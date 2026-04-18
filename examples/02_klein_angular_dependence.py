#!/usr/bin/env python3
"""
Example 2 — Klein tunneling at non-zero incidence angles.
==========================================================

Demonstrates the angular dependence of Klein tunneling through a
rectangular electrostatic barrier in the bipolar regime
(V_0 > E_F). A Gaussian wave packet is injected at a sequence of
incidence angles theta into an isotropic, untilted Dirac channel,
and the transmission T(theta) and reflection R(theta) are read off
from the drain contacts. In an isotropic untilted cone the system is
mirror-symmetric under y → -y, so physics requires

    T(+theta) = T(-theta),   R(+theta) = R(-theta).

Any residual asymmetry is numerical (FFT sampling, drain
bookkeeping) and should remain below ~10^-3. This mirror-symmetry
self-consistency is itself a useful validation.

Physical setup
--------------
The channel hosts a symmetric, isotropic 2D Dirac cone

    H = hbar v_F (sigma_x k_x + sigma_y k_y) + V(x, y) I + M(y) sigma_z

with v_x = v_y = v_F and no tilt (w_x = w_y = 0). A single valley is
propagated because, at zero tilt, K and K' are equivalent; the
second-valley run would reproduce the K data bit-for-bit. A
rectangular barrier of height V_0 = 2.0 E_F and thickness d = 100
sim-units (~ 330 nm) is placed at the channel centre. At normal
incidence (theta = 0) the Klein paradox enforces T = 1 exactly for
massless Dirac fermions regardless of V_0; away from normal
incidence, T(theta) drops and develops Fabry–Pérot structure from
the barrier's two interfaces.

By default the script scans theta in {-15°, -5°, +5°, +15°}, which
samples two ±-pairs whose T values must agree by mirror symmetry.
Each angle is run as an independent simulation; the total cost is
4 × (single-valley) ≈ same order as a single dual-valley run of
Example 1.

Running
-------
Default (publication-quality, ~20-40 min total on 8+ cores):

    python 02_klein_angular_dependence.py

Smoke test (~3-5 min total):

    python 02_klein_angular_dependence.py --quick

Custom angle list:

    python 02_klein_angular_dependence.py --angles -20 -10 10 20

Outputs
-------
    <output-dir>/
        theta_-15/                 # one subdir per angle
            final_snapshot.png
            probability_history.png
            transmission_report.png
            wavefunction/
        theta_-05/ ...
        theta_+05/ ...
        theta_+15/ ...
        klein_angular_snapshots.png      # 1x4 grid of final frames
        klein_angular_summary.png        # T(theta), R(theta) scatter
        klein_angular_results.csv        # tab-separated T, R, P_y per angle
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dirac_wavepacket.config import (
    SimConfig, GridConfig, PhysicsConfig, TimeConfig, WavepacketConfig,
    PotentialConfig, AbsorberConfig, DetectorConfig, CouplingConfig,
)
from dirac_wavepacket.simulation import run_simulation
from dirac_wavepacket.wf_io import WavefunctionLoader


# =============================================================================
#  Configuration builder
# =============================================================================

def build_config(*, theta_deg: float, quick: bool = False) -> SimConfig:
    """
    Construct a single-angle Klein-tunneling configuration.

    Parameters
    ----------
    theta_deg : float
        Injection angle in degrees (positive: beam drifts toward +y).
    quick : bool
        If True, use a reduced grid and step count for a smoke test.

    Returns
    -------
    cfg : SimConfig
    """
    cfg = SimConfig()
    cfg.title = (
        f"Klein tunneling (isotropic, untilted, theta = {theta_deg:+g} deg)"
    )

    # ---- Grid ---------------------------------------------------------------
    # Lx = Ly = 300 gives the oblique beam enough room in y so that even
    # the largest |theta| = 15° trajectory remains well inside the
    # reflecting walls throughout the traversal.
    if quick:
        cfg.grid = GridConfig(Nx=192, Ny=192, Lx=300.0, Ly=300.0)
    else:
        cfg.grid = GridConfig(Nx=512, Ny=512, Lx=300.0, Ly=300.0)

    # ---- Physics: isotropic, untilted ---------------------------------------
    # Single valley only. At zero tilt K ≡ K', and propagating a second
    # valley adds no information.
    cfg.physics = PhysicsConfig(
        vf=1.0,
        tilt=[0.0, 0.0],
        valleys="K",
    )
    cfg.coupling = CouplingConfig(enabled=False)

    # ---- Wavepacket ---------------------------------------------------------
    # sigma = 8 gives an angular FWHM of ~5°, comparable to the step
    # between sampled angles. This is narrow enough that each run
    # represents a directed beam around theta_deg, and broad enough
    # that the Fabry-Pérot resonances of the barrier are gently
    # averaged rather than perfectly resolved.
    cfg.wavepacket = WavepacketConfig(
        k0=0.8,                    # E_F = 0.8 (sim units)
        theta=float(theta_deg),    # degrees (converted internally)
        sigma_x=9.0,
        sigma_y=9.0,
        x_source=-0.75,            # x = -0.75 * Lx/2 = -112.5
    )

    # ---- Potential: rectangular barrier at alpha = 0 -----------------------
    d = 100.0
    cfg.potential = PotentialConfig(
        type="shaped",
        height=1.6,                # V_0 = 2.00 E_F (deep bipolar Klein regime)
        edges=[-d / 2, +d / 2, -d / 2, +d / 2],
        smoothing_width=0.0,
    )

    # ---- Boundaries --------------------------------------------------------
    # Reflecting y-walls with mass confinement. The largest beam in the
    # scan (theta = 15°) is well inside the wall region throughout.
    cfg.absorber = AbsorberConfig(
        width_frac=0.03,
        strength=0.08,
        y_bc="reflecting",
        y_width_frac=0.03,
        wall_mass=5.0,
    )

    # ---- Time evolution ----------------------------------------------------
    cfg.time = TimeConfig(
        dt=0.04,
        n_steps=6000 if quick else 15000,
        save_every=200,
    )

    # ---- Detectors ---------------------------------------------------------
    cfg.detector = DetectorConfig(
        enabled=True,
        auto_source=False,
        auto_stop=True,
        stop_threshold=0.005,
    )

    return cfg


# =============================================================================
#  Snapshot grid  (1 row × N angles)
# =============================================================================

def plot_angle_grid(
    per_angle: list[dict],
    out_path: Path,
    frame_fraction: float = 0.75,
) -> None:
    """
    Render a single-row snapshot grid, one panel per incidence angle,
    at a common fractional time through each run. The shared colourmap
    scale makes the angular dependence of the transmitted beam
    directly comparable across panels.

    Parameters
    ----------
    per_angle : list of dict
        Each dict has keys: "theta", "cfg", "wf_dir", "T", "R".
    out_path : Path
        Output PNG path.
    frame_fraction : float
        Fraction (0 < f ≤ 1) into each run's snapshot sequence to use
        for the displayed frame. 0.75 shows late-time post-scattering
        state in most configurations.
    """
    from dirac_wavepacket.visualizer import (
        _plot_density,
        _vmax_from_density,
        _add_current_overlay,
    )

    n_angles = len(per_angle)
    if n_angles == 0:
        print("  [warn] no per-angle data; skipping grid")
        return

    # --- Load one snapshot per angle (at the same fractional time). ----
    snaps = []
    vmax = 0.0
    for entry in per_angle:
        wf_dir = entry["wf_dir"]
        if not wf_dir.exists():
            snaps.append(None)
            continue
        loader = WavefunctionLoader(wf_dir)
        if len(loader) == 0:
            snaps.append(None)
            continue
        j = int(np.clip(round((len(loader) - 1) * frame_fraction),
                        0, len(loader) - 1))
        step = loader.steps[j]
        snap = loader.load_step(step)
        rho = (np.abs(snap["psi_K"][0]) ** 2
               + np.abs(snap["psi_K"][1]) ** 2)
        snaps.append((snap, rho))
        vmax = max(vmax, _vmax_from_density(rho))

    # --- Figure skeleton. ----------------------------------------------
    fig_w = 3.5 * n_angles + 0.8
    fig, axes = plt.subplots(
        1, n_angles, figsize=(fig_w, 3.4), dpi=150,
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    if n_angles == 1:
        axes = np.array([axes])

    a00 = 6.58212   # nm per sim-unit
    im_last = None

    for col, (entry, snap_rho) in enumerate(zip(per_angle, snaps)):
        ax = axes[col]
        theta = float(entry["theta"])
        cfg = entry["cfg"]
        Lx2, Ly2 = cfg.grid.Lx / 2.0, cfg.grid.Ly / 2.0

        if snap_rho is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        snap, rho = snap_rho
        im_last = _plot_density(ax, rho, cfg, vmax)
        _add_current_overlay(
            ax, snap["psi_K"], cfg,
            wx=cfg.physics.wx, wy=cfg.physics.wy,
        )

        # Title and T/R annotations.
        ax.set_title(rf"$\theta = {theta:+g}^\circ$", fontsize=10, pad=3)

        x_info = -Lx2 * a00 + 0.03 * cfg.grid.Lx * a00
        y_info = -Ly2 * a00 + 0.04 * cfg.grid.Ly * a00
        info_kw = dict(fontsize=7, ha="left", va="bottom",
                       color="#333333",
                       bbox=dict(facecolor="white", alpha=0.85,
                                 edgecolor="none", pad=1.5))
        ax.text(x_info, y_info,
                f"T = {entry['T']:.3f}   R = {entry['R']:.3f}",
                **info_kw)

        if col > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    # Shared colorbar.
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.06, right=0.90)
    if im_last is not None:
        cbar_ax = fig.add_axes([0.915, 0.14, 0.014, 0.74])
        cbar = fig.colorbar(im_last, cax=cbar_ax)
        cbar.set_label(r"$|\psi|^2$", fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    # Suptitle.
    cfg0 = per_angle[0]["cfg"]
    V0 = cfg0.potential.height
    E_F = cfg0.physics.vf * cfg0.wavepacket.k0
    fig.suptitle(
        rf"Klein tunneling, angular scan: "
        rf"$V_0/E_F = {V0/E_F:.2f}$,  isotropic untilted cone",
        fontsize=11, y=0.97,
    )

    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_tr_summary(
    per_angle: list[dict],
    out_path: Path,
) -> None:
    """
    T(theta) and R(theta) scatter summary. Paired ±theta points are
    plotted with matching colours so mirror symmetry is visible at a
    glance.
    """
    thetas = np.array([e["theta"] for e in per_angle], dtype=float)
    T = np.array([e["T"] for e in per_angle], dtype=float)
    R = np.array([e["R"] for e in per_angle], dtype=float)
    Py = np.array([e["P_y"] for e in per_angle], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    fig.patch.set_facecolor("white")

    order = np.argsort(thetas)
    ax.plot(thetas[order], T[order], "o-", color="#1f77b4",
            label=r"$T(\theta)$", markersize=7)
    ax.plot(thetas[order], R[order], "s-", color="#d62728",
            label=r"$R(\theta)$", markersize=6)
    if np.any(Py > 1e-4):
        ax.plot(thetas[order], Py[order], "^--", color="#7f7f7f",
                label=r"$P_y(\theta)$  (wall absorption)",
                markersize=5, alpha=0.8)

    # Reference Klein peak at theta = 0.
    ax.axhline(1.0, linestyle=":", color="#bbbbbb", linewidth=0.8)
    ax.text(0.02, 0.96, r"Klein paradox: $T(0) = 1$",
            transform=ax.transAxes,
            fontsize=8, color="#666666", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7,
                      edgecolor="none", pad=2))

    ax.set_xlabel(r"Injection angle $\theta$ (deg)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="center right", fontsize=9)
    ax.set_title(
        r"Angular dependence of Klein tunneling  ($V_0 > E_F$, untilted)",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# =============================================================================
#  Main
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Klein tunneling angular dependence, isotropic untilted.",
    )
    parser.add_argument(
        "--angles", nargs="+", type=float,
        default=[-15.0, -5.0, 5.0, 15.0],
        help="Injection angles in degrees (default: -15 -5 5 15).",
    )
    parser.add_argument(
        "--output-dir", default="results/ex02_klein_angular",
        help="Top-level output directory; per-angle runs live in sub-dirs.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced grid / step count for CI and local iteration.",
    )
    parser.add_argument(
        "--frame-fraction", type=float, default=0.75,
        help="Fraction (0-1) of each run's snapshot series to display "
             "in the grid (default: 0.75).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Example 2 — Klein tunneling, angular scan")
    print("=" * 70)
    print(f"  angles            : {args.angles}  deg")
    print(f"  V_0 / E_F         : 2.00  (Klein regime)")
    print(f"  barrier width d   : 100  sim-units")
    print(f"  cone              : isotropic, untilted (w_x = w_y = 0)")
    print(f"  valley            : K  (single; K' equivalent at zero tilt)")
    print(f"  grid / time       : {'smoke test' if args.quick else 'production'}")
    print()

    per_angle = []
    for theta_deg in args.angles:
        print("-" * 70)
        print(f"  theta = {theta_deg:+g} deg")
        print("-" * 70)

        # Tag per-angle output sub-directories with a zero-padded sign.
        sign = "+" if theta_deg >= 0 else "-"
        tag = f"theta_{sign}{int(round(abs(theta_deg))):02d}"
        angle_out = out_dir / tag
        angle_out.mkdir(parents=True, exist_ok=True)

        cfg = build_config(theta_deg=theta_deg, quick=args.quick)
        cfg.output_dir = str(angle_out)

        result = run_simulation(
            cfg,
            make_animation=False,
            verbose=False,
            data_only=False,
            save_wf=True,
            wf_save_every=max(cfg.time.n_steps // 20, 100),
        )

        T, R, P_y = result["T"], result["R"], result.get("P_y", 0.0)
        P_rem = 1.0 - T - R - P_y
        print(f"    T = {T:.4f}   R = {R:.4f}   "
              f"P_y = {P_y:.4f}   P_rem = {P_rem:.4f}")

        per_angle.append({
            "theta": theta_deg,
            "cfg": cfg,
            "wf_dir": angle_out / "wavefunction",
            "T": T,
            "R": R,
            "P_y": P_y,
        })
        print()

    # --- Summary table -----------------------------------------------------
    print("=" * 70)
    print("  ANGULAR SCAN SUMMARY")
    print("=" * 70)
    print(f"  {'theta (deg)':>12}  {'T':>8}  {'R':>8}  {'P_y':>8}  {'T+R+P_y':>10}")
    print("  " + "-" * 56)
    for e in per_angle:
        budget = e["T"] + e["R"] + e["P_y"]
        print(f"  {e['theta']:>+12.1f}  {e['T']:>8.4f}  {e['R']:>8.4f}  "
              f"{e['P_y']:>8.4f}  {budget:>10.4f}")
    print()

    # --- Mirror-symmetry check --------------------------------------------
    # For every pair (+theta, -theta) that was actually scanned, report
    # the asymmetry. These should be numerically zero.
    pos = {e["theta"]: e for e in per_angle if e["theta"] > 0}
    for e in per_angle:
        th = e["theta"]
        if th < 0 and (-th) in pos:
            p = pos[-th]
            dT = e["T"] - p["T"]
            dR = e["R"] - p["R"]
            print(f"  symmetry  theta = ±{-th:.1f}°:  "
                  f"ΔT = {dT:+.2e},  ΔR = {dR:+.2e}")
    print()

    # --- CSV dump ----------------------------------------------------------
    csv_path = out_dir / "klein_angular_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_deg", "T", "R", "P_y", "P_rem"])
        for e in per_angle:
            P_rem = 1.0 - e["T"] - e["R"] - e["P_y"]
            w.writerow([f"{e['theta']:+.4f}",
                        f"{e['T']:.6f}",
                        f"{e['R']:.6f}",
                        f"{e['P_y']:.6f}",
                        f"{P_rem:.6f}"])
    print(f"  wrote {csv_path}")

    # --- Figures -----------------------------------------------------------
    plot_angle_grid(
        per_angle,
        out_path=out_dir / "klein_angular_snapshots.png",
        frame_fraction=args.frame_fraction,
    )
    plot_tr_summary(
        per_angle,
        out_path=out_dir / "klein_angular_summary.png",
    )

    print()
    print(f"  full output tree  : {out_dir.resolve()}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
