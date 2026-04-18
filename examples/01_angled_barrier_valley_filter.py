#!/usr/bin/env python3
"""
Example 1 — All-electrostatic valley filtering by barrier rotation.
====================================================================

Demonstrates a purely electrostatic valley-filtering mechanism in a 2D
channel hosting tilted Dirac fermions. Two Gaussian wave packets — one
at valley K and one at K' — are launched at normal incidence and
propagated independently under the tilted Dirac Hamiltonian across an
electrostatic barrier. Two geometries are compared via the
``--barrier`` flag:

  * ``rect``    :  barrier interfaces normal to the transport axis.
                   Control run. Both valleys see the same barrier and
                   the transmission is valley-degenerate
                   (T_K = T_K', eta = 0). The tilt produces only a
                   transverse separation of the two valley packets.

  * ``angled``  :  barrier rotated by alpha = 20 deg about the y-axis.
                   The two valleys have oblique Klein-tunneling peaks
                   at equal-and-opposite angles +/- theta_K; rotating
                   the barrier shifts one valley closer to its peak
                   and the other away from it. The same barrier
                   therefore transmits one valley and reflects the
                   other, producing a net valley polarization
                       eta = (T_K - T_K') / (T_K + T_K') > 0
                   without any magnetic field or strain.

Physical parameters
-------------------
The model is intentionally material-agnostic. The dispersion is an
isotropic 2D tilted Dirac cone,

    H_tau = hbar v_F (sigma_x k_x + sigma_y k_y)
            + tau hbar (w_x k_x + w_y k_y) I
            + V(x, y) I  +  M(y) sigma_z ,

with v_x = v_y = v_F and a purely transverse tilt w_y = 0.3 v_F,
representative of the type-I tilted Dirac / Weyl regime. Valley index
tau = +1 for K, -1 for K'. Users should substitute their own
material-specific values of v_F, w_x, w_y for quantitative
predictions. The in-code units are natural units (hbar = v_F = 1);
an absolute scale can be applied post-hoc via the helpers in
``dirac_wavepacket.units``.

Running
-------
Default (publication-quality, ~15-30 min on 8+ cores):

    python 01_angled_barrier_valley_filter.py --barrier angled

Smoke test (~1-3 min, visibly under-resolved but pipeline-exact):

    python 01_angled_barrier_valley_filter.py --barrier angled --quick

Control run (rectangular barrier):

    python 01_angled_barrier_valley_filter.py --barrier rect

Outputs
-------
    <output-dir>/
        final_snapshot.png         # K and K' side-by-side at t_final
        probability_history.png    # P(t) for each valley
        transmission_report.png    # cumulative drain absorption + eta(t)
        wavefunction/              # .npz snapshot series
        valley_filter_snapshots.png    # multi-frame K / K' overlay
"""
from __future__ import annotations

import argparse
import math
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

def build_config(*, barrier: str, quick: bool = False) -> SimConfig:
    """
    Construct the valley-filter configuration.

    Parameters
    ----------
    barrier : {"rect", "angled"}
        "rect"   -> alpha = 0, control run, valley-degenerate
        "angled" -> alpha = 20 deg, filtering geometry
    quick : bool
        If True, use a reduced grid and step count for a smoke test.

    Returns
    -------
    cfg : SimConfig
    """
    if barrier not in ("rect", "angled"):
        raise ValueError(f"barrier must be 'rect' or 'angled', got {barrier!r}")

    cfg = SimConfig()
    alpha_deg = 0.0 if barrier == "rect" else 20.0
    cfg.title = (
        f"Valley filter (tilted Dirac, alpha = {alpha_deg:g} deg, {barrier})"
    )

    # ---- Grid ---------------------------------------------------------------
    # Lx, Ly chosen so the source packet, the barrier region, and the
    # drains are comfortably separated. Nx, Ny are FFT-friendly.
    if quick:
        cfg.grid = GridConfig(Nx=192, Ny=192, Lx=300.0, Ly=200.0)
    else:
        cfg.grid = GridConfig(Nx=512, Ny=512, Lx=300.0, Ly=200.0)

    # ---- Physics ------------------------------------------------------------
    # Isotropic v_x = v_y = v_F = 1 (sim units). Transverse tilt w_y = 0.3 v_F.
    # The K' valley automatically uses reversed tilt and opposite chirality.
    cfg.physics = PhysicsConfig(
        vf=1.0,
        tilt=[0.0, 0.3],           # (w_x, w_y) at K; K' uses (-w_x, -w_y)
        valleys="both",            # propagate K and K' side-by-side
    )
    cfg.coupling = CouplingConfig(enabled=False)

    # ---- Wavepacket ---------------------------------------------------------
    # k_0 = 0.8 gives E_F = hbar v_F k_0 = 0.8 (sim units). Normal incidence.
    # sigma_x = sigma_y = 7 gives angular spread
    #     Delta_theta ~ 1 / (2 sigma k_0) ~ 0.09 rad ~ 5 deg,
    # narrower than the oblique Klein angle theta_K = arctan(w_y) ~ 17 deg.
    # This lets the rotated-barrier angular shift be the dominant source of
    # valley-selective transmission.
    cfg.wavepacket = WavepacketConfig(
        k0=0.8,
        theta=0.0,
        sigma_x=7.0,
        sigma_y=7.0,
        x_source=-0.75,            # x = -0.75 * Lx/2 = -112.5
    )

    # ---- Potential ---------------------------------------------------------
    # 'shaped' builds a parallelogram whose left/right edges vary linearly
    # in y. Edge positions for a barrier of thickness d, centred on x = 0,
    # rotated by alpha about the y-axis:
    #       x_L(y) = -d/2 + y tan(alpha),   x_R(y) = +d/2 + y tan(alpha)
    # edges = [xL_bot, xR_bot, xL_top, xR_top] gives these values at
    # y = -Ly/2 and y = +Ly/2. For alpha = 0 the edges are vertical.
    d = 100.0                                   # barrier thickness at y = 0
    half_Ly = cfg.grid.Ly / 2.0
    shift = half_Ly * math.tan(math.radians(alpha_deg))
    cfg.potential = PotentialConfig(
        type="shaped",
        height=1.4,                             # V_0 = 1.75 E_F; n-p-n regime
        edges=[-d / 2 - shift,                  # xL_bot
               +d / 2 - shift,                  # xR_bot
               -d / 2 + shift,                  # xL_top
               +d / 2 + shift],                 # xR_top
        smoothing_width=0.0,                    # sharp interfaces
    )

    # ---- Boundaries --------------------------------------------------------
    # x-drains absorb transmitted / reflected beam and record T, R.
    # Reflecting y-walls via M(y) sigma_z mass confinement keep the packet
    # inside the channel without Klein leakage through the walls (which a
    # scalar wall would allow).
    cfg.absorber = AbsorberConfig(
        width_frac=0.05,
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
#  Snapshot visualisation
# =============================================================================

def plot_snapshots(
    wf_dir: Path,
    cfg: SimConfig,
    out_path: Path,
    n_frames: int = 4,
    alpha_deg: float = 20.0,
    drain_histories: dict | None = None,
) -> None:
    """
    Multi-panel snapshot grid rendered in the same publication style as
    ``dirac_wavepacket.visualizer.render_frame_dual`` (i.e. the frames produced by
    the live animation pipeline): white background, physical-unit axes in
    nanometres, barrier outline, drain / source / wall shading, a single
    shared colourbar, and per-panel T / R annotations.

    Layout::

        row 0:   K  @ t_1      K  @ t_2      K  @ t_3     ...
        row 1:   K' @ t_1      K' @ t_2      K' @ t_3     ...

    All panels share the same colour scale so relative intensities are
    directly readable.

    Parameters
    ----------
    wf_dir : Path
        Wavefunction snapshot directory produced by the simulator.
    cfg : SimConfig
        Configuration used for the run (supplies extent, barrier geometry,
        drain / wall regions).
    out_path : Path
        Output PNG path.
    n_frames : int
        Number of evenly spaced snapshot times to display.
    alpha_deg : float
        Barrier rotation angle, used in the figure super-title.
    drain_histories : dict or None
        Optional mapping {"T_K": np.ndarray, "T_Kp": np.ndarray,
        "R_K": np.ndarray, "R_Kp": np.ndarray, "times": np.ndarray}
        giving cumulative drain absorption vs time so per-panel T/R
        annotations can be produced at each snapshot's physical time.
        When None, the annotations are omitted.
    """
    # Pull in the package's own panel helpers. They are underscore-private
    # but fully self-contained; we import them directly to keep the styling
    # synchronised with the single-frame renderer without duplicating code.
    from dirac_wavepacket.visualizer import (
        _plot_density,
        _vmax_from_density,
        _add_current_overlay,
    )

    loader = WavefunctionLoader(wf_dir)
    n_total = len(loader)
    if n_total == 0:
        print(f"  [warn] no snapshots in {wf_dir}; skipping figure")
        return

    idxs = np.linspace(0, n_total - 1, n_frames, dtype=int)
    step_list = [loader.steps[int(i)] for i in idxs]
    snaps = [loader.load_step(s) for s in step_list]

    # --- Shared vmax across every panel so intensities are comparable. --
    vmax = 0.0
    panels = []   # list of (snap, rho_K, rho_Kp)
    for snap in snaps:
        rho_K = np.abs(snap["psi_K"][0]) ** 2 + np.abs(snap["psi_K"][1]) ** 2
        rho_Kp = np.abs(snap["psi_Kp"][0]) ** 2 + np.abs(snap["psi_Kp"][1]) ** 2
        panels.append((snap, rho_K, rho_Kp))
        vmax = max(vmax, _vmax_from_density(rho_K),
                   _vmax_from_density(rho_Kp))

    # --- Figure skeleton. dpi and facecolor match the single-frame dump. -
    fig_w = 3.5 * n_frames + 0.8    # extra for the colourbar
    fig, axes = plt.subplots(
        2, n_frames, figsize=(fig_w, 6.0), dpi=150,
        gridspec_kw={"wspace": 0.08, "hspace": 0.18},
    )
    fig.patch.set_facecolor("white")
    if n_frames == 1:
        axes = axes.reshape(2, 1)

    # For per-panel T/R annotations: if a history was supplied, look up
    # the cumulative values at each snapshot's step index.
    a00 = 6.58212   # nm per sim-unit; mirrors _plot_density

    im_last = None
    for col, (snap, rho_K, rho_Kp) in enumerate(panels):
        t_phys = float(snap["time"])
        step = int(snap["step"])

        # K panel (top row)
        ax_K = axes[0, col]
        im_last = _plot_density(ax_K, rho_K, cfg, vmax)
        _add_current_overlay(
            ax_K, snap["psi_K"], cfg,
            wx=cfg.physics.wx, wy=cfg.physics.wy,
        )

        # K' panel (bottom row)
        ax_Kp = axes[1, col]
        _plot_density(ax_Kp, rho_Kp, cfg, vmax)
        _add_current_overlay(
            ax_Kp, snap["psi_Kp"], cfg,
            wx=-cfg.physics.wx, wy=-cfg.physics.wy,
        )

        # Per-column time label at the top
        ax_K.set_title(f"$t$ = {t_phys:.1f}", fontsize=9, pad=3)

        # Valley labels (top-left corner of each panel).
        Lx2, Ly2 = cfg.grid.Lx / 2.0, cfg.grid.Ly / 2.0
        x_lab = -Lx2 * a00 + 0.03 * cfg.grid.Lx * a00
        y_lab = +Ly2 * a00 - 0.04 * cfg.grid.Ly * a00
        box_kw = dict(facecolor="white", alpha=0.85,
                      edgecolor="none", pad=2)
        ax_K.text(x_lab, y_lab, r"$K$", fontsize=10, fontweight="bold",
                  ha="left", va="top", color="#222222", bbox=box_kw)
        ax_Kp.text(x_lab, y_lab, r"$K'$", fontsize=10, fontweight="bold",
                   ha="left", va="top", color="#222222", bbox=box_kw)

        # T / R annotations if histories provided.
        if drain_histories is not None:
            t_hist = np.asarray(drain_histories["times"])
            j = int(np.searchsorted(t_hist, t_phys))
            j = min(j, len(t_hist) - 1)
            T_K = float(drain_histories["T_K"][j])
            R_K = float(drain_histories["R_K"][j])
            T_Kp = float(drain_histories["T_Kp"][j])
            R_Kp = float(drain_histories["R_Kp"][j])
            x_info = -Lx2 * a00 + 0.03 * cfg.grid.Lx * a00
            y_info = -Ly2 * a00 + 0.04 * cfg.grid.Ly * a00
            info_kw = dict(fontsize=7, ha="left", va="bottom",
                           color="#333333",
                           bbox=dict(facecolor="white", alpha=0.85,
                                     edgecolor="none", pad=1.5))
            ax_K.text(x_info, y_info,
                      f"T = {T_K:.3f}   R = {R_K:.3f}", **info_kw)
            ax_Kp.text(x_info, y_info,
                       f"T = {T_Kp:.3f}   R = {R_Kp:.3f}", **info_kw)

        # Hide inner y-labels for readability.
        if col > 0:
            ax_K.set_ylabel("")
            ax_Kp.set_ylabel("")
            ax_K.set_yticklabels([])
            ax_Kp.set_yticklabels([])
        # Hide x-labels on top row.
        ax_K.set_xlabel("")
        ax_K.set_xticklabels([])

    # Shared colorbar on the right.
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.90)
    cbar_ax = fig.add_axes([0.915, 0.10, 0.014, 0.80])
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label(r"$|\psi|^2$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Suptitle.
    V0 = cfg.potential.height
    E_F = cfg.physics.vf * cfg.wavepacket.k0
    fig.suptitle(
        rf"Valley filter: $\alpha = {alpha_deg:g}^\circ$,  "
        rf"$V_0/E_F = {V0/E_F:.2f}$,  "
        rf"$w_y = {cfg.physics.wy:g}\,v_F$",
        fontsize=11, y=0.975,
    )

    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# =============================================================================
#  Main
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="All-electrostatic valley filtering by barrier rotation.",
    )
    parser.add_argument(
        "--barrier", choices=["rect", "angled"], default="angled",
        help="'rect' for the alpha = 0 control run, 'angled' for alpha = 20 "
             "deg (default: angled).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/ex01_{barrier}).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced grid / step count for CI and local iteration.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=4,
        help="Number of snapshot frames to plot (default: 4).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(
        args.output_dir if args.output_dir
        else f"results/ex01_{args.barrier}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(barrier=args.barrier, quick=args.quick)
    cfg.output_dir = str(out_dir)

    alpha_deg = 0.0 if args.barrier == "rect" else 20.0
    E_F = cfg.physics.vf * cfg.wavepacket.k0
    V0 = cfg.potential.height

    print("=" * 70)
    print(f"  Example 1 — Valley filtering ({args.barrier} barrier)")
    print("=" * 70)
    print(f"  tilt (w_x, w_y)   : ({cfg.physics.wx}, {cfg.physics.wy}) * v_F")
    print(f"  barrier rotation  : {alpha_deg:g} deg")
    print(f"  V_0 / E_F         : {V0 / E_F:.2f}  (n-p-n regime)")
    print(f"  grid              : {cfg.grid.Nx} x {cfg.grid.Ny}  "
          f"(Lx={cfg.grid.Lx}, Ly={cfg.grid.Ly})")
    print(f"  time              : n_steps={cfg.time.n_steps}  dt={cfg.time.dt}")
    print()

    result = run_simulation(
        cfg,
        make_animation=False,
        verbose=True,
        data_only=False,
        save_wf=True,
        wf_save_every=max(cfg.time.n_steps // 20, 100),
    )

    T_K, T_Kp = result["T"], result["T_Kp"]
    R_K, R_Kp = result["R"], result["R_Kp"]
    eta = result["eta"]

    print()
    print("=" * 70)
    print(f"  VALLEY-FILTER RESULTS ({args.barrier} barrier)")
    print("=" * 70)
    print(f"  T_K  = {T_K:.4f}     R_K  = {R_K:.4f}")
    print(f"  T_K' = {T_Kp:.4f}     R_K' = {R_Kp:.4f}")
    print(f"  Delta T = T_K - T_K' = {T_K - T_Kp:+.4f}")
    print(f"  eta = (T_K - T_K')/(T_K + T_K') = {eta:+.4f}")
    print()

    wf_dir = out_dir / "wavefunction"
    if wf_dir.exists():
        # Build drain histories dict so per-panel T/R annotations appear.
        drain_histories = None
        if "det_times" in result and "T_history" in result:
            drain_histories = {
                "times": np.asarray(result["det_times"]),
                "T_K":   np.asarray(result["T_history"]),
                "R_K":   np.asarray(result["R_history"]),
                "T_Kp":  np.asarray(result.get("T_history_Kp",
                                               result["T_history"])),
                "R_Kp":  np.asarray(result.get("R_history_Kp",
                                               result["R_history"])),
            }
        plot_snapshots(
            wf_dir=wf_dir,
            cfg=cfg,
            out_path=out_dir / "valley_filter_snapshots.png",
            n_frames=args.n_frames,
            alpha_deg=alpha_deg,
            drain_histories=drain_histories,
        )

    print(f"  full output tree  : {out_dir.resolve()}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
