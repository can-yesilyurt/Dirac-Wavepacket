#!/usr/bin/env python3
"""
Valley qubit phase gate — source-drain bias sweep.

Sweeps V_sd (source-drain bias) and optionally V_0 (barrier height) to
extract I–V characteristics and 2D conductance density maps.

Modes (auto-detected from CLI flags)
------------------------------------
  1D I–V              : --v0 <value>          --vsd-min/--vsd-max/--vsd-step
                        Fixes V_0, sweeps V_sd. Produces:
                          - T_K(V_sd), T_Kp(V_sd)
                          - I_K(V_sd), I_Kp(V_sd)  (Landauer: I = T · V_sd)
                          - G_K(V_sd), G_Kp(V_sd)  (dI/dV_sd, finite difference)
                          - eta(V_sd)
                        plus two PNG plots.

  2D density plot     : --v0-min/--v0-max/--v0-step + --vsd-min/--vsd-max/--vsd-step
                        Sweeps both. Produces a 4-panel heatmap:
                          (a) T_K(V_0, V_sd)
                          (b) T_Kp(V_0, V_sd)
                          (c) eta(V_0, V_sd) — gate visibility landscape
                          (d) ΔG(V_0, V_sd) — differential-conductance asymmetry
                                              G_K − G_Kp (per V_sd column)

Usage
-----
    # Small bias I–V curve at the design operating point
    python sweep_Vsd.py --config configs/reflecting_walls_w35_W50.yaml \\
                       --v0 0.228 \\
                       --vsd-min -0.02 --vsd-max 0.02 --vsd-step 0.004 \\
                       --jobs 8 --no-anim

    # 2D density plot
    python sweep_Vsd.py --config configs/reflecting_walls_w35_W50.yaml \\
                       --v0-min 0.0  --v0-max 0.5  --v0-step 0.05 \\
                       --vsd-min -0.02 --vsd-max 0.02 --vsd-step 0.004 \\
                       --jobs 16 --threads-per-job 8 --no-anim

    # Dry run — print plan only
    python sweep_Vsd.py --config configs/X.yaml --v0 0.228 \\
                       --vsd-min -0.02 --vsd-max 0.02 --vsd-step 0.004 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import multiprocessing as _mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force 'spawn' so worker env vars take effect BEFORE numpy/pyFFTW import.
# Same rationale as sweep_V0.py — on Linux, 'fork' inherits the parent's
# already-initialised BLAS/FFT thread pools, silently defeating our
# per-worker thread caps.
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ─── worker ──────────────────────────────────────────────────────────────────

def _run_one(job: dict) -> dict:
    """
    Subprocess entry point: set thread caps BEFORE importing heavy modules,
    load YAML, override V_0 and V_sd, run one simulation, return a row.
    """
    threads = int(job["threads"])
    os.environ["DIRAC2D_FFT_THREADS"] = str(threads)
    # "auto" degrades gracefully to numpy FFT if pyFFTW is not installed;
    # the propagator auto-selects pyFFTW when available for ~3× speedup.
    os.environ["DIRAC2D_FFT_BACKEND"] = "auto"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "PYFFTW_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)

    # Re-import inside subprocess so env vars take effect
    from dirac_wavepacket.config import load_config
    from dirac_wavepacket.simulation import run_simulation
    from dirac_wavepacket.transport_metrics import valley_group_delay_from_results

    if job.get("verbose_threads", False):
        try:
            import pyfftw
            pf_ok = True
        except ImportError:
            pf_ok = False
        print(f"  [pid={os.getpid()}] V0={job['V0']:.4f} V_sd={job['V_sd']:+.4f}  "
              f"threads={threads}  backend=pyfftw  pyFFTW={'OK' if pf_ok else 'MISSING'}",
              flush=True)

    cfg = load_config(job["config_path"])
    cfg.potential.height = float(job["V0"])
    cfg.output_dir = job["out_dir"]

    # Override bias block — this is the whole point of this sweep
    cfg.bias.enabled = True
    cfg.bias.V_sd = float(job["V_sd"])
    cfg.bias.type = job["bias_type"]
    cfg.bias.x_center = float(job["bias_x_center"])
    cfg.bias.L_drop = job["bias_L_drop"]     # may be None → auto

    if job.get("source_mode") is not None:
        cfg.physics.source_mode = job["source_mode"]

    t0 = time.perf_counter()
    res = run_simulation(cfg, make_animation=job["make_anim"],
                         save_wf=False, verbose=False)
    t_elapsed = time.perf_counter() - t0

    Ef = cfg.physics.vf * cfg.wavepacket.k0
    T_K  = float(res["T"])
    R_K  = float(res["R"])
    T_Kp = float(res.get("T_Kp", res["T"]))
    R_Kp = float(res.get("R_Kp", res["R"]))
    V_sd = float(job["V_sd"])

    # Landauer small-bias current (dimensionless here — multiply by e²/h in SI).
    # Valid as an I–V datum as long as we stay in the linear-response window
    # V_sd ≪ Ef. With V_sd=0 the current is identically 0; G is recovered
    # from finite differences across the V_sd grid (done in post-processing).
    I_K  = T_K  * V_sd
    I_Kp = T_Kp * V_sd

    # eta is undefined at V_sd = 0 for the trivial reason that both currents
    # are zero there; but η of *transmissions* is still meaningful and is
    # the gate-visibility quantity Can has been tracking. So always report η
    # of T, not of I.
    denom = T_K + T_Kp
    eta = (T_K - T_Kp) / denom if denom > 1e-12 else 0.0

    row = {
        "V0":         float(job["V0"]),
        "V_sd":       V_sd,
        "Ef":         Ef,
        "V0_over_Ef": float(job["V0"]) / Ef if Ef > 0 else 0.0,
        "Vsd_over_Ef": V_sd / Ef if Ef > 0 else 0.0,
        "T_K":        T_K,
        "T_Kp":       T_Kp,
        "R_K":        R_K,
        "R_Kp":       R_Kp,
        "I_K":        I_K,
        "I_Kp":       I_Kp,
        "eta":        eta,
        "time_s":     round(t_elapsed, 1),
    }
    gd = valley_group_delay_from_results(res)
    row["delta_tau_T"] = gd["delta_tau_T"] if gd else float("nan")
    return row


# ─── post-processing helpers ─────────────────────────────────────────────────

def _compute_G_1d(rows_sorted_by_Vsd):
    """
    Finite-difference conductance G = dI/dV_sd along a V_sd axis (fixed V_0).
    Central differences interior, forward/backward at endpoints.
    Modifies each row in place to add 'G_K' and 'G_Kp'.
    """
    if len(rows_sorted_by_Vsd) < 2:
        for r in rows_sorted_by_Vsd:
            r["G_K"] = float("nan")
            r["G_Kp"] = float("nan")
        return

    Vsd = np.array([r["V_sd"] for r in rows_sorted_by_Vsd])
    I_K = np.array([r["I_K"]  for r in rows_sorted_by_Vsd])
    I_Kp = np.array([r["I_Kp"] for r in rows_sorted_by_Vsd])

    G_K  = np.gradient(I_K,  Vsd)
    G_Kp = np.gradient(I_Kp, Vsd)

    for i, r in enumerate(rows_sorted_by_Vsd):
        r["G_K"]  = float(G_K[i])
        r["G_Kp"] = float(G_Kp[i])


def _pivot_2d(rows, V0_grid, Vsd_grid, key):
    """
    Build a 2D array [len(V0_grid), len(Vsd_grid)] of `key` from the
    flat list of rows. NaN-filled for any missing combination.
    """
    idx_V0  = {v: i for i, v in enumerate(V0_grid)}
    idx_Vsd = {v: i for i, v in enumerate(Vsd_grid)}
    out = np.full((len(V0_grid), len(Vsd_grid)), np.nan, dtype=float)
    for r in rows:
        i = idx_V0.get(r["V0"])
        j = idx_Vsd.get(r["V_sd"])
        if i is not None and j is not None:
            out[i, j] = float(r[key])
    return out


def _plot_1d_IV(rows, out_path, V0, Ef):
    """Plot the 1D I–V curve + differential conductance + eta(V_sd)."""
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["V_sd"])
    Vsd = np.array([r["V_sd"] for r in rows])
    T_K = np.array([r["T_K"] for r in rows])
    T_Kp = np.array([r["T_Kp"] for r in rows])
    I_K = np.array([r["I_K"] for r in rows])
    I_Kp = np.array([r["I_Kp"] for r in rows])
    G_K = np.array([r["G_K"] for r in rows])
    G_Kp = np.array([r["G_Kp"] for r in rows])
    eta = np.array([r["eta"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    (a, b), (c, d) = axes

    a.plot(Vsd, T_K,  "o-", color="C0", label=r"$T_K$")
    a.plot(Vsd, T_Kp, "s--", color="C3", label=r"$T_{K'}$")
    a.axvline(0, color="k", lw=0.5, alpha=0.3)
    a.set_xlabel(r"$V_\mathrm{sd}$"); a.set_ylabel("transmission")
    a.set_title(rf"Transmission vs $V_\mathrm{{sd}}$  "
                rf"($V_0={V0:.3f}$, $E_F={Ef:.2f}$)")
    a.legend(); a.grid(alpha=0.3)

    b.plot(Vsd, I_K,  "o-", color="C0", label=r"$I_K$")
    b.plot(Vsd, I_Kp, "s--", color="C3", label=r"$I_{K'}$")
    b.axhline(0, color="k", lw=0.5, alpha=0.3)
    b.axvline(0, color="k", lw=0.5, alpha=0.3)
    b.set_xlabel(r"$V_\mathrm{sd}$"); b.set_ylabel(r"$I = T\,V_\mathrm{sd}$")
    b.set_title("I–V characteristic (Landauer, linear response)")
    b.legend(); b.grid(alpha=0.3)

    c.plot(Vsd, G_K,  "o-", color="C0", label=r"$G_K = dI_K/dV_\mathrm{sd}$")
    c.plot(Vsd, G_Kp, "s--", color="C3", label=r"$G_{K'} = dI_{K'}/dV_\mathrm{sd}$")
    c.axvline(0, color="k", lw=0.5, alpha=0.3)
    c.set_xlabel(r"$V_\mathrm{sd}$"); c.set_ylabel(r"differential conductance $G$")
    c.set_title("Differential conductance (flat plateau = ballistic regime)")
    c.legend(); c.grid(alpha=0.3)

    d.plot(Vsd, eta, "o-", color="C2", label=r"$\eta = (T_K - T_{K'})/(T_K + T_{K'})$")
    d.axvline(0, color="k", lw=0.5, alpha=0.3)
    d.axhline(0, color="k", lw=0.5, alpha=0.3)
    d.set_xlabel(r"$V_\mathrm{sd}$"); d.set_ylabel(r"$\eta$")
    d.set_title(r"Valley polarisation — $V_\mathrm{sd}$-independent in small-bias window")
    d.legend(); d.grid(alpha=0.3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_2d_density(rows, V0_grid, Vsd_grid, out_path, Ef):
    """4-panel heatmap over (V_0, V_sd): T_K, T_Kp, eta, ΔG."""
    import matplotlib.pyplot as plt

    T_K_map  = _pivot_2d(rows, V0_grid, Vsd_grid, "T_K")
    T_Kp_map = _pivot_2d(rows, V0_grid, Vsd_grid, "T_Kp")
    eta_map  = _pivot_2d(rows, V0_grid, Vsd_grid, "eta")

    # G_K / G_Kp were computed row-wise along V_sd for each V_0 already; pivot them
    G_K_map  = _pivot_2d(rows, V0_grid, Vsd_grid, "G_K")
    G_Kp_map = _pivot_2d(rows, V0_grid, Vsd_grid, "G_Kp")
    dG_map = G_K_map - G_Kp_map

    # Use pcolormesh with cell-edge coordinates so each data point is rendered
    # as its own cell (no smoothing, no off-by-half-pixel artefacts).
    def _edges(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 1:
            return np.array([arr[0] - 0.5, arr[0] + 0.5])
        dx = np.diff(arr)
        left  = np.concatenate([[arr[0] - dx[0] / 2], arr[:-1] + dx / 2])
        right = np.array([arr[-1] + dx[-1] / 2])
        return np.concatenate([left, right])

    V0_edges  = _edges(V0_grid)
    Vsd_edges = _edges(Vsd_grid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    (a, b), (c, d) = axes

    # Pick a reasonable max number of x-tick labels to avoid overlap
    from matplotlib.ticker import MaxNLocator

    def _decorate(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    im = a.pcolormesh(Vsd_edges, V0_edges, T_K_map, cmap="viridis", shading="flat")
    fig.colorbar(im, ax=a, label=r"$T_K$")
    _decorate(a, r"$V_\mathrm{sd}$", r"$V_0$",
              rf"$T_K(V_0, V_\mathrm{{sd}})$   $E_F={Ef:.2f}$")

    im = b.pcolormesh(Vsd_edges, V0_edges, T_Kp_map, cmap="viridis", shading="flat")
    fig.colorbar(im, ax=b, label=r"$T_{K'}$")
    _decorate(b, r"$V_\mathrm{sd}$", r"$V_0$",
              r"$T_{K'}(V_0, V_\mathrm{sd})$")

    vmax = float(np.nanmax(np.abs(eta_map))) if np.isfinite(eta_map).any() else 1.0
    im = c.pcolormesh(Vsd_edges, V0_edges, eta_map, cmap="RdBu_r",
                      shading="flat", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=c, label=r"$\eta$")
    _decorate(c, r"$V_\mathrm{sd}$", r"$V_0$",
              r"$\eta(V_0, V_\mathrm{sd})$ — valley visibility landscape")

    vmax = float(np.nanmax(np.abs(dG_map))) if np.isfinite(dG_map).any() else 1.0
    im = d.pcolormesh(Vsd_edges, V0_edges, dG_map, cmap="PuOr_r",
                      shading="flat", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=d, label=r"$G_K - G_{K'}$")
    _decorate(d, r"$V_\mathrm{sd}$", r"$V_0$",
              r"$\Delta G = G_K - G_{K'}$ — differential valley conductance")

    for ax in (a, b, c, d):
        ax.tick_params(axis="x", rotation=30)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── arg parsing ─────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Sweep V_sd (and optionally V_0) for I–V and 2D conductance maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/reflecting_walls_w35_W50.yaml",
                   help="Base YAML config")
    p.add_argument("--output-root", default="./results",
                   help="Root output directory")
    p.add_argument("--prefix", default="vqubit_vsd",
                   help="Output folder prefix")

    # V_sd axis
    p.add_argument("--vsd-min", type=float, default=-0.02)
    p.add_argument("--vsd-max", type=float, default=+0.02)
    p.add_argument("--vsd-step", type=float, default=0.004,
                   help="V_sd step size. Symmetric grids including V_sd=0 "
                        "give clean finite-difference G estimates.")

    # V_0 axis — mutually exclusive with --v0-min/max/step
    g = p.add_mutually_exclusive_group()
    g.add_argument("--v0", type=float, default=None,
                   help="Single V_0 value (1D I–V mode). Default: use "
                        "potential.height from the YAML.")
    g.add_argument("--v0-min", type=float, default=None,
                   help="Lower V_0 (triggers 2D sweep when combined with "
                        "--v0-max).")
    p.add_argument("--v0-max", type=float, default=None)
    p.add_argument("--v0-step", type=float, default=None)

    # Bias profile
    p.add_argument("--bias-type", default="linear", choices=["linear", "localized"],
                   help="Spatial profile of the bias field.")
    p.add_argument("--bias-xc", type=float, default=0.0,
                   help="x_center of the bias (linear: ramp zero; localized: sigmoid centre)")
    p.add_argument("--L-drop", type=float, default=None,
                   help="Total drop length (linear) or 5–95%% width (localized). "
                        "None → auto (linear: grid.Lx, localized: 10).")

    p.add_argument("--source-mode", default=None,
                   choices=["both", "K_only", "Kp_only"],
                   help="Overrides physics.source_mode from the YAML.")
    p.add_argument("--no-anim", action="store_true",
                   help="Skip GIF generation (recommended for large sweeps)")

    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Runs in parallel")
    p.add_argument("--threads-per-job", type=int, default=None,
                   help="FFT/BLAS threads per subprocess")

    p.add_argument("--dry-run", action="store_true",
                   help="Print sweep plan without running simulations")
    return p


def _linspace(lo, hi, step):
    arr = np.arange(lo, hi + 1e-9, step)
    return np.round(arr, 6)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    from dirac_wavepacket.config import load_config

    base_cfg = load_config(args.config)
    Ef = base_cfg.physics.vf * base_cfg.wavepacket.k0

    Vsd_values = _linspace(args.vsd_min, args.vsd_max, args.vsd_step)

    # Decide 1D vs 2D
    is_2d = (args.v0_min is not None and args.v0_max is not None
             and args.v0_step is not None)
    if is_2d:
        V0_values = _linspace(args.v0_min, args.v0_max, args.v0_step)
        mode = "2D"
    else:
        V0_values = np.array(
            [args.v0 if args.v0 is not None else float(base_cfg.potential.height)]
        )
        mode = "1D"

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║   Valley qubit phase gate — V_sd sweep ({mode})     ║")
    print(f"  ╚══════════════════════════════════════════════════╝\n")
    print(f"  Base config : {args.config}")
    print(f"  Ef          : {Ef:.4f}")
    print(f"  V0 axis     : {V0_values[0]:.4f} → {V0_values[-1]:.4f}  "
          f"({len(V0_values)} points)")
    print(f"  V_sd axis   : {Vsd_values[0]:+.4f} → {Vsd_values[-1]:+.4f}  "
          f"({len(Vsd_values)} points)")
    print(f"  Bias profile: type={args.bias_type}  x_c={args.bias_xc}  "
          f"L_drop={args.L_drop}")
    Vsd_over_Ef_max = max(abs(Vsd_values[0]), abs(Vsd_values[-1])) / Ef if Ef > 0 else 0
    print(f"  |V_sd|/Ef   : {Vsd_over_Ef_max:.4f}   "
          f"({'small-bias regime ✓' if Vsd_over_Ef_max < 0.1 else 'WARNING — nonlinear response likely'})")
    print(f"  Total runs  : {len(V0_values) * len(Vsd_values)}")
    print()

    # ── dry-run summary ────────────────────────────────────────────────
    if args.dry_run:
        for V0 in V0_values:
            for Vsd in Vsd_values:
                tag_v0  = f"{float(V0):.4f}".replace(".", "p").replace("-", "m")
                tag_vsd = f"{float(Vsd):+.4f}".replace(".", "p").replace("-", "m").replace("+", "p")
                out_dir = f"{args.output_root}/{args.prefix}_V0-{tag_v0}_Vsd-{tag_vsd}"
                print(f"  [dry-run] V0={float(V0):+.4f}  V_sd={float(Vsd):+.4f}  → {out_dir}")
        print(f"\n  {len(V0_values) * len(Vsd_values)} runs planned. "
              "Remove --dry-run to execute.\n")
        return

    # ── parallel dispatch ──────────────────────────────────────────────
    n_jobs = max(1, int(args.jobs))
    total_cpus = os.cpu_count() or 1
    if args.threads_per_job is not None:
        threads = int(args.threads_per_job)
    else:
        threads = min(16, max(1, total_cpus // n_jobs))

    if n_jobs > 1:
        print(f"  Parallel   : {n_jobs} jobs × {threads} threads "
              f"(host has {total_cpus} CPUs)")
        if n_jobs * threads > total_cpus:
            print(f"  ⚠ oversubscription: {n_jobs}×{threads}={n_jobs*threads} "
                  f"> {total_cpus} cores")

    # Build job specs (row-major: V0 outer, V_sd inner — keeps wall-time
    # prediction roughly linear even on a heterogeneous queue)
    jobs = []
    idx = 0
    for V0 in V0_values:
        for Vsd in Vsd_values:
            tag_v0  = f"{float(V0):.4f}".replace(".", "p").replace("-", "m")
            tag_vsd = f"{float(Vsd):+.4f}".replace(".", "p").replace("-", "m").replace("+", "p")
            out_dir = f"{args.output_root}/{args.prefix}_V0-{tag_v0}_Vsd-{tag_vsd}"
            jobs.append({
                "V0":          float(V0),
                "V_sd":        float(Vsd),
                "config_path": args.config,
                "out_dir":     out_dir,
                "bias_type":   args.bias_type,
                "bias_x_center": args.bias_xc,
                "bias_L_drop": args.L_drop,
                "source_mode": args.source_mode,
                "make_anim":   not args.no_anim,
                "threads":     threads,
                "verbose_threads": (idx < min(2, n_jobs)),
            })
            idx += 1

    results_table = []
    t_start_all = time.perf_counter()

    if n_jobs == 1:
        for i, job in enumerate(jobs):
            print(f"\n{'━' * 70}")
            print(f"  RUN {i+1}/{len(jobs)}:  V0={job['V0']:+.4f}  V_sd={job['V_sd']:+.4f}  →  {job['out_dir']}")
            print(f"{'━' * 70}")
            results_table.append(_run_one(job))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_run_one, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    row = fut.result()
                    results_table.append(row)
                    done += 1
                    print(f"  [{done}/{len(jobs)}] V0={j['V0']:+.4f} "
                          f"V_sd={j['V_sd']:+.4f} done in {row['time_s']:.1f}s  "
                          f"T_K={row['T_K']:.4f} T_K'={row['T_Kp']:.4f} "
                          f"η={row['eta']:+.4f}", flush=True)
                except Exception as e:
                    done += 1
                    print(f"  [{done}/{len(jobs)}] V0={j['V0']:+.4f} "
                          f"V_sd={j['V_sd']:+.4f} FAILED: {e}", flush=True)
        # Stable sort: V0 outer, V_sd inner
        results_table.sort(key=lambda r: (r["V0"], r["V_sd"]))

    t_total = time.perf_counter() - t_start_all

    # ── compute differential conductance G = dI/dV_sd ──────────────────
    # Compute per-V_0 slice so G is a proper derivative in the V_sd direction.
    from itertools import groupby
    results_table.sort(key=lambda r: (r["V0"], r["V_sd"]))
    for V0_key, group_iter in groupby(results_table, key=lambda r: r["V0"]):
        slab = list(group_iter)
        _compute_G_1d(slab)  # adds G_K, G_Kp in place

    # ── summary table ──────────────────────────────────────────────────
    print(f"\n\n{'═' * 92}")
    print(f"  SWEEP SUMMARY — I–V and differential conductance")
    print(f"{'─' * 92}")
    print(f"  {'V0':>7s}  {'V_sd':>8s}  {'T_K':>7s}  {'T_Kp':>7s}  "
          f"{'I_K':>9s}  {'I_Kp':>9s}  {'G_K':>8s}  {'G_Kp':>8s}  "
          f"{'η':>8s}  {'t(s)':>6s}")
    print(f"{'─' * 92}")
    for row in results_table:
        print(f"  {row['V0']:+7.4f}  {row['V_sd']:+8.4f}  "
              f"{row['T_K']:7.4f}  {row['T_Kp']:7.4f}  "
              f"{row['I_K']:+9.5f}  {row['I_Kp']:+9.5f}  "
              f"{row.get('G_K', float('nan')):+8.4f}  "
              f"{row.get('G_Kp', float('nan')):+8.4f}  "
              f"{row['eta']:+8.5f}  {row['time_s']:6.1f}")
    print(f"{'═' * 92}")
    print(f"  Total time: {t_total:.0f} s  ({t_total/60:.1f} min)")
    print(f"  {len(results_table)} simulations completed.\n")

    # ── save JSON ──────────────────────────────────────────────────────
    for row in results_table:
        for k, v in row.items():
            if isinstance(v, (np.floating, np.float64)):
                row[k] = float(v) if not (isinstance(v, float) and np.isnan(v)) else None
            elif isinstance(v, np.integer):
                row[k] = int(v)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"{args.prefix}_sweep.json"
    with open(json_path, "w") as f:
        json.dump({
            "sweep":        "Vsd" if not is_2d else "V0_x_Vsd",
            "base_config":  args.config,
            "Ef":           Ef,
            "V0_values":    [float(v) for v in V0_values],
            "Vsd_values":   [float(v) for v in Vsd_values],
            "bias":         {
                "type":     args.bias_type,
                "x_center": args.bias_xc,
                "L_drop":   args.L_drop,
            },
            "results":      results_table,
        }, f, indent=2)
    print(f"  Sweep data saved → {json_path}")

    # ── plots ──────────────────────────────────────────────────────────
    if is_2d:
        _plot_2d_density(results_table, list(V0_values), list(Vsd_values),
                         out_root / f"{args.prefix}_2D_density.png", Ef)
    else:
        V0_here = float(V0_values[0])
        slab = [r for r in results_table if r["V0"] == V0_here]
        _plot_1d_IV(slab, out_root / f"{args.prefix}_IV_curve.png", V0_here, Ef)

    print()


if __name__ == "__main__":
    main()
