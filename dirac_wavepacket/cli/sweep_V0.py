#!/usr/bin/env python3
"""
Valley qubit phase gate — V0 sweep.

Sweeps barrier height V0 from 0 to 6 while keeping Ef = 2 (k0 = 2, vf = 1).
Each run uses the reflecting_walls.yaml base config with the vertical-face-first
triangle geometry (phase gate orientation).

Usage:
    python sweep_V0.py                        # full sweep, default config
    python sweep_V0.py --config configs/reflecting_walls.yaml
    python sweep_V0.py --v0-min 1 --v0-max 4 --v0-step 0.5
    python sweep_V0.py --dry-run              # print configs without running
"""

import argparse
import os
import sys
import json
import time
import multiprocessing as _mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force 'spawn' so each worker gets a clean interpreter and our thread
# env vars take effect BEFORE numpy/pyFFTW are imported. 'fork' (Linux
# default) inherits the parent's already-initialized BLAS/FFT thread
# pools — which means OMP_NUM_THREADS=8 set inside the worker has no
# effect, and all runs end up single-threaded.
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set by caller

from dirac_wavepacket.config import load_config, config_summary
from dirac_wavepacket.simulation import run_simulation
from dirac_wavepacket.transport_metrics import valley_group_delay_from_results


def _run_one_V0(job: dict) -> dict:
    """
    Worker executed in a subprocess. Sets per-process thread caps BEFORE
    importing numpy-heavy code, then runs a single V0 case and returns a
    summary row.

    Thread caps matter because each FFT/BLAS library independently grabs
    all cores by default; N jobs × all-cores = massive oversubscription
    and 1.0x speedup instead of N.
    """
    threads = int(job["threads"])
    # DIRAC2D_FFT_THREADS is what _resolve_fft_backend() reads — this is
    # the one that actually controls per-run pyFFTW parallelism. The
    # others (OMP, MKL, OPENBLAS) still matter for BLAS-heavy operations
    # if numpy is linked against a threaded BLAS (MKL/OpenBLAS). On
    # systems with reference BLAS (Ubuntu libblas 3.x) they are inert.
    os.environ["DIRAC2D_FFT_THREADS"] = str(threads)
    os.environ["DIRAC2D_FFT_BACKEND"] = "pyfftw"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "PYFFTW_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)

    # Re-import inside subprocess so env vars take effect
    from dirac_wavepacket.config import load_config
    from dirac_wavepacket.simulation import run_simulation
    from dirac_wavepacket.transport_metrics import valley_group_delay_from_results

    # Verify thread setup — print from one worker so the user can confirm
    # threads > 1 without scanning htop.
    if job.get("verbose_threads", False):
        try:
            import pyfftw
            pf_ok = True
        except ImportError:
            pf_ok = False
        print(f"  [pid={os.getpid()}] V0={job['V0']:.3f}  "
              f"DIRAC2D_FFT_THREADS={os.environ['DIRAC2D_FFT_THREADS']}  "
              f"backend={os.environ['DIRAC2D_FFT_BACKEND']}  "
              f"pyFFTW={'OK' if pf_ok else 'MISSING'}", flush=True)

    V0 = float(job["V0"])
    cfg = load_config(job["config_path"])
    cfg.potential.height = V0
    cfg.output_dir = job["out_dir"]

    if job.get("source_mode") is not None:
        cfg.physics.source_mode = job["source_mode"]
    if job.get("disorder"):
        d = job["disorder"]
        cfg.disorder.enabled = True
        cfg.disorder.dv = d["dv"]
        cfg.disorder.lc = d["lc"]
        cfg.disorder.seed = d["seed"]

    t0 = time.perf_counter()
    res = run_simulation(cfg, make_animation=job["make_anim"],
                         save_wf=job.get("save_wf", False),
                         verbose=False)
    t_elapsed = time.perf_counter() - t0

    Ef = cfg.physics.vf * cfg.wavepacket.k0
    row = {
        "V0":         V0,
        "Ef":         Ef,
        "V0_over_Ef": V0 / Ef if Ef > 0 else 0,
        "T_K":        res["T"],
        "T_Kp":       res.get("T_Kp", res["T"]),
        "R_K":        res["R"],
        "R_Kp":       res.get("R_Kp", res["R"]),
        "eta":        res.get("eta", 0.0),
        "time_s":     round(t_elapsed, 1),
    }
    gd = valley_group_delay_from_results(res)
    row["delta_tau_T"] = gd["delta_tau_T"] if gd else float("nan")
    return row



def build_parser():
    p = argparse.ArgumentParser(
        description="Sweep V0 for valley qubit phase gate characterisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/reflecting_walls.yaml",
                   help="Base YAML config (default: configs/reflecting_walls.yaml)")
    p.add_argument("--output-root", default="./results",
                   help="Root output directory (default: ./results)")
    p.add_argument("--prefix", default="vqubit_v0",
                   help="Output folder prefix (default: vqubit_v0)")
    p.add_argument("--v0-min", type=float, default=0.0)
    p.add_argument("--v0-max", type=float, default=6.0)
    p.add_argument("--v0-step", type=float, default=0.5)
    p.add_argument("--no-anim", action="store_true",
                   help="Skip GIF generation (saves time)")

    # Electrostatic disorder (charge puddles)
    p.add_argument("--disorder-dv", type=float, default=0.0,
                   help="Disorder amplitude as fraction of V0 (e.g. 0.05 = 5%%). "
                        "Adds spatially correlated δV(x,y) inside the barrier.")
    p.add_argument("--disorder-lc", type=float, default=3.0,
                   help="Disorder correlation length in sim units (default: 3.0). "
                        "Controls charge-puddle size.")
    p.add_argument("--disorder-seed", type=int, default=None,
                   help="Random seed for reproducible disorder realisations")

    p.add_argument("--source-mode", default=None,
                   choices=["both", "K_only", "Kp_only"],
                   help="Which valley block is sourced at t=0 in the coupled "
                        "4-spinor. Overrides physics.source_mode from YAML. "
                        "Use K_only / Kp_only for readout-visibility runs "
                        "(two runs per V0 with coupling enabled).")
    p.add_argument("--subdir", default=None,
                   help="Optional subdirectory under each per-V0 run folder, "
                        "e.g. --subdir K_only --source-mode K_only to produce "
                        "the layout expected by compute_readout.py")

    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Number of V0 points to run in parallel (default: 1). "
                        "On a 128-core box, try --jobs 16 with "
                        "--threads-per-job 8 for a 16-point sweep.")
    p.add_argument("--threads-per-job", type=int, default=None,
                   help="FFT/BLAS threads per subprocess. Default: "
                        "total_cpus // jobs, capped at 16 (pyFFTW 1024² "
                        "FFTs saturate around there).")
    p.add_argument("--save-wf", action="store_true",
                   help="Save raw wavefunction snapshots to "
                        "{out_dir}/wavefunction/wf_*.npz + grid.npz + "
                        "index.json. REQUIRED for compute_readout.py and "
                        "compute_coupling_scan.py post-processing. Adds "
                        "~1-5 GB disk per run depending on save_every.")

    p.add_argument("--dry-run", action="store_true",
                   help="Print sweep plan without running simulations")
    return p


def main():
    args = build_parser().parse_args()

    base_cfg = load_config(args.config)
    V0_values = np.arange(args.v0_min, args.v0_max + 1e-9, args.v0_step)
    V0_values = np.round(V0_values, 4)

    Ef = base_cfg.physics.vf * base_cfg.wavepacket.k0

    # Disorder settings
    has_disorder = args.disorder_dv > 0

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║   Valley qubit phase gate — V0 sweep             ║")
    print(f"  ╚══════════════════════════════════════════════════╝\n")
    print(f"  Base config : {args.config}")
    print(f"  Ef          : {Ef:.2f}")
    print(f"  V0 range    : {V0_values[0]:.2f} → {V0_values[-1]:.2f}  "
          f"(step {args.v0_step}, {len(V0_values)} runs)")
    if has_disorder:
        print(f"  Disorder    : δV_rms={args.disorder_dv*100:.1f}% of V0  "
              f"L_c={args.disorder_lc}  seed={args.disorder_seed}")
    print(f"  Output root : {args.output_root}")
    print(f"  Animation   : {'off' if args.no_anim else 'on'}")
    print()

    if args.dry_run:
        for V0 in V0_values:
            tag = f"{V0:.4f}".replace(".", "p")
            out_dir = f"{args.output_root}/{args.prefix}_{tag}"
            print(f"  [dry-run] V0={V0:.2f}  → {out_dir}")
        print(f"\n  {len(V0_values)} runs planned. Remove --dry-run to execute.\n")
        return

    # ── sweep ─────────────────────────────────────────────────────────
    # ── Parallel dispatch ─────────────────────────────────────────────
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

    # Build job specs
    jobs = []
    for idx, V0 in enumerate(V0_values):
        tag = f"{float(V0):.4f}".replace(".", "p")
        if args.subdir:
            out_dir = f"{args.output_root}/{args.prefix}_{tag}/{args.subdir}"
        else:
            out_dir = f"{args.output_root}/{args.prefix}_{tag}"
        jobs.append({
            "V0":          float(V0),
            "config_path": args.config,
            "out_dir":     out_dir,
            "source_mode": args.source_mode,
            "make_anim":   not args.no_anim,
            "save_wf":     args.save_wf,
            "threads":     threads,
            "verbose_threads": (idx < min(2, n_jobs)),
            "disorder":    ({"dv": args.disorder_dv,
                             "lc": args.disorder_lc,
                             "seed": args.disorder_seed}
                            if has_disorder else None),
        })

    results_table = []
    t_start_all = time.perf_counter()

    if n_jobs == 1:
        for i, job in enumerate(jobs):
            print(f"\n{'━' * 60}")
            print(f"  RUN {i+1}/{len(jobs)}:  V0 = {job['V0']:.2f}  →  {job['out_dir']}")
            print(f"{'━' * 60}")
            results_table.append(_run_one_V0(job))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_run_one_V0, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    row = fut.result()
                    results_table.append(row)
                    done += 1
                    print(f"  [{done}/{len(jobs)}] V0={j['V0']:.3f} done "
                          f"in {row['time_s']:.1f}s  T_K={row['T_K']:.4f} "
                          f"T_Kp={row['T_Kp']:.4f}", flush=True)
                except Exception as e:
                    done += 1
                    print(f"  [{done}/{len(jobs)}] V0={j['V0']:.3f} FAILED: {e}",
                          flush=True)
        results_table.sort(key=lambda r: r["V0"])

    t_total = time.perf_counter() - t_start_all

    # ── summary table ─────────────────────────────────────────────────
    print(f"\n\n{'═' * 80}")
    print(f"  SWEEP SUMMARY — V0 dependence of valley phase gate")
    print(f"{'─' * 80}")
    print(f"  {'V0':>5s}  {'V0/Ef':>6s}  {'T_K':>7s}  {'T_Kp':>7s}  "
          f"{'η':>8s}  {'Δτ_T':>8s}  {'t(s)':>6s}")
    print(f"{'─' * 80}")

    for row in results_table:
        print(f"  {row['V0']:5.2f}  {row['V0_over_Ef']:6.2f}  "
              f"{row['T_K']:7.4f}  {row['T_Kp']:7.4f}  "
              f"{row['eta']:+8.5f}  {row['delta_tau_T']:+8.3f}  "
              f"{row['time_s']:6.1f}")

    print(f"{'═' * 80}")
    print(f"  Total time: {t_total:.0f} s  ({t_total/60:.1f} min)")
    print(f"  {len(V0_values)} simulations completed.\n")

    # ── save to JSON ──────────────────────────────────────────────────
    json_path = Path(args.output_root) / f"{args.prefix}_sweep.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialisation
    for row in results_table:
        for k, v in row.items():
            if isinstance(v, (np.floating, np.float64)):
                row[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, np.integer):
                row[k] = int(v)

    with open(json_path, "w") as f:
        payload = {
            "sweep":      "V0",
            "base_config": args.config,
            "Ef":          Ef,
            "results":     results_table,
        }
        if has_disorder:
            payload["disorder"] = {
                "enabled":    True,
                "dv":         args.disorder_dv,
                "lc":         args.disorder_lc,
                "seed":       args.disorder_seed,
                "description": f"δV_rms = {args.disorder_dv*100:.1f}% of V0, "
                               f"L_c = {args.disorder_lc}",
            }
        json.dump(payload, f, indent=2)
    print(f"  Sweep data saved → {json_path}\n")


if __name__ == "__main__":
    main()
