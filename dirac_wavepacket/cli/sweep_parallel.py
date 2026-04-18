#!/usr/bin/env python3
"""
Parallel V0 / geometry sweep for the transport simulator.

Key features in this version:
  - resumable sweeps via per-task checkpoint JSON files
  - aggregate sweep JSON created immediately at launch
  - append-only progress log written after each completed point
  - configurable aggregate JSON flush cadence during the run
  - optional rebuild of the aggregate JSON from existing checkpoints
  - optional serial mode with threaded FFT (workers=1, --fft-threads N)
  - optional worker recycling (maxtasksperchild) for long unstable jobs

Usage examples:
    # Regular parallel V0 sweep
    dwp-sweep --v0-min 0.0 --v0-max 1.6 --v0-step 0.01 --no-anim

    # Resume a crashed run
    dwp-sweep --resume --v0-min 0.0 --v0-max 1.6 --v0-step 0.01 --no-anim

    # Rebuild the JSON only from completed checkpoints (no simulations run)
    dwp-sweep --resume --rebuild-only --v0-min 0.0 --v0-max 1.6 --v0-step 0.01

    # Serial but threaded pyFFTW mode (good for fragile remote sessions)
    dwp-sweep --workers 1 --fft-backend pyfftw --fft-threads 8 --resume --no-anim

    # More stable multiprocessing: respawn workers after each task
    dwp-sweep --resume --maxtasksperchild 1 --workers 16 --no-anim

    # All examples also work via  `python -m dirac_wavepacket.cli.sweep_parallel`.
"""

# Pin BLAS/OpenMP before importing numpy. For sweep workloads, many single-thread
# workers almost always beat a few workers with internal BLAS threading.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import math
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np

from dirac_wavepacket.sweep_state import (
    append_jsonl,
    atomic_write_json,
    load_json,
    load_jsonl,
    load_task_row,
    normalise_results,
    num_tag,
    save_error_text,
    save_task_row,
    task_id_for_value,
    utc_now_iso,
)


def _set_fft_env(fft_backend: str | None, fft_threads: int | None) -> None:
    if fft_backend:
        os.environ["DIRAC2D_FFT_BACKEND"] = str(fft_backend)
    if fft_threads is not None:
        os.environ["DIRAC2D_FFT_THREADS"] = str(int(fft_threads))


def _build_output_json(
    *,
    args,
    results,
    Ef,
    json_path: Path,
    state_root: Path,
    status: str,
    n_total: int,
    n_failed: int,
    wy_tag: str = "",
) -> dict:
    sort_key = "V0" if args.sweep == "V0" else "W"
    rows = normalise_results(results, sort_key=sort_key)

    fft_backends = sorted({str(r.get("fft_backend", "unknown")) for r in rows if r.get("fft_backend") is not None})
    fft_threads = sorted({int(r["fft_threads"]) for r in rows if r.get("fft_threads") is not None})

    data = {
        "sweep": args.sweep if args.sweep == "geometry" else "V0",
        "base_config": args.config,
        "Ef": Ef,
        "status": status,
        "n_total": int(n_total),
        "n_completed": int(len(rows)),
        "n_failed": int(n_failed),
        "updated_at": utc_now_iso(),
        "json_path": str(json_path),
        "checkpoint_dir": str(state_root),
        "resume_supported": True,
        "fft_backend": fft_backends[0] if len(fft_backends) == 1 else (fft_backends or (args.fft_backend or "auto")),
        "fft_threads": fft_threads[0] if len(fft_threads) == 1 else (fft_threads or (args.fft_threads if args.fft_threads is not None else None)),
        "results": rows,
    }
    if args.wy is not None:
        data["tilt_wy"] = args.wy
    if args.sweep == "geometry":
        data["V0"] = args.v0
        data["x_face"] = args.x_face
        data["parameter"] = "W (triangle base width)"
    if wy_tag:
        data["prefix_tag"] = wy_tag
    dis_dv = getattr(args, "disorder_dv", 0.0)
    if dis_dv and dis_dv > 0:
        data["disorder"] = {
            "enabled": True,
            "dv": dis_dv,
            "lc": getattr(args, "disorder_lc", 3.0),
            "seed": getattr(args, "disorder_seed", None),
            "description": f"δV_rms = {dis_dv*100:.1f}% of V0, "
                           f"L_c = {getattr(args, 'disorder_lc', 3.0)}",
        }
    if getattr(args, "coupled", False):
        data["coupling"] = {
            "enabled": True,
            "strength": float(getattr(args, "coupling_strength", 0.0)),
            "type": str(getattr(args, "coupling_type", "barrier")),
            "region_threshold": float(getattr(args, "coupling_region_threshold", 0.5)),
            "line_position": float(getattr(args, "coupling_line_position", 0.0)),
            "line_width": float(getattr(args, "coupling_line_width", 2.0)),
            "description": (
                f"4-component coupled propagator, "
                f"u={getattr(args, 'coupling_strength', 0.0)} ({getattr(args, 'coupling_type', 'barrier')})"
            ),
        }
    return data


def _write_aggregate_json(*, args, results, Ef, json_path: Path, state_root: Path, status: str, n_total: int, n_failed: int, wy_tag: str) -> None:
    data = _build_output_json(
        args=args,
        results=results,
        Ef=Ef,
        json_path=json_path,
        state_root=state_root,
        status=status,
        n_total=n_total,
        n_failed=n_failed,
        wy_tag=wy_tag,
    )
    atomic_write_json(json_path, data)


def _make_task_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _run_single_task(task: dict) -> dict:
    """Worker body for one sweep point. Top-level for multiprocessing pickle."""
    # Keep BLAS pinned to 1 inside worker as well.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    _set_fft_env(task.get("fft_backend"), task.get("fft_threads"))

    from dirac_wavepacket.config import load_config
    from dirac_wavepacket.simulation import run_simulation
    from dirac_wavepacket.transport_metrics import valley_group_delay_from_results

    cfg = load_config(task["config_path"])
    cfg.output_dir = task["out_dir"]

    if task["sweep"] == "V0":
        V0 = float(task["value"])
        cfg.potential.height = V0
        wy = task.get("wy")
        if wy is not None:
            cfg.physics.tilt = [0.0, float(wy)]
    else:
        W = float(task["value"])
        cfg.potential.height = float(task["V0"])
        cfg.potential.type = "shaped"
        cfg.potential.edges = [task["x_face"], task["x_face"] + W, task["x_face"], task["x_face"]]

    # Apply electrostatic disorder if configured
    dis_dv = task.get("disorder_dv", 0.0)
    if dis_dv and dis_dv > 0:
        cfg.disorder.enabled = True
        cfg.disorder.dv = float(dis_dv)
        cfg.disorder.lc = float(task.get("disorder_lc", 3.0))
        cfg.disorder.seed = task.get("disorder_seed")

    # Apply intervalley coupling if configured (v2.0 coupled mode)
    if task.get("coupled", False):
        cfg.coupling.enabled = True
        cfg.coupling.strength = float(task.get("coupling_strength", 0.0))
        cfg.coupling.type = str(task.get("coupling_type", "barrier"))
        cfg.coupling.region_threshold = float(task.get("coupling_region_threshold", 0.5))
        cfg.coupling.line_position = float(task.get("coupling_line_position", 0.0))
        cfg.coupling.line_width = float(task.get("coupling_line_width", 2.0))
        # Coupled mode requires dual-valley simulation
        cfg.physics.valleys = "both"

    t0 = time.perf_counter()
    res = run_simulation(
        cfg,
        make_animation=not task["no_anim"],
        verbose=False,
        data_only=task["data_only"],
        measure_overlap=not task["no_overlap"],
        save_wf=task["save_wf"],
        wf_save_every=task["wf_save_every"],
    )
    elapsed = time.perf_counter() - t0

    Ef = cfg.physics.vf * cfg.wavepacket.k0
    row = {
        "Ef": Ef,
        "time_s": round(elapsed, 1),
        "fft_backend": res.get("fft_backend", "unknown"),
        "fft_threads": res.get("fft_threads", None),
        "T_K": res["T"],
        "T_Kp": res.get("T_Kp", res["T"]),
        "R_K": res["R"],
        "R_Kp": res.get("R_Kp", res["R"]),
        "eta": res.get("eta", 0.0),
    }

    if task["sweep"] == "V0":
        V0 = float(task["value"])
        row.update({
            "V0": V0,
            "V0_over_Ef": V0 / Ef if Ef > 0 else 0.0,
        })
        if task.get("wy") is not None:
            row["wy"] = float(task["wy"])
    else:
        W = float(task["value"])
        V0 = float(task["V0"])
        row.update({
            "W": W,
            "V0": V0,
            "V0_over_Ef": V0 / Ef if Ef > 0 else 0.0,
            "x_face": float(task["x_face"]),
            "xR_bot": float(task["x_face"] + W),
        })

    gd = valley_group_delay_from_results(res)
    row["delta_tau_T"] = gd["delta_tau_T"] if gd else np.nan

    # Record coupling metadata if v2.0 coupled mode was used
    if task.get("coupled", False):
        row["coupled"] = True
        row["coupling_strength"] = float(task.get("coupling_strength", 0.0))
        row["coupling_type"] = str(task.get("coupling_type", "barrier"))

    _make_task_dir(task["out_dir"])
    save_task_row(
        row,
        checkpoint_path=task["checkpoint_path"],
        task_result_path=task["task_result_path"],
        task_id=task["task_id"],
        task_meta={
            "sweep": task["sweep"],
            "value": float(task["value"]),
            "out_dir": str(task["out_dir"]),
        },
    )
    return row


def _row_task_id(row: dict, sweep: str) -> str | None:
    try:
        if sweep == "V0" and row.get("V0") is not None:
            return task_id_for_value("V0", float(row["V0"]), decimals=4)
        if sweep == "geometry" and row.get("W") is not None:
            return task_id_for_value("W", float(row["W"]), decimals=4)
    except Exception:
        return None
    return None


def _load_completed_rows(tasks: list[dict], *, sweep: str, json_path: Path | None = None, progress_path: Path | None = None) -> dict[str, dict]:
    task_ids = {task["task_id"] for task in tasks}
    completed: dict[str, dict] = {}

    for task in tasks:
        row = load_task_row(task["checkpoint_path"], task["task_result_path"])
        if row is not None:
            completed[task["task_id"]] = row

    if progress_path is not None:
        for item in load_jsonl(progress_path):
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            task_id = item.get("task_id") or _row_task_id(row, sweep)
            if task_id in task_ids and task_id not in completed:
                completed[task_id] = row

    if json_path is not None:
        data = load_json(json_path)
        rows = data.get("results") if isinstance(data, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                task_id = _row_task_id(row, sweep)
                if task_id in task_ids and task_id not in completed:
                    completed[task_id] = row

    return completed


def _task_key_display(row: dict, sweep: str) -> str:
    if sweep == "V0":
        return f"V0={row['V0']:.2f}"
    return f"W={row['W']:.0f}"


def _coupling_extras(args) -> dict:
    """Coupling fields shared by both V0 and geometry task dicts."""
    return {
        "coupled": bool(getattr(args, "coupled", False)),
        "coupling_strength": float(getattr(args, "coupling_strength", 0.0)),
        "coupling_type": str(getattr(args, "coupling_type", "barrier")),
        "coupling_region_threshold": float(getattr(args, "coupling_region_threshold", 0.5)),
        "coupling_line_position": float(getattr(args, "coupling_line_position", 0.0)),
        "coupling_line_width": float(getattr(args, "coupling_line_width", 2.0)),
    }


def _build_tasks(args, values, wy_tag: str, state_root: Path) -> tuple[list[dict], str]:
    rows_dir = state_root / "rows"

    tasks = []
    if args.sweep == "V0":
        for V0 in values:
            tag = f"{float(V0):.4f}".replace(".", "p")
            out_dir = Path(args.output_root) / f"{args.prefix}{wy_tag}_v0_{tag}"
            task_id = task_id_for_value("V0", float(V0), decimals=4)
            task = {
                "task_id": task_id,
                "sweep": "V0",
                "value": float(V0),
                "config_path": args.config,
                "wy": args.wy,
                "out_dir": str(out_dir),
                "no_anim": args.no_anim,
                "data_only": args.data_only,
                "no_overlap": args.no_overlap,
                "save_wf": args.save_wf,
                "wf_save_every": args.wf_save_every,
                "fft_backend": args.fft_backend,
                "fft_threads": args.fft_threads,
                "disorder_dv": getattr(args, "disorder_dv", 0.0),
                "disorder_lc": getattr(args, "disorder_lc", 3.0),
                "disorder_seed": getattr(args, "disorder_seed", None),
                "checkpoint_path": str(rows_dir / f"{task_id}.json"),
                "task_result_path": str(out_dir / "task_result.json"),
            }
            task.update(_coupling_extras(args))
            tasks.append(task)
        sweep_desc = f"V0 = {values[0]:.2f} → {values[-1]:.2f} (step {args.v0_step})"
        if args.wy is not None:
            sweep_desc += f", wy = {args.wy}"
    else:
        for W in values:
            tag = f"W{float(W):.2f}".replace(".", "p")
            out_dir = Path(args.output_root) / f"{args.prefix}_geom_{tag}"
            task_id = task_id_for_value("W", float(W), decimals=4)
            task = {
                "task_id": task_id,
                "sweep": "geometry",
                "value": float(W),
                "config_path": args.config,
                "V0": float(args.v0),
                "x_face": float(args.x_face),
                "out_dir": str(out_dir),
                "no_anim": args.no_anim,
                "data_only": args.data_only,
                "no_overlap": args.no_overlap,
                "save_wf": args.save_wf,
                "wf_save_every": args.wf_save_every,
                "fft_backend": args.fft_backend,
                "fft_threads": args.fft_threads,
                "disorder_dv": getattr(args, "disorder_dv", 0.0),
                "disorder_lc": getattr(args, "disorder_lc", 3.0),
                "disorder_seed": getattr(args, "disorder_seed", None),
                "checkpoint_path": str(rows_dir / f"{task_id}.json"),
                "task_result_path": str(out_dir / "task_result.json"),
            }
            task.update(_coupling_extras(args))
            tasks.append(task)
        sweep_desc = f"W = {values[0]:.0f} → {values[-1]:.0f} (step {args.w_step}), V0 = {args.v0}"
    return tasks, sweep_desc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable parallel sweep for the transport simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/reflecting_walls.yaml")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument("--prefix", default="vqubit_par")
    parser.add_argument("--sweep", choices=["V0", "geometry"], default="V0")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU core count)")

    # V0 sweep parameters
    parser.add_argument("--v0-min", type=float, default=0.0)
    parser.add_argument("--v0-max", type=float, default=4.0)
    parser.add_argument("--v0-step", type=float, default=0.01)
    parser.add_argument("--wy", type=float, default=None,
                        help="Override tilt wy (default: use config value)")

    # Geometry sweep parameters
    parser.add_argument("--v0", type=float, default=0.58,
                        help="Fixed V0 for geometry sweep (default: 0.58)")
    parser.add_argument("--x-face", type=float, default=-10.0)
    parser.add_argument("--w-min", type=float, default=10.0)
    parser.add_argument("--w-max", type=float, default=80.0)
    parser.add_argument("--w-step", type=float, default=5.0)

    # Output / diagnostics
    parser.add_argument("--no-anim", action="store_true")
    parser.add_argument("--data-only", action="store_true",
                        help="Skip all frame rendering and plots — pure data (fastest)")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Deprecated no-op kept for CLI compatibility")
    parser.add_argument("--save-wf", action="store_true",
                        help="Save spinor wavefunctions to .npz files for post-processing")
    parser.add_argument("--wf-save-every", type=int, default=None,
                        help="Save wavefunctions every N steps (default: 5×save_every)")

    # Robustness / resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from per-task checkpoints and existing task_result.json files")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Rebuild/update the aggregate sweep JSON from checkpoints only, then exit")
    parser.add_argument("--json-flush-every", type=int, default=1,
                        help="Rewrite the aggregate sweep JSON after every N completed sweep points (default: 1).")
    parser.add_argument("--maxtasksperchild", type=int, default=None,
                        help="Recycle workers after N tasks (improves stability on long runs)")
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None,
                        help="Optional multiprocessing start method; spawn is safer but slower")

    # FFT control
    parser.add_argument("--fft-backend", choices=["auto", "numpy", "pyfftw"], default=None,
                        help="Forwarded to DIRAC2D_FFT_BACKEND inside workers")
    parser.add_argument("--fft-threads", type=int, default=None,
                        help="Forwarded to DIRAC2D_FFT_THREADS inside workers")

    # Electrostatic disorder (charge puddles)
    parser.add_argument("--disorder-dv", type=float, default=0.0,
                        help="Disorder amplitude as fraction of V0 (e.g. 0.05 = 5%%). "
                             "Adds spatially correlated δV(x,y) inside the barrier.")
    parser.add_argument("--disorder-lc", type=float, default=3.0,
                        help="Disorder correlation length in sim units (default: 3.0). "
                             "Controls charge-puddle size.")
    parser.add_argument("--disorder-seed", type=int, default=None,
                        help="Random seed for reproducible disorder realisations")

    # Intervalley coupling (v2.0 coupled mode)
    parser.add_argument("--coupled", action="store_true",
                        help="Use the 4-component coupled propagator (v2.0). "
                             "Required to enable intervalley coupling. With "
                             "--coupling-strength=0 this gives bit-equivalent "
                             "v1.0 results from a single coupled run.")
    parser.add_argument("--coupling-strength", type=float, default=0.0,
                        help="Intervalley coupling amplitude in sim energy units. "
                             "Only effective when --coupled is set. (default: 0.0)")
    parser.add_argument("--coupling-type", choices=["barrier", "uniform", "defect_line"],
                        default="barrier",
                        help="Spatial profile of the intervalley coupling field: "
                             "'barrier' restricts u to V > threshold·|V0|; "
                             "'uniform' applies u everywhere; "
                             "'defect_line' Gaussian along x = line_position. "
                             "(default: barrier)")
    parser.add_argument("--coupling-region-threshold", type=float, default=0.5,
                        help="For --coupling-type=barrier: V > threshold·|V0| → coupling active. "
                             "(default: 0.5)")
    parser.add_argument("--coupling-line-position", type=float, default=0.0,
                        help="For --coupling-type=defect_line: x-coordinate of line centre. "
                             "(default: 0.0)")
    parser.add_argument("--coupling-line-width", type=float, default=2.0,
                        help="For --coupling-type=defect_line: Gaussian width. "
                             "(default: 2.0)")

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rebuild_only and not args.resume:
        parser.error("--rebuild-only expects --resume so the script knows to load checkpoints.")
    if args.json_flush_every < 1:
        parser.error("--json-flush-every must be >= 1")

    n_workers = max(1, int(args.workers or os.cpu_count() or 1))
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    _set_fft_env(args.fft_backend, args.fft_threads)

    if args.fft_threads is not None and args.fft_threads > 1 and n_workers > 1:
        print("  [warn] fft_threads > 1 with multiple workers will usually oversubscribe the CPU.")
        print("         Best practice for sweeps: workers > 1 and fft_threads = 1.")
        print("         Best practice for one serial run: workers = 1 and fft_threads > 1.\n")

    # Build values / tasks.
    if args.sweep == "V0":
        values = np.arange(args.v0_min, args.v0_max + 1e-9, args.v0_step)
        values = np.round(values, 4)
        wy_tag = ""
        if args.wy is not None:
            wy_tag = f"_wy{abs(args.wy):.2f}".replace(".", "p")
    else:
        values = np.arange(args.w_min, args.w_max + 1e-9, args.w_step)
        values = np.round(values, 2)
        wy_tag = ""

    state_stem = f"{args.prefix}{wy_tag if args.sweep == 'V0' else '_geom'}_sweep_state"
    state_root = Path(args.output_root) / state_stem
    rows_dir = state_root / "rows"
    errors_dir = state_root / "errors"
    rows_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    tasks, sweep_desc = _build_tasks(args, values, wy_tag, state_root)

    if args.resume:
        existing_manifest = load_json(state_root / "manifest.json")
        if isinstance(existing_manifest, dict):
            mismatches = []
            if existing_manifest.get("sweep") != args.sweep:
                mismatches.append("sweep type")
            if existing_manifest.get("base_config") != args.config:
                mismatches.append("base config")
            old_values = existing_manifest.get("values")
            if isinstance(old_values, list):
                cur_values = [float(v) for v in values]
                if len(old_values) != len(cur_values) or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(old_values, cur_values)):
                    mismatches.append("sweep values")
            if args.sweep == "V0" and existing_manifest.get("wy") != args.wy:
                mismatches.append("wy")
            if args.sweep == "geometry":
                if float(existing_manifest.get("v0", args.v0)) != float(args.v0):
                    mismatches.append("geometry V0")
                if float(existing_manifest.get("x_face", args.x_face)) != float(args.x_face):
                    mismatches.append("x_face")
            if mismatches:
                print("  [error] Existing checkpoint state does not match the current sweep plan:")
                for item in mismatches:
                    print(f"          - {item}")
                print("          Use a new --prefix or clear the old state directory before resuming.\n")
                return 2
    n_tasks = len(tasks)

    from dirac_wavepacket.config import load_config
    _cfg = load_config(args.config)
    Ef = _cfg.physics.vf * _cfg.wavepacket.k0
    Nx, Ny = _cfg.grid.Nx, _cfg.grid.Ny

    # Aggregate JSON is now written throughout the run.
    json_name = f"{args.prefix}{wy_tag if args.sweep == 'V0' else '_geom'}_sweep.json"
    json_path = Path(args.output_root) / json_name
    progress_path = state_root / "progress.jsonl"

    manifest = {
        "created_at": utc_now_iso(),
        "sweep": args.sweep,
        "base_config": args.config,
        "output_root": args.output_root,
        "prefix": args.prefix,
        "wy": args.wy,
        "v0": args.v0,
        "x_face": args.x_face,
        "value_count": len(values),
        "values": [float(v) for v in values],
        "json_path": str(json_path),
        "progress_path": str(progress_path),
        "rows_dir": str(rows_dir),
    }
    atomic_write_json(state_root / "manifest.json", manifest)

    # Memory estimate from config.
    mem_per_worker_gb = (Nx * Ny * 16 * 20) / 1e9  # ~20 arrays per worker
    total_mem_gb = n_workers * mem_per_worker_gb

    est_per_task = 1.5 if args.data_only else 3.0
    est_total_seq = n_tasks * est_per_task
    est_total_par = est_total_seq / max(1, n_workers)

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║   Resumable sweep — {n_workers} worker{'s' if n_workers != 1 else ' '}                     ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝\n")
    print(f"  Config       : {args.config}")
    print(f"  Grid         : {Nx}×{Ny}")
    print(f"  Sweep        : {sweep_desc}")
    print(f"  Tasks        : {n_tasks}")
    print(f"  Workers      : {n_workers} / {os.cpu_count()} cores")
    print("  Phase mode   : transport only (overlap via post-process)")
    print(f"  Save ψ       : {'ON' if args.save_wf else 'OFF'}")
    print(f"  BLAS threads : OMP_NUM_THREADS=1 (pinned)")
    print(f"  FFT backend  : {args.fft_backend or os.environ.get('DIRAC2D_FFT_BACKEND', 'auto')}")
    print(f"  FFT threads  : {args.fft_threads if args.fft_threads is not None else os.environ.get('DIRAC2D_FFT_THREADS', 'auto')}")
    if args.coupled:
        print(f"  Coupling     : v2.0 COUPLED  u={args.coupling_strength}  type={args.coupling_type}")
    else:
        print("  Coupling     : v1.0 INDEPENDENT (no intervalley coupling)")
    print(f"  Resume       : {'ON' if args.resume else 'OFF'}")
    print(f"  State dir    : {state_root}")
    print(f"  Live JSON    : {json_path}")
    print(f"  Progress log : {progress_path}")
    print(f"  JSON flush   : every {args.json_flush_every} completed point(s)")
    print(f"  Memory       : ~{mem_per_worker_gb:.1f} GB/worker × {n_workers} = ~{total_mem_gb:.0f} GB total")
    print(f"  Est. time    : ~{est_total_par:.0f} min parallel ({est_total_seq:.0f} min sequential)")
    if args.maxtasksperchild is not None:
        print(f"  Recycle      : worker respawn every {args.maxtasksperchild} task(s)")
    if args.start_method is not None:
        print(f"  Start method : {args.start_method}")
    if args.disorder_dv > 0:
        print(f"  Disorder     : δV_rms={args.disorder_dv*100:.1f}% of V0  "
              f"L_c={args.disorder_lc}  seed={args.disorder_seed}")
    print()

    if total_mem_gb > 0.8 * 512:
        print(f"  ⚠ High memory usage ({total_mem_gb:.0f} GB). Consider fewer workers.")
        print()

    if args.dry_run:
        print(f"  [dry-run] {n_tasks} tasks planned. Remove --dry-run to execute.\n")
        return 0

    # Load completed rows if resuming.
    completed_by_id: dict[str, dict] = {}
    if args.resume:
        completed_by_id = _load_completed_rows(
            tasks,
            sweep=args.sweep,
            json_path=json_path,
            progress_path=progress_path,
        )
        print(f"  Resume scan  : found {len(completed_by_id)} completed task row(s).")
        print()
    else:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    # Always write an initial aggregate JSON so there is never a run with no JSON.
    _write_aggregate_json(
        args=args,
        results=list(completed_by_id.values()),
        Ef=Ef,
        json_path=json_path,
        state_root=state_root,
        status="running" if len(completed_by_id) < n_tasks else "complete",
        n_total=n_tasks,
        n_failed=0,
        wy_tag=wy_tag,
    )
    if not args.resume or not progress_path.exists() or progress_path.stat().st_size == 0:
        append_jsonl(progress_path, {
            "type": "manifest",
            "created_at": utc_now_iso(),
            "sweep": args.sweep,
            "json_path": str(json_path),
            "n_total": int(n_tasks),
            "json_flush_every": int(args.json_flush_every),
        })

    if args.rebuild_only:
        print(f"  Rebuilt aggregate JSON from checkpoints → {json_path}\n")
        return 0

    pending_tasks = [t for t in tasks if t["task_id"] not in completed_by_id]
    if not pending_tasks:
        print(f"  Nothing to do — all {n_tasks} tasks are already complete.\n")
        return 0

    t_start = time.perf_counter()
    completed_live = 0
    failed = 0
    last_heartbeat = t_start
    last_json_flush = t_start
    dirty_aggregate = False

    print(f"  Submitting {len(pending_tasks)} pending task(s) to {n_workers} worker(s) ...", flush=True)

    # Choose multiprocessing context.
    ctx = mp.get_context(args.start_method) if args.start_method else mp

    try:
        with ctx.Pool(processes=n_workers, maxtasksperchild=args.maxtasksperchild) as pool:
            async_results = [(task, pool.apply_async(_run_single_task, (task,))) for task in pending_tasks]
            pending = list(range(len(async_results)))

            while pending:
                still_pending = []
                for idx in pending:
                    task, ar = async_results[idx]
                    if ar.ready():
                        try:
                            row = ar.get()
                        except Exception as e:
                            failed += 1
                            err_text = (
                                f"task_id: {task['task_id']}\n"
                                f"out_dir: {task['out_dir']}\n"
                                f"time: {utc_now_iso()}\n\n"
                                f"{type(e).__name__}: {e}\n\n"
                                f"{traceback.format_exc()}"
                            )
                            save_error_text(errors_dir / f"{task['task_id']}.txt", err_text)
                            print(f"  ⚠ Task {task['task_id']} FAILED: {e}", flush=True)
                            continue

                        completed_live += 1
                        completed_by_id[task["task_id"]] = row
                        append_jsonl(progress_path, {
                            "type": "row",
                            "saved_at": utc_now_iso(),
                            "task_id": task["task_id"],
                            "row": row,
                        })
                        dirty_aggregate = True

                        if (len(completed_by_id) % args.json_flush_every) == 0:
                            _write_aggregate_json(
                                args=args,
                                results=list(completed_by_id.values()),
                                Ef=Ef,
                                json_path=json_path,
                                state_root=state_root,
                                status="running",
                                n_total=n_tasks,
                                n_failed=failed,
                                wy_tag=wy_tag,
                            )
                            dirty_aggregate = False
                            last_json_flush = time.perf_counter()

                        elapsed = time.perf_counter() - t_start
                        total_done = len(completed_by_id)
                        newly_done = completed_live
                        rate = newly_done / elapsed * 60 if elapsed > 0 else 0.0
                        remaining = n_tasks - total_done
                        eta = remaining / rate if rate > 0 else math.inf

                        key = _task_key_display(row, args.sweep)
                        dtau = row.get("delta_tau_T", np.nan)
                        dtau_str = ""
                        if dtau is not None and not (isinstance(dtau, float) and math.isnan(dtau)):
                            dtau_str = f"  Δτ={dtau:+.2f}"

                        eta_str = f"ETA {eta:.0f}min" if math.isfinite(eta) else "ETA --"
                        print(
                            f"  [{total_done:4d}/{n_tasks}]  {key}  "
                            f"T_K={row['T_K']:.3f}  T_Kp={row['T_Kp']:.3f}"
                            f"{dtau_str}  ({row['time_s']:.0f}s)  "
                            f"[{rate:.1f}/min  {eta_str}]",
                            flush=True,
                        )
                    else:
                        still_pending.append(idx)

                pending = still_pending

                now = time.perf_counter()
                if dirty_aggregate and (now - last_json_flush) > 30:
                    _write_aggregate_json(
                        args=args,
                        results=list(completed_by_id.values()),
                        Ef=Ef,
                        json_path=json_path,
                        state_root=state_root,
                        status="running",
                        n_total=n_tasks,
                        n_failed=failed,
                        wy_tag=wy_tag,
                    )
                    dirty_aggregate = False
                    last_json_flush = now
                if pending and (now - last_heartbeat) > 30:
                    elapsed = now - t_start
                    print(
                        f"  ... {len(completed_by_id)}/{n_tasks} done, "
                        f"{len(pending)} running  [{elapsed:.0f}s elapsed]",
                        flush=True,
                    )
                    last_heartbeat = now

                if pending:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  Interrupted — current aggregate JSON and per-task checkpoints were preserved.", flush=True)
    finally:
        final_status = "complete" if len(completed_by_id) == n_tasks and failed == 0 else "partial"
        _write_aggregate_json(
            args=args,
            results=list(completed_by_id.values()),
            Ef=Ef,
            json_path=json_path,
            state_root=state_root,
            status=final_status,
            n_total=n_tasks,
            n_failed=failed,
            wy_tag=wy_tag,
        )
        append_jsonl(progress_path, {
            "type": "summary",
            "saved_at": utc_now_iso(),
            "status": final_status,
            "n_completed": int(len(completed_by_id)),
            "n_total": int(n_tasks),
            "n_failed": int(failed),
            "json_path": str(json_path),
        })

    t_total = time.perf_counter() - t_start
    rows = normalise_results(list(completed_by_id.values()), sort_key=("V0" if args.sweep == "V0" else "W"))
    fft_backends = sorted({str(r.get("fft_backend", "unknown")) for r in rows if r.get("fft_backend") is not None})
    fft_threads = sorted({int(r["fft_threads"]) for r in rows if r.get("fft_threads") is not None})

    print(f"\n  {'═' * 64}")
    if len(completed_by_id) == n_tasks and failed == 0:
        print(f"  COMPLETE: {n_tasks} tasks in {t_total:.0f}s ({t_total/60:.1f} min)")
    else:
        print(f"  PARTIAL : {len(completed_by_id)}/{n_tasks} tasks complete, {failed} failed")
        print(f"  Resume with: dwp-sweep --resume ...")
    if t_total > 0 and completed_live > 0:
        speedup = est_total_seq * 60 / t_total
        print(f"  Speedup:   {speedup:.1f}× vs sequential estimate")
    print(f"  FFT seen:  {fft_backends[0] if len(fft_backends) == 1 else (fft_backends or 'unknown')}  (threads={fft_threads[0] if len(fft_threads) == 1 else (fft_threads or 'unknown')})")
    print(f"  Saved:     {json_path}")
    print(f"  State dir: {state_root}")
    print(f"  {'═' * 64}\n")
    print(f"  Load JSON in your post-processing script: {json_path}\n")

    return 0 if len(completed_by_id) == n_tasks and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
