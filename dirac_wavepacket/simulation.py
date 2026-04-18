"""
Simulation module: transport-only wavepacket propagation with drain-contact T/R measurement.

The absorbers ARE the detectors:
  - Right drain absorbs → T(t)
  - Left drain absorbs  → R(t)
  - T + R + P_remaining = 1 exactly at every step
    (plus P_y for absorbing y-walls)

Workflow:
  1. Validate grid / auto-size domain / auto-position source
  2. Build grid, potential, DrainContacts
  3. Initialise Gaussian spinor wavepacket
  4. Time-advance (split-operator FFT)
  5. Each step: propagate → drain.apply(psi) → T,R accumulate
  6. Auto-stop when P_remaining < threshold
  7. Save frames, GIF, T/R report

Phase tracking and overlap extraction are intentionally NOT part of
this module. The live simulation is kept transport-only; any phase /
overlap post-processing is done externally by loading the saved
wavefunction snapshots (``save_wf=True``) from disk.
"""

from __future__ import annotations

import math
import time as _time
from pathlib import Path

import numpy as np

from .absorber import DrainContacts, build_mass_confinement
from .auto import auto_domain, auto_source_x, validate_grid
from .config import SimConfig, config_summary
from .grid import Grid
from .potential import build_potential, build_disorder, add_source_drain_bias
from .propagator import SplitOperatorPropagator
from .valley_propagator import build_valley_propagator
from .visualizer import (
    make_gif,
    plot_final_snapshot,
    plot_final_snapshot_dual,
    plot_probability_history,
    plot_transmission_report,
    plot_transmission_report_dual,
    render_frame,
    render_frame_dual,
)
from .wavepacket import init_wavepacket, total_probability


def _apply_auto_grid(cfg: SimConfig, verbose: bool = True):
    """If detector.auto_grid is True, replace grid/source with optimal values."""
    dc = cfg.detector
    rec = auto_domain(
        cfg,
        n_sigma_source=dc.n_sigma_source,
        n_sigma_detector=dc.n_sigma_det,
        pts_per_wavelength=dc.pts_per_wavelength,
    )
    if verbose:
        print(f"  [auto_grid] Domain: {cfg.grid.Lx}×{cfg.grid.Ly} "
              f"→ {rec['Lx']}×{rec['Ly']}")
        print(f"  [auto_grid] Grid  : {cfg.grid.Nx}×{cfg.grid.Ny} "
              f"→ {rec['Nx']}×{rec['Ny']}  (dx={rec['dx']:.3f})")
    cfg.grid.Lx = rec["Lx"]
    cfg.grid.Ly = rec["Ly"]
    cfg.grid.Nx = rec["Nx"]
    cfg.grid.Ny = rec["Ny"]
    if cfg.time.dt > rec["dt_max"]:
        new_dt = round(rec["dt_max"] * 0.8, 5)
        if verbose:
            print(f"  [auto_grid] dt: {cfg.time.dt} → {new_dt} "
                  f"(stability limit {rec['dt_max']:.5f})")
        cfg.time.dt = new_dt


def _apply_auto_source(cfg: SimConfig, verbose: bool = True):
    """If detector.auto_source is True, reposition the source optimally."""
    dc = cfg.detector
    x_src = auto_source_x(cfg, n_sigma=dc.n_sigma_source)
    frac = x_src / (cfg.grid.Lx / 2.0)
    if verbose:
        old_x = cfg.wavepacket.x_source * (cfg.grid.Lx / 2.0)
        print(f"  [auto_source] x_source: {old_x:.2f} → {x_src:.2f} "
              f"(frac {cfg.wavepacket.x_source:.3f} → {frac:.3f})")
    cfg.wavepacket.x_source = frac


def run_simulation(
    cfg: SimConfig,
    make_animation: bool = True,
    fps: int = 15,
    verbose: bool = True,
    data_only: bool = False,
    measure_overlap: bool = True,
    save_wf: bool = False,
    wf_save_every: int | None = None,
) -> dict:
    """
    Run the split-operator FFT transport simulation with drain-contact T/R.

    Supports three valley modes (cfg.physics.valleys):
      "K"    — single valley with tilt as given
      "Kp"   — single valley with negated tilt (K′)
      "both" — both valleys in parallel, side-by-side output + η(t)

    Parameters
    ----------
    data_only : bool
        If True, skip all frame rendering and post-processing plots.
        Only compute transport observables.
    measure_overlap : bool
        Retained for backward API compatibility, but ignored.
        Overlap / phase tracking is no longer performed inside the
        live transport loop. Use ``save_wf=True`` to dump raw spinor
        snapshots and post-process them externally.
    save_wf : bool
        If True, dump raw barrier-propagated wavefunction snapshots
        (complex64) to output_dir/wavefunction/ for post-processing.
    wf_save_every : int or None
        Save wavefunction every N steps. Defaults to cfg.time.save_every.

    Returns
    -------
    dict with keys:
        'times', 'probabilities', 'T', 'R', 'T_history', 'R_history',
        'det_times', 'output_dir', 'n_steps_actual', 'auto_stopped',
        'fft_backend', 'fft_threads'
    For dual-valley mode, also includes:
        'T_Kp', 'R_Kp', 'T_history_Kp', 'R_history_Kp',
        'probabilities_Kp', 'eta', 'polarization'
    """
    dc = cfg.detector
    p = cfg.physics
    valleys = p.valleys.lower().replace("'", "p")
    dual = valleys == "both"

    if valleys == "kp":
        p.tilt = [-p.tilt[0], -p.tilt[1]]
        _single_chirality = -1
    else:
        _single_chirality = +1

    if dc.auto_grid:
        _apply_auto_grid(cfg, verbose)

    warnings = validate_grid(cfg)
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
        print()

    if dc.auto_source:
        _apply_auto_source(cfg, verbose)

    a = cfg.absorber
    if not a.x_only and a.y_bc == "periodic":
        a.y_bc = "absorbing"
    if (cfg.potential.type in ("barrier", "pn_junction", "double_barrier")
            and a.y_bc == "periodic"):
        a.x_only = True
    y_bc = a.y_bc

    out_dir = Path(cfg.output_dir)
    frame_dir = out_dir / "frames"
    if not data_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(config_summary(cfg))
        if dual:
            print("  *** Dual-valley mode: K and K′ side by side ***\n")
        if measure_overlap:
            print("  [note] Live overlap / phase tracking is disabled in "
                  "transport mode. Use save_wf=True and post-process the "
                  "snapshots externally.\n")

    g = cfg.grid
    grid = Grid(Nx=g.Nx, Ny=g.Ny, Lx=g.Lx, Ly=g.Ly)
    if verbose:
        print(f"  Grid: {grid}")

    V = build_potential(grid, cfg.potential)

    # Apply electrostatic disorder (charge puddles) if configured
    dis = cfg.disorder
    if dis.enabled and dis.dv > 0 and cfg.potential.height != 0:
        V = build_disorder(V, grid, dis, V0=cfg.potential.height, verbose=verbose)

    # Apply source-drain bias (rigid potential gradient) if configured.
    # Must come AFTER disorder: build_disorder() uses |V|>0 to localise the
    # puddle mask, and the bias adds a small nonzero value everywhere.
    bias_cfg = cfg.bias
    if bias_cfg.enabled and bias_cfg.V_sd != 0.0:
        V = add_source_drain_bias(V, grid, bias_cfg, verbose=verbose)

    Vmax = float(V.max())

    Ef = p.vf * cfg.wavepacket.k0
    M = None
    if y_bc == "reflecting":
        M = build_mass_confinement(grid, a, Ef=Ef, Vmax=Vmax)
        M_wall = float(M.max())
        if verbose:
            print(f"  Reflecting y-walls: M(y)·σz confinement, "
                  f"M_wall={M_wall:.1f} (gap=2M={2*M_wall:.1f}, Ef={Ef:.1f})")
        Emax_with_M = p.vf * (np.pi / grid.dx) + Vmax + M_wall
        dt_hard_M = np.pi / Emax_with_M
        if cfg.time.dt > dt_hard_M:
            cfg.time.dt = round(dt_hard_M * 0.8, 5)
            if verbose:
                print(f"  [auto-dt] dt reduced to {cfg.time.dt:.5f} "
                      f"(M raises Emax to {Emax_with_M:.1f})")
    elif y_bc == "absorbing":
        if verbose:
            Sy = a.y_strength if a.y_strength > 0 else a.strength
            Wy = a.y_width_frac if a.y_width_frac > 0 else a.width_frac
            print(f"  Absorbing y-walls: cos² drain, "
                  f"width_frac={Wy:.2f}, strength={Sy:.3f}")
    elif y_bc == "periodic" and verbose:
        print("  Periodic y-boundary (ky conserved)")

    dt = cfg.time.dt
    kmax = np.pi / grid.dx
    kymax = np.pi / grid.dy
    Emax = p.vf * kmax + abs(p.wx) * kmax + abs(p.wy) * kymax + Vmax
    if M is not None:
        Emax += float(M.max())
    dt_hard = np.pi / Emax
    if dt > dt_hard:
        print(f"  [WARN] dt={dt:.4f} > hard limit π/Emax={dt_hard:.4f}! "
              "Results will have phase aliasing.")
    elif dt > 1.0 / Emax:
        print(f"  [NOTE] dt={dt:.4f} > 1/Emax={1.0/Emax:.4f}. "
              f"Splitting error ~{100 * (dt * Emax) ** 2 / 12:.0f}%, still unitary.")

    drains_K = DrainContacts(grid, a)
    if verbose:
        Wx = a.width_frac * cfg.grid.Lx
        print(f"  Drain contacts: y_bc={y_bc}  x-width={Wx:.1f}  strength={a.strength}")
    if dual:
        drains_Kp = DrainContacts(grid, a)

    wx, wy = p.wx, p.wy

    # ── Single-valley v1.0 path ─────────────────────────────────────
    if not dual:
        psi_K = init_wavepacket(grid, cfg.wavepacket, chirality=_single_chirality)
        P0 = total_probability(psi_K, grid)
        if verbose:
            print(f"  Initial probability P0 = {P0:.6f}")
        prop_K = SplitOperatorPropagator(
            grid=grid, V=V, vf=p.vf, wx=wx, wy=wy, dt=dt, M=M,
            chirality=_single_chirality,
        )
        valley_prop = None
    else:
        # ── Dual-valley path: independent (v1.0) or coupled (v2.0) ───
        coupling_cfg = cfg.coupling
        valley_prop = build_valley_propagator(
            grid=grid, V=V, vf=p.vf, wx=wx, wy=wy, dt=dt,
            wp_cfg=cfg.wavepacket, coupling_cfg=coupling_cfg,
            V0=cfg.potential.height, M=M,
            source_mode=getattr(p, "source_mode", "both"),
        )
        psi_K = valley_prop.psi_K
        psi_Kp = valley_prop.psi_Kp
        P0 = total_probability(psi_K, grid)
        if verbose:
            print(f"  Initial probability P0 = {P0:.6f}")
            mode = "COUPLED (4-spinor)" if valley_prop.is_coupled else "INDEPENDENT (2 propagators)"
            print(f"  Dual-valley mode: {mode}")
            if valley_prop.is_coupled:
                print(f"  Coupling: type={coupling_cfg.type}  "
                      f"strength={coupling_cfg.strength}")
            print(f"  Propagators: K (w=[{wx},{wy}])  K′ (w=[{-wx},{-wy}])")

        # Expose prop_K for unified logging below — this is just for the
        # fft backend / threads display, no stepping is done through it
        # in dual mode.
        prop_K = valley_prop

    if verbose:
        print(f"  FFT backend: {prop_K.fft_backend}  (threads={prop_K.fft_threads})")

    auto_stop_enabled = dc.auto_stop and dc.enabled
    v_group_x = p.vf * abs(math.cos(cfg.wavepacket.theta_rad))
    if v_group_x > 0:
        x_src = cfg.wavepacket.x_source * (cfg.grid.Lx / 2.0)
        dist_to_barrier = abs(cfg.potential.x_center - x_src)
        min_steps_before_check = int(1.5 * dist_to_barrier / (v_group_x * dt))
    else:
        min_steps_before_check = cfg.time.n_steps

    n_steps = cfg.time.n_steps
    save_every = cfg.time.save_every
    times = []
    probs_K = []
    probs_Kp = [] if dual else None
    frame_idx = 0
    vmax_fixed = None
    auto_stopped = False
    has_y_loss = y_bc == "absorbing"

    wf_dumper = None
    wf_se = None
    if save_wf:
        from .wf_io import WavefunctionDumper

        wf_dir = out_dir / "wavefunction"
        wf_se = wf_save_every if wf_save_every is not None else save_every
        wf_dumper = WavefunctionDumper(wf_dir, grid, sim_config=cfg)
        est_files = n_steps // wf_se
        est_mb = est_files * (67 if dual else 34) / 2
        if verbose:
            print(f"  Wavefunction dump: every {wf_se} steps → "
                  f"~{est_files} files, ~{est_mb / 1000:.1f} GB to {wf_dir}")

    wall_start = _time.perf_counter()

    for step in range(n_steps + 1):
        t = step * dt

        if step % save_every == 0:
            P_K = total_probability(psi_K, grid)
            times.append(t)
            probs_K.append(P_K)

            if dual:
                P_Kp = total_probability(psi_Kp, grid)
                probs_Kp.append(P_Kp)
                if not data_only:
                    vmax_fixed = render_frame_dual(
                        psi_K=psi_K,
                        psi_Kp=psi_Kp,
                        cfg=cfg,
                        step=step,
                        t=t,
                        P_K=P_K,
                        P_Kp=P_Kp,
                        vmax=vmax_fixed,
                        frame_dir=frame_dir,
                        frame_idx=frame_idx,
                        drains_K=drains_K,
                        drains_Kp=drains_Kp,
                    )
            else:
                if not data_only:
                    vmax_fixed = render_frame(
                        psi=psi_K,
                        cfg=cfg,
                        step=step,
                        t=t,
                        prob_total=P_K,
                        vmax=vmax_fixed,
                        frame_dir=frame_dir,
                        frame_idx=frame_idx,
                        drains=drains_K,
                    )
            frame_idx += 1

            if verbose:
                elapsed = _time.perf_counter() - wall_start
                pct = 100.0 * step / n_steps if n_steps > 0 else 100.0
                budget_K = drains_K.total_absorbed + P_K
                if dual:
                    budget_Kp = drains_Kp.total_absorbed + P_Kp
                    print(
                        f"  step {step:5d}/{n_steps}  t={t:6.2f}  "
                        f"T_K={drains_K.T:.3f}  T_K′={drains_Kp.T:.3f}  "
                        f"Σ_K={budget_K:.4f}  Σ_K′={budget_Kp:.4f}  "
                        f"[{pct:.0f}%  {elapsed:.1f}s]",
                        end="\r",
                    )
                else:
                    py_str = f"  Py={drains_K.P_y:.3f}" if has_y_loss else ""
                    print(
                        f"  step {step:5d}/{n_steps}  t={t:6.2f}  "
                        f"P={P_K:.4f}  T={drains_K.T:.4f}  R={drains_K.R:.4f}"
                        f"{py_str}  Σ={budget_K:.6f}  [{pct:.0f}%  {elapsed:.1f}s]",
                        end="\r",
                    )

        if wf_dumper is not None and step % wf_se == 0:
            _P_K = total_probability(psi_K, grid) if step % save_every != 0 else P_K
            wf_meta = {
                "T_K": drains_K.T,
                "R_K": drains_K.R,
                "P_K": _P_K,
                # Embed a compact self-describing metadata set so offline
                # scripts can work directly from wf_*.npz snapshots.
                "V0": float(cfg.potential.height),
                "Ef": float(Ef),
                "vf": float(p.vf),
                "k0": float(cfg.wavepacket.k0),
                "dt": float(dt),
                "wx": float(p.wx),
                "wy": float(p.wy),
                "Nx": int(grid.Nx),
                "Ny": int(grid.Ny),
                "Lx": float(grid.Lx),
                "Ly": float(grid.Ly),
                "dx": float(grid.dx),
                "dy": float(grid.dy),
            }
            if dis.enabled and dis.dv > 0:
                wf_meta.update({
                    "disorder_dv": float(dis.dv),
                    "disorder_lc": float(dis.lc),
                    "disorder_seed": dis.seed,
                })
            if bias_cfg.enabled and bias_cfg.V_sd != 0.0:
                wf_meta.update({
                    "bias_V_sd": float(bias_cfg.V_sd),
                    "bias_type": str(bias_cfg.type),
                    "bias_x_center": float(bias_cfg.x_center),
                    "bias_L_drop": (float(bias_cfg.L_drop)
                                    if bias_cfg.L_drop is not None else None),
                })
            _psi_Kp_save = None
            _psi_full_save = None
            if dual:
                _P_Kp = total_probability(psi_Kp, grid) if step % save_every != 0 else P_Kp
                _psi_Kp_save = psi_Kp
                wf_meta.update({
                    "T_Kp": drains_Kp.T,
                    "R_Kp": drains_Kp.R,
                    "P_Kp": _P_Kp,
                })
                # In coupled mode, also save the full 4-component spinor so
                # post-processing can reconstruct the joint state (intervalley
                # coherence, cross density matrix, valley Bloch vector).
                if valley_prop is not None and valley_prop.is_coupled:
                    _psi_full_save = valley_prop.psi_full
                    wf_meta.update({
                        "coupled": 1,
                        "coupling_strength": float(cfg.coupling.strength),
                    })
            wf_dumper.save(step, t, psi_K, _psi_Kp_save, wf_meta,
                           psi_full=_psi_full_save)

        if (auto_stop_enabled and step > min_steps_before_check
                and step % save_every == 0):
            P_K = total_probability(psi_K, grid)
            P_check = P_K
            if dual:
                P_Kp = total_probability(psi_Kp, grid)
                P_check = max(P_K, P_Kp)
            if P_check < dc.stop_threshold:
                auto_stopped = True
                if verbose:
                    print(f"\n  [auto-stop] P_remaining={P_check:.6f} < "
                          f"{dc.stop_threshold}. Stopping at step {step}.")
                break

        if step < n_steps:
            if dual:
                valley_prop.step()
                # Re-bind in case the adapter rebound its views
                psi_K = valley_prop.psi_K
                psi_Kp = valley_prop.psi_Kp
                valley_prop.apply_drains(drains_K, drains_Kp, t + dt)
            else:
                prop_K.step(psi_K)
                drains_K.apply(psi_K, t + dt)

    wall_total = _time.perf_counter() - wall_start
    n_steps_actual = step

    if verbose:
        print(f"\n  Simulation complete in {wall_total:.1f} s  ({n_steps_actual} steps)")

    if wf_dumper is not None:
        wf_dumper.write_index()
        if verbose:
            n_wf = len(wf_dumper._index)
            print(f"  Wavefunction: {n_wf} snapshots saved to {wf_dumper.out_dir}")

    P_final_K = total_probability(psi_K, grid)
    T_K_val = drains_K.T
    R_K_val = drains_K.R
    Py_K_val = drains_K.P_y
    budget_K = T_K_val + R_K_val + Py_K_val + P_final_K

    if dual:
        P_final_Kp = total_probability(psi_Kp, grid)
        T_Kp_val = drains_Kp.T
        R_Kp_val = drains_Kp.R
        Py_Kp_val = drains_Kp.P_y
        budget_Kp = T_Kp_val + R_Kp_val + Py_Kp_val + P_final_Kp
        denom = T_K_val + T_Kp_val
        eta = (T_K_val - T_Kp_val) / denom if denom > 1e-8 else 0.0
    else:
        eta = None

    if verbose and dc.enabled:
        print(f"\n  {'═' * 58}")
        if dual:
            print("  TRANSPORT RESULTS  (drain contacts, dual valley)")
            print(f"  {'─' * 58}")
            print(f"  {'':20s}{'K':>12s}{'K′':>12s}")
            print(f"  {'─' * 58}")
            print(f"  {'T (right drain)':20s}{T_K_val:12.6f}{T_Kp_val:12.6f}")
            print(f"  {'R (left drain)':20s}{R_K_val:12.6f}{R_Kp_val:12.6f}")
            if has_y_loss:
                print(f"  {'P_y (side walls)':20s}{Py_K_val:12.6f}{Py_Kp_val:12.6f}")
            print(f"  {'T + R':20s}{T_K_val + R_K_val:12.6f}"
                  f"{T_Kp_val + R_Kp_val:12.6f}")
            print(f"  {'P remaining':20s}{P_final_K:12.6f}{P_final_Kp:12.6f}")
            print(f"  {'Σ (budget)':20s}{budget_K:12.6f}{budget_Kp:12.6f}  "
                  f"{'✓' if abs(budget_K - 1) < 1e-4 and abs(budget_Kp - 1) < 1e-4 else '⚠'}")
            print(f"  {'─' * 58}")
            print(f"  Valley polarisation η = (T_K−T_K′)/(T_K+T_K′) = {eta:+.6f}")
        else:
            print("  TRANSPORT RESULTS  (drain contacts)")
            print(f"  {'─' * 58}")
            print(f"  T (right drain)   = {T_K_val:.6f}")
            print(f"  R (left drain)    = {R_K_val:.6f}")
            if has_y_loss:
                print(f"  P_y (side walls)  = {Py_K_val:.6f}")
            print(f"  T + R             = {T_K_val + R_K_val:.6f}")
            print(f"  P remaining       = {P_final_K:.6f}")
            print(f"  Σ (budget)        = {budget_K:.6f}  "
                  f"{'✓' if abs(budget_K - 1) < 1e-4 else '⚠'}")
            if P_final_K < 0.01 and (T_K_val + R_K_val) > 1e-8:
                print(f"  {'─' * 58}")
                print(f"  T / (T+R)         = {T_K_val / (T_K_val + R_K_val):.6f}")
                print(f"  R / (T+R)         = {R_K_val / (T_K_val + R_K_val):.6f}")
        print(f"  {'═' * 58}\n")

    t_final = n_steps_actual * dt
    if not data_only:
        if dual:
            plot_final_snapshot_dual(
                psi_K=psi_K,
                psi_Kp=psi_Kp,
                cfg=cfg,
                t=t_final,
                output_path=out_dir / "final_snapshot.png",
            )
            plot_probability_history(
                times=times,
                probs=probs_K,
                cfg=cfg,
                output_path=out_dir / "probability_history.png",
                probs_Kp=probs_Kp,
            )
            if dc.enabled:
                plot_transmission_report_dual(
                    drains_K=drains_K,
                    drains_Kp=drains_Kp,
                    cfg=cfg,
                    output_path=out_dir / "transmission_report.png",
                )
        else:
            plot_final_snapshot(
                psi=psi_K,
                cfg=cfg,
                t=t_final,
                output_path=out_dir / "final_snapshot.png",
            )
            plot_probability_history(
                times=times,
                probs=probs_K,
                cfg=cfg,
                output_path=out_dir / "probability_history.png",
            )
            if dc.enabled:
                plot_transmission_report(
                    drains=drains_K,
                    cfg=cfg,
                    output_path=out_dir / "transmission_report.png",
                )

        if make_animation:
            make_gif(
                frame_dir=frame_dir,
                output_path=out_dir / "animation.gif",
                fps=fps,
            )

        if verbose:
            print(f"\n  Output written to: {out_dir.resolve()}")

    result = {
        "times": times,
        "probabilities": probs_K,
        "T": T_K_val,
        "R": R_K_val,
        "P_y": Py_K_val,
        "T_history": drains_K.T_history,
        "R_history": drains_K.R_history,
        "P_y_history": drains_K.P_y_history,
        "det_times": drains_K.times,
        "output_dir": out_dir,
        "n_steps_actual": n_steps_actual,
        "auto_stopped": auto_stopped,
        "y_bc": y_bc,
        "fft_backend": prop_K.fft_backend,
        "fft_threads": prop_K.fft_threads,
    }
    if not data_only:
        result["psi_final"] = psi_K

    if dual:
        result.update({
            "T_Kp": T_Kp_val,
            "R_Kp": R_Kp_val,
            "P_y_Kp": Py_Kp_val,
            "T_history_Kp": drains_Kp.T_history,
            "R_history_Kp": drains_Kp.R_history,
            "P_y_history_Kp": drains_Kp.P_y_history,
            "probabilities_Kp": probs_Kp,
            "eta": eta,
            "polarization": eta,
        })
        if not data_only:
            result["psi_final_Kp"] = psi_Kp

    return result
