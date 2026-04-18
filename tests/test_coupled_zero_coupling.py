"""
Coupled-propagator zero-coupling equivalence test.

At u = 0, the 4-component coupled propagator must reproduce the output
of two independent single-valley propagations to numerical precision.
This is a fundamental self-consistency check: the K-K' block of the
coupled Hamiltonian is diagonal when U_KK'(x, y) = 0, so the block
operator must factor into two independent 2x2 evolutions.

A failure here signals a bug in either the coupled propagator or in
the independent-vs-coupled dispatch logic.
"""
from __future__ import annotations

import numpy as np
import pytest

from dirac_wavepacket.config import (
    SimConfig, GridConfig, PhysicsConfig, TimeConfig, WavepacketConfig,
    PotentialConfig, AbsorberConfig, DetectorConfig, CouplingConfig,
)
from dirac_wavepacket.simulation import run_simulation


def _base_config() -> SimConfig:
    cfg = SimConfig()
    cfg.grid = GridConfig(Nx=96, Ny=96, Lx=180.0, Ly=180.0)
    cfg.physics = PhysicsConfig(vf=1.0, tilt=[0.0, 0.3], valleys="both")
    cfg.wavepacket = WavepacketConfig(
        k0=0.8, theta=0.0, sigma_x=6.0, sigma_y=6.0, x_source=-0.6,
    )
    d = 40.0
    cfg.potential = PotentialConfig(
        type="shaped", height=1.2,
        edges=[-d/2, d/2, -d/2, d/2],
        smoothing_width=0.0,
    )
    cfg.absorber = AbsorberConfig(
        width_frac=0.04, strength=0.08,
        y_bc="reflecting", y_width_frac=0.03, wall_mass=5.0,
    )
    cfg.time = TimeConfig(dt=0.04, n_steps=1500, save_every=5000)
    cfg.detector = DetectorConfig(
        enabled=True, auto_source=False, auto_stop=False,
    )
    return cfg


def test_coupled_zero_u_equals_independent(tmp_path) -> None:
    """4-spinor coupled propagator at u=0 must match independent K + K'."""
    # --- Independent run (default) ---
    cfg_indep = _base_config()
    cfg_indep.coupling = CouplingConfig(enabled=False)
    cfg_indep.output_dir = str(tmp_path / "indep")
    r_indep = run_simulation(
        cfg_indep, make_animation=False, verbose=False,
        data_only=True, save_wf=False,
    )

    # --- Coupled run at u = 0 ---
    cfg_coupled = _base_config()
    cfg_coupled.coupling = CouplingConfig(
        enabled=True, type="barrier", strength=0.0,
    )
    cfg_coupled.output_dir = str(tmp_path / "coupled_u0")
    r_coupled = run_simulation(
        cfg_coupled, make_animation=False, verbose=False,
        data_only=True, save_wf=False,
    )

    # Check each transport observable agrees to single-precision level.
    tol = 5e-5
    for key in ("T", "R", "T_Kp", "R_Kp"):
        vi = r_indep[key]
        vc = r_coupled[key]
        assert abs(vi - vc) < tol, (
            f"{key}: independent={vi:.7f}, coupled@u=0={vc:.7f}, "
            f"diff={vi - vc:+.2e} exceeds tol {tol:.0e}"
        )

    # The valley polarisation must also match.
    eta_i, eta_c = r_indep["eta"], r_coupled["eta"]
    assert abs(eta_i - eta_c) < tol, (
        f"eta: independent={eta_i:+.6f}, coupled@u=0={eta_c:+.6f}"
    )
