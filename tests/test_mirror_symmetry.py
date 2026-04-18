"""
Mirror-symmetry test.

In an isotropic, untilted Dirac cone with a rectangular barrier, the
system is invariant under y → -y, so physics requires

    T(+theta) = T(-theta)   and   R(+theta) = R(-theta).

Any deviation from this is numerical (FFT sampling, drain bookkeeping,
single-precision accumulation) and should stay below ~1e-4.

This is the compressed sibling of the user-facing symmetry check in
examples/02_klein_angular_dependence.py.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dirac_wavepacket.config import (
    SimConfig, GridConfig, PhysicsConfig, TimeConfig, WavepacketConfig,
    PotentialConfig, AbsorberConfig, DetectorConfig, CouplingConfig,
)
from dirac_wavepacket.simulation import run_simulation


def _config_at_angle(theta_deg: float) -> SimConfig:
    cfg = SimConfig()
    cfg.grid = GridConfig(Nx=128, Ny=128, Lx=240.0, Ly=240.0)
    cfg.physics = PhysicsConfig(vf=1.0, tilt=[0.0, 0.0], valleys="K")
    cfg.coupling = CouplingConfig(enabled=False)
    cfg.wavepacket = WavepacketConfig(
        k0=0.8, theta=theta_deg,
        sigma_x=7.0, sigma_y=7.0, x_source=-0.7,
    )
    d = 60.0
    cfg.potential = PotentialConfig(
        type="shaped", height=1.6,
        edges=[-d/2, d/2, -d/2, d/2],
        smoothing_width=0.0,
    )
    cfg.absorber = AbsorberConfig(
        width_frac=0.04, strength=0.08,
        y_bc="reflecting", y_width_frac=0.03, wall_mass=5.0,
    )
    cfg.time = TimeConfig(dt=0.04, n_steps=2500, save_every=5000)
    cfg.detector = DetectorConfig(
        enabled=True, auto_source=False,
        auto_stop=True, stop_threshold=0.005,
    )
    return cfg


@pytest.mark.parametrize("abs_theta", [5.0, 10.0])
def test_mirror_symmetry_untilted(abs_theta: float, tmp_path) -> None:
    """T and R must be invariant under theta -> -theta in an untilted cone."""
    results = {}
    for sign, tag in [(+1, "pos"), (-1, "neg")]:
        cfg = _config_at_angle(sign * abs_theta)
        cfg.output_dir = str(tmp_path / f"{tag}{abs_theta:g}")
        results[sign] = run_simulation(
            cfg, make_animation=False, verbose=False,
            data_only=True, save_wf=False,
        )

    dT = results[+1]["T"] - results[-1]["T"]
    dR = results[+1]["R"] - results[-1]["R"]

    # At this grid / step budget, single-precision residuals are ~1e-5;
    # 1e-4 is a safe gate that still catches any real symmetry breakage.
    assert abs(dT) < 1e-4, (
        f"T asymmetry at theta=±{abs_theta}°: "
        f"T(+)={results[+1]['T']:.6f}, T(-)={results[-1]['T']:.6f}, "
        f"dT={dT:+.2e}"
    )
    assert abs(dR) < 1e-4, (
        f"R asymmetry at theta=±{abs_theta}°: "
        f"R(+)={results[+1]['R']:.6f}, R(-)={results[-1]['R']:.6f}, "
        f"dR={dR:+.2e}"
    )
