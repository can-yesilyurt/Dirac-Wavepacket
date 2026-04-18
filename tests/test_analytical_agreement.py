"""
Analytical-agreement test at the Klein paradox point.

At normal incidence (theta = 0) through a rectangular barrier in an
isotropic untilted Dirac cone, the Klein paradox enforces T = 1 exactly
regardless of V_0 and barrier width. The numerical split-operator
result must reproduce this within a small tolerance — any systematic
deviation signals a bug in the propagator, the drain bookkeeping, or
the wavepacket normalisation.

We also cross-check against the analytical `transmission_coefficient`
function (Katsnelson formula) directly to make sure the analytical
module itself returns ~1 at phi=0.
"""
from __future__ import annotations

import numpy as np

from dirac_wavepacket.analytical import transmission_coefficient
from dirac_wavepacket.config import (
    SimConfig, GridConfig, PhysicsConfig, TimeConfig, WavepacketConfig,
    PotentialConfig, AbsorberConfig, DetectorConfig, CouplingConfig,
)
from dirac_wavepacket.simulation import run_simulation


def test_analytical_klein_peak_is_unity() -> None:
    """Katsnelson formula must return T = 1 at normal incidence."""
    T_analytical = transmission_coefficient(
        phi_deg=0.0, E=0.8, V0=1.6, D=50.0, hvF=1.0,
    )
    # float so pytest prints nicely; expect < 1e-10 deviation
    assert abs(float(T_analytical) - 1.0) < 1e-10, (
        f"Katsnelson formula at phi=0 returned {float(T_analytical)}; "
        f"Klein paradox requires exactly 1"
    )


def test_numerical_klein_peak_close_to_unity(tmp_path) -> None:
    """Split-operator numerical T at theta=0 must be close to the
    Klein-paradox limit T=1 for a rectangular barrier in the bipolar
    regime (V_0 > E_F)."""
    cfg = SimConfig()
    # Compact domain so the packet reaches the drain in reasonable steps.
    cfg.grid = GridConfig(Nx=192, Ny=96, Lx=180.0, Ly=90.0)
    cfg.physics = PhysicsConfig(vf=1.0, tilt=[0.0, 0.0], valleys="K")
    cfg.coupling = CouplingConfig(enabled=False)
    # Wide packet in real space -> narrow in k-space -> close to plane
    # wave at theta=0. This is important: the Klein paradox is a
    # plane-wave statement.
    cfg.wavepacket = WavepacketConfig(
        k0=0.8, theta=0.0, sigma_x=8.0, sigma_y=8.0, x_source=-0.65,
    )
    d = 30.0
    cfg.potential = PotentialConfig(
        type="shaped", height=1.6,
        edges=[-d/2, d/2, -d/2, d/2],
        smoothing_width=0.0,
    )
    cfg.absorber = AbsorberConfig(
        width_frac=0.05, strength=0.10,
        y_bc="reflecting", y_width_frac=0.03, wall_mass=5.0,
    )
    cfg.time = TimeConfig(dt=0.04, n_steps=8000, save_every=20000)
    cfg.detector = DetectorConfig(
        enabled=True, auto_source=False,
        auto_stop=True, stop_threshold=0.002,
    )
    cfg.output_dir = str(tmp_path / "klein")

    result = run_simulation(
        cfg, make_animation=False, verbose=False,
        data_only=True, save_wf=False,
    )
    T = result["T"]
    R = result["R"]

    # A finite-width packet samples a small angular spread around 0,
    # so T will be slightly below 1. 0.97 is a safe floor for the
    # chosen sigma / grid, and rules out any gross propagator defect.
    assert T > 0.97, (
        f"numerical T({'theta'}=0) = {T:.6f}; Klein peak requires "
        f"T ~ 1. R = {R:.6f}. Possible propagator regression."
    )
    # Reflection must be correspondingly small.
    assert R < 0.03, (
        f"numerical R(theta=0) = {R:.6f}; Klein paradox requires near "
        f"zero reflection."
    )
