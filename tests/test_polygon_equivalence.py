"""
Polygon vs shaped-rectangle equivalence test.

The potential builder supports multiple geometry types. A rectangle
specified via the `polygon` type with the four corner vertices must
produce the same transport observables as the same rectangle
specified via the `shaped` type. This guards against regressions in
either the polygon mask generation or the shaped parallelogram logic.
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
    cfg.grid = GridConfig(Nx=128, Ny=128, Lx=240.0, Ly=240.0)
    cfg.physics = PhysicsConfig(vf=1.0, tilt=[0.0, 0.0], valleys="K")
    cfg.coupling = CouplingConfig(enabled=False)
    cfg.wavepacket = WavepacketConfig(
        k0=0.8, theta=0.0, sigma_x=7.0, sigma_y=7.0, x_source=-0.7,
    )
    cfg.absorber = AbsorberConfig(
        width_frac=0.04, strength=0.08,
        y_bc="reflecting", y_width_frac=0.03, wall_mass=5.0,
    )
    cfg.time = TimeConfig(dt=0.04, n_steps=2000, save_every=5000)
    cfg.detector = DetectorConfig(
        enabled=True, auto_source=False, auto_stop=False,
    )
    return cfg


def test_polygon_vs_shaped_rectangle(tmp_path) -> None:
    """Same rectangle via polygon and shaped types must give equal T, R."""
    d = 40.0
    Ly = 240.0
    height = 1.2

    # --- Shaped rectangle (alpha = 0) ---
    cfg_s = _base_config()
    cfg_s.potential = PotentialConfig(
        type="shaped", height=height,
        edges=[-d/2, d/2, -d/2, d/2],
        smoothing_width=0.0,
    )
    cfg_s.output_dir = str(tmp_path / "shaped")
    r_s = run_simulation(
        cfg_s, make_animation=False, verbose=False,
        data_only=True, save_wf=False,
    )

    # --- Polygon rectangle with matching corners ---
    # Vertices in CCW order, spanning the full Ly.
    cfg_p = _base_config()
    cfg_p.potential = PotentialConfig(
        type="polygon", height=height,
        vertices=[
            (-d/2, -Ly/2), (+d/2, -Ly/2),
            (+d/2, +Ly/2), (-d/2, +Ly/2),
        ],
        smoothing_width=0.0,
    )
    cfg_p.output_dir = str(tmp_path / "polygon")
    r_p = run_simulation(
        cfg_p, make_animation=False, verbose=False,
        data_only=True, save_wf=False,
    )

    # The two mask-generation paths differ; a small sub-grid-cell mismatch
    # near the rectangle's boundary is expected. 5e-3 allows for that but
    # still catches any genuine geometry bug.
    tol = 5e-3
    for key in ("T", "R"):
        assert abs(r_s[key] - r_p[key]) < tol, (
            f"{key}: shaped={r_s[key]:.6f}, polygon={r_p[key]:.6f}, "
            f"diff={r_s[key]-r_p[key]:+.2e} exceeds tol {tol:.0e}"
        )
