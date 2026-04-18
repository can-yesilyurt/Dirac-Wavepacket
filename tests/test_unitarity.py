"""
Unitarity test for the single-valley split-operator propagator.

Without absorbers or disorder, the split-operator scheme is unitary by
construction; norm conservation is a minimal correctness check.
"""
from __future__ import annotations

import numpy as np
import pytest

from dirac_wavepacket.config import WavepacketConfig
from dirac_wavepacket.grid import Grid
from dirac_wavepacket.potential import build_potential
from dirac_wavepacket.config import PotentialConfig
from dirac_wavepacket.propagator import SplitOperatorPropagator
from dirac_wavepacket.wavepacket import init_wavepacket, total_probability


@pytest.mark.parametrize("chirality", [+1, -1])
def test_free_propagation_conserves_norm(chirality: int) -> None:
    """Free (V = 0) propagation must conserve the L2 norm of the spinor."""
    grid = Grid(Nx=128, Ny=128, Lx=60.0, Ly=60.0)
    wp_cfg = WavepacketConfig(
        k0=0.8, theta=0.0, sigma_x=4.0, sigma_y=4.0, x_source=-0.5,
    )
    pot_cfg = PotentialConfig(type="free", height=0.0)
    V = build_potential(grid, pot_cfg)

    psi = init_wavepacket(grid, wp_cfg, chirality=chirality)
    P0 = total_probability(psi, grid)
    assert abs(P0 - 1.0) < 1e-4, f"initial norm not unity: {P0}"

    prop = SplitOperatorPropagator(
        grid=grid, V=V, vf=1.0, wx=0.0, wy=0.0, dt=0.02,
        chirality=chirality, fft_backend="numpy",
    )
    for _ in range(200):
        prop.step(psi)

    P_final = total_probability(psi, grid)
    # Single precision + 200 steps ~ a few x 1e-6 drift, be loose.
    assert abs(P_final - 1.0) < 5e-4, (
        f"norm drifted from 1.0 to {P_final} after 200 steps"
    )
