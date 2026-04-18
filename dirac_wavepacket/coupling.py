"""
Intervalley coupling models for the 4-component coupled propagator.

Coupling enters the real-space half-step of the split-operator scheme
as off-diagonal 2×2 blocks connecting the K and K' valley subspaces:

    H_r = | H_K     U_KK'  |
          | U_KK'†  H_K'   |

where H_K = H_K' = V(x,y)·I + M(y)·σ_z  (valley-independent diagonal blocks)
and U_KK'(x,y) is the 2×2 intervalley coupling matrix.

Smooth electrostatic potentials cannot couple valleys because they lack
Fourier content at the intervalley separation |K - K'|. Coupling requires
atomically sharp features: defect lines, ultra-short-range disorder,
lattice-scale potential steps, or spin–orbit interactions.

This module provides functions that build coupling arrays on a given grid.
The coupled propagator accepts these arrays and handles the real-space
matrix exponentials internally.

Coupling classifications supported:
  - Scalar:  U = u(x,y)·I₂    pseudospin-preserving intervalley coupling
  - (Future: pseudospin-flip, SOC, exchange — each a 2×2 matrix field)

At zero coupling (u=0), the propagator reproduces two independent valley
simulations exactly.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .grid import Grid
from .precision import COMPLEX_DTYPE, REAL_DTYPE


# ═══════════════════════════════════════════════════════════════════════════════
#  Scalar (pseudospin-preserving) intervalley coupling: U = u(x,y) · I₂
# ═══════════════════════════════════════════════════════════════════════════════

def scalar_coupling(
    grid: Grid,
    strength: float,
    region_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a uniform scalar intervalley coupling U = u · I₂.

    The coupling is constant everywhere (or within a region mask) and
    preserves the pseudospin structure — it mixes valleys without
    flipping the sublattice character.

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    strength : float
        Coupling strength in simulation energy units.
        Positive real values give a real symmetric coupling.
    region_mask : ndarray, shape (Ny, Nx), optional
        Binary or smooth mask restricting the coupling to a spatial region.
        If None, coupling is active everywhere.

    Returns
    -------
    u : complex ndarray, shape (Ny, Nx)
        Coupling amplitude at each grid point.
    """
    u = np.full(grid.shape, COMPLEX_DTYPE(strength), dtype=COMPLEX_DTYPE)
    if region_mask is not None:
        u *= np.asarray(region_mask, dtype=REAL_DTYPE)
    return u


def defect_line_coupling(
    grid: Grid,
    strength: float,
    y_position: float = 0.0,
    width: float = 2.0,
) -> np.ndarray:
    """
    Intervalley coupling localised along a transverse defect line.

    Models a line defect (e.g. a grain boundary or engineered atomic-step
    edge) running perpendicular to the transport direction (x-axis).
    The coupling has a Gaussian profile centred at x = y_position with
    standard deviation = width.

    Parameters
    ----------
    grid : Grid
    strength : float
        Peak coupling amplitude.
    y_position : float
        Centre of the defect line along x (in simulation units).
    width : float
        Gaussian width of the coupling profile (in simulation units).

    Returns
    -------
    u : complex ndarray, shape (Ny, Nx)
    """
    profile = np.exp(-((grid.X - y_position) ** 2) / (2.0 * width ** 2))
    return np.asarray(strength * profile, dtype=COMPLEX_DTYPE)


def barrier_edge_coupling(
    grid: Grid,
    V: np.ndarray,
    strength: float,
    gradient_threshold: float = 0.01,
) -> np.ndarray:
    """
    Intervalley coupling that activates at sharp potential gradients.

    Real-world short-range scattering is strongest where the electrostatic
    landscape varies on the atomic scale. This model activates coupling
    proportional to |∇V|, mimicking the effect of sharp barrier interfaces
    on intervalley mixing.

    Parameters
    ----------
    grid : Grid
    V : ndarray, shape (Ny, Nx)
        Electrostatic potential.
    strength : float
        Proportionality constant.
    gradient_threshold : float
        Minimum |∇V| below which coupling is zeroed (noise floor).

    Returns
    -------
    u : complex ndarray, shape (Ny, Nx)
    """
    grad_x = np.gradient(V, grid.dx, axis=1)
    grad_y = np.gradient(V, grid.dy, axis=0)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag[grad_mag < gradient_threshold] = 0.0
    return np.asarray(strength * grad_mag, dtype=COMPLEX_DTYPE)


def region_mask_barrier(
    grid: Grid,
    V: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Binary mask selecting the barrier interior (where V > threshold).

    Useful as a region_mask for scalar_coupling when intervalley effects
    should be confined to the gated region.

    Parameters
    ----------
    grid : Grid
    V : ndarray, shape (Ny, Nx)
        Electrostatic potential.
    threshold : float
        Minimum potential to count as "inside the barrier".

    Returns
    -------
    mask : float ndarray, shape (Ny, Nx)
        1.0 inside barrier, 0.0 outside.
    """
    return np.where(np.abs(V) > threshold, REAL_DTYPE(1.0), REAL_DTYPE(0.0))
