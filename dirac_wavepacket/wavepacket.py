"""
Wavepacket module: initialise a Gaussian minimum-uncertainty spinor wavepacket
dressed with the correct positive-energy eigenspinor of the 2D Dirac Hamiltonian.

The initial state is:
    ψ(x,y) = N · exp[-(x-x₀)²/(4σx²)] · exp[-(y-y₀)²/(4σy²)]
              · exp[i(kx0·x + ky0·y)] · |χ(k₀)⟩

where the eigenspinor for positive energy at k=(kx0,ky0) depends on chirality χ:
    χ = +1 (K ):  |χ⟩ = (1/√2) · [1, exp(+iφ)]ᵀ,   φ = arctan2(ky0, kx0)
    χ = -1 (K'):  |χ⟩ = (1/√2) · [1, exp(-iφ)]ᵀ

This ensures the wavepacket is initialised in the correct positive-energy
eigenspace of H = vf(σx·kx + χ·σy·ky), where opposite chirality at K'
reverses the pseudospin-momentum locking.

The spinor is stored as an array of shape (2, Ny, Nx) with the package complex dtype.
Index 0 → spin-up, index 1 → spin-down.
"""

import math
import cmath
import numpy as np
from .grid import Grid
from .config import WavepacketConfig
from .precision import COMPLEX_DTYPE, INV_SQRT2, I


def init_wavepacket(grid: Grid, cfg: WavepacketConfig, chirality: int = +1) -> np.ndarray:
    """
    Build the initial Gaussian spinor wavepacket.

    Parameters
    ----------
    grid : Grid
    cfg  : WavepacketConfig
    chirality : int, +1 or -1
        Valley chirality. +1 for K (standard), -1 for K' (opposite).
        Determines the eigenspinor phase: exp(+iφ) for K, exp(-iφ) for K'.

    Returns
    -------
    psi : complex ndarray, shape (2, Ny, Nx)
        Normalised two-component spinor wavefunction.
    """
    if chirality not in (+1, -1):
        raise ValueError(f"chirality must be +1 or -1, got {chirality}")

    kx0 = cfg.kx0
    ky0 = cfg.ky0
    sigma_x = cfg.sigma_x
    sigma_y = cfg.sigma_y

    # Source position
    x0 = cfg.x_source * (grid.Lx / 2.0)
    y0 = 0.0                              # always centred in y

    X, Y = grid.X, grid.Y

    # ── Gaussian envelope ──────────────────────────────────────────────
    gauss = (
        np.exp(-((X - x0) ** 2) / (4.0 * sigma_x ** 2))
        * np.exp(-((Y - y0) ** 2) / (4.0 * sigma_y ** 2))
    )

    # ── Plane-wave carrier ─────────────────────────────────────────────
    carrier = np.exp(I * (kx0 * X + ky0 * Y)).astype(COMPLEX_DTYPE, copy=False)

    # ── Positive-energy eigenspinor of H = vf(σx kx + χ σy ky) ──────────
    #   χ = +1 (K ):  |χ⟩ = (1/√2) [1, exp(+iφ)]ᵀ
    #   χ = -1 (K'):  |χ⟩ = (1/√2) [1, exp(-iφ)]ᵀ
    #   where φ = arctan2(ky0, kx0)
    phi = math.atan2(ky0, kx0)
    spinor_up = INV_SQRT2
    spinor_down = COMPLEX_DTYPE(cmath.exp(1j * chirality * phi) * INV_SQRT2)

    # ── Assemble spinor ────────────────────────────────────────────────
    envelope = gauss * carrier                       # shape (Ny, Nx)
    psi = np.zeros((2, grid.Ny, grid.Nx), dtype=COMPLEX_DTYPE)
    psi[0] = spinor_up   * envelope
    psi[1] = spinor_down * envelope

    # ── Normalise so ∫|ψ|² dx dy = 1 ─────────────────────────────────
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy)
    psi /= norm

    return psi


def total_probability(psi: np.ndarray, grid: Grid) -> float:
    """Compute ∫|ψ|² dx dy (should remain ≈ 1.0 in the absence of absorbers)."""
    return float(np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy)


def probability_density(psi: np.ndarray) -> np.ndarray:
    """Return |ψ|² = |ψ↑|² + |ψ↓|², shape (Ny, Nx)."""
    return np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2


def probability_current_x(psi: np.ndarray, vf: float) -> np.ndarray:
    """
    Compute the x-component of the Dirac probability current:
        jx = vf · ψ† σx ψ = vf · 2 Re(ψ↑* ψ↓)
    """
    return vf * 2.0 * np.real(np.conj(psi[0]) * psi[1])
