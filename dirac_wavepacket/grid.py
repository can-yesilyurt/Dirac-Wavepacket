"""
Grid module: real-space and k-space coordinate arrays for the 2D simulation domain.
"""

import numpy as np
from dataclasses import dataclass

from .precision import COMPLEX_DTYPE, REAL_DTYPE, TWO_PI, I


@dataclass
class Grid:
    """
    Uniform 2D grid with periodic FFT-compatible layout.

    Real-space arrays (X, Y) are centred on the origin.
    k-space arrays (KX, KY) use numpy's fftfreq ordering so they
    are ready to be applied directly after np.fft.fft2.
    """
    Nx: int
    Ny: int
    Lx: float
    Ly: float

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        # ── real-space coordinates ──────────────────────────────────────
        # x ∈ [-Lx/2, Lx/2), y ∈ [-Ly/2, Ly/2)
        self.x = np.linspace(-self.Lx / 2, self.Lx / 2 - self.dx, self.Nx, dtype=REAL_DTYPE)
        self.y = np.linspace(-self.Ly / 2, self.Ly / 2 - self.dy, self.Ny, dtype=REAL_DTYPE)
        # meshgrid convention: X[row, col] → (y_row, x_col), shape (Ny, Nx)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # ── k-space coordinates (FFT ordering) ──────────────────────────
        self.kx = np.asarray(np.fft.fftfreq(self.Nx, d=self.dx) * TWO_PI, dtype=REAL_DTYPE)
        self.ky = np.asarray(np.fft.fftfreq(self.Ny, d=self.dy) * TWO_PI, dtype=REAL_DTYPE)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)   # shape (Ny, Nx)

        # |k| with a small floor to avoid 0/0 in angular expressions
        self.K = np.sqrt(self.KX ** 2 + self.KY ** 2).astype(REAL_DTYPE, copy=False)
        self._K_safe = np.where(self.K == 0.0, REAL_DTYPE(1.0), self.K).astype(REAL_DTYPE, copy=False)

        # Unit-vector components in k-space: ê_k = (kx + i ky) / |k|
        self.ek = ((self.KX + I * self.KY) / self._K_safe).astype(COMPLEX_DTYPE, copy=False)
        self.ek_conj = self.ek.conj().astype(COMPLEX_DTYPE, copy=False)

    # ── convenience ----------------------------------------------------------

    @property
    def shape(self):
        return (self.Ny, self.Nx)

    @property
    def kx_max(self):
        return np.pi / self.dx

    @property
    def ky_max(self):
        return np.pi / self.dy

    def x_index(self, x_val: float) -> int:
        """Return the grid column index closest to x_val."""
        return int(np.argmin(np.abs(self.x - x_val)))

    def __repr__(self):
        return (
            f"Grid(Nx={self.Nx}, Ny={self.Ny}, "
            f"Lx={self.Lx}, Ly={self.Ly}, "
            f"dx={self.dx:.3f}, dy={self.dy:.3f}, "
            f"kx_max={self.kx_max:.2f})"
        )
