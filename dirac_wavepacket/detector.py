"""
Detector module: flux-based transmission and reflection measurement.

Places virtual "current meters" at vertical lines x = x_T (after barrier)
and x = x_R (before barrier).  Each detector integrates the x-component
of the Dirac probability current  jx = vf·ψ†σx·ψ + wx·|ψ|²  across
the full y-extent at every time step.

Rightward and leftward contributions are tracked separately, giving:
    T(t) = cumulative rightward flux through the transmission detector
    R(t) = cumulative leftward  flux through the reflection  detector

Spatial cross-check (region integration) is also provided:
    T_spatial = ∫|ψ|² dx dy   for x > x_T
    R_spatial = ∫|ψ|² dx dy   for x < x_R
"""

import numpy as np
from .grid import Grid


class FluxDetector:
    """
    Integrates the Dirac probability current across a vertical line.

    Tracks rightward (+x) and leftward (−x) flux separately so that
    transmitted and reflected components can be distinguished.
    """

    def __init__(self, grid: Grid, x_position: float,
                 vf: float, wx: float = 0.0, label: str = ""):
        self.ix = grid.x_index(x_position)
        self.x_actual = float(grid.x[self.ix])
        self.dy = grid.dy
        self.vf = vf
        self.wx = wx
        self.label = label

        # cumulative integrated fluxes
        self.rightward = 0.0
        self.leftward  = 0.0

        # time-series for post-processing / plotting
        self.times          = []
        self.hist_rightward = []
        self.hist_leftward  = []

    # ── per-step measurement ─────────────────────────────────────────────

    def measure(self, psi: np.ndarray, dt: float, t: float) -> float:
        """
        Accumulate flux through this detector for one time step.

        Parameters
        ----------
        psi : complex128 array, shape (2, Ny, Nx)
        dt  : time step size
        t   : current simulation time

        Returns
        -------
        net rightward flux contribution this step
        """
        up = psi[0, :, self.ix]
        dn = psi[1, :, self.ix]

        # jx = vf · 2 Re(ψ↑* ψ↓)  +  wx · |ψ|²
        jx = self.vf * 2.0 * np.real(np.conj(up) * dn)
        if self.wx != 0.0:
            rho = np.abs(up) ** 2 + np.abs(dn) ** 2
            jx += self.wx * rho

        # separate rightward and leftward contributions per y-point
        flux_r = float(np.sum(np.maximum(jx, 0.0))) * self.dy * dt
        flux_l = float(np.sum(np.maximum(-jx, 0.0))) * self.dy * dt

        self.rightward += flux_r
        self.leftward  += flux_l

        self.times.append(t)
        self.hist_rightward.append(self.rightward)
        self.hist_leftward.append(self.leftward)

        return flux_r - flux_l

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def net(self) -> float:
        return self.rightward - self.leftward

    def __repr__(self):
        return (f"FluxDetector('{self.label}', x={self.x_actual:.2f}, "
                f"R→={self.rightward:.4f}, L←={self.leftward:.4f})")


# ── spatial cross-check ──────────────────────────────────────────────────────

def region_probability(psi: np.ndarray, grid: Grid,
                       x_left: float, x_right: float) -> float:
    """
    Integrate |ψ|² over the strip  x_left ≤ x ≤ x_right, full y.

    Parameters
    ----------
    psi : shape (2, Ny, Nx)
    grid : Grid
    x_left, x_right : strip boundaries

    Returns
    -------
    P_region : float
    """
    ix_l = grid.x_index(x_left)
    ix_r = grid.x_index(x_right)
    if ix_l > ix_r:
        ix_l, ix_r = ix_r, ix_l
    rho = np.abs(psi[0, :, ix_l:ix_r + 1]) ** 2 + \
          np.abs(psi[1, :, ix_l:ix_r + 1]) ** 2
    return float(np.sum(rho)) * grid.dx * grid.dy


def barrier_region_probability(psi: np.ndarray, grid: Grid,
                               x_center: float, half_width: float,
                               margin: float = 0.0) -> float:
    """Probability in the barrier ± margin region."""
    return region_probability(psi, grid,
                              x_center - half_width - margin,
                              x_center + half_width + margin)
