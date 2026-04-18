"""
Drain module: absorbing boundaries as Ohmic drain contacts.

The key insight: the absorber IS the detector. Everything absorbed by the
right drain was transmitted; everything absorbed by the left drain was
reflected. This gives T + R + P_y + P_remaining = 1 exactly by construction,
with no flux-detector placement issues and no probability leaks.

Implementation:
  - Two separate cos²-profile masks: mask_L (left drain) and mask_R (right drain)
  - Each step: apply mask_L, measure ΔP_L; apply mask_R, measure ΔP_R
  - Cumulative ΔP_R = T,  cumulative ΔP_L = R

Transverse boundary conditions (y_bc):
  - "periodic"   — no y-treatment, FFT periodic wrap (correct for V=V(x))
  - "absorbing"  — cos² damping at y-edges, tracked as P_y (side-wall loss)
                    Budget: T + R + P_y + P_remaining = 1
  - "reflecting" — handled upstream via mass confinement M(y)·σz in propagator
                    (no y-absorbers needed here)

The cos² profile ensures C¹-smooth damping that minimises spurious
reflections at the inner edge of the drain region.
"""

import numpy as np
from .grid import Grid
from .config import AbsorberConfig
from .precision import REAL_DTYPE


def _cos2_mask(distance_array: np.ndarray, width: float,
               strength: float) -> np.ndarray:
    """
    Build a cos²-profile damping mask.

    Returns 1.0 in the bulk, smoothly decreasing to (1 - strength) at the
    boundary over a transition region of `width`.
    """
    ratio = np.asarray(np.clip(distance_array / width, 0.0, 1.0), dtype=REAL_DTYPE)
    mask = 1.0 - strength * np.cos(0.5 * np.pi * ratio) ** 2
    return np.asarray(mask, dtype=REAL_DTYPE)


class DrainContacts:
    """
    Left and right absorbing drain contacts that track absorbed probability.
    Optionally includes y-edge absorbers for non-periodic transverse BCs.

    Usage:
        drains = DrainContacts(grid, cfg)
        # in the time loop:
        drains.apply(psi, t)       # absorbs + counts
        T = drains.T               # cumulative transmission
        R = drains.R               # cumulative reflection
        P_y = drains.P_y           # cumulative y-edge loss (absorbing y_bc only)
    """

    def __init__(self, grid: Grid, cfg: AbsorberConfig):
        S = cfg.strength
        Wx = cfg.width_frac * grid.Lx
        X = grid.X

        dx_left  = X + grid.Lx / 2.0       # distance from left boundary
        dx_right = grid.Lx / 2.0 - X       # distance from right boundary

        # Separate x-masks: each is 1 in bulk, < 1 only near its own edge
        self.mask_L = _cos2_mask(dx_left, Wx, S)
        self.mask_R = _cos2_mask(dx_right, Wx, S)

        # Pre-compute drain fraction: |ψ_new|² = mask² |ψ_old|²
        self.drain_L = 1.0 - self.mask_L ** 2
        self.drain_R = 1.0 - self.mask_R ** 2

        self.dxdy = grid.dx * grid.dy
        self.dx = grid.dx
        self.dy = grid.dy

        # Cumulative absorbed probability
        self._T = 0.0       # right drain = transmission
        self._R = 0.0       # left drain  = reflection

        # y-resolved absorption: per-row accumulation for angle-resolved T(θ)
        self.Ny = grid.Ny
        self.y = grid.y.copy()
        self.T_of_y = np.zeros(grid.Ny, dtype=np.float64)
        self.R_of_y = np.zeros(grid.Ny, dtype=np.float64)

        # Time series
        self.T_history = []
        self.R_history = []
        self.times = []

        # ── y-edge absorbers (only for y_bc="absorbing") ─────────────
        self._has_y_drains = (cfg.y_bc == "absorbing")
        self._P_y = 0.0
        self.P_y_history = []

        if self._has_y_drains:
            Y = grid.Y
            Sy = cfg.y_strength if cfg.y_strength > 0 else S
            Wy = (cfg.y_width_frac if cfg.y_width_frac > 0
                  else cfg.width_frac) * grid.Ly

            dy_bot = Y + grid.Ly / 2.0     # distance from bottom
            dy_top = grid.Ly / 2.0 - Y     # distance from top

            self.mask_B = _cos2_mask(dy_bot, Wy, Sy)
            self.mask_T = _cos2_mask(dy_top, Wy, Sy)
            self.drain_B = 1.0 - self.mask_B ** 2
            self.drain_T = 1.0 - self.mask_T ** 2

    def apply(self, psi: np.ndarray, t: float = 0.0) -> None:
        """
        Apply all drain masks to the spinor and accumulate T, R, P_y.

        Parameters
        ----------
        psi : complex128 array, shape (2, Ny, Nx) — modified in-place
        t   : current simulation time (for history tracking)
        """
        # Current probability density
        rho = np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2

        # ── Right drain (transmission) ────────────────────────────────
        absorbed_R = rho * self.drain_R
        dP_R_per_y = np.sum(absorbed_R, axis=1) * self.dx
        dP_R = float(np.sum(dP_R_per_y)) * self.dy
        self.T_of_y += dP_R_per_y * self.dy
        psi[0] *= self.mask_R
        psi[1] *= self.mask_R

        # After right drain, recompute rho for left drain
        rho = np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2

        # ── Left drain (reflection) ───────────────────────────────────
        absorbed_L = rho * self.drain_L
        dP_L_per_y = np.sum(absorbed_L, axis=1) * self.dx
        dP_L = float(np.sum(dP_L_per_y)) * self.dy
        self.R_of_y += dP_L_per_y * self.dy
        psi[0] *= self.mask_L
        psi[1] *= self.mask_L

        # ── y-edge drains (side-wall loss) ────────────────────────────
        dP_y = 0.0
        if self._has_y_drains:
            rho = np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2

            # Bottom y-edge
            dP_B = float(np.sum(rho * self.drain_B)) * self.dx * self.dy
            psi[0] *= self.mask_B
            psi[1] *= self.mask_B

            rho = np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2

            # Top y-edge
            dP_T = float(np.sum(rho * self.drain_T)) * self.dx * self.dy
            psi[0] *= self.mask_T
            psi[1] *= self.mask_T

            dP_y = dP_B + dP_T
            self._P_y += dP_y

        # Accumulate
        self._T += dP_R
        self._R += dP_L

        self.T_history.append(self._T)
        self.R_history.append(self._R)
        self.P_y_history.append(self._P_y)
        self.times.append(t)

    @property
    def T(self) -> float:
        """Cumulative transmission (right drain absorption)."""
        return self._T

    @property
    def R(self) -> float:
        """Cumulative reflection (left drain absorption)."""
        return self._R

    @property
    def P_y(self) -> float:
        """Cumulative y-edge (side-wall) absorption."""
        return self._P_y

    @property
    def T_plus_R(self) -> float:
        return self._T + self._R

    @property
    def total_absorbed(self) -> float:
        """Total probability removed by all drains."""
        return self._T + self._R + self._P_y

    def __repr__(self):
        s = (f"DrainContacts(T={self._T:.6f}, R={self._R:.6f}")
        if self._has_y_drains:
            s += f", P_y={self._P_y:.6f}"
        s += f", absorbed={self.total_absorbed:.6f})"
        return s


# ── mass confinement for reflecting y-walls ──────────────────────────────────

def build_mass_confinement(grid: Grid, cfg: AbsorberConfig,
                           Ef: float = 1.0, Vmax: float = 0.0) -> np.ndarray:
    """
    Build the mass confinement profile M(y) for reflecting transverse walls.

    M(y)·σz opens a gap 2M at the y-edges, confining Dirac fermions without
    Klein tunneling (a scalar potential wall would be transparent).

    The profile ramps smoothly from 0 in the bulk to M_wall at the boundary
    using the same cos² shape as the absorbing drains.

    Parameters
    ----------
    grid : Grid
    cfg  : AbsorberConfig
    Ef   : Fermi energy (for auto-setting wall_mass)
    Vmax : barrier height (for auto-setting wall_mass)

    Returns
    -------
    M : ndarray, shape (Ny, Nx), dtype float64
        Mass confinement term. Zero in bulk, large near y-edges.
    """
    Wy = (cfg.y_width_frac if cfg.y_width_frac > 0
          else cfg.width_frac) * grid.Ly

    # Wall mass: large enough that gap 2M >> max(Ef, V0)
    if cfg.wall_mass > 0:
        M_wall = cfg.wall_mass
    else:
        M_wall = 20.0 * max(Ef, Vmax, 1.0)

    Y = grid.Y
    dy_bot = Y + grid.Ly / 2.0     # distance from bottom wall
    dy_top = grid.Ly / 2.0 - Y     # distance from top wall

    def _mass_ramp(d_arr, W):
        """M = M_wall * cos²(π/2 · d/W) inside the wall region, 0 outside."""
        ratio = np.clip(d_arr / W, 0.0, 1.0)
        return M_wall * np.cos(0.5 * np.pi * ratio) ** 2

    M = np.maximum(_mass_ramp(dy_bot, Wy), _mass_ramp(dy_top, Wy))
    return np.asarray(M, dtype=REAL_DTYPE)


# ── backward-compatible wrappers ─────────────────────────────────────────────

def build_absorber_mask(grid: Grid, cfg: AbsorberConfig) -> np.ndarray:
    """
    Build a combined absorber mask (for use without drain tracking).
    Left and right x-drains only when x_only=True.
    """
    S = cfg.strength
    Wx = cfg.width_frac * grid.Lx
    X, Y = grid.X, grid.Y

    dx_left  = X + grid.Lx / 2.0
    dx_right = grid.Lx / 2.0 - X

    def _damp(d_arr, W):
        ratio = np.clip(d_arr / W, 0.0, 1.0)
        return 1.0 - S * np.cos(0.5 * np.pi * ratio) ** 2

    mask = np.ones(grid.shape, dtype=REAL_DTYPE)
    mask = np.minimum(mask, _damp(dx_left, Wx))
    mask = np.minimum(mask, _damp(dx_right, Wx))

    if not cfg.x_only:
        Wy = cfg.width_frac * grid.Ly
        dy_bot = Y + grid.Ly / 2.0
        dy_top = grid.Ly / 2.0 - Y
        mask = np.minimum(mask, _damp(dy_bot, Wy))
        mask = np.minimum(mask, _damp(dy_top, Wy))

    return mask


def apply_absorber(psi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply both spinor components by the absorbing mask in-place."""
    psi[0] *= mask
    psi[1] *= mask
    return psi
