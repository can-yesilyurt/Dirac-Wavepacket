"""
Auto module: automatic domain sizing, source positioning, and grid validation.

Given the wavepacket parameters (k0, theta, sigma), the barrier geometry, and
the absorber fraction, this module computes the minimal simulation domain and
grid resolution that guarantee:

  1. No aliasing:  dx < π / (safety_factor · k0)
  2. Source far enough from barrier:  x_src = barrier_left − n_sigma · sigma_x
  3. Source and detectors inside the physical (non-absorber) region
  4. Ly wide enough for the oblique trajectory + beam width
  5. Nx, Ny rounded up to FFT-friendly sizes (multiples of 32, or powers of 2)

All sizing functions return plain numbers / dicts — they never mutate the config.
"""

import math
import numpy as np
from .config import SimConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _next_fft_friendly(n: int, base: int = 64) -> int:
    """Round up to the next multiple of `base` (good for FFT performance)."""
    return int(math.ceil(n / base)) * base


def _nyquist_dx(k0: float, safety: float = 3.0) -> float:
    """Maximum dx satisfying Nyquist with a safety margin (points per λ)."""
    # We want at least `safety` grid points per shortest half-wavelength
    # dx < π / (k0)  is Nyquist;  dx < π / (safety * k0) for safety
    return math.pi / (safety * k0)


def _barrier_x_bounds(cfg: SimConfig) -> tuple[float, float]:
    """
    Return the bounding-box x-extent (x_left, x_right) of the barrier,
    regardless of barrier type.  Works for barrier, shaped, etc.
    """
    v = cfg.potential
    if v.type == "shaped" and v.edges is not None:
        xL_b, xR_b, xL_t, xR_t = v.edges
        return min(xL_b, xL_t), max(xR_b, xR_t)
    # All other types use x_center ± width/2
    return v.x_center - v.width / 2.0, v.x_center + v.width / 2.0


# ── auto source position ────────────────────────────────────────────────────

def auto_source_x(cfg: SimConfig, n_sigma: float = 3.5) -> float:
    """
    Compute optimal source x-position in physical units.

    Places the wavepacket centre at  barrier_left_edge − n_sigma · sigma_x,
    ensuring the Gaussian tail (3σ) doesn't overlap the barrier and the
    packet body stays inside the absorber-free region.

    Returns
    -------
    x_source : float (physical x-coordinate, NOT a fraction)
    """
    sigma_x = cfg.wavepacket.sigma_x
    barrier_left, _ = _barrier_x_bounds(cfg)

    x_src = barrier_left - n_sigma * sigma_x

    # Check that 3σ tail doesn't hit left absorber boundary
    absorber_w = cfg.absorber.width_frac * cfg.grid.Lx
    x_min_safe = -cfg.grid.Lx / 2.0 + absorber_w + 2.5 * sigma_x
    if x_src < x_min_safe:
        x_src = x_min_safe

    return x_src


def auto_source_fraction(cfg: SimConfig, n_sigma: float = 3.5) -> float:
    """Return auto source position as fraction of Lx/2 (for config compatibility)."""
    x_src = auto_source_x(cfg, n_sigma)
    return x_src / (cfg.grid.Lx / 2.0)


# ── auto detector positions ─────────────────────────────────────────────────

def auto_detector_positions(cfg: SimConfig, n_sigma: float = 2.0):
    """
    Compute transmission and reflection detector x-positions.

    Returns
    -------
    x_trans, x_refl : float, float
        Transmission detector placed n_sigma·sigma_x after barrier right edge.
        Reflection detector placed midway between source and barrier left edge.
    """
    sigma_x = cfg.wavepacket.sigma_x
    barrier_left, barrier_right = _barrier_x_bounds(cfg)

    x_trans = barrier_right + n_sigma * sigma_x

    # Reflection detector: between source and barrier
    x_src = auto_source_x(cfg)
    x_refl = (x_src + barrier_left) / 2.0

    # Clamp to valid domain
    absorber_w = cfg.absorber.width_frac * cfg.grid.Lx
    Lx2 = cfg.grid.Lx / 2.0
    x_trans = min(x_trans, Lx2 - absorber_w - sigma_x)
    x_refl  = max(x_refl,  -Lx2 + absorber_w + sigma_x)

    return x_trans, x_refl


# ── auto domain sizing ──────────────────────────────────────────────────────

def auto_domain(cfg: SimConfig, n_sigma_source: float = 3.5,
                n_sigma_detector: float = 2.0,
                pts_per_wavelength: float = 6.0,
                absorber_frac: float = None) -> dict:
    """
    Compute minimal Lx, Ly, Nx, Ny for the given physics.

    Parameters
    ----------
    cfg : SimConfig (uses wavepacket, potential, absorber, physics sections)
    n_sigma_source : source-to-barrier distance in sigma_x units
    n_sigma_detector : detector offset from barrier in sigma_x units
    pts_per_wavelength : minimum grid points per de Broglie wavelength
    absorber_frac : override absorber width fraction (default: from config)

    Returns
    -------
    dict with keys: Lx, Ly, Nx, Ny, dx, dy, dt_max, x_source, x_trans, x_refl
    """
    w = cfg.wavepacket
    v = cfg.potential
    p = cfg.physics
    a_frac = absorber_frac if absorber_frac is not None else cfg.absorber.width_frac

    sigma_x = w.sigma_x
    sigma_y = w.sigma_y
    k0 = w.k0
    theta_rad = w.theta_rad

    barrier_left, barrier_right = _barrier_x_bounds(cfg)
    barrier_hw = (barrier_right - barrier_left) / 2.0  # effective half-width

    # ── Lx: physical extent needed ─────────────────────────────────────
    # Left of barrier centre: barrier_hw + n_sigma_source*σx + 3σx (tail)
    left_phys = barrier_hw + (n_sigma_source + 3.0) * sigma_x
    # Right of barrier centre: barrier_hw + n_sigma_detector*σx + 3σx (tail) + extra for transmitted wave
    right_phys = barrier_hw + (n_sigma_detector + 3.0) * sigma_x
    # Symmetrise for simplicity (domain centred on barrier centre)
    half_phys = max(left_phys, right_phys)
    # Include absorber: Lx_physical = Lx * (1 - 2*a_frac)
    # So Lx = 2 * half_phys / (1 - 2*a_frac)
    Lx = 2.0 * half_phys / max(1.0 - 2.0 * a_frac, 0.3)

    # ── Ly: depends on boundary condition ────────────────────────────
    if cfg.absorber.x_only:
        # Periodic y (ky conserved): beam wraps freely, Ly only needs
        # to contain the Gaussian without aliasing onto itself.
        # 7σ contains 99.97% of a Gaussian.
        half_y_phys = 3.5 * sigma_y
        Ly = 2.0 * half_y_phys
    else:
        # y-absorbers present: must physically contain the trajectory
        y_travel = half_phys * abs(math.tan(theta_rad)) if abs(theta_rad) < 1.4 else half_phys * 5.0
        half_y_phys = y_travel + 3.5 * sigma_y
        y_absorber_frac = a_frac
        Ly = 2.0 * half_y_phys / max(1.0 - 2.0 * y_absorber_frac, 0.3)

    # Minimum Ly for beam width
    Ly = max(Ly, 2.0 * 3.5 * sigma_y)

    # ── Grid resolution from Nyquist ───────────────────────────────────
    dx_max = math.pi / (0.5 * pts_per_wavelength * k0)  # π / (k0 * pts/2)
    dy_max = dx_max  # same resolution in both directions

    Nx = _next_fft_friendly(int(math.ceil(Lx / dx_max)))
    Ny = _next_fft_friendly(int(math.ceil(Ly / dy_max)))

    # Recompute actual dx, dy
    dx = Lx / Nx
    dy = Ly / Ny

    # ── Accuracy bound (split-operator is unconditionally unitary) ────
    kmax = math.pi / dx
    Vmax = v.height if v.type != "free" else 0.0
    Emax = p.vf * kmax + Vmax
    dt_max = 1.0 / Emax   # practical: < 1% Strang splitting error

    # ── Source and detector positions ──────────────────────────────────
    x_source = barrier_left - n_sigma_source * sigma_x
    x_trans  = barrier_right + n_sigma_detector * sigma_x
    x_refl   = (x_source + barrier_left) / 2.0

    return {
        "Lx": round(Lx, 2),
        "Ly": round(Ly, 2),
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "dt_max": dt_max,
        "x_source": x_source,
        "x_source_frac": x_source / (Lx / 2.0),
        "x_trans": x_trans,
        "x_refl": x_refl,
    }


# ── grid validation ──────────────────────────────────────────────────────────

def validate_grid(cfg: SimConfig) -> list[str]:
    """
    Check grid resolution, stability, and domain sizing.

    Returns
    -------
    warnings : list of warning strings (empty if all OK)
    """
    warnings = []
    g = cfg.grid
    w = cfg.wavepacket
    p = cfg.physics
    v = cfg.potential

    dx = g.Lx / g.Nx
    dy = g.Ly / g.Ny

    # Nyquist check
    kx_max = math.pi / dx
    ky_max = math.pi / dy
    kx0 = abs(w.kx0)
    ky0 = abs(w.ky0)

    if kx0 > kx_max:
        warnings.append(
            f"CRITICAL: kx0={kx0:.3f} > kx_max={kx_max:.3f} — carrier is ALIASED! "
            f"Need Nx ≥ {int(math.ceil(g.Lx * w.k0 / math.pi))} or reduce Lx."
        )
    elif kx0 > 0.7 * kx_max:
        warnings.append(
            f"WARNING: kx0={kx0:.3f} is > 70% of kx_max={kx_max:.3f} — marginal resolution."
        )

    if ky0 > ky_max:
        warnings.append(
            f"CRITICAL: ky0={ky0:.3f} > ky_max={ky_max:.3f} — carrier is ALIASED in y!"
        )

    # Points per wavelength
    lam = 2 * math.pi / w.k0
    ppw_x = lam / dx
    ppw_y = lam / dy
    if ppw_x < 4:
        warnings.append(f"WARNING: only {ppw_x:.1f} points/wavelength in x (need ≥ 4).")
    if ppw_y < 4:
        warnings.append(f"WARNING: only {ppw_y:.1f} points/wavelength in y (need ≥ 4).")

    # Accuracy check (split-operator is unconditionally unitary)
    Vmax = v.height if v.type != "free" else 0.0
    kmax = math.pi / min(dx, dy)
    Emax = p.vf * kmax + Vmax
    dt_hard = math.pi / Emax
    if cfg.time.dt > dt_hard:
        warnings.append(
            f"CRITICAL: dt={cfg.time.dt:.4f} > π/Emax={dt_hard:.4f} — temporal aliasing!"
        )
    elif cfg.time.dt > 1.0 / Emax:
        warnings.append(
            f"NOTE: dt={cfg.time.dt:.4f} > 1/Emax={1/Emax:.4f}. "
            f"Splitting error grows but still unitary."
        )

    # Source inside absorber?
    x_src = w.x_source * (g.Lx / 2.0)
    absorber_w = cfg.absorber.width_frac * g.Lx
    if x_src < -g.Lx / 2.0 + absorber_w + 2 * w.sigma_x:
        warnings.append(
            f"WARNING: source at x={x_src:.1f} overlaps the absorbing boundary layer."
        )

    return warnings


def print_auto_summary(cfg: SimConfig):
    """Print auto-sizing recommendations for the given config."""
    rec = auto_domain(cfg)
    current_dx = cfg.grid.Lx / cfg.grid.Nx
    current_dy = cfg.grid.Ly / cfg.grid.Ny

    print(f"\n  {'═' * 58}")
    print(f"  AUTO-SIZING RECOMMENDATION")
    print(f"  {'─' * 58}")
    print(f"  Current grid : {cfg.grid.Nx}×{cfg.grid.Ny}  "
          f"({cfg.grid.Lx}×{cfg.grid.Ly})  dx={current_dx:.3f}")
    print(f"  Recommended  : {rec['Nx']}×{rec['Ny']}  "
          f"({rec['Lx']}×{rec['Ly']})  dx={rec['dx']:.3f}")
    print(f"  dt_max       : {rec['dt_max']:.5f}")
    print(f"  Source x     : {rec['x_source']:.2f}  "
          f"(frac = {rec['x_source_frac']:.3f})")
    print(f"  T-detector   : {rec['x_trans']:.2f}")
    print(f"  R-detector   : {rec['x_refl']:.2f}")
    print(f"  {'═' * 58}\n")

    warnings = validate_grid(cfg)
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
        print()
