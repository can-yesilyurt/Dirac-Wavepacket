"""
Potential module: builds the electrostatic potential V(x,y) array on the grid.

Supported profiles:
  free           – no barrier (V = 0 everywhere)
  barrier        – rectangular npn barrier: V = V0 in |x - x_c| < D/2
  pn_junction    – sharp p-n interface: V = V0 for x > x_c
  double_barrier – two rectangular barriers separated by a gap
  shaped         – arbitrary quadrilateral barrier defined by edges at
                   bottom and top walls: edges = [xL_bot, xR_bot, xL_top, xR_top]
                   Covers trapezoid, triangle, rotated rectangle, etc.
"""

import numpy as np
from typing import Optional
from .grid import Grid
from .config import PotentialConfig, BarrierConfig, DisorderConfig, BiasConfig
from .precision import REAL_DTYPE


def build_potential(grid: Grid, cfg: PotentialConfig) -> np.ndarray:
    """
    Return the potential V(x,y) as a real array of shape (Ny, Nx).

    Parameters
    ----------
    grid : Grid
    cfg  : PotentialConfig

    Returns
    -------
    V : ndarray, shape (Ny, Nx), dtype float32/float64 (precision policy)
    """
    X, Y = grid.X, grid.Y
    ptype = cfg.type.lower()

    if ptype == "free":
        return np.zeros(grid.shape, dtype=REAL_DTYPE)

    elif ptype == "barrier":
        # Rectangular npn barrier: V = V0 for |x - x_c| < D/2
        V = np.where(
            np.abs(X - cfg.x_center) < cfg.width / 2.0,
            cfg.height,
            0.0,
        )
        return np.asarray(V, dtype=REAL_DTYPE)

    elif ptype == "pn_junction":
        # Step potential: V = 0 for x < x_c, V = V0 for x >= x_c
        V = np.where(X >= cfg.x_center, cfg.height, 0.0)
        return np.asarray(V, dtype=REAL_DTYPE)

    elif ptype == "double_barrier":
        # Two barriers separated by a gap equal to the barrier width
        gap = cfg.width          # gap between the two barriers
        half_gap = gap / 2.0
        D = cfg.width            # each barrier width = same as 'width'
        left_c  = cfg.x_center - half_gap - D / 2.0
        right_c = cfg.x_center + half_gap + D / 2.0
        V = np.where(
            (np.abs(X - left_c)  < D / 2.0) | (np.abs(X - right_c) < D / 2.0),
            cfg.height,
            0.0,
        )
        return np.asarray(V, dtype=REAL_DTYPE)

    elif ptype == "shaped":
        # Quadrilateral barrier: left/right edges interpolated linearly in y.
        #   edges = [xL_bot, xR_bot, xL_top, xR_top]
        #
        #   (xL_top)────────(xR_top)       y = +Ly/2  (top wall)
        #       \              /
        #        \   V = V0   /
        #         \          /
        #   (xL_bot)────(xR_bot)           y = −Ly/2  (bottom wall)
        #
        # At each y, the barrier spans  x_left(y) ≤ x ≤ x_right(y)
        # where x_left and x_right are linear functions of y.
        if cfg.edges is None or len(cfg.edges) != 4:
            raise ValueError(
                "shaped potential requires edges=[xL_bot, xR_bot, xL_top, xR_top]"
            )
        xL_b, xR_b, xL_t, xR_t = cfg.edges

        # Normalised y: 0 at bottom wall, 1 at top wall
        t = (Y + grid.Ly / 2.0) / grid.Ly     # shape (Ny, Nx), range [0, 1]

        x_left  = xL_b + (xL_t - xL_b) * t    # left edge as function of y
        x_right = xR_b + (xR_t - xR_b) * t    # right edge as function of y

        V = np.where((X >= x_left) & (X <= x_right), cfg.height, 0.0)
        return np.asarray(V, dtype=REAL_DTYPE)

    elif ptype == "polygon":
        # Arbitrary polygon barrier: V = V0 inside the polygon, 0 outside.
        # Vertices may use symbolic strings ("Ly", "-Ly", "Lx/2", etc.)
        # which are resolved against the grid dimensions, with the
        # convention Ly := grid.Ly/2 (so "Ly" is the top-wall location).
        if cfg.vertices is None or len(cfg.vertices) < 3:
            raise ValueError(
                "polygon potential requires vertices=[[x1,y1],[x2,y2],...] "
                "with at least 3 points."
            )
        verts = _resolve_polygon_vertices(cfg.vertices, grid)
        V = _polygon_mask(grid, verts) * cfg.height
        if cfg.smoothing_width is not None and cfg.smoothing_width > 0:
            V = smooth_potential_xrows(V, grid.x, cfg.smoothing_width,
                                       V0=cfg.height)
        return np.asarray(V, dtype=REAL_DTYPE)

    elif ptype == "multi":
        # Multiple independent barriers, each with its own geometry.
        # The final potential is the SUM of all sub-barriers.
        #
        # Height policy (matches all other potential types in this module):
        # the top-level cfg.height is the authoritative V0. Any per-sub-barrier
        # height set in the YAML is OVERRIDDEN by cfg.height so that V0 sweeps
        # via sweep_V0.py work uniformly. If you need sub-barriers at different
        # heights (e.g. a 1:2:4 voltage ladder), use explicit per-barrier
        # scaling on top of cfg.height — the schema does not currently support
        # this; track it as a follow-up.
        if cfg.barriers is None or len(cfg.barriers) == 0:
            raise ValueError(
                "multi potential requires barriers=[...] with at least one "
                "BarrierConfig entry."
            )
        V = np.zeros(grid.shape, dtype=REAL_DTYPE)
        for i, bcfg in enumerate(cfg.barriers):
            # Convert BarrierConfig → temporary PotentialConfig for dispatch.
            # Sub-barrier height is forced to the top-level cfg.height so that
            # the V0 sweep is honored regardless of YAML per-barrier values.
            sub = PotentialConfig(
                type=bcfg.type,
                x_center=bcfg.x_center,
                width=bcfg.width,
                height=cfg.height,
                edges=bcfg.edges,
                vertices=bcfg.vertices,
                smoothing_width=bcfg.smoothing_width,
            )
            Vi = build_potential(grid, sub)
            V += Vi
        return np.asarray(V, dtype=REAL_DTYPE)

    else:
        raise ValueError(
            f"Unknown potential type: '{cfg.type}'. "
            "Choose from: free, barrier, pn_junction, double_barrier, "
            "shaped, polygon, multi."
        )


# ── polygon barrier helpers ──────────────────────────────────────────────────

# Allowed names in the symbolic vertex resolver. The convention is
#   Lx := grid.Lx / 2     (so "Lx" is the right channel wall, "-Lx" the left)
#   Ly := grid.Ly / 2     (so "Ly" is the top channel wall, "-Ly" the bottom)
# This lets the user write "[-15, 'Ly']" to mean "the top wall at x = -15".
_ALLOWED_SYMBOLS = {"Lx", "Ly"}


def _resolve_symbol(value, grid):
    """
    Resolve a polygon vertex coordinate that may be either a number or a
    symbolic string referencing the grid dimensions.

    Accepted forms:
        42                  -> 42.0
        42.5                -> 42.5
        "Ly"                -> grid.Ly / 2.0
        "-Ly"               -> -grid.Ly / 2.0
        "Lx/2"              -> grid.Lx / 4.0
        "Ly*0.6"            -> 0.6 * grid.Ly / 2.0
        "-Lx/3 + 5"         -> -grid.Lx / 6.0 + 5

    Strings are evaluated against a tiny safe namespace containing only
    Lx and Ly (as defined above). No other names are allowed.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(
            f"polygon vertex coordinate must be a number or symbolic string, "
            f"got {type(value).__name__}: {value!r}"
        )

    expr = value.strip()
    namespace = {
        "Lx": grid.Lx / 2.0,
        "Ly": grid.Ly / 2.0,
        "__builtins__": {},
    }

    # Reject anything that looks unsafe (this is paranoia, since we've already
    # cleared __builtins__, but it's cheap insurance against typos and lets
    # users see a clear error rather than a confusing NameError).
    import re
    allowed = re.compile(r"^[\s0-9.+\-*/()LxLy]+$")
    if not allowed.match(expr):
        raise ValueError(
            f"polygon vertex symbolic expression contains disallowed "
            f"characters: {value!r}. Allowed: digits, + - * / ( ) and "
            f"the symbols Lx, Ly."
        )
    try:
        result = eval(expr, namespace)
    except Exception as e:
        raise ValueError(
            f"failed to evaluate polygon vertex expression {value!r}: {e}"
        )
    return float(result)


def _resolve_polygon_vertices(raw_vertices, grid):
    """
    Convert a list of (x, y) vertices to a numpy array of shape (N, 2),
    resolving any symbolic strings against the grid dimensions.
    """
    out = []
    for i, vert in enumerate(raw_vertices):
        if not hasattr(vert, "__len__") or len(vert) != 2:
            raise ValueError(
                f"polygon vertex {i} must be a pair (x, y), got {vert!r}"
            )
        x = _resolve_symbol(vert[0], grid)
        y = _resolve_symbol(vert[1], grid)
        out.append((x, y))
    return np.asarray(out, dtype=np.float64)


def _polygon_mask(grid, vertices):
    """
    Build a boolean mask of grid points lying inside a closed polygon.

    Uses matplotlib.path.Path for the inside test, which handles
    non-convex polygons via the standard ray-casting algorithm and
    returns a clean (Ny, Nx) bool array.

    Parameters
    ----------
    grid : Grid
    vertices : ndarray, shape (N, 2)

    Returns
    -------
    mask : float ndarray, shape (Ny, Nx)
        1.0 inside the polygon, 0.0 outside.
    """
    from matplotlib.path import Path

    # Stack grid points as (Ny*Nx, 2) and test all at once
    pts = np.column_stack([grid.X.ravel(), grid.Y.ravel()])
    path = Path(vertices, closed=False)
    inside = path.contains_points(pts).reshape(grid.X.shape)
    return inside.astype(REAL_DTYPE)


def smooth_potential_xrows(V: np.ndarray, x_axis: np.ndarray,
                           w: float, V0: Optional[float] = None) -> np.ndarray:
    """
    Apply 1D sigmoid smoothing along x at every horizontal row of a
    pre-built barrier potential V(x, y).

    The function operates *downstream* of any barrier construction
    (shaped, polygon, custom). For each row at fixed y:
        1. Detect the barrier intervals — contiguous regions where
           V(x, y) > V0/2 (so a sharp 0/V0 row gives clean intervals).
        2. For each interval [x_L, x_R], replace it with a product of
           two sigmoids:
                f_L(x) = sigmoid(  2 * (x - x_L) / w )      ramp UP
                f_R(x) = sigmoid( -2 * (x - x_R) / w )      ramp DOWN
           and add V0 · f_L(x) · f_R(x) to the row.
        3. Sum the contributions from all intervals in the row
           (correct for disjoint intervals; mildly inexact only when
           the smoothing width is comparable to the gap between
           neighbouring intervals).

    The sigmoid scale convention is chosen so that `smoothing_width` is
    literally the 5%-to-95% visible transition width. Specifically, the
    profile uses sigmoid(α (x - x_edge)) with α = 2 ln(19) / w, so the
    interface goes from 5%·V0 to 95%·V0 over exactly `w` sim units.
    This matches the intuitive reading: `smoothing_width = 2.0` means
    a transition you can see spanning about 2 units on a plot.

    Physical interpretation
    -----------------------
    For a vertical interface (parallel to y), the row-wise sigmoid is
    *exactly* per-normal smoothing — the normal direction is along x,
    and we're smoothing along x. This is the case for the entrance
    interface of the v1 forward gate.

    For an oblique interface at angle θ to the y-axis (the normal
    points away from the y-axis by angle θ from x), the smoothing
    along x corresponds to per-normal smoothing with effective width
    w · cos(θ). For the v1 forward gate's oblique exit (angle ~25°
    from y-axis), this means the effective normal smoothing is about
    10% smaller than the nominal `w`. This is a ~6-10% inaccuracy in
    the local interface width and does not change the physics
    qualitatively.

    Limits
    ------
    - w → 0 : the function returns V unchanged (no-op, byte identical).
    - w → ∞ : the sigmoids become essentially constant (~V0/2) over
      the entire row, so the function returns approximately V0/2 ·
      (number of intervals) — clearly unphysical at this limit.
    - When two neighbouring intervals in the same row are separated
      by less than ~3w, their sigmoid tails overlap and the sum
      double-counts the gap region. The result is still finite and
      well-behaved, but no longer represents a clean union of
      smoothed barriers. For typical use w ≪ gap and this is fine.

    Parameters
    ----------
    V : (Ny, Nx) float ndarray
        Pre-built sharp barrier potential.
    x_axis : (Nx,) float ndarray
        Cell-centered x coordinates.
    w : float
        Smoothing width in sim units. w ≤ 0 returns V unchanged.
    V0 : float or None
        The barrier height. If None, inferred from V.max(). Required
        only to set the threshold for interval detection (V > V0/2).

    Returns
    -------
    V_smooth : (Ny, Nx) float ndarray
        Smoothed potential with the same shape and dtype as V.
    """
    if w is None or w <= 0:
        return V
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got shape {V.shape}")
    if x_axis.shape[0] != V.shape[1]:
        raise ValueError(
            f"x_axis length {x_axis.shape[0]} does not match "
            f"V.shape[1] = {V.shape[1]}"
        )

    Ny, Nx = V.shape
    if V0 is None:
        V0 = float(V.max())
        if V0 <= 0:
            return V              # nothing to smooth — V is identically zero

    threshold = 0.5 * V0
    # Sigmoid scale chosen so that the 5%-95% transition width equals w.
    # The 5% to 95% width of sigmoid(α x) is 2 ln(19) / α, so we set
    # α = 2 ln(19) / w. The result: smoothing_width = w gives a visible
    # transition spanning exactly w sim units.
    alpha = 2.0 * np.log(19.0) / w

    V_smooth = np.zeros_like(V)

    for iy in range(Ny):
        row = V[iy]
        # Binary mask for "inside the barrier" along this row
        mask = row > threshold
        if not mask.any():
            continue                # row entirely outside barrier; leave at 0

        # Find transitions: 0→1 (left edges) and 1→0 (right edges)
        # Pad with False on both sides so transitions at row boundaries
        # are detected correctly.
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(np.int8))
        left_edges  = np.where(diff ==  1)[0]   # indices in original row
        right_edges = np.where(diff == -1)[0] - 1  # last index inside

        # For each interval, compute its smoothed contribution
        for iL, iR in zip(left_edges, right_edges):
            x_L = x_axis[iL]
            x_R = x_axis[iR]
            f_up   = _sigmoid(alpha * (x_axis - x_L))
            f_down = _sigmoid(alpha * (x_R - x_axis))
            V_smooth[iy] += V0 * f_up * f_down

    return V_smooth.astype(V.dtype)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp for very negative x
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


# ── visualisation helpers ────────────────────────────────────────────────────

def potential_extents(cfg: PotentialConfig) -> list[tuple[float, float]]:
    """
    Return a list of (x_left, x_right) extents for barrier shading in plots.
    For shaped/polygon barriers, returns the bounding box (widest extent).
    """
    if cfg.type == "barrier":
        return [(cfg.x_center - cfg.width / 2.0, cfg.x_center + cfg.width / 2.0)]
    elif cfg.type == "pn_junction":
        return [(cfg.x_center, None)]   # None → extends to +∞
    elif cfg.type == "double_barrier":
        gap = cfg.width
        D = cfg.width
        lc = cfg.x_center - gap / 2.0 - D / 2.0
        rc = cfg.x_center + gap / 2.0 + D / 2.0
        return [(lc - D / 2.0, lc + D / 2.0), (rc - D / 2.0, rc + D / 2.0)]
    elif cfg.type == "shaped" and cfg.edges is not None:
        xL_b, xR_b, xL_t, xR_t = cfg.edges
        return [(min(xL_b, xL_t), max(xR_b, xR_t))]
    elif cfg.type == "polygon" and cfg.vertices is not None:
        # Bounding box of the resolved polygon — note this needs the grid
        # to resolve symbolic strings, so we return None and let callers
        # use potential_polygon() instead for full geometry.
        return []
    elif cfg.type == "multi" and cfg.barriers is not None:
        # Aggregate extents from each sub-barrier
        all_extents = []
        for bcfg in cfg.barriers:
            sub = PotentialConfig(
                type=bcfg.type, x_center=bcfg.x_center, width=bcfg.width,
                height=bcfg.height, edges=bcfg.edges, vertices=bcfg.vertices,
                smoothing_width=bcfg.smoothing_width,
            )
            all_extents.extend(potential_extents(sub))
        return all_extents
    return []


def potential_polygon(cfg: PotentialConfig, Ly: float, grid=None) -> list[tuple[float, float]] | None:
    """
    Return the barrier outline as a list of (x, y) vertices for polygon overlay.

    For "shaped" barriers, returns the trapezoid corners.
    For "polygon" barriers, returns the resolved vertex list (requires `grid`
    to resolve symbolic strings like "Ly", "-Ly").
    Returns None for other types (use axvspan instead).
    """
    if cfg.type == "shaped" and cfg.edges is not None:
        xL_b, xR_b, xL_t, xR_t = cfg.edges
        y_bot = -Ly / 2.0
        y_top =  Ly / 2.0
        # CCW polygon: bottom-left → bottom-right → top-right → top-left
        return [
            (xL_b, y_bot),
            (xR_b, y_bot),
            (xR_t, y_top),
            (xL_t, y_top),
        ]
    if cfg.type == "polygon" and cfg.vertices is not None:
        if grid is None:
            # Best-effort: try to resolve assuming Ly came from the caller
            from types import SimpleNamespace
            grid = SimpleNamespace(Lx=Ly, Ly=Ly)
        verts = _resolve_polygon_vertices(cfg.vertices, grid)
        return [(float(x), float(y)) for x, y in verts]
    if cfg.type == "multi" and cfg.barriers is not None:
        # Return the first sub-barrier's polygon for backward compat.
        # For all polygons, use potential_polygons().
        for bcfg in cfg.barriers:
            sub = PotentialConfig(
                type=bcfg.type, height=bcfg.height, edges=bcfg.edges,
                vertices=bcfg.vertices, smoothing_width=bcfg.smoothing_width,
            )
            result = potential_polygon(sub, Ly, grid=grid)
            if result is not None:
                return result
    return None


def potential_polygons(cfg: PotentialConfig, Ly: float, grid=None) -> list[list[tuple[float, float]]]:
    """
    Return ALL barrier outlines as a list of vertex lists.

    For single-barrier types (shaped, polygon), returns a list with one element.
    For type="multi", returns one vertex list per sub-barrier that has polygon
    geometry (skips rectangular barriers that use axvspan instead).
    """
    if cfg.type in ("shaped", "polygon"):
        p = potential_polygon(cfg, Ly, grid=grid)
        return [p] if p is not None else []
    if cfg.type == "multi" and cfg.barriers is not None:
        polys = []
        for bcfg in cfg.barriers:
            sub = PotentialConfig(
                type=bcfg.type, height=bcfg.height, edges=bcfg.edges,
                vertices=bcfg.vertices, smoothing_width=bcfg.smoothing_width,
            )
            p = potential_polygon(sub, Ly, grid=grid)
            if p is not None:
                polys.append(p)
        return polys
    return []


# ── electrostatic disorder (charge puddles) ──────────────────────────────────

def build_disorder(
    V: np.ndarray,
    grid: Grid,
    cfg: DisorderConfig,
    V0: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Add spatially correlated electrostatic disorder on top of V(x,y).

    Models local top-gate voltage variations from oxide impurities or
    substrate defects ("charge puddles").  The disorder is applied ONLY
    inside the barrier region (where V ≠ 0).

    Algorithm
    ---------
    1.  Generate white noise ξ(x,y) on the grid  (mean 0, unit variance).
    2.  Convolve with a Gaussian kernel of width σ = L_c / dx to produce
        spatially correlated fluctuations with correlation length L_c.
    3.  Identify the barrier region (where |V| > 0).
    4.  Normalise so that the RMS of the filtered field *inside the barrier*
        equals 1, ensuring the actual δV_rms matches dv × V0 exactly.
    5.  Scale by D_V × V0  where D_V = cfg.dv.
    6.  Apply δV only inside the barrier region.

    Parameters
    ----------
    V     : ndarray (Ny, Nx) — clean potential (modified in-place).
    grid  : Grid
    cfg   : DisorderConfig
    V0    : float — barrier height (used for amplitude scaling).
    verbose : bool

    Returns
    -------
    V : same array, with disorder added inside the barrier.
    """
    from scipy.ndimage import gaussian_filter

    if not cfg.enabled or cfg.dv <= 0 or V0 == 0:
        return V

    rng = np.random.default_rng(cfg.seed)

    # 1. White noise on the grid
    xi = rng.standard_normal(grid.shape).astype(REAL_DTYPE)

    # 2. Gaussian blur — sigma in grid-cell units
    sigma_x = cfg.lc / grid.dx
    sigma_y = cfg.lc / grid.dy
    correlated = gaussian_filter(xi, sigma=[sigma_y, sigma_x], mode="wrap")

    # 3. Identify barrier region
    barrier_mask = np.abs(V) > 1e-12

    # 4. Normalise to unit RMS *inside the barrier* so that
    #    the actual δV_rms matches the requested dv × V0 exactly.
    corr_inside = correlated[barrier_mask]
    rms = float(np.sqrt(np.mean(corr_inside ** 2)))
    if rms > 0:
        correlated /= rms

    # 5. Scale to desired amplitude and apply only inside the barrier
    dV = correlated * (cfg.dv * abs(V0))
    V[barrier_mask] += dV[barrier_mask].astype(REAL_DTYPE)

    if verbose:
        dV_inside = dV[barrier_mask]
        n_cells = int(np.sum(barrier_mask))
        print(f"  Disorder   : δV_rms={cfg.dv*100:.1f}% of V0={V0:.4f}  "
              f"L_c={cfg.lc:.2f}  σ_grid=({sigma_x:.1f},{sigma_y:.1f}) cells  "
              f"seed={cfg.seed}")
        print(f"               |δV| range [{float(dV_inside.min()):.4f}, "
              f"{float(dV_inside.max()):+.4f}]  "
              f"RMS={float(np.std(dV_inside)):.4f}  "
              f"applied to {n_cells} barrier cells")

    return V


# ── source-drain bias (potential gradient) ───────────────────────────────────

def add_source_drain_bias(
    V: np.ndarray,
    grid: Grid,
    cfg: BiasConfig,
    verbose: bool = True,
) -> np.ndarray:
    """
    Add a source-drain bias term V_bias(x) on top of an existing potential.

    This models a small voltage applied between the source (left, x=-Lx/2)
    and the drain (right, x=+Lx/2). For positive V_sd the source is at
    higher potential than the drain, so the semiclassical force −∂V_bias/∂x
    is in +x — i.e. carriers are accelerated from source toward drain.
    A side benefit is soft "confinement" of the reflected wavepacket: any
    back-scattered component climbs an uphill on the source side and is
    pushed back toward the scattering region, making the transport measurement
    more realistic and numerically better-behaved.

    Profiles (selected by ``cfg.type``)
    -----------------------------------
    linear (default)
        V_bias(x) = -V_sd · (x - x_center) / L_drop
        Uniform electric field E_x = V_sd / L_drop along the whole channel.
        L_drop=None → uses grid.Lx so the total drop across the domain is
        exactly V_sd.

    localized
        V_bias(x) = V_sd · ( 1/2 − sigmoid( α · (x − x_center) ) )
        with α = 2·ln(19) / L_drop, so L_drop is the visible 5%→95% width.
        Far-field limits are +V_sd/2 (left) and -V_sd/2 (right), matching
        the linear profile. Use this when the voltage should drop
        essentially within the scattering region rather than across the
        whole channel (ballistic device geometry).
        L_drop=None → defaults to 10 sim units.

    Application order
    -----------------
    This MUST be called after any disorder has been applied, because
    build_disorder() uses |V| > 0 to detect the barrier region, and the
    bias adds a small nonzero value everywhere.

    Parameters
    ----------
    V     : (Ny, Nx) real ndarray  — existing potential; modified OUT-OF-PLACE.
    grid  : Grid
    cfg   : BiasConfig
    verbose : bool                  — print a diagnostic line if True.

    Returns
    -------
    V_biased : (Ny, Nx) real ndarray
        V + V_bias, same dtype as the input V. If bias is disabled or
        V_sd == 0, returns V unchanged (no copy).
    """
    if not cfg.enabled or cfg.V_sd == 0.0:
        return V

    X = grid.X
    btype = cfg.type.lower()

    if btype == "linear":
        L_drop = float(cfg.L_drop) if (cfg.L_drop is not None and cfg.L_drop > 0) \
                 else float(grid.Lx)
        if L_drop <= 0:
            raise ValueError(f"bias.L_drop must be positive, got {L_drop}")
        V_bias = -float(cfg.V_sd) * (X - float(cfg.x_center)) / L_drop
        E_field = float(cfg.V_sd) / L_drop

    elif btype == "localized":
        L_drop = float(cfg.L_drop) if (cfg.L_drop is not None and cfg.L_drop > 0) \
                 else 10.0
        if L_drop <= 0:
            raise ValueError(f"bias.L_drop must be positive, got {L_drop}")
        alpha = 2.0 * np.log(19.0) / L_drop
        sig = _sigmoid(alpha * (X - float(cfg.x_center)))
        V_bias = float(cfg.V_sd) * (0.5 - sig)
        # Peak field magnitude at the inflection (x = x_center):
        #   d/dx [V_sd · (1/2 − σ(α·x))]|_0 = -V_sd · σ'(0) · α = -V_sd · α/4
        E_field = float(cfg.V_sd) * alpha / 4.0
    else:
        raise ValueError(
            f"Unknown bias.type='{cfg.type}'. Expected 'linear' or 'localized'."
        )

    V_out = (V + V_bias).astype(V.dtype, copy=False)

    if verbose:
        V_left  = float(V_bias[:, 0].mean())
        V_right = float(V_bias[:, -1].mean())
        print(f"  Bias       : V_sd={cfg.V_sd:+.5f}  type={btype}  "
              f"L_drop={L_drop:.2f}  x_c={float(cfg.x_center):.2f}")
        print(f"               V_bias(-Lx/2)={V_left:+.5f}  "
              f"V_bias(+Lx/2)={V_right:+.5f}  "
              f"|E_x|={abs(E_field):.2e}")

    return V_out
