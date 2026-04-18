"""
Config module: dataclass hierarchy for all simulation parameters.
Loaded from a YAML file via load_config().
"""

import yaml
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── sub-configs ────────────────────────────────────────────────────────────────

@dataclass
class GridConfig:
    Nx: int = 256
    Ny: int = 256
    Lx: float = 60.0
    Ly: float = 40.0


@dataclass
class PhysicsConfig:
    vf: float = 1.0
    tilt: List[float] = field(default_factory=lambda: [0.0, 0.0])
    valleys: str = "K"            # "K" | "Kp" | "both"
                                  # K uses tilt as given; Kp negates it;
                                  # both runs K and K' in the same simulation

    source_mode: str = "both"     # "both" | "K_only" | "Kp_only"
                                  # Only meaningful when valleys='both' AND
                                  # coupling is enabled. Selects which valley
                                  # block is populated at t = 0 in the 4-spinor
                                  # coupled propagator. Two single-source runs
                                  # let you reconstruct the full A_{σσ'} channel
                                  # response matrix for readout-visibility and
                                  # arbitrary injection superpositions.

    @property
    def wx(self) -> float:
        return self.tilt[0]

    @property
    def wy(self) -> float:
        return self.tilt[1]


@dataclass
class TimeConfig:
    dt: float = 0.04
    n_steps: int = 800
    save_every: int = 20          # save a frame every N steps


@dataclass
class WavepacketConfig:
    """
    Shared injection for both valleys.

    Design note (vQubit v1.5):
    --------------------------
    Both valleys are injected with the SAME (kx0, ky0). This is the physically
    correct choice: the valley phase gate mechanism requires K and K' to impinge
    on the oblique exit facet with the SAME incident wavevector, so that the
    refraction-angle contrast arises purely from the tilted dispersion
    (opposite tilt sign at K and K'). An earlier v2 attempt injected each
    valley with a different k to align group velocities along x; this destroys
    the mechanism and was removed. Do not reintroduce per-valley injection
    angles without first revisiting the device physics.
    """
    k0: float = 2.0               # central wavevector magnitude
    theta: float = 0.0            # injection angle (deg), shared by K and K'
    sigma_x: float = 5.0          # spatial width in x (ℏ = 1 units)
    sigma_y: float = 5.0          # spatial width in y
    x_source: float = -0.45       # source position as fraction of Lx/2

    @property
    def theta_rad(self) -> float:
        return math.radians(self.theta)

    @property
    def kx0(self) -> float:
        return self.k0 * math.cos(self.theta_rad)

    @property
    def ky0(self) -> float:
        return self.k0 * math.sin(self.theta_rad)


@dataclass
class BarrierConfig:
    """Single barrier definition for use in multi-barrier (type='multi') mode.

    Each barrier has its own type, geometry, height, and smoothing.
    Supported sub-types: 'polygon', 'barrier', 'shaped'.
    """
    type: str = "polygon"
    x_center: float = 0.0         # for type="barrier"
    width: float = 3.0            # for type="barrier"
    height: float = 4.0           # V0 for this barrier
    edges: Optional[List[float]] = None   # for type="shaped"
    vertices: Optional[List[List]] = None  # for type="polygon"
    smoothing_width: float = 0.0


@dataclass
class PotentialConfig:
    type: str = "barrier"         # "barrier" | "pn_junction" | "double_barrier" | "shaped" | "polygon" | "multi" | "free"
    x_center: float = 0.0
    width: float = 3.0            # barrier full width (D in the docs)
    height: float = 4.0           # V0 (barrier height)
    edges: Optional[List[float]] = None
        # shaped barrier: [x_left_bot, x_right_bot, x_left_top, x_right_top]
        # linearly interpolated between y = −Ly/2 (bottom) and y = +Ly/2 (top)
        # covers trapezoid, triangle, rotated rectangle, etc.
    vertices: Optional[List[List]] = None
        # polygon barrier: arbitrary list of (x, y) vertices defining a closed
        # polygon. Each vertex can be either two numbers, or strings that
        # reference grid dimensions: "Lx", "-Lx", "Ly", "-Ly", "Lx/2", etc.
        # The convention is Ly := grid.Ly/2  (so "Ly" is the top-wall location
        # and "-Ly" is the bottom-wall location). Same for Lx.
        # Example (pentagon shape on a 300x300 grid):
        #   vertices:
        #     - [-15, "-Ly"]
        #     - [60, "-Ly"]
        #     - [15, 0]
        #     - [60, "Ly"]
        #     - [-15, "Ly"]
        # The polygon may be given in any order (CW or CCW); it does not
        # need to be explicitly closed (the last vertex is auto-connected
        # to the first). Non-convex shapes are supported.
    smoothing_width: float = 0.0
        # Optional 1D-per-row smoothing width applied to the polygon mask
        # AFTER it is built. Each horizontal row of V(x,y) is processed
        # independently: the contiguous barrier intervals in that row are
        # detected (V > V0/2), and each interval [x_L, x_R] contributes
        #
        #     V_smooth(x) += V0 · sigmoid(α(x − x_L)) · sigmoid(α(x_R − x))
        #
        # to the row, where α = 2·ln(19)/w. This convention makes
        # `smoothing_width` literally the visible 5%-to-95% transition
        # width on a plot, in sim units (same units as Lx/Ly).
        #
        # Formulation: PRODUCT of two sigmoids per interval (not sum
        # from a base value). Both sigmoids are positive and ≤ 1; their
        # product is ≈ 1 in the deep interior of a wide barrier, ≈ 0
        # outside, and = 0.25 on the visible edges. For a barrier
        # narrower than the smoothing width, the two sigmoids overlap
        # destructively and the peak value falls below V0 — this models
        # the physical fringing-field suppression of narrow gates and is
        # NOT a numerical artifact.
        #
        # For w → 0 the smoothing is a no-op (early return) and the
        # barrier is exactly the sharp polygon mask, byte-identical.
        #
        # Per-interface direction: the smoothing is along x at each row.
        # For a vertical interface (parallel to y) this is exactly per-
        # normal smoothing. For an oblique interface at angle θ from y,
        # the equivalent per-normal smoothing width is w·cos(θ); for the
        # v1 forward gate's oblique exit (~25° from y) this is a ~10%
        # geometric correction and does not change the physics
        # qualitatively.
        #
        # Multi-interval rows (e.g., a y row that crosses two disjoint
        # barrier regions) are handled correctly as long as the inter-
        # barrier gap is wider than ~3w; closer than that the sigmoid
        # tails overlap and the gap region picks up a small spurious
        # contribution (~1% of V0 at gap = w).
    barriers: Optional[List] = None
        # List of BarrierConfig dicts for type="multi" mode.
        # Each entry defines an independent barrier with its own type,
        # vertices/edges, height, and smoothing. The final potential is the
        # SUM of all sub-barriers.


@dataclass
class DisorderConfig:
    """Spatially correlated electrostatic disorder (charge puddles).

    Models local potential variations from oxide impurities or substrate
    defects by adding δV(x,y) on top of the barrier potential:

        V(x,y) → V(x,y) + δV(x,y)

    δV is generated as Gaussian-correlated noise: white noise filtered
    with a Gaussian kernel of width L_c, then scaled so that the RMS
    amplitude equals dv × V0.  Applied only inside the barrier region.

    Parameters
    ----------
    enabled : bool
        Master switch for disorder.
    dv : float
        Disorder amplitude as a fraction of V0.
        E.g. dv=0.05 → δV_rms = 5% of V0.
    lc : float
        Spatial correlation length (same units as Lx, Ly).
        Controls the puddle size. Typical: 1–10 nm.
    seed : int or None
        Random seed for reproducible disorder realisations.
    """
    enabled: bool = False
    dv: float = 0.0               # fractional amplitude  δV_rms / V0
    lc: float = 3.0               # correlation length (sim units)
    seed: Optional[int] = None    # RNG seed for reproducibility


@dataclass
class AbsorberConfig:
    width_frac: float = 0.12      # absorbing layer width as fraction of domain size
    strength: float = 0.025       # max damping strength S per step
    x_only: bool = True           # DEPRECATED — use y_bc instead
                                  # kept for backward compatibility only
    y_bc: str = "periodic"        # transverse boundary condition:
                                  #   "periodic"   — FFT wrap, no y treatment (V=V(x) only)
                                  #   "absorbing"  — cos² damping at y-edges (wave exits channel)
                                  #   "reflecting" — mass confinement M(y)·σz at y-edges
                                  #                  (gap opening, correct for Dirac fermions —
                                  #                   scalar V wall fails due to Klein tunneling)
    y_width_frac: float = 0.0     # y-absorber/wall width as fraction of Ly
                                  # 0 → use width_frac (same as x)
    y_strength: float = 0.0       # y-absorber strength; 0 → use strength
    wall_mass: float = 0.0        # mass confinement height for reflecting walls
                                  # 0 → auto-computed as 20 × max(V0, Ef)


@dataclass
class OutputConfig:
    colormap: str = "inferno"
    show_potential: bool = True
    show_spinor: bool = False     # if True, also plot individual spinor components
    show_current: bool = False    # overlay Dirac current field on frames
    current_style: str = "quiver" # "quiver" | "stream" | "both"
    current_stride: int = 0       # 0 → auto-thin; 1 → every grid cell
    current_normalize: bool = True
    current_min_density_frac: float = 0.02
    current_max_arrow_length: float = 1.10
    current_color: str = "#00c8ff"
    current_alpha: float = 0.95
    current_width: float = 0.0055
    current_stream_density: float = 1.25
    current_stream_linewidth: float = 1.15
    current_stream_arrowsize: float = 0.95
    dpi: int = 120
    figsize: List[float] = field(default_factory=lambda: [10.0, 5.0])


@dataclass
class DetectorConfig:
    enabled: bool = True          # enable flux detectors for T/R measurement
    auto_source: bool = True      # auto-position source based on sigma & barrier
    auto_stop: bool = True        # stop simulation when wave has cleared barrier
    auto_grid: bool = False       # auto-compute Lx, Ly, Nx, Ny from physics
    n_sigma_source: float = 3.5   # source distance from barrier (in sigma_x)
    n_sigma_det: float = 2.0      # detector offset from barrier (in sigma_x)
    stop_threshold: float = 0.005 # stop when barrier-region P < this fraction
    stop_patience: int = 200      # confirm convergence for this many steps
    pts_per_wavelength: float = 6.0  # grid resolution target (for auto_grid)


@dataclass
class CouplingConfig:
    """
    Intervalley coupling between K and K' valleys.

    The coupled propagator (CoupledSplitOperatorPropagator) treats K and K'
    as a single 4-component spinor and applies a scalar intervalley coupling
    U_KK'(x,y) = u(x,y) · I_2 in the real-space half-step.

    When ``enabled=False``, the simulation runs in v1.0 mode with two
    independent propagators (one per valley). The coupled propagator with
    ``strength=0`` reproduces the v1.0 result bit-for-bit.
    """
    enabled: bool = False
    type: str = "barrier"         # "barrier"  – localised inside V > threshold·V0
                                  # "uniform"  – constant everywhere
                                  # "defect_line" – Gaussian along x = position
    strength: float = 0.0         # coupling amplitude in simulation energy units
    region_threshold: float = 0.5 # for type="barrier": V > threshold·|V0| → active
    line_position: float = 0.0    # for type="defect_line": x-coordinate
    line_width: float = 2.0       # for type="defect_line": Gaussian width


@dataclass
class BiasConfig:
    """
    Source-drain bias — a rigid potential gradient added on top of V(x,y)
    to model a small voltage applied between the source (left) and drain
    (right) contacts.

    Physics
    -------
    The bias adds an additive term  V_bias(x)  to the barrier potential.
    Two spatial profiles are supported (selected via ``type``):

        type = "linear"  (default):
            V_bias(x) = -V_sd · (x - x_center) / L_drop
            With x_center = 0 and L_drop = grid.Lx, this gives
              V_bias(-Lx/2) = +V_sd/2   (source, higher potential)
              V_bias(+Lx/2) = -V_sd/2   (drain,  lower  potential)
            i.e. the total source→drain drop is exactly V_sd.

        type = "localized":
            V_bias(x) = V_sd · ( 1/2 − sigmoid( α · (x − x_center) ) )
            with α = 2·ln(19)/L_drop so that L_drop is the visible 5%→95%
            transition width. Models a bias that drops almost entirely
            across the scattering region, as in ballistic devices.

    Sign convention
    ---------------
    Positive V_sd means source (left) is at higher potential than drain
    (right). The semiclassical force −∂V_bias/∂x is therefore in +x,
    pushing carriers from source toward drain (conventional bias sense).
    This also softly "confines" the wavepacket on the drain side: reflected
    components climb a potential hill on the source side and are pushed
    back toward the barrier.

    Units
    -----
    ``V_sd`` is in the same energy units as everything else in this package
    (same units as potential.height and physics.vf · wavepacket.k0). For
    typical "small-bias" regime use V_sd ≪ E_F (say a few percent of E_F).

    Application order
    -----------------
    In simulation.py the bias is applied AFTER build_potential() and AFTER
    build_disorder() — applying it before disorder would pollute the
    ``|V| > 0`` barrier mask used by the puddle model.
    """
    enabled: bool = False
    V_sd: float = 0.0             # source-drain voltage drop (sim energy units)
    type: str = "linear"          # "linear" | "localized"
    x_center: float = 0.0         # linear: zero-reference of the ramp
                                  # localized: center of the sigmoid transition
    L_drop: Optional[float] = None  # linear: total drop length
                                    #         None → use grid.Lx (full domain)
                                    # localized: 5%→95% transition width
                                    #         None → default = 10 sim units


# ── master config ──────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    wavepacket: WavepacketConfig = field(default_factory=WavepacketConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    disorder: DisorderConfig = field(default_factory=DisorderConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    absorber: AbsorberConfig = field(default_factory=AbsorberConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    title: str = "Dirac Wavepacket"
    output_dir: str = "./results"


# ── loader ─────────────────────────────────────────────────────────────────────

def _merge(dataclass_instance, d: dict):
    """Recursively update a dataclass from a dictionary."""
    if d is None:
        return dataclass_instance
    for key, value in d.items():
        if not hasattr(dataclass_instance, key):
            raise ValueError(f"Unknown config key: '{key}'")
        current = getattr(dataclass_instance, key)
        # Special case: PotentialConfig.barriers is a list of BarrierConfig
        if key == "barriers" and isinstance(value, list):
            barrier_list = []
            for item in value:
                if isinstance(item, dict):
                    bc = BarrierConfig()
                    _merge(bc, item)
                    barrier_list.append(bc)
                else:
                    barrier_list.append(item)
            setattr(dataclass_instance, key, barrier_list)
        elif hasattr(current, '__dataclass_fields__'):
            _merge(current, value)
        else:
            setattr(dataclass_instance, key, value)
    return dataclass_instance


def load_config(path: str) -> SimConfig:
    """Load a SimConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = SimConfig()
    _merge(cfg, raw)
    return cfg


def config_summary(cfg: SimConfig) -> str:
    """Return a human-readable summary of the simulation config."""
    g = cfg.grid
    p = cfg.physics
    t = cfg.time
    w = cfg.wavepacket
    v = cfg.potential
    Ef = p.vf * w.k0
    if v.type == "shaped" and v.edges is not None:
        pot_line = f"  Potential  : {v.type}  V0={v.height}  edges={v.edges}"
    elif v.type == "multi" and v.barriers is not None:
        pot_line = f"  Potential  : multi  V0={v.height}  ({len(v.barriers)} barriers)"
    elif v.type == "polygon" and v.vertices is not None:
        pot_line = f"  Potential  : polygon  V0={v.height}  ({len(v.vertices)} vertices)"
    else:
        pot_line = f"  Potential  : {v.type}  V0={v.height}  width={v.width}  center={v.x_center}"
    a = cfg.absorber
    abs_line = f"  Absorber   : y_bc={a.y_bc}  width_frac={a.width_frac}  S={a.strength}"
    lines = [
        f"{'─'*55}",
        f"  {cfg.title}",
        f"{'─'*55}",
        f"  Grid       : {g.Nx}×{g.Ny}  ({g.Lx}×{g.Ly} a.u.)",
        f"  Physics    : vf={p.vf}  tilt=[{p.wx},{p.wy}]  valleys={p.valleys}",
        f"  Wavepacket : k0={w.k0}  θ={w.theta}° (K=K')  σ=({w.sigma_x},{w.sigma_y})",
        f"               Ef={Ef:.2f}  Δθ≈{math.degrees(1/(2*w.sigma_x*w.k0)):.1f}°",
        pot_line,
        abs_line,
        f"  Time       : dt={t.dt}  steps={t.n_steps}  save_every={t.save_every}",
        f"  Output     : {cfg.output_dir}",
    ]
    dis = cfg.disorder
    if dis.enabled and dis.dv > 0:
        lines.insert(-1,
            f"  Disorder   : δV_rms={dis.dv*100:.1f}% of V0  "
            f"L_c={dis.lc}  seed={dis.seed}")
    b = cfg.bias
    if b.enabled and b.V_sd != 0.0:
        L_auto = "Lx" if b.type == "linear" else "10"
        L_str = f"{b.L_drop}" if b.L_drop else f"auto({L_auto})"
        lines.insert(-1,
            f"  Bias (V_sd): {b.V_sd:+.4f}  type={b.type}  "
            f"x_c={b.x_center}  L_drop={L_str}")
    lines.append(f"{'─'*55}")
    return "\n".join(lines)
