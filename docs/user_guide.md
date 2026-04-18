# User Guide

This guide walks through practical use of `dirac_wavepacket` at three levels:

1. **Installation and quickstart** — get a simulation running.
2. **Configuration reference** — annotated schema for every YAML field.
3. **Common patterns and checklist** — idioms, production-run tips,
   sweep-driver recipes.

For the physics and numerical method, see [`theory.md`](theory.md).
For end-to-end physics examples, see [`../examples/`](../examples).

---

## Installation

```bash
git clone https://github.com/can-yesilyurt/Dirac-Wavepacket
cd Dirac-Wavepacket
pip install -e .                 # core
pip install -e ".[fft]"          # + pyFFTW acceleration (2-5x speedup)
pip install -e ".[dev]"          # + pytest for running the test suite
```

Python 3.11 or newer is required.

## Quickstart

The shipped reference configuration runs a valley-resolved transport
simulation through a rectangular barrier:

```bash
dwp examples/configs/reflecting_walls_w35_W50.yaml --output results/quickstart
```

This writes a final snapshot, probability-history plot, transmission
report, and wavefunction snapshot series into `results/quickstart/`.

To run one of the worked examples:

```bash
python examples/01_angled_barrier_valley_filter.py --barrier angled
python examples/02_klein_angular_dependence.py --angles -15 -5 5 15
```

Both support `--quick` for a ~3 min smoke test.

---

## Configuration reference

A `SimConfig` object fully specifies a simulation. YAML configs are
loaded by `load_config()`; programmatic configs are built directly
as Python dataclasses. Every field has a sensible default; only the
physics / geometry / wavepacket fields typically need tuning.

### `grid`

Uniform Cartesian grid. `Nx`, `Ny` must be FFT-friendly powers of 2
(or smooth composites) for best performance; the auto-validator will
warn if they are awkward.

```yaml
grid:
  Nx: 512                  # gridpoints along x
  Ny: 256                  # gridpoints along y
  Lx: 300.0                # domain length in simulation units
  Ly: 200.0                # domain width
```

In simulation units `hbar = v_F = 1`, positions are dimensionless.
The default unit scale is 6.582 nm per simulation unit (consistent
with `hbar v_F = 6.582e-16 eV·s × 10⁶ m/s` and `E_F` in eV), but
the code does not impose units — all post-processing can rescale.

### `physics`

Dispersion parameters.

```yaml
physics:
  vf: 1.0                  # isotropic Fermi velocity, sim units
  vx: null                 # anisotropic x-velocity (overrides vf if set)
  vy: null                 # anisotropic y-velocity (overrides vf if set)
  tilt: [0.0, 0.3]         # (w_x, w_y) at K valley; K' uses (-w_x, -w_y)
  valleys: "both"          # "K", "Kp", or "both"
```

For a tilted 8-Pmmn-borophene-like dispersion: `vf: 1.0`, `tilt: [0.0, 0.32]`
(transverse tilt only). For a Weyl-semimetal analogue: set both `vx`
and `vy`. For an isotropic Dirac cone: `vf: 1.0` with `tilt: [0.0, 0.0]`.

### `wavepacket`

Gaussian injection. The `theta` angle is in **degrees** and is
converted internally to (`kx0`, `ky0`) via
`kx0 = k0 cos(theta), ky0 = k0 sin(theta)`.

```yaml
wavepacket:
  k0: 0.8                  # central wavenumber magnitude; E_F = hbar v_F k_0
  theta: 0.0               # injection angle, degrees
  sigma_x: 7.0             # real-space 1/e radius, x
  sigma_y: 7.0             # real-space 1/e radius, y
  x_source: -0.75          # source x as a fraction of Lx/2
```

The y-position of the source is currently fixed at y = 0. Angular
spread in k-space is approximately `1 / (sigma · k0)` radians.

### `potential`

Electrostatic barrier. Several geometries are supported via the `type`
field.

#### Rectangular (`type: barrier`)

```yaml
potential:
  type: barrier
  height: 1.6              # V_0 in sim units
  x_center: 0.0            # barrier centre along x
  width: 50.0              # barrier thickness along x
  smoothing_width: 0.0     # sigmoid edge smoothing (0 = sharp)
```

#### n-p junction (`type: pn_junction`)

A single step from 0 to `height` at `x_step`.

```yaml
potential:
  type: pn_junction
  height: 1.2
  x_step: 0.0
  smoothing_width: 2.0
```

#### Shaped parallelogram (`type: shaped`)

A barrier with four edges given by four x-positions: the left/right
positions at y = −Ly/2 (bottom) and at y = +Ly/2 (top). Setting these
four values makes a rotated rectangle, a trapezoid, or a triangle.

```yaml
potential:
  type: shaped
  height: 1.4
  edges: [-25.0, 25.0, -25.0, 25.0]     # rectangular at alpha=0
  smoothing_width: 0.0
```

For a barrier rotated by angle α about the y-axis:

```python
shift = (Ly/2) * math.tan(math.radians(alpha))
edges = [-d/2 - shift, d/2 - shift, -d/2 + shift, d/2 + shift]
```

#### Polygon (`type: polygon`)

Arbitrary convex or non-convex polygons. Vertices are given as
(x, y) pairs; symbolic `Lx`, `Ly` references are resolved at grid-build
time.

```yaml
potential:
  type: polygon
  height: 1.4
  vertices:
    - [-25.0,  "-Ly/2"]
    - [ 25.0,  "-Ly/2"]
    - [ 25.0,  "+Ly/2"]
    - [-25.0,  "+Ly/2"]
  smoothing_width: 0.0
```

#### Multiple barriers (`type: multi`)

Each sub-barrier has its own `type` and geometry; the total potential
is their sum. Useful for double-barrier resonant structures.

```yaml
potential:
  type: multi
  barriers:
    - type: barrier
      height: 1.2
      x_center: -40.0
      width: 15.0
    - type: barrier
      height: 1.2
      x_center: +40.0
      width: 15.0
```

### `absorber`

Drain contacts at the x-boundaries and wall treatment at the
y-boundaries. Drains absorb with a cos² profile over the outermost
`width_frac × Lx` region; the absorbed probability is recorded as T
(right drain) and R (left drain) by construction.

```yaml
absorber:
  width_frac: 0.05         # drain-region width, fraction of Lx
  strength: 0.08           # absorption rate per FFT step
  y_bc: "reflecting"       # "reflecting" | "absorbing" | "periodic"
  y_width_frac: 0.03       # y-wall region width, fraction of Ly
  wall_mass: 5.0           # mass M for M(y) sigma_z confinement
                           # 0 = auto (20 × max(E_F, V_0))
```

Reflecting walls use the mass-confinement trick `M(y) σ_z` — a local
band gap at the walls — which is the correct Dirac-fermion boundary
condition (a scalar wall would leak via Klein tunneling).

### `time`

Integration parameters.

```yaml
time:
  dt: 0.04                 # time step, sim units
  n_steps: 15000           # maximum number of steps
  save_every: 200          # snapshot cadence for animation frames
```

The built-in validator checks that `dt` satisfies the Nyquist
criterion `dt × max(|H|) < pi/2` for the 2nd-order split-operator
scheme and prints a warning if marginal.

### `detector`

Transport observables.

```yaml
detector:
  enabled: true            # record T(t), R(t), P_y(t), eta(t)
  auto_source: false       # auto-position source based on grid
  auto_stop: true          # stop when P_remaining < stop_threshold
  stop_threshold: 0.005    # P remaining below which to auto-stop
```

### `coupling`

Optional 4-spinor intervalley coupling `U_KK'(x, y) · I_2`. Disabled
by default; when enabled, the coupled propagator is used.

```yaml
coupling:
  enabled: false
  type: "barrier"          # "barrier" | "line" | "region"
  strength: 0.0            # coupling amplitude
  region_threshold: 0.5    # for type=region
  line_position: 0.0       # for type=line
  line_width: 2.0          # for type=line
```

At `strength = 0.0`, the coupled propagator is verified by the test
suite to reproduce two independent single-valley propagations to
single-precision.

---

## Common patterns

### Single-valley vs dual-valley simulation

```yaml
physics:
  valleys: "K"             # single valley, ~half the compute
  # valleys: "Kp"          # single valley K' (tilt sign reversed)
  # valleys: "both"        # both valleys, parallel, reports eta(t)
```

Use single-valley mode when you only care about one valley (e.g. when
tilt is zero, K and K' are equivalent) or when doing angle sweeps
without valley-dependent observables.

### Reflecting channel walls vs open-boundary

```yaml
# Closed channel (realistic device)
absorber:
  y_bc: "reflecting"
  wall_mass: 5.0

# Open-y geometry (for isolated beam paths, negative-refraction demos)
absorber:
  y_bc: "absorbing"
  y_width_frac: 0.05
```

For oblique injection with large `|theta|`, reflecting walls can cause
the beam to bounce and re-interact with the barrier. Use absorbing
y-walls (`"absorbing"`) to let the beam exit cleanly; any absorbed
probability is tracked as `P_y` in the budget.

### Sharp vs smoothed barrier interfaces

```yaml
potential:
  smoothing_width: 0.0     # sharp step, delta-like interface
  smoothing_width: 2.0     # sigmoid interface with 1/e width = 2 sim units
```

Sharp interfaces require more grid resolution (kx reach well above
1/(lattice spacing)) but match the analytical transfer-matrix picture
exactly. Smoothed interfaces are physically realistic for
lithographically gated devices.

### Source-drain bias for I-V characteristics

```python
# In a configuration built programmatically:
from dirac_wavepacket.config import BiasConfig
cfg.bias = BiasConfig(
    enabled=True,
    V_sd=0.02,               # bias voltage
    profile="linear",        # "linear" or "sigmoid"
    x_drop_start=-60.0,
    x_drop_end=+60.0,
)
```

Under bias, `d<kx>/dt = V_sd / L_drop` — verified by the test suite to
better than 0.05 %.

---

## Running sweeps

For parameter scans, use the installed console scripts instead of
launching simulations one at a time.

### V₀ sweep (bipolar regime scan)

```bash
dwp-sweep --config myconfig.yaml \
              --v0-min 0.0 --v0-max 2.0 --v0-step 0.02 \
              --workers 8 --no-anim --save-wf
```

Produces a JSON of per-V₀ results and a per-task state directory.
Resumable: if the run is interrupted, re-invoke with `--resume` and
only missing points are computed.

### Source-drain bias sweep (I-V characteristic)

```bash
dwp-sweep-vsd --config myconfig.yaml --v0 0.8 \
                  --vsd-min -0.02 --vsd-max 0.02 --vsd-step 0.004 \
                  --jobs 8
```

### Dual-valley geometry sweep

```bash
dwp-sweep --config myconfig.yaml --sweep geometry \
              --w-min 0.0 --w-max 0.4 --w-step 0.02 \
              --v0 1.4 --workers 8 --no-anim
```

---

## Production-run checklist

Before committing a long A100-class run, check the following:

1. **Grid resolution.** `dirac_wavepacket` emits automatic warnings when
   `kx0` approaches `kx_max` or when points-per-wavelength drops below
   4. Aim for ≥ 5 points-per-wavelength in both x and y.

2. **Time step.** The validator prints the estimated splitting error
   relative to `1 / max(|H|)`. For < 1% accuracy, stay at
   `dt < 0.5 / max(E_F, V_0)`.

3. **Channel extents.** The source must be several σ away from the
   drain, and the drain-absorber region must be wide enough (≥ 5%) to
   absorb fully. For oblique beams, the y-wall must be far enough that
   the beam reaches the x-drain before the y-wall (or use absorbing
   y-walls).

4. **Auto-stop threshold.** `0.005` is a good default. If you need
   tight probability-budget closure (for paper figures), drop to
   `0.001` and accept ~2× longer runtime.

5. **Snapshot cadence.** `wf_save_every = n_steps // 20` is a good
   default (~20 frames). More frames make animations smoother; fewer
   save disk.

6. **FFT backend.** `pyFFTW` is 2–5× faster than NumPy's FFT for the
   array shapes we use. Install via `pip install -e ".[fft]"`.
   Set `--fft-backend pyfftw --fft-threads 8` on the sweep drivers.

7. **Parallel workers.** For sweeps, one single-threaded worker per
   core is almost always optimal. `dwp-sweep --workers N` with
   `N = ncpus` and `FFT threads = 1` beats `N = 1, FFT threads = ncpus`
   on every benchmark so far.

---

## Troubleshooting

**"kx0 > 70% of kx_max" warning.** The grid is too coarse for the
wavepacket's central k. Either reduce `k0`, or increase `Nx`, or shrink
`Lx`.

**"Only N points/wavelength in x" warning.** Same root cause. The
numerical dispersion of the split-operator scheme is accurate when N ≥
4; fewer than 4 produces visible distortion. Add grid resolution.

**Budget Σ < 0.99.** The probability budget should close within
single-precision round-off. If it falls short, either the packet
hasn't reached the drain yet (extend `n_steps`), or the wall mass is
too low and leakage through the y-wall is occurring (increase
`wall_mass`).

**Valley polarisation η drifts over time.** Expected — for dual-valley
runs, η(t) is time-dependent while each valley drains asymmetrically.
The final steady-state η is what the paper measures.

**Sweep worker crashes after many tasks.** Add
`--maxtasksperchild 1` so each worker is recycled after its task.
Python's multiprocessing leaks a small amount of memory per task
through NumPy, and this avoids OOM on large sweeps.
