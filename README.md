# dirac-wavepacket

**Time-dependent wave-packet transport in 2D tilted Dirac/Weyl systems.**

[![PyPI version](https://img.shields.io/pypi/v/dirac-wavepacket.svg)](https://pypi.org/project/dirac-wavepacket/)
[![Python versions](https://img.shields.io/pypi/pyversions/dirac-wavepacket.svg)](https://pypi.org/project/dirac-wavepacket/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`dirac_wavepacket` solves the time-dependent Dirac equation for a two-component
spinor on a 2D grid using a symmetric split-operator Fourier scheme. It
is designed for continuum-limit transport calculations in Dirac and
Weyl materials with anisotropic Fermi velocities and tilted cones, and
supports arbitrary electrostatic barrier shapes, reflecting mass-wall
confinement, absorbing drain contacts, source–drain bias, and
pseudo-spin-preserving intervalley coupling between K and K'.

## Physical model

For valley index `τ = ±1` (K and K′):

```
H_τ = ℏ (v_x σ_x k_x + v_y σ_y k_y)
      + τ ℏ (w_x k_x + w_y k_y) I
      + V(x, y) I  +  M(y) σ_z
```

The anisotropic Dirac term carries the pseudo-spin texture; the tilt
term is proportional to the identity and rigidly displaces the Fermi
contour along `w`. `V(x, y)` is an arbitrary electrostatic barrier
(rectangular / p-n / shaped / polygon / multi-barrier), and `M(y) σ_z`
implements reflecting channel walls via a local gap — the correct
confinement for Dirac fermions, which cannot be reflected by a scalar
wall because of Klein tunneling.

A 4-component coupled propagator is available for simulations with
spatially local, pseudo-spin-preserving intervalley coupling
`U_KK'(x, y) · I_2`. At zero coupling it reproduces two independent
single-valley propagations bit-for-bit.

## Installation

```bash
pip install dirac-wavepacket
```

Optional extras:

```bash
pip install "dirac-wavepacket[fft]"    # pyFFTW acceleration (2–5× speedup)
pip install "dirac-wavepacket[dev]"    # pytest and test utilities
pip install "dirac-wavepacket[all]"    # both of the above
```

Requires Python ≥ 3.11. Platform-independent (Linux, macOS, Windows).

### For contributors (editable install from source)

```bash
git clone https://github.com/can-yesilyurt/Dirac-Wavepacket
cd Dirac-Wavepacket
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## Quickstart

```python
from dirac_wavepacket import SimConfig, load_config, run_simulation

cfg = load_config("examples/configs/reflecting_walls_w35_W50.yaml")
cfg.output_dir = "results/quickstart"
result = run_simulation(cfg, make_animation=False, verbose=True)
print(f"T = {result['T']:.4f}, R = {result['R']:.4f}")
```

Or from the command line:

```bash
dwp examples/configs/reflecting_walls_w35_W50.yaml \
    --output results/quickstart
```

## Worked examples

Two self-contained scripts in `examples/` reproduce the figures in the
companion paper:

| Script                                      | What it shows                                           |
|---------------------------------------------|---------------------------------------------------------|
| `01_angled_barrier_valley_filter.py`        | All-electrostatic valley filter via barrier rotation    |
| `02_klein_angular_dependence.py`            | Angular dependence of Klein tunneling, T(θ) scan        |

Each script has a `--quick` mode for laptop-scale smoke testing
(~1–3 min) and a default mode for the publication-quality figure
(~10–30 min on a modern multi-core CPU or a single GPU).

## Production drivers

For parameter sweeps over many `V_0`, geometry, or source–drain
voltage values, use the checkpointed parallel drivers installed as
console scripts:

```bash
dwp-sweep      --config examples/configs/reflecting_walls_w35_W50.yaml \
                   --v0-min 0 --v0-max 1.6 --v0-step 0.02 --workers 8 --no-anim
dwp-sweep-vsd  --config ... --v0 0.228 --vsd-min -0.02 --vsd-max 0.02 \
                   --vsd-step 0.004 --jobs 8
```

Sweeps are resumable (per-task JSON checkpoints) and graceful against
worker crashes — see `docs/user_guide.md`.

## Repository layout

```
dirac_wavepacket/     Python package (propagator, potentials, etc.)
  cli/                Console-script entry points (sweep drivers)
examples/             Self-contained worked examples
  configs/            YAML configurations
tests/                pytest suite
docs/                 User guide and theory notes
```

## Citation

If `dirac-wavepacket` contributes to a publication, please cite the
Zenodo archive. A software paper describing the package is under review
at *SoftwareX*; citation details will be updated here once the paper is
accepted. In the meantime, please cite as:

```bibtex
@software{yesilyurt_dirac_wavepacket_2026,
  author    = {Yesilyurt, Can},
  title     = {Dirac-Wavepacket: time-dependent wave-packet transport
               in two-dimensional tilted Dirac and Weyl systems},
  year      = {2026},
  version   = {1.0.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/can-yesilyurt/Dirac-Wavepacket}
}
```

*(Replace `XXXXXXX` with the actual Zenodo DOI minted on the v1.0.1
GitHub release.)*

## License

MIT — see [`LICENSE`](LICENSE).
