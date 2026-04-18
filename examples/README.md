# Examples

Each script below is self-contained and uses only public `dirac_wavepacket`
APIs. They reproduce the figures in the companion
*Computer Physics Communications* paper.

After `pip install -e .` from the repository root, run any example
from anywhere:

```bash
python examples/01_angled_barrier_valley_filter.py --quick
```

## Scripts

### `01_angled_barrier_valley_filter.py`

All-electrostatic valley filter in a tilted-Dirac channel. An
electrostatic barrier rotated by `α = 20°` selectively transmits one
valley and reflects the other, producing a net valley polarization
`η = (T_K − T_K')/(T_K + T_K')` without magnetic fields or strain.

### `02_klein_angular_dependence.py`

Angular dependence of Klein tunneling through a rectangular barrier
in the bipolar regime `V_0 > E_F`, isotropic untilted cone, single
valley. Scans a list of injection angles (default: −15°, −5°, +5°, +15°)
and reports `T(θ)`, `R(θ)`, and the mirror-symmetry residual
`T(+θ) − T(−θ)` as a numerical self-consistency check. Writes both a
snapshot grid (one panel per angle) and a `T(θ)`, `R(θ)` summary plot.

## Configurations

`configs/` contains YAML files that can be loaded with
`dirac_wavepacket.config.load_config()` or passed on the CLI:

```bash
dwp examples/configs/reflecting_walls_w35_W50.yaml --output results/test
```

## Running time

Every example has a `--quick` flag (laptop-scale, ~1–3 min, smoke-test
resolution) and a default mode (~10–30 min on modern hardware) for the
publication-quality figure.
