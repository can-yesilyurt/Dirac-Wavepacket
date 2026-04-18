# Changelog

All notable changes to `dirac_wavepacket` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-18 (Computer Physics Communications submission)

First public release. Captures the development previously tagged as
internal v2.x.

### Added
- 2D tilted Dirac split-operator FFT propagator (`SplitOperatorPropagator`)
  with chirality-aware eigenspinor and k-space unitary (σ_y → −σ_y swap
  at K′).
- 4-component coupled propagator (`CoupledSplitOperatorPropagator`) for
  spatially local, pseudo-spin-preserving intervalley coupling
  U_KK'(x, y) · I_2; reduces to two independent single-valley propagations
  bit-for-bit at zero coupling.
- Arbitrary barrier geometries: `rectangular`, `pn_junction`,
  `double_barrier`, `shaped` (parallelograms), `polygon` (symbolic
  `Lx`, `Ly` expressions), and `multi` (sum of sub-barriers).
- Reflecting channel walls via local mass confinement `M(y) σ_z`
  (gap opening — the correct Dirac-fermion boundary condition).
- Absorbing drain contacts as detectors: `T + R + P_y + P_rem = 1` by
  construction.
- Rigid source–drain bias `V_sd` (linear or sigmoid profile), validated
  against the Bloch-acceleration prediction to 0.02 %.
- Valley-resolved transport observables, overlap-based gate-phase
  extraction, group-delay diagnostics.
- Checkpointed parallel sweep drivers (`dwp-sweep`,
  `dwp-sweep-v0`, `dwp-sweep-vsd`) with resumable state.
- pyFFTW auto-detection with graceful fall-back to NumPy FFT.

### Documentation
- Worked examples reproducing the figures of the accompanying
  *Computer Physics Communications* paper.
- User guide (`docs/user_guide.md`) and theory / conventions
  (`docs/theory.md`).
