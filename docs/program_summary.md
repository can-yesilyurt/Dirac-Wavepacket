# Program Summary

This document follows the *Computer Physics Communications* template for a
Computer Program in Physics (CPiP) submission. It is the authoritative
machine-readable summary of the program; the accompanying journal paper
expands on each section.

---

## Header

| Field | Value |
|---|---|
| **Program title** | `dirac_wavepacket` |
| **CPC Library link to program files** | *(to be supplied by the Technical Editor)* |
| **Developer's repository link** | https://github.com/can-yesilyurt/Dirac-Wavepacket |
| **Code Ocean capsule** | *(optional; to be supplied if preparing a reproducibility capsule)* |
| **Licensing provisions** | MIT License |
| **Programming language** | Python 3.11+ |
| **Nature of problem** | *(below)* |
| **Solution method** | *(below)* |
| **Additional comments, Restrictions, Unusual features** | *(below)* |
| **External routines / libraries** | NumPy, SciPy, Matplotlib, PyYAML, Pillow; optional: pyFFTW |
| **Operating system** | Linux, macOS, Windows (any platform with CPython ≥ 3.11) |
| **RAM** | ~1 GB for a 512² grid in single precision; scales linearly with `Nx · Ny` |
| **Number of processors used** | 1 for a single simulation; up to hundreds via the parallel sweep driver |
| **Classification** | 7.3 Electronic Structure (primary); 7.9 Transport Properties *(final code assigned by Technical Editor from the CPC Program Library Subject Index)* |
| **Keywords** | Dirac equation, Weyl semimetal, tilted cone, valley transport, wave-packet, split-operator, Klein tunneling, intervalley coupling |

---

## Nature of problem

Electronic transport in two-dimensional Dirac and Weyl semimetals with
tilted, anisotropic cones underlies a growing class of proposals for
valley filters, valley-Hall elements, electron optics, and
valley-based quantum gates. Quantitative design of such devices
requires solving the time-dependent 2D Dirac equation for a
two-component spinor wave packet in a spatially inhomogeneous
electrostatic landscape with realistic device boundaries: reflecting
channel walls (non-trivial for Dirac fermions because of Klein
tunneling), absorbing source/drain contacts, and electrostatically
defined barriers of arbitrary polygonal or parallelogram geometry.
Existing open-source quantum-transport codes target either
non-relativistic Schrödinger systems, tight-binding scattering-matrix
problems, or atomic-physics applications of the Dirac equation; none
of them combine continuum tilted-anisotropic Dirac dynamics with
arbitrary-shape barriers, mass-wall confinement, drains-as-detectors
transport bookkeeping, and optional intervalley coupling in a single
installable Python package.

---

## Solution method

`dirac_wavepacket` evolves a two-component spinor on a uniform Cartesian grid
by a symmetric second-order Strang-split Fourier propagator. The
Hamiltonian is decomposed into a real-space part (scalar potential
plus mass confinement, diagonal in the spinor basis) and a
momentum-space part (tilted anisotropic Dirac kinetic term, evaluated
analytically as a 2×2 unitary at each k-point). Time evolution
alternates real-space half-step phase multiplication with a
momentum-space full-step application of the analytical unitary,
connected by FFTs. Absorbing cos²-profile drain contacts at the x
boundaries absorb outgoing probability and simultaneously record
cumulative transmission and reflection — the contacts are the
detectors, so the probability budget T + R + P_y + P_remaining = 1
holds exactly by construction. Reflecting channel walls are
implemented via a transverse mass profile M(y) σ_z that opens a local
gap at the walls, the correct Dirac-fermion confinement. An optional
4-component propagator adds spatially local, pseudo-spin-preserving
intervalley coupling U_KK'(x, y) · I₂ via an analytical 2×2 matrix
exponential in the (K, K′) subspace; at zero coupling it reproduces
two independent single-valley propagations bit-for-bit.

---

## Additional comments, Restrictions, Unusual features

The code is written in pure Python with NumPy, targets single-node
CPU execution, and is installable via `pip install .`. FFT
performance is accelerated automatically when the optional `pyFFTW`
extra is installed. The split-operator scheme is unconditionally
unitary in the closed (drainless) limit and has O(Δt³) local error;
a built-in validator reports the effective splitting error relative
to the maximum Hamiltonian eigenvalue on the grid. A rigid
source–drain bias module (linear or sigmoid profile) adds an
electric field across the channel and has been validated against the
analytical Bloch-acceleration prediction d⟨kx⟩/dt = V_sd / L_drop to
better than 0.05 %. A checkpointed parallel driver
(`dwp-sweep`) resumes interrupted sweeps from per-task JSON
state.

Restrictions. The numerical grid is uniform Cartesian, which
limits efficiency for devices with strongly multi-scale features;
users should prefer the analytically tractable rectangular limit
when it applies. The intervalley coupling model is scalar and local,
so processes mediated by atomic-scale bond physics (phonon-assisted
scattering, long-range SOC-mediated coupling) fall outside the
continuum scope. Intermediate grid resolutions between the single-
wavelength minimum and full device resolution can produce visible
anisotropy of the dispersion on the grid; the code emits automatic
warnings when the Nyquist / points-per-wavelength limits are
approached.

Unusual features. (a) Arbitrary polygonal barriers with symbolic
`Lx`, `Ly` vertex expressions, supporting non-convex shapes;
(b) chirality-aware propagator that automatically applies the
σ_y → −σ_y swap at K′, giving the physically correct
pseudo-spin–momentum locking at each valley;
(c) resumable parallel sweeps over V₀, geometry, or V_sd via
per-task checkpoint files.

---

## References

1. M. D. Feit, J. A. Fleck Jr., A. Steiger.
   *Solution of the Schrödinger equation by a spectral method.*
   Journal of Computational Physics **47**, 412–433 (1982).

2. G. Strang.
   *On the construction and comparison of difference schemes.*
   SIAM Journal on Numerical Analysis **5**, 506–517 (1968).

3. G. R. Mocken, C. H. Keitel.
   *FFT-split-operator code for solving the Dirac equation in 2+1
   dimensions.* Computer Physics Communications **178**, 868–882 (2008).

4. A. Chaves, G. A. Farias, F. M. Peeters, R. Ferreira.
   *The split-operator technique for the study of spinorial wavepacket
   dynamics.* Communications in Computational Physics **17**, 850–866
   (2015).

5. M. V. Berry, R. J. Mondragon.
   *Neutrino billiards: time-reversal symmetry-breaking without
   magnetic fields.* Proceedings of the Royal Society A **412**, 53–74
   (1987).

6. M. I. Katsnelson, K. S. Novoselov, A. K. Geim.
   *Chiral tunnelling and the Klein paradox in graphene.*
   Nature Physics **2**, 620–625 (2006).

7. V. H. Nguyen, J.-C. Charlier.
   *Klein tunneling and electron optics in Dirac-Weyl fermion systems
   with tilted energy dispersion.* Physical Review B **97**, 235113
   (2018).
