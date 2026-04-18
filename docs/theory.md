# Theory and Numerical Method

This document derives the physics model, the numerical scheme, and the
conventions used throughout `dirac_wavepacket`. For hands-on usage, see the
[user guide](user_guide.md). For worked physics examples, see the
[examples directory](../examples).

## Notation

Throughout: `τ = ±1` is the valley index (+1 for K, −1 for K'),
`σ_x`, `σ_y`, `σ_z` are Pauli matrices in sublattice-pseudospin space,
`I` is the 2×2 identity, and natural units `ℏ = v_F = 1` are used.
When physical units are required, recovery is straightforward:
energies multiply by `ℏ v_F`, and lengths divide by the same.

---

## 1. Hamiltonian

The low-energy theory of a type-I tilted, anisotropic 2D Dirac /
Weyl semimetal near a single valley is governed by

```
H_τ =  v_x σ_x k_x  +  v_y σ_y k_y                          (kinetic)
     +  τ (w_x k_x + w_y k_y) I                             (tilt)
     +  V(x, y) I                                           (scalar potential)
     +  M(y) σ_z                                            (mass confinement)
```

where `(v_x, v_y)` are anisotropic Fermi velocities, `(w_x, w_y)` is
the tilt vector (at valley K; valley K' uses `-w`), `V(x, y)` is an
external electrostatic potential, and `M(y) σ_z` is a transverse mass
profile used only at reflecting y-walls.

The identity-term structure of the tilt is not a convention choice:
it is enforced by the crystal symmetry of systems like 8-Pmmn
borophene, α-(BEDT-TTF)₂I₃, and WTe₂. Because the tilt acts on the
identity, it rigidly displaces the Fermi contour without modifying the
pseudospin texture — a fact exploited in the valley-filter mechanism
demonstrated in [Example 1](../examples/01_angled_barrier_valley_filter.py).

### 1.1 Energy dispersion

Diagonalising the kinetic + tilt terms at fixed `k`:

```
E_±(k) = τ (w_x k_x + w_y k_y)  ±  √[(v_x k_x)² + (v_y k_y)²]
```

so the tilt gives an additive, valley-dependent energy shift that
displaces the two Dirac cones oppositely in k-space. The type-I (closed
Fermi surface) regime requires `√(w_x²/v_x² + w_y²/v_y²) < 1`.

### 1.2 Pseudospin texture

At fixed band index λ = ±1 and valley τ, the eigenspinor is

```
|ψ⟩_τ = (1/√2) [ 1, exp(i τ φ_k) ]ᵀ
```

with `φ_k = atan2(v_y k_y, v_x k_x)`. The chirality τ enters only as
the sign of the pseudospin phase — equivalently, `H(K')` is related to
`H(K)` by `σ_y → -σ_y` (at zero tilt) which is exactly the swap the
numerical propagator implements at K'.

---

## 2. Split-operator FFT propagator

The formal evolution from `t` to `t + dt` is `ψ(t + dt) = U(dt) ψ(t)`
with `U(dt) = exp(-i H dt)`. For Strang-splitting into real-space and
momentum-space parts `H = H_R + H_K`, we have

```
U(dt) = exp(-i H_R dt/2) · exp(-i H_K dt) · exp(-i H_R dt/2)  +  O(dt³)
```

The symmetric (second-order) splitting is unconditionally unitary by
construction in both limits; the time-step error appears only in the
commutator `[H_R, H_K]` and vanishes as `dt³` locally (`dt²` globally
over fixed total time).

### 2.1 Real-space step

`H_R = V(x, y) I + M(y) σ_z` is diagonal in the position basis and,
at each grid point, diagonal in the pseudospin basis after a σ_z
diagonalisation:

```
exp(-i H_R dt/2) = exp(-i V dt/2) · diag(exp(-i M dt/2), exp(+i M dt/2))
```

Apply point-wise to `ψ(x, y)`. Cost: O(Nx Ny) per half-step.

### 2.2 Momentum-space step

In the momentum basis, `H_K = v_x σ_x k_x + v_y σ_y k_y + τ(w_x k_x +
w_y k_y) I`. At each `k`, this is a 2×2 matrix whose exponential is
computed analytically. Writing

```
d_k = τ (w_x k_x + w_y k_y)                    (identity piece)
ε_k = √[(v_x k_x)² + (v_y k_y)²]                (kinetic magnitude)
n̂(k) = (v_x k_x, v_y k_y, 0) / ε_k              (unit pseudospin axis)
```

the propagator is

```
exp(-i H_K dt) = exp(-i d_k dt) [ cos(ε_k dt) I  -  i sin(ε_k dt) n̂·σ ]
```

This is implemented exactly as a pointwise multiplication in momentum
space (no eigenvector storage needed). Cost: two FFTs (forward and
inverse) plus a pointwise update, O(Nx Ny log(Nx Ny)) total.

### 2.3 Valley chirality

For valley K' the propagator uses `(w_x, w_y) → (-w_x, -w_y)` and
swaps `σ_y → -σ_y` in `H_K`. Operationally this is a sign flip of the
imaginary off-diagonal term in the 2×2 k-space matrix; no other
branch is taken. This guarantees that valley K and K' are evolved by
the correct chirality without any re-indexing of the wavefunction.

### 2.4 Split-operator error monitor

A built-in validator reports the worst-case eigenvalue `E_max` on the
grid and compares `dt E_max` to the stability bound. For `dt E_max ≲
0.5` the local O(dt³) splitting error is < 1 %.

---

## 3. Boundary conditions

### 3.1 Absorbing drains

The two narrow regions at `x < -Lx/2 + W_d` and `x > Lx/2 - W_d` carry
a cosine-squared absorption mask `A(x)`, applied between steps:

```
ψ(x, y) ← A(x) ψ(x, y),   A(x) = [1 - κ · cos²(π(x - x_d) / (2 W_d))] per step
```

The absorbed probability is *not* discarded: it is accumulated into
cumulative counters `T(t)` (right drain) and `R(t)` (left drain).
Therefore the total budget

```
T(t) + R(t) + P_y(t) + P_remaining(t) = 1
```

holds by construction at every step, where `P_y` is the y-wall
absorption (only non-zero when `y_bc: "absorbing"`) and
`P_remaining = ∫|ψ(x, y, t)|² dx dy` over the non-drain region.

### 3.2 Mass-wall confinement

For `y_bc: "reflecting"`, the channel walls use the local mass profile

```
M(y) = m_0 · h(y)
```

where `h(y)` is a cosine-squared ramp confined to the outermost `W_y`
fraction of the channel. This opens a local gap `2 m_0` at the walls.
Dirac fermions with `E < m_0` cannot propagate through the wall —
Klein tunneling is eliminated because the massive Dirac equation at
these y-values has no propagating modes. A scalar potential wall would
not achieve this: Klein tunneling guarantees finite transmission
through any scalar barrier at normal incidence, no matter how high.

The auto-value of `m_0` is `20 × max(E_F, V_0)` — large enough that
even the fastest-group-velocity component of the wavepacket sees an
effectively infinite gap at the walls. Users can override if the
physics requires a softer wall.

---

## 4. Optional 4-spinor coupled propagator

For simulations with spatially local, pseudospin-preserving
intervalley coupling `U_KK'(x, y)`, the Hamiltonian becomes

```
H_coupled = diag(H_K, H_K')  +  U_KK'(x, y) · τ_x ⊗ I_2
```

where τ_x is the Pauli matrix in valley space. The resulting
4-component spinor evolves under

```
U(dt) = exp(-i H_coupled dt)
```

In real space, at each grid point, `H_coupled` is a 4×4 matrix whose
upper-left 2×2 block is `H_R^K`, lower-right is `H_R^K'`, and the
off-diagonal 2×2 blocks are `U_KK'(x, y) · I_2`. Its exponential is
computed analytically in the (K, K') pair subspace (by rotating to the
± combinations), then applied to the two independent pseudospin
components.

In momentum space, `H_K^{kin}` and `H_K'^{kin}` act on their respective
valleys independently (the intervalley coupling is local in real space
so it has no k-space piece), and each valley is evolved by its own 2×2
k-space unitary.

At `U_KK' = 0`, the propagator must reproduce two independent
single-valley propagations — this equivalence is verified by the
`test_coupled_zero_coupling` test in the pytest suite.

---

## 5. Wavepacket initialisation

The injected packet at time `t = 0` is a Gaussian envelope multiplied
by a positive-energy eigenspinor of `H_K` or `H_K'`:

```
ψ(x, y, 0) = exp( -(x-x_0)² / (4 σ_x²) - (y-y_0)² / (4 σ_y²) )
             · exp( i (k_{x0} x + k_{y0} y) )
             · |ψ⟩_τ(k_0)
```

with `k_{x0} = k_0 cos(θ)`, `k_{y0} = k_0 sin(θ)`. The eigenspinor
carries the valley's pseudospin-momentum locking; at τ = +1 (K) it is
`(1, exp(+iφ))/√2` and at τ = −1 (K') it is `(1, exp(−iφ))/√2`.

Both valleys are initialised with the **same** (k_{x0}, k_{y0}) — the
dynamical asymmetry between them arises entirely from the sign
difference in the tilt and chirality during propagation.

The envelope is normalised so that `∫|ψ|² dx dy = 1` at `t = 0`.

---

## 6. Source–drain bias

For I–V characteristics, a rigid bias `V_sd` is added as a potential
drop across the channel:

```
V_bias(x) = V_sd · f(x),    f(x) = 0  for x < x_drop_start,
                                    (x - x_drop_start) / (x_drop_end - x_drop_start)
                                    for x_drop_start < x < x_drop_end,
                                    1  for x > x_drop_end.
```

(Or a sigmoid profile if `profile: "sigmoid"`.) The wavepacket's
central momentum evolves according to the analytical Bloch-acceleration
prediction:

```
d⟨k_x⟩/dt = -∂V_bias/∂x = V_sd / L_drop
```

where `L_drop = x_drop_end - x_drop_start`. This prediction is
verified by the package test suite to better than 0.05 %.

---

## 7. Units

In-code natural units: `ℏ = v_F = 1`. Positions are dimensionless;
energies have dimensions of inverse length. The default physical
mapping for display (e.g. axis labels on rendered figures) is

```
1 simulation unit = 6.582 nm,   v_F = 10⁶ m/s,   hbar v_F = 0.6582 eV·nm,
```

which is the canonical graphene / Dirac-semimetal convention. All
physical-unit output is purely cosmetic post-processing applied in
the visualiser; the dynamics do not depend on it.

To rescale to a different material, multiply the `v_F` the code uses
by the material's `v_F / 10⁶ m/s`; all output simulation units remain
consistent if the same scaling is used everywhere.

---

## 8. References

1. Feit, Fleck, Steiger, *J. Comp. Phys.* **47**, 412 (1982) —
   split-operator Fourier scheme.
2. Strang, *SIAM J. Numer. Anal.* **5**, 506 (1968) — symmetric
   splitting and its error order.
3. Mocken & Keitel, *Comp. Phys. Commun.* **178**, 868 (2008) —
   2+1D Dirac split-operator in atomic physics.
4. Chaves, Farias, Peeters, Ferreira, *Commun. Comp. Phys.* **17**,
   850 (2015) — spinorial wavepacket dynamics, split-operator review.
5. Berry & Mondragón, *Proc. R. Soc. A* **412**, 53 (1987) — mass
   confinement ("neutrino billiards") and reflecting boundaries for
   Dirac fermions.
6. Katsnelson, Novoselov, Geim, *Nature Phys.* **2**, 620 (2006) —
   Klein paradox in graphene, plane-wave transfer-matrix formula.
7. Nguyen & Charlier, *Phys. Rev. B* **97**, 235113 (2018) — Klein
   tunneling and electron optics in tilted Dirac / Weyl systems.
