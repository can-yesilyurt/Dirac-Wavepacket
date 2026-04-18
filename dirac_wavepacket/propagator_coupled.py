"""
Coupled propagator: split-operator FFT time evolution for the 4-component
two-valley tilted Dirac equation.

Spinor layout:  psi[0] = K↑,  psi[1] = K↓,  psi[2] = K'↑,  psi[3] = K'↓

Full Hamiltonian:

    H = | H_K(+w, χ=+1)   U_KK'           |
        | U_KK'†           H_K'(-w, χ=-1)  |

where each diagonal block is the single-valley tilted Dirac Hamiltonian
with the appropriate chirality:

    H_K  = ℏ(+w·k) I + ℏv_F(σ_x k_x + σ_y k_y) + V I + M σ_z       [χ=+1]
    H_K' = ℏ(-w·k) I + ℏv_F(σ_x k_x − σ_y k_y) + V I + M σ_z       [χ=-1]

The opposite chirality at K' (σ_y → -σ_y) reverses the pseudospin-momentum
locking. In k-space this is implemented by swapping ek ↔ ek* in the K'
block's 2×2 unitary, while leaving the dispersion |b| = v_F|k| unchanged.

and U_KK'(x,y) is the intervalley coupling matrix.

Split-operator decomposition:

    H_k = | H_k,K    0       |     ← block-diagonal in k-space
          | 0         H_k,K'  |

    H_r = | V I+M σ_z    U_KK'     |     ← local in real space
          | U_KK'†       V I+M σ_z |

The k-space step is identical to two independent single-valley steps with
opposite tilt signs. The real-space step is where coupling enters.

For scalar coupling U = u(x,y)·I₂, the 4×4 real-space block decomposes
into two independent 2×2 problems in the (K_s, K'_s) subspace for each
pseudospin component s ∈ {↑,↓}:

    (K↑, K'↑):  [[V+M, u], [u*, V+M]]
    (K↓, K'↓):  [[V−M, u], [u*, V−M]]

Each admits an analytical 2×2 matrix exponential:

    exp(-i [[a,u],[u*,a]] t) = e^{-iat} · [[cos|u|t, -i(u/|u|)sin|u|t],
                                             [-i(u*/|u|)sin|u|t, cos|u|t]]

At zero coupling this reduces exactly to the diagonal phase factors of
the single-valley propagator.

────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .grid import Grid
from .precision import COMPLEX_DTYPE, REAL_DTYPE, MINUS_I

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _pyfftw_fft

    _HAVE_PYFFTW = True
    try:
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60)
    except Exception:
        pass
except Exception:
    pyfftw = None
    _pyfftw_fft = None
    _HAVE_PYFFTW = False


def _resolve_fft_backend(requested="auto", threads=None):
    """Resolve the FFT backend (identical logic to propagator.py)."""
    req = (requested or "auto").strip().lower()
    env_backend = os.environ.get("DIRAC2D_FFT_BACKEND", "").strip().lower()
    if req == "auto" and env_backend:
        req = env_backend

    if req not in {"auto", "numpy", "pyfftw"}:
        raise ValueError(f"Unknown fft_backend='{requested}'.")

    if threads is None:
        env_threads = os.environ.get("DIRAC2D_FFT_THREADS", "1").strip()
        try:
            threads = int(env_threads)
        except ValueError:
            threads = 1
    threads = max(1, int(threads))

    if req == "pyfftw":
        if not _HAVE_PYFFTW:
            raise RuntimeError("pyFFTW requested but not installed.")
        backend = "pyfftw"
    elif req == "numpy":
        backend = "numpy"
    else:
        backend = "pyfftw" if _HAVE_PYFFTW else "numpy"

    if backend == "numpy":
        return backend, 1, np.fft.fft2, np.fft.ifft2

    def fft2(arr):
        return _pyfftw_fft.fft2(
            arr, overwrite_input=False, planner_effort="FFTW_MEASURE",
            threads=threads, auto_align_input=True, auto_contiguous=True,
        )

    def ifft2(arr):
        return _pyfftw_fft.ifft2(
            arr, overwrite_input=False, planner_effort="FFTW_MEASURE",
            threads=threads, auto_align_input=True, auto_contiguous=True,
        )

    return backend, threads, fft2, ifft2


class CoupledSplitOperatorPropagator:
    """
    4-component split-operator propagator for coupled K–K' valleys.

    At zero coupling, this reproduces two independent SplitOperatorPropagator
    runs bit-for-bit (same operator arrays, same arithmetic sequence).

    Parameters
    ----------
    grid : Grid
    V    : real ndarray (Ny, Nx) — electrostatic potential
    vf   : Fermi velocity
    wx, wy : tilt velocity components (K valley; K' uses -wx, -wy)
    dt   : time step
    M    : real ndarray (Ny, Nx) or None — mass confinement
    coupling_u : complex ndarray (Ny, Nx) or None — scalar intervalley coupling
        The coupling matrix is U_KK' = coupling_u(x,y) · I₂.
        Pass None for the decoupled (independent valley) limit.
    fft_backend : {"auto", "numpy", "pyfftw"}
    fft_threads : int or None
    """

    def __init__(
        self,
        grid: Grid,
        V: np.ndarray,
        vf: float,
        wx: float,
        wy: float,
        dt: float,
        M: Optional[np.ndarray] = None,
        coupling_u: Optional[np.ndarray] = None,
        fft_backend: str = "auto",
        fft_threads: Optional[int] = None,
    ):
        self.grid = grid
        self.dt = dt

        V = np.asarray(V, dtype=REAL_DTYPE)
        M = None if M is None else np.asarray(M, dtype=REAL_DTYPE)

        self.fft_backend, self.fft_threads, self._fft2, self._ifft2 = \
            _resolve_fft_backend(requested=fft_backend, threads=fft_threads)

        # ══════════════════════════════════════════════════════════════════
        #  Real-space half-step operators
        # ══════════════════════════════════════════════════════════════════

        self._has_mass = M is not None

        if self._has_mass:
            self._phase_r_half_up = np.exp(
                MINUS_I * (V + M) * REAL_DTYPE(dt / 2.0)
            ).astype(COMPLEX_DTYPE, copy=False)
            self._phase_r_half_down = np.exp(
                MINUS_I * (V - M) * REAL_DTYPE(dt / 2.0)
            ).astype(COMPLEX_DTYPE, copy=False)
        else:
            self._phase_r_half = np.exp(
                MINUS_I * V * REAL_DTYPE(dt / 2.0)
            ).astype(COMPLEX_DTYPE, copy=False)

        # ── Coupling operators ────────────────────────────────────────────
        self._has_coupling = coupling_u is not None
        if self._has_coupling:
            u = np.asarray(coupling_u, dtype=COMPLEX_DTYPE)
            u_mag = np.abs(u)
            # Avoid division by zero at grid points with no coupling
            u_mag_safe = np.where(u_mag == 0.0, REAL_DTYPE(1.0), u_mag)
            u_hat = u / u_mag_safe       # unit phase factor: u/|u|

            half_dt = REAL_DTYPE(dt / 2.0)
            self._cos_u_half = np.asarray(
                np.cos(u_mag * half_dt), dtype=REAL_DTYPE
            )
            # -i · (u/|u|) · sin(|u|·dt/2)  for the (K→K') off-diagonal
            self._mix_KKp = (MINUS_I * u_hat * np.sin(u_mag * half_dt)).astype(
                COMPLEX_DTYPE, copy=False
            )
            # -i · (u*/|u|) · sin(|u|·dt/2)  for the (K'→K) off-diagonal
            self._mix_KpK = (MINUS_I * np.conj(u_hat) * np.sin(u_mag * half_dt)).astype(
                COMPLEX_DTYPE, copy=False
            )

            # Scratch for the coupling rotation (avoid temporaries)
            self._scratch_0 = np.empty(grid.shape, dtype=COMPLEX_DTYPE)
            self._scratch_1 = np.empty(grid.shape, dtype=COMPLEX_DTYPE)

        # ══════════════════════════════════════════════════════════════════
        #  k-space full-step operators (block-diagonal: K and K' separate)
        # ══════════════════════════════════════════════════════════════════

        KX, KY, K = grid.KX, grid.KY, grid.K
        ek = grid.ek
        ek_c = grid.ek_conj

        # Velocity-independent part
        cos_b = np.cos(vf * K * dt)
        sin_b = np.sin(vf * K * dt)

        # ── K valley: tilt = (+wx, +wy), chirality χ = +1 ────────────
        # H_K = vf(σx·kx + σy·ky)  →  M12 uses ek*, M21 uses ek
        a_K = wx * KX + wy * KY
        tilt_K = np.exp(MINUS_I * a_K * REAL_DTYPE(dt)).astype(COMPLEX_DTYPE, copy=False)

        self._M11_K = tilt_K * cos_b
        self._M12_K = MINUS_I * tilt_K * sin_b * ek_c
        self._M21_K = MINUS_I * tilt_K * sin_b * ek
        self._M22_K = tilt_K * cos_b

        # ── K' valley: tilt = (-wx, -wy), chirality χ = -1 ───────────
        # H_K' = vf(σx·kx − σy·ky)  →  M12 uses ek, M21 uses ek*
        # (swapped relative to K: opposite pseudospin-momentum locking)
        # a_Kp = -a_K  →  tilt_Kp = conj(tilt_K)
        tilt_Kp = np.conj(tilt_K)

        self._M11_Kp = tilt_Kp * cos_b
        self._M12_Kp = MINUS_I * tilt_Kp * sin_b * ek      # ← was ek_c
        self._M21_Kp = MINUS_I * tilt_Kp * sin_b * ek_c    # ← was ek
        self._M22_Kp = tilt_Kp * cos_b

        # ── Scratch arrays for k-space step ───────────────────────────────
        self._k_u = np.empty(grid.shape, dtype=COMPLEX_DTYPE)
        self._k_v = np.empty(grid.shape, dtype=COMPLEX_DTYPE)
        self._work = np.empty(grid.shape, dtype=COMPLEX_DTYPE)

    # ══════════════════════════════════════════════════════════════════════
    #  Real-space half-step
    # ══════════════════════════════════════════════════════════════════════

    def _realspace_halfstep(self, psi: np.ndarray) -> None:
        """
        Apply the real-space half-step to the 4-component spinor in-place.

        Without coupling: diagonal phase factors (identical to v1.0).
        With scalar coupling: 2×2 rotation in the (K_s, K'_s) subspace
        for each pseudospin component s, followed by the diagonal phase.

        Because exp(-i·a·I₂) commutes with the valley rotation R,
        the order of operations doesn't affect the result. We apply
        the valley rotation first, then the diagonal phase — this way
        the uncoupled limit is reached by simply skipping the rotation.
        """
        if self._has_coupling:
            self._apply_valley_rotation(psi)

        if self._has_mass:
            psi[0] *= self._phase_r_half_up      # K↑
            psi[1] *= self._phase_r_half_down     # K↓
            psi[2] *= self._phase_r_half_up       # K'↑
            psi[3] *= self._phase_r_half_down     # K'↓
        else:
            psi[0] *= self._phase_r_half
            psi[1] *= self._phase_r_half
            psi[2] *= self._phase_r_half
            psi[3] *= self._phase_r_half

    def _apply_valley_rotation(self, psi: np.ndarray) -> None:
        """
        Apply the intervalley coupling rotation in the (K, K') subspace.

        For scalar coupling U = u·I₂, the rotation is identical for
        ↑ and ↓ pseudospin components:

            | psi_K_s  |     | cos|u|t    -i(u/|u|)sin|u|t  | | psi_K_s  |
            | psi_K'_s |  =  | -i(u*/|u|)sin|u|t   cos|u|t  | | psi_K'_s |

        Applied separately to s=↑ (indices 0,2) and s=↓ (indices 1,3).
        """
        cos_u = self._cos_u_half
        mix_KKp = self._mix_KKp     # -i (u/|u|) sin(|u|dt/2)
        mix_KpK = self._mix_KpK     # -i (u*/|u|) sin(|u|dt/2)

        # ── Pseudospin-up pair: (K↑, K'↑) = (psi[0], psi[2]) ─────────
        np.multiply(cos_u, psi[0], out=self._scratch_0)
        np.multiply(mix_KKp, psi[2], out=self._work)
        self._scratch_0 += self._work

        np.multiply(mix_KpK, psi[0], out=self._scratch_1)
        np.multiply(cos_u, psi[2], out=self._work)
        self._scratch_1 += self._work

        psi[0][...] = self._scratch_0
        psi[2][...] = self._scratch_1

        # ── Pseudospin-down pair: (K↓, K'↓) = (psi[1], psi[3]) ───────
        np.multiply(cos_u, psi[1], out=self._scratch_0)
        np.multiply(mix_KKp, psi[3], out=self._work)
        self._scratch_0 += self._work

        np.multiply(mix_KpK, psi[1], out=self._scratch_1)
        np.multiply(cos_u, psi[3], out=self._work)
        self._scratch_1 += self._work

        psi[1][...] = self._scratch_0
        psi[3][...] = self._scratch_1

    # ══════════════════════════════════════════════════════════════════════
    #  k-space full step (block-diagonal)
    # ══════════════════════════════════════════════════════════════════════

    def _kspace_step_valley(
        self, psi_u: np.ndarray, psi_v: np.ndarray,
        M11, M12, M21, M22,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the 2×2 k-space unitary to one valley's spinor components.
        Returns (psi_u_new, psi_v_new) in the scratch arrays.
        """
        u_k = np.asarray(self._fft2(psi_u), dtype=COMPLEX_DTYPE)
        v_k = np.asarray(self._fft2(psi_v), dtype=COMPLEX_DTYPE)

        np.multiply(M11, u_k, out=self._k_u)
        np.multiply(M12, v_k, out=self._work)
        self._k_u += self._work

        np.multiply(M21, u_k, out=self._k_v)
        np.multiply(M22, v_k, out=self._work)
        self._k_v += self._work

        return (
            np.asarray(self._ifft2(self._k_u), dtype=COMPLEX_DTYPE),
            np.asarray(self._ifft2(self._k_v), dtype=COMPLEX_DTYPE),
        )

    def _kspace_fullstep(self, psi: np.ndarray) -> None:
        """Apply the block-diagonal k-space full step in-place."""
        # K valley: components (0, 1)
        psi[0], psi[1] = self._kspace_step_valley(
            psi[0], psi[1],
            self._M11_K, self._M12_K, self._M21_K, self._M22_K,
        )

        # K' valley: components (2, 3)
        psi[2], psi[3] = self._kspace_step_valley(
            psi[2], psi[3],
            self._M11_Kp, self._M12_Kp, self._M21_Kp, self._M22_Kp,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Public interface
    # ══════════════════════════════════════════════════════════════════════

    def step(self, psi: np.ndarray) -> np.ndarray:
        """
        Advance the 4-component spinor by one time step.

        Symmetric Strang splitting:
            ψ(t+dt) ≈ e^{-iH_r dt/2} · F⁻¹[e^{-iH_k dt} · F[e^{-iH_r dt/2} ψ(t)]]

        Parameters
        ----------
        psi : complex ndarray, shape (4, Ny, Nx) — modified in-place

        Returns
        -------
        psi : same array, updated
        """
        self._realspace_halfstep(psi)
        self._kspace_fullstep(psi)
        self._realspace_halfstep(psi)
        return psi

    @property
    def is_coupled(self) -> bool:
        return self._has_coupling
