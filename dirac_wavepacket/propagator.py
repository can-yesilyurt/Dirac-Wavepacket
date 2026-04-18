"""
Propagator module: split-operator FFT time evolution for the 2D tilted Dirac equation.

Hamiltonian (chirality χ = ±1):
    H = (wx·kx + wy·ky)·I + vf·(σx·kx + χ·σy·ky) + V(x,y)·I + M(x,y)·σz

    χ = +1  →  K valley   (standard Dirac cone)
    χ = -1  →  K' valley  (opposite chirality: σy → -σy)

Decomposition:
    Hr = V(x,y)·I + M(x,y)·σz                          [diagonal in real space]
    Hk = (wx·kx + wy·ky)·I + vf·(σx·kx + χ·σy·ky)      [diagonal in k-space]

The mass term M(x,y)·σz opens a local gap 2M and confines Dirac fermions
without Klein tunneling (unlike a scalar potential wall). Used for
reflecting transverse boundary conditions (y_bc="reflecting").

Symmetric (Strang) splitting — 2nd-order accurate in dt:
    ψ(t+dt) ≈ exp(-i·Hr·dt/2) · IFFT[ exp(-i·Hk·dt) · FFT[ exp(-i·Hr·dt/2)·ψ(t) ] ]

────────────────────────────────────────────────────────────────────────────────
Real-space half-step  (spinor-diagonal):
    Without mass:  ψ↑,↓ → exp(-i·V·dt/2) · ψ↑,↓
    With mass:     ψ↑ → exp(-i·(V+M)·dt/2) · ψ↑
                   ψ↓ → exp(-i·(V−M)·dt/2) · ψ↓

k-space full step  (2×2 unitary at each (kx,ky)):
    Hk = a·I + b·σ   where  a = wx·kx + wy·ky,  b = vf·(kx, ky)
    exp(-i·Hk·dt) = e^{-i·a·dt} [ cos(|b|·dt)·I - i·sin(|b|·dt)·b̂·σ ]

    u' = e^{-ia·dt} [ cos(|b|·dt)·u  - i·sin(|b|·dt)·ek*·v ]
    v' = e^{-ia·dt} [ cos(|b|·dt)·v  - i·sin(|b|·dt)·ek ·u ]
    where ek = (kx + i·ky)/|k|
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from typing import Callable, Optional

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


def _resolve_fft_backend(
    requested: str = "auto",
    threads: Optional[int] = None,
) -> tuple[str, int, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Resolve the FFT backend for the propagator.

    Selection order:
      1. explicit ``requested`` argument
      2. ``DIRAC2D_FFT_BACKEND`` environment variable when requested="auto"
      3. automatic choice: pyFFTW if installed, else numpy

    Threads default to ``DIRAC2D_FFT_THREADS`` or 1.
    """
    req = (requested or "auto").strip().lower()
    env_backend = os.environ.get("DIRAC2D_FFT_BACKEND", "").strip().lower()
    if req == "auto" and env_backend:
        req = env_backend

    if req not in {"auto", "numpy", "pyfftw"}:
        raise ValueError(
            f"Unknown fft_backend='{requested}'. Expected 'auto', 'numpy', or 'pyfftw'."
        )

    if threads is None:
        env_threads = os.environ.get("DIRAC2D_FFT_THREADS", "1").strip()
        try:
            threads = int(env_threads)
        except ValueError:
            threads = 1
    threads = max(1, int(threads))

    if req == "pyfftw":
        if not _HAVE_PYFFTW:
            raise RuntimeError(
                "fft_backend='pyfftw' was requested, but pyFFTW is not installed."
            )
        backend = "pyfftw"
    elif req == "numpy":
        backend = "numpy"
    else:
        backend = "pyfftw" if _HAVE_PYFFTW else "numpy"

    if backend == "numpy":
        return backend, 1, np.fft.fft2, np.fft.ifft2

    def fft2(arr: np.ndarray) -> np.ndarray:
        return _pyfftw_fft.fft2(
            arr,
            overwrite_input=False,
            planner_effort="FFTW_MEASURE",
            threads=threads,
            auto_align_input=True,
            auto_contiguous=True,
        )

    def ifft2(arr: np.ndarray) -> np.ndarray:
        return _pyfftw_fft.ifft2(
            arr,
            overwrite_input=False,
            planner_effort="FFTW_MEASURE",
            threads=threads,
            auto_align_input=True,
            auto_contiguous=True,
        )

    return backend, threads, fft2, ifft2


class SplitOperatorPropagator:
    """
    Pre-computes all grid-level operator arrays to minimise per-step overhead.
    Call .step(psi) to advance the spinor by one time step.

    The propagator automatically uses pyFFTW when available unless
    ``fft_backend='numpy'`` is requested explicitly.
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
        chirality: int = +1,
        fft_backend: str = "auto",
        fft_threads: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        grid : Grid
        V    : real ndarray, shape (Ny, Nx) – electrostatic potential
        vf   : Fermi velocity
        wx,wy: tilt velocity components
        dt   : time step
        M    : real ndarray, shape (Ny, Nx) – mass confinement term (optional)
               Opens a local gap 2M(x,y) via M·σz.
               Used for reflecting y-boundaries in Dirac systems.
        chirality : int, +1 or -1
            Valley chirality. +1 for K valley (σx·kx + σy·ky),
            -1 for K' valley (σx·kx − σy·ky). Opposite chirality swaps
            the role of ek and ek* in the k-space unitary, giving the
            correct pseudospin-momentum locking at each Dirac cone.
        fft_backend : {"auto", "numpy", "pyfftw"}
            FFT backend. "auto" prefers pyFFTW when installed.
        fft_threads : int or None
            Number of FFT threads for pyFFTW. Ignored by numpy.
        """
        if chirality not in (+1, -1):
            raise ValueError(f"chirality must be +1 or -1, got {chirality}")
        self.grid = grid
        self.dt = dt
        self.chirality = chirality

        V = np.asarray(V, dtype=REAL_DTYPE)
        M = None if M is None else np.asarray(M, dtype=REAL_DTYPE)

        self.fft_backend, self.fft_threads, self._fft2, self._ifft2 = _resolve_fft_backend(
            requested=fft_backend,
            threads=fft_threads,
        )

        # ── real-space half-step phase ─────────────────────────────────
        self._has_mass = M is not None
        if self._has_mass:
            self._phase_r_half_up = np.exp(MINUS_I * (V + M) * REAL_DTYPE(dt / 2.0)).astype(COMPLEX_DTYPE, copy=False)
            self._phase_r_half_down = np.exp(MINUS_I * (V - M) * REAL_DTYPE(dt / 2.0)).astype(COMPLEX_DTYPE, copy=False)
        else:
            self._phase_r_half = np.exp(MINUS_I * V * REAL_DTYPE(dt / 2.0)).astype(COMPLEX_DTYPE, copy=False)

        # ── k-space full-step ──────────────────────────────────────────
        # Chirality determines the pseudospin-momentum locking:
        #   χ = +1 (K ):  H_Dirac = vf(σx·kx + σy·ky)  →  ek_12 = ek*, ek_21 = ek
        #   χ = -1 (K'):  H_Dirac = vf(σx·kx − σy·ky)  →  ek_12 = ek,  ek_21 = ek*
        # This swap implements opposite chirality at the two Dirac cones.
        KX, KY, K = grid.KX, grid.KY, grid.K
        if chirality == +1:
            ek_12 = grid.ek_conj    # ek* in M12 position
            ek_21 = grid.ek         # ek  in M21 position
        else:
            ek_12 = grid.ek         # swapped for opposite chirality
            ek_21 = grid.ek_conj

        a = wx * KX + wy * KY
        b_mag = vf * K

        cos_b = np.cos(b_mag * dt)
        sin_b = np.sin(b_mag * dt)
        tilt_p = np.exp(MINUS_I * a * REAL_DTYPE(dt)).astype(COMPLEX_DTYPE, copy=False)

        self._M11 = tilt_p * cos_b
        self._M12 = MINUS_I * tilt_p * sin_b * ek_12
        self._M21 = MINUS_I * tilt_p * sin_b * ek_21
        self._M22 = tilt_p * cos_b

        # Scratch arrays to reduce per-step temporary allocations.
        self._k_u = np.empty(grid.shape, dtype=COMPLEX_DTYPE)
        self._k_v = np.empty(grid.shape, dtype=COMPLEX_DTYPE)
        self._work = np.empty(grid.shape, dtype=COMPLEX_DTYPE)

    def step(self, psi: np.ndarray) -> np.ndarray:
        """
        Advance the spinor by one time step using the symmetric Strang splitting.

        Parameters
        ----------
        psi : complex ndarray, shape (2, Ny, Nx)  – modified in-place

        Returns
        -------
        psi : same array, updated
        """
        if self._has_mass:
            psi[0] *= self._phase_r_half_up
            psi[1] *= self._phase_r_half_down
        else:
            psi[0] *= self._phase_r_half
            psi[1] *= self._phase_r_half

        u_k = np.asarray(self._fft2(psi[0]), dtype=COMPLEX_DTYPE)
        v_k = np.asarray(self._fft2(psi[1]), dtype=COMPLEX_DTYPE)

        np.multiply(self._M11, u_k, out=self._k_u)
        np.multiply(self._M12, v_k, out=self._work)
        self._k_u += self._work

        np.multiply(self._M21, u_k, out=self._k_v)
        np.multiply(self._M22, v_k, out=self._work)
        self._k_v += self._work

        psi[0][...] = np.asarray(self._ifft2(self._k_u), dtype=COMPLEX_DTYPE)
        psi[1][...] = np.asarray(self._ifft2(self._k_v), dtype=COMPLEX_DTYPE)

        if self._has_mass:
            psi[0] *= self._phase_r_half_up
            psi[1] *= self._phase_r_half_down
        else:
            psi[0] *= self._phase_r_half
            psi[1] *= self._phase_r_half

        return psi

    def recommended_dt(self, vf: float, Vmax: float) -> float:
        """
        Practical upper bound on dt for split-operator accuracy:
            dt < 1 / E_max   where E_max = vf·kmax + Vmax
        The split-operator is unconditionally unitary (never blows up).
        Hard limit: dt < π / E_max (temporal phase aliasing).
        """
        kmax = np.pi / self.grid.dx
        Emax = vf * kmax + Vmax
        return 1.0 / Emax
