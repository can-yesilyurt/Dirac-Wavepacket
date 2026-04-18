"""
Valley propagator adapter
=========================

Provides a uniform interface over the two valley propagation strategies:

  - INDEPENDENT (v1.0):
        Two SplitOperatorPropagator instances, one per valley, each holding
        its own (2, Ny, Nx) spinor. Step them separately. Drains, snapshots,
        observables all act on (2, Ny, Nx) arrays.

  - COUPLED (v2.0):
        One CoupledSplitOperatorPropagator holding a single (4, Ny, Nx)
        spinor with K = (0,1) and K' = (2,3). The coupling matrix mixes
        the two valley blocks during the real-space half-step. Adapter
        exposes views psi_K and psi_Kp that look like (2, Ny, Nx) spinors
        to downstream code.

Both modes expose:

    .step()                      → advance one time step
    .psi_K, .psi_Kp              → live views into the K / K' spinors
    .fft_backend, .fft_threads   → for logging
    .is_coupled                  → bool, distinguishes the two modes

The adapter also offers ``apply_drains(drains_K, drains_Kp, t)`` which
correctly accumulates valley-resolved T, R for both modes. In the coupled
case the drain masks are applied to the K and K' blocks of the same
4-spinor in place, since drains are pseudospin-diagonal and local in
space and therefore valley-preserving.

The independent case reproduces v1.0 exactly and is the default.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grid import Grid
from .precision import COMPLEX_DTYPE
from .propagator import SplitOperatorPropagator
from .propagator_coupled import CoupledSplitOperatorPropagator
from .wavepacket import init_wavepacket
from .config import WavepacketConfig
from .coupling import scalar_coupling, region_mask_barrier, defect_line_coupling


def build_coupling_field(
    grid: Grid,
    V: np.ndarray,
    V0: float,
    coupling_cfg,
) -> Optional[np.ndarray]:
    """
    Construct the coupling field array u(x, y) from a CouplingConfig.

    Returns None if coupling is disabled or strength is zero, in which case
    the coupled propagator runs in its zero-coupling limit (mathematically
    identical to two independent propagations, but in a single 4-spinor).
    """
    if not coupling_cfg.enabled or coupling_cfg.strength == 0.0:
        return None

    ctype = coupling_cfg.type.lower()
    strength = float(coupling_cfg.strength)

    if ctype == "barrier":
        threshold = float(coupling_cfg.region_threshold) * abs(float(V0))
        mask = region_mask_barrier(grid, V, threshold=threshold)
        return scalar_coupling(grid, strength=strength, region_mask=mask)
    elif ctype == "uniform":
        return scalar_coupling(grid, strength=strength)
    elif ctype == "defect_line":
        return defect_line_coupling(
            grid, strength=strength,
            y_position=float(coupling_cfg.line_position),
            width=float(coupling_cfg.line_width),
        )
    else:
        raise ValueError(
            f"Unknown coupling.type='{coupling_cfg.type}'. "
            f"Expected one of: barrier, uniform, defect_line."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Independent propagator pair (v1.0 mode, default)
# ═══════════════════════════════════════════════════════════════════════════════

class IndependentValleyPropagator:
    """
    Wraps two SplitOperatorPropagator instances, one per valley.

    The valley spinors live in two separate (2, Ny, Nx) arrays.
    """

    is_coupled = False

    def __init__(
        self,
        grid: Grid,
        V: np.ndarray,
        vf: float,
        wx: float,
        wy: float,
        dt: float,
        wp_cfg: WavepacketConfig,
        M: Optional[np.ndarray] = None,
        fft_backend: str = "auto",
        fft_threads: Optional[int] = None,
    ):
        self.grid = grid
        self.dt = dt

        self._prop_K = SplitOperatorPropagator(
            grid=grid, V=V, vf=vf, wx=wx, wy=wy, dt=dt, M=M,
            chirality=+1,
            fft_backend=fft_backend, fft_threads=fft_threads,
        )
        self._prop_Kp = SplitOperatorPropagator(
            grid=grid, V=V, vf=vf, wx=-wx, wy=-wy, dt=dt, M=M,
            chirality=-1,
            fft_backend=fft_backend, fft_threads=fft_threads,
        )

        self.psi_K = init_wavepacket(grid, wp_cfg, chirality=+1)
        self.psi_Kp = init_wavepacket(grid, wp_cfg, chirality=-1)

    @property
    def psi_full(self):
        """No 4-component spinor in independent mode — returns None."""
        return None

    def step(self) -> None:
        self._prop_K.step(self.psi_K)
        self._prop_Kp.step(self.psi_Kp)

    def apply_drains(self, drains_K, drains_Kp, t: float) -> None:
        drains_K.apply(self.psi_K, t)
        drains_Kp.apply(self.psi_Kp, t)

    @property
    def fft_backend(self) -> str:
        return self._prop_K.fft_backend

    @property
    def fft_threads(self) -> int:
        return self._prop_K.fft_threads


# ═══════════════════════════════════════════════════════════════════════════════
#  Coupled propagator wrapper (v2.0 mode)
# ═══════════════════════════════════════════════════════════════════════════════

class CoupledValleyPropagator:
    """
    Wraps a single CoupledSplitOperatorPropagator and exposes psi_K, psi_Kp
    as 2-component views into its 4-component spinor. Downstream consumers
    (drains, observables, wavefunction I/O) treat them as ordinary spinors.

    Important: psi_K and psi_Kp are NOT independent arrays. They are views
    into psi_full[0:2] and psi_full[2:4] respectively. Modifying one block
    in place is observed by the other; this is exactly what we want for
    drain absorption (which acts on each block separately and locally in
    real space) and for snapshot saving.
    """

    is_coupled = True

    def __init__(
        self,
        grid: Grid,
        V: np.ndarray,
        vf: float,
        wx: float,
        wy: float,
        dt: float,
        wp_cfg: WavepacketConfig,
        coupling_u: Optional[np.ndarray],
        M: Optional[np.ndarray] = None,
        fft_backend: str = "auto",
        fft_threads: Optional[int] = None,
        source_mode: str = "both",
    ):
        """
        source_mode :  {"both", "K_only", "Kp_only"}
            Controls which valley blocks are populated at t = 0.

            - "both"    : both K and K' blocks initialized to the same
                          Gaussian spinor. This is the historical default
                          and is what you want for measuring |O_K|, |O_Kp|
                          with coupling off (independent channels).
            - "K_only"  : only the K block is sourced; K' starts at zero.
                          With coupling on, K' fills up from intervalley
                          scattering — gives the column A_{*,K} of the
                          2x2 channel response matrix.
            - "Kp_only" : only the K' block is sourced; gives A_{*,K'}.

            Two single-source runs let you reconstruct the full coupled
            response for arbitrary input superpositions post-hoc, which
            is what the readout-visibility analysis needs.
        """
        self.grid = grid
        self.dt = dt
        self.source_mode = source_mode

        self._prop = CoupledSplitOperatorPropagator(
            grid=grid, V=V, vf=vf, wx=wx, wy=wy, dt=dt, M=M,
            coupling_u=coupling_u,
            fft_backend=fft_backend, fft_threads=fft_threads,
        )

        psi_init_K  = init_wavepacket(grid, wp_cfg, chirality=+1)   # (2, Ny, Nx)
        psi_init_Kp = init_wavepacket(grid, wp_cfg, chirality=-1)   # (2, Ny, Nx)
        self._psi_full = np.zeros((4, grid.Ny, grid.Nx), dtype=COMPLEX_DTYPE)

        mode = source_mode.lower().replace("'", "p")
        if mode == "both":
            self._psi_full[0:2] = psi_init_K
            self._psi_full[2:4] = psi_init_Kp
        elif mode in ("k_only", "konly", "k"):
            self._psi_full[0:2] = psi_init_K
            # K' block stays zero
        elif mode in ("kp_only", "kponly", "kp"):
            self._psi_full[2:4] = psi_init_Kp
            # K block stays zero
        else:
            raise ValueError(
                f"Unknown source_mode={source_mode!r}. "
                f"Expected one of: 'both', 'K_only', 'Kp_only'."
            )

        # Live views — modifications propagate through to the underlying spinor
        self.psi_K = self._psi_full[0:2]
        self.psi_Kp = self._psi_full[2:4]

    @property
    def psi_full(self) -> np.ndarray:
        """
        The full 4-component coupled spinor (K↑, K↓, K'↑, K'↓), shape (4, Ny, Nx).
        Used by the wavefunction dumper to save the joint state for v2.0
        post-processing (intervalley coherence, Bloch vector, etc.).
        """
        return self._psi_full

    def step(self) -> None:
        self._prop.step(self._psi_full)
        # Re-bind views in case in-place ops above invalidated them
        # (np slice views remain valid for in-place ops, but reassigning
        # is needed if a script ever does psi_K = something_new)
        self.psi_K = self._psi_full[0:2]
        self.psi_Kp = self._psi_full[2:4]

    def apply_drains(self, drains_K, drains_Kp, t: float) -> None:
        # Drains are local in space and pseudospin-diagonal — they don't
        # couple valleys. Apply each drain to its own valley block.
        # The block views modify the underlying 4-spinor in place.
        drains_K.apply(self.psi_K, t)
        drains_Kp.apply(self.psi_Kp, t)

    @property
    def fft_backend(self) -> str:
        return self._prop.fft_backend

    @property
    def fft_threads(self) -> int:
        return self._prop.fft_threads


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_valley_propagator(
    *,
    grid: Grid,
    V: np.ndarray,
    vf: float,
    wx: float,
    wy: float,
    dt: float,
    wp_cfg: WavepacketConfig,
    coupling_cfg,
    V0: float,
    M: Optional[np.ndarray] = None,
    fft_backend: str = "auto",
    fft_threads: Optional[int] = None,
    source_mode: str = "both",
):
    """
    Return the appropriate valley propagator wrapper.

    If coupling_cfg.enabled is True, builds the 4-component CoupledValleyPropagator.
    Otherwise builds the v1.0 IndependentValleyPropagator.

    source_mode : {"both", "K_only", "Kp_only"}
        Forwarded to CoupledValleyPropagator. Ignored in independent mode
        because the two valleys never mix there (single-source has no
        effect on the other valley's output). A warning is emitted if
        source_mode != "both" and coupling is off.
    """
    if coupling_cfg.enabled:
        coupling_u = build_coupling_field(grid, V, V0, coupling_cfg)
        return CoupledValleyPropagator(
            grid=grid, V=V, vf=vf, wx=wx, wy=wy, dt=dt,
            wp_cfg=wp_cfg, coupling_u=coupling_u, M=M,
            fft_backend=fft_backend, fft_threads=fft_threads,
            source_mode=source_mode,
        )
    else:
        if source_mode.lower() != "both":
            import warnings
            warnings.warn(
                f"source_mode={source_mode!r} requested but coupling is "
                f"disabled. In independent mode valleys never mix, so "
                f"single-source runs give the trivial response. Using "
                f"'both' instead. Enable coupling to explore A_{{σσ'}}.",
                stacklevel=2,
            )
        return IndependentValleyPropagator(
            grid=grid, V=V, vf=vf, wx=wx, wy=wy, dt=dt,
            wp_cfg=wp_cfg, M=M,
            fft_backend=fft_backend, fft_threads=fft_threads,
        )
