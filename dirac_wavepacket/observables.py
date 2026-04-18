"""
Physical observables for 2D Dirac spinor wavefunctions.

The spinor convention used throughout the codebase is
    psi.shape == (2, Ny, Nx)
with component 0 = spin-up, component 1 = spin-down.
"""

from __future__ import annotations

import numpy as np


def charge_density(psi: np.ndarray) -> np.ndarray:
    """Probability density rho(x,y) = |psi_up|^2 + |psi_down|^2."""
    return np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2


def total_probability(psi: np.ndarray, grid) -> float:
    """Integral of probability density over the grid."""
    dA = getattr(grid, "dA", None)
    if dA is None:
        dA = float(grid.dx) * float(grid.dy)
    return float(np.sum(charge_density(psi)) * dA)


def current_density(
    psi: np.ndarray,
    grid=None,
    vf: float = 1.0,
    wx: float = 0.0,
    wy: float = 0.0,
):
    """
    Probability current for the tilted 2D Dirac Hamiltonian:
        H = (w · k) I + v_f (sigma · k)
        j = w rho + v_f psi^dagger sigma psi

    Parameters
    ----------
    psi : ndarray, shape (2, Ny, Nx)
        Two-component spinor in real space.
    grid : ignored, optional
        Accepted for compatibility with older helper code.
    vf : float
        Fermi velocity prefactor multiplying the Pauli-current term.
    wx, wy : float
        Tilt / convective velocity components for the valley represented by
        ``psi``. Pass the signed values for the specific valley: typically
        ``(+wx, +wy)`` for K and ``(-wx, -wy)`` for K′.

    Returns
    -------
    jx, jy : ndarray, shape (Ny, Nx)
        Real-valued current-density components.
    """
    rho = charge_density(psi)
    overlap = np.conj(psi[0]) * psi[1]
    jx = vf * 2.0 * np.real(overlap)
    jy = vf * np.real(
        np.conj(psi[0]) * (-1j) * psi[1] + np.conj(psi[1]) * (1j) * psi[0]
    )
    if wx != 0.0:
        jx = jx + float(wx) * rho
    if wy != 0.0:
        jy = jy + float(wy) * rho
    return jx, jy


def spin_density(psi: np.ndarray):
    """
    Spin expectation values <sigma_x>, <sigma_y>, <sigma_z> at each grid point.

    Returns
    -------
    sx, sy, sz : ndarray, shape (Ny, Nx)
    """
    sx = 2.0 * np.real(np.conj(psi[0]) * psi[1])
    sy = 2.0 * np.imag(np.conj(psi[1]) * psi[0])
    sz = np.abs(psi[0]) ** 2 - np.abs(psi[1]) ** 2
    return sx, sy, sz


def expectation_position(psi: np.ndarray, grid):
    """Return <x>, <y> expectation values."""
    rho = charge_density(psi)
    dA = getattr(grid, "dA", None)
    if dA is None:
        dA = float(grid.dx) * float(grid.dy)
    norm = np.sum(rho) * dA
    x_avg = np.sum(rho * grid.X) * dA / norm
    y_avg = np.sum(rho * grid.Y) * dA / norm
    return float(x_avg), float(y_avg)


def _block_centers(coords: np.ndarray, stride: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    stride = max(1, int(stride))
    centers = []
    for i0 in range(0, len(coords), stride):
        i1 = min(i0 + stride, len(coords))
        centers.append(float(np.mean(coords[i0:i1])))
    return np.asarray(centers, dtype=float)




def _regular_block_centers(coords: np.ndarray, stride: int) -> np.ndarray:
    """Equally spaced block centers suitable for streamplot/quiver grids."""
    coords = np.asarray(coords, dtype=float)
    stride = max(1, int(stride))
    if coords.size <= 1 or stride == 1:
        return coords.astype(float, copy=True)
    d = float(coords[1] - coords[0])
    n_blocks = int(np.ceil(len(coords) / stride))
    return coords[0] + 0.5 * (stride - 1) * d + stride * d * np.arange(n_blocks, dtype=float)

def _block_average_2d(arr: np.ndarray, stride_y: int, stride_x: int) -> np.ndarray:
    arr = np.asarray(arr)
    stride_y = max(1, int(stride_y))
    stride_x = max(1, int(stride_x))
    ny, nx = arr.shape
    out = np.empty((int(np.ceil(ny / stride_y)), int(np.ceil(nx / stride_x))), dtype=float)
    oy = 0
    for y0 in range(0, ny, stride_y):
        y1 = min(y0 + stride_y, ny)
        ox = 0
        for x0 in range(0, nx, stride_x):
            x1 = min(x0 + stride_x, nx)
            out[oy, ox] = float(np.mean(arr[y0:y1, x0:x1]))
            ox += 1
        oy += 1
    return out


def current_plot_data(
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    vf: float = 1.0,
    wx: float = 0.0,
    wy: float = 0.0,
    stride: int = 8,
    density_frac: float = 0.02,
    normalize: bool = True,
    block_average: bool = True,
):
    """
    Prepare a visually robust coarse-grained current field for plotting.

    Returns a dictionary with 1D coordinates (x, y), 2D field components (U, V),
    the valid mask, the coarse density rho, and helper magnitude scales.
    """
    stride = max(1, int(stride))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    rho = charge_density(psi)
    jx, jy = current_density(psi, vf=vf, wx=wx, wy=wy)

    if stride > 1 and block_average:
        rho_v = _block_average_2d(rho, stride, stride)
        jx_v = _block_average_2d(jx, stride, stride)
        jy_v = _block_average_2d(jy, stride, stride)
        x_v = _regular_block_centers(x, stride)
        y_v = _regular_block_centers(y, stride)
    else:
        rho_v = rho[::stride, ::stride]
        jx_v = jx[::stride, ::stride]
        jy_v = jy[::stride, ::stride]
        x_v = x[::stride]
        y_v = y[::stride]

    rho_max = max(float(np.nanmax(rho)), 1e-30)
    cutoff = float(density_frac) * rho_max
    mag = np.hypot(jx_v, jy_v)
    mask = np.isfinite(rho_v) & np.isfinite(mag) & (rho_v >= cutoff) & (mag > 1e-30)
    if not np.any(mask):
        return {
            "x": x_v,
            "y": y_v,
            "U": np.empty((0, 0), dtype=float),
            "V": np.empty((0, 0), dtype=float),
            "mask": mask,
            "mag": mag,
            "mag_ref": 0.0,
            "rho": rho_v,
            "rho_max": rho_max,
        }

    if normalize:
        safe_mag = np.where(mask, mag, 1.0)
        U = np.where(mask, jx_v / safe_mag, np.nan)
        V = np.where(mask, jy_v / safe_mag, np.nan)
        mag_ref = 1.0
    else:
        U = np.where(mask, jx_v, np.nan)
        V = np.where(mask, jy_v, np.nan)
        mag_ref = float(np.nanmax(mag[mask])) if np.any(mask) else 0.0

    return {
        "x": x_v,
        "y": y_v,
        "U": U,
        "V": V,
        "mask": mask,
        "mag": mag,
        "mag_ref": float(mag_ref),
        "rho": rho_v,
        "rho_max": rho_max,
    }


def current_quiver_data(
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    vf: float = 1.0,
    wx: float = 0.0,
    wy: float = 0.0,
    stride: int = 8,
    density_frac: float = 0.02,
    normalize: bool = True,
    block_average: bool = True,
):
    """
    Prepare a visually robust quiver field from a spinor snapshot.

    The raw current field is optionally coarse-grained over stride×stride tiles
    before plotting. This is more stable than point sampling and gives much
    clearer arrows on packet snapshots and rendered frames.
    """
    q = current_plot_data(
        psi,
        x,
        y,
        vf=vf,
        wx=wx,
        wy=wy,
        stride=stride,
        density_frac=density_frac,
        normalize=normalize,
        block_average=block_average,
    )
    if q["U"].size == 0:
        return {
            "X": np.empty((0, 0), dtype=float),
            "Y": np.empty((0, 0), dtype=float),
            **q,
        }

    X, Y = np.meshgrid(q["x"], q["y"], indexing="xy")
    return {
        "X": X,
        "Y": Y,
        **q,
    }
