"""
Analytical module: exact Dirac barrier transmission coefficient.

Implements the reflection amplitude from the transfer-matrix solution of
the 1D Dirac equation through a rectangular barrier (Katsnelson et al. 2006),
using the EXACT complex-valued formula — identical to the Mathematica expression:

    kF = E / (ℏvF)
    ky = kF · sin(φ)
    qx = √[(E−V₀)²/(ℏvF)² − ky²]          ← complex if evanescent
    θ  = atan2(ky, qx)

    r = 2i · e^{iφ} · sin(qx·a) · (sinφ − ss′sinθ)
        / [ss′(e^{−iqxa}cos(φ+θ) + e^{iqxa}cos(φ−θ)) − 2i·sin(qx·a)]

    T = 1 − |r|²

where s = sgn(E), s′ = sgn(E − V₀), and all trig/exp functions are complex-valued
so that the propagating ↔ evanescent crossover is handled seamlessly.

Works in natural units (ℏ = vf = 1) where E and V₀ have units of inverse length,
or in physical units (meV, nm) — set hvF accordingly.
"""

import numpy as np

from .precision import COMPLEX_DTYPE
from typing import Optional


def transmission_coefficient(
    phi_deg: float | np.ndarray,
    E: float,
    V0: float,
    D: float,
    hvF: float = 1.0,
) -> np.ndarray | float:
    """
    Compute transmission T(φ) through a Dirac npn barrier.

    Parameters
    ----------
    phi_deg : incidence angle(s) in degrees  (scalar or array)
    E       : Fermi energy  (same units as V0)
    V0      : barrier height
    D       : barrier width
    hvF     : ℏ·vF product  (default 1.0 for natural units;
              use 1.055e-34 * 1e6 for SI with E,V0 in Joules, D in metres;
              use 6.582e-2 for E,V0 in meV, D in nm)

    Returns
    -------
    T : transmission probability, same shape as phi_deg
    """
    scalar = np.isscalar(phi_deg)
    phi_deg = np.atleast_1d(np.asarray(phi_deg, dtype=np.float64))
    phi = np.deg2rad(phi_deg)

    s  = np.sign(E)
    sp = np.sign(E - V0)
    ss = s * sp

    kF = E / hvF
    ky = kF * np.sin(phi)

    # qx as complex: √[(E-V0)²/hvF² - ky²]
    qx2 = ((E - V0) / hvF) ** 2 - ky ** 2
    qx = np.sqrt(qx2.astype(COMPLEX_DTYPE))

    # θ = atan2(ky, qx) — complex-valued
    # numpy atan2 doesn't support complex, use arctan(ky/qx)
    # but need to handle qx→0.  atan2(ky, qx) for real qx = arctan(ky/qx)
    # For complex qx, this generalises naturally.
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.arctan(ky / qx)
        # where qx=0: θ = ±π/2
        theta = np.where(np.abs(qx) < 1e-30, np.sign(ky) * np.pi / 2, theta)

    # Complex products
    qxa = qx * D

    sin_qxa = np.sin(qxa)        # complex sin → sinh for imaginary arg
    exp_pos = np.exp(1j * qxa)
    exp_neg = np.exp(-1j * qxa)

    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)     # complex
    cos_ppt = np.cos(phi + theta)
    cos_pmt = np.cos(phi - theta)
    exp_iphi = np.exp(1j * phi)

    # Numerator:  2i · e^{iφ} · sin(qx·a) · (sinφ − ss′·sinθ)
    numerator = 2j * exp_iphi * sin_qxa * (sin_phi - ss * sin_theta)

    # Denominator: ss′·(e^{-iqxa}cos(φ+θ) + e^{iqxa}cos(φ−θ)) − 2i·sin(qxa)
    denominator = ss * (exp_neg * cos_ppt + exp_pos * cos_pmt) - 2j * sin_qxa

    # r = num / den
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(np.abs(denominator) > 1e-30,
                     numerator / denominator,
                     0.0)

    T = 1.0 - np.abs(r) ** 2
    T = np.clip(T, 0.0, 1.0)

    return float(T[0]) if scalar else T


def transmission_curve(
    E: float,
    V0: float,
    D: float,
    hvF: float = 1.0,
    phi_min: float = -89.5,
    phi_max: float = 89.5,
    n_pts: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute T(φ) over a range of angles.

    Returns
    -------
    phi_deg, T : arrays of shape (n_pts,)
    """
    phi_deg = np.linspace(phi_min, phi_max, n_pts)
    T = transmission_coefficient(phi_deg, E, V0, D, hvF)
    return phi_deg, T


def fabry_perot_angles(
    E: float,
    V0: float,
    D: float,
    hvF: float = 1.0,
    max_n: int = 50,
) -> list[dict]:
    """
    Compute Fabry-Pérot resonance angles where qx·D = nπ → T ≈ 1.

    Returns
    -------
    list of dicts with keys 'n', 'angle_deg', 'qxD'
    """
    kF = E / hvF
    E_in = E - V0
    if abs(E_in) < 1e-30:
        return []

    results = []
    for n in range(1, max_n + 1):
        qx = n * np.pi / D
        ky2 = (E_in / hvF) ** 2 - qx ** 2
        if ky2 < 0:
            continue
        sin_phi = np.sqrt(ky2) / kF
        if sin_phi >= 1:
            continue
        angle = np.degrees(np.arcsin(sin_phi))
        results.append({
            'n': n,
            'angle_deg': angle,
            'qxD': qx * D,
        })
    return results


# ── physical-unit helpers ────────────────────────────────────────────────────

# ℏvF in different unit systems
HVF_SI   = 1.055e-34 * 1e6       # J·m   (E in J, D in m)
HVF_MEVNM = 1.055e-34 * 1e6 / (1.6e-19 * 1e-3 * 1e-9)  # meV·nm ≈ 659.0


def T_physical(phi_deg, En_meV, V0_meV, a_nm):
    """
    Transmission using physical units directly.

    Parameters
    ----------
    phi_deg : angle(s) in degrees
    En_meV  : Fermi energy in meV
    V0_meV  : barrier height in meV
    a_nm    : barrier width in nm

    Returns
    -------
    T : transmission probability
    """
    return transmission_coefficient(phi_deg, En_meV, V0_meV, a_nm, hvF=HVF_MEVNM)


def natural_to_physical(E, V0, D, En_meV=None):
    """
    Convert natural-unit (ℏ=vf=1) parameters to physical units.

    If En_meV is given, scales so that E maps to En_meV.
    Otherwise uses En_meV = E * ℏvF / (qe * 1e-3) with standard constants.

    Returns
    -------
    dict with 'En_meV', 'V0_meV', 'a_nm', 'hvF_meVnm'
    """
    if En_meV is None:
        En_meV = E * HVF_MEVNM  # not very useful, but consistent
    scale = En_meV / E
    return {
        'En_meV': En_meV,
        'V0_meV': V0 * scale,
        'a_nm': D / E * En_meV / HVF_MEVNM,
        'hvF_meVnm': HVF_MEVNM,
    }


# ── angular convolution (finite wavepacket width) ────────────────────────────

def transmission_convolved(
    phi_deg: float | np.ndarray,
    E: float,
    V0: float,
    D: float,
    sigma: float,
    hvF: float = 1.0,
    n_conv: int = 2000,
) -> np.ndarray | float:
    """
    Compute T(φ) convolved with the Gaussian angular profile of a wavepacket.

    A wavepacket with spatial width σ and central wavevector k₀ = E/hvF carries
    an angular spread Δφ ≈ 1/(2·σ·k₀). This function averages the analytical T
    over a Gaussian in ky-space (which maps to a Gaussian in angle near φ₀).

    More precisely, the wavepacket momentum distribution is:
        |g(ky)|² ∝ exp[-(ky - ky0)² · 2σ²]

    where ky0 = kF·sin(φ₀), and we integrate T(ky) · |g(ky)|² over ky,
    which corresponds to a Gaussian convolution in angle with width
    Δφ ≈ 1/(2·σ·k₀·cos(φ₀)).

    Parameters
    ----------
    phi_deg : central incidence angle(s) in degrees
    E, V0, D : barrier parameters
    sigma   : wavepacket spatial width (same units as 1/k₀)
    hvF     : ℏvF (default 1 for natural units)
    n_conv  : number of points for convolution quadrature

    Returns
    -------
    T_conv : convolved transmission, same shape as phi_deg
    """
    scalar = np.isscalar(phi_deg)
    phi_deg = np.atleast_1d(np.asarray(phi_deg, dtype=np.float64))

    k0 = E / hvF  # kF
    # Angular width of the Gaussian in k-space: σ_φ ≈ 1/(2·σ·k0) radians
    sigma_phi_rad = 1.0 / (2.0 * sigma * k0)
    sigma_phi_deg = np.degrees(sigma_phi_rad)

    # For the convolution, sample a fine grid of angles spanning ±4σ_φ
    half_width = 4.0 * sigma_phi_deg
    dphi = np.linspace(-half_width, half_width, n_conv)
    gauss = np.exp(-0.5 * (dphi / sigma_phi_deg) ** 2)
    gauss /= gauss.sum()  # normalise

    results = np.empty_like(phi_deg)
    for i, phi0 in enumerate(phi_deg):
        phi_sample = phi0 + dphi
        # Clip to valid range
        phi_sample = np.clip(phi_sample, -89.9, 89.9)
        T_sample = transmission_coefficient(phi_sample, E, V0, D, hvF)
        results[i] = np.sum(T_sample * gauss)

    return float(results[0]) if scalar else results
