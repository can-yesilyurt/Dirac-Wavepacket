"""
Central numeric precision policy for dirac_wavepacket.

Switch COMPLEX_DTYPE here to move the whole codebase between a consistent
single-precision (complex64/float32) and double-precision
(complex128/float64) path.
"""

from __future__ import annotations

import numpy as np

COMPLEX_DTYPE = np.dtype(np.complex64).type
REAL_DTYPE = np.float32 if np.dtype(COMPLEX_DTYPE) == np.dtype(np.complex64) else np.float64

COMPLEX_NAME = np.dtype(COMPLEX_DTYPE).name
REAL_NAME = np.dtype(REAL_DTYPE).name

PI = REAL_DTYPE(np.pi)
TWO_PI = REAL_DTYPE(2.0 * np.pi)
SQRT2 = REAL_DTYPE(np.sqrt(2.0))
INV_SQRT2 = REAL_DTYPE(1.0) / SQRT2
I = COMPLEX_DTYPE(1j)
MINUS_I = COMPLEX_DTYPE(-1j)


def as_complex(arr, copy: bool = False):
    return np.array(arr, dtype=COMPLEX_DTYPE, copy=copy)


def as_real(arr, copy: bool = False):
    return np.array(arr, dtype=REAL_DTYPE, copy=copy)


def exp_i(theta):
    return np.exp(I * np.asarray(theta, dtype=REAL_DTYPE)).astype(COMPLEX_DTYPE, copy=False)


def exp_minus_i(theta):
    return np.exp(MINUS_I * np.asarray(theta, dtype=REAL_DTYPE)).astype(COMPLEX_DTYPE, copy=False)
