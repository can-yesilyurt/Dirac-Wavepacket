"""
Transport-side post-processing utilities.

These metrics are derived purely from the drain-contact histories recorded by
``DrainContacts`` during transport runs.  They do not depend on any live phase
tracking and are therefore safe to keep in the lean transport pipeline.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def group_delay_from_histories(times, T_history, R_history) -> Dict[str, float]:
    """
    Compute group-delay observables from cumulative drain histories.

    Parameters
    ----------
    times
        Time stamps corresponding to the cumulative drain histories.
    T_history, R_history
        Cumulative transmitted/reflected probabilities recorded at ``times``.

    Returns
    -------
    dict
        ``tau_T`` / ``tau_R`` are the mean arrival times, ``tau_T_peak`` /
        ``tau_R_peak`` are the peak-arrival times from dT/dt and dR/dt, and
        ``sigma_T`` / ``sigma_R`` are the temporal widths.
    """
    t = _as_float_array(times)
    T = _as_float_array(T_history)
    R = _as_float_array(R_history)

    n = min(len(t), len(T), len(R))
    t = t[:n]
    T = T[:n]
    R = R[:n]

    result: Dict[str, float] = {
        "T_final": float(T[-1]) if n > 0 else 0.0,
        "R_final": float(R[-1]) if n > 0 else 0.0,
        "tau_T": np.nan,
        "tau_R": np.nan,
        "tau_T_peak": np.nan,
        "tau_R_peak": np.nan,
        "sigma_T": np.nan,
        "sigma_R": np.nan,
    }

    if n < 2:
        return result

    dt = np.diff(t)
    valid_dt = dt > 0
    if not np.any(valid_dt):
        return result

    t_mid = 0.5 * (t[:-1] + t[1:])
    t_mid = t_mid[valid_dt]
    dt = dt[valid_dt]

    dT = np.diff(T)[valid_dt]
    dR = np.diff(R)[valid_dt]
    dTdt = np.divide(dT, dt, out=np.zeros_like(dT), where=dt > 0)
    dRdt = np.divide(dR, dt, out=np.zeros_like(dR), where=dt > 0)

    # Small numerical negatives can appear from floating-point noise. They are
    # unphysical for cumulative drain histories, so clip them away.
    dTdt = np.clip(dTdt, 0.0, None)
    dRdt = np.clip(dRdt, 0.0, None)

    T_final = float(T[-1])
    if T_final > 1e-8 and np.any(dTdt > 0):
        tau_T = float(np.sum(t_mid * dTdt * dt)) / T_final
        tau_T_peak = float(t_mid[int(np.argmax(dTdt))])
        var_T = float(np.sum((t_mid - tau_T) ** 2 * dTdt * dt)) / T_final
        result.update({
            "tau_T": tau_T,
            "tau_T_peak": tau_T_peak,
            "sigma_T": float(np.sqrt(max(var_T, 0.0))),
        })

    R_final = float(R[-1])
    if R_final > 1e-8 and np.any(dRdt > 0):
        tau_R = float(np.sum(t_mid * dRdt * dt)) / R_final
        tau_R_peak = float(t_mid[int(np.argmax(dRdt))])
        var_R = float(np.sum((t_mid - tau_R) ** 2 * dRdt * dt)) / R_final
        result.update({
            "tau_R": tau_R,
            "tau_R_peak": tau_R_peak,
            "sigma_R": float(np.sqrt(max(var_R, 0.0))),
        })

    return result


def valley_group_delay_from_results(result: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Compute valley-resolved delay differences from a ``run_simulation`` result.

    Returns ``None`` when the result does not contain dual-valley drain
    histories.
    """
    if "T_history_Kp" not in result or "R_history_Kp" not in result:
        return None

    times = result.get("det_times", [])
    gd_K = group_delay_from_histories(times, result.get("T_history", []), result.get("R_history", []))
    gd_Kp = group_delay_from_histories(times, result.get("T_history_Kp", []), result.get("R_history_Kp", []))

    return {
        "gd_K": gd_K,
        "gd_Kp": gd_Kp,
        "delta_tau_T": gd_K["tau_T_peak"] - gd_Kp["tau_T_peak"],
        "delta_tau_T_mean": gd_K["tau_T"] - gd_Kp["tau_T"],
        "delta_tau_R": gd_K["tau_R_peak"] - gd_Kp["tau_R_peak"],
        "delta_tau_R_mean": gd_K["tau_R"] - gd_Kp["tau_R"],
    }
