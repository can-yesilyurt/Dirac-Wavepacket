"""
Wavefunction I/O: save and load spinor snapshots for post-processing.

Design:
  - One compressed .npz file per snapshot: wf_NNNNN.npz
  - Spinors stored using the package complex dtype
  - Free-reference spinors are optional; newer transport-only simulations
    typically save only the barrier-propagated state plus metadata/config

Usage (save):
    dumper = WavefunctionDumper(out_dir, grid)
    # inside loop:
    dumper.save(step, t, psi_K, psi_Kp, metadata_dict)
    # after loop:
    dumper.write_index()

Usage (load):
    loader = WavefunctionLoader(out_dir)
    for snap in loader:
        psi_K = snap['psi_K']       # package complex dtype, (2, Ny, Nx)
        t     = snap['time']
        ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .precision import COMPLEX_DTYPE, REAL_DTYPE


class WavefunctionDumper:
    """
    Save wavefunction snapshots to compressed .npz files.

    Parameters
    ----------
    out_dir : str or Path
        Directory to write wf_NNNNN.npz files into (created if absent).
    grid : Grid object
        Provides dx, dy for metadata.
    compress : bool
        Use np.savez_compressed (default True, ~3× smaller).
    dtype : numpy dtype
        Storage dtype for spinors (default: package complex dtype).
    sim_config : SimConfig or None
        Full simulation config saved into index.json for later reconstruction.
    """

    def __init__(self, out_dir, grid, compress=True, dtype=COMPLEX_DTYPE,
                 sim_config=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.compress = compress
        self.dtype = dtype
        self._index = []
        self._save_fn = np.savez_compressed if compress else np.savez
        self._sim_config = sim_config

        np.savez_compressed(
            self.out_dir / "grid.npz",
            x=grid.x.astype(REAL_DTYPE),
            y=grid.y.astype(REAL_DTYPE),
            Nx=np.int32(grid.Nx),
            Ny=np.int32(grid.Ny),
            Lx=np.float64(grid.Lx),
            Ly=np.float64(grid.Ly),
            dx=np.float64(grid.dx),
            dy=np.float64(grid.dy),
        )

    def save(self, step, time, psi_K, psi_Kp=None, meta=None,
             psi_free_K=None, psi_free_Kp=None, psi_full=None):
        """
        Save one snapshot.

        Parameters
        ----------
        step : int
        time : float
        psi_K : ndarray — barrier-propagated K spinor, shape (2, Ny, Nx)
        psi_Kp : ndarray or None — barrier-propagated K' spinor, shape (2, Ny, Nx)
        meta : dict of scalar metadata (T_K, R_K, P_K, ...)
        psi_free_K : ndarray or None — optional free-propagated K reference
        psi_free_Kp : ndarray or None — optional free-propagated K' reference
        psi_full : ndarray or None — full 4-component spinor (K↑, K↓, K'↑, K'↓)
            shape (4, Ny, Nx). If provided (coupled mode), saved alongside the
            split blocks so post-processing scripts that need the full joint
            state (e.g., intervalley coherence, cross density matrix, Bloch
            vector diagnostics) can reconstruct it.

        Backward compatibility: psi_K and psi_Kp are always saved with their
        v1.0 shape and meaning. psi_full is purely additive.
        """
        fname = f"wf_{step:06d}.npz"
        fpath = self.out_dir / fname

        data = {
            "psi_K": psi_K.astype(self.dtype),
            "step": np.int32(step),
            "time": np.float64(time),
        }
        if psi_Kp is not None:
            data["psi_Kp"] = psi_Kp.astype(self.dtype)
        if psi_free_K is not None:
            data["psi_free_K"] = psi_free_K.astype(self.dtype)
        if psi_free_Kp is not None:
            data["psi_free_Kp"] = psi_free_Kp.astype(self.dtype)
        if psi_full is not None:
            # Full 4-component coupled spinor (K↑, K↓, K'↑, K'↓).
            # Required to reconstruct intervalley coherence and joint
            # density matrix for v2.0 coupled-mode post-processing.
            data["psi_full"] = psi_full.astype(self.dtype)

        if meta:
            scalar_keys = []
            scalar_vals = []
            for key in sorted(meta.keys()):
                try:
                    arr = np.asarray(meta[key])
                except Exception:
                    continue
                if arr.ndim != 0:
                    continue
                value = arr.item()
                # Store scalar metadata directly at the top level so that
                # legacy post-processing scripts can operate on a single
                # snapshot file without separately opening index.json/grid.npz.
                if isinstance(value, (np.integer, int)):
                    data[key] = np.int64(value)
                    scalar_keys.append(key)
                    scalar_vals.append(float(value))
                elif isinstance(value, (np.floating, float)):
                    data[key] = np.float64(value)
                    scalar_keys.append(key)
                    scalar_vals.append(float(value))
                elif isinstance(value, (np.bool_, bool)):
                    data[key] = np.int8(bool(value))
                    scalar_keys.append(key)
                    scalar_vals.append(float(bool(value)))
            if scalar_keys:
                data["meta_keys"] = np.array(scalar_keys)
                data["meta_vals"] = np.array(scalar_vals, dtype=np.float64)

        self._save_fn(fpath, **data)

        self._index.append({
            "file": fname,
            "step": int(step),
            "time": float(time),
        })

    def write_index(self):
        """Write a JSON index of all saved snapshots + simulation config."""
        idx_data = {
            "n_snapshots": len(self._index),
            "dtype": str(self.dtype),
            "grid": {
                "Nx": int(self.grid.Nx), "Ny": int(self.grid.Ny),
                "Lx": float(self.grid.Lx), "Ly": float(self.grid.Ly),
                "dx": float(self.grid.dx), "dy": float(self.grid.dy),
            },
            "snapshots": self._index,
        }
        if self._sim_config is not None:
            try:
                import dataclasses

                cfg = self._sim_config
                idx_data["config"] = {
                    "grid": dataclasses.asdict(cfg.grid)
                        if dataclasses.is_dataclass(cfg.grid) else {},
                    "physics": dataclasses.asdict(cfg.physics)
                        if dataclasses.is_dataclass(cfg.physics) else {},
                    "time": dataclasses.asdict(cfg.time)
                        if dataclasses.is_dataclass(cfg.time) else {},
                    "wavepacket": dataclasses.asdict(cfg.wavepacket)
                        if dataclasses.is_dataclass(cfg.wavepacket) else {},
                    "potential": dataclasses.asdict(cfg.potential)
                        if dataclasses.is_dataclass(cfg.potential) else {},
                    "absorber": dataclasses.asdict(cfg.absorber)
                        if dataclasses.is_dataclass(cfg.absorber) else {},
                    "detector": dataclasses.asdict(cfg.detector)
                        if dataclasses.is_dataclass(cfg.detector) else {},
                    "title": getattr(cfg, "title", None),
                }
                # v2.0: include the coupling block when present so that
                # post-processing scripts can detect coupled-mode snapshots
                # at the directory level (without opening every npz file).
                if hasattr(cfg, "coupling") and dataclasses.is_dataclass(cfg.coupling):
                    idx_data["config"]["coupling"] = dataclasses.asdict(cfg.coupling)
            except Exception:
                pass

        idx_path = self.out_dir / "index.json"
        with open(idx_path, "w") as f:
            json.dump(idx_data, f, indent=2, default=str)


class WavefunctionLoader:
    """
    Load wavefunction snapshots from a directory.

    Usage:
        loader = WavefunctionLoader("results/wavefunction")

        # Iterate all snapshots in order:
        for snap in loader:
            psi_K = snap['psi_K']    # package complex dtype
            t = snap['time']

        # Load a specific step:
        snap = loader.load_step(500)

        # Get grid/config metadata:
        grid_data = loader.grid
        cfg_dict = loader.config
    """

    def __init__(self, wf_dir):
        self.wf_dir = Path(wf_dir)
        if not self.wf_dir.exists():
            raise FileNotFoundError(f"Wavefunction directory not found: {wf_dir}")

        self.index = None
        self.config = None

        idx_path = self.wf_dir / "index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                idx = json.load(f)
            self.index = idx
            self.config = idx.get("config")
            self._files = [(s["step"], s["file"]) for s in idx["snapshots"]]
        else:
            files = sorted(self.wf_dir.glob("wf_*.npz"))
            self._files = []
            for fp in files:
                step = int(fp.stem.split("_")[1])
                self._files.append((step, fp.name))

        self._files.sort(key=lambda x: x[0])

        grid_path = self.wf_dir / "grid.npz"
        if grid_path.exists():
            g = np.load(grid_path)
            self.grid = {k: g[k] for k in g.files}
        else:
            self.grid = None

    def __len__(self):
        return len(self._files)

    @property
    def steps(self):
        return [s for s, _ in self._files]

    def _load_file(self, fname):
        """Load a single .npz and return a dict."""
        data = np.load(self.wf_dir / fname, allow_pickle=True)
        result = {
            "step": int(data["step"]),
            "time": float(data["time"]),
        }
        handled = {"step", "time", "meta_keys", "meta_vals"}

        # ── Spinor blocks: handle both v1.0 (psi_K, psi_Kp split) and
        #    v2.0 coupled (psi_full as 4-component) formats. ──────────
        has_psi_K = "psi_K" in data
        has_psi_Kp = "psi_Kp" in data
        has_psi_full = "psi_full" in data

        if has_psi_K:
            result["psi_K"] = data["psi_K"].astype(COMPLEX_DTYPE)
            handled.add("psi_K")
        if has_psi_Kp:
            result["psi_Kp"] = data["psi_Kp"].astype(COMPLEX_DTYPE)
            handled.add("psi_Kp")
        if has_psi_full:
            psi_full = data["psi_full"].astype(COMPLEX_DTYPE)
            result["psi_full"] = psi_full
            handled.add("psi_full")
            # If the saver was a future deduplicated variant that only stored
            # psi_full and not psi_K / psi_Kp, synthesize the v1.0 split-block
            # views so existing post-processing scripts keep working.
            if not has_psi_K:
                result["psi_K"] = psi_full[0:2]
            if not has_psi_Kp and psi_full.shape[0] >= 4:
                result["psi_Kp"] = psi_full[2:4]
            result["coupled"] = True
        else:
            result["coupled"] = False
        # Don't let the meta-key loop override the coupled flag we just set
        # from the actual presence of psi_full in the file.
        handled.add("coupled")

        if "psi_free_K" in data:
            result["psi_free_K"] = data["psi_free_K"].astype(COMPLEX_DTYPE)
            handled.add("psi_free_K")
        if "psi_free_Kp" in data:
            result["psi_free_Kp"] = data["psi_free_Kp"].astype(COMPLEX_DTYPE)
            handled.add("psi_free_Kp")

        for key in data.files:
            if key in handled:
                continue
            val = data[key]
            if getattr(val, "ndim", 0) == 0:
                result[key] = val.item()
            else:
                result[key] = val

        if "meta_keys" in data and "meta_vals" in data:
            keys = [str(k) for k in data["meta_keys"]]
            vals = data["meta_vals"]
            result["meta"] = {k: float(v) for k, v in zip(keys, vals)}

        # Attach shared grid metadata for convenience when the snapshot file
        # itself only contains spinors + scalars.
        if self.grid is not None:
            for key in ("x", "y", "Nx", "Ny", "Lx", "Ly", "dx", "dy"):
                if key not in result and key in self.grid:
                    result[key] = self.grid[key]

        # Attach a few common config-derived scalars if they were not embedded
        # directly into the snapshot file.
        if self.config is not None:
            try:
                if "V0" not in result:
                    result["V0"] = float(self.config["potential"]["height"])
            except Exception:
                pass
            try:
                if "vf" not in result:
                    result["vf"] = float(self.config["physics"]["vf"])
            except Exception:
                pass
            try:
                if "k0" not in result:
                    result["k0"] = float(self.config["wavepacket"]["k0"])
            except Exception:
                pass
            try:
                if "Ef" not in result:
                    result["Ef"] = float(self.config["physics"]["vf"]) * float(self.config["wavepacket"]["k0"])
            except Exception:
                pass
            try:
                if "dt" not in result:
                    result["dt"] = float(self.config["time"]["dt"])
            except Exception:
                pass

        return result

    def load_step(self, step):
        """Load a specific step. Raises KeyError if not found."""
        for s, fname in self._files:
            if s == step:
                return self._load_file(fname)
        raise KeyError(f"Step {step} not found. Available: {self.steps[:5]}...")

    def __iter__(self):
        """Iterate all snapshots in step order."""
        for _, fname in self._files:
            yield self._load_file(fname)

    def load_range(self, step_min=0, step_max=None):
        """Iterate snapshots in a step range."""
        for s, fname in self._files:
            if s < step_min:
                continue
            if step_max is not None and s > step_max:
                break
            yield self._load_file(fname)


# ═══════════════════════════════════════════════════════════════════════════════
#  Coupled-state diagnostic helpers
#
#  These operate on a 4-component spinor psi_full of shape (4, Ny, Nx) with
#  layout (K↑, K↓, K'↑, K'↓). They compute joint observables that are
#  inaccessible from the v1.0 split-block (psi_K, psi_Kp) format alone.
# ═══════════════════════════════════════════════════════════════════════════════

def valley_block_K(psi_full):
    """Return the K block (2, Ny, Nx) of a 4-component spinor (read-only view)."""
    return psi_full[0:2]


def valley_block_Kp(psi_full):
    """Return the K' block (2, Ny, Nx) of a 4-component spinor (read-only view)."""
    return psi_full[2:4]


def valley_populations(psi_full, dxdy):
    """
    Spatially integrated valley populations.

    Returns
    -------
    P_K, P_Kp : float
        ∫ |ψ_K|² dx dy and ∫ |ψ_K'|² dx dy
    """
    P_K = float((np.abs(psi_full[0]) ** 2 + np.abs(psi_full[1]) ** 2).sum() * dxdy)
    P_Kp = float((np.abs(psi_full[2]) ** 2 + np.abs(psi_full[3]) ** 2).sum() * dxdy)
    return P_K, P_Kp


def cross_density_field(psi_full):
    """
    Compute the spatially-resolved 2×2 cross density matrix between K and K'.

    For each grid point:
        ρ_KK'(x,y)[s,s'] = ψ_K_s(x,y)* · ψ_K'_s'(x,y)

    Returns
    -------
    rho_cross : complex ndarray, shape (2, 2, Ny, Nx)
        rho_cross[s, s'](x,y) = conj(psi_full[s]) * psi_full[2+s']
        s, s' ∈ {0=↑, 1=↓}.

    For scalar (pseudospin-preserving) intervalley coupling, the off-diagonal
    pseudospin elements rho_cross[0,1] and rho_cross[1,0] should be near zero,
    while the diagonal elements rho_cross[0,0] and rho_cross[1,1] carry the
    intervalley coherence. SOC and pseudospin-flip terms populate the
    off-diagonal pseudospin entries.
    """
    Ny, Nx = psi_full.shape[1], psi_full.shape[2]
    rho_cross = np.empty((2, 2, Ny, Nx), dtype=psi_full.dtype)
    rho_cross[0, 0] = np.conj(psi_full[0]) * psi_full[2]   # K↑* · K'↑
    rho_cross[0, 1] = np.conj(psi_full[0]) * psi_full[3]   # K↑* · K'↓
    rho_cross[1, 0] = np.conj(psi_full[1]) * psi_full[2]   # K↓* · K'↑
    rho_cross[1, 1] = np.conj(psi_full[1]) * psi_full[3]   # K↓* · K'↓
    return rho_cross


def intervalley_coherence(psi_full, dxdy):
    """
    Spatially integrated intervalley coherence (scalar).

        |C| = | ∫ (ψ_K↑* ψ_K'↑ + ψ_K↓* ψ_K'↓) dx dy |

    This is the magnitude of the trace of the integrated cross density matrix,
    summed over the pseudospin-diagonal channels. It is the natural figure of
    merit for scalar intervalley coupling: at zero coupling and well-separated
    K, K' wave-packets, |C| ≈ 0; in a coherent KK' superposition with full
    spatial overlap, |C| ≈ √(P_K · P_Kp).

    Returns
    -------
    coh : float
    """
    s_up = (np.conj(psi_full[0]) * psi_full[2]).sum() * dxdy
    s_dn = (np.conj(psi_full[1]) * psi_full[3]).sum() * dxdy
    return float(np.abs(s_up + s_dn))


def valley_density_matrix(psi_full, dxdy):
    """
    Build the 2×2 reduced valley density matrix by tracing over the spatial
    and pseudospin degrees of freedom.

        ρ_v[K,  K ] = P_K
        ρ_v[K', K'] = P_K'
        ρ_v[K,  K'] = ∫ (ψ_K↑* ψ_K'↑ + ψ_K↓* ψ_K'↓) dx dy
        ρ_v[K', K ] = conj(ρ_v[K, K'])

    Returns
    -------
    rho_v : complex ndarray, shape (2, 2)
        Hermitian, trace = P_K + P_Kp.

    Use this for Bloch sphere visualization, gate fidelity calculations,
    and process tomography of the valley qubit.
    """
    P_K, P_Kp = valley_populations(psi_full, dxdy)
    s_up = complex((np.conj(psi_full[0]) * psi_full[2]).sum() * dxdy)
    s_dn = complex((np.conj(psi_full[1]) * psi_full[3]).sum() * dxdy)
    off = s_up + s_dn

    rho_v = np.array([[P_K, off],
                      [np.conj(off), P_Kp]], dtype=np.complex128)
    return rho_v


def valley_bloch_vector(psi_full, dxdy):
    """
    Compute the Bloch vector (rx, ry, rz) of the valley qubit, treating
    the valley index as a two-level system with the basis (|K⟩, |K'⟩).

    The Bloch vector is defined from the reduced valley density matrix:
        ρ_v = (I + r·σ) / 2  · trace(ρ_v)

    so that
        rx = 2 Re(ρ_v[K, K']) / Tr(ρ_v)
        ry = -2 Im(ρ_v[K, K']) / Tr(ρ_v)
        rz =  (P_K - P_K') / Tr(ρ_v)

    For a pure state |Ψ⟩ = α|K⟩ + β|K'⟩, the Bloch vector has |r| = 1
    and lies on the surface of the Bloch sphere. For a mixed state
    (e.g., the transmitted state after orbital decoherence as in v1.0
    section 3), |r| < 1 — the radius gives a direct measure of the
    gate purity.

    Returns
    -------
    rx, ry, rz : float
        The three Bloch vector components. Tr(ρ_v) = P_K + P_Kp ≤ 1
        (drains may have absorbed some probability).
    """
    rho_v = valley_density_matrix(psi_full, dxdy)
    tr = float(np.real(rho_v[0, 0] + rho_v[1, 1]))
    if tr < 1e-30:
        return 0.0, 0.0, 0.0
    rx = float(2.0 * np.real(rho_v[0, 1])) / tr
    ry = float(2.0 * np.imag(rho_v[0, 1])) / tr
    rz = float(np.real(rho_v[0, 0] - rho_v[1, 1])) / tr
    return rx, ry, rz
