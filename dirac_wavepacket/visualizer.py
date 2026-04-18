"""
Visualizer module: render probability-density frames and compile a GIF animation.

Frames show:
  • |ψ(x,y,t)|² as false-colour imshow
  • Potential barrier region (blue overlay)
  • Drain contact regions (red/green shading at edges)
  • Optional current overlay: quiver arrows, streamlines, or both
  • Annotations: time, P, T, R, T+R+P

Dual-valley mode (valleys="both"):
  • Side-by-side panels: K (left) and K′ (right)
  • Shared colour scale, combined T/R report with valley polarisation η
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .wavepacket import probability_density
from .observables import current_plot_data
from .config import SimConfig


def _vmax_from_density(rho: np.ndarray, percentile: float = 99.5) -> float:
    v = float(np.percentile(rho, percentile))
    return max(v, 1e-12)


# ── shared drawing helpers ───────────────────────────────────────────────────


def _white_cmap(base_name: str):
    """Build a colormap that fades from white into the base colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(base_name)
    colors = base(np.linspace(0, 1, 256))
    n_fade = 13  # ~5% fade from white
    for i in range(n_fade):
        f = i / n_fade
        colors[i] = (1 - f) * np.array([1, 1, 1, 1]) + f * colors[i]
    return LinearSegmentedColormap.from_list("white_" + base_name, colors)


def _draw_barrier_outline(ax, cfg, scale=6.58212):
    """Draw barrier as a thin outline polygon (no fill)."""
    from matplotlib.patches import Polygon as MplPolygon
    from .potential import potential_polygon, potential_polygons

    ptype = cfg.potential.type
    edges = cfg.potential.edges
    Ly2 = cfg.grid.Ly / 2.0

    if ptype == "multi" and cfg.potential.barriers is not None:
        # Draw each sub-barrier outline
        from .grid import Grid
        grid = Grid(Nx=cfg.grid.Nx, Ny=cfg.grid.Ny,
                    Lx=cfg.grid.Lx, Ly=cfg.grid.Ly)
        polys = potential_polygons(cfg.potential, cfg.grid.Ly, grid=grid)
        for verts in polys:
            scaled = [(x * scale, y * scale) for x, y in verts]
            poly = MplPolygon(scaled, closed=True, fill=False,
                              edgecolor="#0077cc", linewidth=0.8, zorder=3)
            ax.add_patch(poly)

    elif ptype == "polygon" and cfg.potential.vertices is not None:
        from .grid import Grid
        grid = Grid(Nx=cfg.grid.Nx, Ny=cfg.grid.Ny,
                    Lx=cfg.grid.Lx, Ly=cfg.grid.Ly)
        verts = potential_polygon(cfg.potential, cfg.grid.Ly, grid=grid)
        if verts is not None:
            scaled = [(x * scale, y * scale) for x, y in verts]
            poly = MplPolygon(scaled, closed=True, fill=False,
                              edgecolor="#0077cc", linewidth=0.8, zorder=3)
            ax.add_patch(poly)

    elif edges is not None:
        xLb, xRb, xLt, xRt = edges
        verts = [
            (xLb * scale, -Ly2 * scale),
            (xRb * scale, -Ly2 * scale),
            (xRt * scale,  Ly2 * scale),
            (xLt * scale,  Ly2 * scale),
        ]
        poly = MplPolygon(verts, closed=True, fill=False,
                          edgecolor="#0077cc", linewidth=0.8, zorder=3)
        ax.add_patch(poly)

    elif ptype == "barrier":
        xc = cfg.potential.x_center
        hw = cfg.potential.width / 2.0 * scale
        ax.axvspan(xc - hw, xc + hw, alpha=0.1, color="#0077cc", zorder=2)


def _draw_regions(ax, cfg, extent, scale=6.58212):
    """Draw subtle drain and wall shading."""
    Lx, Ly = cfg.grid.Lx, cfg.grid.Ly
    Wx = cfg.absorber.width_frac * Lx * scale
    ax.axvspan(extent[0], extent[0] + Wx, alpha=0.06, color="#cc4444", zorder=1)
    ax.axvspan(extent[1] - Wx, extent[1], alpha=0.06, color="#44aa44", zorder=1)
    y_bc = cfg.absorber.y_bc
    if y_bc in ("reflecting", "absorbing"):
        yw_frac = (cfg.absorber.y_width_frac if cfg.absorber.y_width_frac > 0
                   else cfg.absorber.width_frac)
        Wy = yw_frac * Ly * scale
        clr = "#6666cc" if y_bc == "reflecting" else "#ccaa44"
        ax.axhspan(extent[2], extent[2] + Wy, alpha=0.06, color=clr, zorder=1)
        ax.axhspan(extent[3] - Wy, extent[3], alpha=0.06, color=clr, zorder=1)


def _plot_density(ax, rho, cfg, vmax):
    """Plot |ψ|² on an axes with potential and drain overlays. Returns im."""
    g = cfg.grid
    Lx2, Ly2 = g.Lx / 2.0, g.Ly / 2.0
    a00 = 6.582120
    extent = [-Lx2 * a00, Lx2 * a00, -Ly2 * a00, Ly2 * a00]
    cmap = _white_cmap(cfg.output.colormap)
    im = ax.imshow(
        rho, origin="lower", extent=extent, aspect="equal",
        cmap=cmap, vmin=0.0, vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_facecolor("white")
    if cfg.output.show_potential and cfg.potential.type != "free":
        _draw_barrier_outline(ax, cfg)
    _draw_regions(ax, cfg, extent)
    ax.set_xlabel(r"$x$ (nm)", fontsize=8)
    ax.set_ylabel(r"$y$ (nm)", fontsize=8)
    ax.tick_params(labelsize=6)
    return im


def _current_stride(cfg) -> int:
    s = int(getattr(cfg.output, "current_stride", 0) or 0)
    if s > 0:
        return s
    return max(1, int(np.ceil(max(cfg.grid.Nx, cfg.grid.Ny) / 48)))


def _current_style(cfg) -> str:
    raw = str(getattr(cfg.output, "current_style", "quiver") or "quiver").strip().lower()
    aliases = {
        "quiver": "quiver",
        "arrow": "quiver",
        "arrows": "quiver",
        "stream": "stream",
        "streamline": "stream",
        "streamlines": "stream",
        "both": "both",
    }
    return aliases.get(raw, "quiver")


def _draw_quiver(ax, X, Y, U, V, *, scale, color, alpha, width):
    U_plot = np.asarray(U, dtype=float)
    V_plot = np.asarray(V, dtype=float)
    U_plot = np.where(np.isfinite(U_plot), U_plot, np.nan)
    V_plot = np.where(np.isfinite(V_plot), V_plot, np.nan)
    if U_plot.size == 0:
        return

    ax.quiver(
        X, Y, U_plot, V_plot,
        angles="xy", scale_units="xy", scale=scale, pivot="mid",
        color="#08222c", alpha=min(1.0, alpha * 0.65),
        width=width * 2.25,
        minlength=0.0, minshaft=1.5,
        headwidth=5.8, headlength=7.0, headaxislength=6.2,
        zorder=4,
    )
    ax.quiver(
        X, Y, U_plot, V_plot,
        angles="xy", scale_units="xy", scale=scale, pivot="mid",
        color=color, alpha=alpha,
        width=width,
        minlength=0.0, minshaft=1.5,
        headwidth=4.8, headlength=6.0, headaxislength=5.2,
        zorder=5,
    )


def _draw_streamlines(ax, field, *, color, alpha, density, linewidth, arrowsize):
    from matplotlib.colors import to_rgba

    if field["U"].size == 0:
        return

    mask = np.asarray(field["mask"], dtype=bool) & np.isfinite(field["U"]) & np.isfinite(field["V"])
    if not np.any(mask):
        return

    rho = np.asarray(field["rho"], dtype=float)
    rho_local = np.where(mask, rho, np.nan)
    rho_max = max(float(np.nanmax(rho_local)), 1e-30)
    rho_norm = np.sqrt(np.clip(rho / rho_max, 0.0, 1.0))
    lw = linewidth * (0.55 + 0.95 * rho_norm)

    U_plot = np.where(mask, np.asarray(field["U"], dtype=float), np.nan)
    V_plot = np.where(mask, np.asarray(field["V"], dtype=float), np.nan)
    lw_plot = np.where(mask, np.asarray(lw, dtype=float), np.nan)

    kwargs = dict(
        density=max(0.25, float(density)),
        linewidth=lw_plot,
        arrowsize=max(0.45, float(arrowsize)),
        arrowstyle="-|>",
        minlength=0.10,
        integration_direction="both",
        zorder=4,
    )

    try:
        ax.streamplot(
            field["x"], field["y"], U_plot, V_plot,
            color=to_rgba("#08222c", min(1.0, alpha * 0.65)),
            linewidth=lw_plot * 2.0,
            arrowsize=max(0.55, float(arrowsize) * 1.10),
            arrowstyle="-|>",
            density=max(0.25, float(density)),
            minlength=0.10,
            integration_direction="both",
            zorder=4,
        )
        ax.streamplot(
            field["x"], field["y"], U_plot, V_plot,
            color=to_rgba(color, alpha),
            **kwargs,
        )
    except Exception:
        pass


def _add_current_overlay(ax, psi, cfg, *, wx: float | None = None, wy: float | None = None):
    if not getattr(cfg.output, "show_current", False):
        return

    stride = _current_stride(cfg)
    dx = float(cfg.grid.Lx) / float(cfg.grid.Nx)
    dy = float(cfg.grid.Ly) / float(cfg.grid.Ny)
    x_all = np.linspace(-float(cfg.grid.Lx) / 2.0, float(cfg.grid.Lx) / 2.0 - dx, int(cfg.grid.Nx))
    y_all = np.linspace(-float(cfg.grid.Ly) / 2.0, float(cfg.grid.Ly) / 2.0 - dy, int(cfg.grid.Ny))

    if wx is None:
        wx = float(cfg.physics.wx)
    if wy is None:
        wy = float(cfg.physics.wy)

    field = current_plot_data(
        psi,
        x_all,
        y_all,
        vf=cfg.physics.vf,
        wx=float(wx),
        wy=float(wy),
        stride=stride,
        density_frac=float(getattr(cfg.output, "current_min_density_frac", 0.02)),
        normalize=bool(getattr(cfg.output, "current_normalize", True)),
        block_average=True,
    )
    if field["mag_ref"] <= 0.0 or field["U"].size == 0:
        return

    style = _current_style(cfg)
    color = getattr(cfg.output, "current_color", "#00c8ff")
    alpha = float(getattr(cfg.output, "current_alpha", 0.95))

    if style in ("stream", "both"):
        _draw_streamlines(
            ax,
            field,
            color=color,
            alpha=alpha,
            density=float(getattr(cfg.output, "current_stream_density", 1.25)),
            linewidth=float(getattr(cfg.output, "current_stream_linewidth", 1.15)),
            arrowsize=float(getattr(cfg.output, "current_stream_arrowsize", 0.95)),
        )

    if style in ("quiver", "both"):
        X, Y = np.meshgrid(field["x"], field["y"], indexing="xy")
        spacing = stride * min(dx, dy)
        max_len = float(getattr(cfg.output, "current_max_arrow_length", 1.10))
        max_len = max(max_len, 1e-6) * spacing
        scale = field["mag_ref"] / max(max_len, 1e-12)
        _draw_quiver(
            ax,
            X,
            Y,
            field["U"],
            field["V"],
            scale=scale,
            color=color,
            alpha=alpha,
            width=float(getattr(cfg.output, "current_width", 0.0055)),
        )


def _plot_rate(ax, t_arr, T_arr, R_arr,
               label_suffix="", ls="-", alpha=1.0):
    """Plot instantaneous dT/dt and dR/dt on the given axes."""
    if len(t_arr) < 2:
        return
    dt_det = np.diff(t_arr)
    dT = np.diff(T_arr)
    dR = np.diff(R_arr)
    window = max(1, len(dT) // 50)
    if window > 1:
        kernel = np.ones(window) / window
        dT_s = np.convolve(dT / dt_det, kernel, mode="same")
        dR_s = np.convolve(dR / dt_det, kernel, mode="same")
    else:
        dT_s = dT / dt_det
        dR_s = dR / dt_det
    t_mid = 0.5 * (t_arr[:-1] + t_arr[1:])
    ax.plot(t_mid, dT_s, lw=1.0, color="#00bb44", ls=ls, alpha=alpha,
            label=f"dT/dt{label_suffix}")
    ax.plot(t_mid, dR_s, lw=1.0, color="#dd4400", ls=ls, alpha=alpha,
            label=f"dR/dt{label_suffix}")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Absorption rate")
    ax.grid(True, alpha=0.3)


# ── single-valley rendering ─────────────────────────────────────────────────
def render_frame(
    psi: np.ndarray,
    cfg: SimConfig,
    step: int,
    t: float,
    prob_total: float,
    vmax: Optional[float],
    frame_dir: Path,
    frame_idx: int,
    drains=None,
) -> float:
    """Render one single-valley frame (publication style). Returns vmax."""
    oc = cfg.output
    w = cfg.wavepacket
    g = cfg.grid
    Lx2, Ly2 = g.Lx / 2.0, g.Ly / 2.0
    extent = [-Lx2, Lx2, -Ly2, Ly2]

    rho = probability_density(psi)
    if vmax is None:
        vmax = _vmax_from_density(rho)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0), dpi=oc.dpi)
    fig.patch.set_facecolor("white")

    im = _plot_density(ax, rho, cfg, vmax)
    _add_current_overlay(ax, psi, cfg, wx=cfg.physics.wx, wy=cfg.physics.wy)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.88)
    cbar.set_label(r"$|\psi|^2$", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    xinfo = extent[0] + 0.03 * g.Lx
    yinfo = extent[2] + 0.04 * g.Ly

    # T/R info (bottom-left)
    if drains is not None:
        ax.text(xinfo * 6.582120, yinfo * 6.582120,
                f"T = {drains.T:.3f}   R = {drains.R:.3f}",
                fontsize=6.5, ha="left", va="bottom", color="#333333",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

    # Title
    Ef = cfg.physics.vf * w.k0 * 100
    V0 = cfg.potential.height * 100
    wy = cfg.physics.wy 
    fig.suptitle(
        rf"$E_\mathrm{{F}}$ = {Ef:.2f} meV   $V_0$ = {V0:.2f}  meV  "
        rf"$w_y$ = {wy} $v_\mathrm{{F}}$    $t$ = {t:.1f} ps",
        fontsize=8.5, y=0.97)

    fig.tight_layout()
    fig.savefig(frame_dir / f"frame_{frame_idx:05d}.png", dpi=oc.dpi,
                facecolor="white")
    plt.close(fig)
    return vmax


# ── dual-valley rendering ───────────────────────────────────────────────────

def render_frame_dual(
    psi_K: np.ndarray,
    psi_Kp: np.ndarray,
    cfg: SimConfig,
    step: int,
    t: float,
    P_K: float,
    P_Kp: float,
    vmax: Optional[float],
    frame_dir: Path,
    frame_idx: int,
    drains_K=None,
    drains_Kp=None,
) -> float:
    """Render one dual-valley frame (publication style). Returns vmax."""
    oc = cfg.output
    w = cfg.wavepacket
    g = cfg.grid
    Lx2, Ly2 = g.Lx / 2.0, g.Ly / 2.0
    extent = [-Lx2, Lx2, -Ly2, Ly2]

    rho_K  = probability_density(psi_K)
    rho_Kp = probability_density(psi_Kp)
    if vmax is None:
        vmax = max(_vmax_from_density(rho_K), _vmax_from_density(rho_Kp))

    fig, (ax_K, ax_Kp) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), dpi=oc.dpi,
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    im_K  = _plot_density(ax_K,  rho_K,  cfg, vmax)
    im_Kp = _plot_density(ax_Kp, rho_Kp, cfg, vmax)
    _add_current_overlay(ax_K, psi_K, cfg, wx=cfg.physics.wx, wy=cfg.physics.wy)
    _add_current_overlay(ax_Kp, psi_Kp, cfg, wx=-cfg.physics.wx, wy=-cfg.physics.wy)

    # Remove redundant y-axis on right panel
    ax_Kp.set_ylabel("")
    ax_Kp.set_yticklabels([])

    # Valley labels (top-left)
    x_lab = extent[0] + 0.03 * g.Lx
    y_lab = extent[3] - 0.04 * g.Ly
    box_kw = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2)
    ax_K.text(x_lab * 6.582120, y_lab * 6.582120, r"$K$", fontsize=11, fontweight="bold",
              ha="left", va="top", color="#222222", bbox=box_kw)
    ax_Kp.text(x_lab * 6.582120, y_lab * 6.582120, r"$K'$", fontsize=11, fontweight="bold",
               ha="left", va="top", color="#222222", bbox=box_kw)

    # T/R labels (bottom-left)
    x_info = extent[0] + 0.03 * g.Lx
    y_info = extent[2] + 0.04 * g.Ly
    info_kw = dict(fontsize=6.5, ha="left", va="bottom", color="#333333",
                   bbox=dict(facecolor="white", alpha=0.85,
                             edgecolor="none", pad=1.5))
    if drains_K is not None:
        ax_K.text(x_info * 6.582120, y_info * 6.582120,
                  f"T = {drains_K.T:.3f}   R = {drains_K.R:.3f}", **info_kw)
    if drains_Kp is not None:
        ax_Kp.text(x_info * 6.582120, y_info * 6.582120,
                   f"T = {drains_Kp.T:.3f}   R = {drains_Kp.R:.3f}", **info_kw)

    # Colorbar (right side)
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.08, right=0.82,
                        wspace=0.08)
    cbar_ax = fig.add_axes([0.84, 0.14, 0.02, 0.76])
    cbar = fig.colorbar(im_Kp, cax=cbar_ax)
    cbar.set_label(r"$|\psi|^2$", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # Title: shared parameters
    Ef = cfg.physics.vf * w.k0 * 100
    V0 = cfg.potential.height * 100
    wy = cfg.physics.wy
    fig.suptitle(
        rf"$E_\mathrm{{F}}$ = {Ef:.2f} meV   $V_0$ = {V0:.2f}  meV  "
        rf"$w_y$ = {wy} $v_\mathrm{{F}}$    $t$ = {t:.2f} ps",
        fontsize=8.5, y=0.97)

    fig.savefig(frame_dir / f"frame_{frame_idx:05d}.png", dpi=oc.dpi,
                facecolor="white")
    plt.close(fig)
    return vmax


# ── GIF compilation ──────────────────────────────────────────────────────────

def make_gif(frame_dir: Path, output_path: Path, fps: int = 15) -> None:
    from PIL import Image
    frames_sorted = sorted(frame_dir.glob("frame_*.png"))
    if not frames_sorted:
        print("  [warn] No frames found for animation.")
        return
    duration_ms = int(1000 / fps)
    images = [Image.open(str(f)).convert("RGBA") for f in frames_sorted]
    images[0].save(
        str(output_path), save_all=True, append_images=images[1:],
        optimize=False, duration=duration_ms, loop=0,
    )
    print(f"  ✓ Animation saved → {output_path}  ({len(images)} frames, {fps} fps)")


# ── probability history ──────────────────────────────────────────────────────

def plot_probability_history(times, probs, cfg, output_path,
                             probs_Kp=None) -> None:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=cfg.output.dpi)
    if probs_Kp is not None:
        ax.plot(times, probs,    lw=1.5, color="steelblue", label="K")
        ax.plot(times, probs_Kp, lw=1.5, color="coral",     label="K′")
        ax.legend(fontsize=8)
    else:
        ax.plot(times, probs, lw=1.5, color="steelblue")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Total probability P(t)")
    ax.set_title(f"{cfg.title} — probability in domain")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=cfg.output.dpi)
    plt.close(fig)
    print(f"  ✓ Probability history → {output_path}")


# ── transmission report (single valley) ─────────────────────────────────────

def plot_transmission_report(drains, cfg, output_path) -> None:
    """Plot T(t) and R(t) from drain contact histories."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=cfg.output.dpi)

    t_arr = np.array(drains.times)
    T_arr = np.array(drains.T_history)
    R_arr = np.array(drains.R_history)

    ax = axes[0]
    ax.plot(t_arr, T_arr, lw=1.5, color="#00bb44", label=f"T = {T_arr[-1]:.4f}")
    ax.plot(t_arr, R_arr, lw=1.5, color="#dd4400", label=f"R = {R_arr[-1]:.4f}")
    ax.plot(t_arr, T_arr + R_arr, lw=1.0, color="gray", ls="--",
            label=f"T+R = {T_arr[-1] + R_arr[-1]:.4f}")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Cumulative drain absorption")
    ax.set_title(f"T & R  (θ = {cfg.wavepacket.theta:.0f}°, drain contacts)")
    ax.legend(fontsize=8, loc="center right")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    _plot_rate(axes[1], t_arr, T_arr, R_arr)
    axes[1].set_title("Instantaneous drain absorption rate")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=cfg.output.dpi)
    plt.close(fig)
    print(f"  ✓ Transmission report → {output_path}")


# ── transmission report (dual valley) ───────────────────────────────────────

def plot_transmission_report_dual(drains_K, drains_Kp, cfg, output_path) -> None:
    """Plot dual-valley T/R histories + valley polarisation η(t)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=cfg.output.dpi)

    t_K  = np.array(drains_K.times)
    T_K  = np.array(drains_K.T_history)
    R_K  = np.array(drains_K.R_history)
    t_Kp = np.array(drains_Kp.times)
    T_Kp = np.array(drains_Kp.T_history)
    R_Kp = np.array(drains_Kp.R_history)

    # Panel 1: cumulative T and R for both valleys
    ax = axes[0]
    ax.plot(t_K,  T_K,  lw=1.5, color="#00bb44",
            label=f"T(K) = {T_K[-1]:.4f}")
    ax.plot(t_Kp, T_Kp, lw=1.5, color="#00bb44", ls="--",
            label=f"T(K′) = {T_Kp[-1]:.4f}")
    ax.plot(t_K,  R_K,  lw=1.5, color="#dd4400",
            label=f"R(K) = {R_K[-1]:.4f}")
    ax.plot(t_Kp, R_Kp, lw=1.5, color="#dd4400", ls="--",
            label=f"R(K′) = {R_Kp[-1]:.4f}")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Cumulative drain absorption")
    ax.set_title(f"K vs K′  (θ = {cfg.wavepacket.theta:.0f}°)")
    ax.legend(fontsize=7, loc="center right")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel 2: valley polarisation η(t) = (T_K − T_K′) / (T_K + T_K′)
    ax2 = axes[1]
    n = min(len(T_K), len(T_Kp))
    t_common = t_K[:n]
    T_sum = T_K[:n] + T_Kp[:n]
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(T_sum > 1e-8, (T_K[:n] - T_Kp[:n]) / T_sum, 0.0)
    ax2.plot(t_common, eta, lw=1.5, color="#8855dd")
    ax2.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Time (a.u.)")
    ax2.set_ylabel("η(t)")
    eta_final = float(eta[-1]) if len(eta) > 0 else 0.0
    ax2.set_title(f"Valley polarisation  η = {eta_final:+.4f}")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # Panel 3: instantaneous rates
    ax3 = axes[2]
    _plot_rate(ax3, t_K, T_K, R_K, label_suffix=" K")
    _plot_rate(ax3, t_Kp, T_Kp, R_Kp, label_suffix=" K′", ls="--", alpha=0.7)
    ax3.set_title("Instantaneous drain absorption rate")
    ax3.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=cfg.output.dpi)
    plt.close(fig)
    print(f"  ✓ Dual transmission report → {output_path}")


# ── final snapshot ───────────────────────────────────────────────────────────

def plot_final_snapshot(psi, cfg, t, output_path) -> None:
    g, oc = cfg.grid, cfg.output
    rho = probability_density(psi)
    vmax = _vmax_from_density(rho)

    fig, ax = plt.subplots(figsize=tuple(oc.figsize), dpi=oc.dpi * 1.5)
    im = _plot_density(ax, rho, cfg, vmax)
    _add_current_overlay(ax, psi, cfg, wx=cfg.physics.wx, wy=cfg.physics.wy)
    fig.colorbar(im, ax=ax, shrink=0.92, label=r"$|\psi(x,y,t_{\rm final})|^2$")

    Ef = cfg.physics.vf * cfg.wavepacket.k0
    ax.set_title(
        f"{cfg.title}   t_final = {t:.2f}   "
        f"θ = {cfg.wavepacket.theta:.0f}°   Ef = {Ef:.1f}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=oc.dpi * 1.5)
    plt.close(fig)
    print(f"  ✓ Final snapshot    → {output_path}")


def plot_final_snapshot_dual(psi_K, psi_Kp, cfg, t, output_path) -> None:
    g, oc = cfg.grid, cfg.output
    rho_K  = probability_density(psi_K)
    rho_Kp = probability_density(psi_Kp)
    vmax = max(_vmax_from_density(rho_K), _vmax_from_density(rho_Kp))

    fw, fh = oc.figsize
    fig, (ax_K, ax_Kp) = plt.subplots(
        1, 2, figsize=(fw * 1.7, fh), dpi=oc.dpi * 1.5,
        gridspec_kw={"wspace": 0.25},
        layout="constrained",
    )
    im_K  = _plot_density(ax_K,  rho_K,  cfg, vmax)
    im_Kp = _plot_density(ax_Kp, rho_Kp, cfg, vmax)
    _add_current_overlay(ax_K, psi_K, cfg, wx=cfg.physics.wx, wy=cfg.physics.wy)
    _add_current_overlay(ax_Kp, psi_Kp, cfg, wx=-cfg.physics.wx, wy=-cfg.physics.wy)

    wx, wy = cfg.physics.wx, cfg.physics.wy
    ax_K.set_title(f"K  (w=[{wx},{wy}])", fontsize=10)
    ax_Kp.set_title(f"K′  (w=[{-wx},{-wy}])", fontsize=10)

    fig.colorbar(im_Kp, ax=[ax_K, ax_Kp], shrink=0.92,
                 label=r"$|\psi(x,y,t_{\rm final})|^2$")

    Ef = cfg.physics.vf * cfg.wavepacket.k0
    fig.suptitle(
        f"{cfg.title}   t_final = {t:.2f}   "
        f"θ = {cfg.wavepacket.theta:.0f}°   Ef = {Ef:.1f}",
        fontsize=10, y=1.01)
    fig.savefig(str(output_path), dpi=oc.dpi * 1.5)
    plt.close(fig)
    print(f"  ✓ Dual final snapshot → {output_path}")


# ── phase / group delay plots ────────────────────────────────────────────────

def plot_phase_report_dual(tracker_K_summary, tracker_Kp_summary,
                           valley_phase, cfg, output_path) -> None:
    """
    Plot dual-valley phase comparison: centroid, carrier phase, and group delay.

    Parameters
    ----------
    tracker_K_summary  : dict from PhaseTracker.summary() for K
    tracker_Kp_summary : dict from PhaseTracker.summary() for K′
    valley_phase       : dict from valley_phase_difference()
    cfg                : SimConfig
    output_path        : Path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=cfg.output.dpi)

    t_K  = tracker_K_summary["times"]
    t_Kp = tracker_Kp_summary["times"]
    gd_K  = valley_phase["gd_K"]
    gd_Kp = valley_phase["gd_Kp"]

    # ── Panel (0,0): centroid ⟨x⟩(t) ─────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_K,  tracker_K_summary["centroid_x"],  lw=1.5,
            color="steelblue", label="⟨x⟩  K")
    ax.plot(t_Kp, tracker_Kp_summary["centroid_x"], lw=1.5,
            color="coral", label="⟨x⟩  K′")
    # Mark barrier position
    ax.axhline(cfg.potential.x_center, color="gray", lw=0.8, ls=":",
               alpha=0.5, label="barrier")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("⟨x⟩ (a.u.)")
    ax.set_title("Wavepacket centroid")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel (0,1): carrier phase (barrier-induced) ───────────────────
    ax = axes[0, 1]

    # Full-domain carrier phase — the gate observable
    phi_K_full  = tracker_K_summary["carrier_phase"]
    phi_Kp_full = tracker_Kp_summary["carrier_phase"]
    amp_K_full  = tracker_K_summary["carrier_amplitude"]
    amp_Kp_full = tracker_Kp_summary["carrier_amplitude"]

    # Post-barrier amplitude (for timing reference only)
    amp_K_post  = tracker_K_summary["carrier_amplitude_post"]
    amp_Kp_post = tracker_Kp_summary["carrier_amplitude_post"]

    # Mask where amplitude is too low
    amp_max_full = max(np.nanmax(amp_K_full), np.nanmax(amp_Kp_full), 1e-30)
    threshold_full = 0.05 * amp_max_full
    valid_K_full  = amp_K_full  > threshold_full
    valid_Kp_full = amp_Kp_full > threshold_full

    # Post-barrier amplitude threshold (for Δφ timing)
    amp_max_post = max(
        np.nanmax(amp_K_post) if np.any(~np.isnan(amp_K_post)) else 0,
        np.nanmax(amp_Kp_post) if np.any(~np.isnan(amp_Kp_post)) else 0,
        1e-30)
    threshold_post = 0.05 * amp_max_post
    valid_K_post  = (~np.isnan(amp_K_post))  & (amp_K_post  > threshold_post)
    valid_Kp_post = (~np.isnan(amp_Kp_post)) & (amp_Kp_post > threshold_post)

    # Plot full-domain carrier phase — bold
    phi_K_full_plot  = np.where(valid_K_full,  phi_K_full,  np.nan)
    phi_Kp_full_plot = np.where(valid_Kp_full, phi_Kp_full, np.nan)
    ax.plot(t_K,  phi_K_full_plot,  lw=1.8, color="steelblue",
            label=r"$\varphi$ K")
    ax.plot(t_Kp, phi_Kp_full_plot, lw=1.8, color="coral",
            label=r"$\varphi$ K$'$")

    # Mark plateau windows on the full-domain curve
    mask_K  = valley_phase.get("plateau_mask_K")
    mask_Kp = valley_phase.get("plateau_mask_Kp")
    if mask_K is not None and np.any(mask_K):
        ax.scatter(t_K[mask_K], phi_K_full[mask_K], s=18, color="steelblue",
                   zorder=5, alpha=0.7, edgecolors="none",
                   label="plateau K")
    if mask_Kp is not None and np.any(mask_Kp):
        ax.scatter(t_Kp[mask_Kp], phi_Kp_full[mask_Kp], s=18, color="coral",
                   zorder=5, alpha=0.7, edgecolors="none",
                   label="plateau K′")

    # Horizontal lines at the extracted gate phase values
    phi_K_gate  = valley_phase.get("gate_phase_K", np.nan)
    phi_Kp_gate = valley_phase.get("gate_phase_Kp", np.nan)
    if not np.isnan(phi_K_gate):
        ax.axhline(phi_K_gate, color="steelblue", lw=1.0, ls=":",
                   alpha=0.6)
    if not np.isnan(phi_Kp_gate):
        ax.axhline(phi_Kp_gate, color="coral", lw=1.0, ls=":",
                   alpha=0.6)

    # Phase difference where both post-barrier amplitudes are valid
    n = min(len(t_K), len(t_Kp))
    both_valid = valid_K_post[:n] & valid_Kp_post[:n]
    dphi_full = np.where(both_valid, phi_K_full[:n] - phi_Kp_full[:n], np.nan)
    ax.plot(t_K[:n], dphi_full, lw=1.2, color="#8855dd", ls="--",
            label="Δφ = φ_K − φ_K′")

    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Carrier phase at k₀\n(barrier-induced, free-prop subtracted)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel (1,0): group delay from drains ──────────────────────────
    ax = axes[1, 0]
    # Re-plot the dT/dt curves with group delay markers
    from .absorber import DrainContacts  # type hint only; we use arrays
    _draw_delay_panel(ax, valley_phase, cfg)

    # ── Panel (1,1): summary text box ─────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    dtau = valley_phase["delta_tau_T"]
    Ef = valley_phase["Ef"]
    wx, wy = cfg.physics.wx, cfg.physics.wy

    # Barrier-induced gate phase from carrier tracking
    gate = valley_phase.get("gate_phase", np.nan)
    method = valley_phase.get("gate_phase_method", "unavailable")
    phi_K_gate  = valley_phase.get("gate_phase_K", np.nan)
    phi_Kp_gate = valley_phase.get("gate_phase_Kp", np.nan)

    lines = [
        f"Valley Phase Gate Summary",
        f"{'─' * 40}",
        f"Tilt:   w = [{wx}, {wy}]",
        f"θ = {cfg.wavepacket.theta:.1f}°    Ef = {Ef:.2f}",
        f"",
        f"Group delay (transmitted):",
        f"  τ_T(K)   = {gd_K['tau_T']:.3f}  (peak {gd_K['tau_T_peak']:.3f})",
        f"  τ_T(K′)  = {gd_Kp['tau_T']:.3f}  (peak {gd_Kp['tau_T_peak']:.3f})",
        f"  Δτ_T (peak) = {dtau:+.4f} a.u.",
    ]

    if method == "carrier":
        gate_deg = np.degrees(gate)
        gate_pi = gate / np.pi
        lines += [
            f"",
            f"Barrier-induced gate phase:",
            f"  φ(K)     = {phi_K_gate:+.4f} rad",
            f"  φ(K′)    = {phi_Kp_gate:+.4f} rad",
            f"  Δφ_gate  = {gate:+.4f} rad",
            f"           = {gate_deg:+.2f}°",
            f"           = {gate_pi:+.4f}π",
        ]
    else:
        lines += [
            f"",
            f"Gate phase: insufficient",
            f"  post-barrier amplitude too low",
        ]

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                      edgecolor="#cccccc", alpha=0.9))

    fig.suptitle(f"{cfg.title} — Phase Analysis", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=cfg.output.dpi)
    plt.close(fig)
    print(f"  ✓ Phase report       → {output_path}")


def _draw_delay_panel(ax, valley_phase, cfg):
    """Draw the group delay panel: dT/dt for K and K′ with τ markers."""
    gd_K  = valley_phase["gd_K"]
    gd_Kp = valley_phase["gd_Kp"]

    # We need to recompute dT/dt from the drain histories stored in the
    # phase_data — but we don't store raw drains here. Instead, accept
    # that this is called from plot_phase_report_dual which receives
    # the drains through valley_phase_difference().
    # For now, draw the summary markers on a schematic timeline.
    tau_K  = gd_K["tau_T"]
    tau_Kp = gd_Kp["tau_T"]
    sig_K  = gd_K["sigma_T"]
    sig_Kp = gd_Kp["sigma_T"]

    # Draw Gaussian arrival distributions
    if not np.isnan(tau_K) and not np.isnan(sig_K) and sig_K > 0:
        t_range = np.linspace(tau_K - 4 * sig_K, tau_K + 4 * sig_K, 200)
        g_K = np.exp(-0.5 * ((t_range - tau_K) / sig_K) ** 2)
        ax.fill_between(t_range, g_K, alpha=0.3, color="steelblue")
        ax.plot(t_range, g_K, lw=1.5, color="steelblue", label="T arrival K")
        ax.axvline(tau_K, color="steelblue", lw=1.2, ls="--", alpha=0.7)

    if not np.isnan(tau_Kp) and not np.isnan(sig_Kp) and sig_Kp > 0:
        t_range_p = np.linspace(tau_Kp - 4 * sig_Kp, tau_Kp + 4 * sig_Kp, 200)
        g_Kp = np.exp(-0.5 * ((t_range_p - tau_Kp) / sig_Kp) ** 2)
        ax.fill_between(t_range_p, g_Kp, alpha=0.3, color="coral")
        ax.plot(t_range_p, g_Kp, lw=1.5, color="coral", label="T arrival K′")
        ax.axvline(tau_Kp, color="coral", lw=1.2, ls="--", alpha=0.7)

    # Annotate Δτ
    if not np.isnan(tau_K) and not np.isnan(tau_Kp):
        y_arrow = 0.5
        ax.annotate("", xy=(tau_K, y_arrow), xytext=(tau_Kp, y_arrow),
                     arrowprops=dict(arrowstyle="<->", color="#8855dd", lw=1.5))
        dtau = tau_K - tau_Kp
        ax.text(0.5 * (tau_K + tau_Kp), y_arrow + 0.08,
                f"Δτ = {dtau:+.2f}", ha="center", fontsize=8, color="#8855dd")

    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Arrival distribution (normalised)")
    ax.set_title("Transmitted group delay")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.2)


def plot_phase_report_single(tracker_summary, gd, cfg, output_path) -> None:
    """
    Plot single-valley phase analysis: centroid, carrier phase, group delay.

    Parameters
    ----------
    tracker_summary : dict from PhaseTracker.summary()
    gd              : dict from group_delay_from_drains()
    cfg             : SimConfig
    output_path     : Path
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=cfg.output.dpi)

    t = tracker_summary["times"]

    # Centroid
    ax = axes[0]
    ax.plot(t, tracker_summary["centroid_x"], lw=1.5, color="steelblue",
            label="⟨x⟩")
    ax.axhline(cfg.potential.x_center, color="gray", lw=0.8, ls=":",
               alpha=0.5, label="barrier")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("⟨x⟩ (a.u.)")
    ax.set_title("Wavepacket centroid")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Carrier phase (full-domain — barrier-induced)
    ax = axes[1]
    phi_full = tracker_summary["carrier_phase"]
    amp_full = tracker_summary["carrier_amplitude"]
    amp_post = tracker_summary["carrier_amplitude_post"]

    thresh_full = 0.05 * max(np.nanmax(amp_full), 1e-30)
    phi_full_plot = np.where(amp_full > thresh_full, phi_full, np.nan)

    # Full-domain phase: faint before wavepacket exits, bold after
    thresh_post = 0.05 * max(
        np.nanmax(amp_post) if np.any(~np.isnan(amp_post)) else 0, 1e-30)
    post_valid = (~np.isnan(amp_post)) & (amp_post > thresh_post)

    ax.plot(t, phi_full_plot, lw=0.7, color="steelblue", alpha=0.3)
    phi_full_post = np.where(post_valid, phi_full, np.nan)
    ax.plot(t, phi_full_post, lw=1.8, color="steelblue",
            label="φ (barrier-induced)")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Carrier phase at k₀\n(barrier-induced, free-prop subtracted)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Group delay summary
    ax = axes[2]
    ax.axis("off")
    lines = [
        f"Group Delay (drain)",
        f"{'─' * 30}",
        f"τ_T = {gd['tau_T']:.3f} a.u.",
        f"τ_T_peak = {gd['tau_T_peak']:.3f}",
        f"σ_T = {gd['sigma_T']:.3f}",
        f"τ_R = {gd['tau_R']:.3f} a.u.",
        f"σ_R = {gd['sigma_R']:.3f}",
    ]
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                      edgecolor="#cccccc", alpha=0.9))

    fig.suptitle(f"{cfg.title} — Phase Analysis", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=cfg.output.dpi)
    plt.close(fig)
    print(f"  ✓ Phase report       → {output_path}")
