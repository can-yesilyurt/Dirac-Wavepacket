#!/usr/bin/env python3
"""
Generate fig_arch.pdf — the package-architecture diagram for the
SoftwareX manuscript. Boxes-and-arrows layout showing the flow
from config through grid / wavepacket / potential builders into
the propagator and detector subsystems.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


def box(ax, xy, w, h, text, face="#f6f6f6", edge="#333333",
        fontsize=9, weight="normal", text_color="#111111"):
    """Rounded rectangle with centered text."""
    p = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color)


def arrow(ax, xy1, xy2, color="#333333", style="->"):
    a = FancyArrowPatch(
        xy1, xy2, arrowstyle=style, color=color,
        mutation_scale=12, linewidth=1.0,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main() -> None:
    out = Path(__file__).parent / "fig_arch.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- Column 1: inputs ----------
    box(ax, (0.1, 3.7), 1.8, 0.6, "YAML config", face="#eef4fb")
    box(ax, (0.1, 2.9), 1.8, 0.6, "Python SimConfig", face="#eef4fb")
    # bracket these together as "User input"
    ax.text(1.0, 4.55, "User input", ha="center", va="center",
            fontsize=9, color="#555555", style="italic")

    # ---------- Column 2: builders ----------
    box(ax, (2.6, 4.0), 1.8, 0.6, "Grid builder")
    box(ax, (2.6, 3.2), 1.8, 0.6, "Wavepacket")
    box(ax, (2.6, 2.4), 1.8, 0.6, "Potential\n(5 geometry types)",
        fontsize=8)
    box(ax, (2.6, 1.6), 1.8, 0.6, "Absorber / walls")

    # arrows from inputs to builders (fan-out)
    for ty in (4.3, 3.5, 2.7, 1.9):
        arrow(ax, (1.9, 3.7), (2.6, ty))
        arrow(ax, (1.9, 3.2), (2.6, ty))

    # ---------- Column 3: propagator ----------
    box(ax, (5.1, 3.2), 2.2, 0.8,
        "Propagator\n(split-operator FFT)",
        face="#fff3e0", weight="bold", fontsize=9)
    box(ax, (5.1, 1.6), 2.2, 0.8,
        "Coupled propagator\n(4-spinor, optional)",
        face="#fff3e0", fontsize=9)

    # builders into propagator
    for ty in (4.3, 3.5, 2.7):
        arrow(ax, (4.4, ty), (5.1, 3.6))
    arrow(ax, (4.4, 1.9), (5.1, 3.4))
    # coupling option
    arrow(ax, (6.2, 3.2), (6.2, 2.4), color="#888888", style="<->")
    ax.text(6.3, 2.8, "if U$_{KK'}$\n$>0$", fontsize=7, color="#555555",
            va="center", ha="left")

    # ---------- Column 4: detectors / outputs ----------
    box(ax, (8.0, 3.6), 1.8, 0.8,
        "Detector\n(T, R, $\\eta$ vs t)",
        face="#e8f5e9", fontsize=9)
    box(ax, (8.0, 2.4), 1.8, 0.8,
        "Snapshots\n($\\psi_K, \\psi_{K'}$)",
        face="#e8f5e9", fontsize=9)
    box(ax, (8.0, 1.2), 1.8, 0.8,
        "Reports / plots",
        face="#e8f5e9", fontsize=9)

    # propagator -> outputs
    arrow(ax, (7.3, 3.6), (8.0, 4.0))
    arrow(ax, (7.3, 3.4), (8.0, 2.8))
    arrow(ax, (7.3, 2.0), (8.0, 1.6))

    # column headers
    ax.text(1.0, 0.6, "Configuration", ha="center", va="center",
            fontsize=9, color="#333333", fontweight="bold")
    ax.text(3.5, 0.6, "Domain builders", ha="center", va="center",
            fontsize=9, color="#333333", fontweight="bold")
    ax.text(6.2, 0.6, "Evolution", ha="center", va="center",
            fontsize=9, color="#333333", fontweight="bold")
    ax.text(8.9, 0.6, "Observables", ha="center", va="center",
            fontsize=9, color="#333333", fontweight="bold")

    # light separators between columns
    for xs in (2.3, 4.75, 7.7):
        ax.plot([xs, xs], [1.0, 4.7],
                color="#dddddd", linewidth=0.8,
                linestyle=":", zorder=0)

    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
