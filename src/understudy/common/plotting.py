"""Shared chart styling (reference palette, light mode). Text never wears series colours."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Sequential ramp (one hue, light -> dark) for magnitudes such as confusion counts.
SEQUENTIAL = ["#f3f7fd", "#c9dcf5", "#8fb8ea", "#4f8fdc", "#2a78d6", "#1d5aa6", "#123d72"]


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the house style: recessive grid and axes, left-aligned title."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
