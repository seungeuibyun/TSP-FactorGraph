"""IEEE-style figures for the continuous-traffic start-hour sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .ieee import (
    EXPECTED_METHODS,
    METHOD_LABELS,
    METHOD_STYLES,
    _boxed_legend,
    _configure_ieee_style,
    _fit_legend_band,
)


def _plot(frame: pd.DataFrame, column: str, ylabel: str, stem: str,
          output_dir: Path) -> None:
    _configure_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    adaptive = frame[frame["policy"] == "adaptive"]
    for method in EXPECTED_METHODS:
        part = adaptive[adaptive["method"] == method].sort_values("start_hour")
        if part.empty:
            continue
        ax.plot(
            part["start_hour"], part[column], label=METHOD_LABELS[method],
            markevery=1, markerfacecolor="white", markeredgewidth=0.75,
            **METHOD_STYLES[method],
        )
    ax.set_xlabel("Starting hour")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.grid(True, which="major", axis="both")
    ax.tick_params(direction="in", top=True, right=True, width=0.6)
    legend = _boxed_legend(
        ax, ncol=2, loc="upper left", handlelength=2.3,
        columnspacing=1.0)
    _fit_legend_band(ax, legend, adaptive[column].max())
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}", bbox_inches="tight",
            pad_inches=0.02)
    plt.close(fig)


def create_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    _plot(frame, "cumulative_energy_kwh", "Energy consumption (kWh)",
          "hourly_online_energy_ieee", output_dir)
    _plot(frame, "cumulative_operating_s", "Completion time (s)",
          "hourly_online_completion_time_ieee", output_dir)
