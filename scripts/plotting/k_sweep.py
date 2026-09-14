"""Create one IEEE-style exact-K comparison graph per figure file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .ieee import (
    EXPECTED_METHODS,
    METHOD_LABELS,
    METHOD_STYLES,
    _configure_ieee_style,
)


DEFAULT_HOURS = (4, 11, 19)
DEFAULT_VEHICLES = tuple(range(6, 15))
METRICS = {
    "energy": ("energy_kwh", "Energy consumption (kWh)"),
    "battery_remaining": (
        "battery_remaining_mean_kwh",
        "Mean remaining battery (kWh/vehicle)",
    ),
    "makespan": ("makespan_s", "Makespan (s)"),
}


def _validated_frame(
    path: Path,
    hours: tuple[int, ...],
    vehicles: tuple[int, ...],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "method", "start_hour", "vehicles", "energy_kwh",
        "battery_remaining_mean_kwh", "makespan_s", "feasible",
        "active_vehicles",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"K-sweep results are missing {sorted(missing)}")
    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    expected = {
        (method, hour, vehicle_count)
        for method in EXPECTED_METHODS
        for hour in hours
        for vehicle_count in vehicles
    }
    observed = set(zip(
        frame["method"], frame["start_hour"].astype(int),
        frame["vehicles"].astype(int),
    ))
    if observed != expected:
        raise ValueError(
            f"incomplete K sweep: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}")
    if frame.duplicated(["method", "start_hour", "vehicles"]).any():
        raise ValueError("duplicate method/hour/K observations")
    if not frame["feasible"].astype(bool).all():
        raise ValueError("infeasible observation in K sweep")
    if not (frame["active_vehicles"].astype(int)
            == frame["vehicles"].astype(int)).all():
        raise ValueError("active vehicle count differs from exact K")
    return frame.sort_values(["start_hour", "vehicles", "method"])


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _boxed_legend(axis: plt.Axes):
    legend = axis.legend(
        ncol=2, loc="upper left", frameon=True, fancybox=False,
        facecolor="white", edgecolor="0.15", framealpha=1.0,
        borderpad=0.45, handlelength=2.2, columnspacing=0.9,
    )
    legend.get_frame().set_linewidth(0.7)
    return legend


def _fit_legend_band(axis: plt.Axes, legend, data_upper: float) -> None:
    """Leave only a narrow data-free gap directly below the legend."""

    lower = axis.get_ylim()[0]
    for _ in range(2):
        axis.figure.canvas.draw()
        renderer = axis.figure.canvas.get_renderer()
        axes_box = axis.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        legend_bottom = (legend_box.y0 - axes_box.y0) / axes_box.height
        data_ceiling = legend_bottom - 0.025
        if data_ceiling <= 0.0:
            raise ValueError("legend leaves no usable plotting area")
        axis.set_ylim(
            lower, lower + (float(data_upper) - lower) / data_ceiling)


def _cloud_color(color: str) -> tuple[float, float, float]:
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(component + (1.0 - component) * 0.84 for component in rgb)


def _normalized_metric_summary(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    normalized = frame[[
        "method", "start_hour", "vehicles", column
    ]].copy()
    hourly_reference = normalized.groupby(
        "start_hour")[column].transform("min")
    normalized["normalized_gap_pct"] = 100.0 * (
        normalized[column] / hourly_reference - 1.0)
    return (normalized.groupby(["method", "vehicles"], as_index=False)
            .agg(
                hours=("start_hour", "count"),
                normalized_gap_mean_pct=(
                    "normalized_gap_pct", "mean"),
                normalized_gap_min_pct=(
                    "normalized_gap_pct", "min"),
                normalized_gap_max_pct=(
                    "normalized_gap_pct", "max"),
            ))


def _draw_normalized_metric(
    summary: pd.DataFrame,
    stem: Path,
    vehicles: tuple[int, ...],
    ylabel: str,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    tick_values = list(vehicles)
    if len(tick_values) > 6:
        tick_values = tick_values[::2]
        if tick_values[-1] != vehicles[-1]:
            tick_values.append(vehicles[-1])
    for method in EXPECTED_METHODS:
        part = summary[summary["method"] == method].sort_values("vehicles")
        band = axis.fill_between(
            part["vehicles"],
            part["normalized_gap_min_pct"],
            part["normalized_gap_max_pct"],
            facecolor=_cloud_color(METHOD_STYLES[method]["color"]),
            edgecolor="none", alpha=0.38, zorder=1,
        )
        band.set_rasterized(True)
        axis.plot(
            part["vehicles"], part["normalized_gap_mean_pct"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, zorder=2.0, **METHOD_STYLES[method],
        )
    axis.axhline(0.0, color="0.35", linewidth=0.6, zorder=0.4)
    axis.set_xlabel("Exact active vehicles, $K$")
    axis.set_ylabel(ylabel)
    axis.set_xticks(tick_values)
    axis.set_ylim(bottom=0.0)
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    legend = _boxed_legend(axis)
    _fit_legend_band(
        axis, legend, summary["normalized_gap_max_pct"].max())
    return _save(fig, stem)


def _draw_metric(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    stem: Path,
    hour: int,
    vehicles: tuple[int, ...],
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    tick_values = list(vehicles)
    if len(tick_values) > 6:
        tick_values = tick_values[::2]
        if tick_values[-1] != vehicles[-1]:
            tick_values.append(vehicles[-1])
    for method in EXPECTED_METHODS:
        part = frame[
            (frame["start_hour"].astype(int) == hour)
            & (frame["method"] == method)
        ].sort_values("vehicles")
        axis.plot(
            part["vehicles"], part[column],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, **METHOD_STYLES[method],
        )
    axis.set_title(f"{hour:02d}:00")
    axis.set_xlabel("Exact active vehicles, $K$")
    axis.set_ylabel(ylabel)
    axis.set_xticks(tick_values)
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    legend = _boxed_legend(axis)
    _fit_legend_band(axis, legend, frame.loc[
        frame["start_hour"].astype(int) == hour, column].max())
    return _save(fig, stem)


def create_figures(
    input_csv: Path,
    output_dir: Path,
    hours: tuple[int, ...] = DEFAULT_HOURS,
    vehicles: tuple[int, ...] = DEFAULT_VEHICLES,
) -> list[Path]:
    _configure_ieee_style()
    frame = _validated_frame(input_csv, hours, vehicles)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    normalized_energy = _normalized_metric_summary(frame, "energy_kwh")
    normalized_energy.to_csv(
        input_csv.with_name("k_sweep_normalized_energy_summary.csv"),
        index=False,
    )
    outputs.extend(_draw_normalized_metric(
        normalized_energy,
        output_dir / "k_sweep_normalized_energy_ieee",
        vehicles,
        "Energy gap to hourly best (%)",
    ))
    normalized_makespan = _normalized_metric_summary(frame, "makespan_s")
    normalized_makespan.to_csv(
        input_csv.with_name("k_sweep_normalized_makespan_summary.csv"),
        index=False,
    )
    outputs.extend(_draw_normalized_metric(
        normalized_makespan,
        output_dir / "k_sweep_normalized_makespan_ieee",
        vehicles,
        "Makespan gap to hourly best (%)",
    ))
    for hour in hours:
        for name, (column, ylabel) in METRICS.items():
            outputs.extend(_draw_metric(
                frame, column, ylabel,
                output_dir / f"k_sweep_h{hour:02d}_{name}_ieee",
                hour, vehicles))
    manifest = {
        "source": input_csv.resolve().relative_to(
            Path(__file__).resolve().parents[2]).as_posix(),
        "observations": int(len(frame)),
        "hours": list(hours),
        "vehicles": list(vehicles),
        "methods": list(EXPECTED_METHODS),
        "primary_figures": [
            "k_sweep_normalized_energy_ieee",
            "k_sweep_normalized_makespan_ieee",
        ],
        "normalization": (
            "100 * (metric / minimum metric over all methods and K within "
            "the same hour - 1), independently for energy and makespan"),
        "normalized_line": "arithmetic mean across the three hours",
        "normalized_cloud": "minimum-to-maximum across the three hours",
        "layout": (
            "one normalized primary graph plus single-hour supplementary "
            "graphs"),
        "legend": "boxed upper-left inside a reserved data-free y-axis band",
        "formats": ["PDF", "PNG"],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int, nargs="+", default=DEFAULT_HOURS)
    parser.add_argument(
        "--vehicles", type=int, nargs="+", default=DEFAULT_VEHICLES)
    args = parser.parse_args(arguments)
    for path in create_figures(
        args.input, args.output_dir, tuple(args.hours), tuple(args.vehicles)):
        print(path)


if __name__ == "__main__":
    main()
