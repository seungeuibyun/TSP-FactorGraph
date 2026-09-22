"""Create IEEE-style hourly proposed-versus-baseline comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


EXPECTED_METHODS = ("proposed", "nn", "aco", "pso", "ga")
METHOD_LABELS = {
    "proposed": "Proposed",
    "nn": "NN",
    "aco": "ACO",
    "pso": "PSO",
    "ga": "GA",
}
METHOD_STYLES = {
    "proposed": {"marker": "*", "linestyle": "-", "color": "#000000"},
    "nn": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
    "aco": {"marker": "s", "linestyle": "--", "color": "#D55E00"},
    "pso": {"marker": "^", "linestyle": "-.", "color": "#009E73"},
    "ga": {"marker": "D", "linestyle": ":", "color": "#CC79A7"},
}
METRICS = {
    "energy": ("energy_kwh", "Energy consumption (kWh)"),
    "battery_remaining": (
        "battery_remaining_mean_kwh", "Mean remaining battery (kWh/vehicle)"),
    "makespan": ("makespan_s", "Makespan (s)"),
}


def _configure_ieee_style() -> None:
    """Apply a compact IEEE-compatible serif style with embedded TrueType text."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "grid.color": "0.88",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _boxed_legend(axis: plt.Axes, **kwargs: object):
    legend = axis.legend(
        frameon=True, fancybox=False, facecolor="white", edgecolor="0.15",
        framealpha=1.0, borderpad=0.45, **kwargs)
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
    return tuple(0.18 * component + 0.82 for component in rgb)


def _vehicle_time_summary(
    frame: pd.DataFrame,
    route_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_records = []
    observation_records = []
    missing = []
    for row in frame.itertuples(index=False):
        route_path = route_dir / f"{row.run_id}.json"
        if not route_path.exists():
            missing.append(route_path.name)
            continue
        payload = json.loads(route_path.read_text(encoding="utf-8"))
        times = np.asarray(payload["vehicle_time_s"], dtype=float)
        energies = np.asarray(payload["vehicle_energy_kwh"], dtype=float)
        routes = payload["routes"]
        if len(times) != len(routes) or len(energies) != len(routes):
            raise ValueError(
                f"{route_path.name} has inconsistent route/metric lengths")
        active_indices = [
            vehicle for vehicle, route in enumerate(routes) if len(route) > 0
        ]
        active_times = times[np.asarray(active_indices, dtype=int)]
        if not len(active_times):
            raise ValueError(f"{route_path.name} has no active vehicle time")
        expected_active = int(payload["run"]["active_vehicles"])
        if len(active_times) != expected_active:
            raise ValueError(
                f"{route_path.name} records {expected_active} active vehicles "
                f"but contains {len(active_times)} nonempty routes")
        if np.any(active_times <= 0.0):
            raise ValueError(
                f"{route_path.name} has a nonpositive active-vehicle time")
        active_energies = energies[np.asarray(active_indices, dtype=int)]
        if np.any(active_energies <= 0.0):
            raise ValueError(
                f"{route_path.name} has nonpositive active-vehicle energy")
        summary_records.append({
            "run_id": row.run_id,
            "method": row.method,
            "start_hour": int(row.start_hour),
            "active_vehicles": expected_active,
            "vehicle_time_mean_s": float(np.mean(active_times)),
            "vehicle_time_min_s": float(np.min(active_times)),
            "vehicle_time_max_s": float(np.max(active_times)),
        })
        observation_records.extend({
            "run_id": row.run_id,
            "method": row.method,
            "start_hour": int(row.start_hour),
            "vehicle": int(vehicle),
            "completion_time_s": float(times[vehicle]),
            "energy_kwh": float(energies[vehicle]),
        } for vehicle in active_indices)
    if missing:
        raise ValueError(
            "vehicle-time route details are incomplete; missing "
            f"{missing[:8]}" + (" ..." if len(missing) > 8 else ""))
    summary = pd.DataFrame(summary_records).sort_values(
        ["method", "start_hour"])
    observations = pd.DataFrame(observation_records).sort_values(
        ["method", "start_hour", "vehicle"])
    return summary, observations


def _draw_vehicle_time_distribution(
    summary: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in EXPECTED_METHODS:
        part = summary[summary["method"] == method].sort_values("start_hour")
        axis.plot(
            part["start_hour"], part["vehicle_time_mean_s"],
            label=METHOD_LABELS[method], markevery=1,
            markerfacecolor="white", markeredgewidth=0.75, zorder=2.0,
            **METHOD_STYLES[method],
        )
    axis.set_xlabel("Starting hour")
    axis.set_ylabel("Mean active-vehicle completion time (s)")
    axis.set_xlim(0, 23)
    axis.set_xticks([0, 4, 8, 12, 16, 20, 23])
    data_lower = float(summary["vehicle_time_mean_s"].min())
    data_upper = float(summary["vehicle_time_mean_s"].max())
    lower_padding = 0.04 * (data_upper - data_lower)
    axis.set_ylim(bottom=data_lower - lower_padding)
    axis.grid(True, which="major", axis="both")
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    legend = _boxed_legend(
        axis, ncol=2, loc="upper left", handlelength=2.3,
        columnspacing=1.0)
    _fit_legend_band(axis, legend, data_upper)
    paths = []
    for suffix in ("pdf", "eps", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def _draw_active_vehicle_cdf(
    observations: pd.DataFrame,
    column: str,
    xlabel: str,
    output_stem: Path,
) -> list[Path]:
    """Plot a smooth monotone CDF over all 24 hourly active-vehicle runs."""

    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in EXPECTED_METHODS:
        values = np.sort(observations.loc[
            observations["method"] == method, column
        ].to_numpy(dtype=float))
        if not len(values):
            raise ValueError(
                f"no active-vehicle {column} values are available for {method}")
        last_at_value = np.r_[
            np.flatnonzero(values[:-1] != values[1:]), len(values) - 1
        ]
        support = np.r_[0.0, values[last_at_value]]
        probability = np.r_[
            0.0, (last_at_value + 1).astype(float) / len(values)
        ]
        interpolator = PchipInterpolator(
            support, probability, extrapolate=False)
        grid = np.linspace(
            0.0, float(observations[column].max()) * 1.02, 1200)
        curve = np.where(
            grid <= support[-1], interpolator(np.minimum(grid, support[-1])),
            1.0)
        style = {
            key: value for key, value in METHOD_STYLES[method].items()
            if key != "marker"
        }
        axis.plot(
            grid, np.clip(curve, 0.0, 1.0),
            label=METHOD_LABELS[method], **style)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("CDF")
    axis.set_xlim(0.0, float(observations[column].max()) * 1.02)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.grid(True, which="major", axis="both")
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    _boxed_legend(
        axis, ncol=2, loc="upper left", handlelength=2.3,
        columnspacing=1.0)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def _common_kde_bandwidth(
    observations: pd.DataFrame,
    column: str,
) -> float:
    values = observations[column].to_numpy(dtype=float)
    deviation = float(np.std(values, ddof=1))
    if not np.isfinite(deviation) or deviation <= 0.0:
        raise ValueError(f"cannot estimate a KDE bandwidth for {column}")
    return deviation * len(values) ** (-1.0 / 5.0)


def _draw_active_vehicle_kde_cdf(
    observations: pd.DataFrame,
    column: str,
    xlabel: str,
    output_stem: Path,
) -> list[Path]:
    """Plot a boundary-corrected Gaussian KDE-CDF with common bandwidth."""

    bandwidth = _common_kde_bandwidth(observations, column)
    upper = float(observations[column].max()) + 4.0 * bandwidth
    grid = np.linspace(0.0, upper, 1200)
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in EXPECTED_METHODS:
        values = observations.loc[
            observations["method"] == method, column
        ].to_numpy(dtype=float)
        if not len(values):
            raise ValueError(
                f"no active-vehicle {column} values are available for {method}")
        scaled_values = values / bandwidth
        forward = ndtr(
            (grid[:, None] - values[None, :]) / bandwidth
        ) - ndtr(-scaled_values)[None, :]
        reflected = ndtr(
            (grid[:, None] + values[None, :]) / bandwidth
        ) - ndtr(scaled_values)[None, :]
        curve = np.mean(forward + reflected, axis=1)
        style = {
            key: value for key, value in METHOD_STYLES[method].items()
            if key != "marker"
        }
        axis.plot(
            grid, np.clip(curve, 0.0, 1.0),
            label=METHOD_LABELS[method], **style)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("CDF")
    axis.set_xlim(0.0, upper)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.grid(True, which="major", axis="both")
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    _boxed_legend(
        axis, ncol=2, loc="upper left", handlelength=2.3,
        columnspacing=1.0)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def _validated_frame(input_csv: Path | list[Path]) -> pd.DataFrame:
    paths = [input_csv] if isinstance(input_csv, Path) else list(input_csv)
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {
        "method", "start_hour", "energy_kwh",
        "battery_remaining_mean_kwh", "makespan_s", "feasible",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hourly results are missing columns: {sorted(missing)}")
    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    duplicates = frame.duplicated(["method", "start_hour"], keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, ["method", "start_hour"]].to_dict("records")
        raise ValueError(f"duplicate method/hour observations: {rows[:4]}")
    expected = {(method, hour) for method in EXPECTED_METHODS for hour in range(24)}
    observed = set(zip(frame["method"], frame["start_hour"].astype(int)))
    absent = sorted(expected - observed)
    if absent:
        raise ValueError(f"hourly experiment is incomplete; missing {absent[:8]}")
    if not frame["feasible"].astype(bool).all():
        failed = frame.loc[~frame["feasible"].astype(bool),
                           ["method", "start_hour"]].to_dict("records")
        raise ValueError(f"infeasible runs cannot be plotted: {failed}")
    return frame.sort_values(["method", "start_hour"])


def _draw_metric(frame: pd.DataFrame, column: str, ylabel: str,
                 output_stem: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in EXPECTED_METHODS:
        part = frame[frame["method"] == method].sort_values("start_hour")
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
    _fit_legend_band(ax, legend, frame[column].max())
    paths = []
    for suffix in ("pdf", "eps", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def _draw_combined(frame: pd.DataFrame, output_stem: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6.7), sharex=True,
                             constrained_layout=True)
    panel_labels = ("(a)", "(b)", "(c)")
    for ax, (metric, (column, ylabel)), panel in zip(
            axes, METRICS.items(), panel_labels):
        del metric
        for method in EXPECTED_METHODS:
            part = frame[frame["method"] == method].sort_values("start_hour")
            ax.plot(
                part["start_hour"], part[column], label=METHOD_LABELS[method],
                markerfacecolor="white", markeredgewidth=0.75,
                **METHOD_STYLES[method],
            )
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 23)
        ax.grid(True, which="major", axis="both")
        ax.tick_params(direction="in", top=True, right=True, width=0.6)
        ax.text(0.98, 0.93, panel, transform=ax.transAxes,
                ha="right", va="top")
    axes[-1].set_xlabel("Starting hour")
    axes[-1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    legend = _boxed_legend(
        axes[0], ncol=2, loc="upper left", handlelength=2.3,
        columnspacing=1.0)
    first_column = next(iter(METRICS.values()))[0]
    _fit_legend_band(axes[0], legend, frame[first_column].max())
    paths = []
    for suffix in ("pdf", "eps", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def create_figures(input_csv: Path | list[Path], output_dir: Path,
                   merged_output: Path | None = None,
                   route_dir: Path | None = None) -> list[Path]:
    """Validate hourly runs and export metric and vehicle-time figures."""

    _configure_ieee_style()
    frame = _validated_frame(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    if merged_output is not None:
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values(["start_hour", "method"]).to_csv(
            merged_output, index=False)
    outputs: list[Path] = []
    for name, (column, ylabel) in METRICS.items():
        outputs.extend(_draw_metric(
            frame, column, ylabel, output_dir / f"hourly_{name}_ieee"))
    outputs.extend(_draw_combined(frame, output_dir / "hourly_comparison_ieee"))
    vehicle_time_summary = None
    vehicle_time_observations = None
    if route_dir is not None:
        vehicle_time_summary, vehicle_time_observations = _vehicle_time_summary(
            frame, route_dir.resolve())
        vehicle_time_summary.to_csv(
            route_dir.resolve().parent
            / "hourly_vehicle_completion_time_summary.csv",
            index=False,
        )
        outputs.extend(_draw_vehicle_time_distribution(
            vehicle_time_summary,
            output_dir / "hourly_vehicle_completion_time_ieee"))
        outputs.extend(_draw_active_vehicle_cdf(
            vehicle_time_observations,
            "completion_time_s",
            "Active-vehicle completion time (s)",
            output_dir / "hourly_active_vehicle_completion_cdf_ieee"))
        outputs.extend(_draw_active_vehicle_cdf(
            vehicle_time_observations,
            "energy_kwh",
            "Active-vehicle energy consumption (kWh)",
            output_dir / "hourly_active_vehicle_energy_cdf_ieee"))
        outputs.extend(_draw_active_vehicle_kde_cdf(
            vehicle_time_observations,
            "completion_time_s",
            "Active-vehicle completion time (s)",
            output_dir / "hourly_active_vehicle_completion_kde_cdf_ieee"))
        outputs.extend(_draw_active_vehicle_kde_cdf(
            vehicle_time_observations,
            "energy_kwh",
            "Active-vehicle energy consumption (kWh)",
            output_dir / "hourly_active_vehicle_energy_kde_cdf_ieee"))

    summary = (frame.groupby("method", as_index=False)
               .agg(energy_mean_kwh=("energy_kwh", "mean"),
                    energy_std_kwh=("energy_kwh", "std"),
                    battery_remaining_mean_kwh=(
                        "battery_remaining_mean_kwh", "mean"),
                    battery_remaining_min_kwh=(
                        "battery_remaining_min_kwh", "min"),
                    makespan_mean_s=("makespan_s", "mean"),
                    makespan_std_s=("makespan_s", "std"),
                    feasible_rate=("feasible", "mean")))
    summary["method"] = pd.Categorical(
        summary["method"], categories=EXPECTED_METHODS, ordered=True)
    summary = summary.sort_values("method")
    if merged_output is not None:
        summary.to_csv(
            merged_output.with_name(f"{merged_output.stem}_summary.csv"),
            index=False,
        )

    manifest = {
        "source": ([_portable_path(path) for path in input_csv]
                   if isinstance(input_csv, list)
                   else _portable_path(input_csv)),
        "observations": int(len(frame)),
        "hours": list(range(24)),
        "methods": list(EXPECTED_METHODS),
        "battery_metric": (
            "mean across vehicles of initial battery minus route energy"),
        "formats": ["PDF", "EPS", "PNG"],
        "figure_width_in": 3.5,
        "legend": "boxed upper-left inside a reserved data-free y-axis band",
        "vehicle_time_figure": (
            "mean active-vehicle completion time"
            if vehicle_time_summary is not None else None),
        "vehicle_time_cdf": (
            "monotone interpolation of the pooled active-vehicle empirical "
            "CDF over the 24 hourly runs"
            if vehicle_time_observations is not None else None),
        "vehicle_time_cdf_samples_per_method": (
            int(len(vehicle_time_observations) / len(EXPECTED_METHODS))
            if vehicle_time_observations is not None else None),
        "vehicle_energy_cdf": (
            "monotone interpolation of the pooled active-vehicle energy "
            "empirical CDF over the 24 hourly runs"
            if vehicle_time_observations is not None else None),
        "vehicle_kde_cdfs": (
            "zero-boundary-reflected Gaussian KDE-CDFs using a common pooled "
            "Scott bandwidth for each metric"
            if vehicle_time_observations is not None else None),
        "vehicle_kde_common_bandwidth": ({
            "completion_time_s": _common_kde_bandwidth(
                vehicle_time_observations, "completion_time_s"),
            "energy_kwh": _common_kde_bandwidth(
                vehicle_time_observations, "energy_kwh"),
        } if vehicle_time_observations is not None else None),
        "raster_dpi": 600,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, nargs="+",
        default=[PACKAGE_ROOT / "results" / "hourly_24h"
                 / "hourly_dual_path_with_baselines_seed7.csv"])
    parser.add_argument(
        "--output-dir", type=Path,
        default=PACKAGE_ROOT / "figures" / "hourly_24h")
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--route-dir", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    for path in create_figures(
            args.input, args.output_dir, merged_output=args.merged_output,
            route_dir=args.route_dir):
        print(path)


if __name__ == "__main__":
    main()
