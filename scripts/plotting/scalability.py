"""Create IEEE-style energy and service-time scalability figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ieee import (
    METHOD_LABELS,
    METHOD_STYLES,
    _boxed_legend,
    _configure_ieee_style,
    _fit_legend_band,
    _portable_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_ORDER = ("proposed", "nn", "aco", "pso", "ga")


def _cloud_color(color: str) -> tuple[float, float, float]:
    """Match the pale uncertainty bands used by the online figures."""

    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * 0.84)


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    try:
        for suffix in ("pdf", "png"):
            path = stem.with_suffix(f".{suffix}")
            temporary = stem.with_name(
                f".{stem.name}.tmp").with_suffix(f".{suffix}")
            try:
                fig.savefig(
                    temporary, bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if suffix == "png" else None)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(path)
    finally:
        plt.close(fig)
    return outputs


def _load(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_csv)
    required = {
        "method", "n_bins", "vehicles", "runtime_s", "oracle_prep_s",
        "total_runtime_s", "energy_kwh", "energy_per_bin_kwh",
        "makespan_s", "feasible", "assignment_edges",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scalability results are missing {sorted(missing)}")
    frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
    if frame.empty:
        raise ValueError("scalability results contain no supported methods")
    frame["feasible"] = frame["feasible"].astype(str).str.lower().eq("true")
    valid = frame[frame["feasible"]].copy()
    if valid.empty:
        raise ValueError("no feasible scalability observation is available")
    summary = (valid.groupby(
        ["method", "n_bins", "vehicles", "assignment_edges"],
        as_index=False)
        .agg(
            runs=("run_id", "count"),
            solver_runtime_s=("runtime_s", "mean"),
            solver_runtime_std_s=("runtime_s", "std"),
            oracle_runtime_s=("oracle_prep_s", "mean"),
            total_runtime_s=("total_runtime_s", "mean"),
            total_runtime_std_s=("total_runtime_s", "std"),
            total_energy_kwh=("energy_kwh", "mean"),
            total_energy_std_kwh=("energy_kwh", "std"),
            energy_per_bin_kwh=("energy_per_bin_kwh", "mean"),
            energy_per_bin_std_kwh=("energy_per_bin_kwh", "std"),
            makespan_s=("makespan_s", "mean"),
            makespan_std_s=("makespan_s", "std"),
        ))
    return frame, summary


def _method_sequence(frame: pd.DataFrame) -> list[str]:
    observed = set(frame["method"])
    return [method for method in METHOD_ORDER if method in observed]


def _place_legend(axis: plt.Axes, data_upper: float,
                  reserve_band: bool = True) -> None:
    legend = _boxed_legend(
        axis, loc="upper right", ncol=2, handlelength=2.0,
        columnspacing=0.9)
    if reserve_band:
        _fit_legend_band(axis, legend, data_upper)


def _energy_figure(summary: pd.DataFrame, methods: list[str],
                   stem: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in methods:
        part = summary[summary["method"] == method].sort_values("n_bins")
        axis.plot(
            part["n_bins"], part["energy_per_bin_kwh"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, **METHOD_STYLES[method])
        deviations = part["energy_per_bin_std_kwh"].fillna(0.0)
        if np.any(deviations.to_numpy() > 0.0):
            band = axis.fill_between(
                part["n_bins"],
                part["energy_per_bin_kwh"] - deviations,
                part["energy_per_bin_kwh"] + deviations,
                facecolor=_cloud_color(METHOD_STYLES[method]["color"]),
                edgecolor="none", alpha=0.38, zorder=1)
            band.set_rasterized(True)
    axis.set_xlabel("Number of bins, $N$")
    axis.set_ylabel("Energy per serviced bin (kWh/bin)")
    axis.set_axisbelow(True)
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    deviations = summary["energy_per_bin_std_kwh"].fillna(0.0)
    data_lower = float(
        (summary["energy_per_bin_kwh"] - deviations).min())
    data_upper = float(
        (summary["energy_per_bin_kwh"] + deviations).max())
    padding = 0.035 * (data_upper - data_lower)
    axis.set_ylim(data_lower - padding, data_upper + padding)
    _place_legend(
        axis,
        data_upper,
        reserve_band=False,
    )
    return _save(fig, stem)


def _total_energy_figure(summary: pd.DataFrame, methods: list[str],
                         stem: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in methods:
        part = summary[summary["method"] == method].sort_values("n_bins")
        axis.plot(
            part["n_bins"], part["total_energy_kwh"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, **METHOD_STYLES[method])
        deviations = part["total_energy_std_kwh"].fillna(0.0)
        if np.any(deviations.to_numpy() > 0.0):
            band = axis.fill_between(
                part["n_bins"],
                part["total_energy_kwh"] - deviations,
                part["total_energy_kwh"] + deviations,
                facecolor=_cloud_color(METHOD_STYLES[method]["color"]),
                edgecolor="none", alpha=0.38, zorder=1)
            band.set_rasterized(True)
    axis.set_xlabel("Number of bins, $N$")
    axis.set_ylabel("Total energy consumption (kWh)")
    axis.set_axisbelow(True)
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    _place_legend(
        axis,
        float((summary["total_energy_kwh"]
               + summary["total_energy_std_kwh"].fillna(0.0)).max()),
    )
    return _save(fig, stem)


def _makespan_figure(summary: pd.DataFrame, methods: list[str],
                     stem: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in methods:
        part = summary[summary["method"] == method].sort_values("n_bins")
        axis.plot(
            part["n_bins"], part["makespan_s"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, **METHOD_STYLES[method])
        deviations = part["makespan_std_s"].fillna(0.0)
        if np.any(deviations.to_numpy() > 0.0):
            band = axis.fill_between(
                part["n_bins"],
                part["makespan_s"] - deviations,
                part["makespan_s"] + deviations,
                facecolor=_cloud_color(METHOD_STYLES[method]["color"]),
                edgecolor="none", alpha=0.38, zorder=1)
            band.set_rasterized(True)
    axis.set_xlabel("Number of bins, $N$")
    axis.set_ylabel("Makespan (s)")
    axis.set_axisbelow(True)
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    _place_legend(
        axis,
        float((summary["makespan_s"]
               + summary["makespan_std_s"].fillna(0.0)).max()),
    )
    return _save(fig, stem)


def create_figures(input_csv: Path, output_dir: Path) -> list[Path]:
    _configure_ieee_style()
    frame, summary = _load(input_csv)
    methods = _method_sequence(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(_energy_figure(
        summary, methods, output_dir / "scalability_energy_per_bin_ieee"))
    outputs.extend(_total_energy_figure(
        summary, methods, output_dir / "scalability_total_energy_ieee"))
    outputs.extend(_makespan_figure(
        summary, methods, output_dir / "scalability_makespan_ieee"))
    manifest = {
        "source": _portable_path(input_csv),
        "observations": len(frame),
        "completed_sizes": sorted(frame["n_bins"].astype(int).unique().tolist()),
        "methods": methods,
        "energy_metric": "fleet energy divided by N",
        "total_energy_metric": "fleet energy consumption",
        "makespan_metric": "maximum vehicle completion time",
        "energy_cloud": "mean plus or minus one sample standard deviation",
        "makespan_cloud": "mean plus or minus one sample standard deviation",
        "formats": ["PDF", "PNG"],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_temporary = output_dir / ".figure_manifest.json.tmp"
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_temporary.replace(manifest_path)

    summary_path = input_csv.with_name(
        f"{input_csv.stem}_plot_summary.csv")
    summary_temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    summary.to_csv(summary_temporary, index=False)
    summary_temporary.replace(summary_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    for path in create_figures(args.input.resolve(), args.output_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
