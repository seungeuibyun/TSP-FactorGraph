"""Create cumulative Monte Carlo K-sweep figures with paired uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ieee import (
    EXPECTED_METHODS,
    METHOD_LABELS,
    METHOD_STYLES,
    _boxed_legend,
    _configure_ieee_style,
    _fit_legend_band,
    _portable_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VEHICLES = tuple(range(6, 15))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    try:
        for suffix in ("pdf", "png"):
            path = stem.with_suffix(f".{suffix}")
            temporary = stem.with_name(
                f".{stem.name}.{os.getpid()}.tmp").with_suffix(f".{suffix}")
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


def _cloud_color(color: str) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * 0.84)


def _stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _bootstrap_interval(
    values: np.ndarray,
    *,
    confidence_level: float,
    draws: int,
    random_seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    sample_means = samples.mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(sample_means, [alpha, 1.0 - alpha])
    return mean, float(lower), float(upper)


def _load_partial(input_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    required = {
        "run_id", "method", "vehicles", "start_hour", "energy_kwh",
        "makespan_s", "feasible", "active_vehicles",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"K-sweep Monte Carlo results miss {sorted(missing)}")
    if "demand_seed" not in frame.columns:
        if "seed" not in frame.columns:
            raise ValueError("results contain neither demand_seed nor seed")
        frame["demand_seed"] = frame["seed"]
    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    frame["feasible"] = frame["feasible"].astype(str).str.lower().eq("true")
    frame = frame[
        frame["feasible"]
        & (frame["active_vehicles"].astype(int)
           == frame["vehicles"].astype(int))
    ].copy()
    if frame.empty:
        raise ValueError("no feasible exact-K observation is available")
    frame = frame.drop_duplicates("run_id", keep="last")
    instance_columns = ["vehicles", "demand_seed", "start_hour"]
    complete = (frame.groupby(instance_columns)["method"].agg(
        lambda values: set(values) >= set(EXPECTED_METHODS)))
    complete_index = complete[complete].index
    indexed = frame.set_index(instance_columns)
    frame = indexed[indexed.index.isin(complete_index)].reset_index()
    if frame.empty:
        raise ValueError(
            "no instance has a complete proposed/NN/ACO/PSO/GA comparison")
    return frame.sort_values(
        ["vehicles", "demand_seed", "start_hour", "method"])


def _absolute_summary(
    frame: pd.DataFrame,
    column: str,
    *,
    confidence_level: float,
    draws: int,
) -> pd.DataFrame:
    # Hours are repeated conditions within a demand realization.  Average them
    # before bootstrapping so the uncertainty unit remains the independent seed.
    seed_level = (frame.groupby(
        ["method", "vehicles", "demand_seed"], as_index=False)
        .agg(value=(column, "mean"), hours=("start_hour", "nunique")))
    rows = []
    for (method, vehicles), part in seed_level.groupby(
            ["method", "vehicles"], sort=True):
        mean, lower, upper = _bootstrap_interval(
            part["value"].to_numpy(),
            confidence_level=confidence_level,
            draws=draws,
            random_seed=_stable_seed("absolute", column, method, vehicles),
        )
        rows.append({
            "method": method,
            "vehicles": int(vehicles),
            "seeds": int(part["demand_seed"].nunique()),
            "minimum_hours_per_seed": int(part["hours"].min()),
            "maximum_hours_per_seed": int(part["hours"].max()),
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        })
    return pd.DataFrame(rows)


def _paired_gap_summary(
    frame: pd.DataFrame,
    column: str,
    *,
    confidence_level: float,
    draws: int,
) -> pd.DataFrame:
    paired = frame.pivot_table(
        index=["vehicles", "demand_seed", "start_hour"],
        columns="method", values=column, aggfunc="last")
    if "proposed" not in paired.columns:
        return pd.DataFrame()
    records = []
    for method in EXPECTED_METHODS:
        if method not in paired.columns:
            continue
        if method == "proposed":
            valid = paired[["proposed"]].dropna().copy()
            valid["gap_pct"] = 0.0
        else:
            valid = paired[["proposed", method]].dropna().copy()
            valid["gap_pct"] = 100.0 * (
                valid[method] / valid["proposed"] - 1.0)
        seed_level = (valid.reset_index().groupby(
            ["vehicles", "demand_seed"], as_index=False)
            .agg(value=("gap_pct", "mean"), hours=("start_hour", "nunique")))
        for vehicles, part in seed_level.groupby("vehicles", sort=True):
            mean, lower, upper = _bootstrap_interval(
                part["value"].to_numpy(),
                confidence_level=confidence_level,
                draws=draws,
                random_seed=_stable_seed("gap", column, method, vehicles),
            )
            records.append({
                "method": method,
                "vehicles": int(vehicles),
                "seeds": int(part["demand_seed"].nunique()),
                "minimum_hours_per_seed": int(part["hours"].min()),
                "maximum_hours_per_seed": int(part["hours"].max()),
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            })
    return pd.DataFrame(records)


def _draw_summary(
    summary: pd.DataFrame,
    stem: Path,
    *,
    ylabel: str,
    vehicles: tuple[int, ...],
    zero_line: bool = False,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    data_upper = -np.inf
    data_lower = np.inf
    plotted = 0
    for method in EXPECTED_METHODS:
        part = summary[summary["method"] == method].sort_values("vehicles")
        if part.empty:
            continue
        x = part["vehicles"].to_numpy(dtype=float)
        mean = part["mean"].to_numpy(dtype=float)
        lower = part["ci_lower"].to_numpy(dtype=float)
        upper = part["ci_upper"].to_numpy(dtype=float)
        if np.any(upper > lower):
            band = axis.fill_between(
                x, lower, upper,
                facecolor=_cloud_color(METHOD_STYLES[method]["color"]),
                edgecolor="none", alpha=0.38, zorder=1)
            band.set_rasterized(True)
        axis.plot(
            x, mean, label=METHOD_LABELS[method],
            markerfacecolor="white", markeredgewidth=0.75,
            zorder=2.0, **METHOD_STYLES[method])
        data_lower = min(data_lower, float(np.nanmin(lower)))
        data_upper = max(data_upper, float(np.nanmax(upper)))
        plotted += 1
    if not plotted:
        plt.close(fig)
        return []
    if zero_line:
        axis.axhline(0.0, color="0.35", linewidth=0.6, zorder=0.5)
        data_lower = min(data_lower, 0.0)
        data_upper = max(data_upper, 0.0)
    data_range = max(data_upper - data_lower, max(abs(data_upper), 1.0) * 0.08)
    axis.set_ylim(data_lower - 0.04 * data_range,
                  data_upper + 0.04 * data_range)
    axis.set_xlabel("Exact active vehicles, $K$")
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(vehicles))
    axis.set_xlim(min(vehicles) - 0.25, max(vehicles) + 0.25)
    axis.set_axisbelow(True)
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    legend = _boxed_legend(
        axis, ncol=2, loc="upper left", handlelength=2.2,
        columnspacing=0.9)
    _fit_legend_band(axis, legend, data_upper)
    return _save(fig, stem)


def create_figures(
    input_csv: Path,
    output_dir: Path,
    *,
    vehicles: tuple[int, ...] = DEFAULT_VEHICLES,
    confidence_level: float = 0.95,
    bootstrap_draws: int = 2000,
) -> list[Path]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    _configure_ieee_style()
    frame = _load_partial(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "energy": _absolute_summary(
            frame, "energy_kwh", confidence_level=confidence_level,
            draws=bootstrap_draws),
        "makespan": _absolute_summary(
            frame, "makespan_s", confidence_level=confidence_level,
            draws=bootstrap_draws),
        "energy_gap": _paired_gap_summary(
            frame, "energy_kwh", confidence_level=confidence_level,
            draws=bootstrap_draws),
        "makespan_gap": _paired_gap_summary(
            frame, "makespan_s", confidence_level=confidence_level,
            draws=bootstrap_draws),
    }
    for name, summary in summaries.items():
        _atomic_csv(
            summary,
            input_csv.with_name(f"k_sweep_mc_{name}_summary.csv"))

    outputs = []
    outputs.extend(_draw_summary(
        summaries["energy"], output_dir / "k_sweep_mc_energy_ieee",
        ylabel="Energy consumption (kWh)", vehicles=vehicles))
    outputs.extend(_draw_summary(
        summaries["makespan"], output_dir / "k_sweep_mc_makespan_ieee",
        ylabel="Makespan (s)", vehicles=vehicles))
    outputs.extend(_draw_summary(
        summaries["energy_gap"],
        output_dir / "k_sweep_mc_normalized_energy_ieee",
        ylabel="Energy gap to proposed (%)", vehicles=vehicles,
        zero_line=True))
    outputs.extend(_draw_summary(
        summaries["makespan_gap"],
        output_dir / "k_sweep_mc_normalized_makespan_ieee",
        ylabel="Makespan gap to proposed (%)", vehicles=vehicles,
        zero_line=True))

    counts = (frame.groupby("vehicles")["demand_seed"].nunique()
              .sort_index().astype(int))
    manifest = {
        "source": _portable_path(input_csv),
        "observations": int(len(frame)),
        "vehicles": list(vehicles),
        "methods": [
            method for method in EXPECTED_METHODS
            if method in set(frame["method"])
        ],
        "demand_seeds_by_k": {
            str(int(vehicle)): int(count)
            for vehicle, count in counts.items()
        },
        "confidence_level": confidence_level,
        "bootstrap_draws": bootstrap_draws,
        "uncertainty_unit": (
            "demand seed after averaging the currently completed traffic "
            "hours within each seed"
        ),
        "relative_metric": (
            "100 * (baseline metric / proposed metric - 1), paired by K, "
            "demand seed, and traffic hour before seed-level averaging"
        ),
        "formats": ["PDF", "PNG"],
    }
    manifest_path = output_dir / "k_sweep_mc_figure_manifest.json"
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return outputs


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vehicles", type=int, nargs="+", default=DEFAULT_VEHICLES)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args(arguments)
    for path in create_figures(
            args.input.resolve(), args.output_dir.resolve(),
            vehicles=tuple(args.vehicles),
            confidence_level=args.confidence_level,
            bootstrap_draws=args.bootstrap_draws):
        print(path)


if __name__ == "__main__":
    main()
