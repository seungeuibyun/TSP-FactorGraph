"""Plot matched Seongbuk-gu fleet routes for the manuscript comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, TextArea, VPacker
from matplotlib.patches import Rectangle
import numpy as np

from solver.model import load_table_i_instance
from .ieee import _configure_ieee_style


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("proposed", "aco", "ga")
METHOD_LABELS = {"proposed": "Proposed", "aco": "ACO", "ga": "GA"}
ROUTE_COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#7A5195", "#2F4B7C", "#8C564B", "#6B8E23",
)

# Hand-set normalized positions for the fixed manuscript snapshot.  Automatic
# collision avoidance is retained for other hours/seeds, but this comparison
# needs each payload bar to stay visibly associated with its route.
MANUSCRIPT_BAR_POSITIONS = {
    "proposed": np.asarray([
        [0.468, 0.355], [0.622, 0.581], [0.493, 0.678],
        [0.115, 0.346], [0.380, 0.694], [0.874, 0.442],
        [0.605, 0.777], [0.295, 0.198], [0.368, 0.435],
        [0.157, 0.684],
    ]),
    "aco": np.asarray([
        [0.401, 0.728], [0.489, 0.625], [0.628, 0.580],
        [0.612, 0.747], [0.892, 0.491], [0.268, 0.535],
        [0.329, 0.226], [0.520, 0.307], [0.164, 0.670],
        [0.123, 0.313],
    ]),
    "ga": np.asarray([
        [0.856, 0.486], [0.394, 0.693], [0.182, 0.716],
        [0.455, 0.327], [0.057, 0.509], [0.151, 0.233],
        [0.828, 0.781], [0.535, 0.495], [0.609, 0.716],
        [0.517, 0.700],
    ]),
}


def _points(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return []
    return [tuple(map(float, point.split(",")[:2]))
            for point in value.split()]


def _network_geometry(net_file: Path, required_edge_ids: set[str]) -> tuple[
        dict[str, tuple[float, float]],
        dict[str, list[tuple[float, float]]]]:
    junctions: dict[str, tuple[float, float]] = {}
    raw_edges: dict[str, tuple[str, str, list[tuple[float, float]]]] = {}
    for _, elem in ET.iterparse(net_file, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "junction":
            junctions[str(elem.get("id"))] = (
                float(elem.get("x", 0.0)), float(elem.get("y", 0.0)))
            elem.clear()
        elif tag == "edge":
            edge_id = str(elem.get("id", ""))
            if (edge_id in required_edge_ids
                    and elem.get("function") != "internal"
                    and not edge_id.startswith(":")):
                shape = _points(elem.get("shape"))
                if not shape:
                    lane = elem.find("lane")
                    shape = _points(None if lane is None else lane.get("shape"))
                raw_edges[edge_id] = (
                    str(elem.get("from")), str(elem.get("to")), shape)
            elem.clear()
    edges: dict[str, list[tuple[float, float]]] = {}
    for edge_id, (source, target, shape) in raw_edges.items():
        if len(shape) < 2 and source in junctions and target in junctions:
            shape = [junctions[source], junctions[target]]
        if len(shape) >= 2:
            edges[edge_id] = shape
    return junctions, edges


def _route_load_bar(
    axis: plt.Axes,
    anchor: np.ndarray,
    position: np.ndarray,
    load_kg: float,
    capacity_kg: float,
    color: str,
) -> None:
    """Place a route-linked payload bar matching the manuscript map style."""

    bar_width = 3.7
    bar_height = 10.5
    utilization = np.clip(load_kg / capacity_kg, 0.0, 1.0)
    route_rgb = np.asarray(to_rgb(color))
    light_fill = tuple(route_rgb + 0.72 * (1.0 - route_rgb))
    drawing = DrawingArea(5.7, 14.2, 0.0, 0.0)
    drawing.add_artist(Rectangle(
        (1.0, 0.7), bar_width, max(0.35, bar_height * utilization),
        facecolor=light_fill, edgecolor=color, linewidth=0.80))
    drawing.add_artist(Line2D(
        [0.0, 5.7], [13.0, 13.0], color="0.15",
        linewidth=0.65, linestyle=(0, (2.0, 1.5))))
    label = TextArea(
        f"{int(round(load_kg))}",
        textprops={"fontsize": 7.2, "fontweight": "bold", "ha": "center"})
    packed = VPacker(children=[label, drawing], align="center", pad=0.0, sep=0.2)
    annotation = AnnotationBbox(
        packed, anchor, xybox=position,
        xycoords="data", boxcoords="data", frameon=False,
        zorder=5,
    )
    axis.add_artist(annotation)


def _service_area_roads(
    edge_shapes: dict[str, list[tuple[float, float]]],
    service_points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius_m: float = 425.0,
) -> list[list[tuple[float, float]]]:
    """Keep only operational road fragments surrounding the service data."""

    radius_squared = radius_m * radius_m
    selected = []
    for shape in edge_shapes.values():
        values = np.asarray(shape)
        if (values[:, 0].max() < lower[0]
                or values[:, 0].min() > upper[0]
                or values[:, 1].max() < lower[1]
                or values[:, 1].min() > upper[1]):
            continue
        sample_indices = np.unique(np.linspace(
            0, len(values) - 1, min(len(values), 5), dtype=int))
        samples = values[sample_indices]
        delta = samples[:, None, :] - service_points[None, :, :]
        if np.any(np.sum(delta * delta, axis=2) <= radius_squared):
            selected.append(shape)
    return selected


def _bar_positions(
    anchors: np.ndarray,
    depot_xy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    route_polylines: list[np.ndarray],
    bin_xy: np.ndarray,
) -> np.ndarray:
    """Place opaque load bars in nearby gaps without covering the routes."""

    span = np.maximum(upper - lower, 1.0)
    normalized = (anchors - lower) / span
    obstacle_parts = [(bin_xy - lower) / span,
                      ((depot_xy - lower) / span)[None, :]]
    for polyline in route_polylines:
        if len(polyline):
            stride = max(1, len(polyline) // 250)
            obstacle_parts.append((polyline[::stride] - lower) / span)
    obstacles = np.vstack(obstacle_parts)

    radii = np.asarray([0.025, 0.045, 0.065, 0.085, 0.105, 0.125, 0.145])
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    selected: list[np.ndarray] = []
    for anchor in normalized:
        candidates = [anchor]
        candidates.extend(
            anchor + radius * np.asarray([np.cos(angle), np.sin(angle)])
            for radius in radii for angle in angles)
        best, best_score = None, np.inf
        for candidate in candidates:
            x_coord, y_coord = candidate
            if not (0.045 <= x_coord <= 0.955
                    and 0.135 <= y_coord <= 0.845):
                continue
            if x_coord > 0.55 and y_coord < 0.265:
                continue
            delta = obstacles - candidate
            scaled = np.column_stack((delta[:, 0] / 0.032,
                                      delta[:, 1] / 0.105))
            clearance = float(np.exp(
                -0.5 * np.sum(scaled * scaled, axis=1)).max())
            occupied = float(np.count_nonzero(
                (np.abs(delta[:, 0]) < 0.050)
                & (np.abs(delta[:, 1]) < 0.115)))
            distance = float(np.linalg.norm(
                (candidate - anchor) / np.asarray([0.20, 0.28])))
            overlap = 0.0
            for other in selected:
                separation = np.abs(candidate - other)
                if separation[0] < 0.110 and separation[1] < 0.190:
                    overlap += 500.0 * (
                        1.0 - separation[0] / 0.110) * (
                        1.0 - separation[1] / 0.190)
            score = (50.0 * occupied + 2.0 * clearance
                     + 3.0 * distance + overlap)
            if score < best_score:
                best, best_score = candidate, score
        selected.append(anchor if best is None else best)
    return lower + np.asarray(selected) * span


def _route_polylines(instance, routes: list[list[int]],
                     junctions: dict[str, tuple[float, float]],
                     edge_shapes: dict[str, list[tuple[float, float]]]
                     ) -> tuple[list[np.ndarray], set[str]]:
    """Join directed SUMO edge shapes into one continuous line per vehicle."""

    all_polylines: list[np.ndarray] = []
    used_edges: set[str] = set()
    service_time = float(instance.network.params.service_time_s)
    for vehicle, route in enumerate(routes):
        state = instance.vehicle_states[vehicle]
        source = state.node
        elapsed = 0.0
        payload = float(state.payload_kg)
        points = [np.asarray(junctions[source], dtype=float)]

        def append_path(path: list[str]) -> None:
            for edge_id in path:
                if edge_id not in edge_shapes:
                    continue
                shape = np.asarray(edge_shapes[edge_id], dtype=float)
                if np.linalg.norm(points[-1] - shape[-1]) < np.linalg.norm(
                        points[-1] - shape[0]):
                    shape = shape[::-1]
                if np.linalg.norm(points[-1] - shape[0]) < 1e-7:
                    points.extend(shape[1:])
                else:
                    points.extend(shape)

        for raw_index in route:
            index = int(raw_index)
            target = instance.bin_nodes[index]
            path, _, travel_time = instance.network.shortest_path(
                source, target, instance.start_time_s + elapsed, payload)
            used_edges.update(map(str, path))
            append_path(list(map(str, path)))
            elapsed += float(travel_time) + service_time
            payload += float(instance.demand_kg[index])
            source = target
        path, _, _ = instance.network.shortest_path(
            source, instance.depot_node,
            instance.start_time_s + elapsed, payload)
        used_edges.update(map(str, path))
        append_path(list(map(str, path)))
        all_polylines.append(np.asarray(points))
    return all_polylines, used_edges


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=600 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _draw_map(method: str, run: dict, instance,
              junctions: dict[str, tuple[float, float]],
              edge_shapes: dict[str, list[tuple[float, float]]],
              output_dir: Path,
              routes: list[list[int]],
              route_polylines: list[np.ndarray],
              lower: np.ndarray,
              upper: np.ndarray) -> list[Path]:
    bin_xy = np.asarray([junctions[node] for node in instance.bin_nodes])
    depot_xy = np.asarray(junctions[instance.depot_node])

    background = _service_area_roads(
        edge_shapes, np.vstack([bin_xy, depot_xy]), lower, upper)

    fig, axis = plt.subplots(figsize=(3.53, 2.38))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    axis.add_collection(LineCollection(
        background, colors="#D0D0D0", linewidths=0.28, zorder=0))
    for vehicle, polyline in enumerate(route_polylines):
        if len(polyline) >= 2:
            axis.plot(
                polyline[:, 0], polyline[:, 1],
                color=ROUTE_COLORS[vehicle % len(ROUTE_COLORS)],
                linewidth=1.10, solid_capstyle="round",
                solid_joinstyle="round", zorder=2)

    axis.scatter(
        bin_xy[:, 0], bin_xy[:, 1], s=6.0, c="#333333", linewidths=0.0,
        zorder=3, label="Smart bin")
    axis.scatter(
        [depot_xy[0]], [depot_xy[1]], marker="*", s=34,
        facecolor="black", edgecolor="white", linewidth=0.35, zorder=4,
        label="Depot")
    capacity_kg = float(instance.capacity_kg)
    anchors = np.asarray([
        bin_xy[np.asarray(route, dtype=int)].mean(axis=0)
        for route in routes
    ])
    if (int(run["run"]["start_hour"]) == 11
            and int(run["run"]["seed"]) == 7
            and method in MANUSCRIPT_BAR_POSITIONS):
        bar_positions = (
            lower + MANUSCRIPT_BAR_POSITIONS[method] * (upper - lower))
    else:
        bar_positions = _bar_positions(
            anchors, depot_xy, lower, upper, route_polylines, bin_xy)
    for vehicle, route in enumerate(routes):
        load_kg = float(instance.demand_kg[np.asarray(route, dtype=int)].sum())
        _route_load_bar(
            axis, anchors[vehicle], bar_positions[vehicle], load_kg, capacity_kg,
            ROUTE_COLORS[vehicle % len(ROUTE_COLORS)])
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D([0], [0], color=ROUTE_COLORS[0], linewidth=1.2,
               label="Vehicle route"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#333333",
               markeredgecolor="#333333", markersize=4.2,
               label="Smart bin"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="black",
               markeredgecolor="white", markeredgewidth=0.35,
               markersize=7.0, label="Depot"),
        Line2D([0], [0], color="0.10", linewidth=0.7, linestyle=":",
               label=r"Capacity $\ell_{\max}$ (kg)"),
    ]
    legend = axis.legend(
        handles=handles, loc="lower right", ncol=1, frameon=True,
        fancybox=False, facecolor="white", edgecolor="0.2", framealpha=1.0,
        fontsize=7.0, handlelength=1.8,
        borderpad=0.40, labelspacing=0.35, borderaxespad=0.15,
        scatterpoints=1, markerscale=1.0)
    legend.get_frame().set_linewidth(0.6)
    stem = output_dir / f"route_h{int(run['run']['start_hour']):02d}_{method}_ieee"
    return _save(fig, stem)


def create_figures(route_dir: Path, output_dir: Path, hour: int = 11,
                   seed: int = 7) -> list[Path]:
    _configure_ieee_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    instance = load_table_i_instance(
        n_bins=84, vehicles=10, start_hour=hour, seed=seed)
    junctions, edge_shapes = _network_geometry(
        PROJECT_ROOT / "simul/seongbuk_buffer_elevation.net.xml",
        set(instance.network.edge_data))
    runs = {}
    prepared = {}
    for method in METHODS:
        path = route_dir / f"{method}_n84_k10_h{hour:02d}_s{seed}.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        runs[method] = (path, run)
        routes = [[int(index) for index in route] for route in run["routes"]]
        route_polylines, _ = _route_polylines(
            instance, routes, junctions, edge_shapes)
        prepared[method] = (routes, route_polylines)

    bin_xy = np.asarray([junctions[node] for node in instance.bin_nodes])
    depot_xy = np.asarray(junctions[instance.depot_node])
    route_points = [
        polyline
        for _, polylines in prepared.values()
        for polyline in polylines
        if len(polyline) >= 2
    ]
    all_points = np.vstack([bin_xy, depot_xy, *route_points])
    span = np.ptp(all_points, axis=0)
    margin = np.maximum(0.015 * span, 1.0)
    lower = all_points.min(axis=0) - margin
    upper = all_points.max(axis=0) + margin

    outputs = []
    manifest = {
        "hour": int(hour), "seed": int(seed), "n_bins": 84,
        "vehicles": 10,
        "background": (
            "operational traffic-data links within 425 m of a service bin "
            "or the depot"),
        "route_bars": (
            "colored height is assigned waste load in kg; no enclosing "
            "capacity rectangle; dashed mark is Q=2000 kg"),
        "methods": {},
    }
    for method in METHODS:
        path, run = runs[method]
        routes, route_polylines = prepared[method]
        outputs.extend(_draw_map(
            method, run, instance, junctions, edge_shapes, output_dir,
            routes, route_polylines, lower, upper))
        manifest["methods"][method] = {
            "label": METHOD_LABELS[method],
            "route_file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "energy_kwh": float(run["run"]["energy_kwh"]),
            "makespan_s": float(run["run"]["makespan_s"]),
            "feasible": bool(run["run"]["feasible"]),
        }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route-dir", type=Path,
        default=PROJECT_ROOT / "results/hourly_24h/hourly_routes")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "figures/route_comparison")
    parser.add_argument("--hour", type=int, default=11)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    for path in create_figures(
            args.route_dir.resolve(), args.output_dir.resolve(),
            hour=args.hour, seed=args.seed):
        print(path)


if __name__ == "__main__":
    main()
