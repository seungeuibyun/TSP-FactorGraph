"""Build a deterministic nested 500-bin set on the operational SUMO graph."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simul.energy import read_sumo_grades


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reachable(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    reached = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for target in adjacency.get(node, ()):
            if target not in reached:
                reached.add(target)
                queue.append(target)
    return reached


def generate(output: Path, manifest_path: Path, maximum_bins: int,
             seed: int) -> pd.DataFrame:
    if maximum_bins < 84:
        raise ValueError("maximum_bins must retain all 84 measured bins")

    network_path = PROJECT_ROOT / "simul/seongbuk_buffer_elevation.net.xml"
    traffic_path = PROJECT_ROOT / "simul/sumo_hourly_cost_operational_v3.csv"
    measured_path = PROJECT_ROOT / "simul/seongbuk_bins_84.csv"
    depot_path = PROJECT_ROOT / "simul/operational_depot_v3.csv"

    traffic_edges = set(pd.read_csv(
        traffic_path, usecols=["SUMO_EDGE_ID"], dtype=str,
        encoding="utf-8-sig")["SUMO_EDGE_ID"])
    edges = read_sumo_grades(
        network_path, required_edge_ids=traffic_edges)
    edges = edges[edges["SUMO_EDGE_ID"].astype(str).isin(traffic_edges)].copy()
    edges["SUMO_EDGE_ID"] = edges["SUMO_EDGE_ID"].astype(str)
    edges["FROM_NODE"] = edges["FROM_NODE"].astype(str)
    edges["TO_NODE"] = edges["TO_NODE"].astype(str)
    by_edge = edges.set_index("SUMO_EDGE_ID", drop=False)

    depot_edge = str(pd.read_csv(
        depot_path, dtype={"SUMO_EDGE_ID": str},
        encoding="utf-8-sig").iloc[0]["SUMO_EDGE_ID"])
    depot_node = str(by_edge.at[depot_edge, "TO_NODE"])

    forward: dict[str, list[str]] = defaultdict(list)
    reverse: dict[str, list[str]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        source = str(row.FROM_NODE)
        target = str(row.TO_NODE)
        forward[source].append(target)
        reverse[target].append(source)
    outbound = _reachable(depot_node, forward)
    inbound = _reachable(depot_node, reverse)

    measured = pd.read_csv(
        measured_path, dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
    measured = measured[
        measured["ROUND_TRIP_FEASIBLE"].astype(str).str.lower() == "true"
    ].copy().reset_index(drop=True)
    if len(measured) != 84:
        raise ValueError(f"expected 84 measured bins, found {len(measured)}")
    measured["idx"] = np.arange(len(measured), dtype=int)
    measured["SYNTHETIC"] = False
    measured["SOURCE"] = "measured_smart_bin"

    used_edges = set(measured["SUMO_EDGE_ID"].astype(str))
    used_nodes = {
        str(by_edge.at[edge_id, "TO_NODE"]) for edge_id in used_edges
    }
    candidates = edges[
        edges["FROM_NODE"].isin(outbound)
        & edges["TO_NODE"].isin(inbound)
        & ~edges["SUMO_EDGE_ID"].isin(used_edges)
        & ~edges["TO_NODE"].isin(used_nodes)
    ].sort_values("SUMO_EDGE_ID", kind="stable").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    selected_rows: list[dict] = []
    selected_nodes = set(used_nodes)
    for index in order:
        row = candidates.iloc[int(index)]
        target = str(row["TO_NODE"])
        if target in selected_nodes:
            continue
        selected_nodes.add(target)
        selected_rows.append({
            "idx": len(measured) + len(selected_rows),
            "node_id": target,
            "lon": np.nan,
            "lat": np.nan,
            "SUMO_EDGE_ID": str(row["SUMO_EDGE_ID"]),
            "SNAP_DISTANCE_M": 0.0,
            "OUTBOUND_REACHABLE": True,
            "RETURN_REACHABLE": True,
            "ROUND_TRIP_FEASIBLE": True,
            "SYNTHETIC": True,
            "SOURCE": "seeded_operational_road_node",
        })
        if len(measured) + len(selected_rows) == maximum_bins:
            break
    if len(measured) + len(selected_rows) != maximum_bins:
        raise ValueError(
            f"only {len(measured) + len(selected_rows)} unique round-trip "
            f"service nodes are available; requested {maximum_bins}")

    synthetic = pd.DataFrame(selected_rows)
    columns = list(dict.fromkeys([*measured.columns, *synthetic.columns]))
    frame = pd.concat([
        measured.reindex(columns=columns),
        synthetic.reindex(columns=columns),
    ], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")

    manifest = {
        "maximum_bins": maximum_bins,
        "measured_bins": len(measured),
        "synthetic_bins": len(synthetic),
        "seed": seed,
        "nested_prefix_rule": (
            "every N-bin instance uses the first N rows of the same file"),
        "synthetic_sampling": (
            "uniform seeded permutation of operational edges, retaining one "
            "unique destination node per bin and exact depot round-trip "
            "reachability"),
        "network": "simul/seongbuk_buffer_elevation.net.xml",
        "traffic": "simul/sumo_hourly_cost_operational_v3.csv",
        "measured_source": "simul/seongbuk_bins_84.csv",
        "output": output.resolve().relative_to(PROJECT_ROOT).as_posix(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-bins", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "simul/scalability_bins_500.csv")
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "simul/scalability_bins_500_manifest.json")
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    frame = generate(
        args.output.resolve(), args.manifest.resolve(),
        args.maximum_bins, args.seed)
    print(f"wrote {len(frame)} nested service locations to {args.output}")


if __name__ == "__main__":
    main()
