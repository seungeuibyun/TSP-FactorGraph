# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import unary_union


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = BASE / "seongbuk_buffer.net.xml"
MAPPING_FILE = BASE / "topis_sumo_mapping_loop_cleaned.csv"
REVIEW_FILE = BASE / "final_mapping_review_v2.csv"

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT = BASE / "review14_candidate_comparison.csv"


# ============================================================
# 1. SUMO
# ============================================================

SUMO_HOME = os.environ["SUMO_HOME"]

sys.path.append(
    os.path.join(
        SUMO_HOME,
        "tools"
    )
)

import sumolib
from sumolib.route import mapTrace

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 2. REVIEW 14
# ============================================================

review = pd.read_csv(
    REVIEW_FILE,
    dtype={"TOPIS_LINK_ID": str}
)

review_links = (
    review["TOPIS_LINK_ID"]
    .str.strip()
    .tolist()
)

print(
    "review links:",
    len(review_links)
)

print(review_links)


# ============================================================
# 3. TOPIS GEOMETRY
# ============================================================

vtx = pd.read_excel(
    VERTEX_FILE,
    dtype={"LINK_ID": str}
)

vtx["LINK_ID"] = (
    vtx["LINK_ID"]
    .str.strip()
)

vtx = vtx[
    vtx["LINK_ID"].isin(
        review_links
    )
].copy()


to_wgs84 = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


coords = []

for x, y in zip(
    vtx["GRS80TM_X"],
    vtx["GRS80TM_Y"]
):

    lon, lat = to_wgs84.transform(
        x, y
    )

    coords.append(
        net.convertLonLat2XY(
            lon, lat
        )
    )


vtx["SUMO_X"] = [
    p[0] for p in coords
]

vtx["SUMO_Y"] = [
    p[1] for p in coords
]

vtx = vtx.sort_values(
    ["LINK_ID", "VER_SEQ"]
)


topis_lines = {}

for link_id, g in vtx.groupby(
    "LINK_ID"
):

    xy = list(
        zip(
            g["SUMO_X"],
            g["SUMO_Y"]
        )
    )

    if len(xy) >= 2:

        topis_lines[
            link_id
        ] = LineString(xy)


# ============================================================
# 4. MAPTRACE
# ============================================================

def trace_match(
    line,
    spacing,
    delta,
    fill_gaps,
    reversal_penalty
):

    if line.length <= 10:

        ds = [
            line.length / 2
        ]

    else:

        ds = np.arange(
            5,
            line.length - 5,
            spacing
        )

    trace = [
        (
            line.interpolate(d).x,
            line.interpolate(d).y
        )
        for d in ds
    ]

    route = mapTrace(
        trace,
        net,
        delta=delta,
        fillGaps=fill_gaps,
        direction=True,
        reversalPenalty=reversal_penalty
    )

    return [
        e.getID()
        for e in route
    ]


def primary(line):

    return trace_match(
        line,
        spacing=10,
        delta=30,
        fill_gaps=100,
        reversal_penalty=1000
    )


def secondary(line):

    return trace_match(
        line,
        spacing=5,
        delta=20,
        fill_gaps=30,
        reversal_penalty=10000
    )


# ============================================================
# 5. HELPERS
# ============================================================

def physical_id(eid):

    return (
        eid[1:]
        if eid.startswith("-")
        else eid
    )


def broken_count(edges):

    n = 0

    for a, b in zip(
        edges[:-1],
        edges[1:]
    ):

        ea = net.getEdge(a)
        eb = net.getEdge(b)

        if (
            ea.getToNode().getID()
            !=
            eb.getFromNode().getID()
        ):
            n += 1

    return n


def sample_line(
    line,
    spacing=10
):

    ds = list(
        np.arange(
            0,
            line.length,
            spacing
        )
    )

    ds.append(
        line.length
    )

    return [
        line.interpolate(d)
        for d in ds
    ]


def route_geometry(edges):

    lines = []

    for eid in edges:

        shape = (
            net.getEdge(eid)
            .getShape()
        )

        if len(shape) >= 2:

            lines.append(
                LineString(shape)
            )

    if not lines:
        return None

    return unary_union(lines)


# ============================================================
# 6. V2 QC
# ============================================================

def evaluate(
    topis_line,
    edges
):

    if not edges:

        return {
            "n_edges": 0,
            "duplicate": 999,
            "physical_dup": 999,
            "broken": 999,
            "mean_t2r": 9999,
            "max_t2r": 9999,
            "mean_interior": 9999,
            "max_interior": 9999,
            "length_ratio": 9999,
            "pass": False
        }

    duplicate = (
        len(edges)
        -
        len(set(edges))
    )

    physical = [
        physical_id(e)
        for e in edges
    ]

    physical_dup = (
        len(physical)
        -
        len(set(physical))
    )

    broken = broken_count(
        edges
    )

    # ----------------------
    # TOPIS -> route
    # ----------------------

    rg = route_geometry(
        edges
    )

    pts = sample_line(
        topis_line
    )

    d = [
        p.distance(rg)
        for p in pts
    ]

    mean_t2r = float(
        np.mean(d)
    )

    max_t2r = float(
        np.max(d)
    )

    # ----------------------
    # interior route -> TOPIS
    # ----------------------

    d2 = []

    if len(edges) > 2:

        for eid in edges[1:-1]:

            e_line = LineString(
                net.getEdge(
                    eid
                ).getShape()
            )

            for p in sample_line(
                e_line
            ):

                d2.append(
                    p.distance(
                        topis_line
                    )
                )

    if d2:

        mean_interior = float(
            np.mean(d2)
        )

        max_interior = float(
            np.max(d2)
        )

    else:

        mean_interior = 0.0
        max_interior = 0.0

    # ----------------------
    # length ratio
    # ----------------------

    route_len = sum(
        net.getEdge(e)
        .getLength()
        for e in edges
    )

    length_ratio = (
        route_len /
        topis_line.length
    )

    detour = (
        (
            len(edges) > 2
            and (
                mean_interior > 20
                or
                max_interior > 80
            )
        )
        or
        (
            len(edges) >= 3
            and length_ratio > 3
        )
    )

    passed = (
        duplicate == 0
        and
        physical_dup == 0
        and
        broken == 0
        and
        mean_t2r <= 15
        and
        max_t2r <= 50
        and
        not detour
    )

    return {
        "n_edges":
            len(edges),

        "duplicate":
            duplicate,

        "physical_dup":
            physical_dup,

        "broken":
            broken,

        "mean_t2r":
            mean_t2r,

        "max_t2r":
            max_t2r,

        "mean_interior":
            mean_interior,

        "max_interior":
            max_interior,

        "length_ratio":
            length_ratio,

        "pass":
            passed
    }


# ============================================================
# 7. COMPARE PRIMARY / SECONDARY
# ============================================================

rows = []

for link_id in review_links:

    line = topis_lines[
        link_id
    ]

    candidates = {
        "primary":
            primary(line),

        "secondary":
            secondary(line)
    }

    print(
        "\n========================="
    )

    print(
        "TOPIS:",
        link_id
    )

    for method, edges in (
        candidates.items()
    ):

        q = evaluate(
            line,
            edges
        )

        rows.append({
            "TOPIS_LINK_ID":
                link_id,

            "METHOD":
                method,

            **q
        })

        print(
            method,
            "| edges =",
            q["n_edges"],
            "| broken =",
            q["broken"],
            "| physical dup =",
            q["physical_dup"],
            "| mean =",
            round(
                q["mean_t2r"],
                1
            ),
            "| max =",
            round(
                q["max_t2r"],
                1
            ),
            "| interior mean =",
            round(
                q["mean_interior"],
                1
            ),
            "| ratio =",
            round(
                q["length_ratio"],
                2
            ),
            "| PASS =",
            q["pass"]
        )


# ============================================================
# 8. OUTPUT
# ============================================================

result = pd.DataFrame(
    rows
)

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. SUMMARY
# ============================================================

passed = result[
    result["pass"]
].copy()

print(
    "\n=========================="
)

print(
    "RE-MATCH SUMMARY"
)

print(
    "=========================="
)

print(
    "links with at least one passing candidate:",
    passed[
        "TOPIS_LINK_ID"
    ].nunique()
)

if len(passed) > 0:

    print(
        "\nPASSING CANDIDATES:"
    )

    print(
        passed[
            [
                "TOPIS_LINK_ID",
                "METHOD",
                "n_edges",
                "mean_t2r",
                "max_t2r",
                "mean_interior",
                "max_interior",
                "length_ratio"
            ]
        ].to_string(
            index=False
        )
    )

print(
    "\nCreated:",
    OUTPUT.name
)