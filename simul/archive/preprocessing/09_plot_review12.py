# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import LineString


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = BASE / "seongbuk_buffer.net.xml"

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_corridor.csv"
)

REVIEW_FILE = (
    BASE / "final_mapping_review_v2.csv"
)

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT_DIR = (
    BASE / "review12_compare"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


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
# 2. REVIEW 12
# ============================================================

review = pd.read_csv(
    REVIEW_FILE,
    dtype={"TOPIS_LINK_ID": str}
)

review_links = (
    review[
        "TOPIS_LINK_ID"
    ]
    .str.strip()
    .tolist()
)

print(
    "review links:",
    len(review_links)
)


# ============================================================
# 3. CURRENT MAPPING
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

mapping["TOPIS_LINK_ID"] = (
    mapping[
        "TOPIS_LINK_ID"
    ].str.strip()
)


# ============================================================
# 4. TOPIS GEOMETRY
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

    lon, lat = (
        to_wgs84.transform(
            x,
            y
        )
    )

    coords.append(
        net.convertLonLat2XY(
            lon,
            lat
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
# 5. mapTrace candidate
# ============================================================

def trace_match(
    line,
    spacing,
    delta,
    fill_gaps,
    reversal_penalty
):

    ds = np.arange(
        5,
        max(
            5.1,
            line.length - 5
        ),
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
# 6. plot helper
# ============================================================

def draw_background(
    ax,
    line,
    margin=150
):

    minx, miny, maxx, maxy = (
        line.bounds
    )

    for e in net.getEdges():

        shape = e.getShape()

        if len(shape) < 2:
            continue

        xs = [
            p[0] for p in shape
        ]

        ys = [
            p[1] for p in shape
        ]

        if (
            max(xs) < minx - margin
            or
            min(xs) > maxx + margin
            or
            max(ys) < miny - margin
            or
            min(ys) > maxy + margin
        ):
            continue

        ax.plot(
            xs,
            ys,
            linewidth=0.4,
            alpha=0.25
        )

    ax.set_xlim(
        minx - margin,
        maxx + margin
    )

    ax.set_ylim(
        miny - margin,
        maxy + margin
    )

    ax.set_aspect(
        "equal"
    )


def draw_topis(
    ax,
    line
):

    x, y = line.xy

    ax.plot(
        x,
        y,
        linewidth=4,
        alpha=0.65,
        label="TOPIS"
    )

    # 시작점
    ax.scatter(
        [x[0]],
        [y[0]],
        s=50,
        marker="o"
    )

    # 끝점
    ax.scatter(
        [x[-1]],
        [y[-1]],
        s=60,
        marker="x"
    )


def draw_route(
    ax,
    edge_ids
):

    for eid in edge_ids:

        e = net.getEdge(
            eid
        )

        shape = e.getShape()

        if len(shape) < 2:
            continue

        xs = [
            p[0]
            for p in shape
        ]

        ys = [
            p[1]
            for p in shape
        ]

        ax.plot(
            xs,
            ys,
            linewidth=2.5,
            alpha=0.9
        )


# ============================================================
# 7. 12개 overlay 비교 그림
# ============================================================

for idx, link_id in enumerate(
    review_links,
    1
):

    print(
        f"[{idx}/{len(review_links)}]",
        link_id
    )

    line = topis_lines[
        link_id
    ]

    current = (
        mapping[
            mapping[
                "TOPIS_LINK_ID"
            ] == link_id
        ]
        .sort_values("SEQ")
        ["SUMO_EDGE_ID"]
        .tolist()
    )

    p = primary(line)
    s = secondary(line)

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    minx, miny, maxx, maxy = (
        line.bounds
    )

    margin = 150

    # --------------------------------
    # SUMO background
    # --------------------------------

    for e in net.getEdges():

        shape = e.getShape()

        if len(shape) < 2:
            continue

        xs = [
            q[0] for q in shape
        ]

        ys = [
            q[1] for q in shape
        ]

        if (
            max(xs) < minx - margin
            or min(xs) > maxx + margin
            or max(ys) < miny - margin
            or min(ys) > maxy + margin
        ):
            continue

        ax.plot(
            xs,
            ys,
            color="lightgray",
            linewidth=0.5,
            alpha=0.5,
            zorder=1
        )

    # --------------------------------
    # helper
    # --------------------------------

    def plot_route(
        edges,
        color,
        label,
        linewidth,
        alpha,
        zorder
    ):

        first = True

        for eid in edges:

            e = net.getEdge(eid)

            shape = e.getShape()

            if len(shape) < 2:
                continue

            xs = [
                q[0]
                for q in shape
            ]

            ys = [
                q[1]
                for q in shape
            ]

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=(
                    label
                    if first
                    else None
                ),
                zorder=zorder
            )

            first = False

    # --------------------------------
    # CURRENT
    # --------------------------------

    plot_route(
        current,
        color="red",
        label=f"Current ({len(current)})",
        linewidth=4,
        alpha=0.45,
        zorder=2
    )

    # --------------------------------
    # PRIMARY
    # --------------------------------

    plot_route(
        p,
        color="blue",
        label=f"Primary ({len(p)})",
        linewidth=3,
        alpha=0.65,
        zorder=3
    )

    # --------------------------------
    # SECONDARY
    # --------------------------------

    plot_route(
        s,
        color="green",
        label=f"Secondary ({len(s)})",
        linewidth=2,
        alpha=0.8,
        zorder=4
    )

    # --------------------------------
    # TOPIS는 제일 위에
    # --------------------------------

    tx, ty = line.xy

    ax.plot(
        tx,
        ty,
        color="black",
        linewidth=2,
        linestyle="--",
        label="TOPIS",
        zorder=10
    )

    # 시작 / 끝
    ax.scatter(
        [tx[0]],
        [ty[0]],
        color="black",
        marker="o",
        s=80,
        zorder=11,
        label="TOPIS start"
    )

    ax.scatter(
        [tx[-1]],
        [ty[-1]],
        color="black",
        marker="x",
        s=100,
        linewidths=3,
        zorder=11,
        label="TOPIS end"
    )

    ax.set_xlim(
        minx - margin,
        maxx + margin
    )

    ax.set_ylim(
        miny - margin,
        maxy + margin
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        f"TOPIS {link_id}"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        f"{link_id}_overlay.png",
        dpi=180
    )

    plt.close(fig)


print("\nDONE")
print("Saved in:", OUTPUT_DIR)