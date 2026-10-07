# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import Transformer
from shapely.geometry import LineString


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = BASE / "seongbuk_buffer.net.xml"

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_loop_cleaned.csv"
)

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT_FILE = (
    BASE /
    "topis_sumo_mapping_after_rematch.csv"
)


# ============================================================
# 1. 이번에 primary로 교체할 2개
# ============================================================

fix_links = [
    "1050003200",
    "1070000400",
]


# ============================================================
# 2. SUMO
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
# 3. 기존 mapping
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

mapping["TOPIS_LINK_ID"] = (
    mapping["TOPIS_LINK_ID"]
    .str.strip()
)


# ============================================================
# 4. TOPIS geometry 2개만 읽기
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
        fix_links
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
        x,
        y
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

    topis_lines[
        link_id
    ] = LineString(xy)


# ============================================================
# 5. primary mapTrace
# ============================================================

def primary_match(line):

    distances = np.arange(
        5,
        max(
            5.1,
            line.length - 5
        ),
        10
    )

    trace = [
        (
            line.interpolate(d).x,
            line.interpolate(d).y
        )
        for d in distances
    ]

    route = mapTrace(
        trace,
        net,
        delta=30,
        fillGaps=100,
        direction=True,
        reversalPenalty=1000
    )

    return [
        e.getID()
        for e in route
    ]


# ============================================================
# 6. 새 mapping
# ============================================================

replacement = {}

for link_id in fix_links:

    edges = primary_match(
        topis_lines[
            link_id
        ]
    )

    replacement[
        link_id
    ] = edges

    print(
        link_id,
        "->",
        len(edges),
        "edges"
    )

    print(edges)


# ============================================================
# 7. 기존 2개 제거
# ============================================================

keep = mapping[
    ~mapping[
        "TOPIS_LINK_ID"
    ].isin(
        fix_links
    )
].copy()


# ============================================================
# 8. 새로운 2개 추가
# ============================================================

rows = []

for link_id, edges in (
    replacement.items()
):

    for seq, eid in enumerate(
        edges
    ):

        rows.append({
            "TOPIS_LINK_ID":
                link_id,

            "SUMO_EDGE_ID":
                eid,

            "SEQ":
                seq
        })


replacement_df = pd.DataFrame(
    rows
)


final = pd.concat(
    [
        keep,
        replacement_df
    ],
    ignore_index=True
)


final = (
    final
    .sort_values(
        [
            "TOPIS_LINK_ID",
            "SEQ"
        ]
    )
    .reset_index(
        drop=True
    )
)


final.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nCreated:",
    OUTPUT_FILE.name
)

print(
    "TOPIS links:",
    final[
        "TOPIS_LINK_ID"
    ].nunique()
)