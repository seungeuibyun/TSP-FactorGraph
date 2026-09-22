import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import Transformer
from shapely.geometry import LineString


# ============================================================
# 0. paths
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = BASE / "seongbuk_buffer.net.xml"
MAPPING_FILE = BASE / "topis_sumo_mapping_best.csv"
QC_FILE = BASE / "topis_sumo_mapping_qc.csv"
VERTEX_FILE = BASE / "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"

OUTPUT_FIXED = BASE / "topis_sumo_mapping_direction_fixed.csv"
OUTPUT_RECHECK = BASE / "direction_fix_qc.csv"


# ============================================================
# 1. SUMO
# ============================================================

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(
    os.path.join(SUMO_HOME, "tools")
)

import sumolib

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 2. review 결과 읽기
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

qc = pd.read_csv(
    QC_FILE,
    dtype={"TOPIS_LINK_ID": str}
)

direction_links = qc.loc[
    qc["physical_duplicates"] > 0,
    "TOPIS_LINK_ID"
].tolist()

print(
    "direction oscillation links:",
    len(direction_links)
)

print(direction_links)


# ============================================================
# 3. 문제 링크 TOPIS geometry만 읽기
# ============================================================

vtx = pd.read_excel(
    VERTEX_FILE,
    dtype={"LINK_ID": str}
)

vtx["LINK_ID"] = (
    vtx["LINK_ID"].str.strip()
)

vtx = vtx[
    vtx["LINK_ID"].isin(
        direction_links
    )
].copy()


to_wgs84 = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


def topis_to_sumo(x, y):

    lon, lat = to_wgs84.transform(
        x, y
    )

    return net.convertLonLat2XY(
        lon, lat
    )


coords = [
    topis_to_sumo(x, y)
    for x, y in zip(
        vtx["GRS80TM_X"],
        vtx["GRS80TM_Y"]
    )
]

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

for link_id, group in vtx.groupby(
    "LINK_ID"
):

    xy = list(
        zip(
            group["SUMO_X"],
            group["SUMO_Y"]
        )
    )

    topis_lines[link_id] = (
        LineString(xy)
    )


# ============================================================
# 4. heading functions
# ============================================================

def heading(p1, p2):

    return (
        math.degrees(
            math.atan2(
                p2[1] - p1[1],
                p2[0] - p1[0]
            )
        )
        % 360
    )


def angle_diff(a, b):

    d = abs(a - b) % 360

    return min(
        d,
        360 - d
    )


def local_heading(
    line,
    distance,
    delta=5.0
):

    d1 = max(
        0,
        distance - delta
    )

    d2 = min(
        line.length,
        distance + delta
    )

    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)

    return heading(
        (p1.x, p1.y),
        (p2.x, p2.y)
    )


# ============================================================
# 5. edge가 TOPIS 진행방향과 얼마나 잘 맞는지
# ============================================================

def direction_error(
    eid,
    topis_line
):

    e = net.getEdge(eid)

    shape = e.getShape()

    if len(shape) < 2:
        return 999

    edge_line = LineString(
        shape
    )

    midpoint = edge_line.interpolate(
        0.5,
        normalized=True
    )

    # 이 SUMO edge가 TOPIS 선 어디쯤인지
    d = topis_line.project(
        midpoint
    )

    topis_h = local_heading(
        topis_line,
        d
    )

    edge_h = local_heading(
        edge_line,
        edge_line.length / 2,
        delta=min(
            5.0,
            max(
                0.1,
                edge_line.length / 3
            )
        )
    )

    return angle_diff(
        topis_h,
        edge_h
    )


# ============================================================
# 6. +edge / -edge oscillation 제거
# ============================================================

def physical_id(eid):

    if eid.startswith("-"):
        return eid[1:]

    return eid


def fix_direction_oscillation(
    edge_ids,
    topis_line
):

    # 같은 physical edge가 여러 번 등장하면
    # TOPIS local direction에 가장 잘 맞는 orientation 선택

    best_orientation = {}

    for eid in edge_ids:

        pid = physical_id(eid)

        err = direction_error(
            eid,
            topis_line
        )

        if (
            pid not in best_orientation
            or
            err <
            best_orientation[pid][1]
        ):

            best_orientation[pid] = (
                eid,
                err
            )

    preferred = {
        pid: value[0]
        for pid, value
        in best_orientation.items()
    }

    cleaned = []

    seen_physical = set()

    for eid in edge_ids:

        pid = physical_id(eid)

        # 같은 물리 edge는 한 번만
        if pid in seen_physical:
            continue

        selected = preferred[pid]

        cleaned.append(
            selected
        )

        seen_physical.add(
            pid
        )

    return cleaned


# ============================================================
# 7. connectivity 검사
# ============================================================

def count_broken(
    edge_ids
):

    broken_pairs = []

    for a, b in zip(
        edge_ids[:-1],
        edge_ids[1:]
    ):

        ea = net.getEdge(a)
        eb = net.getEdge(b)

        if (
            ea.getToNode().getID()
            !=
            eb.getFromNode().getID()
        ):

            broken_pairs.append(
                (a, b)
            )

    return broken_pairs


# ============================================================
# 8. 13개만 수정
# ============================================================

replacement = {}

qc_rows = []

for link_id in direction_links:

    old = (
        mapping[
            mapping[
                "TOPIS_LINK_ID"
            ] == link_id
        ]
        .sort_values("SEQ")
        ["SUMO_EDGE_ID"]
        .tolist()
    )

    line = topis_lines[
        link_id
    ]

    new = fix_direction_oscillation(
        old,
        line
    )

    broken = count_broken(
        new
    )

    exact_dup = (
        len(new)
        - len(set(new))
    )

    physical = [
        physical_id(e)
        for e in new
    ]

    physical_dup = (
        len(physical)
        - len(set(physical))
    )

    replacement[
        link_id
    ] = new

    qc_rows.append({
        "TOPIS_LINK_ID":
            link_id,
        "OLD_EDGES":
            len(old),
        "NEW_EDGES":
            len(new),
        "DUPLICATES":
            exact_dup,
        "PHYSICAL_DUPLICATES":
            physical_dup,
        "BROKEN":
            len(broken),
        "BROKEN_PAIRS":
            str(broken)
    })

    print(
        link_id,
        "|",
        len(old),
        "->",
        len(new),
        "| physical dup =",
        physical_dup,
        "| broken =",
        len(broken)
    )


# ============================================================
# 9. 13개를 기존 mapping에 교체
# ============================================================

good_mapping = mapping[
    ~mapping[
        "TOPIS_LINK_ID"
    ].isin(direction_links)
].copy()

new_rows = []

for link_id, edges in (
    replacement.items()
):

    for seq, eid in enumerate(
        edges
    ):

        new_rows.append({
            "TOPIS_LINK_ID":
                link_id,
            "SUMO_EDGE_ID":
                eid,
            "SEQ":
                seq
        })


new_df = pd.DataFrame(
    new_rows
)

final = pd.concat(
    [
        good_mapping,
        new_df
    ],
    ignore_index=True
)

final = final.sort_values(
    [
        "TOPIS_LINK_ID",
        "SEQ"
    ]
)

final.to_csv(
    OUTPUT_FIXED,
    index=False,
    encoding="utf-8-sig"
)


fix_qc = pd.DataFrame(
    qc_rows
)

fix_qc.to_csv(
    OUTPUT_RECHECK,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. summary
# ============================================================

print(
    "\n========================"
)

print(
    "DIRECTION FIX SUMMARY"
)

print(
    "========================"
)

print(
    "processed:",
    len(direction_links)
)

print(
    "remaining physical duplicates:",
    fix_qc[
        "PHYSICAL_DUPLICATES"
    ].sum()
)

print(
    "broken links:",
    (
        fix_qc["BROKEN"] > 0
    ).sum()
)

print(
    "\nCreated:",
    OUTPUT_FIXED.name
)

print(
    "Created:",
    OUTPUT_RECHECK.name
)