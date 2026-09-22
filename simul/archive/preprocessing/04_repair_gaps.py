# -*- coding: utf-8 -*-

import os
import sys
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

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_direction_fixed.csv"
)

QC_FILE = (
    BASE / "direction_fix_qc.csv"
)

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT_FILE = (
    BASE /
    "topis_sumo_mapping_gap_repaired.csv"
)

OUTPUT_QC = (
    BASE /
    "gap_repair_qc.csv"
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

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 2. mapping + QC 읽기
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

broken_links = qc.loc[
    qc["BROKEN"] > 0,
    "TOPIS_LINK_ID"
].tolist()

print(
    "broken links:",
    len(broken_links)
)

print(broken_links)


# ============================================================
# 3. 문제 링크 TOPIS geometry
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
        broken_links
    )
].copy()


to_wgs84 = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


def topis_to_sumo(x, y):

    lon, lat = to_wgs84.transform(
        x,
        y
    )

    return net.convertLonLat2XY(
        lon,
        lat
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
# 4. connectivity
# ============================================================

def broken_pairs(edge_ids):

    broken = []

    for i, (a, b) in enumerate(
        zip(
            edge_ids[:-1],
            edge_ids[1:]
        )
    ):

        ea = net.getEdge(a)
        eb = net.getEdge(b)

        if (
            ea.getToNode().getID()
            !=
            eb.getFromNode().getID()
        ):

            broken.append(
                (i, a, b)
            )

    return broken


# ============================================================
# 5. path가 TOPIS line 근처인지 검사
# ============================================================

def path_geometry_error(
    edge_ids,
    topis_line
):

    distances = []

    for eid in edge_ids:

        edge = net.getEdge(eid)

        shape = edge.getShape()

        if len(shape) < 2:
            continue

        edge_line = LineString(
            shape
        )

        ds = list(
            np.arange(
                0,
                edge_line.length,
                5.0
            )
        )

        ds.append(
            edge_line.length
        )

        for d in ds:

            p = edge_line.interpolate(d)

            distances.append(
                p.distance(
                    topis_line
                )
            )

    if not distances:

        return 9999, 9999

    return (
        float(
            np.mean(distances)
        ),
        float(
            np.max(distances)
        )
    )


# ============================================================
# 6. 끊긴 pair 사이만 local shortest path
# ============================================================

def repair_one_link(
    edge_ids,
    topis_line
):

    result = list(
        edge_ids
    )

    # 앞에서부터 수리.
    # 수리할 때마다 다시 broken 검사함.
    max_iterations = 20

    for iteration in range(
        max_iterations
    ):

        broken = broken_pairs(
            result
        )

        if not broken:
            return result, True, []

        # 첫 번째 broken pair
        i, a, b = broken[0]

        ea = net.getEdge(a)
        eb = net.getEdge(b)

        shortest = net.getShortestPath(
            ea,
            eb
        )

        if (
            shortest is None
            or shortest[0] is None
        ):

            return (
                result,
                False,
                [
                    (
                        a,
                        b,
                        "NO_PATH"
                    )
                ]
            )

        path, cost = shortest

        path_ids = [
            e.getID()
            for e in path
        ]

        # path는 보통
        # [a, ..., b] 형태
        if len(path_ids) < 2:

            return (
                result,
                False,
                [
                    (
                        a,
                        b,
                        "EMPTY_PATH"
                    )
                ]
            )

        # 국소 경로가 TOPIS에서
        # 너무 멀리 빠지는지 검사
        mean_dist, max_dist = (
            path_geometry_error(
                path_ids,
                topis_line
            )
        )

        print(
            "   gap:",
            a,
            "->",
            b,
            "| inserted path =",
            len(path_ids),
            "| mean =",
            round(mean_dist, 1),
            "| max =",
            round(max_dist, 1)
        )

        # 상당히 보수적인 기준
        if (
            mean_dist > 20
            or max_dist > 60
        ):

            return (
                result,
                False,
                [
                    (
                        a,
                        b,
                        "PATH_TOO_FAR",
                        mean_dist,
                        max_dist
                    )
                ]
            )

        # a와 b는 이미 기존 리스트에 있으므로
        # 중간 edge만 삽입
        middle = path_ids[
            1:-1
        ]

        result = (
            result[:i + 1]
            + middle
            + result[i + 1:]
        )

    return (
        result,
        False,
        [
            (
                "MAX_ITERATIONS",
            )
        ]
    )


# ============================================================
# 7. 7개만 repair
# ============================================================

replacement = {}

qc_rows = []

for link_id in broken_links:

    print(
        "\n========================"
    )

    print(
        "TOPIS:",
        link_id
    )

    old = (
        mapping[
            mapping[
                "TOPIS_LINK_ID"
            ] == link_id
        ]
        .sort_values(
            "SEQ"
        )["SUMO_EDGE_ID"]
        .tolist()
    )

    line = topis_lines[
        link_id
    ]

    old_broken = broken_pairs(
        old
    )

    print(
        "before edges:",
        len(old)
    )

    print(
        "before broken:",
        len(old_broken)
    )

    new, success, errors = (
        repair_one_link(
            old,
            line
        )
    )

    new_broken = broken_pairs(
        new
    )

    mean_dist, max_dist = (
        path_geometry_error(
            new,
            line
        )
    )

    print(
        "after edges:",
        len(new)
    )

    print(
        "after broken:",
        len(new_broken)
    )

    print(
        "geometry:",
        round(mean_dist, 1),
        "/",
        round(max_dist, 1),
        "m"
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
        "OLD_BROKEN":
            len(old_broken),
        "NEW_BROKEN":
            len(new_broken),
        "MEAN_DIST":
            mean_dist,
        "MAX_DIST":
            max_dist,
        "SUCCESS":
            (
                success
                and len(new_broken) == 0
            ),
        "ERROR":
            str(errors)
    })


# ============================================================
# 8. 기존 mapping에서 7개 교체
# ============================================================

mapping_good = mapping[
    ~mapping[
        "TOPIS_LINK_ID"
    ].isin(
        broken_links
    )
].copy()

rows = []

for link_id, edge_ids in (
    replacement.items()
):

    for seq, eid in enumerate(
        edge_ids
    ):

        rows.append({
            "TOPIS_LINK_ID":
                link_id,
            "SUMO_EDGE_ID":
                eid,
            "SEQ":
                seq
        })


repair_df = pd.DataFrame(
    rows
)

final_mapping = pd.concat(
    [
        mapping_good,
        repair_df
    ],
    ignore_index=True
)

final_mapping = (
    final_mapping
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

final_mapping.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. QC
# ============================================================

repair_qc = pd.DataFrame(
    qc_rows
)

repair_qc.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================"
)

print(
    "GAP REPAIR SUMMARY"
)

print(
    "========================"
)

print(
    "processed:",
    len(repair_qc)
)

print(
    "success:",
    repair_qc[
        "SUCCESS"
    ].sum()
)

print(
    "remaining broken:",
    (
        repair_qc[
            "NEW_BROKEN"
        ] > 0
    ).sum()
)

print(
    "\nCreated:",
    OUTPUT_FILE.name
)

print(
    "Created:",
    OUTPUT_QC.name
)