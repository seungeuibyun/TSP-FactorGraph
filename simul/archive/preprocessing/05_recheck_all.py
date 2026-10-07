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

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_after_rematch.csv"
)

SPEED_FILE = BASE / "topis_speed.xlsx"

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT_QC = (
    BASE / "final_mapping_qc_v2.csv"
)

OUTPUT_REVIEW = (
    BASE / "final_mapping_review_v2.csv"
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
# 2. MAPPING
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
# 3. TOPIS LINK IDS
# ============================================================

speed = pd.read_excel(
    SPEED_FILE,
    dtype={
        "링크아이디": str
    }
)

speed["링크아이디"] = (
    speed["링크아이디"]
    .str.strip()
)

speed_ids = set(
    speed["링크아이디"]
)


# ============================================================
# 4. TOPIS GEOMETRY
# ============================================================

vtx = pd.read_excel(
    VERTEX_FILE,
    dtype={
        "LINK_ID": str
    }
)

vtx["LINK_ID"] = (
    vtx["LINK_ID"]
    .str.strip()
)

vtx = vtx[
    vtx["LINK_ID"].isin(
        speed_ids
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
    [
        "LINK_ID",
        "VER_SEQ"
    ]
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
# 5. HELPERS
# ============================================================

def physical_id(eid):

    if eid.startswith("-"):
        return eid[1:]

    return eid


def sample_line(
    line,
    spacing=10.0
):

    if line.length <= 0:
        return []

    ds = list(
        np.arange(
            0,
            line.length,
            spacing
        )
    )

    if (
        len(ds) == 0
        or ds[-1] != line.length
    ):
        ds.append(
            line.length
        )

    return [
        line.interpolate(d)
        for d in ds
    ]


def route_geometry(
    edge_ids
):

    lines = []

    for eid in edge_ids:

        e = net.getEdge(eid)

        shape = e.getShape()

        if len(shape) >= 2:

            lines.append(
                LineString(shape)
            )

    if not lines:
        return None

    return unary_union(
        lines
    )


def broken_pairs(
    edge_ids
):

    broken = []

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

            broken.append(
                (a, b)
            )

    return broken


# ============================================================
# 6. TOPIS -> ROUTE DISTANCE
#
# TOPIS 전체가 matched route에 잘 덮이는지 검사
# ============================================================

def topis_to_route_error(
    topis_line,
    edge_ids
):

    rg = route_geometry(
        edge_ids
    )

    if rg is None:
        return 9999, 9999

    points = sample_line(
        topis_line,
        spacing=10
    )

    dist = [
        p.distance(rg)
        for p in points
    ]

    return (
        float(np.mean(dist)),
        float(np.max(dist))
    )


# ============================================================
# 7. INTERIOR ROUTE -> TOPIS
#
# 핵심 수정:
# first / last edge는 제외한다.
#
# 이유:
# SUMO edge가 TOPIS link보다 길 수 있어서
# boundary edge 전체를 비교하면 false positive가 생김.
#
# 대신 중간 edge가 엉뚱한 도로로 우회하는지만 검사.
# ============================================================

def interior_route_to_topis_error(
    topis_line,
    edge_ids
):

    # 1~2 edge 매핑이면
    # interior 자체가 없음.
    if len(edge_ids) <= 2:

        return (
            np.nan,
            np.nan,
            0
        )

    interior = edge_ids[
        1:-1
    ]

    dist = []

    for eid in interior:

        e = net.getEdge(eid)

        shape = e.getShape()

        if len(shape) < 2:
            continue

        line = LineString(
            shape
        )

        points = sample_line(
            line,
            spacing=10
        )

        for p in points:

            dist.append(
                p.distance(
                    topis_line
                )
            )

    if len(dist) == 0:

        return (
            np.nan,
            np.nan,
            len(interior)
        )

    return (
        float(np.mean(dist)),
        float(np.max(dist)),
        len(interior)
    )


# ============================================================
# 8. ROUTE LENGTH
#
# 극단적인 shortest-path detour를 확인하기 위한
# 보조 지표.
# ============================================================

def route_length(
    edge_ids
):

    total = 0.0

    for eid in edge_ids:

        total += (
            net.getEdge(
                eid
            ).getLength()
        )

    return total


# ============================================================
# 9. EVALUATE
# ============================================================

def evaluate(
    link_id,
    edge_ids
):

    topis_line = (
        topis_lines[
            link_id
        ]
    )

    # exact duplicate
    duplicate = (
        len(edge_ids)
        -
        len(set(edge_ids))
    )

    # +/- same road
    physical = [
        physical_id(e)
        for e in edge_ids
    ]

    physical_duplicate = (
        len(physical)
        -
        len(set(physical))
    )

    # connectivity
    broken = broken_pairs(
        edge_ids
    )

    # TOPIS -> route
    mean_t2r, max_t2r = (
        topis_to_route_error(
            topis_line,
            edge_ids
        )
    )

    # interior route -> TOPIS
    (
        mean_interior,
        max_interior,
        n_interior
    ) = (
        interior_route_to_topis_error(
            topis_line,
            edge_ids
        )
    )

    tlen = (
        topis_line.length
    )

    rlen = route_length(
        edge_ids
    )

    length_ratio = (
        rlen / tlen
        if tlen > 0
        else np.nan
    )

    # --------------------------------
    # DETOUR
    # --------------------------------

    detour = False

    # interior edge가 있을 때만
    # route -> TOPIS 검사
    if n_interior > 0:

        if (
            not np.isnan(
                mean_interior
            )
            and (
                mean_interior > 20
                or
                max_interior > 80
            )
        ):

            detour = True

    # 극단적으로 긴 route도 review
    #
    # 단, 1~2 edge는 SUMO edge 자체가
    # 길 수 있으므로 length ratio로
    # fail시키지 않는다.
    if (
        len(edge_ids) >= 3
        and length_ratio > 3.0
    ):

        detour = True

    # --------------------------------
    # PASS
    # --------------------------------

    passed = (
        duplicate == 0
        and
        physical_duplicate == 0
        and
        len(broken) == 0

        and
        mean_t2r <= 15

        and
        max_t2r <= 50

        and
        not detour
    )

    return {
        "TOPIS_LINK_ID":
            link_id,

        "N_EDGES":
            len(edge_ids),

        "DUPLICATES":
            duplicate,

        "PHYSICAL_DUPLICATES":
            physical_duplicate,

        "BROKEN":
            len(broken),

        "MEAN_TOPIS_TO_ROUTE":
            mean_t2r,

        "MAX_TOPIS_TO_ROUTE":
            max_t2r,

        "N_INTERIOR":
            n_interior,

        "MEAN_INTERIOR_TO_TOPIS":
            mean_interior,

        "MAX_INTERIOR_TO_TOPIS":
            max_interior,

        "TOPIS_LENGTH":
            tlen,

        "ROUTE_LENGTH":
            rlen,

        "LENGTH_RATIO":
            length_ratio,

        "DETOUR":
            detour,

        "PASS":
            passed
    }


# ============================================================
# 10. RUN ALL 183
# ============================================================

rows = []

link_ids = sorted(
    mapping[
        "TOPIS_LINK_ID"
    ].unique()
)

for i, link_id in enumerate(
    link_ids,
    1
):

    edge_ids = (
        mapping[
            mapping[
                "TOPIS_LINK_ID"
            ] == link_id
        ]
        .sort_values(
            "SEQ"
        )[
            "SUMO_EDGE_ID"
        ]
        .tolist()
    )

    q = evaluate(
        link_id,
        edge_ids
    )

    rows.append(q)

    print(
        f"[{i}/{len(link_ids)}]",
        link_id,
        "|",
        "PASS" if q["PASS"]
        else "REVIEW"
    )


qc = pd.DataFrame(
    rows
)


# ============================================================
# 11. FAIL REASON
# ============================================================

def fail_reason(row):

    reasons = []

    if row[
        "DUPLICATES"
    ] > 0:

        reasons.append(
            "duplicate"
        )

    if row[
        "PHYSICAL_DUPLICATES"
    ] > 0:

        reasons.append(
            "direction_oscillation"
        )

    if row[
        "BROKEN"
    ] > 0:

        reasons.append(
            "broken"
        )

    if row[
        "MEAN_TOPIS_TO_ROUTE"
    ] > 15:

        reasons.append(
            "mean_distance"
        )

    if row[
        "MAX_TOPIS_TO_ROUTE"
    ] > 50:

        reasons.append(
            "max_distance"
        )

    if row[
        "DETOUR"
    ]:

        reasons.append(
            "detour"
        )

    return ", ".join(
        reasons
    )


qc[
    "FAIL_REASON"
] = qc.apply(
    fail_reason,
    axis=1
)


qc.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


review = qc[
    ~qc["PASS"]
].copy()

review.to_csv(
    OUTPUT_REVIEW,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "FINAL QC V2"
)

print(
    "=============================="
)

print(
    "links:",
    len(qc)
)

print(
    "PASS:",
    qc[
        "PASS"
    ].sum()
)

print(
    "REVIEW:",
    len(review)
)


print(
    "\n===== REVIEW LINKS =====\n"
)

cols = [
    "TOPIS_LINK_ID",
    "N_EDGES",
    "MEAN_TOPIS_TO_ROUTE",
    "MAX_TOPIS_TO_ROUTE",
    "MEAN_INTERIOR_TO_TOPIS",
    "MAX_INTERIOR_TO_TOPIS",
    "LENGTH_RATIO",
    "FAIL_REASON"
]

if len(review) > 0:

    print(
        review[
            cols
        ].to_string(
            index=False
        )
    )


print(
    "\n===== REASON COUNTS =====\n"
)

if len(review) > 0:

    print(
        review[
            "FAIL_REASON"
        ].value_counts()
    )