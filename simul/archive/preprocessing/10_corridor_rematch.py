# -*- coding: utf-8 -*-

import os
import sys
import math
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import LineString


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_after_rematch.csv"
)

REVIEW_FILE = (
    BASE / "final_mapping_review_v2.csv"
)

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT_MAPPING = (
    BASE /
    "topis_sumo_mapping_corridor.csv"
)

OUTPUT_QC = (
    BASE /
    "corridor_rematch_qc.csv"
)

PLOT_DIR = (
    BASE /
    "corridor_review"
)

PLOT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 1. PARAMETERS
# ============================================================

# divided highway까지 포함하기 위해 넉넉하게
CORRIDOR = 100.0

# 시작/끝 edge 탐색 반경
ENDPOINT_RADIUS = 100.0

# TOPIS 방향과 90도 이상 다르면 제외
MAX_HEADING_ERROR = 85.0

# cost weights
DIST_WEIGHT = 4.0
HEADING_WEIGHT = 0.5

# edge shape sampling
SAMPLE_SPACING = 10.0

ENDPOINT_PROGRESS_TOL = 25.0

BACKTRACK_TOL = 20.0

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

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 3. REVIEW LINKS
# ============================================================

review = pd.read_csv(
    REVIEW_FILE,
    dtype={
        "TOPIS_LINK_ID": str
    }
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
# 4. EXISTING MAPPING
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
# 5. TOPIS GEOMETRY
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
    p[0]
    for p in coords
]

vtx["SUMO_Y"] = [
    p[1]
    for p in coords
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
# 6. GEOMETRY HELPERS
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

    d = (
        abs(a - b)
        % 360
    )

    return min(
        d,
        360 - d
    )


def sample_line(
    line,
    spacing=10
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

    ds.append(
        line.length
    )

    return [
        line.interpolate(d)
        for d in ds
    ]


def local_heading(
    line,
    d,
    delta=10
):

    d1 = max(
        0,
        d - delta
    )

    d2 = min(
        line.length,
        d + delta
    )

    if d2 <= d1:

        return None

    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)

    return heading(
        (p1.x, p1.y),
        (p2.x, p2.y)
    )


# ============================================================
# 7. EDGE METRICS RELATIVE TO TOPIS
# ============================================================

def edge_metrics(
    edge,
    topis_line
):

    shape = edge.getShape()

    if len(shape) < 2:
        return None

    e_line = LineString(
        shape
    )

    midpoint = e_line.interpolate(
        0.5,
        normalized=True
    )

    # --------------------------------
    # nearest position on TOPIS
    # --------------------------------

    d_topis = topis_line.project(
        midpoint
    )

    topis_h = local_heading(
        topis_line,
        d_topis,
        delta=10
    )

    edge_h = local_heading(
        e_line,
        e_line.length / 2,
        delta=min(
            10,
            max(
                1,
                e_line.length / 3
            )
        )
    )

    if (
        topis_h is None
        or edge_h is None
    ):

        h_error = 180

    else:

        h_error = angle_diff(
            topis_h,
            edge_h
        )

    # --------------------------------
    # edge sampling
    # --------------------------------

    points = sample_line(
        e_line,
        spacing=SAMPLE_SPACING
    )

    distances = [
        p.distance(
            topis_line
        )
        for p in points
    ]

    # ★ TOPIS를 따라 진행한 위치
    progress = [
        topis_line.project(p)
        for p in points
    ]

    min_dist = e_line.distance(
        topis_line
    )

    return {
        "line":
            e_line,

        "min_dist":
            float(min_dist),

        "mean_dist":
            float(
                np.mean(distances)
            ),

        "heading_error":
            float(h_error),

        # ★ 새로 추가
        "s_min":
            float(
                np.min(progress)
            ),

        "s_max":
            float(
                np.max(progress)
            ),

        "s_mid":
            float(d_topis)
    }

# ============================================================
# 8. BUILD CORRIDOR SUBGRAPH
# ============================================================

def build_allowed_edges(
    topis_line
):

    allowed = {}

    minx, miny, maxx, maxy = (
        topis_line.bounds
    )

    # bbox로 먼저 거름
    margin = (
        CORRIDOR + 50
    )

    for edge in net.getEdges():

        shape = edge.getShape()

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

        if (
            max(xs)
            < minx - margin

            or min(xs)
            > maxx + margin

            or max(ys)
            < miny - margin

            or min(ys)
            > maxy + margin
        ):

            continue

        m = edge_metrics(
            edge,
            topis_line
        )

        if m is None:
            continue

        # TOPIS corridor 안에 있고
        # 진행방향도 맞는 edge만
        if (
            m["min_dist"]
            <= CORRIDOR

            and
            m["heading_error"]
            <= MAX_HEADING_ERROR
        ):

            allowed[
                edge.getID()
            ] = m

    return allowed


# ============================================================
# 9. START / END EDGE CANDIDATES
# ============================================================

def get_endpoint_candidates(
    topis_line,
    allowed
):

    L = topis_line.length

    start_p = topis_line.interpolate(
        0
    )

    end_p = topis_line.interpolate(
        L
    )

    starts = []
    ends = []

    for eid, m in allowed.items():

        e_line = m["line"]

        # --------------------------------
        # START 후보
        #
        # edge가 TOPIS 시작부를 실제로
        # 커버해야 함
        # --------------------------------

        if (
            m["s_min"]
            <= ENDPOINT_PROGRESS_TOL
        ):

            ds = start_p.distance(
                e_line
            )

            if ds <= CORRIDOR:

                starts.append(
                    (
                        eid,
                        ds
                    )
                )

        # --------------------------------
        # END 후보
        # --------------------------------

        if (
            m["s_max"]
            >=
            L - ENDPOINT_PROGRESS_TOL
        ):

            de = end_p.distance(
                e_line
            )

            if de <= CORRIDOR:

                ends.append(
                    (
                        eid,
                        de
                    )
                )

    starts = sorted(
        starts,
        key=lambda x: x[1]
    )[:20]

    ends = sorted(
        ends,
        key=lambda x: x[1]
    )[:20]

    return (
        starts,
        ends
    )


# ============================================================
# 10. EDGE COST
# ============================================================

def edge_cost(
    eid,
    allowed
):

    e = net.getEdge(
        eid
    )

    m = allowed[
        eid
    ]

    return (
        e.getLength()

        +
        DIST_WEIGHT
        *
        m["mean_dist"]

        +
        HEADING_WEIGHT
        *
        m["heading_error"]
    )


# ============================================================
# 11. CORRIDOR-CONSTRAINED DIJKSTRA
# ============================================================

def corridor_path(
    topis_line
):

    allowed = build_allowed_edges(
        topis_line
    )

    starts, ends = (
        get_endpoint_candidates(
            topis_line,
            allowed
        )
    )

    if (
        len(starts) == 0
        or len(ends) == 0
    ):

        return (
            None,
            "NO_ENDPOINT_CANDIDATE"
        )

    end_set = {
        eid
        for eid, _
        in ends
    }

    end_penalty = {
        eid: d
        for eid, d
        in ends
    }

    dist = {}
    parent = {}

    pq = []

    # 여러 start edge를 동시에 시작
    for eid, endpoint_dist in starts:

        c = (
            endpoint_dist
            +
            edge_cost(
                eid,
                allowed
            )
        )

        if (
            eid not in dist
            or c < dist[eid]
        ):

            dist[eid] = c

            parent[eid] = None

            heapq.heappush(
                pq,
                (
                    c,
                    eid
                )
            )

    best_end = None
    best_total = float(
        "inf"
    )

    while pq:

        d,eid = heapq.heappop(
            pq
        )

        if d != dist[eid]:
            continue

        # end 후보
        if eid in end_set:

            total = (
                d
                +
                end_penalty[eid]
            )

            if total < best_total:

                best_total = total
                best_end = eid

        # 이미 현재 최선보다 비싸면 stop
        if d > best_total:
            break

        edge = net.getEdge(
            eid
        )

        outgoing = (
            edge.getOutgoing()
        )

        for next_edge in outgoing.keys():

            nid = next_edge.getID()

            if nid not in allowed:
                continue

            # --------------------------------
            # TOPIS 진행 방향으로 전진해야 함
            # --------------------------------

            current_s = allowed[
                eid
            ]["s_mid"]

            next_s = allowed[
                nid
            ]["s_mid"]

            if (
                next_s
                <
                current_s
                - BACKTRACK_TOL
            ):
                continue

            nd = (
                d
                +
                edge_cost(
                    nid,
                    allowed
                )
            )

            if (
                nid not in dist
                or nd < dist[nid]
            ):

                dist[nid] = nd

                parent[nid] = eid

                heapq.heappush(
                    pq,
                    (
                        nd,
                        nid
                    )
                )

    if best_end is None:

        return (
            None,
            "NO_CONNECTED_PATH"
        )

    # backtracking
    route = []

    x = best_end

    while x is not None:

        route.append(
            x
        )

        x = parent[
            x
        ]

    route.reverse()

    return (
        route,
        "OK"
    )


# ============================================================
# 12. QC
# ============================================================

def route_qc(
    topis_line,
    edge_ids
):

    if not edge_ids:

        return {
            "n_edges": 0,
            "broken": 999,
            "mean_dist": 9999,
            "max_dist": 9999,
            "route_length": 9999
        }

    broken = 0

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

            broken += 1

    # TOPIS points -> route
    route_lines = [
        LineString(
            net.getEdge(eid)
            .getShape()
        )
        for eid in edge_ids
    ]

    topis_points = sample_line(
        topis_line,
        spacing=10
    )

    # --------------------------------
    # TOPIS longitudinal coverage
    # --------------------------------

    progress = []

    for eid in edge_ids:

        e_line = LineString(
            net.getEdge(
                eid
            ).getShape()
        )

        for p in sample_line(
            e_line,
            spacing=10
        ):

            progress.append(
                topis_line.project(p)
            )

    if progress:

        s_min = min(progress)
        s_max = max(progress)

        coverage_ratio = (
            (s_max - s_min)
            /
            topis_line.length
        )

    else:

        coverage_ratio = 0.0

    dist = []

    for p in topis_points:

        dist.append(
            min(
                p.distance(r)
                for r
                in route_lines
            )
        )

    route_length = sum(
        net.getEdge(eid)
        .getLength()
        for eid in edge_ids
    )

    return {
        "n_edges":
            len(edge_ids),

        "broken":
            broken,

        "mean_dist":
            float(
                np.mean(dist)
            ),

        "max_dist":
            float(
                np.max(dist)
            ),

        "route_length":
            route_length,

        "length_ratio":
            route_length
            /
            topis_line.length,

        "coverage_ratio":
            coverage_ratio
    }


# ============================================================
# 13. RUN 12 LINKS
# ============================================================

replacement = {}
qc_rows = []


for idx, link_id in enumerate(
    review_links,
    1
):

    print(
        "\n========================"
    )

    print(
        f"[{idx}/{len(review_links)}]",
        link_id
    )

    line = topis_lines[
        link_id
    ]

    route, status = corridor_path(
        line
    )

    if route is None:

        print(
            "STATUS:",
            status
        )

        qc_rows.append({
            "TOPIS_LINK_ID":
                link_id,

            "STATUS":
                status
        })

        continue

    q = route_qc(
        line,
        route
    )

    replacement[
        link_id
    ] = route

    print(
        "edges =",
        q["n_edges"],
        "| broken =",
        q["broken"],
        "| mean =",
        round(q["mean_dist"], 1),
        "| max =",
        round(q["max_dist"], 1),
        "| ratio =",
        round(q["length_ratio"], 2),
        "| coverage =",
        round(q["coverage_ratio"], 3)
    )

    qc_rows.append({
        "TOPIS_LINK_ID":
            link_id,

        "STATUS":
            status,

        **q
    })


# ============================================================
# 14. REPLACE ONLY SUCCESSFUL LINKS
# ============================================================

successful_links = set(
    replacement.keys()
)

keep = mapping[
    ~mapping[
        "TOPIS_LINK_ID"
    ].isin(
        successful_links
    )
].copy()


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


new_df = pd.DataFrame(
    rows
)


final = pd.concat(
    [
        keep,
        new_df
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
    OUTPUT_MAPPING,
    index=False,
    encoding="utf-8-sig"
)


qc_df = pd.DataFrame(
    qc_rows
)

qc_df.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================"
)

print(
    "CORRIDOR REMATCH SUMMARY"
)

print(
    "========================"
)

print(
    "requested:",
    len(review_links)
)

print(
    "successful:",
    len(successful_links)
)

print(
    "\nCreated:",
    OUTPUT_MAPPING.name
)

print(
    "Created:",
    OUTPUT_QC.name
)

import pandas as pd

qc = pd.read_csv(
    "corridor_rematch_qc.csv",
    dtype={"TOPIS_LINK_ID": str}
)

print(
    qc[
        [
            "TOPIS_LINK_ID",
            "STATUS",
            "n_edges",
            "broken",
            "mean_dist",
            "max_dist",
            "length_ratio"
        ]
    ].to_string(index=False)
)