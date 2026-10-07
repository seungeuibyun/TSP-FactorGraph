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
# 0. CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

# 10번에서 만든 최신 mapping
MAPPING_FILE = (
    BASE / "topis_sumo_mapping_corridor.csv"
)

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

OUTPUT_MAPPING = (
    BASE / "topis_sumo_mapping_final.csv"
)

OUTPUT_ROUTE = (
    BASE / "1070012500_anchor_route.csv"
)

OUTPUT_PLOT = (
    BASE / "1070012500_anchor_fix.png"
)


TARGET_LINK = "1070012500"


# ============================================================
# 1. PARAMETERS
# ============================================================

# START / 25 / 50 / 75 / END
ANCHOR_FRACTIONS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

# 후보 edge가 TOPIS에서 허용되는 거리
CORRIDOR = 100.0

# 각 anchor 근처 후보 탐색 반경
ANCHOR_RADIUS = 25.0

# anchor 하나당 후보 edge 수
N_CANDIDATES = 12

# 진행방향 차이
MAX_HEADING_ERROR = 85.0

# 뒤로 조금 가는 것은 junction 때문에 허용
BACKTRACK_TOL = 20.0

# geometry cost
DIST_WEIGHT = 4.0
HEADING_WEIGHT = 0.5

SAMPLE_SPACING = 10.0


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
# 3. TOPIS geometry 읽기
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
    vtx["LINK_ID"] == TARGET_LINK
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
    "VER_SEQ"
)


topis_line = LineString(
    list(
        zip(
            vtx["SUMO_X"],
            vtx["SUMO_Y"]
        )
    )
)

print(
    "TOPIS length:",
    round(
        topis_line.length,
        1
    ),
    "m"
)


# ============================================================
# 4. GEOMETRY HELPERS
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
        not ds
        or ds[-1] < line.length
    ):

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
    delta=10.0
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
# 5. SUMO EDGE와 TOPIS 관계
# ============================================================

def edge_metrics(edge):

    shape = edge.getShape()

    if len(shape) < 2:
        return None

    e_line = LineString(shape)

    midpoint = e_line.interpolate(
        0.5,
        normalized=True
    )

    s_mid = topis_line.project(
        midpoint
    )

    topis_h = local_heading(
        topis_line,
        s_mid,
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

        h_error = 180.0

    else:

        h_error = angle_diff(
            topis_h,
            edge_h
        )

    pts = sample_line(
        e_line,
        SAMPLE_SPACING
    )

    distances = [
        p.distance(
            topis_line
        )
        for p in pts
    ]

    progress = [
        topis_line.project(p)
        for p in pts
    ]

    return {
        "line":
            e_line,

        "mean_dist":
            float(
                np.mean(distances)
            ),

        "min_dist":
            float(
                e_line.distance(
                    topis_line
                )
            ),

        "heading_error":
            float(h_error),

        "s_mid":
            float(s_mid),

        "s_min":
            float(
                min(progress)
            ),

        "s_max":
            float(
                max(progress)
            ),
    }


# ============================================================
# 6. TOPIS corridor 내 directed edges
# ============================================================

def build_allowed_edges():

    allowed = {}

    minx, miny, maxx, maxy = (
        topis_line.bounds
    )

    margin = CORRIDOR + 50

    for edge in net.getEdges():

        shape = edge.getShape()

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

        m = edge_metrics(
            edge
        )

        if m is None:
            continue

        if (
            m["min_dist"] <= CORRIDOR
            and
            m["heading_error"]
            <= MAX_HEADING_ERROR
        ):

            allowed[
                edge.getID()
            ] = m

    return allowed


allowed = build_allowed_edges()

print(
    "allowed edges:",
    len(allowed)
)


# ============================================================
# 7. 5개 anchor
# ============================================================

anchors = []

for f in ANCHOR_FRACTIONS:

    s = (
        f
        * topis_line.length
    )

    p = topis_line.interpolate(
        s
    )

    anchors.append({
        "fraction": f,
        "s": s,
        "point": p,
    })


# ============================================================
# 8. 각 anchor 후보 edge
# ============================================================

def anchor_candidates(anchor):

    results = []

    point = anchor[
        "point"
    ]

    s_anchor = anchor[
        "s"
    ]

    for eid, m in allowed.items():

        d_xy = point.distance(
            m["line"]
        )

        if d_xy > ANCHOR_RADIUS:
            continue

        # edge의 TOPIS longitudinal 위치도
        # anchor에서 너무 멀면 제외
        d_s = abs(
            m["s_mid"]
            -
            s_anchor
        )
        if not (
            m["s_min"] - 15.0
            <= s_anchor
            <= m["s_max"] + 15.0
        ):
            continue

        # 공간거리 + 진행좌표 거리
        score = (
            d_xy
            +
            0.25 * d_s
            +
            0.2 * m[
                "heading_error"
            ]
        )

        results.append(
            (
                score,
                eid,
                d_xy,
                d_s
            )
        )

    results.sort(
        key=lambda x: x[0]
    )

    return results[
        :N_CANDIDATES
    ]


candidate_sets = []

for i, anchor in enumerate(
    anchors
):

    cands = anchor_candidates(
        anchor
    )

    candidate_sets.append(
        cands
    )

    print(
        f"\nAnchor {i}",
        f"({anchor['fraction']:.2f})",
        "| candidates =",
        len(cands)
    )

    for x in cands[:5]:

        print(
            " ",
            x[1],
            "| xy =",
            round(x[2], 1),
            "| ds =",
            round(x[3], 1)
        )


# ============================================================
# 9. Edge traversal cost
# ============================================================

def edge_cost(eid):

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
        * m["mean_dist"]
        +
        HEADING_WEIGHT
        * m["heading_error"]
    )


# ============================================================
# 10. 두 anchor 후보 사이 corridor Dijkstra
# ============================================================

def dijkstra_between(
    start_eid,
    end_eid,
    s_low,
    s_high
):

    if (
        start_eid not in allowed
        or end_eid not in allowed
    ):

        return (
            None,
            float("inf")
        )

    dist = {
        start_eid:
            edge_cost(start_eid)
    }

    parent = {
        start_eid:
            None
    }

    pq = [
        (
            dist[start_eid],
            start_eid
        )
    ]

    while pq:

        d, eid = heapq.heappop(
            pq
        )

        if d != dist[eid]:
            continue

        if eid == end_eid:

            route = []

            x = eid

            while x is not None:

                route.append(x)

                x = parent[x]

            route.reverse()

            return (
                route,
                d
            )

        edge = net.getEdge(
            eid
        )

        current_s = allowed[
            eid
        ]["s_mid"]

        for next_edge in (
            edge.getOutgoing().keys()
        ):

            nid = next_edge.getID()

            if nid not in allowed:
                continue

            next_s = allowed[
                nid
            ]["s_mid"]

            # 뒤로 크게 후퇴 금지
            if (
                next_s
                <
                current_s
                -
                BACKTRACK_TOL
            ):

                continue

            # 이 anchor segment의
            # 진행범위를 너무 벗어나지 않게
            if (
                next_s
                <
                s_low
                -
                40
            ):

                continue

            if (
                next_s
                >
                s_high
                +
                40
            ):

                continue

            nd = (
                d
                +
                edge_cost(nid)
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

    return (
        None,
        float("inf")
    )

# ============================================================
# DEBUG: consecutive anchor connectivity
# ============================================================

print("\n========================")
print("ANCHOR CONNECTIVITY DEBUG")
print("========================")

for i in range(1, len(anchors)):

    print(
        f"\n--- Anchor {i-1} -> {i} "
        f"({ANCHOR_FRACTIONS[i-1]:.2f}"
        f" -> {ANCHOR_FRACTIONS[i]:.2f}) ---"
    )

    s_low = anchors[i - 1]["s"]
    s_high = anchors[i]["s"]

    connected = 0
    total = 0

    for _, start_eid, _, _ in candidate_sets[i - 1]:

        for _, end_eid, _, _ in candidate_sets[i]:

            total += 1

            # --------------------------------
            # 1. 우리가 만든 constrained path
            # --------------------------------
            route, cost = dijkstra_between(
                start_eid,
                end_eid,
                s_low,
                s_high
            )

            if route is not None:

                connected += 1

                print(
                    "OK:",
                    start_eid,
                    "->",
                    end_eid,
                    "| edges =",
                    len(route)
                )

    print(
        "connected:",
        connected,
        "/",
        total
    )


# ============================================================
# DEBUG 2:
# Anchor 0 -> 1 실제 SUMO topology 확인
# ============================================================

print("\n========================")
print("NATIVE SUMO PATH CHECK")
print("========================")

c0 = candidate_sets[0]
c1 = candidate_sets[1]

forward_found = 0
reverse_found = 0


# ------------------------------------------------------------
# 0% -> 25%
# corridor 제약 없이 SUMO 자체 shortest path
# ------------------------------------------------------------

print("\n--- FORWARD: Anchor 0 -> Anchor 1 ---")

for _, start_eid, _, _ in c0:

    for _, end_eid, _, _ in c1:

        start_edge = net.getEdge(start_eid)
        end_edge = net.getEdge(end_eid)

        result = net.getShortestPath(
            start_edge,
            end_edge
        )

        if (
            result is not None
            and result[0] is not None
        ):

            path, cost = result

            path_ids = [
                e.getID()
                for e in path
            ]

            forward_found += 1

            print(
                "FORWARD OK:",
                start_eid,
                "->",
                end_eid,
                "| edges =",
                len(path_ids),
                "| length =",
                round(
                    sum(
                        e.getLength()
                        for e in path
                    ),
                    1
                ),
                "m"
            )

            print(
                " ",
                path_ids
            )


# ------------------------------------------------------------
# 반대로 25% -> 0%도 확인
# ------------------------------------------------------------

print("\n--- REVERSE: Anchor 1 -> Anchor 0 ---")

for _, start_eid, _, _ in c1:

    for _, end_eid, _, _ in c0:

        start_edge = net.getEdge(start_eid)
        end_edge = net.getEdge(end_eid)

        result = net.getShortestPath(
            start_edge,
            end_edge
        )

        if (
            result is not None
            and result[0] is not None
        ):

            path, cost = result

            path_ids = [
                e.getID()
                for e in path
            ]

            reverse_found += 1

            print(
                "REVERSE OK:",
                start_eid,
                "->",
                end_eid,
                "| edges =",
                len(path_ids),
                "| length =",
                round(
                    sum(
                        e.getLength()
                        for e in path
                    ),
                    1
                ),
                "m"
            )

            print(
                " ",
                path_ids
            )


print("\n========================")
print("TOPOLOGY SUMMARY")
print("========================")

print(
    "forward paths:",
    forward_found
)

print(
    "reverse paths:",
    reverse_found
)

sys.exit()


# ============================================================
# 11. Dynamic programming
#
# anchor 0 후보
# -> anchor 1 후보
# -> ...
# -> anchor 4 후보
#
# 전체 비용 최소 조합 탐색
# ============================================================

if any(
    len(c) == 0
    for c in candidate_sets
):

    raise RuntimeError(
        "At least one anchor has no candidate edge."
    )


dp = {}
parent_choice = {}


# 첫 anchor 후보
for _, eid, d_xy, d_s in (
    candidate_sets[0]
):

    dp[
        (0, eid)
    ] = (
        d_xy
        +
        0.25 * d_s
    )

    parent_choice[
        (0, eid)
    ] = None


segment_routes = {}


# anchor 1 -> 4
for i in range(
    1,
    len(anchors)
):

    s_low = anchors[
        i - 1
    ]["s"]

    s_high = anchors[
        i
    ]["s"]

    for (
        _,
        end_eid,
        end_xy,
        end_ds
    ) in candidate_sets[i]:

        best_cost = float(
            "inf"
        )

        best_prev = None
        best_route = None

        for (
            _,
            start_eid,
            _,
            _
        ) in candidate_sets[
            i - 1
        ]:

            prev_key = (
                i - 1,
                start_eid
            )

            if prev_key not in dp:
                continue

            route, route_cost = (
                dijkstra_between(
                    start_eid,
                    end_eid,
                    s_low,
                    s_high
                )
            )

            if route is None:
                continue

            total = (
                dp[prev_key]
                +
                route_cost
                +
                end_xy
                +
                0.25 * end_ds
            )

            if total < best_cost:

                best_cost = total
                best_prev = start_eid
                best_route = route

        if best_prev is not None:

            key = (
                i,
                end_eid
            )

            dp[key] = best_cost

            parent_choice[
                key
            ] = best_prev

            segment_routes[
                (
                    i,
                    best_prev,
                    end_eid
                )
            ] = best_route


# ============================================================
# 12. 마지막 anchor에서 최적 후보 선택
# ============================================================

last_i = (
    len(anchors) - 1
)

last_options = [
    (
        cost,
        eid
    )
    for (
        i,
        eid
    ),
    cost in dp.items()
    if i == last_i
]

if not last_options:

    raise RuntimeError(
        "No complete anchor-constrained route found."
    )


best_cost, best_last = min(
    last_options,
    key=lambda x: x[0]
)


# ============================================================
# 13. anchor edge sequence 역추적
# ============================================================

chosen_anchor_edges = [
    None
] * len(anchors)

chosen_anchor_edges[
    last_i
] = best_last


for i in range(
    last_i,
    0,
    -1
):

    current = (
        chosen_anchor_edges[i]
    )

    prev = parent_choice[
        (i, current)
    ]

    chosen_anchor_edges[
        i - 1
    ] = prev


print(
    "\nChosen anchor edges:"
)

for i, eid in enumerate(
    chosen_anchor_edges
):

    print(
        i,
        ANCHOR_FRACTIONS[i],
        eid
    )


# ============================================================
# 14. segment routes 합치기
# ============================================================

final_route = []

for i in range(
    1,
    len(anchors)
):

    a = chosen_anchor_edges[
        i - 1
    ]

    b = chosen_anchor_edges[
        i
    ]

    route = dijkstra_between(
        a,
        b,
        anchors[i - 1]["s"],
        anchors[i]["s"]
    )[0]

    if route is None:

        raise RuntimeError(
            f"No route between anchor {i-1} and {i}"
        )

    if not final_route:

        final_route.extend(
            route
        )

    else:

        # 첫 edge는 이전 segment의
        # 마지막 edge와 동일하므로 제외
        final_route.extend(
            route[1:]
        )


# 연속 동일 edge 제거
cleaned_route = []

for eid in final_route:

    if (
        not cleaned_route
        or
        cleaned_route[-1]
        != eid
    ):

        cleaned_route.append(
            eid
        )


final_route = cleaned_route


# ============================================================
# 15. QC
# ============================================================

def broken_count(edges):

    broken = 0

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

            broken += 1

    return broken


route_lines = [
    LineString(
        net.getEdge(eid)
        .getShape()
    )
    for eid in final_route
]


topis_pts = sample_line(
    topis_line,
    spacing=10
)

distances = [
    min(
        p.distance(r)
        for r in route_lines
    )
    for p in topis_pts
]


# 진짜 coverage:
# TOPIS sample point 각각이 route의
# 60m 이내에 있는 비율
COVERAGE_RADIUS = 25.0

point_coverage = (
    sum(
        d <= COVERAGE_RADIUS
        for d in distances
    )
    /
    len(distances)
)


route_length = sum(
    net.getEdge(eid)
    .getLength()
    for eid in final_route
)


print(
    "\n========================"
)

print(
    "ANCHOR FIX RESULT"
)

print(
    "========================"
)

print(
    "edges:",
    len(final_route)
)

print(
    "broken:",
    broken_count(
        final_route
    )
)

print(
    "mean distance:",
    round(
        np.mean(distances),
        1
    ),
    "m"
)

print(
    "max distance:",
    round(
        np.max(distances),
        1
    ),
    "m"
)

print(
    "length ratio:",
    round(
        route_length
        /
        topis_line.length,
        3
    )
)

print(
    "point coverage:",
    round(
        point_coverage,
        3
    )
)


# ============================================================
# 16. route CSV 저장
# ============================================================

route_df = pd.DataFrame({
    "TOPIS_LINK_ID":
        TARGET_LINK,

    "SUMO_EDGE_ID":
        final_route,

    "SEQ":
        range(
            len(final_route)
        )
})

route_df.to_csv(
    OUTPUT_ROUTE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 기존 corridor mapping에서
#     1070012500만 교체
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

mapping[
    "TOPIS_LINK_ID"
] = mapping[
    "TOPIS_LINK_ID"
].str.strip()


keep = mapping[
    mapping[
        "TOPIS_LINK_ID"
    ] != TARGET_LINK
].copy()


final_mapping = pd.concat(
    [
        keep,
        route_df
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
    OUTPUT_MAPPING,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)


minx, miny, maxx, maxy = (
    topis_line.bounds
)

margin = 120


for e in net.getEdges():

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
        alpha=0.2
    )


# final route
first = True

for eid in final_route:

    shape = net.getEdge(
        eid
    ).getShape()

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
        linewidth=3,
        alpha=0.9,
        label=(
            "Anchor route"
            if first
            else None
        )
    )

    first = False


# TOPIS
tx, ty = topis_line.xy

ax.plot(
    tx,
    ty,
    "k--",
    linewidth=2,
    label="TOPIS"
)


# anchors
for i, a in enumerate(
    anchors
):

    p = a["point"]

    ax.scatter(
        p.x,
        p.y,
        s=60,
        marker="o"
    )

    ax.annotate(
        f"{int(a['fraction'] * 100)}%",
        (
            p.x,
            p.y
        ),
        xytext=(5, 5),
        textcoords="offset points"
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
    "TOPIS 1070012500 - anchor constrained"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_PLOT,
    dpi=180
)

plt.show()


print(
    "\nCreated:",
    OUTPUT_ROUTE.name
)

print(
    "Created:",
    OUTPUT_MAPPING.name
)

print(
    "Created:",
    OUTPUT_PLOT.name
)