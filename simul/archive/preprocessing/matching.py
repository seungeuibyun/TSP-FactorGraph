import numpy as np
import math
import pandas as pd
from pyproj import Transformer
import os
import sys

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib

net = sumolib.net.readNet("seongbuk_buffer.net.xml")
# ------------------------------------
# 1. 성북구 TOPIS 속도 데이터
# ------------------------------------
speed = pd.read_excel(
    "topis_speed.xlsx",
    dtype={"링크아이디": str}
)

speed["링크아이디"] = speed["링크아이디"].str.strip()
speed_ids = set(speed["링크아이디"])

print("TOPIS speed links:", len(speed_ids))


# ------------------------------------
# 2. 서울 전체 LINK_VERTEX
# ------------------------------------
vtx = pd.read_excel(
    r"C:\Users\guild\Dropbox\연구\코드\TSP_new_new\simul\서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx",
    dtype={"LINK_ID": str}
)

vtx["LINK_ID"] = vtx["LINK_ID"].str.strip()

# ★ 성북구 속도 데이터에 있는 링크만 남김
vtx = vtx[vtx["LINK_ID"].isin(speed_ids)].copy()

print("matched vertex links:", vtx["LINK_ID"].nunique())
print("vertices:", len(vtx))

to_wgs84 = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)

def topis_to_sumo(x, y):
    lon, lat = to_wgs84.transform(x, y)
    sx, sy = net.convertLonLat2XY(lon, lat)
    return sx, sy

coords = [
    topis_to_sumo(x, y)
    for x, y in zip(vtx["GRS80TM_X"], vtx["GRS80TM_Y"])
]

vtx["SUMO_X"] = [p[0] for p in coords]
vtx["SUMO_Y"] = [p[1] for p in coords]

from shapely.geometry import LineString

vtx = vtx.sort_values(["LINK_ID", "VER_SEQ"])

topis_lines = {}

for link_id, group in vtx.groupby("LINK_ID"):

    coords = list(
        zip(
            group["SUMO_X"],
            group["SUMO_Y"]
        )
    )

    if len(coords) >= 2:
        topis_lines[link_id] = LineString(coords)


import numpy as np
from sumolib.route import mapTrace

link_id = "1070014000"
line = topis_lines[link_id]

# TOPIS 선을 10 m 간격의 좌표 trace로 만듦
distances = np.arange(
    5.0,
    max(5.1, line.length - 5.0),
    10.0
)

trace = [
    (line.interpolate(d).x, line.interpolate(d).y)
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

matched_edges = [e.getID() for e in route]

print("TOPIS:", link_id)
print("SUMO:", matched_edges)

for e in route:
    print(
        e.getID(),
        "| road =", e.getName(),
        "| length =", round(e.getLength(), 1),
        "| from =", e.getFromNode().getID(),
        "| to =", e.getToNode().getID()
    )

topis_geom_length = line.length
sumo_route_length = sum(e.getLength() for e in route)

print("TOPIS geometry length:", round(topis_geom_length, 1), "m")
print("SUMO route length:", round(sumo_route_length, 1), "m")
print("ratio:", round(sumo_route_length / topis_geom_length, 3))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# 주변 SUMO 전체 도로
for e in net.getEdges():
    shape = e.getShape()

    if len(shape) >= 2:
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, linewidth=0.4, alpha=0.3)

# TOPIS 원본 선
x, y = line.xy
ax.plot(x, y, linewidth=4, alpha=0.6, label="TOPIS")

# mapTrace가 선택한 SUMO edges
for e in route:
    shape = e.getShape()

    xs = [p[0] for p in shape]
    ys = [p[1] for p in shape]

    ax.plot(
        xs, ys,
        linewidth=2.5,
        alpha=0.9
    )

# 해당 링크 주변만 확대
minx, miny, maxx, maxy = line.bounds
margin = 100

ax.set_xlim(minx - margin, maxx + margin)
ax.set_ylim(miny - margin, maxy + margin)

ax.set_aspect("equal")
plt.show()

def match_link(line):

    distances = np.arange(
        5.0,
        max(5.1, line.length - 5.0),
        10.0
    )

    trace = [
        (line.interpolate(d).x,
         line.interpolate(d).y)
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

    return [e.getID() for e in route]

rows = []

total = len(topis_lines)

for idx, (link_id, line) in enumerate(topis_lines.items(), 1):
    print(f"[{idx}/{total}] matching {link_id}...")

    edges = match_link(line)

    for seq, edge_id in enumerate(edges):
        rows.append({
            "TOPIS_LINK_ID": link_id,
            "SUMO_EDGE_ID": edge_id,
            "SEQ": seq
        })

mapping_df = pd.DataFrame(rows)

mapping_df.to_csv(
    "topis_sumo_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)

print("DONE")

print(mapping_df)
print("TOPIS links:", mapping_df["TOPIS_LINK_ID"].nunique())

print(link_id)


bad_links = [
"1050021500",
"1070004700",
"1070005900",
"1070013300",
"1070013800",
"1070014900",
"1070015000",
"1070015300",
"1070015400",
"1070015700",
"1070016600",
"1070016900",
"1070017000",
"1070017700",
"1070017800",
"1070018200",
"1070018400",
"1070018700",
"1070018800",
"1070019100",
"1070019200",
"1070019500",
"1070019600",
"1070020802",
"1070021300",
"1070021400",
"1070022000",
"1080001000"
]


def rematch_link(line):

    # 더 촘촘하게 5 m 간격
    distances = np.arange(
        5.0,
        max(5.1, line.length - 5.0),
        5.0
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
        delta=20,              # 기존 30 -> 20 m
        fillGaps=30,           # 기존 100 -> 30 m
        direction=True,
        reversalPenalty=10000  # 역방향 전환 강하게 억제
    )

    return [e.getID() for e in route]

for link_id in bad_links:

    edges = rematch_link(
        topis_lines[link_id]
    )

    n_duplicate = len(edges) - len(set(edges))

    print(
        link_id,
        "| edges =", len(edges),
        "| duplicates =", n_duplicate
    )

from collections import defaultdict

remaining = [
    "1070013300",
    "1070015000",
    "1070016600",
    "1070016900",
    "1070018200",
    "1070018400",
    "1080001000"
]

for link_id in remaining:

    edges = rematch_link(topis_lines[link_id])

    pos = defaultdict(list)

    for i, eid in enumerate(edges):
        pos[eid].append(i)

    repeated = {
        eid: idxs
        for eid, idxs in pos.items()
        if len(idxs) > 1
    }

    print("\n============================")
    print("TOPIS:", link_id)
    print("EDGE SEQUENCE:")
    print(edges)
    print("REPEATED:")
    print(repeated)

import math

def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def get_heading(p1, p2):
    return math.degrees(
        math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0]
        )
    ) % 360


def clean_direction(edges, line, net):

    # TOPIS 링크 전체 진행방향
    p1 = line.interpolate(0.05 * line.length)
    p2 = line.interpolate(0.95 * line.length)

    topis_heading = get_heading(
        (p1.x, p1.y),
        (p2.x, p2.y)
    )

    cleaned = []

    for eid in edges:

        e = net.getEdge(eid)
        shape = e.getShape()

        if len(shape) < 2:
            continue

        edge_heading = get_heading(
            shape[0],
            shape[-1]
        )

        # TOPIS 진행방향과 90도 이상 반대면 제거
        if angle_diff(
            topis_heading,
            edge_heading
        ) >= 90:
            continue

        # 연속 중복 방지
        if not cleaned or cleaned[-1] != eid:
            cleaned.append(eid)

    return cleaned

link_id = "1080001000"

edges = rematch_link(topis_lines[link_id])

cleaned = clean_direction(
    edges,
    topis_lines[link_id],
    net
)

print("BEFORE:")
print(edges)

print("\nAFTER:")
print(cleaned)

remaining = [
    "1070013300",
    "1070015000",
    "1070016600",
    "1070016900",
    "1070018200",
    "1070018400",
    "1080001000"
]

cleaned_results = {}

for link_id in remaining:

    # 재매칭
    edges = rematch_link(topis_lines[link_id])

    # 방향 정리
    cleaned = clean_direction(
        edges,
        topis_lines[link_id],
        net
    )

    cleaned_results[link_id] = cleaned

    duplicates = len(cleaned) - len(set(cleaned))

    # edge들이 실제로 순서대로 이어지는지도 검사
    broken = []

    for e1_id, e2_id in zip(cleaned[:-1], cleaned[1:]):

        e1 = net.getEdge(e1_id)
        e2 = net.getEdge(e2_id)

        if e1.getToNode().getID() != e2.getFromNode().getID():
            broken.append((e1_id, e2_id))

    print(
        link_id,
        "| edges =", len(cleaned),
        "| duplicates =", duplicates,
        "| broken =", len(broken)
    )

    import math
from shapely.geometry import LineString


def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def heading_xy(p1, p2):
    return math.degrees(
        math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0]
        )
    ) % 360


def tangent_heading(line, d, delta=5.0):
    d1 = max(0, d - delta)
    d2 = min(line.length, d + delta)

    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)

    return heading_xy(
        (p1.x, p1.y),
        (p2.x, p2.y)
    )


def clean_direction_local(edges, topis_line, net):

    cleaned = []

    for eid in edges:

        edge = net.getEdge(eid)
        shape = edge.getShape()

        if len(shape) < 2:
            continue

        edge_line = LineString(shape)

        # SUMO edge의 중간 위치
        midpoint = edge_line.interpolate(
            0.5,
            normalized=True
        )

        # 그 위치가 TOPIS line의 어디쯤인지
        d_topis = topis_line.project(midpoint)

        # 그 위치의 TOPIS 진행방향
        topis_heading = tangent_heading(
            topis_line,
            d_topis
        )

        # SUMO edge 자체 진행방향
        edge_heading = tangent_heading(
            edge_line,
            edge_line.length / 2,
            delta=min(5.0, edge_line.length / 3)
        )

        # 실제로 반대 방향인 edge만 제거
        if angle_diff(topis_heading, edge_heading) > 100:
            continue

        if not cleaned or cleaned[-1] != eid:
            cleaned.append(eid)

    return cleaned

still_bad = [
    "1070013300",
    "1070016900",
    "1070018200",
    "1070018400"
]

for link_id in still_bad:

    before = rematch_link(
        topis_lines[link_id]
    )

    after = clean_direction_local(
        before,
        topis_lines[link_id],
        net
    )

    duplicates = len(after) - len(set(after))

    broken = []

    for e1_id, e2_id in zip(after[:-1], after[1:]):

        e1 = net.getEdge(e1_id)
        e2 = net.getEdge(e2_id)

        if e1.getToNode().getID() != e2.getFromNode().getID():
            broken.append(
                (e1_id, e2_id)
            )

    print(
        link_id,
        "| before =", len(before),
        "| after =", len(after),
        "| duplicates =", duplicates,
        "| broken =", len(broken)
    )

    if broken:
        print(" broken pairs:", broken)