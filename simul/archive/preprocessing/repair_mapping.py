import os
import sys
import numpy as np
import pandas as pd

from pyproj import Transformer
from shapely.geometry import LineString
from sumolib.route import mapTrace


# ============================================================
# 0. SUMO network
# ============================================================

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib

net = sumolib.net.readNet("seongbuk_buffer.net.xml")


# ============================================================
# 1. 지금 남아 있는 문제 링크 4개
# ============================================================

problem4 = [
    "1070013300",
    "1070016900",
    "1070018200",
    "1070018400",
]


# ============================================================
# 2. TOPIS vertex 데이터
# ============================================================

vtx = pd.read_excel(
    r"C:\Users\guild\Dropbox\연구\코드\TSP_new_new\simul"
    r"\서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx",
    dtype={"LINK_ID": str}
)

vtx["LINK_ID"] = vtx["LINK_ID"].str.strip()

# ★ 서울 전체 34,000개 vertex 필요 없음
# 문제 4개 링크 vertex만 가져옴
vtx = vtx[
    vtx["LINK_ID"].isin(problem4)
].copy()

print(
    "loaded TOPIS links:",
    vtx["LINK_ID"].nunique()
)


# ============================================================
# 3. GRS80 TM -> SUMO XY
# ============================================================

to_wgs84 = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


def topis_to_sumo(x, y):

    lon, lat = to_wgs84.transform(x, y)

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


# ============================================================
# 4. TOPIS LineString 생성
# ============================================================

vtx = vtx.sort_values(
    ["LINK_ID", "VER_SEQ"]
)

topis_lines = {}

for link_id, group in vtx.groupby("LINK_ID"):

    xy = list(
        zip(
            group["SUMO_X"],
            group["SUMO_Y"]
        )
    )

    if len(xy) >= 2:
        topis_lines[link_id] = LineString(xy)


print(
    "TOPIS lines:",
    len(topis_lines)
)


# ============================================================
# 5. 기존의 보수적 mapTrace
# ============================================================

def rematch_link(line):

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
        delta=20,
        fillGaps=30,
        direction=True,
        reversalPenalty=10000
    )

    return [e.getID() for e in route]


# # ============================================================
# # 6. 시작/끝 anchor 사이 SUMO shortest path
# # ============================================================

# reconstructed = {}

# for link_id in problem4:

#     line = topis_lines[link_id]

#     before = rematch_link(line)

#     print("\n================================")
#     print("TOPIS:", link_id)
#     print("mapTrace edges:", len(before))

#     if len(before) == 0:
#         print("NO MAPTRACE RESULT")
#         continue

#     start_edge = net.getEdge(
#         before[0]
#     )

#     end_edge = net.getEdge(
#         before[-1]
#     )

#     result = net.getShortestPath(
#         start_edge,
#         end_edge
#     )

#     path, cost = result

#     if path is None:
#         print("NO PATH")
#         continue

#     path_ids = [
#         e.getID()
#         for e in path
#     ]

#     reconstructed[link_id] = path_ids

#     print(
#         "reconstructed edges:",
#         len(path_ids)
#     )

#     print(path_ids)


# # ============================================================
# # 7. reconstructed route가 TOPIS 선을 잘 따라가는지 검사
# # ============================================================

# def path_distance_to_topis(
#     path_ids,
#     topis_line
# ):

#     distances = []

#     for eid in path_ids:

#         edge = net.getEdge(eid)

#         shape = edge.getShape()

#         if len(shape) < 2:
#             continue

#         edge_line = LineString(shape)

#         sample_ds = np.arange(
#             0,
#             edge_line.length,
#             10.0
#         )

#         for d in sample_ds:

#             p = edge_line.interpolate(d)

#             distances.append(
#                 p.distance(topis_line)
#             )

#     if len(distances) == 0:
#         return None, None

#     return (
#         float(np.mean(distances)),
#         float(np.max(distances))
#     )


# print("\n\n===== QUALITY CHECK =====")

# for link_id, path_ids in reconstructed.items():

#     mean_dist, max_dist = (
#         path_distance_to_topis(
#             path_ids,
#             topis_lines[link_id]
#         )
#     )

#     print(
#         link_id,
#         "| edges =", len(path_ids),
#         "| mean =", round(mean_dist, 1), "m",
#         "| max =", round(max_dist, 1), "m"
#     )

import matplotlib.pyplot as plt

problem3 = [
    "1070013300",
    "1070018200",
    "1070018400",
]

for link_id in problem3:

    line = topis_lines[link_id]
    before = rematch_link(line)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 주변 SUMO 도로
    minx, miny, maxx, maxy = line.bounds
    margin = 150

    for e in net.getEdges():

        shape = e.getShape()

        if len(shape) < 2:
            continue

        # 너무 먼 edge는 그리지 않음
        ex = [p[0] for p in shape]
        ey = [p[1] for p in shape]

        if (
            max(ex) < minx - margin or
            min(ex) > maxx + margin or
            max(ey) < miny - margin or
            min(ey) > maxy + margin
        ):
            continue

        ax.plot(
            ex,
            ey,
            linewidth=0.4,
            alpha=0.25
        )

    # TOPIS 원본
    x, y = line.xy

    ax.plot(
        x,
        y,
        linewidth=4,
        alpha=0.6,
        label="TOPIS"
    )

    # 원래 mapTrace 결과
    for eid in before:

        e = net.getEdge(eid)
        shape = e.getShape()

        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]

        ax.plot(
            xs,
            ys,
            linewidth=2.5,
            alpha=0.9
        )

    ax.set_xlim(
        minx - margin,
        maxx + margin
    )

    ax.set_ylim(
        miny - margin,
        maxy + margin
    )

    ax.set_aspect("equal")

    ax.set_title(
        f"{link_id} | mapTrace edges={len(before)}"
    )

    plt.show()