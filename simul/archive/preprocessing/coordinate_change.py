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

print("TOPIS lines:", len(topis_lines))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))

# SUMO roads
for e in net.getEdges():
    shape = e.getShape()

    if len(shape) < 2:
        continue

    xs = [p[0] for p in shape]
    ys = [p[1] for p in shape]

    ax.plot(
        xs, ys,
        linewidth=0.4,
        alpha=0.5
    )

# 성북구 TOPIS links only
for line in topis_lines.values():
    x, y = line.xy

    ax.plot(
        x, y,
        linewidth=1.5,
        alpha=0.8
    )

ax.set_aspect("equal")
plt.show()