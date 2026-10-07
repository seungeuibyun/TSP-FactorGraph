# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import pandas as pd

from shapely.geometry import (
    Point,
    LineString,
    MultiPoint
)


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

TRUCK_COST_FILE = (
    BASE / "sumo_hourly_cost_truck_drivable_v3.csv"
)

OUTPUT_COST = (
    BASE / "sumo_hourly_cost_operational_v3.csv"
)

OUTPUT_DEPOT = (
    BASE / "operational_depot_v3.csv"
)

OUTPUT_EDGE_LIST = (
    BASE / "operational_edge_list_v3.csv"
)


# node mapping 파일명 자동 탐색
node_candidates = list(
    BASE.glob(
        "seongbuk_node_mapping*.csv"
    )
)

if len(node_candidates) == 0:
    raise FileNotFoundError(
        "seongbuk_node_mapping*.csv not found"
    )

NODE_FILE = node_candidates[0]


# ============================================================
# 1. PARAMETER
# ============================================================

# bin convex hull 바깥으로 허용할 우회 범위
BUFFER_M = 2000.0

# 임의 depot을 truck road로 snap할 때 탐색반경
DEPOT_SEARCH_RADIUS = 3000.0


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
# 3. BIN LOCATIONS
# ============================================================

nodes = pd.read_csv(
    NODE_FILE
)


required = [
    "idx",
    "node_id",
    "lon",
    "lat"
]

missing = [
    c
    for c in required
    if c not in nodes.columns
]

if missing:
    raise RuntimeError(
        f"Missing node columns: {missing}"
    )


nodes["lon"] = pd.to_numeric(
    nodes["lon"],
    errors="coerce"
)

nodes["lat"] = pd.to_numeric(
    nodes["lat"],
    errors="coerce"
)

nodes = nodes.dropna(
    subset=[
        "lon",
        "lat"
    ]
).copy()


print(
    "Bins:",
    len(nodes)
)


# ============================================================
# 4. BIN lon/lat -> SUMO XY
# ============================================================

bin_points = []


for lon, lat in zip(
    nodes["lon"],
    nodes["lat"]
):

    x, y = net.convertLonLat2XY(
        float(lon),
        float(lat)
    )

    bin_points.append(
        Point(
            float(x),
            float(y)
        )
    )


# ============================================================
# 5. ARBITRARY DEPOT
#
# 우선 bin들의 중심을 임시 depot으로 잡고,
# 가장 가까운 truck-drivable edge 위로 snap.
# ============================================================

raw_depot_lon = float(
    nodes["lon"].mean()
)

raw_depot_lat = float(
    nodes["lat"].mean()
)


raw_depot_x, raw_depot_y = (
    net.convertLonLat2XY(
        raw_depot_lon,
        raw_depot_lat
    )
)


raw_depot_point = Point(
    raw_depot_x,
    raw_depot_y
)


# ------------------------------------------------------------
# nearby truck edges
# ------------------------------------------------------------

nearby = net.getNeighboringEdges(
    raw_depot_x,
    raw_depot_y,
    DEPOT_SEARCH_RADIUS
)


truck_candidates = []


for edge, dist in nearby:

    if not edge.allows(
        "truck"
    ):
        continue

    shape = edge.getShape()

    if len(shape) < 2:
        continue

    line = LineString(
        shape
    )

    true_dist = (
        raw_depot_point.distance(
            line
        )
    )

    truck_candidates.append(
        (
            true_dist,
            edge,
            line
        )
    )


if len(truck_candidates) == 0:

    raise RuntimeError(
        "No truck-drivable edge found near depot."
    )


truck_candidates.sort(
    key=lambda x: x[0]
)


depot_dist, depot_edge, depot_line = (
    truck_candidates[0]
)


# ------------------------------------------------------------
# depot을 실제 edge geometry에 snap
# ------------------------------------------------------------

depot_s = depot_line.project(
    raw_depot_point
)


depot_point = depot_line.interpolate(
    depot_s
)


depot_x = float(
    depot_point.x
)

depot_y = float(
    depot_point.y
)


depot_lon, depot_lat = (
    net.convertXY2LonLat(
        depot_x,
        depot_y
    )
)


print(
    "\nArbitrary depot:"
)

print(
    "edge:",
    depot_edge.getID()
)

print(
    "lon:",
    round(
        depot_lon,
        7
    )
)

print(
    "lat:",
    round(
        depot_lat,
        7
    )
)

print(
    "snap distance:",
    round(
        depot_dist,
        1
    ),
    "m"
)


# ============================================================
# 6. OPERATIONAL REGION
#
# bins + depot
# -> convex hull
# -> 2 km buffer
# ============================================================

all_points = (
    bin_points
    +
    [
        depot_point
    ]
)


hull = MultiPoint(
    all_points
).convex_hull


operational_region = hull.buffer(
    BUFFER_M
)


print(
    "\nConvex-hull area:",
    round(
        hull.area
        /
        1_000_000,
        2
    ),
    "km^2"
)


print(
    "Operational region area:",
    round(
        operational_region.area
        /
        1_000_000,
        2
    ),
    "km^2"
)


# ============================================================
# 7. LOAD TRUCK COST
# ============================================================

cost = pd.read_csv(
    TRUCK_COST_FILE,
    dtype={
        "SUMO_EDGE_ID": str
    }
)


cost["SUMO_EDGE_ID"] = (
    cost["SUMO_EDGE_ID"]
    .str.strip()
)


print(
    "\nInput truck edges:",
    len(cost)
)


# ============================================================
# 8. SPATIAL FILTER
#
# truck edge geometry가 operational region과
# 조금이라도 교차하면 유지.
#
# midpoint 방식보다 boundary route를 덜 끊음.
# ============================================================

keep_ids = []

edge_rows = []


for i, eid in enumerate(
    cost["SUMO_EDGE_ID"],
    start=1
):

    try:

        edge = net.getEdge(
            eid
        )

    except Exception:

        continue


    shape = edge.getShape()

    if len(shape) < 2:
        continue


    line = LineString(
        shape
    )


    if not line.intersects(
        operational_region
    ):
        continue


    keep_ids.append(
        eid
    )


    midpoint = line.interpolate(
        0.5,
        normalized=True
    )


    # hull 자체 안인지,
    # buffer에서만 살아남은 edge인지 기록
    if line.intersects(
        hull
    ):

        region_type = (
            "CORE"
        )

    else:

        region_type = (
            "BUFFER"
        )


    edge_rows.append({

        "SUMO_EDGE_ID":
            eid,

        "REGION_TYPE":
            region_type,

        "MID_X":
            float(
                midpoint.x
            ),

        "MID_Y":
            float(
                midpoint.y
            )
    })


keep_set = set(
    keep_ids
)


# ============================================================
# 9. FILTER COST
# ============================================================

operational = cost[
    cost[
        "SUMO_EDGE_ID"
    ].isin(
        keep_set
    )
].copy()


operational = (
    operational
    .sort_values(
        "SUMO_EDGE_ID"
    )
    .reset_index(
        drop=True
    )
)


edge_list = pd.DataFrame(
    edge_rows
)


operational = operational.merge(
    edge_list[
        [
            "SUMO_EDGE_ID",
            "REGION_TYPE"
        ]
    ],
    on="SUMO_EDGE_ID",
    how="left"
)


# ============================================================
# 10. SAVE DEPOT
# ============================================================

depot_df = pd.DataFrame([
    {
        "DEPOT_ID":
            "D0",

        "SUMO_EDGE_ID":
            depot_edge.getID(),

        "EDGE_POSITION_M":
            float(
                depot_s
            ),

        "LON":
            float(
                depot_lon
            ),

        "LAT":
            float(
                depot_lat
            ),

        "SUMO_X":
            depot_x,

        "SUMO_Y":
            depot_y,

        "METHOD":
            "BIN_CENTROID_SNAPPED_TO_TRUCK_EDGE"
    }
])


depot_df.to_csv(
    OUTPUT_DEPOT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. SAVE EDGE LIST / COST
# ============================================================

edge_list.to_csv(
    OUTPUT_EDGE_LIST,
    index=False,
    encoding="utf-8-sig"
)


operational.to_csv(
    OUTPUT_COST,
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
    "OPERATIONAL TRUCK NETWORK"
)

print(
    "=============================="
)


print(
    "Truck-drivable edges:",
    len(cost)
)


print(
    "Operational edges:",
    len(operational)
)


print(
    "Removed by spatial filter:",
    len(cost)
    -
    len(operational)
)


print(
    "Remaining ratio:",
    round(
        len(operational)
        /
        len(cost),
        3
    )
)


print(
    "\nRegion type:"
)


print(
    operational[
        "REGION_TYPE"
    ].value_counts()
)


print(
    "\nCost source:"
)


print(
    operational[
        "COST_SOURCE"
    ].value_counts()
)


print(
    "\nImputation method:"
)


print(
    operational[
        "IMPUTATION_METHOD"
    ].value_counts()
)


print(
    "\nCreated:",
    OUTPUT_COST.name
)

print(
    "Created:",
    OUTPUT_DEPOT.name
)

print(
    "Created:",
    OUTPUT_EDGE_LIST.name
)