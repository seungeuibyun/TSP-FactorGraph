# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

COST_FILE = (
    BASE / "sumo_hourly_cost_full_network_v3.csv"
)

OUTPUT = (
    BASE / "sumo_hourly_cost_truck_drivable_v3.csv"
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
# 2. COST
# ============================================================

cost = pd.read_csv(
    COST_FILE,
    dtype={
        "SUMO_EDGE_ID": str
    }
)

cost["SUMO_EDGE_ID"] = (
    cost["SUMO_EDGE_ID"]
    .str.strip()
)


print(
    "Total cost edges:",
    len(cost)
)


# ============================================================
# 3. TRUCK-DRIVABLE EDGE
# ============================================================

drivable_ids = set()

for edge in net.getEdges():

    # 폐기물 수거차 = SUMO truck class
    if edge.allows("truck"):

        drivable_ids.add(
            edge.getID()
        )


print(
    "Truck-drivable SUMO edges:",
    len(drivable_ids)
)


# ============================================================
# 4. FILTER COST
# ============================================================

drivable = cost[
    cost[
        "SUMO_EDGE_ID"
    ].isin(
        drivable_ids
    )
].copy()


drivable = (
    drivable
    .sort_values(
        "SUMO_EDGE_ID"
    )
    .reset_index(
        drop=True
    )
)


drivable.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "TRUCK-DRIVABLE NETWORK"
)

print(
    "=============================="
)


print(
    "Original edges:",
    len(cost)
)


print(
    "Truck-drivable edges:",
    len(drivable)
)


print(
    "Removed edges:",
    len(cost) - len(drivable)
)


print(
    "Remaining ratio:",
    round(
        len(drivable)
        /
        len(cost),
        3
    )
)


print(
    "\nCost source:"
)

print(
    drivable[
        "COST_SOURCE"
    ].value_counts()
)


print(
    "\nImputation method:"
)

print(
    drivable[
        "IMPUTATION_METHOD"
    ].value_counts()
)


print(
    "\nCreated:",
    OUTPUT.name
)