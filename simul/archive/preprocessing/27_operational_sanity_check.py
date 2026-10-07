# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

COST_FILE = (
    BASE / "sumo_hourly_cost_operational_v3.csv"
)

DEPOT_FILE = (
    BASE / "operational_depot_v3.csv"
)


node_candidates = list(
    BASE.glob("seongbuk_node_mapping*.csv")
)

if len(node_candidates) == 0:
    raise FileNotFoundError(
        "seongbuk_node_mapping*.csv not found"
    )

NODE_FILE = node_candidates[0]


OUTPUT_HOURLY = (
    BASE / "sanity_check_hourly_qc_v3.csv"
)

OUTPUT_BINS = (
    BASE / "sanity_check_bin_reachability_v3.csv"
)

OUTPUT_IMPUTATION = (
    BASE / "sanity_check_imputation_qc_v3.csv"
)

OUTPUT_OUTLIERS = (
    BASE / "sanity_check_outlier_edges_v3.csv"
)


# ============================================================
# 1. PARAMETERS
# ============================================================

EPS = 1e-9

# bin -> road snap
BIN_SEARCH_RADIUS = 300.0     # m

# 아래는 삭제/clip 기준이 아니라 QC용 flag 기준
VERY_LOW_SPEED = 2.0          # km/h
VERY_HIGH_FACTOR = 10.0
VERY_LOW_FACTOR = 0.5

# free-flow 대비 지나치게 빠른 경우
MAX_SPEED_FF_RATIO = 1.5

# TT consistency
TT_REL_TOL = 1e-6


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


print(
    "Loading SUMO network..."
)

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 3. LOAD DATA
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


depot = pd.read_csv(
    DEPOT_FILE,
    dtype={
        "SUMO_EDGE_ID": str
    }
)


depot_edge_id = str(
    depot.iloc[0][
        "SUMO_EDGE_ID"
    ]
).strip()


nodes = pd.read_csv(
    NODE_FILE
)


print(
    "Operational edges:",
    len(cost)
)

print(
    "Bins:",
    len(nodes)
)

print(
    "Depot edge:",
    depot_edge_id
)


# ============================================================
# 4. CHECK OPERATIONAL EDGES
# ============================================================

print(
    "\nChecking edge validity..."
)


valid_edge_ids = set()

missing_network_edges = []

non_truck_edges = []


for eid in cost[
    "SUMO_EDGE_ID"
]:

    try:

        edge = net.getEdge(
            eid
        )

    except Exception:

        missing_network_edges.append(
            eid
        )

        continue


    if not edge.allows(
        "truck"
    ):

        non_truck_edges.append(
            eid
        )

        continue


    valid_edge_ids.add(
        eid
    )


print(
    "Missing SUMO edges:",
    len(missing_network_edges)
)

print(
    "Non-truck edges:",
    len(non_truck_edges)
)


# ============================================================
# 5. BUILD DIRECTED OPERATIONAL GRAPH
#
# Junction graph:
#
#    fromNode ---> toNode
#
# operational edges만 사용
# ============================================================

print(
    "\nBuilding directed operational graph..."
)


adj = defaultdict(list)

reverse_adj = defaultdict(list)


edge_from = {}
edge_to = {}
edge_geometry = {}


for eid in valid_edge_ids:

    edge = net.getEdge(
        eid
    )


    u = edge.getFromNode().getID()

    v = edge.getToNode().getID()


    edge_from[eid] = u

    edge_to[eid] = v


    adj[u].append(
        v
    )

    reverse_adj[v].append(
        u
    )


    shape = edge.getShape()

    if len(shape) >= 2:

        edge_geometry[eid] = (
            LineString(
                shape
            )
        )


# ============================================================
# 6. GRAPH BFS
# ============================================================

def bfs(
    graph,
    start
):

    visited = {
        start
    }

    q = deque([
        start
    ])


    while q:

        u = q.popleft()


        for v in graph.get(
            u,
            []
        ):

            if v in visited:
                continue


            visited.add(
                v
            )

            q.append(
                v
            )


    return visited


# ============================================================
# 7. DEPOT CHECK
# ============================================================

if depot_edge_id not in valid_edge_ids:

    raise RuntimeError(
        f"Depot edge {depot_edge_id} "
        "is not in operational truck network."
    )


depot_from = edge_from[
    depot_edge_id
]

depot_to = edge_to[
    depot_edge_id
]


# 차량이 depot edge를 떠난 뒤
# 갈 수 있는 모든 junction
reachable_from_depot = bfs(
    adj,
    depot_to
)


# 어떤 junction에서 depot의 시작점으로
# 돌아올 수 있는지를 reverse BFS
can_return_to_depot = bfs(
    reverse_adj,
    depot_from
)


all_graph_nodes = set(
    adj.keys()
) | set(
    reverse_adj.keys()
)


print(
    "\n=============================="
)

print(
    "DIRECTED NETWORK"
)

print(
    "=============================="
)


print(
    "Operational graph nodes:",
    len(all_graph_nodes)
)


print(
    "Reachable from depot:",
    len(reachable_from_depot)
)


print(
    "Can return to depot:",
    len(can_return_to_depot)
)


print(
    "Outbound reachable ratio:",
    round(
        len(reachable_from_depot)
        /
        max(
            len(all_graph_nodes),
            1
        ),
        4
    )
)


print(
    "Return reachable ratio:",
    round(
        len(can_return_to_depot)
        /
        max(
            len(all_graph_nodes),
            1
        ),
        4
    )
)


# ============================================================
# 8. BIN -> NEAREST OPERATIONAL TRUCK EDGE
# ============================================================

print(
    "\nSnapping bins to operational roads..."
)


bin_rows = []


for _, row in nodes.iterrows():

    idx = row[
        "idx"
    ]

    node_id = row[
        "node_id"
    ]

    lon = float(
        row[
            "lon"
        ]
    )

    lat = float(
        row[
            "lat"
        ]
    )


    x, y = net.convertLonLat2XY(
        lon,
        lat
    )


    p = Point(
        x,
        y
    )


    nearby = net.getNeighboringEdges(
        x,
        y,
        BIN_SEARCH_RADIUS
    )


    candidates = []


    for edge, _ in nearby:

        eid = edge.getID()


        if eid not in valid_edge_ids:
            continue


        line = edge_geometry.get(
            eid
        )


        if line is None:
            continue


        dist = p.distance(
            line
        )


        candidates.append(
            (
                dist,
                eid
            )
        )


    if len(candidates) == 0:

        bin_rows.append({
            "idx":
                idx,

            "node_id":
                node_id,

            "lon":
                lon,

            "lat":
                lat,

            "SUMO_EDGE_ID":
                None,

            "SNAP_DISTANCE_M":
                np.nan,

            "OUTBOUND_REACHABLE":
                False,

            "RETURN_REACHABLE":
                False,

            "ROUND_TRIP_FEASIBLE":
                False
        })

        continue


    candidates.sort(
        key=lambda z: z[0]
    )


    dist, eid = candidates[0]


    bin_edge_from = (
        edge_from[
            eid
        ]
    )

    bin_edge_to = (
        edge_to[
            eid
        ]
    )


    # Depot -> bin
    #
    # depot edge를 빠져나온 뒤
    # bin edge 시작 junction까지 도달 가능해야 함
    outbound_ok = (
        bin_edge_from
        in
        reachable_from_depot
    )


    # bin -> depot
    #
    # bin edge를 지나고 난 junction에서
    # depot edge 시작 junction까지 돌아올 수 있어야 함
    return_ok = (
        bin_edge_to
        in
        can_return_to_depot
    )


    bin_rows.append({

        "idx":
            idx,

        "node_id":
            node_id,

        "lon":
            lon,

        "lat":
            lat,

        "SUMO_EDGE_ID":
            eid,

        "SNAP_DISTANCE_M":
            float(
                dist
            ),

        "OUTBOUND_REACHABLE":
            bool(
                outbound_ok
            ),

        "RETURN_REACHABLE":
            bool(
                return_ok
            ),

        "ROUND_TRIP_FEASIBLE":
            bool(
                outbound_ok
                and
                return_ok
            )
    })


bin_qc = pd.DataFrame(
    bin_rows
)


bin_qc.to_csv(
    OUTPUT_BINS,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. BIN REACHABILITY SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "BIN REACHABILITY"
)

print(
    "=============================="
)


print(
    "Bins:",
    len(bin_qc)
)


print(
    "Bins successfully snapped:",
    bin_qc[
        "SUMO_EDGE_ID"
    ].notna().sum()
)


print(
    "Max snap distance:",
    round(
        bin_qc[
            "SNAP_DISTANCE_M"
        ].max(),
        2
    ),
    "m"
)


print(
    "Median snap distance:",
    round(
        bin_qc[
            "SNAP_DISTANCE_M"
        ].median(),
        2
    ),
    "m"
)


print(
    "Outbound reachable:",
    int(
        bin_qc[
            "OUTBOUND_REACHABLE"
        ].sum()
    )
)


print(
    "Return reachable:",
    int(
        bin_qc[
            "RETURN_REACHABLE"
        ].sum()
    )
)


print(
    "Round-trip feasible:",
    int(
        bin_qc[
            "ROUND_TRIP_FEASIBLE"
        ].sum()
    ),
    "/",
    len(bin_qc)
)


# ============================================================
# 10. IMPUTATION-ONLY QC
#
# IMPORTANT:
# IMPUTATION_METHOD는 fully-observed edge에도
# 기록되어 있으므로
#
# UNOBSERVED_LENGTH > 0
#
# 인 edge만 여기서 분석.
# ============================================================

if (
    "UNOBSERVED_LENGTH"
    not in
    cost.columns
):

    raise RuntimeError(
        "UNOBSERVED_LENGTH column missing."
    )


needs_imputation = cost[
    cost[
        "UNOBSERVED_LENGTH"
    ]
    >
    EPS
].copy()


imputation_rows = []


for region in [
    "ALL",
    "CORE",
    "BUFFER"
]:

    if region == "ALL":

        g = needs_imputation.copy()

    else:

        g = needs_imputation[
            needs_imputation[
                "REGION_TYPE"
            ]
            ==
            region
        ].copy()


    n = len(
        g
    )


    total_unobs_length = (
        g[
            "UNOBSERVED_LENGTH"
        ].sum()
    )


    for method in [
        "LOCAL_SAME_CLASS",
        "LOCAL_ANY_CLASS",
        "MEDIAN_FALLBACK"
    ]:

        m = g[
            g[
                "IMPUTATION_METHOD"
            ]
            ==
            method
        ]


        edge_count = len(
            m
        )


        unobs_length = (
            m[
                "UNOBSERVED_LENGTH"
            ].sum()
        )


        imputation_rows.append({

            "REGION":
                region,

            "METHOD":
                method,

            "N_EDGES":
                edge_count,

            "EDGE_RATIO":
                (
                    edge_count
                    /
                    n
                    if n > 0
                    else np.nan
                ),

            "UNOBSERVED_LENGTH_M":
                unobs_length,

            "LENGTH_RATIO":
                (
                    unobs_length
                    /
                    total_unobs_length
                    if total_unobs_length > 0
                    else np.nan
                )
        })


imputation_qc = pd.DataFrame(
    imputation_rows
)


imputation_qc.to_csv(
    OUTPUT_IMPUTATION,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n=============================="
)

print(
    "IMPUTATION-ONLY QC"
)

print(
    "=============================="
)


for region in [
    "ALL",
    "CORE",
    "BUFFER"
]:

    tmp = imputation_qc[
        imputation_qc[
            "REGION"
        ]
        ==
        region
    ]


    print(
        f"\n[{region}]"
    )


    for _, r in tmp.iterrows():

        print(
            f"{r['METHOD']:20s}"
            f" edges="
            f"{int(r['N_EDGES']):6d}"
            f"  edge%="
            f"{100*r['EDGE_RATIO']:6.2f}"
            f"  length%="
            f"{100*r['LENGTH_RATIO']:6.2f}"
        )


# ============================================================
# 11. HOURLY SANITY CHECK
# ============================================================

hourly_rows = []

outlier_rows = []


for h in range(
    1,
    25
):

    label = (
        f"~{h:02d}시"
    )

    tt_col = (
        f"TT_{label}"
    )

    speed_col = (
        f"SPEED_{label}"
    )

    factor_col = (
        f"IMPUTED_FACTOR_{label}"
    )


    if tt_col not in cost.columns:
        continue


    tt = pd.to_numeric(
        cost[
            tt_col
        ],
        errors="coerce"
    )


    speed = pd.to_numeric(
        cost[
            speed_col
        ],
        errors="coerce"
    )


    ff = pd.to_numeric(
        cost[
            "FREEFLOW_SPEED_KMH"
        ],
        errors="coerce"
    )


    length = pd.to_numeric(
        cost[
            "SUMO_LENGTH"
        ],
        errors="coerce"
    )


    # --------------------------------------------------------
    # TT ↔ speed internal consistency
    # --------------------------------------------------------

    expected_tt = (
        length
        /
        speed
        *
        3.6
    )


    rel_error = (
        (
            tt
            -
            expected_tt
        ).abs()
        /
        expected_tt.clip(
            lower=EPS
        )
    )


    # --------------------------------------------------------
    # implied congestion ratio
    #
    # pace / freeflow pace
    # = FF speed / current speed
    # --------------------------------------------------------

    congestion_ratio = (
        ff
        /
        speed
    )


    # --------------------------------------------------------
    # factor if available
    # --------------------------------------------------------

    if factor_col in cost.columns:

        factor = pd.to_numeric(
            cost[
                factor_col
            ],
            errors="coerce"
        )

    else:

        factor = pd.Series(
            np.nan,
            index=cost.index
        )


    hourly_rows.append({

        "HOUR":
            label,

        "N_EDGES":
            len(cost),

        "N_MISSING_TT":
            int(
                tt.isna().sum()
            ),

        "N_NONPOSITIVE_TT":
            int(
                (
                    tt <= 0
                ).sum()
            ),

        "N_MISSING_SPEED":
            int(
                speed.isna().sum()
            ),

        "N_NONPOSITIVE_SPEED":
            int(
                (
                    speed <= 0
                ).sum()
            ),

        "MEAN_SPEED_KMH":
            float(
                speed.mean()
            ),

        "MEDIAN_SPEED_KMH":
            float(
                speed.median()
            ),

        "P01_SPEED_KMH":
            float(
                speed.quantile(
                    0.01
                )
            ),

        "P05_SPEED_KMH":
            float(
                speed.quantile(
                    0.05
                )
            ),

        "P95_SPEED_KMH":
            float(
                speed.quantile(
                    0.95
                )
            ),

        "P99_SPEED_KMH":
            float(
                speed.quantile(
                    0.99
                )
            ),

        "N_SPEED_LT_2":
            int(
                (
                    speed
                    <
                    VERY_LOW_SPEED
                ).sum()
            ),

        "N_SPEED_GT_1P5_FF":
            int(
                (
                    speed
                    >
                    MAX_SPEED_FF_RATIO
                    *
                    ff
                ).sum()
            ),

        "N_CONGESTION_LT_0P5":
            int(
                (
                    congestion_ratio
                    <
                    VERY_LOW_FACTOR
                ).sum()
            ),

        "N_CONGESTION_GT_10":
            int(
                (
                    congestion_ratio
                    >
                    VERY_HIGH_FACTOR
                ).sum()
            ),

        "MAX_TT_REL_ERROR":
            float(
                rel_error.max()
            ),

        "N_TT_INCONSISTENT":
            int(
                (
                    rel_error
                    >
                    TT_REL_TOL
                ).sum()
            )
    })


    # ========================================================
    # OUTLIER EDGE LOG
    # ========================================================

    mask = (
        (
            speed
            <
            VERY_LOW_SPEED
        )
        |
        (
            speed
            >
            MAX_SPEED_FF_RATIO
            *
            ff
        )
        |
        (
            congestion_ratio
            >
            VERY_HIGH_FACTOR
        )
        |
        (
            congestion_ratio
            <
            VERY_LOW_FACTOR
        )
    )


    if mask.any():

        tmp = cost.loc[
            mask,
            [
                "SUMO_EDGE_ID",
                "REGION_TYPE",
                "ROAD_CLASS",
                "COST_SOURCE",
                "IMPUTATION_METHOD",
                "SUMO_LENGTH",
                "FREEFLOW_SPEED_KMH"
            ]
        ].copy()


        tmp[
            "HOUR"
        ] = label


        tmp[
            "FINAL_SPEED_KMH"
        ] = speed[
            mask
        ].values


        tmp[
            "CONGESTION_RATIO"
        ] = congestion_ratio[
            mask
        ].values


        if factor_col in cost.columns:

            tmp[
                "IMPUTED_FACTOR"
            ] = factor[
                mask
            ].values


        outlier_rows.append(
            tmp
        )


hourly_qc = pd.DataFrame(
    hourly_rows
)


hourly_qc.to_csv(
    OUTPUT_HOURLY,
    index=False,
    encoding="utf-8-sig"
)


if len(
    outlier_rows
) > 0:

    outliers = pd.concat(
        outlier_rows,
        ignore_index=True
    )

else:

    outliers = pd.DataFrame()


outliers.to_csv(
    OUTPUT_OUTLIERS,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. HOURLY OUTPUT
# ============================================================

print(
    "\n=============================="
)

print(
    "HOURLY TRAFFIC SANITY"
)

print(
    "=============================="
)


display_cols = [
    "HOUR",
    "MEAN_SPEED_KMH",
    "P01_SPEED_KMH",
    "P99_SPEED_KMH",
    "N_SPEED_LT_2",
    "N_SPEED_GT_1P5_FF",
    "N_CONGESTION_GT_10",
    "N_TT_INCONSISTENT"
]


print(
    hourly_qc[
        display_cols
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. FINAL STATUS
# ============================================================

hard_failures = []


if len(
    missing_network_edges
) > 0:

    hard_failures.append(
        "operational edges missing from SUMO"
    )


if len(
    non_truck_edges
) > 0:

    hard_failures.append(
        "non-truck edges in operational network"
    )


if (
    bin_qc[
        "SUMO_EDGE_ID"
    ].isna().any()
):

    hard_failures.append(
        "some bins could not be snapped"
    )


if not (
    bin_qc[
        "ROUND_TRIP_FEASIBLE"
    ].all()
):

    hard_failures.append(
        "some bins are not round-trip reachable"
    )


if (
    hourly_qc[
        "N_MISSING_TT"
    ].sum()
    >
    0
):

    hard_failures.append(
        "missing travel times"
    )


if (
    hourly_qc[
        "N_NONPOSITIVE_TT"
    ].sum()
    >
    0
):

    hard_failures.append(
        "non-positive travel times"
    )


if (
    hourly_qc[
        "N_TT_INCONSISTENT"
    ].sum()
    >
    0
):

    hard_failures.append(
        "TT-speed internal inconsistency"
    )


print(
    "\n=============================="
)

print(
    "FINAL SANITY STATUS"
)

print(
    "=============================="
)


if len(
    hard_failures
) == 0:

    print(
        "HARD CHECKS: PASS"
    )

else:

    print(
        "HARD CHECKS: FAIL"
    )

    for x in hard_failures:

        print(
            " -",
            x
        )


# ------------------------------------------------------------
# warnings
# ------------------------------------------------------------

warnings = []


if (
    bin_qc[
        "SNAP_DISTANCE_M"
    ].max()
    >
    50
):

    warnings.append(
        "Some bins are >50 m from their snapped road."
    )


if (
    hourly_qc[
        "N_SPEED_GT_1P5_FF"
    ].sum()
    >
    0
):

    warnings.append(
        "Some inferred speeds exceed 1.5 × SUMO free-flow speed."
    )


if (
    hourly_qc[
        "N_CONGESTION_GT_10"
    ].sum()
    >
    0
):

    warnings.append(
        "Some edges have congestion ratio >10."
    )


if (
    hourly_qc[
        "N_SPEED_LT_2"
    ].sum()
    >
    0
):

    warnings.append(
        "Some edges have inferred speed <2 km/h."
    )


core_imputed = needs_imputation[
    needs_imputation[
        "REGION_TYPE"
    ]
    ==
    "CORE"
]


core_fallback = (
    core_imputed[
        "IMPUTATION_METHOD"
    ]
    ==
    "MEDIAN_FALLBACK"
).sum()


if core_fallback > 0:

    warnings.append(
        f"CORE contains {core_fallback} "
        "actually-imputed MEDIAN_FALLBACK edges."
    )


print(
    "\nWarnings:",
    len(warnings)
)


for w in warnings:

    print(
        " -",
        w
    )


# ============================================================
# 14. FILES
# ============================================================

print(
    "\nCreated:"
)

print(
    " -",
    OUTPUT_HOURLY.name
)

print(
    " -",
    OUTPUT_BINS.name
)

print(
    " -",
    OUTPUT_IMPUTATION.name
)

print(
    " -",
    OUTPUT_OUTLIERS.name
)