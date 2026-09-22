# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from shapely.geometry import LineString


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

OBSERVED_FILE = (
    BASE / "sumo_topis_observed_edge_cost.csv"
)

OUTPUT = (
    BASE / "sumo_hourly_cost_full_network.csv"
)

OUTPUT_QC = (
    BASE / "sumo_hourly_cost_full_network_qc.csv"
)


# ============================================================
# 1. PARAMETERS
# ============================================================

# 같은 road class에서 가까운 관측 edge
K_SAME_CLASS = 5
RADIUS_SAME_CLASS = 1500.0

# 같은 class가 없을 때 주변 모든 road
K_GLOBAL = 8
RADIUS_GLOBAL = 2000.0

EPS = 1e-8


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
# 3. ROAD CLASS
# ============================================================

def get_road_class(edge):

    try:
        t = edge.getType()

        if t is None:
            t = ""

        t = str(t).lower()

    except Exception:
        t = ""


    # OSM/SUMO type을 큰 category로만 묶음
    classes = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "living_street",
        "service",
        "unclassified"
    ]


    for c in classes:

        if c in t:
            return c


    # type 정보가 애매하면 speed band
    speed_kmh = (
        float(
            edge.getSpeed()
        )
        * 3.6
    )

    band = (
        int(
            round(
                speed_kmh / 10
            )
            * 10
        )
    )

    return f"other_{band}"


# ============================================================
# 4. ENTIRE SUMO NETWORK
# ============================================================

network_rows = []


for edge in net.getEdges():

    eid = edge.getID()

    L = float(
        edge.getLength()
    )

    speed_mps = float(
        edge.getSpeed()
    )


    if speed_mps <= 0:
        continue


    shape = edge.getShape()


    if len(shape) >= 2:

        line = LineString(
            shape
        )

        p = line.interpolate(
            0.5,
            normalized=True
        )

        x = float(p.x)
        y = float(p.y)

    else:

        x, y = (
            edge
            .getFromNode()
            .getCoord()
        )


    network_rows.append({

        "SUMO_EDGE_ID":
            eid,

        "SUMO_LENGTH":
            L,

        "FREEFLOW_SPEED_MPS":
            speed_mps,

        "FREEFLOW_SPEED_KMH":
            speed_mps * 3.6,

        # sec / meter
        "FREEFLOW_PACE":
            1.0 / speed_mps,

        "ROAD_CLASS":
            get_road_class(
                edge
            ),

        "MID_X":
            x,

        "MID_Y":
            y
    })


network = pd.DataFrame(
    network_rows
)


print(
    "Total SUMO edges:",
    len(network)
)


# ============================================================
# 5. OBSERVED EDGE COST
# ============================================================

obs = pd.read_csv(
    OBSERVED_FILE,
    dtype={
        "SUMO_EDGE_ID": str
    }
)

obs["SUMO_EDGE_ID"] = (
    obs["SUMO_EDGE_ID"]
    .str.strip()
)


hour_names = [
    f"~{h:02d}시"
    for h in range(1, 25)
]


obs_pace_cols = [
    f"OBS_PACE_{h}"
    for h in hour_names
]


obs_tt_cols = [
    f"OBS_TT_{h}"
    for h in hour_names
]


keep_cols = [
    "SUMO_EDGE_ID",
    "OBSERVED_LENGTH",
    "COVERAGE_RATIO",
    "UNOBSERVED_LENGTH",
    "OVERLAP_LENGTH",
    "OVERLAP_RATIO"
] + obs_pace_cols + obs_tt_cols


obs = obs[
    keep_cols
].copy()


# ============================================================
# 6. MERGE
# ============================================================

df = network.merge(
    obs,
    on="SUMO_EDGE_ID",
    how="left"
)


df["OBSERVED_LENGTH"] = (
    df["OBSERVED_LENGTH"]
    .fillna(0.0)
)


df["COVERAGE_RATIO"] = (
    df["COVERAGE_RATIO"]
    .fillna(0.0)
)


df["UNOBSERVED_LENGTH"] = (
    df["SUMO_LENGTH"]
    -
    df["OBSERVED_LENGTH"]
)


df["UNOBSERVED_LENGTH"] = (
    df["UNOBSERVED_LENGTH"]
    .clip(
        lower=0.0
    )
)


# ============================================================
# 7. OBSERVED SOURCE EDGES
#
# TOPIS가 실제로 조금이라도 덮은 edge만
# traffic inference source로 사용.
# ============================================================

source = df[
    df["OBSERVED_LENGTH"] > EPS
].copy()


source = source.reset_index(
    drop=True
)


print(
    "Edges with direct TOPIS support:",
    len(source)
)


# ============================================================
# 8. CONGESTION FACTOR
#
# observed pace / SUMO free-flow pace
#
# 1.0 = free-flow와 동일
# 2.0 = free-flow보다 2배 오래 걸림
# ============================================================

factor_cols = []


for h in hour_names:

    pace_col = (
        f"OBS_PACE_{h}"
    )

    factor_col = (
        f"FACTOR_{h}"
    )


    source[
        factor_col
    ] = (
        source[
            pace_col
        ]
        /
        source[
            "FREEFLOW_PACE"
        ]
    )


    factor_cols.append(
        factor_col
    )


# ============================================================
# 9. SPATIAL TREES
# ============================================================

# global source tree
global_coords = (
    source[
        [
            "MID_X",
            "MID_Y"
        ]
    ]
    .to_numpy()
)


global_tree = cKDTree(
    global_coords
)


# road-class trees
class_trees = {}


for road_class, g in (
    source.groupby(
        "ROAD_CLASS"
    )
):

    positions = (
        g.index
        .to_numpy()
    )


    coords = (
        g[
            [
                "MID_X",
                "MID_Y"
            ]
        ]
        .to_numpy()
    )


    class_trees[
        road_class
    ] = {
        "tree":
            cKDTree(
                coords
            ),

        "positions":
            positions
    }


# ============================================================
# 10. NEIGHBOR QUERY
# ============================================================

def query_tree(
    tree,
    positions,
    x,
    y,
    target_eid,
    k,
    radius
):

    if len(positions) == 0:
        return []


    # 자기 자신이 source이면 제외해야 해서 +1
    kk = min(
        k + 1,
        len(positions)
    )


    distances, local_idx = tree.query(
        [x, y],
        k=kk
    )


    distances = np.atleast_1d(
        distances
    )

    local_idx = np.atleast_1d(
        local_idx
    )


    result = []


    for d, j in zip(
        distances,
        local_idx
    ):

        if not np.isfinite(d):
            continue


        if d > radius:
            continue


        src_pos = int(
            positions[
                int(j)
            ]
        )


        src_eid = source.loc[
            src_pos,
            "SUMO_EDGE_ID"
        ]


        if src_eid == target_eid:
            continue


        result.append(
            src_pos
        )


        if len(result) >= k:
            break


    return result


# ============================================================
# 11. 각 SUMO EDGE의 imputation neighbor 선택
# ============================================================

neighbor_map = {}
method_map = {}


global_positions = (
    source.index.to_numpy()
)


for i, row in df.iterrows():

    eid = row[
        "SUMO_EDGE_ID"
    ]

    x = row[
        "MID_X"
    ]

    y = row[
        "MID_Y"
    ]

    road_class = row[
        "ROAD_CLASS"
    ]


    # --------------------------------------------------------
    # 1) local + same road class
    # --------------------------------------------------------

    same = []


    if road_class in class_trees:

        info = (
            class_trees[
                road_class
            ]
        )


        same = query_tree(
            info["tree"],
            info["positions"],
            x,
            y,
            eid,
            K_SAME_CLASS,
            RADIUS_SAME_CLASS
        )


    if len(same) > 0:

        neighbor_map[
            eid
        ] = same

        method_map[
            eid
        ] = "LOCAL_SAME_CLASS"

        continue


    # --------------------------------------------------------
    # 2) local all road classes
    # --------------------------------------------------------

    nearby = query_tree(
        global_tree,
        global_positions,
        x,
        y,
        eid,
        K_GLOBAL,
        RADIUS_GLOBAL
    )


    if len(nearby) > 0:

        neighbor_map[
            eid
        ] = nearby

        method_map[
            eid
        ] = "LOCAL_ANY_CLASS"

        continue


    # --------------------------------------------------------
    # 3) 이후 class/global median fallback
    # --------------------------------------------------------

    neighbor_map[
        eid
    ] = []

    method_map[
        eid
    ] = "MEDIAN_FALLBACK"


df[
    "IMPUTATION_METHOD"
] = (
    df[
        "SUMO_EDGE_ID"
    ].map(
        method_map
    )
)


# ============================================================
# 12. HOURLY FULL COST
# ============================================================

for h in hour_names:

    factor_col = (
        f"FACTOR_{h}"
    )

    obs_tt_col = (
        f"OBS_TT_{h}"
    )

    final_tt_col = (
        f"TT_{h}"
    )

    final_speed_col = (
        f"SPEED_{h}"
    )

    imputed_factor_col = (
        f"IMPUTED_FACTOR_{h}"
    )


    # --------------------------------------------------------
    # fallback medians
    # --------------------------------------------------------

    global_median = float(
        source[
            factor_col
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .median()
    )


    class_median = (
        source
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .groupby(
            "ROAD_CLASS"
        )[
            factor_col
        ]
        .median()
        .to_dict()
    )


    inferred_factors = []


    for _, row in df.iterrows():

        eid = (
            row[
                "SUMO_EDGE_ID"
            ]
        )

        road_class = (
            row[
                "ROAD_CLASS"
            ]
        )


        neighbors = (
            neighbor_map[
                eid
            ]
        )


        # ----------------------------------------
        # local neighbors
        # ----------------------------------------

        if len(neighbors) > 0:

            values = (
                source.loc[
                    neighbors,
                    factor_col
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )


            if len(values) > 0:

                # robust local estimate
                factor = float(
                    np.median(
                        values
                    )
                )

            else:

                factor = np.nan

        else:

            factor = np.nan


        # ----------------------------------------
        # same road class median fallback
        # ----------------------------------------

        if (
            not np.isfinite(
                factor
            )
        ):

            factor = (
                class_median.get(
                    road_class,
                    np.nan
                )
            )


        # ----------------------------------------
        # global median fallback
        # ----------------------------------------

        if (
            not np.isfinite(
                factor
            )
        ):

            factor = (
                global_median
            )


        inferred_factors.append(
            float(
                factor
            )
        )


    df[
        imputed_factor_col
    ] = (
        inferred_factors
    )


    # ========================================================
    # DIRECTLY OBSERVED PART
    # ========================================================

    observed_tt = (
        pd.to_numeric(
            df[
                obs_tt_col
            ],
            errors="coerce"
        )
        .fillna(0.0)
    )


    # ========================================================
    # UNOBSERVED PART
    #
    # pace_imputed
    # =
    # SUMO free-flow pace × local congestion factor
    # ========================================================

    imputed_pace = (
        df[
            "FREEFLOW_PACE"
        ]
        *
        df[
            imputed_factor_col
        ]
    )


    imputed_tt = (
        df[
            "UNOBSERVED_LENGTH"
        ]
        *
        imputed_pace
    )


    final_tt = (
        observed_tt
        +
        imputed_tt
    )


    df[
        final_tt_col
    ] = (
        final_tt
    )


    df[
        final_speed_col
    ] = (
        df[
            "SUMO_LENGTH"
        ]
        /
        final_tt
        *
        3.6
    )


# ============================================================
# 13. SOURCE LABEL
# ============================================================

def source_label(row):

    c = row[
        "COVERAGE_RATIO"
    ]


    if c >= 0.999:

        return "OBSERVED"

    if c > 0:

        return "PARTIAL_OBSERVED_PLUS_IMPUTED"

    return "IMPUTED"


df[
    "COST_SOURCE"
] = df.apply(
    source_label,
    axis=1
)


# ============================================================
# 14. OUTPUT
# ============================================================

output_cols = [
    "SUMO_EDGE_ID",
    "SUMO_LENGTH",
    "ROAD_CLASS",
    "FREEFLOW_SPEED_KMH",
    "OBSERVED_LENGTH",
    "UNOBSERVED_LENGTH",
    "COVERAGE_RATIO",
    "COST_SOURCE",
    "IMPUTATION_METHOD"
]


for h in hour_names:

    output_cols.extend(
        [
            f"TT_{h}",
            f"SPEED_{h}",
            f"IMPUTED_FACTOR_{h}"
        ]
    )


out = df[
    output_cols
].copy()


out.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. QC
# ============================================================

qc_rows = []


for h in hour_names:

    tt_col = (
        f"TT_{h}"
    )

    speed_col = (
        f"SPEED_{h}"
    )


    qc_rows.append({

        "HOUR":
            h,

        "N_EDGES":
            len(df),

        "N_MISSING_TT":
            int(
                df[
                    tt_col
                ].isna().sum()
            ),

        "N_NONPOSITIVE_TT":
            int(
                (
                    df[
                        tt_col
                    ]
                    <= 0
                ).sum()
            ),

        "MEAN_SPEED_KMH":
            float(
                df[
                    speed_col
                ].mean()
            ),

        "P05_SPEED_KMH":
            float(
                df[
                    speed_col
                ]
                .quantile(
                    0.05
                )
            ),

        "P95_SPEED_KMH":
            float(
                df[
                    speed_col
                ]
                .quantile(
                    0.95
                )
            )
    })


qc = pd.DataFrame(
    qc_rows
)


qc.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "FULL NETWORK HOURLY COST"
)

print(
    "=============================="
)


print(
    "Total SUMO edges:",
    len(df)
)


print(
    "\nCost source:"
)


print(
    df[
        "COST_SOURCE"
    ].value_counts()
)


print(
    "\nImputation method:"
)


print(
    df[
        "IMPUTATION_METHOD"
    ].value_counts()
)


print(
    "\nMissing final TT:",
    int(
        sum(
            df[
                f"TT_{h}"
            ].isna().sum()
            for h in hour_names
        )
    )
)


print(
    "\nHourly QC:"
)


print(
    qc.to_string(
        index=False
    )
)


print(
    "\nCreated:",
    OUTPUT.name
)

print(
    "Created:",
    OUTPUT_QC.name
)