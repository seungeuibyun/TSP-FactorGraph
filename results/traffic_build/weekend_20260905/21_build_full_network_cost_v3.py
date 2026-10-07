# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shapely.geometry import LineString
from scipy.spatial import cKDTree


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

OBSERVED_COST_FILE = (
    BASE / "sumo_topis_observed_edge_cost.csv"
)

OUTPUT_COST = (
    BASE / "sumo_hourly_cost_full_network_v3.csv"
)

OUTPUT_QC = (
    BASE / "sumo_hourly_cost_full_network_v3_qc.csv"
)

OUTPUT_SOURCE_QC = (
    BASE / "topis_relative_congestion_factor_qc.csv"
)


# ============================================================
# 1. PARAMETERS
# ============================================================

EPS = 1e-9

# 주변 같은 road class
SAME_CLASS_RADIUS = 1500.0
SAME_CLASS_K = 5

# same-class 관측 source 자체가 전혀 없을 때
ANY_CLASS_RADIUS = 2000.0
ANY_CLASS_K = 8

FULL_OBS_THRESHOLD = 0.999


HOURS = [
    f"~{h:02d}시"
    for h in range(1, 25)
]


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


print("Loading SUMO network...")

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 3. ROAD CLASS
# ============================================================

def get_road_class(edge):

    try:
        road_type = (
            edge.getType()
            or ""
        ).lower()

    except Exception:
        road_type = ""


    classes = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "living_street",
        "service",
        "unclassified",
    ]


    for c in classes:

        if c in road_type:
            return c


    # --------------------------------------------------------
    # fallback: SUMO nominal speed band
    # --------------------------------------------------------

    speed_kmh = (
        edge.getSpeed()
        *
        3.6
    )


    if speed_kmh < 10:
        return "other_lt10"

    elif speed_kmh < 20:
        return "other_10_20"

    elif speed_kmh < 30:
        return "other_20_30"

    elif speed_kmh < 40:
        return "other_30_40"

    elif speed_kmh < 50:
        return "other_40_50"

    elif speed_kmh < 60:
        return "other_50_60"

    elif speed_kmh < 80:
        return "other_60_80"

    else:
        return "other_80plus"


# ============================================================
# 4. FULL SUMO EDGE TABLE
# ============================================================

print("Building SUMO edge table...")


edge_rows = []


for edge in net.getEdges():

    eid = edge.getID()

    length = float(
        edge.getLength()
    )

    speed_mps = float(
        edge.getSpeed()
    )


    if (
        length <= 0
        or
        speed_mps <= 0
    ):
        continue


    shape = edge.getShape()


    if len(shape) >= 2:

        line = LineString(
            shape
        )

        midpoint = line.interpolate(
            0.5,
            normalized=True
        )

        mid_x = float(
            midpoint.x
        )

        mid_y = float(
            midpoint.y
        )


    else:

        mid_x, mid_y = (
            edge
            .getFromNode()
            .getCoord()
        )


    edge_rows.append({

        "SUMO_EDGE_ID":
            eid,

        "SUMO_LENGTH":
            length,

        "ROAD_CLASS":
            get_road_class(
                edge
            ),

        "FREEFLOW_SPEED_KMH":
            speed_mps * 3.6,

        # s / m
        "FREEFLOW_PACE":
            1.0 / speed_mps,

        "MID_X":
            mid_x,

        "MID_Y":
            mid_y
    })


edges = pd.DataFrame(
    edge_rows
)


print(
    "Total SUMO edges:",
    len(edges)
)


# ============================================================
# 5. LOAD PIECEWISE TOPIS OBSERVATIONS
# ============================================================

obs = pd.read_csv(
    OBSERVED_COST_FILE,
    dtype={
        "SUMO_EDGE_ID": str
    }
)


obs["SUMO_EDGE_ID"] = (
    obs["SUMO_EDGE_ID"]
    .str.strip()
)


keep_cols = [
    "SUMO_EDGE_ID",
    "OBSERVED_LENGTH",
    "COVERAGE_RATIO",
]


for hour in HOURS:

    keep_cols.extend([
        f"OBS_PACE_{hour}",
        f"OBS_TT_{hour}",
    ])


keep_cols = [
    c
    for c in keep_cols
    if c in obs.columns
]


obs = obs[
    keep_cols
].copy()


# ============================================================
# 6. MERGE
# ============================================================

df = edges.merge(
    obs,
    on="SUMO_EDGE_ID",
    how="left"
)


df["OBSERVED_LENGTH"] = (
    pd.to_numeric(
        df["OBSERVED_LENGTH"],
        errors="coerce"
    )
    .fillna(0.0)
)


df["OBSERVED_LENGTH"] = np.minimum(
    df["OBSERVED_LENGTH"],
    df["SUMO_LENGTH"]
)


df["UNOBSERVED_LENGTH"] = (
    df["SUMO_LENGTH"]
    -
    df["OBSERVED_LENGTH"]
).clip(
    lower=0.0
)


df["COVERAGE_RATIO"] = (
    df["OBSERVED_LENGTH"]
    /
    df["SUMO_LENGTH"]
).clip(
    0.0,
    1.0
)


print(
    "Edges with direct TOPIS support:",
    int(
        (
            df["OBSERVED_LENGTH"]
            >
            EPS
        ).sum()
    )
)


# ============================================================
# 7. COST SOURCE
# ============================================================

def classify_cost_source(row):

    if (
        row["COVERAGE_RATIO"]
        >=
        FULL_OBS_THRESHOLD
    ):

        return "OBSERVED"

    elif (
        row["OBSERVED_LENGTH"]
        >
        EPS
    ):

        return (
            "PARTIAL_OBSERVED_PLUS_IMPUTED"
        )

    return "IMPUTED"


df["COST_SOURCE"] = (
    df.apply(
        classify_cost_source,
        axis=1
    )
)


# ============================================================
# 8. OBSERVED SOURCE EDGES
# ============================================================

source_df = (
    df[
        df["OBSERVED_LENGTH"]
        >
        EPS
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


print(
    "TOPIS-supported source edges:",
    len(source_df)
)


# ============================================================
# 9. DAILY REFERENCE PACE
#
# 핵심 V3 변경
#
# 각 source edge에 대해
#
# p_ref = min_h p_obs(h)
#
# 즉 하루 중 가장 빠른 관측 상태.
#
# c(h) = p_obs(h) / p_ref
#
# 따라서 이론적으로 c(h) >= 1
# ============================================================

print(
    "Computing TOPIS-relative congestion factors..."
)


pace_cols = [
    f"OBS_PACE_{hour}"
    for hour in HOURS
]


pace_matrix = (
    source_df[
        pace_cols
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


# invalid pace는 reference 계산에서 제외
pace_valid = np.where(
    (
        np.isfinite(
            pace_matrix
        )
        &
        (
            pace_matrix > 0
        )
    ),
    pace_matrix,
    np.nan
)


reference_pace = np.nanmin(
    pace_valid,
    axis=1
)


if np.isnan(
    reference_pace
).any():

    raise RuntimeError(
        "Some directly observed source edges "
        "have no valid hourly pace."
    )


source_df[
    "REFERENCE_PACE"
] = reference_pace


source_df[
    "REFERENCE_SPEED_KMH"
] = (
    3.6
    /
    source_df[
        "REFERENCE_PACE"
    ]
)


# ------------------------------------------------------------
# 각 시간대 상대 혼잡 factor
# ------------------------------------------------------------

for hour in HOURS:

    pace_col = (
        f"OBS_PACE_{hour}"
    )

    factor_col = (
        f"SOURCE_FACTOR_{hour}"
    )


    pace = pd.to_numeric(
        source_df[
            pace_col
        ],
        errors="coerce"
    )


    factor = (
        pace
        /
        source_df[
            "REFERENCE_PACE"
        ]
    )


    # floating-point 수준에서만 1 미만 방지
    factor = factor.where(
        factor.isna(),
        np.maximum(
            factor,
            1.0
        )
    )


    source_df[
        factor_col
    ] = factor


# ============================================================
# 10. SOURCE FACTOR QC
# ============================================================

source_qc_rows = []


for hour in HOURS:

    vals = pd.to_numeric(
        source_df[
            f"SOURCE_FACTOR_{hour}"
        ],
        errors="coerce"
    )


    source_qc_rows.append({

        "HOUR":
            hour,

        "N_SOURCE_EDGES":
            len(vals),

        "N_MISSING":
            int(
                vals.isna().sum()
            ),

        "MIN_FACTOR":
            float(
                vals.min()
            ),

        "MEDIAN_FACTOR":
            float(
                vals.median()
            ),

        "MEAN_FACTOR":
            float(
                vals.mean()
            ),

        "P95_FACTOR":
            float(
                vals.quantile(
                    0.95
                )
            ),

        "P99_FACTOR":
            float(
                vals.quantile(
                    0.99
                )
            ),

        "MAX_FACTOR":
            float(
                vals.max()
            )
    })


source_qc = pd.DataFrame(
    source_qc_rows
)


source_qc.to_csv(
    OUTPUT_SOURCE_QC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. SPATIAL SOURCE TREES
# ============================================================

print(
    "Building spatial source index..."
)


source_xy = source_df[
    [
        "MID_X",
        "MID_Y"
    ]
].to_numpy(
    dtype=float
)


global_tree = cKDTree(
    source_xy
)


global_source_positions = np.arange(
    len(source_df),
    dtype=int
)


class_trees = {}

class_source_positions = {}


for road_class, g in source_df.groupby(
    "ROAD_CLASS"
):

    positions = (
        g.index
        .to_numpy(
            dtype=int
        )
    )


    xy = source_df.loc[
        positions,
        [
            "MID_X",
            "MID_Y"
        ]
    ].to_numpy(
        dtype=float
    )


    class_trees[
        road_class
    ] = cKDTree(
        xy
    )


    class_source_positions[
        road_class
    ] = positions


source_id_to_position = {

    eid: i

    for i, eid in enumerate(
        source_df[
            "SUMO_EDGE_ID"
        ]
    )
}


class_source_counts = (
    source_df[
        "ROAD_CLASS"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# 12. FACTOR MEDIANS
# ============================================================

class_median_factor = {}

global_median_factor = {}


for hour in HOURS:

    factor_col = (
        f"SOURCE_FACTOR_{hour}"
    )


    vals = pd.to_numeric(
        source_df[
            factor_col
        ],
        errors="coerce"
    )


    vals = vals[
        np.isfinite(
            vals
        )
        &
        (
            vals >= 1.0
        )
    ]


    global_median_factor[
        hour
    ] = (
        float(
            np.median(
                vals
            )
        )
        if len(vals) > 0
        else 1.0
    )


    for road_class, g in source_df.groupby(
        "ROAD_CLASS"
    ):

        cvals = pd.to_numeric(
            g[
                factor_col
            ],
            errors="coerce"
        )


        cvals = cvals[
            np.isfinite(
                cvals
            )
            &
            (
                cvals >= 1.0
            )
        ]


        if len(cvals) > 0:

            class_median_factor[
                (
                    road_class,
                    hour
                )
            ] = float(
                np.median(
                    cvals
                )
            )


# ============================================================
# 13. KD-TREE QUERY
# ============================================================

def query_neighbors(
    tree,
    source_positions,
    point,
    k,
    radius,
    target_source_position=None
):

    if tree is None:
        return []


    n = len(
        source_positions
    )


    if n == 0:
        return []


    query_k = min(
        k + 1,
        n
    )


    dist, idx = tree.query(
        point,
        k=query_k,
        distance_upper_bound=radius
    )


    dist = np.atleast_1d(
        dist
    )

    idx = np.atleast_1d(
        idx
    )


    result = []


    for d, j in zip(
        dist,
        idx
    ):

        if not np.isfinite(
            d
        ):
            continue


        if j >= n:
            continue


        source_pos = int(
            source_positions[
                int(j)
            ]
        )


        if (
            target_source_position
            is not None
            and
            source_pos
            ==
            target_source_position
        ):
            continue


        result.append(
            source_pos
        )


        if len(result) >= k:
            break


    return result


# ============================================================
# 14. CHOOSE IMPUTATION METHOD
#
# hierarchy:
#
# 1 LOCAL_SAME_CLASS
# 2 ROAD_CLASS_MEDIAN
# 3 LOCAL_ANY_CLASS
# 4 GLOBAL_MEDIAN
# ============================================================

print(
    "Selecting V3 imputation sources..."
)


method_list = []

neighbor_positions_list = []

n_same_sources = []

n_any_sources = []


for _, row in df.iterrows():

    if (
        row[
            "UNOBSERVED_LENGTH"
        ]
        <=
        EPS
    ):

        method_list.append(
            "NOT_NEEDED"
        )

        neighbor_positions_list.append(
            []
        )

        n_same_sources.append(
            0
        )

        n_any_sources.append(
            0
        )

        continue


    eid = row[
        "SUMO_EDGE_ID"
    ]

    road_class = row[
        "ROAD_CLASS"
    ]


    point = np.array([
        row[
            "MID_X"
        ],
        row[
            "MID_Y"
        ]
    ])


    target_source_position = (
        source_id_to_position.get(
            eid
        )
    )


    # --------------------------------------------------------
    # 1. LOCAL SAME CLASS
    # --------------------------------------------------------

    same_neighbors = []


    if road_class in class_trees:

        same_neighbors = query_neighbors(

            class_trees[
                road_class
            ],

            class_source_positions[
                road_class
            ],

            point,

            SAME_CLASS_K,

            SAME_CLASS_RADIUS,

            target_source_position
        )


    n_same_sources.append(
        len(
            same_neighbors
        )
    )


    if len(
        same_neighbors
    ) > 0:

        method_list.append(
            "LOCAL_SAME_CLASS"
        )

        neighbor_positions_list.append(
            same_neighbors
        )

        n_any_sources.append(
            0
        )

        continue


    # --------------------------------------------------------
    # 2. ROAD CLASS MEDIAN
    # --------------------------------------------------------

    if (
        class_source_counts.get(
            road_class,
            0
        )
        >
        0
    ):

        method_list.append(
            "ROAD_CLASS_MEDIAN"
        )

        neighbor_positions_list.append(
            []
        )

        n_any_sources.append(
            0
        )

        continue


    # --------------------------------------------------------
    # 3. LOCAL ANY CLASS
    # --------------------------------------------------------

    any_neighbors = query_neighbors(

        global_tree,

        global_source_positions,

        point,

        ANY_CLASS_K,

        ANY_CLASS_RADIUS,

        target_source_position
    )


    n_any_sources.append(
        len(
            any_neighbors
        )
    )


    if len(
        any_neighbors
    ) > 0:

        method_list.append(
            "LOCAL_ANY_CLASS"
        )

        neighbor_positions_list.append(
            any_neighbors
        )

        continue


    # --------------------------------------------------------
    # 4. GLOBAL MEDIAN
    # --------------------------------------------------------

    method_list.append(
        "GLOBAL_MEDIAN"
    )

    neighbor_positions_list.append(
        []
    )


df[
    "IMPUTATION_METHOD"
] = method_list


df[
    "N_LOCAL_SAME_SOURCES"
] = n_same_sources


df[
    "N_LOCAL_ANY_SOURCES"
] = n_any_sources


df[
    "ROAD_CLASS_SOURCE_COUNT"
] = (
    df[
        "ROAD_CLASS"
    ]
    .map(
        class_source_counts
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# 15. BUILD FINAL HOURLY COST
# ============================================================

print(
    "Building V3 hourly costs..."
)


qc_rows = []


for hour in HOURS:

    obs_tt_col = (
        f"OBS_TT_{hour}"
    )

    factor_col = (
        f"SOURCE_FACTOR_{hour}"
    )


    final_tt_list = []

    final_speed_list = []

    imputed_factor_list = []


    for row_idx, row in df.iterrows():

        observed_tt = pd.to_numeric(
            pd.Series([
                row.get(
                    obs_tt_col,
                    np.nan
                )
            ]),
            errors="coerce"
        ).iloc[0]


        if not np.isfinite(
            observed_tt
        ):

            observed_tt = 0.0


        unobs_len = float(
            row[
                "UNOBSERVED_LENGTH"
            ]
        )


        # ----------------------------------------------------
        # Fully observed
        # ----------------------------------------------------

        if unobs_len <= EPS:

            factor = np.nan

            total_tt = (
                observed_tt
            )


        else:

            method = row[
                "IMPUTATION_METHOD"
            ]


            neighbors = (
                neighbor_positions_list[
                    row_idx
                ]
            )


            # ------------------------------------------------
            # LOCAL SAME CLASS
            # ------------------------------------------------

            if (
                method
                ==
                "LOCAL_SAME_CLASS"
            ):

                vals = pd.to_numeric(
                    source_df.loc[
                        neighbors,
                        factor_col
                    ],
                    errors="coerce"
                ).to_numpy(
                    dtype=float
                )


                vals = vals[
                    np.isfinite(
                        vals
                    )
                    &
                    (
                        vals >= 1.0
                    )
                ]


                if len(vals) > 0:

                    factor = float(
                        np.median(
                            vals
                        )
                    )

                else:

                    factor = (
                        class_median_factor.get(
                            (
                                row[
                                    "ROAD_CLASS"
                                ],
                                hour
                            ),
                            global_median_factor[
                                hour
                            ]
                        )
                    )


            # ------------------------------------------------
            # ROAD CLASS MEDIAN
            # ------------------------------------------------

            elif (
                method
                ==
                "ROAD_CLASS_MEDIAN"
            ):

                factor = (
                    class_median_factor.get(
                        (
                            row[
                                "ROAD_CLASS"
                            ],
                            hour
                        ),
                        global_median_factor[
                            hour
                        ]
                    )
                )


            # ------------------------------------------------
            # LOCAL ANY CLASS
            # ------------------------------------------------

            elif (
                method
                ==
                "LOCAL_ANY_CLASS"
            ):

                vals = pd.to_numeric(
                    source_df.loc[
                        neighbors,
                        factor_col
                    ],
                    errors="coerce"
                ).to_numpy(
                    dtype=float
                )


                vals = vals[
                    np.isfinite(
                        vals
                    )
                    &
                    (
                        vals >= 1.0
                    )
                ]


                factor = (
                    float(
                        np.median(
                            vals
                        )
                    )
                    if len(vals) > 0
                    else global_median_factor[
                        hour
                    ]
                )


            # ------------------------------------------------
            # GLOBAL MEDIAN
            # ------------------------------------------------

            else:

                factor = (
                    global_median_factor[
                        hour
                    ]
                )


            # numerical sanity
            factor = max(
                1.0,
                float(
                    factor
                )
            )


            imputed_pace = (
                float(
                    row[
                        "FREEFLOW_PACE"
                    ]
                )
                *
                factor
            )


            imputed_tt = (
                unobs_len
                *
                imputed_pace
            )


            total_tt = (
                observed_tt
                +
                imputed_tt
            )


        # ----------------------------------------------------
        # final safety
        # ----------------------------------------------------

        if (
            not np.isfinite(
                total_tt
            )
            or
            total_tt <= 0
        ):

            total_tt = (
                float(
                    row[
                        "SUMO_LENGTH"
                    ]
                )
                *
                float(
                    row[
                        "FREEFLOW_PACE"
                    ]
                )
            )


        final_speed = (
            float(
                row[
                    "SUMO_LENGTH"
                ]
            )
            /
            total_tt
            *
            3.6
        )


        final_tt_list.append(
            total_tt
        )


        final_speed_list.append(
            final_speed
        )


        imputed_factor_list.append(
            factor
        )


    df[
        f"TT_{hour}"
    ] = final_tt_list


    df[
        f"SPEED_{hour}"
    ] = final_speed_list


    df[
        f"IMPUTED_FACTOR_{hour}"
    ] = imputed_factor_list


    # ========================================================
    # QC
    # ========================================================

    tt_series = pd.Series(
        final_tt_list
    )


    speed_series = pd.Series(
        final_speed_list
    )


    ff = df[
        "FREEFLOW_SPEED_KMH"
    ]


    qc_rows.append({

        "HOUR":
            hour,

        "N_EDGES":
            len(df),

        "N_MISSING_TT":
            int(
                tt_series.isna().sum()
            ),

        "N_NONPOSITIVE_TT":
            int(
                (
                    tt_series <= 0
                ).sum()
            ),

        "MEAN_SPEED_KMH":
            float(
                speed_series.mean()
            ),

        "MEDIAN_SPEED_KMH":
            float(
                speed_series.median()
            ),

        "P05_SPEED_KMH":
            float(
                speed_series.quantile(
                    0.05
                )
            ),

        "P95_SPEED_KMH":
            float(
                speed_series.quantile(
                    0.95
                )
            ),

        # 완전 관측/partial 때문에 일부 존재 가능
        "N_SPEED_GT_1P5_FF":
            int(
                (
                    speed_series
                    >
                    1.5
                    *
                    ff
                ).sum()
            )
    })


# ============================================================
# 16. SAVE
# ============================================================

df = df.drop(
    columns=[
        "FREEFLOW_PACE",
        "MID_X",
        "MID_Y"
    ],
    errors="ignore"
)


df.to_csv(
    OUTPUT_COST,
    index=False,
    encoding="utf-8-sig"
)


qc = pd.DataFrame(
    qc_rows
)


qc.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "FULL NETWORK HOURLY COST V3"
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


needed = df[
    df[
        "UNOBSERVED_LENGTH"
    ]
    >
    EPS
]


print(
    "\nImputation method "
    "(actually requiring imputation):"
)

print(
    needed[
        "IMPUTATION_METHOD"
    ].value_counts()
)


print(
    "\n=============================="
)

print(
    "TOPIS RELATIVE FACTOR QC"
)

print(
    "=============================="
)


print(
    source_qc.to_string(
        index=False
    )
)


print(
    "\n=============================="
)

print(
    "FINAL HOURLY QC"
)

print(
    "=============================="
)


print(
    qc.to_string(
        index=False
    )
)


print(
    "\nCreated:"
)

print(
    " -",
    OUTPUT_COST.name
)

print(
    " -",
    OUTPUT_QC.name
)

print(
    " -",
    OUTPUT_SOURCE_QC.name
)