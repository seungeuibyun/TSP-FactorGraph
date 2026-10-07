# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from collections import defaultdict

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
    BASE / "sumo_hourly_cost_full_network_v2.csv"
)

OUTPUT_QC = (
    BASE / "sumo_hourly_cost_full_network_v2_qc.csv"
)


# ============================================================
# 1. PARAMETERS
# ============================================================

EPS = 1e-9

# 같은 road class의 주변 관측 edge
SAME_CLASS_RADIUS = 1500.0
SAME_CLASS_K = 5

# class median도 없을 때만 다른 도로 종류 사용
ANY_CLASS_RADIUS = 2000.0
ANY_CLASS_K = 8

# geometry상 사실상 전부 관측된 것으로 볼 threshold
FULL_OBS_THRESHOLD = 0.999

# ------------------------------------------------------------
# 핵심 physical sanity constraint
#
# imputed congestion factor c = pace / free-flow pace
#
# c >= 1
#
# 즉, 추정 때문에 SUMO free-flow보다 더 빨라지지 않도록 함.
#
# 직접 TOPIS 관측값에는 적용하지 않는다.
# ------------------------------------------------------------

MIN_IMPUTED_FACTOR = 1.0


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

    """
    SUMO / OSM road type을 논문에서 다루기 쉬운
    coarse road class로 정규화.

    type 정보가 불분명하면 free-flow speed band 사용.
    """

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
    # fallback: SUMO free-flow speed band
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
# 4. BUILD FULL SUMO EDGE TABLE
# ============================================================

print("Building full SUMO edge table...")


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

        # 극히 예외적인 경우
        from_node = edge.getFromNode()

        mid_x, mid_y = (
            from_node.getCoord()
        )


    ff_speed_kmh = (
        speed_mps
        *
        3.6
    )


    ff_pace = (
        1.0
        /
        speed_mps
    )   # sec / m


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
            ff_speed_kmh,

        "FREEFLOW_PACE":
            ff_pace,

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
# 5. LOAD DIRECT TOPIS OBSERVATION
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


# geometry 오차 때문에 observed length가
# SUMO length보다 살짝 길게 나오는 경우 제한
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
    lower=0.0,
    upper=1.0
)


direct_support = (
    df["OBSERVED_LENGTH"]
    >
    EPS
)


print(
    "Edges with direct TOPIS support:",
    int(
        direct_support.sum()
    )
)


# ============================================================
# 7. COST SOURCE
# ============================================================

def cost_source(row):

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

    else:

        return "IMPUTED"


df["COST_SOURCE"] = (
    df.apply(
        cost_source,
        axis=1
    )
)


# ============================================================
# 8. OBSERVED SOURCE FACTORS
#
# c_e(h) =
#
#    observed pace
#    -------------
#    free-flow pace
#
# 여기서는 절대 clip하지 않는다.
#
# 실제 TOPIS 관측값은 그대로 보존한다.
# ============================================================

source_mask = (
    df["OBSERVED_LENGTH"]
    >
    EPS
)


source_df = (
    df[
        source_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


for hour in HOURS:

    pace_col = (
        f"OBS_PACE_{hour}"
    )

    factor_col = (
        f"SOURCE_FACTOR_{hour}"
    )


    obs_pace = pd.to_numeric(
        source_df[
            pace_col
        ],
        errors="coerce"
    )


    source_df[
        factor_col
    ] = (
        obs_pace
        /
        source_df[
            "FREEFLOW_PACE"
        ]
    )


# ============================================================
# 9. SOURCE INDEX / KD TREE
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


# ------------------------------------------------------------
# class별 source tree
# ------------------------------------------------------------

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


    if len(xy) == 0:
        continue


    class_trees[
        road_class
    ] = cKDTree(
        xy
    )


    class_source_positions[
        road_class
    ] = positions


# ============================================================
# 10. ROAD-CLASS / GLOBAL MEDIAN FACTORS
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
            vals > 0
        )
    ]


    if len(vals) > 0:

        global_median_factor[
            hour
        ] = float(
            np.median(
                vals
            )
        )

    else:

        # 최후의 numerical fallback
        global_median_factor[
            hour
        ] = 1.0


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
                cvals > 0
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
# 11. KD QUERY
# ============================================================

source_id_to_position = {
    eid: i
    for i, eid in enumerate(
        source_df[
            "SUMO_EDGE_ID"
        ]
    )
}


def query_neighbors(
    tree,
    source_positions,
    point,
    k,
    radius,
    target_source_position=None
):

    """
    KDTree result -> source_df row positions.

    target edge가 직접 TOPIS source인 경우에는
    자기 자신은 제외한다.
    """

    if tree is None:
        return []


    n = len(
        source_positions
    )


    if n == 0:
        return []


    # 자기 자신을 제외할 여유로 +1
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


# global tree는 source_df position과 KD index가 동일
global_source_positions = np.arange(
    len(source_df),
    dtype=int
)


# ============================================================
# 12. PRECOMPUTE IMPUTATION SOURCE
#
# New hierarchy:
#
# 1 LOCAL_SAME_CLASS
# 2 ROAD_CLASS_MEDIAN
# 3 LOCAL_ANY_CLASS
# 4 GLOBAL_MEDIAN
# ============================================================

print(
    "Selecting imputation sources..."
)


method_list = []

neighbor_positions_list = []

n_same_sources = []

n_any_sources = []

class_source_counts = (
    source_df[
        "ROAD_CLASS"
    ]
    .value_counts()
    .to_dict()
)


for i, row in df.iterrows():

    unobserved_length = float(
        row[
            "UNOBSERVED_LENGTH"
        ]
    )


    # --------------------------------------------------------
    # 사실상 전부 관측된 edge
    # --------------------------------------------------------

    if unobserved_length <= EPS:

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
    # 1. local same road class
    # --------------------------------------------------------

    same_neighbors = []


    if (
        road_class
        in
        class_trees
    ):

        same_neighbors = query_neighbors(

            tree=
                class_trees[
                    road_class
                ],

            source_positions=
                class_source_positions[
                    road_class
                ],

            point=
                point,

            k=
                SAME_CLASS_K,

            radius=
                SAME_CLASS_RADIUS,

            target_source_position=
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
    # 2. global median within SAME road class
    # --------------------------------------------------------

    if (
        road_class
        in
        class_source_counts
        and
        class_source_counts[
            road_class
        ]
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
    # 3. local any road class
    #
    # SAME CLASS 데이터 자체가 전혀 없을 때만 사용
    # --------------------------------------------------------

    any_neighbors = query_neighbors(

        tree=
            global_tree,

        source_positions=
            global_source_positions,

        point=
            point,

        k=
            ANY_CLASS_K,

        radius=
            ANY_CLASS_RADIUS,

        target_source_position=
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
    # 4. global median
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
# 13. COMPUTE FINAL HOURLY COST
# ============================================================

print(
    "Building hourly costs..."
)


qc_rows = []


for hour in HOURS:

    obs_tt_col = (
        f"OBS_TT_{hour}"
    )

    source_factor_col = (
        f"SOURCE_FACTOR_{hour}"
    )


    final_tt = []

    final_speed = []

    imputed_factor = []

    raw_imputed_factor = []

    factor_was_clamped = []


    for i, row in df.iterrows():

        observed_length = float(
            row[
                "OBSERVED_LENGTH"
            ]
        )

        unobserved_length = float(
            row[
                "UNOBSERVED_LENGTH"
            ]
        )


        ff_pace = float(
            row[
                "FREEFLOW_PACE"
            ]
        )


        # ----------------------------------------------------
        # observed travel time
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # no imputation required
        # ----------------------------------------------------

        if unobserved_length <= EPS:

            factor_raw = np.nan

            factor_final = np.nan

            was_clamped = False

            total_tt = observed_tt


        else:

            method = row[
                "IMPUTATION_METHOD"
            ]


            neighbor_positions = (
                neighbor_positions_list[
                    i
                ]
            )


            # ------------------------------------------------
            # 1. LOCAL SAME CLASS
            # ------------------------------------------------

            if (
                method
                ==
                "LOCAL_SAME_CLASS"
            ):

                vals = pd.to_numeric(
                    source_df.loc[
                        neighbor_positions,
                        source_factor_col
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
                        vals > 0
                    )
                ]


                if len(vals) > 0:

                    factor_raw = float(
                        np.median(
                            vals
                        )
                    )

                else:

                    factor_raw = (
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
            # 2. ROAD CLASS MEDIAN
            # ------------------------------------------------

            elif (
                method
                ==
                "ROAD_CLASS_MEDIAN"
            ):

                factor_raw = (
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
            # 3. LOCAL ANY CLASS
            # ------------------------------------------------

            elif (
                method
                ==
                "LOCAL_ANY_CLASS"
            ):

                vals = pd.to_numeric(
                    source_df.loc[
                        neighbor_positions,
                        source_factor_col
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
                        vals > 0
                    )
                ]


                if len(vals) > 0:

                    factor_raw = float(
                        np.median(
                            vals
                        )
                    )

                else:

                    factor_raw = (
                        global_median_factor[
                            hour
                        ]
                    )


            # ------------------------------------------------
            # 4. GLOBAL MEDIAN
            # ------------------------------------------------

            else:

                factor_raw = (
                    global_median_factor[
                        hour
                    ]
                )


            # ------------------------------------------------
            # PHYSICAL CONSTRAINT
            #
            # imputed part only:
            #
            # c_hat >= 1
            #
            # ------------------------------------------------

            factor_final = max(
                MIN_IMPUTED_FACTOR,
                float(
                    factor_raw
                )
            )


            was_clamped = (
                factor_final
                >
                factor_raw
                +
                EPS
            )


            imputed_pace = (
                ff_pace
                *
                factor_final
            )


            imputed_tt = (
                unobserved_length
                *
                imputed_pace
            )


            total_tt = (
                observed_tt
                +
                imputed_tt
            )


        # ----------------------------------------------------
        # safety
        # ----------------------------------------------------

        if (
            not np.isfinite(
                total_tt
            )
            or
            total_tt <= 0
        ):

            # numerical final fallback
            total_tt = (
                float(
                    row[
                        "SUMO_LENGTH"
                    ]
                )
                *
                ff_pace
            )


        speed_kmh = (
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


        final_tt.append(
            total_tt
        )

        final_speed.append(
            speed_kmh
        )

        imputed_factor.append(
            factor_final
        )

        raw_imputed_factor.append(
            factor_raw
        )

        factor_was_clamped.append(
            was_clamped
        )


    # --------------------------------------------------------
    # SAVE HOURLY COLUMNS
    # --------------------------------------------------------

    df[
        f"TT_{hour}"
    ] = final_tt


    df[
        f"SPEED_{hour}"
    ] = final_speed


    df[
        f"IMPUTED_FACTOR_{hour}"
    ] = imputed_factor


    df[
        f"RAW_IMPUTED_FACTOR_{hour}"
    ] = raw_imputed_factor


    df[
        f"FACTOR_CLAMPED_{hour}"
    ] = factor_was_clamped


    speed_series = pd.Series(
        final_speed
    )


    tt_series = pd.Series(
        final_tt
    )


    ff_series = df[
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

        "N_FACTOR_CLAMPED_TO_1":
            int(
                np.sum(
                    factor_was_clamped
                )
            ),

        "N_SPEED_GT_1P5_FF":
            int(
                (
                    speed_series
                    >
                    1.5
                    *
                    ff_series
                ).sum()
            )
    })


# ============================================================
# 14. CLEAN INTERNAL WORKING COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "FREEFLOW_PACE",
        "MID_X",
        "MID_Y",
    ],
    errors="ignore"
)


# ============================================================
# 15. SAVE
# ============================================================

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
# 16. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "FULL NETWORK HOURLY COST V2"
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


# ------------------------------------------------------------
# 실제 imputation이 필요한 edge만 별도 집계
# ------------------------------------------------------------

needed = df[
    df[
        "UNOBSERVED_LENGTH"
    ]
    >
    EPS
]


print(
    "\nImputation method "
    "(edges actually requiring imputation):"
)

print(
    needed[
        "IMPUTATION_METHOD"
    ].value_counts()
)


missing_tt = 0


for hour in HOURS:

    missing_tt += int(
        df[
            f"TT_{hour}"
        ].isna().sum()
    )


print(
    "\nMissing final TT:",
    missing_tt
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
    OUTPUT_COST.name
)

print(
    "Created:",
    OUTPUT_QC.name
)