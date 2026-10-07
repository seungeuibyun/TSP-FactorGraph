# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent

INPUT = (
    BASE / "sanity_check_outlier_edges_v3.csv"
)

OUTPUT = (
    BASE / "traffic_outlier_summary_v3.csv"
)

df = pd.read_csv(
    INPUT,
    dtype={
        "SUMO_EDGE_ID": str
    }
)


# ============================================================
# 1. OUTLIER TYPE
# ============================================================

def classify(row):

    types = []

    speed = row["FINAL_SPEED_KMH"]
    ff = row["FREEFLOW_SPEED_KMH"]
    c = row["CONGESTION_RATIO"]

    if speed < 2:
        types.append(
            "VERY_LOW_SPEED"
        )

    if speed > 1.5 * ff:
        types.append(
            "ABOVE_1P5_FREEFLOW"
        )

    if c > 10:
        types.append(
            "CONGESTION_GT_10"
        )

    if c < 0.5:
        types.append(
            "CONGESTION_LT_0P5"
        )

    return "|".join(types)


df["OUTLIER_TYPE"] = (
    df.apply(
        classify,
        axis=1
    )
)


# ============================================================
# 2. BASIC DISTRIBUTION
# ============================================================

print(
    "\n=============================="
)

print(
    "OUTLIER TYPE"
)

print(
    "=============================="
)

print(
    df[
        "OUTLIER_TYPE"
    ].value_counts()
)


print(
    "\n=============================="
)

print(
    "REGION"
)

print(
    "=============================="
)

print(
    pd.crosstab(
        df["REGION_TYPE"],
        df["OUTLIER_TYPE"]
    )
)


print(
    "\n=============================="
)

print(
    "IMPUTATION METHOD"
)

print(
    "=============================="
)

print(
    pd.crosstab(
        df["IMPUTATION_METHOD"],
        df["OUTLIER_TYPE"]
    )
)


print(
    "\n=============================="
)

print(
    "COST SOURCE"
)

print(
    "=============================="
)

print(
    pd.crosstab(
        df["COST_SOURCE"],
        df["OUTLIER_TYPE"]
    )
)


# ============================================================
# 3. EDGE-LEVEL SUMMARY
# ============================================================

summary = (
    df.groupby(
        "SUMO_EDGE_ID"
    )
    .agg(
        N_OUTLIER_HOURS=(
            "HOUR",
            "nunique"
        ),

        OUTLIER_TYPES=(
            "OUTLIER_TYPE",
            lambda x:
                "|".join(
                    sorted(
                        set(x)
                    )
                )
        ),

        REGION_TYPE=(
            "REGION_TYPE",
            "first"
        ),

        ROAD_CLASS=(
            "ROAD_CLASS",
            "first"
        ),

        COST_SOURCE=(
            "COST_SOURCE",
            "first"
        ),

        IMPUTATION_METHOD=(
            "IMPUTATION_METHOD",
            "first"
        ),

        FREEFLOW_SPEED_KMH=(
            "FREEFLOW_SPEED_KMH",
            "first"
        ),

        MIN_SPEED_KMH=(
            "FINAL_SPEED_KMH",
            "min"
        ),

        MAX_SPEED_KMH=(
            "FINAL_SPEED_KMH",
            "max"
        ),

        MAX_CONGESTION_RATIO=(
            "CONGESTION_RATIO",
            "max"
        ),

        MIN_CONGESTION_RATIO=(
            "CONGESTION_RATIO",
            "min"
        ),
    )
    .reset_index()
)


summary = (
    summary.sort_values(
        [
            "N_OUTLIER_HOURS",
            "MAX_CONGESTION_RATIO"
        ],
        ascending=[
            False,
            False
        ]
    )
)


summary.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. RECURRING OUTLIERS
# ============================================================

recurring = summary[
    summary[
        "N_OUTLIER_HOURS"
    ]
    >=
    3
]


print(
    "\n=============================="
)

print(
    "RECURRING OUTLIERS >= 3 HOURS"
)

print(
    "=============================="
)


print(
    "Number of edges:",
    len(recurring)
)


if len(recurring) > 0:

    print(
        recurring[
            [
                "SUMO_EDGE_ID",
                "N_OUTLIER_HOURS",
                "OUTLIER_TYPES",
                "REGION_TYPE",
                "ROAD_CLASS",
                "COST_SOURCE",
                "IMPUTATION_METHOD",
                "FREEFLOW_SPEED_KMH",
                "MIN_SPEED_KMH",
                "MAX_SPEED_KMH",
                "MAX_CONGESTION_RATIO"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 5. VERY HIGH SPEED CASES
# ============================================================

fast = df[
    df[
        "OUTLIER_TYPE"
    ].str.contains(
        "ABOVE_1P5_FREEFLOW",
        na=False
    )
]


print(
    "\n=============================="
)

print(
    "ABOVE 1.5 × FREE-FLOW"
)

print(
    "=============================="
)


if len(fast) == 0:

    print(
        "None"
    )

else:

    print(
        fast[
            [
                "SUMO_EDGE_ID",
                "HOUR",
                "REGION_TYPE",
                "ROAD_CLASS",
                "COST_SOURCE",
                "IMPUTATION_METHOD",
                "FREEFLOW_SPEED_KMH",
                "FINAL_SPEED_KMH",
                "CONGESTION_RATIO"
            ]
        ]
        .sort_values(
            "FINAL_SPEED_KMH",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


print(
    "\nCreated:",
    OUTPUT.name
)