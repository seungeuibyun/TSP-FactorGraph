# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent

INPUT = (
    BASE / "sumo_hourly_cost_operational.csv"
)

OUTPUT = (
    BASE / "operational_cost_qc_summary.csv"
)


df = pd.read_csv(
    INPUT,
    dtype={
        "SUMO_EDGE_ID": str
    }
)


# ============================================================
# 1. REGION × IMPUTATION METHOD
# ============================================================

table = pd.crosstab(
    df["REGION_TYPE"],
    df["IMPUTATION_METHOD"]
)


print(
    "\n=============================="
)

print(
    "REGION × IMPUTATION METHOD"
)

print(
    "=============================="
)

print(
    table
)


# ============================================================
# 2. REGION별 비율
# ============================================================

pct = (
    table.div(
        table.sum(axis=1),
        axis=0
    )
    * 100
)


print(
    "\n=============================="
)

print(
    "REGION × IMPUTATION METHOD (%)"
)

print(
    "=============================="
)

print(
    pct.round(2)
)


# ============================================================
# 3. REGION × COST SOURCE
# ============================================================

source_table = pd.crosstab(
    df["REGION_TYPE"],
    df["COST_SOURCE"]
)


print(
    "\n=============================="
)

print(
    "REGION × COST SOURCE"
)

print(
    "=============================="
)

print(
    source_table
)


source_pct = (
    source_table.div(
        source_table.sum(axis=1),
        axis=0
    )
    * 100
)


print(
    "\n=============================="
)

print(
    "REGION × COST SOURCE (%)"
)

print(
    "=============================="
)

print(
    source_pct.round(2)
)


# ============================================================
# 4. SUMMARY FILE
# ============================================================

rows = []


for region, g in df.groupby(
    "REGION_TYPE"
):

    n = len(g)

    rows.append({

        "REGION_TYPE":
            region,

        "N_EDGES":
            n,

        "OBSERVED":
            int(
                (
                    g["COST_SOURCE"]
                    ==
                    "OBSERVED"
                ).sum()
            ),

        "PARTIAL":
            int(
                (
                    g["COST_SOURCE"]
                    ==
                    "PARTIAL_OBSERVED_PLUS_IMPUTED"
                ).sum()
            ),

        "IMPUTED":
            int(
                (
                    g["COST_SOURCE"]
                    ==
                    "IMPUTED"
                ).sum()
            ),

        "LOCAL_SAME_CLASS":
            int(
                (
                    g["IMPUTATION_METHOD"]
                    ==
                    "LOCAL_SAME_CLASS"
                ).sum()
            ),

        "LOCAL_ANY_CLASS":
            int(
                (
                    g["IMPUTATION_METHOD"]
                    ==
                    "LOCAL_ANY_CLASS"
                ).sum()
            ),

        "MEDIAN_FALLBACK":
            int(
                (
                    g["IMPUTATION_METHOD"]
                    ==
                    "MEDIAN_FALLBACK"
                ).sum()
            ),

        "SAME_CLASS_RATIO":
            float(
                (
                    g["IMPUTATION_METHOD"]
                    ==
                    "LOCAL_SAME_CLASS"
                ).mean()
            ),

        "ANY_CLASS_RATIO":
            float(
                (
                    g["IMPUTATION_METHOD"]
                    ==
                    "LOCAL_ANY_CLASS"
                ).mean()
            ),

        "FALLBACK_RATIO":
            float(
                (
                    g["IMPUTATION_METHOD"]
                    ==
                    "MEDIAN_FALLBACK"
                ).mean()
            )
    })


summary = pd.DataFrame(
    rows
)


summary.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nCreated:",
    OUTPUT.name
)