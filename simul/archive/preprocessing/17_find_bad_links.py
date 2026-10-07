# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(__file__).resolve().parent

INPUT = (
    BASE / "reconciliation_v2_link_error.csv"
)

OUTPUT = (
    BASE / "reconciliation_bad_links.csv"
)


# ============================================================
# 1. LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype={
        "TOPIS_LINK_ID": str
    }
)

df["TOPIS_LINK_ID"] = (
    df["TOPIS_LINK_ID"]
    .str.strip()
)


# ============================================================
# 2. LINK-LEVEL SUMMARY
# ============================================================

summary = (
    df.groupby(
        "TOPIS_LINK_ID"
    )
    .agg(
        AFFECTED=(
            "AFFECTED",
            "max"
        ),

        MEAN_REL_ERROR=(
            "REL_ERROR",
            "mean"
        ),

        MAX_REL_ERROR=(
            "REL_ERROR",
            "max"
        ),

        MEAN_ABS_ERROR_SEC=(
            "ABS_ERROR_SEC",
            "mean"
        ),

        MAX_ABS_ERROR_SEC=(
            "ABS_ERROR_SEC",
            "max"
        ),

        N_HOURS_GT_10PCT=(
            "REL_ERROR",
            lambda x: int(
                (x > 0.10).sum()
            )
        ),

        N_HOURS_GT_20PCT=(
            "REL_ERROR",
            lambda x: int(
                (x > 0.20).sum()
            )
        ),

        N_HOURS_GT_50PCT=(
            "REL_ERROR",
            lambda x: int(
                (x > 0.50).sum()
            )
        )
    )
    .reset_index()
)


# ============================================================
# 3. BADNESS SCORE
#
# 지속적으로 큰 오류가 나는 링크를 위로
# ============================================================

summary["BADNESS_SCORE"] = (
    summary["MEAN_REL_ERROR"]
    +
    0.5
    * summary["MAX_REL_ERROR"]
)


summary = summary.sort_values(
    [
        "BADNESS_SCORE",
        "MAX_REL_ERROR"
    ],
    ascending=False
)


summary.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. PRINT
# ============================================================

print(
    "\n=============================="
)

print(
    "WORST TOPIS LINKS"
)

print(
    "==============================\n"
)


print(
    summary[
        [
            "TOPIS_LINK_ID",
            "MEAN_REL_ERROR",
            "MAX_REL_ERROR",
            "MEAN_ABS_ERROR_SEC",
            "MAX_ABS_ERROR_SEC",
            "N_HOURS_GT_10PCT",
            "N_HOURS_GT_20PCT",
            "N_HOURS_GT_50PCT"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\n=============================="
)

print(
    "ERROR COUNTS"
)

print(
    "=============================="
)


print(
    "Links ever >10%:",
    (
        summary[
            "MAX_REL_ERROR"
        ]
        > 0.10
    ).sum()
)

print(
    "Links ever >20%:",
    (
        summary[
            "MAX_REL_ERROR"
        ]
        > 0.20
    ).sum()
)

print(
    "Links ever >50%:",
    (
        summary[
            "MAX_REL_ERROR"
        ]
        > 0.50
    ).sum()
)

print(
    "Links mean error >20%:",
    (
        summary[
            "MEAN_REL_ERROR"
        ]
        > 0.20
    ).sum()
)


print(
    "\nCreated:",
    OUTPUT.name
)