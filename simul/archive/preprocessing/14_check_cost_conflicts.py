# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

COST_FILE = (
    BASE / "sumo_topis_hourly_cost_final.csv"
)

OUTPUT_SUMMARY = (
    BASE / "sumo_edge_conflict_summary.csv"
)

OUTPUT_DETAIL = (
    BASE / "sumo_edge_conflict_detail.csv"
)


# ============================================================
# 1. COST DATA
# ============================================================

df = pd.read_csv(
    COST_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

df["TOPIS_LINK_ID"] = (
    df["TOPIS_LINK_ID"]
    .str.strip()
)

df["SUMO_EDGE_ID"] = (
    df["SUMO_EDGE_ID"]
    .str.strip()
)


hour_cols = [
    f"TT_~{h:02d}시"
    for h in range(1, 25)
]


# ============================================================
# 2. 실제 conflict edge만
# ============================================================

n_links = (
    df.groupby(
        "SUMO_EDGE_ID"
    )["TOPIS_LINK_ID"]
    .nunique()
)


conflict_ids = (
    n_links[
        n_links > 1
    ]
    .index
)


conflict = df[
    df["SUMO_EDGE_ID"].isin(
        conflict_ids
    )
].copy()


print(
    "Conflict SUMO edges:",
    len(conflict_ids)
)


# ============================================================
# 3. 몇 개 TOPIS link가 겹치는지
# ============================================================

print(
    "\n===== NUMBER OF TOPIS LINKS PER EDGE ====="
)

print(
    n_links[
        n_links > 1
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. conflict severity
#
# 동일 SUMO edge에 대해
# TOPIS link별 TT 후보가 얼마나 다른지 측정
#
# relative spread:
#
#     (max TT - min TT) / mean TT
#
# ============================================================

summary_rows = []


for edge_id, g in conflict.groupby(
    "SUMO_EDGE_ID"
):

    links = sorted(
        g["TOPIS_LINK_ID"]
        .unique()
    )

    hourly_relative_spreads = []
    hourly_abs_spreads = []

    for col in hour_cols:

        values = (
            pd.to_numeric(
                g[col],
                errors="coerce"
            )
            .dropna()
            .values
        )

        if len(values) < 2:
            continue

        vmin = float(
            np.min(values)
        )

        vmax = float(
            np.max(values)
        )

        vmean = float(
            np.mean(values)
        )

        abs_spread = (
            vmax - vmin
        )

        if vmean > 0:

            rel_spread = (
                abs_spread
                / vmean
            )

        else:

            rel_spread = np.nan

        hourly_abs_spreads.append(
            abs_spread
        )

        hourly_relative_spreads.append(
            rel_spread
        )


    summary_rows.append({

        "SUMO_EDGE_ID":
            edge_id,

        "N_TOPIS_LINKS":
            len(links),

        "TOPIS_LINKS":
            ",".join(links),

        "SUMO_LENGTH":
            float(
                g[
                    "SUMO_LENGTH"
                ].iloc[0]
            ),

        "MEAN_REL_SPREAD":
            float(
                np.nanmean(
                    hourly_relative_spreads
                )
            ),

        "MAX_REL_SPREAD":
            float(
                np.nanmax(
                    hourly_relative_spreads
                )
            ),

        "MEAN_ABS_SPREAD_SEC":
            float(
                np.nanmean(
                    hourly_abs_spreads
                )
            ),

        "MAX_ABS_SPREAD_SEC":
            float(
                np.nanmax(
                    hourly_abs_spreads
                )
            )
    })


summary = pd.DataFrame(
    summary_rows
)


summary = summary.sort_values(
    "MAX_REL_SPREAD",
    ascending=False
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. raw detail도 저장
# ============================================================

detail_cols = [
    "SUMO_EDGE_ID",
    "TOPIS_LINK_ID",
    "SEQ",
    "SUMO_LENGTH",
    "TOPIS_LENGTH",
    "LENGTH_WEIGHT"
] + hour_cols


conflict[
    detail_cols
].sort_values(
    [
        "SUMO_EDGE_ID",
        "TOPIS_LINK_ID"
    ]
).to_csv(
    OUTPUT_DETAIL,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. severity buckets
# ============================================================

def severity(x):

    if x <= 0.10:
        return "<=10%"

    elif x <= 0.20:
        return "10-20%"

    elif x <= 0.50:
        return "20-50%"

    else:
        return ">50%"


summary["SEVERITY"] = (
    summary["MAX_REL_SPREAD"]
    .apply(
        severity
    )
)


print(
    "\n===== CONFLICT SEVERITY ====="
)

print(
    summary[
        "SEVERITY"
    ].value_counts()
)


# ============================================================
# 7. 가장 심한 것 20개
# ============================================================

print(
    "\n===== TOP 20 CONFLICTS =====\n"
)

print(
    summary[
        [
            "SUMO_EDGE_ID",
            "N_TOPIS_LINKS",
            "TOPIS_LINKS",
            "MEAN_REL_SPREAD",
            "MAX_REL_SPREAD",
            "MAX_ABS_SPREAD_SEC"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nCreated:",
    OUTPUT_SUMMARY.name
)

print(
    "Created:",
    OUTPUT_DETAIL.name
)