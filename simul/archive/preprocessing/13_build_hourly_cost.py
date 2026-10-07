# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_final.csv"
)

SPEED_FILE = (
    BASE / "topis_speed.xlsx"
)

OUTPUT_COST = (
    BASE / "sumo_topis_hourly_cost_final.csv"
)

OUTPUT_QC = (
    BASE / "hourly_cost_qc.csv"
)

OUTPUT_CONFLICT = (
    BASE / "sumo_edge_topis_conflicts.csv"
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
# 2. FINAL MAPPING
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

mapping["TOPIS_LINK_ID"] = (
    mapping["TOPIS_LINK_ID"]
    .str.strip()
)

mapping["SUMO_EDGE_ID"] = (
    mapping["SUMO_EDGE_ID"]
    .str.strip()
)


print(
    "Mapped TOPIS links:",
    mapping["TOPIS_LINK_ID"].nunique()
)

print(
    "Mapping rows:",
    len(mapping)
)


# ============================================================
# 3. SUMO EDGE LENGTH
# ============================================================

def get_edge_length(eid):

    try:

        return float(
            net.getEdge(
                eid
            ).getLength()
        )

    except Exception:

        raise RuntimeError(
            f"SUMO edge not found: {eid}"
        )


mapping["SUMO_LENGTH"] = (
    mapping["SUMO_EDGE_ID"]
    .apply(
        get_edge_length
    )
)


# ============================================================
# 4. 한 TOPIS link에 매칭된
#    SUMO edge 총 길이
# ============================================================

matched_length = (
    mapping
    .groupby(
        "TOPIS_LINK_ID"
    )["SUMO_LENGTH"]
    .sum()
    .rename(
        "MATCHED_SUMO_LENGTH"
    )
)


mapping = mapping.merge(
    matched_length,
    on="TOPIS_LINK_ID",
    how="left"
)


mapping["LENGTH_WEIGHT"] = (
    mapping["SUMO_LENGTH"]
    /
    mapping["MATCHED_SUMO_LENGTH"]
)


# ============================================================
# 5. TOPIS SPEED DATA
# ============================================================

speed = pd.read_excel(
    SPEED_FILE,
    dtype={
        "링크아이디": str
    }
)

speed["링크아이디"] = (
    speed["링크아이디"]
    .str.strip()
)


# mapping에 실제 사용되는 182개만
speed = speed[
    speed["링크아이디"].isin(
        set(
            mapping[
                "TOPIS_LINK_ID"
            ]
        )
    )
].copy()


print(
    "TOPIS speed links used:",
    speed["링크아이디"].nunique()
)


# ============================================================
# 6. 필요한 column 검사
# ============================================================

hour_cols = [
    f"~{h:02d}시"
    for h in range(
        1,
        25
    )
]


required_cols = [
    "링크아이디",
    "거리"
] + hour_cols


missing_cols = [
    col
    for col in required_cols
    if col not in speed.columns
]


if missing_cols:

    raise RuntimeError(
        "Missing TOPIS columns: "
        + str(
            missing_cols
        )
    )


# ============================================================
# 7. numeric conversion
# ============================================================

speed["거리"] = pd.to_numeric(
    speed["거리"],
    errors="coerce"
)


for col in hour_cols:

    speed[col] = pd.to_numeric(
        speed[col],
        errors="coerce"
    )


# ============================================================
# 8. mapping + TOPIS data merge
# ============================================================

speed = speed.rename(
    columns={
        "링크아이디":
            "TOPIS_LINK_ID",

        "거리":
            "TOPIS_LENGTH"
    }
)


df = mapping.merge(
    speed,
    on="TOPIS_LINK_ID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 9. 누락 검사
# ============================================================

missing_length = df[
    "TOPIS_LENGTH"
].isna().sum()


print(
    "Rows missing TOPIS length:",
    missing_length
)


if missing_length > 0:

    bad = (
        df.loc[
            df["TOPIS_LENGTH"].isna(),
            "TOPIS_LINK_ID"
        ]
        .unique()
    )

    raise RuntimeError(
        "Missing TOPIS speed/length data for: "
        + str(
            bad
        )
    )


# ============================================================
# 10. HOURLY TRAVEL TIME
# ============================================================
#
# TOPIS:
#
#   T_l(h)
#   =
#   L_topis / (v_topis / 3.6)
#
# SUMO edge:
#
#   T_e(h)
#   =
#   T_l(h) * LENGTH_WEIGHT
#
# ============================================================

for col in hour_cols:

    speed_kmh = df[
        col
    ]

    # 0 또는 음수 속도는 유효하지 않음
    valid_speed = (
        speed_kmh > 0
    )

    topis_tt = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    topis_tt.loc[
        valid_speed
    ] = (
        df.loc[
            valid_speed,
            "TOPIS_LENGTH"
        ]
        /
        (
            speed_kmh.loc[
                valid_speed
            ]
            /
            3.6
        )
    )

    # 원 TOPIS link travel time도 저장
    df[
        f"TOPIS_TT_{col}"
    ] = topis_tt

    # SUMO edge travel time
    df[
        f"TT_{col}"
    ] = (
        topis_tt
        *
        df[
            "LENGTH_WEIGHT"
        ]
    )


# ============================================================
# 11. 최종 cost table
# ============================================================

tt_cols = [
    f"TT_{col}"
    for col in hour_cols
]


output_cols = [
    "TOPIS_LINK_ID",
    "SUMO_EDGE_ID",
    "SEQ",
    "SUMO_LENGTH",
    "MATCHED_SUMO_LENGTH",
    "TOPIS_LENGTH",
    "LENGTH_WEIGHT"
] + tt_cols


cost = df[
    output_cols
].copy()


cost = cost.sort_values(
    [
        "TOPIS_LINK_ID",
        "SEQ"
    ]
)


cost.to_csv(
    OUTPUT_COST,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. TRAVEL-TIME CONSERVATION QC
#
# SUM_e T_e(h) == T_l(h)
# ============================================================

qc_rows = []


for col in hour_cols:

    tt_col = f"TT_{col}"
    topis_tt_col = (
        f"TOPIS_TT_{col}"
    )

    edge_sum = (
        df.groupby(
            "TOPIS_LINK_ID"
        )[tt_col]
        .sum(
            min_count=1
        )
    )

    observed = (
        df.groupby(
            "TOPIS_LINK_ID"
        )[topis_tt_col]
        .first()
    )

    error = (
        edge_sum
        -
        observed
    ).abs()

    qc_rows.append({
        "HOUR":
            col,

        "N_VALID_LINKS":
            int(
                observed.notna().sum()
            ),

        "N_MISSING_LINKS":
            int(
                observed.isna().sum()
            ),

        "MAX_ABS_ERROR_SEC":
            float(
                error.max()
            )
            if error.notna().any()
            else np.nan,

        "MEAN_ABS_ERROR_SEC":
            float(
                error.mean()
            )
            if error.notna().any()
            else np.nan
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
# 13. SUMO edge가 여러 TOPIS link에
#     겹쳐 매칭됐는지 확인
# ============================================================
#
# 이건 지금 cost table 생성 자체에는 문제 없음.
# 하지만 나중에 SUMO edge 하나에 최종 cost를
# 넣을 때는 반드시 정리해야 함.
#
# ============================================================

edge_link_count = (
    cost.groupby(
        "SUMO_EDGE_ID"
    )["TOPIS_LINK_ID"]
    .nunique()
)


conflict_edges = (
    edge_link_count[
        edge_link_count > 1
    ]
    .index
)


conflicts = cost[
    cost[
        "SUMO_EDGE_ID"
    ].isin(
        conflict_edges
    )
].copy()


if len(conflicts) > 0:

    conflicts.to_csv(
        OUTPUT_CONFLICT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 14. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "HOURLY COST SUMMARY"
)

print(
    "=============================="
)


print(
    "TOPIS links:",
    cost[
        "TOPIS_LINK_ID"
    ].nunique()
)


print(
    "Mapped SUMO edges:",
    cost[
        "SUMO_EDGE_ID"
    ].nunique()
)


print(
    "Cost rows:",
    len(cost)
)


print(
    "SUMO edges mapped by >1 TOPIS link:",
    len(conflict_edges)
)


print(
    "\nTravel-time conservation:"
)


print(
    qc[
        [
            "HOUR",
            "N_VALID_LINKS",
            "N_MISSING_LINKS",
            "MAX_ABS_ERROR_SEC"
        ]
    ].to_string(
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


if len(conflicts) > 0:

    print(
        "Created:",
        OUTPUT_CONFLICT.name
    )