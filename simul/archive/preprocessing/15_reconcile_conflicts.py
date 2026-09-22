# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, diags, vstack


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

INPUT_COST = (
    BASE / "sumo_topis_hourly_cost_final.csv"
)

OUTPUT_EDGE_COST = (
    BASE / "sumo_topis_hourly_cost_reconciled.csv"
)

OUTPUT_QC = (
    BASE / "reconciliation_qc.csv"
)

OUTPUT_LINK_ERROR = (
    BASE / "reconciliation_link_error.csv"
)


# ============================================================
# 1. PARAMETER
# ============================================================

# 작을수록 TOPIS 관측식 fitting을 더 중시
# 클수록 기존 edge TT 후보값에 더 가까이 유지
LAMBDA = 1e-6

EPS = 1e-6


# ============================================================
# 2. LOAD
# ============================================================

df = pd.read_csv(
    INPUT_COST,
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
# 3. CONFLICT / UNIQUE EDGE 구분
# ============================================================

edge_n_links = (
    df.groupby(
        "SUMO_EDGE_ID"
    )["TOPIS_LINK_ID"]
    .nunique()
)

conflict_edges = sorted(
    edge_n_links[
        edge_n_links > 1
    ].index
)

unique_edges = sorted(
    edge_n_links[
        edge_n_links == 1
    ].index
)


print(
    "Conflict edges:",
    len(conflict_edges)
)

print(
    "Non-conflict edges:",
    len(unique_edges)
)

print(
    "Total unique edges:",
    df["SUMO_EDGE_ID"].nunique()
)


conflict_set = set(
    conflict_edges
)


# conflict edge index
conflict_index = {
    eid: j
    for j, eid
    in enumerate(conflict_edges)
}


# ============================================================
# 4. EDGE 기본 정보
# ============================================================

edge_info = (
    df.groupby(
        "SUMO_EDGE_ID",
        as_index=False
    )
    .agg(
        SUMO_LENGTH=(
            "SUMO_LENGTH",
            "first"
        )
    )
)


# ============================================================
# 5. AFFECTED TOPIS LINKS
#
# conflict edge를 하나라도 포함하는 TOPIS link만
# reconciliation equation에 들어감.
# ============================================================

affected_links = sorted(
    df.loc[
        df["SUMO_EDGE_ID"].isin(
            conflict_set
        ),
        "TOPIS_LINK_ID"
    ].unique()
)


print(
    "Affected TOPIS links:",
    len(affected_links)
)


link_index = {
    lid: i
    for i, lid
    in enumerate(affected_links)
}


# ============================================================
# 6. B MATRIX
#
# B[l,e] = 1
# if conflict SUMO edge e belongs to TOPIS link l
# ============================================================

rows = []
cols = []
data = []


for lid in affected_links:

    edges = (
        df.loc[
            df["TOPIS_LINK_ID"] == lid,
            "SUMO_EDGE_ID"
        ]
        .drop_duplicates()
    )

    for eid in edges:

        if eid in conflict_set:

            rows.append(
                link_index[lid]
            )

            cols.append(
                conflict_index[eid]
            )

            data.append(
                1.0
            )


B = csr_matrix(
    (
        data,
        (
            rows,
            cols
        )
    ),
    shape=(
        len(affected_links),
        len(conflict_edges)
    )
)


# ============================================================
# 7. OUTPUT edge table 준비
# ============================================================

edge_cost = (
    edge_info
    .copy()
    .set_index(
        "SUMO_EDGE_ID"
    )
)


qc_rows = []
link_error_rows = []


# ============================================================
# 8. HOUR-BY-HOUR RECONCILIATION
# ============================================================

for hour_col in hour_cols:

    print(
        "\nSolving",
        hour_col,
        "..."
    )

    # --------------------------------------------------------
    # 8-1. TOPIS 관측 total TT
    #
    # 현재 row-level cost는 링크별 conservation이 정확하므로
    # 합하면 원래 TOPIS 관측 TT가 됨.
    # --------------------------------------------------------

    topis_total = (
        df.groupby(
            "TOPIS_LINK_ID"
        )[hour_col]
        .sum()
    )


    # --------------------------------------------------------
    # 8-2. non-conflict edges는 기존 TT 그대로
    # --------------------------------------------------------

    nonconflict_df = df[
        ~df["SUMO_EDGE_ID"].isin(
            conflict_set
        )
    ]


    fixed_edge_tt = (
        nonconflict_df
        .groupby(
            "SUMO_EDGE_ID"
        )[hour_col]
        .first()
    )


    # --------------------------------------------------------
    # 8-3. 각 TOPIS link에서
    #      conflict edges가 담당해야 할 residual
    #
    # r_l =
    # T_l - sum(non-conflict edge TT)
    # --------------------------------------------------------

    residual_target = []

    full_topis_tt = []


    for lid in affected_links:

        g = df[
            df["TOPIS_LINK_ID"] == lid
        ]

        y = float(
            topis_total.loc[
                lid
            ]
        )

        fixed_sum = float(
            g.loc[
                ~g[
                    "SUMO_EDGE_ID"
                ].isin(
                    conflict_set
                ),
                hour_col
            ].sum()
        )

        r = (
            y
            -
            fixed_sum
        )

        residual_target.append(
            r
        )

        full_topis_tt.append(
            y
        )


    residual_target = np.asarray(
        residual_target,
        dtype=float
    )

    full_topis_tt = np.asarray(
        full_topis_tt,
        dtype=float
    )


    # --------------------------------------------------------
    # 8-4. conflict edge prior
    #
    # 같은 SUMO edge에 여러 TOPIS-derived TT 후보가 있으므로
    # median을 prior로 사용.
    # 평균보다 outlier에 덜 민감.
    # --------------------------------------------------------

    prior = (
        df[
            df["SUMO_EDGE_ID"].isin(
                conflict_set
            )
        ]
        .groupby(
            "SUMO_EDGE_ID"
        )[hour_col]
        .median()
        .reindex(
            conflict_edges
        )
        .to_numpy(
            dtype=float
        )
    )


    prior = np.maximum(
        prior,
        EPS
    )


    # --------------------------------------------------------
    # 8-5. 상대 TOPIS error를 최소화
    #
    # (B x - r_l) / T_l
    #
    # 를 최소화.
    # --------------------------------------------------------

    link_scale = np.maximum(
        full_topis_tt,
        EPS
    )

    W = diags(
        1.0
        /
        link_scale
    )


    A_data = (
        W @ B
    )

    b_data = (
        residual_target
        /
        link_scale
    )


    # --------------------------------------------------------
    # 8-6. Ridge regularization around prior
    #
    # lambda *
    # ((x_e - prior_e) / prior_e)^2
    # --------------------------------------------------------

    sqrt_lambda = np.sqrt(
        LAMBDA
    )


    A_prior = (
        sqrt_lambda
        *
        diags(
            1.0
            /
            prior
        )
    )


    b_prior = (
        sqrt_lambda
        *
        np.ones(
            len(conflict_edges)
        )
    )


    A_aug = vstack(
        [
            A_data,
            A_prior
        ],
        format="csr"
    )


    b_aug = np.concatenate(
        [
            b_data,
            b_prior
        ]
    )


    # --------------------------------------------------------
    # 8-7. positive least squares
    # --------------------------------------------------------

    result = lsq_linear(
        A_aug,
        b_aug,
        bounds=(
            EPS,
            np.inf
        ),
        lsmr_tol="auto",
        # 작은 lambda에서 수렴할 시간을 더 줌
        max_iter=2000,
        lsmr_maxiter=10000,

        verbose=0
    )


    if not result.success:

        raise RuntimeError(
            f"Optimization failed for {hour_col}: "
            f"{result.message}"
        )


    x = result.x


    # --------------------------------------------------------
    # 8-8. 하나의 TT per SUMO edge 저장
    # --------------------------------------------------------

    # unique/non-conflict edges
    for eid, value in (
        fixed_edge_tt.items()
    ):

        edge_cost.loc[
            eid,
            hour_col
        ] = float(value)


    # conflict edges
    for eid, value in zip(
        conflict_edges,
        x
    ):

        edge_cost.loc[
            eid,
            hour_col
        ] = float(value)


    # ========================================================
    # 9. RECONSTRUCTION QC
    # ========================================================

    # 모든 182개 TOPIS 링크에 대해
    # reconciled edge TT를 다시 합산
    # --------------------------------------------------------

    abs_errors = []
    rel_errors = []


    for lid, g in df.groupby(
        "TOPIS_LINK_ID"
    ):

        mapped_edges = (
            g[
                "SUMO_EDGE_ID"
            ]
            .drop_duplicates()
            .tolist()
        )


        reconstructed = sum(
            float(
                edge_cost.loc[
                    eid,
                    hour_col
                ]
            )
            for eid in mapped_edges
        )


        observed = float(
            topis_total.loc[
                lid
            ]
        )


        abs_error = abs(
            reconstructed
            -
            observed
        )


        rel_error = (
            abs_error
            /
            observed
            if observed > 0
            else np.nan
        )


        abs_errors.append(
            abs_error
        )

        rel_errors.append(
            rel_error
        )


        link_error_rows.append({
            "HOUR":
                hour_col,

            "TOPIS_LINK_ID":
                lid,

            "OBSERVED_TT":
                observed,

            "RECONSTRUCTED_TT":
                reconstructed,

            "ABS_ERROR_SEC":
                abs_error,

            "REL_ERROR":
                rel_error
        })


    abs_errors = np.asarray(
        abs_errors,
        dtype=float
    )

    rel_errors = np.asarray(
        rel_errors,
        dtype=float
    )


    qc_rows.append({

        "HOUR":
            hour_col,

        "MEAN_ABS_ERROR_SEC":
            float(
                np.mean(
                    abs_errors
                )
            ),

        "MAX_ABS_ERROR_SEC":
            float(
                np.max(
                    abs_errors
                )
            ),

        "MEAN_REL_ERROR":
            float(
                np.nanmean(
                    rel_errors
                )
            ),

        "P95_REL_ERROR":
            float(
                np.nanpercentile(
                    rel_errors,
                    95
                )
            ),

        "MAX_REL_ERROR":
            float(
                np.nanmax(
                    rel_errors
                )
            )
    })


# ============================================================
# 10. SPEED도 같이 계산
#
# 나중에 sanity check용.
# travel-time cost 자체는 TT를 사용.
# ============================================================

for h in range(
    1,
    25
):

    tt_col = (
        f"TT_~{h:02d}시"
    )

    speed_col = (
        f"SPEED_~{h:02d}시"
    )

    edge_cost[
        speed_col
    ] = (
        edge_cost[
            "SUMO_LENGTH"
        ]
        /
        edge_cost[
            tt_col
        ]
        *
        3.6
    )


# ============================================================
# 11. SAVE
# ============================================================

edge_cost = (
    edge_cost
    .reset_index()
)


edge_cost.to_csv(
    OUTPUT_EDGE_COST,
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


link_error = pd.DataFrame(
    link_error_rows
)

link_error.to_csv(
    OUTPUT_LINK_ERROR,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "GLOBAL RECONCILIATION SUMMARY"
)

print(
    "=============================="
)

print(
    "Unique SUMO edges:",
    len(edge_cost)
)

print(
    "Adjusted conflict edges:",
    len(conflict_edges)
)

print(
    "Unchanged non-conflict edges:",
    len(unique_edges)
)


print(
    "\nTOPIS reconstruction error:"
)


print(
    qc.to_string(
        index=False
    )
)


print(
    "\nCreated:",
    OUTPUT_EDGE_COST.name
)

print(
    "Created:",
    OUTPUT_QC.name
)

print(
    "Created:",
    OUTPUT_LINK_ERROR.name
)