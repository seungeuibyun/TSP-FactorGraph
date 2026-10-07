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
    BASE / "sumo_topis_hourly_cost_reconciled_v2.csv"
)

OUTPUT_QC = (
    BASE / "reconciliation_v2_qc.csv"
)

OUTPUT_LINK_ERROR = (
    BASE / "reconciliation_v2_link_error.csv"
)


# ============================================================
# 1. PARAMETERS
# ============================================================

# data fitting이 최우선.
# prior는 edge TT가 터지는 것만 약하게 방지.
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
# 3. CONFLICT EDGES
# ============================================================

edge_n_links = (
    df.groupby(
        "SUMO_EDGE_ID"
    )["TOPIS_LINK_ID"]
    .nunique()
)


conflict_edges = set(
    edge_n_links[
        edge_n_links > 1
    ].index
)


print(
    "Conflict edges:",
    len(conflict_edges)
)


# ============================================================
# 4. AFFECTED LINKS
#
# conflict edge를 하나라도 포함하는 TOPIS link
# ============================================================

affected_links = sorted(
    df.loc[
        df["SUMO_EDGE_ID"].isin(
            conflict_edges
        ),
        "TOPIS_LINK_ID"
    ].unique()
)


affected_link_set = set(
    affected_links
)


print(
    "Affected TOPIS links:",
    len(affected_links)
)


# ============================================================
# 5. AFFECTED EDGES
#
# affected link에 포함된 모든 edge를 조정 가능하게 함.
# ============================================================

affected_edges = sorted(
    df.loc[
        df["TOPIS_LINK_ID"].isin(
            affected_link_set
        ),
        "SUMO_EDGE_ID"
    ].unique()
)


affected_edge_set = set(
    affected_edges
)


unaffected_edges = sorted(
    set(
        df["SUMO_EDGE_ID"].unique()
    )
    -
    affected_edge_set
)


print(
    "Affected SUMO edges:",
    len(affected_edges)
)

print(
    "Unchanged SUMO edges:",
    len(unaffected_edges)
)


# ============================================================
# 6. INDEX
# ============================================================

link_index = {
    lid: i
    for i, lid
    in enumerate(
        affected_links
    )
}


edge_index = {
    eid: j
    for j, eid
    in enumerate(
        affected_edges
    )
}


# ============================================================
# 7. INCIDENCE MATRIX A
#
# A[l,e] = 해당 TOPIS route에서 edge e가
#          등장하는 횟수
#
# 보통 1이지만 혹시 반복 edge가 있어도 처리.
# ============================================================

rows = []
cols = []
data = []


for lid in affected_links:

    g = df[
        df["TOPIS_LINK_ID"]
        ==
        lid
    ]

    counts = (
        g["SUMO_EDGE_ID"]
        .value_counts()
    )

    for eid, count in counts.items():

        if eid not in affected_edge_set:
            continue

        rows.append(
            link_index[lid]
        )

        cols.append(
            edge_index[eid]
        )

        data.append(
            float(count)
        )


A = csr_matrix(
    (
        data,
        (
            rows,
            cols
        )
    ),
    shape=(
        len(affected_links),
        len(affected_edges)
    )
)


# ============================================================
# 8. EDGE INFO
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


edge_cost = (
    edge_info
    .set_index(
        "SUMO_EDGE_ID"
    )
    .copy()
)


qc_rows = []
link_error_rows = []


# ============================================================
# 9. HOUR-BY-HOUR
# ============================================================

for hour_col in hour_cols:

    print(
        "\nSolving",
        hour_col,
        "..."
    )


    # --------------------------------------------------------
    # 9-1. 원래 TOPIS total travel time
    #
    # 13번에서 conservation이 정확했으므로
    # row sum = 관측 TOPIS TT
    # --------------------------------------------------------

    topis_total = (
        df.groupby(
            "TOPIS_LINK_ID"
        )[hour_col]
        .sum(
            min_count=1
        )
    )


    # --------------------------------------------------------
    # 9-2. PRIOR per SUMO edge
    #
    # non-conflict:
    #   원래 값 그대로
    #
    # conflict:
    #   여러 TOPIS-derived 후보의 median
    # --------------------------------------------------------

    prior_series = (
        df[
            df["SUMO_EDGE_ID"].isin(
                affected_edge_set
            )
        ]
        .groupby(
            "SUMO_EDGE_ID"
        )[hour_col]
        .median()
        .reindex(
            affected_edges
        )
    )


    prior = (
        prior_series
        .to_numpy(
            dtype=float
        )
    )


    if np.isnan(prior).any():

        raise RuntimeError(
            f"NaN prior found in {hour_col}"
        )


    prior = np.maximum(
        prior,
        EPS
    )


    # --------------------------------------------------------
    # 9-3. affected link target
    # --------------------------------------------------------

    y = np.array(
        [
            float(
                topis_total.loc[lid]
            )
            for lid in affected_links
        ],
        dtype=float
    )


    y = np.maximum(
        y,
        EPS
    )


    # ========================================================
    # 9-4. SCALE VARIABLE
    #
    # x_e = prior_e * z_e
    #
    # 우리가 푸는 변수는 z.
    #
    # A diag(prior) z = y
    #
    # 양변을 y로 나누면
    #
    # [A diag(prior) / y] z = 1
    #
    # ========================================================

    A_scaled = (
        A
        @
        diags(
            prior
        )
    )


    W = diags(
        1.0 / y
    )


    A_data = (
        W
        @
        A_scaled
    )


    b_data = np.ones(
        len(
            affected_links
        )
    )


    # --------------------------------------------------------
    # 9-5. PRIOR REGULARIZATION
    #
    # z ≈ 1
    # --------------------------------------------------------

    sqrt_lambda = np.sqrt(
        LAMBDA
    )


    A_prior = (
        sqrt_lambda
        *
        diags(
            np.ones(
                len(
                    affected_edges
                )
            )
        )
    )


    b_prior = (
        sqrt_lambda
        *
        np.ones(
            len(
                affected_edges
            )
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
    # 9-6. POSITIVE LEAST SQUARES
    #
    # z >= small positive
    # --------------------------------------------------------

    result = lsq_linear(
        A_aug,
        b_aug,
        bounds=(
            1e-4,
            np.inf
        ),
        lsmr_tol="auto",
        max_iter=5000,
        lsmr_maxiter=20000,
        verbose=0
    )


    if not result.success:

        raise RuntimeError(
            f"Optimization failed for {hour_col}: "
            f"{result.message}"
        )


    z = result.x


    x = (
        prior
        *
        z
    )


    # ========================================================
    # 10. STORE AFFECTED EDGE TT
    # ========================================================

    for eid, value in zip(
        affected_edges,
        x
    ):

        edge_cost.loc[
            eid,
            hour_col
        ] = float(value)


    # ========================================================
    # 11. UNAFFECTED EDGES
    #
    # 기존 값을 그대로 유지
    # ========================================================

    unaffected_df = df[
        df["SUMO_EDGE_ID"].isin(
            unaffected_edges
        )
    ]


    unaffected_tt = (
        unaffected_df
        .groupby(
            "SUMO_EDGE_ID"
        )[hour_col]
        .first()
    )


    for eid, value in (
        unaffected_tt.items()
    ):

        edge_cost.loc[
            eid,
            hour_col
        ] = float(value)


    # ========================================================
    # 12. QC — 182 TOPIS LINKS 모두 재구성
    # ========================================================

    abs_errors = []
    rel_errors = []


    for lid, g in df.groupby(
        "TOPIS_LINK_ID"
    ):

        # edge가 중복될 경우 그 횟수까지 포함
        mapped_edges = (
            g["SUMO_EDGE_ID"]
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

            "AFFECTED":
                int(
                    lid
                    in
                    affected_link_set
                ),

            "OBSERVED_TT":
                observed,

            "RECONSTRUCTED_TT":
                reconstructed,

            "ABS_ERROR_SEC":
                abs_error,

            "REL_ERROR":
                rel_error
        })


    abs_errors = np.array(
        abs_errors
    )

    rel_errors = np.array(
        rel_errors
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
# 13. SPEED
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
# 14. SAVE
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
# 15. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "AFFECTED-EDGE RECONCILIATION"
)

print(
    "=============================="
)


print(
    "Total SUMO edges:",
    len(edge_cost)
)

print(
    "Conflict edges:",
    len(conflict_edges)
)

print(
    "Affected links:",
    len(affected_links)
)

print(
    "Adjusted affected edges:",
    len(affected_edges)
)

print(
    "Unchanged edges:",
    len(unaffected_edges)
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