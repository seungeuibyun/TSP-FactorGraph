# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

COVERAGE_FILE = (
    BASE / "topis_sumo_edge_coverage.csv"
)

SPEED_FILE = (
    BASE / "topis_speed.xlsx"
)

OUTPUT_SEGMENTS = (
    BASE / "sumo_topis_piecewise_segments.csv"
)

OUTPUT_EDGE_COST = (
    BASE / "sumo_topis_observed_edge_cost.csv"
)

OUTPUT_QC = (
    BASE / "sumo_topis_observed_coverage_qc.csv"
)


EPS = 1e-6


# ============================================================
# 1. LOAD COVERAGE
# ============================================================

cov = pd.read_csv(
    COVERAGE_FILE,
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

cov["TOPIS_LINK_ID"] = (
    cov["TOPIS_LINK_ID"]
    .str.strip()
)

cov["SUMO_EDGE_ID"] = (
    cov["SUMO_EDGE_ID"]
    .str.strip()
)


numeric_cols = [
    "SUMO_LENGTH",
    "EDGE_S_START",
    "EDGE_S_END",
    "COVERED_LENGTH",
    "MEAN_LATERAL_DIST"
]

for col in numeric_cols:
    cov[col] = pd.to_numeric(
        cov[col],
        errors="coerce"
    )


print(
    "TOPIS links:",
    cov["TOPIS_LINK_ID"].nunique()
)

print(
    "Mapped SUMO edges:",
    cov["SUMO_EDGE_ID"].nunique()
)


# ============================================================
# 2. TOPIS SPEED
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

speed = speed.rename(
    columns={
        "링크아이디":
            "TOPIS_LINK_ID"
    }
)


hour_cols = [
    f"~{h:02d}시"
    for h in range(1, 25)
]


for col in hour_cols:

    speed[col] = pd.to_numeric(
        speed[col],
        errors="coerce"
    )


speed = (
    speed
    .drop_duplicates(
        "TOPIS_LINK_ID"
    )
    .set_index(
        "TOPIS_LINK_ID"
    )
)


# ============================================================
# 3. CHECK SPEED
# ============================================================

needed_links = set(
    cov["TOPIS_LINK_ID"]
)

missing_speed_links = (
    needed_links
    -
    set(speed.index)
)


if missing_speed_links:

    raise RuntimeError(
        "Missing speed data: "
        + str(
            sorted(
                missing_speed_links
            )
        )
    )


# ============================================================
# 4. POSITIVE COVERAGE INTERVALS
# ============================================================

valid = cov[
    cov["EDGE_S_START"].notna()
    &
    cov["EDGE_S_END"].notna()
    &
    (
        cov["COVERED_LENGTH"]
        > EPS
    )
].copy()


# ============================================================
# 5. SUMO EDGE를 interval boundary로 분할
#
# 예:
#
# 0 -------- A -------- B -------- edge end
#
# TOPIS 1:   [-----]
# TOPIS 2:          [----------]
#
# 모든 start/end를 breakpoint로 사용.
# ============================================================

segment_rows = []


edge_info = (
    cov.groupby(
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


for idx, edge_row in edge_info.iterrows():

    eid = edge_row[
        "SUMO_EDGE_ID"
    ]

    L = float(
        edge_row[
            "SUMO_LENGTH"
        ]
    )


    g = valid[
        valid[
            "SUMO_EDGE_ID"
        ] == eid
    ].copy()


    # ----------------------------------------
    # 이 edge에 coverage가 아예 없는 경우
    # ----------------------------------------

    if len(g) == 0:

        segment_rows.append({
            "SUMO_EDGE_ID":
                eid,

            "SUMO_LENGTH":
                L,

            "S_START":
                0.0,

            "S_END":
                L,

            "SEGMENT_LENGTH":
                L,

            "OBSERVED":
                0,

            "N_TOPIS_LINKS":
                0,

            "TOPIS_LINKS":
                ""
        })

        continue


    # ----------------------------------------
    # breakpoint
    # ----------------------------------------

    breakpoints = [
        0.0,
        L
    ]


    for _, r in g.iterrows():

        a = max(
            0.0,
            min(
                L,
                float(
                    r[
                        "EDGE_S_START"
                    ]
                )
            )
        )

        b = max(
            0.0,
            min(
                L,
                float(
                    r[
                        "EDGE_S_END"
                    ]
                )
            )
        )


        if b < a:
            a, b = b, a


        breakpoints.extend(
            [
                a,
                b
            ]
        )


    # floating-point duplicate 제거
    breakpoints = sorted(
        set(
            round(
                x,
                6
            )
            for x in breakpoints
        )
    )


    # ----------------------------------------
    # elementary segments
    # ----------------------------------------

    for a, b in zip(
        breakpoints[:-1],
        breakpoints[1:]
    ):

        if (
            b - a
            <= EPS
        ):
            continue


        midpoint = (
            a + b
        ) / 2


        active = g[
            (
                g[
                    "EDGE_S_START"
                ]
                <= midpoint
                + EPS
            )
            &
            (
                g[
                    "EDGE_S_END"
                ]
                >= midpoint
                - EPS
            )
        ]


        links = sorted(
            active[
                "TOPIS_LINK_ID"
            ].unique()
        )


        observed = int(
            len(links) > 0
        )


        segment_rows.append({

            "SUMO_EDGE_ID":
                eid,

            "SUMO_LENGTH":
                L,

            "S_START":
                a,

            "S_END":
                b,

            "SEGMENT_LENGTH":
                b - a,

            "OBSERVED":
                observed,

            "N_TOPIS_LINKS":
                len(links),

            "TOPIS_LINKS":
                ",".join(
                    links
                )
        })


segments = pd.DataFrame(
    segment_rows
)


# ============================================================
# 6. HOURLY PACE
#
# pace = seconds / meter
#
#       = 3.6 / speed[km/h]
#
# 동일 physical subsegment를 여러 TOPIS가 덮으면
# median pace 사용.
# ============================================================

for hour in hour_cols:

    pace_col = (
        f"PACE_{hour}"
    )

    tt_col = (
        f"TT_{hour}"
    )


    pace_values = []


    for _, row in segments.iterrows():

        if row[
            "OBSERVED"
        ] == 0:

            pace_values.append(
                np.nan
            )

            continue


        links = [
            x
            for x in str(
                row[
                    "TOPIS_LINKS"
                ]
            ).split(",")
            if x
        ]


        values = []


        for lid in links:

            v = speed.loc[
                lid,
                hour
            ]


            if (
                pd.notna(v)
                and v > 0
            ):

                values.append(
                    3.6
                    /
                    float(v)
                )


        if len(values) == 0:

            pace_values.append(
                np.nan
            )

        else:

            # robust한 대표 pace
            pace_values.append(
                float(
                    np.median(
                        values
                    )
                )
            )


    segments[
        pace_col
    ] = pace_values


    segments[
        tt_col
    ] = (
        segments[
            "SEGMENT_LENGTH"
        ]
        *
        segments[
            pace_col
        ]
    )


# ============================================================
# 7. EDGE-LEVEL OBSERVED COVERAGE
# ============================================================

edge_rows = []


for eid, g in segments.groupby(
    "SUMO_EDGE_ID"
):

    L = float(
        g[
            "SUMO_LENGTH"
        ].iloc[0]
    )


    observed_g = g[
        g[
            "OBSERVED"
        ] == 1
    ]


    observed_length = float(
        observed_g[
            "SEGMENT_LENGTH"
        ].sum()
    )


    overlap_length = float(
        g.loc[
            g[
                "N_TOPIS_LINKS"
            ] > 1,
            "SEGMENT_LENGTH"
        ].sum()
    )


    row = {

        "SUMO_EDGE_ID":
            eid,

        "SUMO_LENGTH":
            L,

        "OBSERVED_LENGTH":
            observed_length,

        "COVERAGE_RATIO":
            (
                observed_length
                /
                L
                if L > 0
                else np.nan
            ),

        "UNOBSERVED_LENGTH":
            max(
                0.0,
                L
                -
                observed_length
            ),

        "OVERLAP_LENGTH":
            overlap_length,

        "OVERLAP_RATIO":
            (
                overlap_length
                /
                L
                if L > 0
                else np.nan
            )
    }


    # ----------------------------------------
    # hourly observed TT / mean observed pace
    # ----------------------------------------

    for hour in hour_cols:

        tt_col = (
            f"TT_{hour}"
        )

        pace_out = (
            f"OBS_PACE_{hour}"
        )

        tt_out = (
            f"OBS_TT_{hour}"
        )


        obs_tt = (
            observed_g[
                tt_col
            ].sum(
                min_count=1
            )
        )


        row[
            tt_out
        ] = obs_tt


        if (
            observed_length > 0
            and pd.notna(
                obs_tt
            )
        ):

            row[
                pace_out
            ] = (
                float(
                    obs_tt
                )
                /
                observed_length
            )

        else:

            row[
                pace_out
            ] = np.nan


    edge_rows.append(
        row
    )


edge_cost = pd.DataFrame(
    edge_rows
)


# ============================================================
# 8. QC
# ============================================================

qc = edge_cost[
    [
        "SUMO_EDGE_ID",
        "SUMO_LENGTH",
        "OBSERVED_LENGTH",
        "UNOBSERVED_LENGTH",
        "COVERAGE_RATIO",
        "OVERLAP_LENGTH",
        "OVERLAP_RATIO"
    ]
].copy()


def coverage_class(x):

    if x >= 0.95:
        return ">=95%"

    if x >= 0.50:
        return "50-95%"

    if x > 0:
        return "<50%"

    return "0%"


qc[
    "COVERAGE_CLASS"
] = (
    qc[
        "COVERAGE_RATIO"
    ].apply(
        coverage_class
    )
)


# ============================================================
# 9. SAVE
# ============================================================

segments.to_csv(
    OUTPUT_SEGMENTS,
    index=False,
    encoding="utf-8-sig"
)


edge_cost.to_csv(
    OUTPUT_EDGE_COST,
    index=False,
    encoding="utf-8-sig"
)


qc.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "PIECEWISE OBSERVED COST"
)

print(
    "=============================="
)


print(
    "Mapped SUMO edges:",
    len(
        edge_cost
    )
)


print(
    "Piecewise segments:",
    len(
        segments
    )
)


print(
    "\nCoverage classes:"
)


print(
    qc[
        "COVERAGE_CLASS"
    ]
    .value_counts()
)


print(
    "\nEdge coverage ratio:"
)


print(
    qc[
        "COVERAGE_RATIO"
    ].describe()
)


print(
    "\nEdges with overlapping TOPIS observations:",
    int(
        (
            qc[
                "OVERLAP_LENGTH"
            ] > 0
        ).sum()
    )
)


print(
    "Total overlapping length:",
    round(
        qc[
            "OVERLAP_LENGTH"
        ].sum(),
        1
    ),
    "m"
)


print(
    "\nCreated:",
    OUTPUT_SEGMENTS.name
)

print(
    "Created:",
    OUTPUT_EDGE_COST.name
)

print(
    "Created:",
    OUTPUT_QC.name
)