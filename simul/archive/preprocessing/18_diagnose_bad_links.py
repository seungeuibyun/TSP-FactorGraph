# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent

INPUT = (
    BASE / "sumo_topis_hourly_cost_final.csv"
)

TARGETS = [
    "1070012300",
    "1070012400",
    "1005003800",
]

OUTPUT = (
    BASE / "bad_link_overlap_diagnosis.csv"
)


# ============================================================
# 1. LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
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


# ============================================================
# 2. 각 TOPIS link의 edge set / length
# ============================================================

link_edges = {}

link_sumo_length = {}

link_topis_length = {}


for lid, g in df.groupby(
    "TOPIS_LINK_ID"
):

    # edge 중복 없다고 가정하지 않고
    # unique 기준으로 geometry overlap 계산
    edges = (
        g[
            [
                "SUMO_EDGE_ID",
                "SUMO_LENGTH"
            ]
        ]
        .drop_duplicates(
            "SUMO_EDGE_ID"
        )
    )

    link_edges[lid] = set(
        edges["SUMO_EDGE_ID"]
    )

    link_sumo_length[lid] = float(
        edges["SUMO_LENGTH"].sum()
    )

    link_topis_length[lid] = float(
        g["TOPIS_LENGTH"].iloc[0]
    )


# edge length dictionary
edge_length = (
    df[
        [
            "SUMO_EDGE_ID",
            "SUMO_LENGTH"
        ]
    ]
    .drop_duplicates(
        "SUMO_EDGE_ID"
    )
    .set_index(
        "SUMO_EDGE_ID"
    )[
        "SUMO_LENGTH"
    ]
    .to_dict()
)


# ============================================================
# 3. TARGET별 overlap
# ============================================================

rows = []


for target in TARGETS:

    print(
        "\n========================================"
    )

    print(
        "TARGET:",
        target
    )

    print(
        "========================================"
    )

    target_edges = link_edges[
        target
    ]

    target_sumo_len = (
        link_sumo_length[
            target
        ]
    )

    target_topis_len = (
        link_topis_length[
            target
        ]
    )


    print(
        "TOPIS length:",
        round(
            target_topis_len,
            1
        )
    )

    print(
        "Mapped SUMO length:",
        round(
            target_sumo_len,
            1
        )
    )

    print(
        "Length ratio:",
        round(
            target_sumo_len
            /
            target_topis_len,
            3
        )
    )

    print(
        "N edges:",
        len(
            target_edges
        )
    )


    overlaps = []


    for other, other_edges in (
        link_edges.items()
    ):

        if other == target:
            continue

        common = (
            target_edges
            &
            other_edges
        )

        if not common:
            continue


        overlap_len = sum(
            edge_length[eid]
            for eid in common
        )


        target_frac = (
            overlap_len
            /
            target_sumo_len
        )


        other_frac = (
            overlap_len
            /
            link_sumo_length[
                other
            ]
        )


        overlaps.append({

            "TARGET_LINK":
                target,

            "OTHER_LINK":
                other,

            "N_COMMON_EDGES":
                len(common),

            "OVERLAP_LENGTH":
                overlap_len,

            "TARGET_OVERLAP_RATIO":
                target_frac,

            "OTHER_OVERLAP_RATIO":
                other_frac,

            "TARGET_N_EDGES":
                len(target_edges),

            "OTHER_N_EDGES":
                len(other_edges),

            "COMMON_EDGES":
                ",".join(
                    sorted(common)
                )
        })


    overlaps = sorted(
        overlaps,
        key=lambda x:
            x[
                "TARGET_OVERLAP_RATIO"
            ],
        reverse=True
    )


    print(
        "\nTop overlapping TOPIS links:"
    )


    for x in overlaps[:10]:

        print(
            x["OTHER_LINK"],
            "| common edges =",
            x["N_COMMON_EDGES"],
            "| target overlap =",
            round(
                x[
                    "TARGET_OVERLAP_RATIO"
                ],
                3
            ),
            "| other overlap =",
            round(
                x[
                    "OTHER_OVERLAP_RATIO"
                ],
                3
            )
        )


    rows.extend(
        overlaps
    )


# ============================================================
# 4. SAVE
# ============================================================

out = pd.DataFrame(
    rows
)

out.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nCreated:",
    OUTPUT.name
)