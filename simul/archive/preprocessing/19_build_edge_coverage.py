# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import Transformer
from shapely.geometry import LineString


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

VERTEX_FILE = (
    BASE /
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"
)

SPEED_FILE = (
    BASE / "topis_speed.xlsx"
)

OUTPUT_COVERAGE = (
    BASE / "topis_sumo_edge_coverage.csv"
)

OUTPUT_QC = (
    BASE / "topis_sumo_coverage_qc.csv"
)


# ============================================================
# 1. PARAMETERS
# ============================================================

# TOPIS line sampling 간격
SAMPLE_SPACING = 2.0  # meters

# divided road / centerline offset 허용
MAX_LATERAL_DIST = 80.0  # meters


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

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 3. FINAL MAPPING
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
    mapping[
        "TOPIS_LINK_ID"
    ].nunique()
)

print(
    "Mapping rows:",
    len(mapping)
)


# ============================================================
# 4. TOPIS VERTICES
# ============================================================

vtx = pd.read_excel(
    VERTEX_FILE,
    dtype={
        "LINK_ID": str
    }
)

vtx["LINK_ID"] = (
    vtx["LINK_ID"]
    .str.strip()
)


# final 182개만
valid_links = set(
    mapping[
        "TOPIS_LINK_ID"
    ].unique()
)

vtx = vtx[
    vtx["LINK_ID"].isin(
        valid_links
    )
].copy()


# ============================================================
# 5. EPSG:5181 -> SUMO XY
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


sumo_x = []
sumo_y = []


for x, y in zip(
    vtx["GRS80TM_X"],
    vtx["GRS80TM_Y"]
):

    lon, lat = (
        transformer.transform(
            x,
            y
        )
    )

    sx, sy = (
        net.convertLonLat2XY(
            lon,
            lat
        )
    )

    sumo_x.append(sx)
    sumo_y.append(sy)


vtx["SUMO_X"] = sumo_x
vtx["SUMO_Y"] = sumo_y


# ============================================================
# 6. TOPIS reported length
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

speed["거리"] = pd.to_numeric(
    speed["거리"],
    errors="coerce"
)


reported_length = (
    speed
    .drop_duplicates(
        "링크아이디"
    )
    .set_index(
        "링크아이디"
    )[
        "거리"
    ]
    .to_dict()
)


# ============================================================
# 7. HELPERS
# ============================================================

def sample_line(
    line,
    spacing=2.0
):

    if line.length <= 0:
        return [
            line.interpolate(0)
        ]

    distances = list(
        np.arange(
            0,
            line.length,
            spacing
        )
    )

    if (
        len(distances) == 0
        or
        distances[-1] < line.length
    ):

        distances.append(
            line.length
        )

    return [
        (
            float(d),
            line.interpolate(d)
        )
        for d in distances
    ]


# SUMO edge geometry cache
edge_lines = {}

edge_lengths = {}


for eid in (
    mapping[
        "SUMO_EDGE_ID"
    ].unique()
):

    edge = net.getEdge(
        eid
    )

    line = LineString(
        edge.getShape()
    )

    edge_lines[eid] = line
    edge_lengths[eid] = float(
        edge.getLength()
    )


# ============================================================
# 8. COVERAGE CALCULATION
#
# 각 TOPIS sample point를
# 그 TOPIS link에 mapping된 SUMO edges 중
# 가장 가까운 edge 하나에 할당한다.
#
# 그러면 junction 부근에서 같은 TOPIS 구간을
# 여러 SUMO edge가 중복해서 먹는 문제를 줄일 수 있음.
# ============================================================

coverage_rows = []
qc_rows = []


link_ids = sorted(
    mapping[
        "TOPIS_LINK_ID"
    ].unique()
)


for idx, lid in enumerate(
    link_ids,
    start=1
):

    print(
        f"[{idx}/{len(link_ids)}]",
        lid
    )


    # --------------------------------------------------------
    # TOPIS geometry
    # --------------------------------------------------------

    vg = (
        vtx[
            vtx["LINK_ID"] == lid
        ]
        .sort_values(
            "VER_SEQ"
        )
    )


    coords = list(
        zip(
            vg["SUMO_X"],
            vg["SUMO_Y"]
        )
    )


    if len(coords) < 2:

        print(
            "  WARNING: insufficient TOPIS vertices"
        )

        continue


    topis_line = LineString(
        coords
    )


    samples = sample_line(
        topis_line,
        spacing=SAMPLE_SPACING
    )


    # --------------------------------------------------------
    # mapped SUMO edges
    # --------------------------------------------------------

    mg = (
        mapping[
            mapping[
                "TOPIS_LINK_ID"
            ] == lid
        ]
        .sort_values(
            "SEQ"
        )
        .copy()
    )


    edge_ids = (
        mg["SUMO_EDGE_ID"]
        .tolist()
    )


    # edge별 assigned samples
    assigned = {
        eid: []
        for eid in edge_ids
    }


    # --------------------------------------------------------
    # 각 TOPIS sample -> nearest mapped SUMO edge
    # --------------------------------------------------------

    n_unassigned = 0


    for topis_s, p in samples:

        candidates = []

        for eid in edge_ids:

            e_line = (
                edge_lines[eid]
            )

            lateral = (
                p.distance(
                    e_line
                )
            )

            candidates.append(
                (
                    lateral,
                    eid
                )
            )


        candidates.sort(
            key=lambda x: x[0]
        )


        best_dist, best_eid = (
            candidates[0]
        )


        if (
            best_dist
            >
            MAX_LATERAL_DIST
        ):

            n_unassigned += 1
            continue


        e_line = (
            edge_lines[
                best_eid
            ]
        )


        edge_s = float(
            e_line.project(p)
        )


        assigned[
            best_eid
        ].append(
            {
                "topis_s":
                    topis_s,

                "edge_s":
                    edge_s,

                "lateral":
                    float(
                        best_dist
                    )
            }
        )


    # --------------------------------------------------------
    # edge별 covered interval
    # --------------------------------------------------------

    total_covered = 0.0
    n_edges_with_support = 0


    for _, row in mg.iterrows():

        eid = (
            row[
                "SUMO_EDGE_ID"
            ]
        )

        seq = row[
            "SEQ"
        ]

        edge_length = (
            edge_lengths[
                eid
            ]
        )

        values = (
            assigned[
                eid
            ]
        )


        if len(values) == 0:

            edge_s_start = np.nan
            edge_s_end = np.nan
            covered_length = 0.0

            mean_lat = np.nan
            max_lat = np.nan

            topis_s_start = np.nan
            topis_s_end = np.nan

        else:

            edge_positions = [
                x["edge_s"]
                for x in values
            ]

            topis_positions = [
                x["topis_s"]
                for x in values
            ]

            lateral_values = [
                x["lateral"]
                for x in values
            ]


            edge_s_start = max(
                0.0,
                min(
                    edge_positions
                )
                -
                SAMPLE_SPACING / 2
            )


            edge_s_end = min(
                edge_length,
                max(
                    edge_positions
                )
                +
                SAMPLE_SPACING / 2
            )


            covered_length = max(
                0.0,
                edge_s_end
                -
                edge_s_start
            )


            topis_s_start = min(
                topis_positions
            )

            topis_s_end = max(
                topis_positions
            )


            mean_lat = float(
                np.mean(
                    lateral_values
                )
            )

            max_lat = float(
                np.max(
                    lateral_values
                )
            )


            if covered_length > 0:

                n_edges_with_support += 1


        total_covered += (
            covered_length
        )


        coverage_rows.append({

            "TOPIS_LINK_ID":
                lid,

            "SEQ":
                seq,

            "SUMO_EDGE_ID":
                eid,

            "SUMO_LENGTH":
                edge_length,

            # SUMO edge 내부의 실제 대응 구간
            "EDGE_S_START":
                edge_s_start,

            "EDGE_S_END":
                edge_s_end,

            "COVERED_LENGTH":
                covered_length,

            "EDGE_COVERAGE_RATIO":
                (
                    covered_length
                    /
                    edge_length
                    if edge_length > 0
                    else np.nan
                ),

            # 어떤 TOPIS progress가 이 edge로 배정됐는지
            "TOPIS_S_START":
                topis_s_start,

            "TOPIS_S_END":
                topis_s_end,

            "N_ASSIGNED_SAMPLES":
                len(values),

            "MEAN_LATERAL_DIST":
                mean_lat,

            "MAX_LATERAL_DIST":
                max_lat,
        })


    # --------------------------------------------------------
    # LINK QC
    # --------------------------------------------------------

    geom_length = float(
        topis_line.length
    )

    official_length = (
        reported_length.get(
            lid,
            np.nan
        )
    )


    if (
        pd.notna(
            official_length
        )
        and official_length > 0
    ):

        covered_vs_topis = (
            total_covered
            /
            official_length
        )

    else:

        covered_vs_topis = (
            np.nan
        )


    qc_rows.append({

        "TOPIS_LINK_ID":
            lid,

        "N_MAPPED_EDGES":
            len(edge_ids),

        "N_EDGES_WITH_SUPPORT":
            n_edges_with_support,

        "N_TOPIS_SAMPLES":
            len(samples),

        "N_UNASSIGNED_SAMPLES":
            n_unassigned,

        "UNASSIGNED_RATIO":
            (
                n_unassigned
                /
                len(samples)
                if len(samples) > 0
                else np.nan
            ),

        "TOPIS_GEOM_LENGTH":
            geom_length,

        "TOPIS_REPORTED_LENGTH":
            official_length,

        "TOTAL_COVERED_SUMO_LENGTH":
            total_covered,

        "COVERED_TO_TOPIS_RATIO":
            covered_vs_topis,
    })


# ============================================================
# 9. SAVE
# ============================================================

coverage = pd.DataFrame(
    coverage_rows
)


coverage.to_csv(
    OUTPUT_COVERAGE,
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
# 10. SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "EDGE COVERAGE SUMMARY"
)

print(
    "=============================="
)


print(
    "TOPIS links:",
    qc[
        "TOPIS_LINK_ID"
    ].nunique()
)


print(
    "Coverage rows:",
    len(
        coverage
    )
)


print(
    "Links with unassigned samples:",
    int(
        (
            qc[
                "N_UNASSIGNED_SAMPLES"
            ] > 0
        ).sum()
    )
)


print(
    "\nCOVERED / TOPIS ratio:"
)


print(
    qc[
        "COVERED_TO_TOPIS_RATIO"
    ].describe()
)


# ============================================================
# 11. 문제였던 3개 직접 확인
# ============================================================

targets = [
    "1070012300",
    "1070012400",
    "1005003800",
]


print(
    "\n=============================="
)

print(
    "TARGET DIAGNOSTICS"
)

print(
    "=============================="
)


for lid in targets:

    print(
        "\n---",
        lid,
        "---"
    )


    x = coverage[
        coverage[
            "TOPIS_LINK_ID"
        ] == lid
    ]


    print(
        x[
            [
                "SUMO_EDGE_ID",
                "SUMO_LENGTH",
                "EDGE_S_START",
                "EDGE_S_END",
                "COVERED_LENGTH",
                "EDGE_COVERAGE_RATIO",
                "N_ASSIGNED_SAMPLES",
                "MEAN_LATERAL_DIST"
            ]
        ].to_string(
            index=False
        )
    )


    q = qc[
        qc[
            "TOPIS_LINK_ID"
        ] == lid
    ]


    if len(q) > 0:

        q = q.iloc[0]

        print(
            "TOPIS reported =",
            round(
                q[
                    "TOPIS_REPORTED_LENGTH"
                ],
                1
            ),
            "| covered SUMO =",
            round(
                q[
                    "TOTAL_COVERED_SUMO_LENGTH"
                ],
                1
            ),
            "| ratio =",
            round(
                q[
                    "COVERED_TO_TOPIS_RATIO"
                ],
                3
            )
        )


print(
    "\nCreated:",
    OUTPUT_COVERAGE.name
)

print(
    "Created:",
    OUTPUT_QC.name
)