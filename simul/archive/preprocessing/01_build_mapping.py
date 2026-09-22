# -*- coding: utf-8 -*-

import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import unary_union


# ============================================================
# 0. 설정
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = BASE / "seongbuk_buffer.net.xml"
SPEED_FILE = BASE / "topis_speed.xlsx"
VERTEX_FILE = BASE / "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx"

OUTPUT_MAPPING = BASE / "topis_sumo_mapping_best.csv"
OUTPUT_QC = BASE / "topis_sumo_mapping_qc.csv"
OUTPUT_REVIEW = BASE / "topis_sumo_mapping_review.csv"
OUTPUT_FINAL = BASE / "topis_sumo_mapping_final.csv"
OUTPUT_COST = BASE / "sumo_topis_hourly_cost.csv"

PLOT_DIR = BASE / "mapping_review_plots"

EXPECTED_LINKS = 183


# ============================================================
# 1. SUMO 설정
# ============================================================

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib
from sumolib.route import mapTrace

print("Loading SUMO network...")

net = sumolib.net.readNet(
    str(NETWORK_FILE)
)


# ============================================================
# 2. TOPIS 속도 데이터
# ============================================================

print("Loading TOPIS speed data...")

speed = pd.read_excel(
    SPEED_FILE,
    dtype={"링크아이디": str}
)

speed["링크아이디"] = (
    speed["링크아이디"]
    .str.strip()
)

speed_ids = set(
    speed["링크아이디"]
)

print(
    "TOPIS links in speed file:",
    len(speed_ids)
)

if len(speed_ids) != EXPECTED_LINKS:
    print(
        "WARNING: expected",
        EXPECTED_LINKS,
        "but found",
        len(speed_ids)
    )


# ============================================================
# 3. LINK_VERTEX 읽기
# ============================================================

print("Loading LINK_VERTEX...")

vtx = pd.read_excel(
    VERTEX_FILE,
    dtype={"LINK_ID": str}
)

vtx["LINK_ID"] = (
    vtx["LINK_ID"]
    .str.strip()
)

# 성북구 183개 링크만
vtx = vtx[
    vtx["LINK_ID"].isin(speed_ids)
].copy()

print(
    "vertex links:",
    vtx["LINK_ID"].nunique()
)


# ============================================================
# 4. TOPIS GRS80 TM -> SUMO XY
# ============================================================

print("Converting coordinates...")

to_wgs84 = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


def topis_to_sumo(x, y):

    lon, lat = to_wgs84.transform(
        x,
        y
    )

    return net.convertLonLat2XY(
        lon,
        lat
    )


coords = [
    topis_to_sumo(x, y)
    for x, y in zip(
        vtx["GRS80TM_X"],
        vtx["GRS80TM_Y"]
    )
]

vtx["SUMO_X"] = [
    p[0] for p in coords
]

vtx["SUMO_Y"] = [
    p[1] for p in coords
]


# ============================================================
# 5. TOPIS LineString 생성
# ============================================================

vtx = vtx.sort_values(
    ["LINK_ID", "VER_SEQ"]
)

topis_lines = {}

for link_id, group in vtx.groupby(
    "LINK_ID"
):

    xy = list(
        zip(
            group["SUMO_X"],
            group["SUMO_Y"]
        )
    )

    if len(xy) >= 2:

        topis_lines[link_id] = (
            LineString(xy)
        )


print(
    "TOPIS LineStrings:",
    len(topis_lines)
)


# ============================================================
# 6. 기본 mapTrace
# ============================================================

def trace_match(
    line,
    spacing,
    delta,
    fill_gaps,
    reversal_penalty
):

    if line.length <= 10:

        ds = [
            line.length / 2
        ]

    else:

        ds = np.arange(
            5.0,
            line.length - 5.0,
            spacing
        )

    trace = [
        (
            line.interpolate(d).x,
            line.interpolate(d).y
        )
        for d in ds
    ]

    if len(trace) < 2:

        trace = [
            (
                line.interpolate(
                    0.25,
                    normalized=True
                ).x,
                line.interpolate(
                    0.25,
                    normalized=True
                ).y
            ),
            (
                line.interpolate(
                    0.75,
                    normalized=True
                ).x,
                line.interpolate(
                    0.75,
                    normalized=True
                ).y
            )
        ]

    route = mapTrace(
        trace,
        net,
        delta=delta,
        fillGaps=fill_gaps,
        direction=True,
        reversalPenalty=reversal_penalty
    )

    return [
        e.getID()
        for e in route
    ]


def primary_match(line):

    return trace_match(
        line=line,
        spacing=10,
        delta=30,
        fill_gaps=100,
        reversal_penalty=1000
    )


def secondary_match(line):

    return trace_match(
        line=line,
        spacing=5,
        delta=20,
        fill_gaps=30,
        reversal_penalty=10000
    )


# ============================================================
# 7. QC 함수
# ============================================================

def route_geometry(edge_ids):

    geometries = []

    for eid in edge_ids:

        try:

            edge = net.getEdge(eid)

        except Exception:

            continue

        shape = edge.getShape()

        if len(shape) >= 2:

            geometries.append(
                LineString(shape)
            )

    if not geometries:

        return None

    return unary_union(
        geometries
    )


def sample_line(line, spacing=10):

    if line.length == 0:

        return []

    ds = list(
        np.arange(
            0,
            line.length,
            spacing
        )
    )

    ds.append(
        line.length
    )

    return [
        line.interpolate(d)
        for d in ds
    ]


def evaluate_mapping(
    topis_line,
    edge_ids
):

    if len(edge_ids) == 0:

        return {
            "n_edges": 0,
            "duplicates": 0,
            "physical_duplicates": 0,
            "broken": 999,
            "mean_topis_dist": 9999,
            "max_topis_dist": 9999,
            "endpoint_gap": 9999,
            "score": 1e9,
            "pass": False
        }

    # --------------------------------
    # exact duplicate edge
    # --------------------------------

    duplicates = (
        len(edge_ids)
        - len(set(edge_ids))
    )

    # +edge/-edge 왕복 oscillation 검사
    canonical_ids = [
        eid[1:]
        if eid.startswith("-")
        else eid
        for eid in edge_ids
    ]

    physical_duplicates = (
        len(canonical_ids)
        - len(set(canonical_ids))
    )

    # --------------------------------
    # network connectivity
    # --------------------------------

    broken = 0

    for a, b in zip(
        edge_ids[:-1],
        edge_ids[1:]
    ):

        ea = net.getEdge(a)
        eb = net.getEdge(b)

        if (
            ea.getToNode().getID()
            !=
            eb.getFromNode().getID()
        ):

            broken += 1

    # --------------------------------
    # geometry distance
    # --------------------------------

    rg = route_geometry(
        edge_ids
    )

    if rg is None:

        mean_dist = 9999
        max_dist = 9999

    else:

        points = sample_line(
            topis_line,
            spacing=10
        )

        distances = [
            p.distance(rg)
            for p in points
        ]

        mean_dist = float(
            np.mean(distances)
        )

        max_dist = float(
            np.max(distances)
        )

    # --------------------------------
    # endpoint coverage
    # --------------------------------

    first_edge = net.getEdge(
        edge_ids[0]
    )

    last_edge = net.getEdge(
        edge_ids[-1]
    )

    first_geom = LineString(
        first_edge.getShape()
    )

    last_geom = LineString(
        last_edge.getShape()
    )

    start_point = topis_line.interpolate(
        0
    )

    end_point = topis_line.interpolate(
        topis_line.length
    )

    endpoint_gap = (
        start_point.distance(first_geom)
        +
        end_point.distance(last_geom)
    )

    # --------------------------------
    # score
    # --------------------------------

    score = (
        mean_dist
        + 0.2 * max_dist
        + 20 * duplicates
        + 20 * physical_duplicates
        + 50 * broken
        + 0.2 * endpoint_gap
    )

    # buffered network 기준 자동 QC
    passed = (
        duplicates == 0
        and physical_duplicates == 0
        and broken == 0
        and mean_dist <= 15
        and max_dist <= 50
        and endpoint_gap <= 80
    )

    return {
        "n_edges": len(edge_ids),
        "duplicates": duplicates,
        "physical_duplicates":
            physical_duplicates,
        "broken": broken,
        "mean_topis_dist":
            mean_dist,
        "max_topis_dist":
            max_dist,
        "endpoint_gap":
            endpoint_gap,
        "score":
            score,
        "pass":
            passed
    }


# ============================================================
# 8. shortest path fallback candidate
# ============================================================

def shortest_path_candidate(
    edge_ids
):

    if len(edge_ids) < 2:

        return None

    try:

        start_edge = net.getEdge(
            edge_ids[0]
        )

        end_edge = net.getEdge(
            edge_ids[-1]
        )

        result = net.getShortestPath(
            start_edge,
            end_edge
        )

        if result is None:

            return None

        path, cost = result

        if path is None:

            return None

        return [
            e.getID()
            for e in path
        ]

    except Exception:

        return None


# ============================================================
# 9. 183개 매칭
# ============================================================

print("\nStarting TOPIS -> SUMO matching...\n")

mapping_results = {}
qc_rows = []

total = len(topis_lines)

for idx, (
    link_id,
    line
) in enumerate(
    topis_lines.items(),
    1
):

    print(
        f"[{idx}/{total}] {link_id}"
    )

    candidates = []

    # --------------------------------
    # candidate 1: primary
    # --------------------------------

    p1 = primary_match(
        line
    )

    q1 = evaluate_mapping(
        line,
        p1
    )

    candidates.append(
        (
            "primary",
            p1,
            q1
        )
    )

    # --------------------------------
    # candidate 2: conservative
    # --------------------------------

    p2 = secondary_match(
        line
    )

    q2 = evaluate_mapping(
        line,
        p2
    )

    candidates.append(
        (
            "secondary",
            p2,
            q2
        )
    )

    # --------------------------------
    # candidate 3:
    # shortest path between
    # secondary start/end
    # --------------------------------

    sp = shortest_path_candidate(
        p2
    )

    if sp is not None:

        q3 = evaluate_mapping(
            line,
            sp
        )

        candidates.append(
            (
                "shortest_path",
                sp,
                q3
            )
        )

    # --------------------------------
    # passing candidate가 있으면
    # 그중 score 최소 선택
    # --------------------------------

    passed_candidates = [
        c
        for c in candidates
        if c[2]["pass"]
    ]

    if passed_candidates:

        best = min(
            passed_candidates,
            key=lambda x:
                x[2]["score"]
        )

    else:

        # pass한 게 없으면
        # score가 가장 낮은 것 저장,
        # 대신 review 표시
        best = min(
            candidates,
            key=lambda x:
                x[2]["score"]
        )

    method, edge_ids, qc = best

    mapping_results[
        link_id
    ] = edge_ids

    qc_rows.append({
        "TOPIS_LINK_ID":
            link_id,
        "METHOD":
            method,
        **qc
    })


# ============================================================
# 10. mapping CSV 저장
# ============================================================

mapping_rows = []

for link_id, edge_ids \
        in mapping_results.items():

    for seq, eid \
            in enumerate(edge_ids):

        mapping_rows.append({
            "TOPIS_LINK_ID":
                link_id,
            "SUMO_EDGE_ID":
                eid,
            "SEQ":
                seq
        })


mapping_df = pd.DataFrame(
    mapping_rows
)

mapping_df.to_csv(
    OUTPUT_MAPPING,
    index=False,
    encoding="utf-8-sig"
)


qc_df = pd.DataFrame(
    qc_rows
)

qc_df.to_csv(
    OUTPUT_QC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. QC 결과
# ============================================================

review = qc_df[
    ~qc_df["pass"]
].copy()

print("\n==============================")
print("MAPPING SUMMARY")
print("==============================")

print(
    "TOPIS links:",
    len(mapping_results)
)

print(
    "PASS:",
    qc_df["pass"].sum()
)

print(
    "REVIEW:",
    len(review)
)

print(
    "mean geometry error:",
    round(
        qc_df[
            "mean_topis_dist"
        ].mean(),
        2
    ),
    "m"
)


# ============================================================
# 12. 문제 링크만 그림 저장
# ============================================================

if len(review) > 0:

    print(
        "\nSaving review plots..."
    )

    PLOT_DIR.mkdir(
        exist_ok=True
    )

    review.to_csv(
        OUTPUT_REVIEW,
        index=False,
        encoding="utf-8-sig"
    )

    for link_id in review[
        "TOPIS_LINK_ID"
    ]:

        line = topis_lines[
            link_id
        ]

        edge_ids = mapping_results[
            link_id
        ]

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        minx, miny, maxx, maxy = (
            line.bounds
        )

        margin = 150

        # 주변 SUMO road
        for e in net.getEdges():

            shape = e.getShape()

            if len(shape) < 2:

                continue

            xs = [
                p[0]
                for p in shape
            ]

            ys = [
                p[1]
                for p in shape
            ]

            if (
                max(xs) <
                    minx - margin
                or
                min(xs) >
                    maxx + margin
                or
                max(ys) <
                    miny - margin
                or
                min(ys) >
                    maxy + margin
            ):

                continue

            ax.plot(
                xs,
                ys,
                linewidth=0.3,
                alpha=0.25
            )

        # TOPIS
        tx, ty = line.xy

        ax.plot(
            tx,
            ty,
            linewidth=4,
            alpha=0.6
        )

        # matched edges
        for eid in edge_ids:

            e = net.getEdge(
                eid
            )

            shape = e.getShape()

            xs = [
                p[0]
                for p in shape
            ]

            ys = [
                p[1]
                for p in shape
            ]

            ax.plot(
                xs,
                ys,
                linewidth=2.5,
                alpha=0.9
            )

        q = qc_df[
            qc_df[
                "TOPIS_LINK_ID"
            ] == link_id
        ].iloc[0]

        ax.set_title(
            f"{link_id}"
            f" | {q['METHOD']}"
            f" | mean="
            f"{q['mean_topis_dist']:.1f}m"
        )

        ax.set_xlim(
            minx - margin,
            maxx + margin
        )

        ax.set_ylim(
            miny - margin,
            maxy + margin
        )

        ax.set_aspect(
            "equal"
        )

        fig.tight_layout()

        fig.savefig(
            PLOT_DIR /
            f"{link_id}.png",
            dpi=180
        )

        plt.close(
            fig
        )


# ============================================================
# 13. 모든 링크가 통과했으면 FINAL mapping
# ============================================================

if len(review) == 0:

    mapping_df.to_csv(
        OUTPUT_FINAL,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nAll links passed QC."
    )

    print(
        "Created:",
        OUTPUT_FINAL.name
    )

else:

    print(
        "\nSome links still need review."
    )

    print(
        "Check:",
        OUTPUT_REVIEW.name
    )

    print(
        "and folder:",
        PLOT_DIR.name
    )


# ============================================================
# 14. hourly travel-time cost
# ============================================================
#
# review가 0개일 때만 자동 생성.
#
# TOPIS 관측 링크 통행시간:
#
# T_l(h) =
#     TOPIS_LENGTH /
#     (TOPIS_SPEED(h) / 3.6)
#
# 을 해당 SUMO edge 길이에 비례해서 분배.
#
# ============================================================

if len(review) == 0:

    print(
        "\nBuilding hourly cost..."
    )

    mapping_cost = (
        mapping_df.copy()
    )

    mapping_cost[
        "SUMO_LENGTH"
    ] = mapping_cost[
        "SUMO_EDGE_ID"
    ].apply(
        lambda eid:
            net.getEdge(
                eid
            ).getLength()
    )

    sumo_total_length = (
        mapping_cost
        .groupby(
            "TOPIS_LINK_ID"
        )["SUMO_LENGTH"]
        .sum()
        .rename(
            "MATCHED_SUMO_LENGTH"
        )
    )

    mapping_cost = (
        mapping_cost.merge(
            sumo_total_length,
            on="TOPIS_LINK_ID",
            how="left"
        )
    )

    speed2 = speed.rename(
        columns={
            "링크아이디":
                "TOPIS_LINK_ID",
            "거리":
                "TOPIS_LENGTH"
        }
    )

    df = mapping_cost.merge(
        speed2,
        on="TOPIS_LINK_ID",
        how="left"
    )

    hour_cols = [
        f"~{h:02d}시"
        for h in range(
            1,
            25
        )
    ]

    missing_hours = [
        c
        for c in hour_cols
        if c not in df.columns
    ]

    if missing_hours:

        raise RuntimeError(
            "Missing hourly speed columns: "
            + str(
                missing_hours
            )
        )

    for col in hour_cols:

        v_ms = (
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
            /
            3.6
        )

        topis_length = (
            pd.to_numeric(
                df[
                    "TOPIS_LENGTH"
                ],
                errors="coerce"
            )
        )

        topis_tt = (
            topis_length
            /
            v_ms
        )

        df[
            f"TT_{col}"
        ] = (
            topis_tt
            *
            df["SUMO_LENGTH"]
            /
            df[
                "MATCHED_SUMO_LENGTH"
            ]
        )

    output_cols = [
        "TOPIS_LINK_ID",
        "SUMO_EDGE_ID",
        "SEQ",
        "SUMO_LENGTH"
    ] + [
        f"TT_{c}"
        for c in hour_cols
    ]

    hourly_cost = df[
        output_cols
    ].copy()

    hourly_cost.to_csv(
        OUTPUT_COST,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Created:",
        OUTPUT_COST.name
    )


# ============================================================
# 15. 마지막 sanity check
# ============================================================

print("\n==============================")
print("DONE")
print("==============================")

print(
    "Mapped unique TOPIS links:",
    mapping_df[
        "TOPIS_LINK_ID"
    ].nunique()
)

print(
    "Total TOPIS-SUMO rows:",
    len(mapping_df)
)

if len(review) == 0:

    print(
        "FINAL STATUS: READY"
    )

else:

    print(
        "FINAL STATUS:",
        len(review),
        "links require visual review"
    )