# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

COST_FILE = (
    BASE / "sumo_hourly_cost_operational_v3.csv"
)

OUTPUT_DIR = (
    BASE / "traffic_plots"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 1. SETTINGS
# ============================================================

# 시간 → CSV hour column
#
# TOPIS의 ~01시 = 00:00~01:00 로 해석
#
HOURS = {
    0:  "~01시",
    1: "~02시",
    2: "~03시",
    3: "~04시",
    4: "~05시",
    5: "~06시",
    6:  "~07시",
    7:  "~08시",
    8:  "~09시",
    9:  "~10시",
    10: "~11시",
    11: "~12시",
    12: "~13시",
    13: "~14시",
    14: "~15시",
    15: "~16시",
    16: "~17시",
    17: "~18시",
    18: "~19시",
    19: "~20시",
    20: "~21시",
    21: "~22시",
    22: "~23시",
    23: "~24시",
}


# ------------------------------------------------------------
# COLOR THRESHOLDS
#
# <= LOW_SPEED  : red
# >= HIGH_SPEED : green
# between       : red -> yellow -> green
# ------------------------------------------------------------

LOW_SPEED = 10.0       # km/h
HIGH_SPEED = 60.0      # km/h


# 도로 굵기
LINE_WIDTH = 0.75

# figure size
FIGSIZE = (10, 10)

# PNG resolution
DPI = 250


# ------------------------------------------------------------
# True:
#   bin convex hull 내부 CORE만 그림
#
# False:
#   2 km BUFFER까지 포함한 operational network 전체
# ------------------------------------------------------------

CORE_ONLY = False


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
# 3. COST
# ============================================================

df = pd.read_csv(
    COST_FILE,
    dtype={
        "SUMO_EDGE_ID": str
    }
)

df["SUMO_EDGE_ID"] = (
    df["SUMO_EDGE_ID"]
    .str.strip()
)


if CORE_ONLY:

    df = df[
        df["REGION_TYPE"]
        ==
        "CORE"
    ].copy()


print(
    "Edges to plot:",
    len(df)
)


# ============================================================
# 4. EDGE GEOMETRY CACHE
# ============================================================

edge_segments = {}
edge_lengths = {}


for eid in df["SUMO_EDGE_ID"]:

    try:

        edge = net.getEdge(
            eid
        )

    except Exception:

        continue


    shape = edge.getShape()

    if len(shape) < 2:
        continue


    # LineCollection에서는
    # [(x1,y1), (x2,y2), ...] 그대로 사용 가능
    edge_segments[eid] = np.asarray(
        shape,
        dtype=float
    )


    edge_lengths[eid] = float(
        edge.getLength()
    )


# geometry 없는 행 제거
df = df[
    df["SUMO_EDGE_ID"].isin(
        edge_segments.keys()
    )
].copy()


print(
    "Edges with geometry:",
    len(df)
)


# ============================================================
# 5. PLOT BOUNDS
#
# 네 그림 모두 완전히 동일한 범위 사용
# ============================================================

all_x = []
all_y = []


for shape in edge_segments.values():

    all_x.extend(
        shape[:, 0]
    )

    all_y.extend(
        shape[:, 1]
    )


xmin = min(all_x)
xmax = max(all_x)

ymin = min(all_y)
ymax = max(all_y)


# 약간 여백
x_pad = (
    xmax - xmin
) * 0.02

y_pad = (
    ymax - ymin
) * 0.02


# ============================================================
# 6. COLORMAP
#
# 빨강 -> 노랑 -> 초록
# ============================================================

traffic_cmap = (
    LinearSegmentedColormap.from_list(
        "traffic",
        [
            "#d73027",   # red
            "#fee08b",   # yellow
            "#1a9850",   # green
        ]
    )
)


# LOW 이하와 HIGH 이상은 자동 clip
norm = Normalize(
    vmin=LOW_SPEED,
    vmax=HIGH_SPEED,
    clip=True
)


# ============================================================
# 7. NETWORK STATISTICS
# ============================================================

def calculate_network_stats(
    data,
    tt_col,
    speed_col
):

    """
    단순 edge 평균 TT는 SUMO edge segmentation에
    민감하므로 주요 지표로 쓰지 않음.

    대신:
      total_time / total_length
    로 전체 네트워크의 평균 travel time per km 계산.

    Equivalent speed:
      total_length / total_time
    """

    valid = data[
        data[tt_col].notna()
        &
        (
            data[tt_col] > 0
        )
    ].copy()


    lengths = (
        valid["SUMO_EDGE_ID"]
        .map(
            edge_lengths
        )
        .astype(float)
    )


    tt = valid[
        tt_col
    ].astype(float)


    total_length = (
        lengths.sum()
    )

    total_tt = (
        tt.sum()
    )


    # seconds / km
    mean_tt_per_km = (
        total_tt
        /
        total_length
        *
        1000.0
    )


    # harmonic/network-equivalent speed
    equivalent_speed = (
        total_length
        /
        total_tt
        *
        3.6
    )


    # 참고용 length-weighted speed
    weighted_mean_speed = np.average(
        valid[
            speed_col
        ],
        weights=lengths
    )


    return (
        mean_tt_per_km,
        equivalent_speed,
        weighted_mean_speed
    )


# ============================================================
# 8. PLOT FUNCTION
# ============================================================

def plot_hour(
    display_hour,
    hour_label
):

    speed_col = (
        f"SPEED_{hour_label}"
    )

    tt_col = (
        f"TT_{hour_label}"
    )


    if speed_col not in df.columns:

        raise RuntimeError(
            f"Missing column: {speed_col}"
        )


    if tt_col not in df.columns:

        raise RuntimeError(
            f"Missing column: {tt_col}"
        )


    plot_df = df[
        df[speed_col].notna()
    ].copy()


    # --------------------------------------------------------
    # LineCollection용 geometry / speed
    # --------------------------------------------------------

    segments = []

    speeds = []


    for _, row in plot_df.iterrows():

        eid = row[
            "SUMO_EDGE_ID"
        ]


        if eid not in edge_segments:
            continue


        segments.append(
            edge_segments[
                eid
            ]
        )


        speeds.append(
            float(
                row[
                    speed_col
                ]
            )
        )


    speeds = np.asarray(
        speeds,
        dtype=float
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    (
        mean_tt_per_km,
        equivalent_speed,
        weighted_mean_speed
    ) = calculate_network_stats(
        plot_df,
        tt_col,
        speed_col
    )


    print(
        f"\n{display_hour:02d}:00"
    )

    print(
        "Mean travel time per km:",
        round(
            mean_tt_per_km,
            1
        ),
        "sec/km"
    )

    print(
        "Equivalent network speed:",
        round(
            equivalent_speed,
            2
        ),
        "km/h"
    )

    print(
        "Length-weighted mean speed:",
        round(
            weighted_mean_speed,
            2
        ),
        "km/h"
    )


    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )


    collection = LineCollection(
        segments,
        cmap=traffic_cmap,
        norm=norm,
        linewidths=LINE_WIDTH,
        alpha=0.95
    )


    collection.set_array(
        speeds
    )


    ax.add_collection(
        collection
    )


    # --------------------------------------------------------
    # same limits for all figures
    # --------------------------------------------------------

    ax.set_xlim(
        xmin - x_pad,
        xmax + x_pad
    )

    ax.set_ylim(
        ymin - y_pad,
        ymax + y_pad
    )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.axis(
        "off"
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    minutes_per_km = (
        mean_tt_per_km
        /
        60.0
    )


    ax.set_title(
        (
            f"{display_hour:02d}:00 Traffic Conditions\n"
            f"Average travel time = "
            f"{minutes_per_km:.2f} min/km"
            f"   |   "
            f"Equivalent speed = "
            f"{equivalent_speed:.1f} km/h"
        ),
        fontsize=13,
        pad=12
    )


    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap=traffic_cmap
    )

    sm.set_array([])


    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.015
    )


    cbar.set_label(
        "Travel speed (km/h)"
    )


    # threshold ticks를 명확하게
    cbar.set_ticks(
        [
            LOW_SPEED,
            (
                LOW_SPEED
                +
                HIGH_SPEED
            ) / 2,
            HIGH_SPEED
        ]
    )


    cbar.set_ticklabels(
        [
            f"≤ {LOW_SPEED:.0f}",
            f"{(LOW_SPEED + HIGH_SPEED)/2:.0f}",
            f"≥ {HIGH_SPEED:.0f}",
        ]
    )


    fig.tight_layout()


    filename = (
        OUTPUT_DIR
        /
        f"traffic_{display_hour:02d}00.png"
    )


    fig.savefig(
        filename,
        dpi=DPI,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        "Created:",
        filename.name
    )


# ============================================================
# 9. RUN FOUR TIMES
# ============================================================

for hour, label in HOURS.items():

    plot_hour(
        hour,
        label
    )


print(
    "\n=============================="
)

print(
    "DONE"
)

print(
    "=============================="
)

print(
    "Plots saved in:",
    OUTPUT_DIR
)


from pathlib import Path
from PIL import Image


BASE = Path(__file__).resolve().parent

INPUT_DIR = BASE / "traffic_plots"
OUTPUT_GIF = BASE / "traffic_24h.gif"


# 00:00 ~ 23:00 순서대로
files = [
    INPUT_DIR / f"traffic_{h:02d}00.png"
    for h in range(24)
]


images = []

for f in files:

    if not f.exists():
        raise FileNotFoundError(
            f"Missing file: {f}"
        )

    img = Image.open(f).convert("RGB")

    images.append(img)


# 첫 프레임
first = images[0]

# 나머지 프레임
rest = images[1:]


first.save(
    OUTPUT_GIF,
    save_all=True,
    append_images=rest,

    # 한 프레임 표시 시간 [ms]
    duration=400,

    # 무한 반복
    loop=0,

    optimize=True
)


print("Created:", OUTPUT_GIF)