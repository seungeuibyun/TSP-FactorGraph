import numpy as np
import math
import pandas as pd
from pyproj import Transformer
import os
import sys

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib

net = sumolib.net.readNet("seongbuk.net.xml")


mapping = pd.read_csv(
    "topis_sumo_mapping.csv",
    dtype={"TOPIS_LINK_ID": str}
)

speed = pd.read_excel(
    "topis_speed.xlsx",
    dtype={"링크아이디": str}
)

speed["링크아이디"] = speed["링크아이디"].str.strip()
mapping["TOPIS_LINK_ID"] = mapping["TOPIS_LINK_ID"].str.strip()

print("속도 데이터 TOPIS links:", speed["링크아이디"].nunique())
print("매핑된 TOPIS links:", mapping["TOPIS_LINK_ID"].nunique())

missing = set(speed["링크아이디"]) - set(mapping["TOPIS_LINK_ID"])

print("매핑 안 된 TOPIS links:", len(missing))
print(missing)

mapping["SUMO_LENGTH"] = mapping["SUMO_EDGE_ID"].apply(
    lambda eid: net.getEdge(eid).getLength()
)

sumo_link_length = (
    mapping.groupby("TOPIS_LINK_ID")["SUMO_LENGTH"]
    .sum()
    .rename("MATCHED_SUMO_LENGTH")
)

mapping = mapping.merge(
    sumo_link_length,
    on="TOPIS_LINK_ID"
)

speed2 = speed.rename(
    columns={
        "링크아이디": "TOPIS_LINK_ID",
        "거리": "TOPIS_LENGTH"
    }
)

df = mapping.merge(
    speed2,
    on="TOPIS_LINK_ID",
    how="left"
)

hour_cols = [f"~{h:02d}시" for h in range(1, 25)]

for col in hour_cols:

    # km/h -> m/s
    speed_ms = df[col] / 3.6

    # TOPIS 링크 전체 관측 travel time
    topis_tt = df["TOPIS_LENGTH"] / speed_ms

    # SUMO edge에 길이 비례 배분
    df[f"TT_{col}"] = (
        topis_tt
        * df["SUMO_LENGTH"]
        / df["MATCHED_SUMO_LENGTH"]
    )

    output_cols = [
    "TOPIS_LINK_ID",
    "SUMO_EDGE_ID",
    "SEQ",
    "SUMO_LENGTH"
] + [f"TT_{c}" for c in hour_cols]

edge_cost = df[output_cols]

edge_cost.to_csv(
    "sumo_topis_hourly_cost.csv",
    index=False,
    encoding="utf-8-sig"
)

print(edge_cost.head())

import pandas as pd

df = pd.read_csv(
    "sumo_topis_hourly_cost.csv",
    dtype={
        "TOPIS_LINK_ID": str,
        "SUMO_EDGE_ID": str
    }
)

# 같은 TOPIS 링크 안에서 같은 SUMO edge가 두 번 이상 등장하는 경우
bad = df[
    df.duplicated(
        ["TOPIS_LINK_ID", "SUMO_EDGE_ID"],
        keep=False
    )
]

bad_links = sorted(
    bad["TOPIS_LINK_ID"].unique()
)

print("problematic TOPIS links:", len(bad_links))

