# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 0. PATH
# ============================================================

BASE = Path(__file__).resolve().parent

NETWORK_FILE = (
    BASE / "seongbuk_buffer.net.xml"
)

MAPPING_FILE = (
    BASE / "topis_sumo_mapping_gap_repaired.csv"
)

OUTPUT_FILE = (
    BASE / "topis_sumo_mapping_loop_cleaned.csv"
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
# 2. MAPPING
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


# ============================================================
# 3. 방금 gap repair했던 7개
# ============================================================

repair_links = [
    "1005004100",
    "1070014900",
    "1070016000",
    "1070017400",
    "1070017700",
    "1070018500",
    "1080001400",
]


# ============================================================
# 4. connectivity
# ============================================================

def count_broken(edge_ids):

    broken = []

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

            broken.append(
                (a, b)
            )

    return broken


def physical_id(eid):

    if eid.startswith("-"):
        return eid[1:]

    return eid


# ============================================================
# 5. 핵심: connected route에서 loop 제거
# ============================================================

def erase_loops(edge_ids):

    if not edge_ids:
        return []

    first = net.getEdge(
        edge_ids[0]
    )

    # 현재 단순 경로의 node sequence
    nodes = [
        first.getFromNode().getID()
    ]

    # 현재 단순 경로의 edge sequence
    edges = []

    # node -> nodes 내 위치
    node_position = {
        nodes[0]: 0
    }

    for eid in edge_ids:

        e = net.getEdge(eid)

        u = e.getFromNode().getID()
        v = e.getToNode().getID()

        # 원래 route 자체가 연결돼 있다는 전제
        if u != nodes[-1]:

            raise RuntimeError(
                f"Input route is broken: "
                f"{nodes[-1]} -> {u} at {eid}"
            )

        # ----------------------------------------------------
        # v를 이미 방문했다면 cycle 발생
        #
        # 예:
        # A-B-C-D-C
        #
        # D-C까지가 loop이므로
        # A-B-C로 돌아감
        # ----------------------------------------------------

        if v in node_position:

            keep_idx = node_position[v]

            # v 이후 nodes 제거
            removed_nodes = nodes[
                keep_idx + 1:
            ]

            for n in removed_nodes:
                node_position.pop(
                    n,
                    None
                )

            nodes = nodes[
                :keep_idx + 1
            ]

            # nodes가 k개면 edge는 k-1개
            edges = edges[
                :keep_idx
            ]

        else:

            edges.append(
                eid
            )

            nodes.append(
                v
            )

            node_position[v] = (
                len(nodes) - 1
            )

    return edges


# ============================================================
# 6. 7개 처리
# ============================================================

replacement = {}

print(
    "\n===== LOOP ERASURE =====\n"
)

for link_id in repair_links:

    old = (
        mapping[
            mapping[
                "TOPIS_LINK_ID"
            ] == link_id
        ]
        .sort_values(
            "SEQ"
        )[
            "SUMO_EDGE_ID"
        ]
        .tolist()
    )

    old_broken = count_broken(
        old
    )

    if old_broken:

        print(
            link_id,
            "SKIP: input already broken"
        )
        continue

    new = erase_loops(
        old
    )

    new_broken = count_broken(
        new
    )

    exact_dup = (
        len(new)
        -
        len(set(new))
    )

    physical = [
        physical_id(e)
        for e in new
    ]

    physical_dup = (
        len(physical)
        -
        len(set(physical))
    )

    replacement[
        link_id
    ] = new

    print(
        link_id,
        "|",
        len(old),
        "->",
        len(new),
        "| duplicate =",
        exact_dup,
        "| physical dup =",
        physical_dup,
        "| broken =",
        len(new_broken)
    )


# ============================================================
# 7. mapping 교체
# ============================================================

keep = mapping[
    ~mapping[
        "TOPIS_LINK_ID"
    ].isin(
        replacement.keys()
    )
].copy()


rows = []

for link_id, edge_ids in (
    replacement.items()
):

    for seq, eid in enumerate(
        edge_ids
    ):

        rows.append({
            "TOPIS_LINK_ID":
                link_id,
            "SUMO_EDGE_ID":
                eid,
            "SEQ":
                seq
        })


replacement_df = pd.DataFrame(
    rows
)


final = pd.concat(
    [
        keep,
        replacement_df
    ],
    ignore_index=True
)

final = (
    final
    .sort_values(
        [
            "TOPIS_LINK_ID",
            "SEQ"
        ]
    )
    .reset_index(
        drop=True
    )
)


final.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nCreated:",
    OUTPUT_FILE.name
)