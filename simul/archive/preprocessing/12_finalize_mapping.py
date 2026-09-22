import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

SOURCE = (
    BASE / "topis_sumo_mapping_corridor.csv"
)

FINAL = (
    BASE / "topis_sumo_mapping_final.csv"
)

EXCLUDED = (
    BASE / "topis_sumo_mapping_excluded.csv"
)

TARGET = "1070012500"


# ============================================================
# 1. 최신 mapping
# ============================================================

mapping = pd.read_csv(
    SOURCE,
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
# 2. topology mismatch 링크 제외
# ============================================================

final = mapping[
    mapping["TOPIS_LINK_ID"] != TARGET
].copy()

final = (
    final
    .sort_values(
        ["TOPIS_LINK_ID", "SEQ"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 3. final mapping 저장
# ============================================================

final.to_csv(
    FINAL,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. 제외 사유 기록
# ============================================================

excluded = pd.DataFrame([
    {
        "TOPIS_LINK_ID": TARGET,
        "REASON":
            "TOPIS-SUMO topology mismatch: "
            "no local directed connection between "
            "0%-25% anchors; native SUMO path "
            "requires large detour."
    }
])

excluded.to_csv(
    EXCLUDED,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. sanity check
# ============================================================

print(
    "Mapped TOPIS links:",
    final["TOPIS_LINK_ID"].nunique()
)

print(
    "Mapping rows:",
    len(final)
)

print(
    "Excluded TOPIS links:",
    len(excluded)
)

print(
    "\nCreated:",
    FINAL.name
)

print(
    "Created:",
    EXCLUDED.name
)