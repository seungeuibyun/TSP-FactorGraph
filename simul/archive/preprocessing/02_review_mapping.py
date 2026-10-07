import pandas as pd

qc = pd.read_csv(
    "topis_sumo_mapping_review.csv",
    dtype={"TOPIS_LINK_ID": str}
)

def get_reason(row):

    reasons = []

    if row["duplicates"] > 0:
        reasons.append("duplicate")

    if row["physical_duplicates"] > 0:
        reasons.append("direction_oscillation")

    if row["broken"] > 0:
        reasons.append("broken")

    if row["mean_topis_dist"] > 15:
        reasons.append("mean_distance")

    if row["max_topis_dist"] > 50:
        reasons.append("max_distance")

    if row["endpoint_gap"] > 80:
        reasons.append("endpoint_gap")

    return ", ".join(reasons)

qc["FAIL_REASON"] = qc.apply(
    get_reason,
    axis=1
)

cols = [
    "TOPIS_LINK_ID",
    "METHOD",
    "n_edges",
    "duplicates",
    "physical_duplicates",
    "broken",
    "mean_topis_dist",
    "max_topis_dist",
    "endpoint_gap",
    "FAIL_REASON"
]

qc = qc[cols]

print("\n===== REVIEW LINKS =====\n")

print(
    qc.to_string(
        index=False
    )
)

print("\n===== REASON COUNTS =====\n")

print(
    qc["FAIL_REASON"]
    .value_counts()
)

qc.to_csv(
    "topis_sumo_mapping_review_detailed.csv",
    index=False,
    encoding="utf-8-sig"
)