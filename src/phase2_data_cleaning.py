"""
Phase 2 — Data Cleaning (Deep Dive)
====================================
AgroMill Corp — Data Mining II (GW3)

Tasks:
  1. Missing data analysis (by column / time / shift)
  2. Illogical / impossible value detection
  3. Outlier analysis
  4. Cleaning decisions → cleaned datasets + log
"""

from pathlib import Path
import warnings
import sys

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
OUTPUTS = Path("outputs")
TABLES = OUTPUTS / "tables"

# ── Helpers ────────────────────────────────────────────────────────────────────


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def save(df: pd.DataFrame, name: str, suffix: str):
    stem = f"{name}_{suffix}"
    df.to_parquet(PROCESSED / f"{stem}.parquet", index=False)
    print(f"  ✓  {stem}.parquet  ({len(df)} rows)")


def missing_report(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a DataFrame with per-column null counts & percentages."""
    missing = df.isnull().sum()
    total = len(df)
    out = pd.DataFrame({
        "dataset": label,
        "column": missing.index,
        "null_count": missing.values,
        "null_pct": (missing.values / total * 100).round(2),
    })
    out = out[out.null_count > 0].reset_index(drop=True)
    return out


SRC_TAG_GROUPS = {
    "scale":       [f"scale_{i:02d}" for i in range(1, 14)],
    "flow_rate":   [f"flow_rate_scale_{i:02d}" for i in range(1, 14)],
    "consumo":     ["node_a.consumo_moinho_88", "node_a.consumo_moinho_89"],
    "sp_vel":      ["node_a.sp_vel_moinho_88", "node_a.sp_vel_moinho_89"],
    "pv_vel":      ["node_a.pv_vel_moinho_88", "node_a.pv_vel_moinho_89"],
    "start":       ["node_a.start_moinho_88", "node_a.start_moinho_89"],
    "avaria":      ["node_a.avaria_moinho_88", "node_a.avaria_moinho_89"],
    "corrente":    ["node_b.corrente_motor_89", "node_b.corrente_motor_99"],
    "nivel":       ["node_b.storage098_nivel", "node_b.storage099_nivel"],
}
TAG_GROUP_LOOKUP = {}
for group, tags in SRC_TAG_GROUPS.items():
    for t in tags:
        TAG_GROUP_LOOKUP[t] = group


def tag_group(tag: str) -> str:
    return TAG_GROUP_LOOKUP.get(tag, "other")


# 0) Helper: column groups
GRANULOMETRY_COLS = [
    "fraction_8_236", "fraction_10_200", "fraction_14_140",
    "fraction_18_100", "fraction_25_071", "under_000",
]
PRODUCT_VALUE_COLS = ["density", "moisture"] + GRANULOMETRY_COLS

# ── 1.  LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 68)
print("PHASE 2 — DATA CLEANING")
print("=" * 68)

print("\n── Loading datasets ──")
sensors     = load("sensors")
product_01  = load("product_01")
product_01c = load("product_01_client2")
product_02  = load("product_02")
equip       = load("equipment_separation")
mesh        = load("mesh")
print("  ✓  All datasets loaded\n")


# ── 2.  SENSORS ───────────────────────────────────────────────────────────────
print("── Sensors ──")
sns_before = len(sensors)

# 2a. Tag metadata for reporting
sensors["tag_group"] = sensors["tag"].map(tag_group)
g_scale = sensors["tag_group"] == "scale"
g_flow  = sensors["tag_group"] == "flow_rate"
g_nivel = sensors["tag_group"] == "nivel"

# 2b. Illogical: flow_rate spikes (saturation / error code ≈ 66000)
#     Domain reasoning: max plausible flow rate is ~1300 (99.9th percentile ≈ 400–500).
#     Values near 66000 are the sensor's error / saturation sentinel.
flow_hi_mask = g_flow & (sensors["value"] >= 65000)
n_flow_hi = flow_hi_mask.sum()
print(f"  flow_rate ≥ 65000 (sensor error): {n_flow_hi} rows  →  set to NaN")

# 2c. Illogical: storage nivel spikes at exactly 65951
nivel_hi_mask = g_nivel & (sensors["value"] >= 65000)
n_nivel_hi = nivel_hi_mask.sum()
print(f"  storage nivel ≥ 65000 (sensor error): {n_nivel_hi} rows  →  set to NaN")

# 2d. Illogical: negative values in scale / flow_rate (physical impossibility)
neg_mask = (g_scale | g_flow) & (sensors["value"] < 0)
n_neg = neg_mask.sum()
print(f"  scale / flow_rate negative: {n_neg} rows  →  set to NaN")

# 2e. Invalid timestamp duplicates (duplicate timestamp + tag + same value)
ts_dup_mask = sensors.duplicated(subset=["tag", "datetime", "value"], keep="first")
n_ts_dup = ts_dup_mask.sum()
print(f"  Exact dup (tag+ts+value): {n_ts_dup} rows  →  drop")

# Execute cleaning
sensors.loc[flow_hi_mask | nivel_hi_mask | neg_mask, "value"] = np.nan
sensors.drop(ts_dup_mask.index[ts_dup_mask], inplace=True)
sensors.drop(columns=["tag_group"], inplace=True)

sns_after = len(sensors)
sns_removed = sns_before - sns_after
print(f"\n  Sensors: {sns_before} → {sns_after}  ({sns_removed} removed, "
      f"{flow_hi_mask.sum() + nivel_hi_mask.sum() + neg_mask.sum()} values set to NaN)\n")


# ── 3.  EQUIPMENT SEPARATION ──────────────────────────────────────────────────
print("── Equipment Separation ──")
eqp_before = len(equip)

# 3a. Drop exact duplicates
eqp_dup_mask = equip.duplicated(keep="first")
n_eqp_dup = eqp_dup_mask.sum()
print(f"  Exact duplicate rows: {n_eqp_dup}")

# 3b. Decimal mesh_opening_spec — values like 2.38, 5.37, 8.06 are likely
#     mm values accidentally mixed into µm column. Flag but keep as-is
#     (business must confirm intent).
eqp_decimals = equip["mesh_opening_spec"].apply(
    lambda x: isinstance(x, (int, float)) and not pd.isna(x) and x != int(x)
)
n_eqp_dec = eqp_decimals.sum()
print(f"  Decimal mesh_opening_spec ({n_eqp_dec} rows): likely mm vs µm entry error — FLAGGED")

# 3c. "Sem rede" values — keep as NaN-sentinel
n_sem_rede = equip["mesh_opening_spec"].apply(
    lambda x: isinstance(x, str) and "sem" in x.lower()
).sum()
print(f"  'Sem rede' entries: {n_sem_rede}  →  treated as NaN")

# Execute
equip.drop_duplicates(keep="first", inplace=True)
equip["cleaning_flag"] = ""
if eqp_decimals.any():
    equip.loc[eqp_decimals, "cleaning_flag"] = "decimal_mesh_spec"

eqp_after = len(equip)
print(f"  Equipment: {eqp_before} → {eqp_after}  ({eqp_before - eqp_after} duplicates removed)\n")


# ── 4.  MESH ──────────────────────────────────────────────────────────────────
print("── Mesh ──")
msh_before = len(mesh)

# 4a. Convert "-" to NaN in numeric columns
dash_cols = ["calibre_size_3mm", "calibre_size_5mm", "calibre_freq_3mm", "calibre_freq_5mm"]
dash_counts = {}
for col in dash_cols:
    n_dash = (mesh[col] == "-").sum()
    dash_counts[col] = n_dash
    if n_dash > 0:
        mesh[col] = mesh[col].replace("-", np.nan)
        print(f"  '{col}': {n_dash} dashes converted to NaN")

# 4b. Drop exact duplicates
msh_dup_mask = mesh.duplicated(keep="first")
n_msh_dup = msh_dup_mask.sum()
print(f"  Exact duplicate rows: {n_msh_dup}")

# 4c. Validate frequency values — expected set {40, 45, 50}
valid_freqs = {40, 45, 50}
for col in ["calibre_freq_3mm", "calibre_freq_5mm"]:
    vals = pd.to_numeric(mesh[col], errors="coerce")
    suspicious = ~vals.isna() & ~vals.isin(valid_freqs)
    n_sus = suspicious.sum()
    if n_sus > 0:
        print(f"  '{col}': {n_sus} non-standard values {vals[suspicious].unique().tolist()} → FLAGGED")

# Execute
mesh.drop_duplicates(keep="first", inplace=True)

msh_after = len(mesh)
print(f"  Mesh: {msh_before} → {msh_after}  ({msh_before - msh_after} duplicates removed)\n")


# ── 5.  PRODUCT TABLES ────────────────────────────────────────────────────────
product_dfs = {
    "product_01": product_01,
    "product_01_client2": product_01c,
    "product_02": product_02,
}

product_log = []

for pname, pdf in product_dfs.items():
    print(f"── {pname} ──")
    n_before = len(pdf)

    # 5a. Rows that are entirely null in all value columns
    all_null_vals = pdf[PRODUCT_VALUE_COLS].isnull().all(axis=1)
    n_empty_val = all_null_vals.sum()

    # 5b. Rows with nulls in key metadata (test_id, dates)
    has_null_key = pdf[["test_id"]].isnull().any(axis=1)
    n_null_key = has_null_key.sum()

    # 5c. Granulometry sum check
    frac_cols = [c for c in GRANULOMETRY_COLS if c in pdf.columns]
    gran_sum = pdf[frac_cols].sum(axis=1)
    gran_outside = (gran_sum < 99) | (gran_sum > 101)
    n_gran_bad = gran_outside.sum()

    # 5d. Impossible density values
    has_density = "density" in pdf.columns
    n_implaus_density = 0
    if has_density:
        implaus_density = (pdf["density"] > 100) | (pdf["density"] < 1)
        n_implaus_density = int(implaus_density.sum())
        if implaus_density.any():
            print(f"  Implausible density ({n_implaus_density} rows): "
                  f"values {pdf.loc[implaus_density, 'density'].unique().tolist()} → NaN")

    # 5e. Moisture == 0 (impossible in real product)
    if "moisture" in pdf.columns:
        zero_moist = (pdf["moisture"] == 0) & ~pdf["moisture"].isna()
        print(f"  Zero moisture: {zero_moist.sum()} rows → KEPT (flagged)")

    # 5f. Duplicate test_ids
    dup_tid = pdf["test_id"].duplicated(keep=False) & pdf["test_id"].notna()
    n_dup_tid = dup_tid.sum()
    # Show which test_ids are duplicated and across which datasets
    dup_ids = pdf.loc[dup_tid, "test_id"].unique()
    if len(dup_ids) > 0:
        print(f"  Duplicate test_ids found: {len(dup_ids)} IDs ({n_dup_tid} rows across products)")

    print(f"  All-null value rows: {n_empty_val}")
    print(f"  Null test_id rows: {n_null_key}")
    print(f"  Granulometry sum outside [99,101]: {n_gran_bad}/{n_before - n_empty_val} non-null rows")

    # Apply cleaning: set implausible density to NaN
    if has_density and implaus_density.any():
        pdf.loc[implaus_density, "density"] = np.nan

    # Apply: set granulometry where sum is wildly wrong (e.g. sum > 150) to NaN
    # Only if the sum is way off (likely corrupted data)
    gran_wild = gran_sum > 150
    if gran_wild.any():
        pdf.loc[gran_wild, frac_cols] = np.nan
        print(f"  Wild granulometry sum > 150: {gran_wild.sum()} rows → cleared to NaN")

    n_after = len(pdf)
    product_log.append({
        "dataset": pname,
        "rows_before": n_before,
        "rows_after": n_after,
        "all_null_rows": n_empty_val,
        "null_key_rows": n_null_key,
        "gran_sum_outside_99101": int(gran_outside.sum()),
        "gran_sum_outside_pct": round(gran_outside.sum() / max(n_before - n_empty_val, 1) * 100, 2),
        "n_implaus_density": n_implaus_density,
    })

# ── 6.  CROSS-DATASET: shared test_ids ────────────────────────────────────────
# Identify test_ids that appear across multiple product tables
all_prods = {
    "product_01": product_01,
    "product_01_client2": product_01c,
    "product_02": product_02,
}
all_tids = {}
for name, df in all_prods.items():
    tids = set(df["test_id"].dropna().unique())
    all_tids[name] = tids

shared = all_tids["product_01"] & all_tids["product_02"]
shared_all = all_tids["product_01"] & all_tids["product_01_client2"] & all_tids["product_02"]
print(f"\n── Cross-dataset ──")
print(f"  test_ids shared between product_01 & product_02: {len(shared)}")
print(f"  test_ids shared across all 3 product tables: {len(shared_all)}")


# ── 7.  SUMMARY LOG ───────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("CLEANING LOG")
print("=" * 68)

log_entries = []

# Sensors
log_entries.append({
    "dataset": "sensors",
    "issue": "flow_rate_scale_09/11/12 — values ≥65000 (sensor saturation/error code)",
    "action": "Set to NaN",
    "rows_affected": int(n_flow_hi),
})
log_entries.append({
    "dataset": "sensors",
    "issue": "storage098/099_nivel — values ≥65000 (sensor saturation/error code)",
    "action": "Set to NaN",
    "rows_affected": int(n_nivel_hi),
})
log_entries.append({
    "dataset": "sensors",
    "issue": "scale / flow_rate negative values (physically impossible)",
    "action": "Set to NaN",
    "rows_affected": int(n_neg),
})
log_entries.append({
    "dataset": "sensors",
    "issue": "Exact (tag, timestamp, value) duplicates",
    "action": "Dropped rows",
    "rows_affected": int(n_ts_dup),
})
distinct_nanned = sensors.isnull().sum().sum()
log_entries.append({
    "dataset": "sensors",
    "issue": f"Total distinct rows with NaN values after cleaning",
    "action": "Reported",
    "rows_affected": int(distinct_nanned),
})

# Equipment
log_entries.append({
    "dataset": "equipment_separation",
    "issue": "Exact duplicate rows (identical across all columns)",
    "action": "Dropped duplicates (keep=first)",
    "rows_affected": int(n_eqp_dup),
})
log_entries.append({
    "dataset": "equipment_separation",
    "issue": "Decimal mesh_opening_spec — likely mm/µm unit confusion (2.38, 5.37, 8.06)",
    "action": "Flagged (not modified — requires domain confirmation)",
    "rows_affected": int(n_eqp_dec),
})
log_entries.append({
    "dataset": "equipment_separation",
    "issue": "'Sem rede' entries (no mesh installed)",
    "action": "Kept as sentinel (no change needed)",
    "rows_affected": int(n_sem_rede),
})

# Mesh
for col, cnt in dash_counts.items():
    if cnt > 0:
        log_entries.append({
            "dataset": "mesh",
            "issue": f"'{col}' has '-' placeholder values",
            "action": "Converted to NaN",
            "rows_affected": int(cnt),
        })
log_entries.append({
    "dataset": "mesh",
    "issue": "Exact duplicate rows",
    "action": "Dropped duplicates (keep=first)",
    "rows_affected": int(n_msh_dup),
})

# Products
for pl in product_log:
    log_entries.append({
        "dataset": pl["dataset"],
        "issue": f"All-null value rows (empty records)",
        "action": "Kept (may be metadata-only)",
        "rows_affected": int(pl["all_null_rows"]),
    })
    log_entries.append({
        "dataset": pl["dataset"],
        "issue": "Granulometry sum outside [99, 101]",
        "action": "Flagged (kept as-is except wild >150)",
        "rows_affected": int(pl["gran_sum_outside_99101"]),
    })
    if pl["n_implaus_density"] > 0:
        log_entries.append({
            "dataset": pl["dataset"],
            "issue": "Implausible density values (< 1 or > 100)",
            "action": "Set to NaN",
            "rows_affected": pl["n_implaus_density"],
        })

log_df = pd.DataFrame(log_entries)
log_df.to_csv(TABLES / "cleaning_log.csv", index=False)
print(log_df.to_string(index=False))

# ── 8.  MISSING DATA REPORT ──────────────────────────────────────────────────
print("\n" + "=" * 68)
print("MISSING DATA REPORT (post-cleaning)")
print("=" * 68)

all_datasets = {
    "sensors": sensors,
    "product_01": product_01,
    "product_01_client2": product_01c,
    "product_02": product_02,
    "equipment_separation": equip,
    "mesh": mesh,
}
missing_frames = []
for label, df in all_datasets.items():
    m = missing_report(df, label)
    if len(m) > 0:
        missing_frames.append(m)
missing_all = pd.concat(missing_frames, ignore_index=True)
missing_all.to_csv(TABLES / "missing_data_post_cleaning.csv", index=False)
print(missing_all.to_string(index=False))


# ── 9.  SAVE CLEANED DATASETS ────────────────────────────────────────────────
print("\n" + "=" * 68)
print("SAVING CLEANED DATASETS")
print("=" * 68)

save(sensors, "sensors", "cleaned")
save(product_01, "product_01", "cleaned")
save(product_01c, "product_01_client2", "cleaned")
save(product_02, "product_02", "cleaned")
save(equip, "equipment_separation", "cleaned")
save(mesh, "mesh", "cleaned")

print("\n" + "=" * 68)
print("Phase 2 complete.")
print("=" * 68)
