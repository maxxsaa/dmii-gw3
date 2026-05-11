"""
Phase 4 — Feature Engineering
==============================
AgroMill Corp — Data Mining II (GW3)

Tasks:
  1. Time features (hour, weekday, shift, elapsed since maintenance)
  2. Sensor aggregation features (rolling stats per tag group)
  3. Process consistency metrics (mass balance, stability)
  4. Product-specific features for quality modeling
  5. Feature matrix documentation

Outputs:
  - data/processed/feature_matrix_quality_p01.parquet   (Quality Score)
  - data/processed/feature_matrix_throughput.parquet     (Throughput)
  - outputs/tables/feature_dictionary.md
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
TABLES = Path("outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)

TZ_LISBON = "Europe/Lisbon"

# ── Tag group mapping (same as Phase 3) ───────────────────────────────────────
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
for grp, tags in SRC_TAG_GROUPS.items():
    for t in tags:
        TAG_GROUP_LOOKUP[t] = grp

ALL_FLOW_TAGS = [f"flow_rate_scale_{i:02d}" for i in range(1, 14)]
ALL_SCALE_TAGS = [f"scale_{i:02d}" for i in range(1, 14)]


# ── 0. LOAD DATA ───────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 4 — Feature Engineering")
print("=" * 60)

print("\nLoading cleaned datasets ...")
sensors     = pd.read_parquet(PROCESSED / "sensors_cleaned.parquet")
product_01  = pd.read_parquet(PROCESSED / "product_01_cleaned.parquet")
product_01c = pd.read_parquet(PROCESSED / "product_01_client2_cleaned.parquet")
product_02  = pd.read_parquet(PROCESSED / "product_02_cleaned.parquet")
equip       = pd.read_parquet(PROCESSED / "equipment_separation_cleaned.parquet")
mesh        = pd.read_parquet(PROCESSED / "mesh_cleaned.parquet")
print("  Done.")

# ── 1. PRE-COMPUTE HOURLY SENSOR AGGREGATES ──────────────────────────────────
print("\n── Pre-computing hourly sensor aggregates ──")

# Assign tag group
sensors["tag_group"] = sensors["tag"].map(TAG_GROUP_LOOKUP).fillna("other")
# Local time
sensors["datetime_local"] = sensors["datetime"].dt.tz_convert(TZ_LISBON)

# Hourly aggregation per tag group
sensors["hour_bin"] = sensors["datetime_local"].dt.floor("h")
hourly_agg = (
    sensors.groupby(["hour_bin", "tag_group"])["value"]
    .agg(["mean", "std", "min", "max", "median", "count"])
    .reset_index()
)
hourly_agg.columns = ["hour_bin", "tag_group", "val_mean", "val_std", "val_min", "val_max", "val_median", "val_count"]

# Also compute total flow and total scale per hour
sensors["tag"] = sensors["tag"].astype(str)
def sum_group(tag_list, name):
    mask = sensors["tag"].isin(tag_list)
    grp = sensors[mask].groupby("hour_bin")["value"].sum().reset_index()
    grp.columns = ["hour_bin", name]
    return grp

hourly_flow_sum = sum_group(ALL_FLOW_TAGS, "total_flow_rate")
hourly_scale_sum = sum_group(ALL_SCALE_TAGS, "total_scale")

# Merge totals into hourly_agg (broadcast to each tag_group row — will deduplicate later)
hourly_agg = hourly_agg.merge(hourly_flow_sum, on="hour_bin", how="left")
hourly_agg = hourly_agg.merge(hourly_scale_sum, on="hour_bin", how="left")

print(f"  Hourly aggregates: {len(hourly_agg)} rows ({hourly_agg['hour_bin'].nunique()} hours)")

# ── 2. PRODUCT-LEVEL FEATURES (for Quality Score) ────────────────────────────
print("\n── Building Quality Score feature matrix (Product 01) ──")

def build_quality_features(pdf, label, hourly_agg, mesh, equip):
    """Build feature matrix for quality prediction from a product DataFrame."""
    rows = pdf.dropna(subset=["test_date"]).copy()
    if len(rows) == 0:
        return pd.DataFrame()

    # 2a. Time features
    rows["test_hour"] = rows["test_date"].dt.tz_convert(TZ_LISBON).dt.hour
    rows["test_weekday"] = rows["test_date"].dt.tz_convert(TZ_LISBON).dt.weekday
    rows["test_month"] = rows["test_date"].dt.tz_convert(TZ_LISBON).dt.month
    rows["test_is_weekend"] = rows["test_weekday"].isin([5, 6]).astype(int)
    rows["test_shift"] = rows["test_hour"].apply(
        lambda h: "Morning" if 6 <= h < 14 else ("Afternoon" if 14 <= h < 22 else "Night")
    )

    # 2b. Sensor features — window aggregates before each test
    test_times = rows["test_date"].values  # numpy array of Timestamps
    window_hours = [1, 4, 8, 24]

    # For efficiency: convert hourly_agg to dict-of-arrays by tag_group
    tag_groups = hourly_agg["tag_group"].unique()
    agg_by_group = {}
    for grp in tag_groups:
        gdata = hourly_agg[hourly_agg["tag_group"] == grp].set_index("hour_bin")
        agg_by_group[grp] = gdata

    window_feats = []
    for _, row in rows.iterrows():
        t = row["test_date"]
        feats = {}
        for wh in window_hours:
            t_start = t - pd.Timedelta(hours=wh)
            # Filter hourly aggregates to this window
            for grp in tag_groups:
                gdata = agg_by_group[grp]
                mask = (gdata.index >= t_start) & (gdata.index < t)
                window_data = gdata[mask]
                if len(window_data) == 0:
                    continue
                prefix = f"sns_{grp}_win{wh}h"
                feats[f"{prefix}_mean"] = window_data["val_mean"].mean()
                feats[f"{prefix}_std"] = window_data["val_mean"].std()
                feats[f"{prefix}_min"] = window_data["val_min"].min()
                feats[f"{prefix}_max"] = window_data["val_max"].max()
                feats[f"{prefix}_count"] = window_data["val_count"].sum()
            # Total flow / scale for this window
            flow_win = hourly_flow_sum[
                (hourly_flow_sum["hour_bin"] >= t_start) & (hourly_flow_sum["hour_bin"] < t)
            ]
            scale_win = hourly_scale_sum[
                (hourly_scale_sum["hour_bin"] >= t_start) & (hourly_scale_sum["hour_bin"] < t)
            ]
            if len(flow_win) > 0:
                feats[f"total_flow_win{wh}h"] = flow_win["total_flow_rate"].sum()
            if len(scale_win) > 0:
                feats[f"total_scale_win{wh}h"] = scale_win["total_scale"].sum()
            if len(flow_win) > 0 and len(scale_win) > 0 and scale_win["total_scale"].sum() > 0:
                feats[f"flow_scale_ratio_win{wh}h"] = (
                    flow_win["total_flow_rate"].sum() / scale_win["total_scale"].sum()
                )
        window_feats.append(feats)

    win_df = pd.DataFrame(window_feats, index=rows.index)
    rows = pd.concat([rows, win_df], axis=1)

    # 2c. Mesh features (join by date)
    mesh_local = mesh.copy()
    mesh_local["mesh_date"] = pd.to_datetime(mesh_local["date"]).dt.date
    mesh_num = ["calibre_size_3mm", "calibre_size_5mm", "calibre_freq_3mm", "calibre_freq_5mm"]
    for col in mesh_num:
        mesh_local[col] = pd.to_numeric(mesh_local[col], errors="coerce")

    rows["test_date_only"] = rows["test_date"].dt.tz_convert(TZ_LISBON).dt.date
    rows = rows.merge(
        mesh_local[["mesh_date"] + mesh_num].drop_duplicates("mesh_date"),
        left_on="test_date_only", right_on="mesh_date", how="left"
    )
    rows.drop(columns=["mesh_date"], inplace=True, errors="ignore")

    # 2d. Elapsed since last maintenance
    maint_dates = equip["component_maintenance_date"].dropna().unique()
    maint_dates = sorted(pd.to_datetime(maint_dates))
    if len(maint_dates) > 0:
        # Convert to Unix timestamps (seconds) for tz-naive comparison
        maint_epoch = np.array([md.timestamp() for md in pd.DatetimeIndex(maint_dates).tz_localize(TZ_LISBON)])
        test_lisbon = rows["test_date"].dt.tz_convert(TZ_LISBON)
        test_epoch = test_lisbon.apply(lambda x: x.timestamp()).values.astype(float)
        hours_since = np.full(len(rows), np.nan)
        for i, te in enumerate(test_epoch):
            prior = maint_epoch[maint_epoch <= te]
            if len(prior) > 0:
                hours_since[i] = (te - prior[-1]) / 3600
        rows["last_maint_hours"] = hours_since

    # 2e. Detail_type and origin encoding
    rows["detail_type"] = rows["detail_type"].fillna("UNKNOWN")

    # 2f. Process consistency: granulometry sum deviation
    frac_cols = [c for c in ["fraction_8_236", "fraction_10_200", "fraction_14_140",
                              "fraction_18_100", "fraction_25_071", "under_000"] if c in rows.columns]
    rows["gran_sum"] = rows[frac_cols].sum(axis=1)
    rows["gran_deviation"] = (rows["gran_sum"] - 100).abs()

    # 2g. Interaction features
    for wh in window_hours:
        f_mean = f"sns_consumo_win{wh}h_mean"
        f_std = f"sns_consumo_win{wh}h_std"
        if f_mean in rows.columns and f_std in rows.columns:
            rows[f"consumo_cv_win{wh}h"] = rows[f_std] / rows[f_mean].replace(0, np.nan)

    return rows


print("  Building feature matrix for product_01 ...")
fmat_p01 = build_quality_features(product_01, "product_01", hourly_agg, mesh, equip)
print(f"  Feature matrix: {fmat_p01.shape[0]} rows × {fmat_p01.shape[1]} cols")

print("  Building feature matrix for product_01_client2 ...")
fmat_p01c = build_quality_features(product_01c, "product_01_client2", hourly_agg, mesh, equip)
print(f"  Feature matrix: {fmat_p01c.shape[0]} rows × {fmat_p01c.shape[1]} cols")

print("  Building feature matrix for product_02 ...")
fmat_p02 = build_quality_features(product_02, "product_02", hourly_agg, mesh, equip)
print(f"  Feature matrix: {fmat_p02.shape[0]} rows × {fmat_p02.shape[1]} cols")


# ── 3. THROUGHPUT FEATURE MATRIX ──────────────────────────────────────────────
print("\n── Building Throughput feature matrix ──")

def build_throughput_features(hourly_agg, sensors, equip, mesh):
    """Build feature matrix for throughput estimation at hourly level."""
    # One row per hour
    hours = pd.DataFrame({"hour_bin": sorted(hourly_agg["hour_bin"].unique())})
    hours["hour"] = hours["hour_bin"].dt.hour
    hours["weekday"] = hours["hour_bin"].dt.weekday
    hours["month"] = hours["hour_bin"].dt.month
    hours["is_weekend"] = hours["weekday"].isin([5, 6]).astype(int)
    hours["shift"] = hours["hour"].apply(
        lambda h: "Morning" if 6 <= h < 14 else ("Afternoon" if 14 <= h < 22 else "Night")
    )

    # Sensor features per hour (pivot tag_groups wide)
    for grp in hourly_agg["tag_group"].unique():
        gdata = hourly_agg[hourly_agg["tag_group"] == grp][
            ["hour_bin", "val_mean", "val_std", "val_min", "val_max", "val_count"]
        ].copy()
        gdata.columns = ["hour_bin"] + [f"sns_{grp}_{c}" for c in ["mean", "std", "min", "max", "count"]]
        hours = hours.merge(gdata, on="hour_bin", how="left")

    # Total flow and scale
    hours = hours.merge(hourly_flow_sum, on="hour_bin", how="left")
    hours = hours.merge(hourly_scale_sum, on="hour_bin", how="left")

    # Lag features for total_flow_rate
    hours = hours.sort_values("hour_bin")
    for lag in [1, 2, 3, 6, 12, 24]:
        hours[f"total_flow_lag{lag}h"] = hours["total_flow_rate"].shift(lag)
        hours[f"total_scale_lag{lag}h"] = hours["total_scale"].shift(lag)

    # Rolling features for total_flow_rate
    for w in [3, 6, 12, 24]:
        hours[f"total_flow_roll{w}h_mean"] = hours["total_flow_rate"].rolling(w, min_periods=1).mean()
        hours[f"total_flow_roll{w}h_std"] = hours["total_flow_rate"].rolling(w, min_periods=1).std()
        hours[f"total_scale_roll{w}h_mean"] = hours["total_scale"].rolling(w, min_periods=1).mean()

    # Elapsed since last maintenance
    maint_dates = equip["component_maintenance_date"].dropna().unique()
    maint_dates = pd.to_datetime(sorted(maint_dates))
    if len(maint_dates) > 0:
        # Localize maintenance dates to Lisbon
        maint_local = maint_dates.tz_localize(TZ_LISBON)
        # For each hour, find most recent maintenance before it
        def hours_since_maint(ts):
            prior = maint_local[maint_local <= ts]
            if len(prior) == 0:
                return np.nan
            return (ts - prior[-1]).total_seconds() / 3600

        hours["last_maint_hours"] = hours["hour_bin"].apply(hours_since_maint)

    # Target: total_flow_rate (throughput proxy)
    # Also individual flow_rate columns for multi-target
    return hours


fmat_throughput = build_throughput_features(hourly_agg, sensors, equip, mesh)
print(f"  Throughput feature matrix: {fmat_throughput.shape[0]} rows × {fmat_throughput.shape[1]} cols")

# ── 4. SAVE FEATURE MATRICES ─────────────────────────────────────────────────
print("\n── Saving feature matrices ──")

fmat_p01.to_parquet(PROCESSED / "feature_matrix_quality_p01.parquet", index=False)
print(f"  feature_matrix_quality_p01.parquet  ({len(fmat_p01)} rows)")

fmat_p01c.to_parquet(PROCESSED / "feature_matrix_quality_p01c.parquet", index=False)
print(f"  feature_matrix_quality_p01c.parquet  ({len(fmat_p01c)} rows)")

fmat_p02.to_parquet(PROCESSED / "feature_matrix_quality_p02.parquet", index=False)
print(f"  feature_matrix_quality_p02.parquet  ({len(fmat_p02)} rows)")

fmat_throughput.to_parquet(PROCESSED / "feature_matrix_throughput.parquet", index=False)
print(f"  feature_matrix_throughput.parquet  ({len(fmat_throughput)} rows)")

# ── 5. FEATURE DICTIONARY ────────────────────────────────────────────────────
print("\n── Writing feature dictionary ──")

feature_doc = """
# Feature Dictionary — AgroMill Corp

## 1. Quality Score Feature Matrix (product_01 / product_01_client2 / product_02)

### Target Columns (from product tables)
| Feature | Type | Description |
|---------|------|-------------|
| density | continuous | Density measurement (kg/m³) |
| moisture | continuous | Moisture content (%) |
| fraction_8_236 | continuous | Particle fraction retained on 8 mesh (2.36 mm) (%) |
| fraction_10_200 | continuous | Particle fraction retained on 10 mesh (2.00 mm) (%) |
| fraction_14_140 | continuous | Particle fraction retained on 14 mesh (1.40 mm) (%) |
| fraction_18_100 | continuous | Particle fraction retained on 18 mesh (1.00 mm) (%) |
| fraction_25_071 | continuous | Particle fraction retained on 25 mesh (0.71 mm) (%) |
| under_000 | continuous | Particles passing through finest mesh (%) |

### Time Features
| Feature | Type | Description |
|---------|------|-------------|
| test_hour | int | Hour of test (Lisbon local time, 0-23) |
| test_weekday | int | Day of week (0=Mon, 6=Sun) |
| test_month | int | Month (1-12) |
| test_is_weekend | int | 1 if weekend, else 0 |
| test_shift | str | Shift: Morning/Afternoon/Night |

### Sensor Features (per window)
Pattern: `sns_{tag_group}_win{WH}h_{stat}` where WH ∈ {1, 4, 8, 24}

| Stat | Description |
|------|-------------|
| mean | Mean of hourly means in window |
| std | Std of hourly means in window |
| min | Minimum of hourly mins in window |
| max | Maximum of hourly maxes in window |
| count | Total sensor readings in window |

Tag groups: scale, flow_rate, consumo, sp_vel, pv_vel, corrente, nivel

### Total Flow / Scale Features
| Feature | Description |
|---------|-------------|
| total_flow_win{WH}h | Sum of all flow_rate readings in window |
| total_scale_win{WH}h | Sum of all scale readings in window |
| flow_scale_ratio_win{WH}h | Ratio of total flow to total scale in window |

### Mesh Features
| Feature | Type | Description |
|---------|------|-------------|
| calibre_size_3mm | int | Mesh calibre size for 3mm sieves |
| calibre_size_5mm | int | Mesh calibre size for 5mm sieves |
| calibre_freq_3mm | int | Vibration frequency for 3mm sieves (Hz) |
| calibre_freq_5mm | int | Vibration frequency for 5mm sieves (Hz) |

### Other Features
| Feature | Type | Description |
|---------|------|-------------|
| last_maint_hours | continuous | Hours since most recent equipment maintenance |
| detail_type | str | LABS / OUTRO / ENSAIO |
| origin | str | Production area identifier |
| gran_sum | continuous | Sum of all granulometry fractions |
| gran_deviation | continuous | |gran_sum - 100| |
| consumo_cv_win{WH}h | continuous | Coefficient of variation of consumo in window |
| test_date_only | date | Date of test (for reference) |

---

## 2. Throughput Feature Matrix

### Target
| Feature | Type | Description |
|---------|------|-------------|
| total_flow_rate | continuous (target) | Sum of all flow_rate_scale tags (hourly) |

### Time Features
| Feature | Description |
|---------|-------------|
| hour | Hour of day (0-23, Lisbon local) |
| weekday | Day of week |
| month | Month |
| is_weekend | Weekend flag |
| shift | Morning / Afternoon / Night |

### Sensor Features (hourly level)
Pattern: `sns_{tag_group}_{stat}` where stat ∈ {mean, std, min, max, count}

| Feature | Description |
|---------|-------------|
| total_scale | Sum of all scale readings in hour |
| total_flow_rate | Sum of all flow_rate readings in hour |

### Lag Features
| Feature | Description |
|---------|-------------|
| total_flow_lag{WH}h | Total flow N hours before |
| total_scale_lag{WH}h | Total scale N hours before |

Lags: 1, 2, 3, 6, 12, 24 hours

### Rolling Features
| Feature | Description |
|---------|-------------|
| total_flow_roll{Wh}_mean | Rolling mean of total flow over N hours |
| total_flow_roll{Wh}_std | Rolling std of total flow over N hours |
| total_scale_roll{Wh}_mean | Rolling mean of total scale over N hours |

Windows: 3, 6, 12, 24 hours

### Maintenance
| Feature | Description |
|---------|-------------|
| last_maint_hours | Hours since most recent equipment maintenance |
"""

with open(TABLES / "feature_dictionary.md", "w") as f:
    f.write(feature_doc.strip())
print("  feature_dictionary.md")

# ── 6. SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 4 complete.")
print("=" * 60)
print(f"  Quality matrices: {fmat_p01.shape[0]} (p01) / {fmat_p01c.shape[0]} (p01c) / {fmat_p02.shape[0]} (p02)")
print(f"  Throughput matrix: {fmat_throughput.shape[0]} rows")
print(f"  Feature columns (quality): {fmat_p01.shape[1]}")
print(f"  Feature columns (throughput): {fmat_throughput.shape[1]}")
