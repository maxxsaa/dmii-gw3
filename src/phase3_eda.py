"""
Phase 3 — Exploratory Data Analysis
=====================================
AgroMill Corp — Data Mining II (GW3)

Tasks:
  1. Descriptive statistics by product, shift, and period
  2. Distribution plots (moisture, density, granulometry)
  3. Correlation analysis + multicollinearity (VIF)
  4. Time-based trend and drift analysis
  5. Operational bottleneck exploration (shifts, reset, maintenance, mesh)
  6. Initial hypotheses list

Outputs to: outputs/figures/, outputs/tables/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
FIGURES = Path("outputs/figures")
TABLES = Path("outputs/tables")
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load cleaned data ─────────────────────────────────────────────────────────
print("Loading cleaned datasets ...")
sensors     = pd.read_parquet(PROCESSED / "sensors_cleaned.parquet")
product_01  = pd.read_parquet(PROCESSED / "product_01_cleaned.parquet")
product_01c = pd.read_parquet(PROCESSED / "product_01_client2_cleaned.parquet")
product_02  = pd.read_parquet(PROCESSED / "product_02_cleaned.parquet")
equip       = pd.read_parquet(PROCESSED / "equipment_separation_cleaned.parquet")
mesh        = pd.read_parquet(PROCESSED / "mesh_cleaned.parquet")
print("  Done.")

# ── Shift assignment ──────────────────────────────────────────────────────────
TZ_LISBON = "Europe/Lisbon"

def assign_shift(dt_local):
    """Return shift name given a timezone-aware local datetime."""
    h = dt_local.hour
    if 6 <= h < 14:
        return "Morning"
    elif 14 <= h < 22:
        return "Afternoon"
    else:
        return "Night"

# Sensor shift assignment (22M rows → vectorized)
print("Assigning shifts ...")
sensors["datetime_local"] = sensors["datetime"].dt.tz_convert(TZ_LISBON)
sensors["shift"] = sensors["datetime_local"].apply(assign_shift)

# Product shift assignment
for df, name in [(product_01, "product_01"), (product_01c, "product_01c"), (product_02, "product_02")]:
    if "test_date" in df.columns and df["test_date"].notna().any():
        df["test_date_local"] = df["test_date"].dt.tz_convert(TZ_LISBON)
        df["shift"] = df["test_date_local"].apply(assign_shift)
print("  Done.")

# ── Tag group mapping ─────────────────────────────────────────────────────────
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
sensors["tag_group"] = sensors["tag"].map(TAG_GROUP_LOOKUP).fillna("other")

GRAN_COLS = [
    "fraction_8_236", "fraction_10_200", "fraction_14_140",
    "fraction_18_100", "fraction_25_071", "under_000",
]
PROD_VAL_COLS = ["density", "moisture"] + GRAN_COLS

# ── 1. DESCRIPTIVE STATISTICS ─────────────────────────────────────────────────
print("\n═══ 1. Descriptive statistics ═══\n")

# Helper: styled tex table
def save_descriptive(df, label, value_cols):
    desc = df[value_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    desc.to_csv(TABLES / f"desc_{label}.csv")
    print(f"  ✓  desc_{label}.csv")

print("  Product_01 ...")
save_descriptive(product_01, "product_01", PROD_VAL_COLS)
print("  Product_01_client2 ...")
save_descriptive(product_01c, "product_01_client2", PROD_VAL_COLS)
print("  Product_02 ...")
save_descriptive(product_02, "product_02", PROD_VAL_COLS)

# Descriptive by shift
def desc_by_shift(df, label):
    if "shift" not in df.columns or df["shift"].isna().all():
        return
    gb = df.groupby("shift")[PROD_VAL_COLS].describe().round(2)
    gb.to_csv(TABLES / f"desc_{label}_by_shift.csv")
    print(f"  ✓  desc_{label}_by_shift.csv")

print("  By shift ...")
desc_by_shift(product_01, "product_01")
desc_by_shift(product_01c, "product_01_client2")
desc_by_shift(product_02, "product_02")

# ── 2. DISTRIBUTION PLOTS ──────────────────────────────────────────────────
print("\n═══ 2. Distribution plots ═══\n")

DIST_COLORS = {"Morning": "#3498db", "Afternoon": "#e67e22", "Night": "#2c3e50"}

def plot_distributions(df, label):
    n_cols = len(PROD_VAL_COLS)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(PROD_VAL_COLS):
        if i >= 9:
            break
        ax = axes[i]
        data = df[col].dropna()
        if len(data) == 0:
            ax.set_title(f"{col} (no data)")
            continue
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.7, density=True)
        ax.set_title(col)
        ax.set_ylabel("Density")
    # KDE overlay on a separate fig
    fig2, axes2 = plt.subplots(3, 3, figsize=(14, 10))
    axes2 = axes2.flatten()
    for i, col in enumerate(PROD_VAL_COLS):
        if i >= 9:
            break
        ax = axes2[i]
        data = df[col].dropna()
        if len(data) == 0:
            ax.set_title(f"{col} (no data)")
            continue
        sns.kdeplot(data, ax=ax, fill=True, color="steelblue")
        ax.set_title(col)
    plt.tight_layout()
    fig2.savefig(FIGURES / f"dist_kde_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # By-shift boxplots
    if "shift" in df.columns and df["shift"].notna().any():
        fig3, axes3 = plt.subplots(3, 3, figsize=(14, 10))
        axes3 = axes3.flatten()
        for i, col in enumerate(PROD_VAL_COLS):
            if i >= 9:
                break
            ax = axes3[i]
            data = df[["shift", col]].dropna()
            if len(data) == 0:
                ax.set_title(f"{col} (no data)")
                continue
            sns.boxplot(x="shift", y=col, data=data, ax=ax,
                        palette=DIST_COLORS, order=["Morning", "Afternoon", "Night"])
            ax.set_title(col)
        plt.tight_layout()
        fig3.savefig(FIGURES / f"boxplot_{label}_by_shift.png", dpi=120, bbox_inches="tight")
        plt.close(fig3)

    # Clean up first figure
    plt.tight_layout()
    fig.savefig(FIGURES / f"dist_hist_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  dist/hist/KDE/boxplot for {label}")

plot_distributions(product_01, "product_01")
plot_distributions(product_01c, "product_01_client2")
plot_distributions(product_02, "product_02")

# ── 3. CORRELATION + VIF ───────────────────────────────────────────────────
print("\n═══ 3. Correlation & multicollinearity ═══\n")

def correlation_analysis(df, label):
    corr = df[PROD_VAL_COLS].corr()
    corr.to_csv(TABLES / f"corr_{label}.csv")
    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation Matrix — {label}")
    plt.tight_layout()
    fig.savefig(FIGURES / f"corr_heatmap_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    # VIF
    clean = df[PROD_VAL_COLS].dropna()
    if len(clean) > 50 and clean.shape[1] > 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean)
        vif_data = pd.DataFrame()
        vif_data["feature"] = PROD_VAL_COLS
        vif_data["VIF"] = [
            variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])
        ]
        vif_data.to_csv(TABLES / f"vif_{label}.csv", index=False)
        print(f"  ✓  VIF table for {label}")
    print(f"  ✓  Corr heatmap + table for {label}")

correlation_analysis(product_01, "product_01")
correlation_analysis(product_01c, "product_01_client2")
correlation_analysis(product_02, "product_02")

# ── 4. TIME-BASED TRENDS ──────────────────────────────────────────────────
print("\n═══ 4. Time-based trend analysis ═══\n")

def time_trend_product(df, label):
    if "test_date" not in df.columns:
        return
    ts = df.dropna(subset=["test_date"]).copy()
    ts["date"] = ts["test_date"].dt.date
    daily = ts.groupby("date")[PROD_VAL_COLS].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Line plot
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(PROD_VAL_COLS):
        if i >= 9:
            break
        ax = axes[i]
        ax.plot(daily["date"], daily[col], color="steelblue", linewidth=0.8)
        # 7-day rolling mean
        if len(daily) > 7:
            roll = daily[col].rolling(7, center=True).mean()
            ax.plot(daily["date"], roll, color="crimson", linewidth=1.5, label="7d MA")
        ax.set_title(col)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if i == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES / f"trend_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # CDF of time gaps between consecutive tests
    gaps = ts.sort_values("test_date")["test_date"].diff().dropna()
    if len(gaps) > 5:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(gaps.dt.total_seconds() / 3600, bins=50, color="steelblue", edgecolor="white")
        ax2.set_xlabel("Hours between consecutive tests")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Test interval distribution — {label}")
        plt.tight_layout()
        fig2.savefig(FIGURES / f"test_intervals_{label}.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)

    print(f"  ✓  Time trends for {label}")

time_trend_product(product_01, "product_01")
time_trend_product(product_01c, "product_01_client2")
time_trend_product(product_02, "product_02")

# Sensor time trends (aggregated)
print("  Aggregating sensor data for time trends ...")
# Sample for speed: daily mean per tag group
sensors_daily = (
    sensors.groupby([pd.Grouper(key="datetime_local", freq="D"), "tag_group"])["value"]
    .mean()
    .reset_index()
)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, (grp, grp_data) in enumerate(sensors_daily.groupby("tag_group")):
    if i >= 9:
        break
    ax = axes[i]
    grp_data = grp_data.sort_values("datetime_local")
    ax.plot(grp_data["datetime_local"], grp_data["value"], linewidth=0.7, alpha=0.7)
    ax.set_title(grp)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.tight_layout()
fig.savefig(FIGURES / "sensor_daily_trends.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  ✓  Sensor daily trends")

# ── 5. BOTTLENECK ANALYSIS ────────────────────────────────────────────────
print("\n═══ 5. Operational bottleneck exploration ═══\n")

# 5a. Shift effects on sensor metrics
print("  5a. Sensor metrics by shift ...")
sns_shift = sensors.groupby(["tag_group", "shift"])["value"].agg(["mean", "std", "count"]).reset_index()
sns_shift.to_csv(TABLES / "sensor_stats_by_shift.csv", index=False)
print("  ✓  Sensor stats by shift")

# Boxplot of key tag groups by shift
key_groups = ["scale", "flow_rate", "consumo", "corrente"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, grp in enumerate(key_groups):
    ax = axes[i]
    data = sensors[sensors["tag_group"] == grp].copy()
    # Sample to 50k per group for plot
    if len(data) > 50000:
        data = data.sample(50000, random_state=42)
    sns.boxplot(x="shift", y="value", data=data, ax=ax,
                palette=DIST_COLORS, order=["Morning", "Afternoon", "Night"])
    ax.set_title(f"{grp} by shift")
    ax.set_ylabel("Value")
plt.tight_layout()
fig.savefig(FIGURES / "sensor_by_shift_boxplot.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 5b. Within-shift sensor patterns (reset effect)
print("  5b. Shift reset patterns (hourly profile) ...")
sensors["hour_local"] = sensors["datetime_local"].dt.hour
hourly_profile = (
    sensors.groupby(["tag_group", "shift", "hour_local"])["value"]
    .mean()
    .reset_index()
)
# Plot hourly profile for each tag group, faceted by shift
for grp in ["scale", "flow_rate", "consumo", "corrente", "nivel"]:
    fig, ax = plt.subplots(figsize=(8, 4))
    grp_data = hourly_profile[hourly_profile["tag_group"] == grp]
    for shift_name in ["Morning", "Afternoon", "Night"]:
        sdata = grp_data[grp_data["shift"] == shift_name]
        if len(sdata) == 0:
            continue
        ax.plot(sdata["hour_local"], sdata["value"], marker=".", label=shift_name, alpha=0.8)
    ax.set_title(f"{grp} — Hourly mean by shift (local time)")
    ax.set_xlabel("Hour of day (local)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES / f"hourly_profile_{grp}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
print("  ✓  Hourly profiles")

# 5c. Mesh configuration vs granulometry
print("  5c. Mesh vs granulometry ...")
# Merge product_01 (test_date date) with mesh (date)
p1 = product_01.dropna(subset=["test_date", "shift"]).copy()
p1["mesh_date"] = p1["test_date"].dt.date
mesh["mesh_date"] = pd.to_datetime(mesh["date"]).dt.date
p1_mesh = p1.merge(mesh, on="mesh_date", how="left")

if len(p1_mesh) > 0:
    mesh_num_cols = ["calibre_size_3mm", "calibre_size_5mm", "calibre_freq_3mm", "calibre_freq_5mm"]
    for mc in mesh_num_cols:
        p1_mesh[mc] = pd.to_numeric(p1_mesh[mc], errors="coerce")

    # How calibre settings affect granulometry
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, frac in enumerate(GRAN_COLS):
        if i >= 6:
            break
        ax = axes[i]
        # Group by calibre_size_3mm
        means = p1_mesh.groupby("calibre_size_3mm")[frac].mean().dropna()
        if len(means) > 0:
            means.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"{frac} by calibre_size_3mm")
        ax.set_xlabel("calibre_size_3mm")
    plt.tight_layout()
    fig.savefig(FIGURES / "mesh_vs_granulometry.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Frequency effect on key fractions
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    axes2 = axes2.flatten()
    for i, frac in enumerate(GRAN_COLS):
        if i >= 6:
            break
        ax = axes2[i]
        means = p1_mesh.groupby("calibre_freq_3mm")[frac].mean().dropna()
        if len(means) > 0:
            means.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"{frac} by calibre_freq_3mm")
        ax.set_xlabel("calibre_freq_3mm")
    plt.tight_layout()
    fig2.savefig(FIGURES / "mesh_freq_vs_granulometry.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)
print("  ✓  Mesh vs granulometry plots")

# 5d. Density / moisture by shift (statistical test)
print("  5d. ANOVA: quality metrics by shift ...")
anova_results = []
for metric in ["density", "moisture"]:
    for pname, pdf in [("product_01", product_01), ("product_01_client2", product_01c), ("product_02", product_02)]:
        data = pdf.dropna(subset=[metric, "shift"])
        groups = [g[metric].values for _, g in data.groupby("shift")]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            anova_results.append({
                "product": pname,
                "metric": metric,
                "F_stat": round(f_stat, 3),
                "p_value": round(p_val, 5),
            })
anova_df = pd.DataFrame(anova_results)
anova_df.to_csv(TABLES / "anova_shift_quality.csv", index=False)
print("  ✓  ANOVA table")
print(anova_df.to_string(index=False))

# 5e. Maintenance impact: before/after maintenance dates
print("  5e. Maintenance impact on sensors ...")
# Use equipment_separation maintenance dates as cut points
maint_dates = equip["component_maintenance_date"].dropna().unique()
maint_dates = pd.to_datetime(sorted(maint_dates))
# Maintenance dates are timezone-naive; localize to Europe/Lisbon for comparison
maint_dates = maint_dates.tz_localize(TZ_LISBON)
if len(maint_dates) > 0:
    # For each maintenance date, look at sensor readings 3 days before vs 3 days after
    maint_effects = []
    for md in maint_dates:
        md_begin = md - pd.Timedelta(days=3)
        md_end = md + pd.Timedelta(days=3)
        before = sensors[(sensors["datetime_local"] >= md_begin)
                        & (sensors["datetime_local"] < md)]
        after = sensors[(sensors["datetime_local"] >= md)
                       & (sensors["datetime_local"] < md_end)]
        if len(before) > 0 and len(after) > 0:
            for grp in ["scale", "flow_rate", "consumo", "corrente"]:
                b = before[before["tag_group"] == grp]["value"].mean()
                a = after[after["tag_group"] == grp]["value"].mean()
                if not (pd.isna(b) or pd.isna(a)):
                    maint_effects.append({
                        "maint_date": md.date(),
                        "tag_group": grp,
                        "mean_before": round(b, 2),
                        "mean_after": round(a, 2),
                        "delta_pct": round((a - b) / b * 100, 2) if b != 0 else None,
                    })
    if maint_effects:
        me_df = pd.DataFrame(maint_effects)
        me_df.to_csv(TABLES / "maintenance_impact_on_sensors.csv", index=False)
        print(f"  ✓  Maintenance impact table ({len(me_df)} entries)")
    else:
        print("  (no maintenance events with sufficient sensor data)")
else:
    print("  (no maintenance dates found)")

# 5f. Mass balance: scale vs flow_rate
print("  5f. Mass balance scatter (scales vs flow_rates) ...")
# Join scale and flow_rate readings by proximity (same timestamp ± small window)
scales = sensors[sensors["tag_group"] == "scale"].copy()
flows = sensors[sensors["tag_group"] == "flow_rate"].copy()
# Pivot to wide: one row per timestamp with scale_01..13 and flow_rate_01..13
if len(scales) > 0 and len(flows) > 0:
    scale_pivot = scales.pivot_table(index="datetime", columns="tag", values="value", aggfunc="mean")
    flow_pivot = flows.pivot_table(index="datetime", columns="tag", values="value", aggfunc="mean")
    # Find common timestamps
    common_ts = scale_pivot.index.intersection(flow_pivot.index)
    if len(common_ts) > 0:
        # Sum across all scales and all flow rates at each common timestamp
        scale_sum = scale_pivot.loc[common_ts].sum(axis=1)
        flow_sum = flow_pivot.loc[common_ts].sum(axis=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(scale_sum, flow_sum, alpha=0.1, s=5, c="steelblue")
        # 1:1 line
        lims = [min(scale_sum.min(), flow_sum.min()), max(scale_sum.max(), flow_sum.max())]
        ax.plot(lims, lims, "r--", alpha=0.5, label="1:1")
        ax.set_xlabel("Sum of scale readings")
        ax.set_ylabel("Sum of flow_rate readings")
        ax.set_title("Mass balance: total scales vs total flow rates")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIGURES / "mass_balance_scatter.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        # Also do per-timestamp ratio
        ratio = flow_sum / scale_sum.replace(0, np.nan)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(ratio.dropna().clip(0, 5), bins=100, color="steelblue", edgecolor="white")
        ax2.set_xlabel("Flow / Scale ratio")
        ax2.set_ylabel("Count")
        ax2.set_title("Mass balance ratio distribution")
        plt.tight_layout()
        fig2.savefig(FIGURES / "mass_balance_ratio.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"  ✓  Mass balance plots ({len(common_ts)} common timestamps)")
    else:
        print("  (no common timestamps between scales and flow rates)")
else:
    print("  (no scale/flow_rate data found)")

# ── 6. INITIAL HYPOTHESES ─────────────────────────────────────────────────
print("\n═══ 6. Generating hypotheses list ═══\n")

hypotheses = """
AGROMILL CORP — INITIAL HYPOTHESES (Phase 3 EDA)
==================================================

[1] Shift Effects
    H1: Afternoon and Night shifts produce higher moisture variability
        due to residual heat from Morning production, affecting
        drying dynamics.
    H2: The first hours of each shift (post-reset) show lower throughput
        and different quality profiles — the reset/setup period
        meaningfully reduces effective production time.

[2] Granulometry Control
    H3: Mesh calibre configuration (especially 3mm vs 5mm) is the
        dominant driver of the fraction_14_140 / fraction_18_100 split
        in Product 01. Finer meshes shift mass toward finer fractions.
    H4: Frequency settings (40/45/50 Hz) modulate throughput but have
        a secondary effect on granulometry — a potential tuning lever
        for quality without changing mesh hardware.

[3] Maintenance Impact
    H5: Equipment maintenance events cause a temporary shift in
        sensor readings (scale accuracy, motor current) for 1-3 days
        post-maintenance, after which readings stabilize.
    H6: Mesh replacement or cleaning shifts the granulometry
        distribution noticeably for 1-2 shifts until the mesh "seats".

[4] Mass Balance & Data Quality
    H7: Systematic discrepancies between scale totals and flow-rate
        totals reveal measurement drift or calibration loss in
        specific scale tags, compromising mass balance closure.
    H8: The flow_rate_scale sensor saturation at ~66000 correlates
        with high-throughput periods, meaning the production line
        occasionally exceeds sensor measurement range.

[5] Product Differentiation
    H9: Product_02's distinct granulometry profile (dominated by
        fraction_25_071 + fraction_18_100 vs fraction_14_140 for
        Product_01) confirms it follows a different production
        line/configuration — models must be built separately.
    H10: The shared "corrupted" test_ids across all three product
         tables suggest a common data-entry point or shared LIMS
         integration that occasionally produces garbage records.

[6] Sensor-Product Relationship
    H11: Mill motor current (corrente_motor) and consumption (consumo)
         correlate with throughput — higher load = more material
         being processed. These can serve as proxy throughput sensors.
    H12: Storage bin levels (nivel) show daily production cycles,
         filling during Morning shift and drawing down overnight —
         a potential inventory-velocity metric.

[7] Temporal Patterns
    H13: Granulometry and density drift gradually over weeks,
         reflecting raw material seasonality or gradual equipment wear.
    H14: Weekly patterns exist: Monday morning startup differs from
         Friday afternoon wind-down in both throughput and quality.

[8] Client-Specific Quality
    H15: Product_01_client2 (with ENSAIO detail type = 282/377 rows)
         receives more testing per batch than client 1, suggesting
         tighter contractual specifications.
"""

with open(FIGURES.parent.parent / "outputs" / "initial_hypotheses.md", "w") as f:
    f.write(hypotheses.strip())
print("  ✓  initial_hypotheses.md")

# ── WRAP UP ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3 EDA complete.")
print("=" * 60)
print(f"  Figures saved to:  {FIGURES}/")
print(f"  Tables saved to:   {TABLES}/")
print(f"  Hypotheses:        outputs/initial_hypotheses.md")
