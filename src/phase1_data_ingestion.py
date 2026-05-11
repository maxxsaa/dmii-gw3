"""
Phase 1 - Data Ingestion and Understanding
============================================
Loads all datasets, standardizes column names,
validates timestamps, and produces a schema report.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW = Path("data/raw")
PROC = Path("data/processed")
OUTPUTS = Path("outputs/tables")

for p in [PROC, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Load all datasets
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1 — DATA INGESTION & UNDERSTANDING")
print("=" * 60)

datasets_raw = {}

# -- Equipment separation
datasets_raw["equipment_separation"] = pd.read_csv(
    RAW / "equipamento_separacao.csv", encoding="utf-8-sig"
)

# -- Mesh
datasets_raw["mesh"] = pd.read_csv(RAW / "malhas.csv", encoding="utf-8-sig")

# -- Product 01 (main client)
datasets_raw["product_01"] = pd.read_csv(
    RAW / "labs_produto_01.csv", encoding="utf-8-sig"
)

# -- Product 01 (second client)
datasets_raw["product_01_client2"] = pd.read_csv(
    RAW / "labs_produto_01_2.csv", encoding="utf-8-sig"
)

# -- Product 02
datasets_raw["product_02"] = pd.read_csv(
    RAW / "labs_produto_02.csv", encoding="utf-8-sig"
)

# -- Sensors (Parquet)
datasets_raw["sensors"] = pd.read_parquet(RAW / "sensores.parquet")

print(f"\nLoaded {len(datasets_raw)} datasets.\n")

# ---------------------------------------------------------------------------
# 2.  Schema standardisation mapping
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # equipment_separation
    "id_unidade_prod": "production_unit_id",
    "id_equipamento_separacao": "equipment_separation_id",
    "id_malha_ref": "mesh_ref_id",
    "espec_abertura_malha": "mesh_opening_spec",
    "dt_manut_componente": "component_maintenance_date",
    # mesh
    "data": "date",
    "tamanho_calibre_malha_3mm": "calibre_size_3mm",
    "tamanho_calibre_malha_5mm": "calibre_size_5mm",
    "freq_calibre_malha_3mm": "calibre_freq_3mm",
    "freq_calibre_malha_5mm": "calibre_freq_5mm",
    "chefia": "supervisor",
    # product quality (shared)
    "densidade": "density",
    "humidade": "moisture",
    "data_prod": "production_date",
    "data_teste": "test_date",
    "Detail": "detail_type",
    "id_ensaio": "test_id",
    "Under_00": "under_000",
    "Under_000": "under_000",
    "10_200": "fraction_10_200",
    "14_140": "fraction_14_140",
    "18_100": "fraction_18_100",
    "25_071": "fraction_25_071",
    "8_236": "fraction_8_236",
    "origem": "origin",
    # sensors
    "Tag": "tag",
    "tag": "tag",
    "Value": "value",
    "Date time": "datetime",
    "DateTime": "datetime",
}

datasets = {}
rename_log = []

for name, df in datasets_raw.items():
    rename_map = {c: COLUMN_MAP.get(c, c) for c in df.columns}
    changes = [(old, new) for old, new in rename_map.items() if old != new]
    df = df.rename(columns=rename_map)
    datasets[name] = df
    if changes:
        rename_log.append((name, changes))

# ---------------------------------------------------------------------------
# 3.  Build dataset dictionary / inventory
# ---------------------------------------------------------------------------
inventory_rows = []

for name, df in datasets.items():
    for col in df.columns:
        dtype = df[col].dtype
        nulls = int(df[col].isna().sum())
        null_pct = round(nulls / len(df) * 100, 2)
        n_unique = int(df[col].nunique())
        sample_vals = (
            df[col].dropna().unique()[:5].tolist()
            if df[col].nunique() <= 20
            else []
        )
        inventory_rows.append(
            {
                "dataset": name,
                "field": col,
                "dtype": str(dtype),
                "nulls": nulls,
                "null_pct": null_pct,
                "n_unique": n_unique,
                "sample_values": sample_vals,
                "n_rows": len(df),
            }
        )

inventory = pd.DataFrame(inventory_rows)
inventory.to_csv(OUTPUTS / "data_inventory.csv", index=False)
print("Saved: outputs/tables/data_inventory.csv")

# ---------------------------------------------------------------------------
# 4.  Timestamp validation
# ---------------------------------------------------------------------------
timestamp_cols = {
    "equipment_separation": ["component_maintenance_date"],
    "mesh": ["date"],
    "product_01": ["production_date", "test_date"],
    "product_01_client2": ["production_date", "test_date"],
    "product_02": ["production_date", "test_date"],
    "sensors": ["datetime"],
}

ts_report = []

for name, cols in timestamp_cols.items():
    df = datasets[name]
    for col in cols:
        if col not in df.columns:
            continue
        raw_vals = datasets_raw[name][
            [c for c in datasets_raw[name].columns if COLUMN_MAP.get(c) == col][0]
        ]
        sample_raw = raw_vals.dropna().head(10).tolist()
        # Check parsed
        parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
        total = len(df[col].dropna())
        parse_fail = parsed.isna().sum() - df[col].isna().sum()
        has_tz = parsed.dropna().apply(lambda x: x.tz is not None).sum()
        tz_aware = bool(has_tz > 0) if total > 0 else False
        tz_info = str(parsed.dropna().iloc[0].tz) if tz_aware else "None"
        min_ts = parsed.min()
        max_ts = parsed.max()
        ts_report.append(
            {
                "dataset": name,
                "field": col,
                "total_values": total,
                "parse_failures": int(parse_fail),
                "tz_aware": tz_aware,
                "tz_info": tz_info,
                "min": str(min_ts) if pd.notna(min_ts) else "",
                "max": str(max_ts) if pd.notna(max_ts) else "",
                "sample_raw": sample_raw[:3],
            }
        )

ts_df = pd.DataFrame(ts_report)
ts_df.to_csv(OUTPUTS / "timestamp_validation.csv", index=False)
print("Saved: outputs/tables/timestamp_validation.csv")

# ---------------------------------------------------------------------------
# 5.  Schema consistency report (text)
# ---------------------------------------------------------------------------
report_lines = []
report_lines.append("# Schema Consistency Report")
report_lines.append("")
report_lines.append(
    f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
)
report_lines.append("")

for name, df in datasets.items():
    report_lines.append(f"## Dataset: {name}")
    report_lines.append(f"- Rows: {len(df):,}")
    report_lines.append(f"- Columns: {list(df.columns)}")
    report_lines.append("")
    report_lines.append("| Field | Dtype | Non-null | Nulls | Null % |")
    report_lines.append("|-------|-------|----------|-------|--------|")
    for col in df.columns:
        nn = df[col].notna().sum()
        n = df[col].isna().sum()
        pct = n / len(df) * 100
        report_lines.append(f"| {col} | {df[col].dtype} | {nn} | {n} | {pct:.2f}% |")
    report_lines.append("")

report_text = "\n".join(report_lines)
with open(OUTPUTS / "schema_consistency_report.md", "w") as f:
    f.write(report_text)
print("Saved: outputs/tables/schema_consistency_report.md")

# ---------------------------------------------------------------------------
# 6.  Convert timestamp columns to proper datetime dtypes
# ---------------------------------------------------------------------------
for name, df in datasets.items():
    cols = timestamp_cols.get(name, [])
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

# ---------------------------------------------------------------------------
# 7.  Save standardised datasets
# ---------------------------------------------------------------------------
for name, df in datasets.items():
    path = PROC / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# 7.  Quick sensor summary (too large to print fully)
# ---------------------------------------------------------------------------
sensors = datasets["sensors"]
print(f"\nSensor data: {len(sensors):,} rows x {len(sensors.columns)} cols")
print(f"  Tags: {sensors['tag'].nunique()}")
print(f"  Date range: {sensors['datetime'].min()} -> {sensors['datetime'].max()}")
print(f"  Value range: {sensors['value'].min():.2f} -> {sensors['value'].max():.2f}")

print("\nPhase 1 complete.")
