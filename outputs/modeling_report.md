# Phase 5 — Modeling Results

## Quality Prediction (Product 01)

**Features used:** test_hour, test_weekday, test_month, test_is_weekend, calibre_size_3mm, calibre_size_5mm, calibre_freq_3mm, calibre_freq_5mm, last_maint_hours, shift_Morning, shift_Night

**Rows:** 1,164 (after dropping tests without mesh records)

### Density

| Model | R² (shuffled) | R² (time-series) | MAE |
|-------|:-:|:-:|:-:|
| Ridge | 0.05 ±0.04 | -0.17 ±0.12 | 1.20 kg/m³ |
| **RandomForest** | **0.16 ±0.02** | **-0.12 ±0.15** | **1.10 kg/m³** |
| LightGBM | 0.09 ±0.04 | -0.27 ±0.18 | 1.13 kg/m³ |

### Moisture

| Model | R² (shuffled) | R² (time-series) | MAE |
|-------|:-:|:-:|:-:|
| Ridge | 0.09 ±0.04 | -0.30 ±0.36 | 1.24% |
| **RandomForest** | **0.28 ±0.02** | **-0.30 ±0.41** | **1.08%** |
| LightGBM | 0.27 ±0.06 | -0.45 ±0.28 | 1.08% |

### Top Drivers (Permutation Importance — Moisture)

| Feature | Importance | Interpretation |
|---------|:-:|---|
| test_hour | 0.414 | Time of day is the strongest signal — quality varies across shifts |
| last_maint_hours | 0.378 | Fresher maintenance → more consistent quality |
| test_month | 0.258 | Gradual drift across months (seasonality / equipment wear) |
| test_weekday | 0.164 | Day-of-week effects (Monday startup, Friday wind-down) |
| calibre_freq_3mm | 0.127 | Mesh vibration frequency affects particle size distribution |
| calibre_size_5mm | 0.060 | Coarser mesh setting shifts granulometry |
| calibre_freq_5mm | 0.040 | Secondary frequency effect |
| calibre_size_3mm | 0.032 | Fine mesh size effect |
| shift_Night | 0.024 | Night shift differs from baseline (Afternoon) |
| shift_Morning | 0.016 | Morning shift differs from baseline |
| test_is_weekend | 0.002 | Negligible weekend effect |

---

## Throughput Estimation

**Features:** 71 (lagged flow rates, rolling statistics, sensor aggregates by tag group)

**Rows:** 2,385 (hourly, after zero-flow balancing)

| Model | R² | MAE |
|-------|:-:|----:|
| Ridge | 0.97 ±0.00 | 2,420 ±75 |
| RandomForest | 0.98 ±0.01 | 1,284 ±185 |
| **LightGBM** | **0.99 ±0.00** | **961 ±77** |

### Top Drivers (Permutation Importance)

| Feature | Importance | Interpretation |
|---------|:-:|---|
| sns_flow_rate_count | 1.045 | Number of flow sensor readings in the hour (proxy for uptime) |
| sns_flow_rate_mean | 0.284 | Average flow rate during the hour |
| total_flow_roll6h_std | 0.029 | Flow variability over last 6 hours |
| sns_scale_count | 0.021 | Scale reading count (activity level) |
| total_flow_roll3h_mean | 0.009 | Recent flow trend |
| total_flow_roll3h_std | 0.004 | Short-term flow stability |

---

## Key Takeaways

1. **Throughput is highly predictable** (R²=0.99). The model essentially learns that flow rate correlates with itself over time. Operationally useful for real-time monitoring and anomaly detection.

2. **Quality is weakly predictable** (R²=0.16–0.28 at best). The main signals come from operational timing (hour, month, days-since-maintenance), not from sensor readings. Most quality variation is driven by unmeasured factors — raw material batches, ambient conditions, lab measurement noise.

3. **Sensor data does NOT improve quality prediction.** Hourly aggregates are too coarse to capture the specific conditions during each sample's production window. The sensor variation mostly follows daily cycles that `test_hour` already captures.

4. **To improve quality models:** install in-line NIR quality sensors, track raw material lot IDs, and record exact production timestamps (not just dates).
