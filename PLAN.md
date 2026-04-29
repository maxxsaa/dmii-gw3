# AgroMill Project Execution Plan

## Objective

Deliver a complete, data-driven strategy for AgroMill Corp covering:

1. Quality Score prediction for Product 1.
2. Throughput estimation for 5 main products/sub-products.
3. Identification of key operational drivers.
4. Actionable business recommendations and next-step analytics roadmap.

---

## Project Structure

Suggested working folders:

- `data/raw/` - original files (read-only)
- `data/processed/` - cleaned, merged, feature-engineered datasets
- `notebooks/` - exploratory and modeling notebooks
- `src/` - reusable scripts/functions
- `outputs/figures/` - charts for report/slides
- `outputs/models/` - trained models and artifacts
- `outputs/tables/` - evaluation summaries and feature-importance tables

---

## Work Plan by Phase

## Phase 1 - Data Ingestion and Understanding

### Tasks
- Load all datasets and document schemas.
- Standardize column names and data types.
- Build a dataset dictionary (fields, units, expected ranges).
- Validate timestamp formats and timezone assumptions.

### Deliverables
- Data inventory sheet.
- Schema consistency report.

---

## Phase 2 - Data Cleaning (Deep Dive Requirement)

### Tasks
- Missing data analysis: percentage by column/time/shift.
- Illogical/impossible value detection using domain rules:
  - Negative or impossible physical measures.
  - Mass balance inconsistencies between input/output.
  - Timestamp-order problems.
- Outlier analysis (IQR/z-score/robust quantiles + domain checks).
- Document every cleaning decision with rationale.

### Deliverables
- Cleaned datasets.
- Cleaning log with before/after metrics.

---

## Phase 3 - Exploratory Data Analysis

### Tasks
- Descriptive statistics by product, shift, and period.
- Distribution plots for moisture, density, granulometry fractions.
- Correlation analysis and multicollinearity checks.
- Time-based trend and drift analysis.
- Operational bottleneck exploration:
  - Shift effects.
  - Reset/setup impact.
  - Maintenance and mesh configuration effects.

### Deliverables
- EDA notebook.
- Visual pack for presentation.
- Initial hypotheses list.

---

## Phase 4 - Feature Engineering

### Tasks
- Time features: hour, weekday, shift, elapsed time since maintenance/reset.
- Aggregation features from sensors (rolling means/std, lag features).
- Process consistency metrics (e.g., mass-balance deviation scores).
- Product-specific engineered features for quality modeling.

### Deliverables
- Feature matrix documentation.
- Reproducible feature-generation code.

---

## Phase 5 - Modeling (Deep Dive Requirement)

### A) Quality Score Prediction (Product 1)
- Try both regression and classification framing (depending on target definition).
- Baselines: Linear/Logistic models.
- Tree ensembles: Random Forest, Gradient Boosting, XGBoost/LightGBM.
- Model selection via cross-validation and metric comparison.

### B) Throughput Estimation (5 products/sub-products)
- Multi-target or per-product regressors.
- Compare linear, tree-based, and boosting models.
- Evaluate by product segment and time windows.

### C) Feature Importance and Drivers
- Global importances (gain/permutation).
- Local explanations (SHAP) for key predictions.
- Stability checks of top drivers across folds/time splits.

### Deliverables
- Trained candidate models.
- Evaluation report with metrics and interpretation.
- Driver analysis report.

---

## Phase 6 - Business Recommendations and Open Reflection

### Tasks
- Translate model findings into operational actions:
  - Shift planning
  - Maintenance scheduling
  - Sensor/data collection improvements
  - Quality-control prioritization
- Build a prioritized action roadmap:
  - Quick wins (0-3 months)
  - Mid-term (3-9 months)
  - Long-term advanced analytics vision

### Deliverables
- Final recommendation brief.
- Executive-ready narrative for presentation.

---

## Validation and Quality Assurance

- Use train/validation/test split with time-awareness when needed.
- Track all assumptions and data leakage risks.
- Keep reproducibility through fixed random seeds and versioned outputs.
- Verify that conclusions respect physical constraints and process reality.

---

## Suggested Timeline (7 Weeks)

- **Week 1:** Ingestion, schema checks, data dictionary.
- **Week 2:** Cleaning and integrity validation.
- **Week 3:** EDA and initial hypotheses.
- **Week 4:** Feature engineering and baseline models.
- **Week 5:** Advanced models + tuning.
- **Week 6:** Driver analysis + business interpretation.
- **Week 7:** Final story, slides, rehearsal, and QA.

---

## Python Libraries to Use

- **Core data handling:** `pandas`, `numpy`, `pyarrow`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Statistics/diagnostics:** `scipy`, `statsmodels`, `missingno`
- **Machine learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Model explainability:** `shap`
- **Notebook workflow:** `jupyterlab`, `ipykernel`

These libraries are listed in `requirements.txt`.
