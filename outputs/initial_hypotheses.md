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