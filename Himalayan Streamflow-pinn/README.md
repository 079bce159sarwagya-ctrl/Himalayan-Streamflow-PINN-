"""
Cross-Basin Himalayan Streamflow Prediction Benchmark:

Benchmarks three modeling paradigms trained strictly on the Karnali River Basin
(2020-2023) and evaluated under three progressively harder regimes:

  1. Temporal Holdout (In-Basin)     : Karnali Basin (2024)
  2. Zero-Shot Cross-Basin (Large)   : Koshi Basin (2020-2024, 54,100 km2)
  3. Zero-Shot Cross-Basin (Small)   : Babai Basin (2020-2024,  2,570 km2)

Models Benchmarked:

  1. Random Forest Regressor  (Tabular ML baseline with multi-scale rolling features)
  2. Temporal Transformer     (Pure data-driven self-attention sequence model)
  3. Hydro-PINN               (Physics-Informed Neural Network with learnable
                               water-balance & dry-spell recession constraints)

Artifacts Generated in ./results/:

  - metrics_summary.csv               (Full numerical benchmark table)
  - RESULTS_TABLE.md                  (Ready-to-paste Markdown table for GitHub README)
  - fig1_metric_comparison.png        (NSE, KGE, Flood F1, Flood Recall across basins)
  - fig2_hydrographs_2024.png         (2024 daily hydrographs + monsoon & Q90 flood line)
  - fig3_scatter_diagnostics.png      (3x3 Observed vs. Predicted scatter + 1:1 line)
  - fig4_flood_detection_cm.png       (Zero-shot Q90 flood confusion matrices)
  - fig5_training_convergence.png     (Data MSE & Hydro-PINN physics loss curves)
"""
