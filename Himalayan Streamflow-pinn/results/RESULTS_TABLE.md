# Cross-Basin Benchmark Results

| Basin | Model | NSE | KGE | R2 | RMSE (m3/s) | PBIAS (%) | Flood Accuracy | Flood Precision | Flood Recall | Flood F1 | Neg Flow (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Karnali (In-Basin 2024) | **Random Forest** | 0.958 | 0.892 | 0.958 | 184.3 | -1.3% | 0.989 | 0.946 | 0.946 | 0.946 | 0.00% |
| Karnali (In-Basin 2024) | **Transformer** | 0.971 | 0.918 | 0.971 | 153.2 | -0.7% | 0.986 | 0.900 | 0.973 | 0.935 | 0.00% |
| Karnali (In-Basin 2024) | **Hydro-PINN** | 0.980 | 0.944 | 0.980 | 125.8 | -0.4% | 0.992 | 0.947 | 0.973 | 0.960 | 0.00% |
| Koshi (Zero-Shot Large) | **Random Forest** | 0.970 | 0.909 | 0.970 | 206.4 | -5.9% | 0.984 | 0.987 | 0.850 | 0.913 | 0.00% |
| Koshi (Zero-Shot Large) | **Transformer** | 0.986 | 0.955 | 0.986 | 142.3 | -4.0% | 0.992 | 0.977 | 0.944 | 0.960 | 0.00% |
| Koshi (Zero-Shot Large) | **Hydro-PINN** | 0.985 | 0.946 | 0.985 | 144.8 | -4.9% | 0.991 | 0.982 | 0.928 | 0.954 | 0.00% |
| Babai (Zero-Shot Small) | **Random Forest** | 0.967 | 0.883 | 0.967 | 8.1 | +11.5% | 0.978 | 0.851 | 0.950 | 0.898 | 0.00% |
| Babai (Zero-Shot Small) | **Transformer** | 0.968 | 0.874 | 0.968 | 7.9 | +12.6% | 0.980 | 0.840 | 0.989 | 0.908 | 0.00% |
| Babai (Zero-Shot Small) | **Hydro-PINN** | 0.970 | 0.881 | 0.970 | 7.8 | +11.9% | 0.982 | 0.852 | 0.989 | 0.915 | 0.00% |

## Learned Hydro-PINN Physical Parameters

- **beta_p (Orographic Rain Factor):** `0.5614`
- **alpha_et (Effective ET Fraction):** `0.4436`
- **k_melt (Snowmelt mm/d/degC):** `0.1012`
- **k_sm (Soil Storage Scale):** `1.6243`
