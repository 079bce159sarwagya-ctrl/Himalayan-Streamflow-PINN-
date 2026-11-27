from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


print("Imports successful! Torch:", torch.__version__)

# Create synthetic test CSVs with exact schema matching datas.py to test the pipeline end-to-end
rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
n = len(dates)
doy = dates.day_of_year.values


def make_synthetic_basin(area_km2, rain_scale, base_q, seed):
    r = np.random.default_rng(seed)
    monsoon = np.exp(-0.5 * ((doy - 210) / 40.0) ** 2)
    rainfall = np.maximum(0, r.gamma(1.2, 2.5, n) *
                          (0.2 + 4.0 * monsoon) * rain_scale)
    temp = 15.0 + 10.0 * \
        np.sin(2 * np.pi * (doy - 105) / 365.25) + r.normal(0, 1.5, n)
    et = np.maximum(0.5, 2.0 + 2.5 * np.sin(2 * np.pi *
                    (doy - 95) / 365.25) + r.normal(0, 0.3, n))
    solar = np.maximum(5.0, 18.0 + 6.0 * np.sin(2 * np.pi *
                       (doy - 80) / 365.25) + r.normal(0, 1.5, n))
    humidity = np.clip(55.0 + 30.0 * monsoon + r.normal(0, 5.0, n), 20, 99)

    # Simple conceptual bucket for realistic soil moisture & runoff
    sm = np.zeros(n)
    q_mm = np.zeros(n)
    s = 25.0
    for t in range(n):
        s = max(5.0, s + 0.35 * rainfall[t] - 0.25 * et[t])
        runoff = base_q + 0.08 * s + 0.12 * rainfall[t]
        s = max(5.0, s - 0.08 * s)
        sm[t] = np.clip(s / 100.0, 0.05, 0.6)
        q_mm[t] = max(0.1, runoff + r.normal(0, 0.15))

    q_cumecs = q_mm * area_km2 / 86.4
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "rainfall_mm": rainfall,
        "temp_c": temp,
        "evapotranspiration_mm": et,
        "solar_radiation_mj": solar,
        "humidity_pct": humidity,
        "soil_moisture": sm,
        "streamflow_cumecs": q_cumecs,
        "basin_area_km2": area_km2,
        "specific_discharge_mm": q_mm,
    })


make_synthetic_basin(43679.0, 1.0, 0.5, 1).to_csv(
    "karnali_train_2020_2024.csv", index=False)
make_synthetic_basin(54100.0, 1.15, 0.6, 2).to_csv(
    "koshi_test1_2020_2024.csv", index=False)
make_synthetic_basin(2570.0, 0.9, 0.3, 3).to_csv(
    "babai_test2_2020_2024.csv", index=False)
print("Synthetic CSVs created successfully.")
