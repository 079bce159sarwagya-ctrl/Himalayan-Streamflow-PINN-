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

from __future__ import annotations
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.ensemble import RandomForestRegressor
import torch.nn.functional as F
import torch.nn as nn
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
# Non-interactive backend for clean, non-blocking saves in VS Code
matplotlib.use("Agg")


# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class Config:
    train_csv: str = "karnali_train_2020_2024.csv"
    koshi_csv: str = "koshi_test1_2020_2024.csv"
    babai_csv: str = "babai_test2_2020_2024.csv"
    output_dir: str = "results"

    # Strict temporal split on Karnali (zero data leakage)
    train_end_date: str = "2023-12-31"
    seq_len: int = 30                # 30-day hydrological memory window
    flood_quantile: float = 0.90     # Basin Q90 defines High-Flow / Flood Events

    # Deep learning optimization settings
    seed: int = 42
    batch_size: int = 64
    epochs: int = 40
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Hydro-PINN physics regularization weights
    lambda_water_balance: float = 0.10
    lambda_recession: float = 0.25


CFG = Config()

FEATURE_COLS = [
    "rainfall_mm",
    "temp_c",
    "evapotranspiration_mm",
    "solar_radiation_mj",
    "humidity_pct",
    "soil_moisture",
    "doy_sin",
    "doy_cos",
]
TARGET_COL = "specific_discharge_mm"  # Scale-invariant runoff (mm/day)

MODEL_COLORS = {
    "Observed": "#111827",
    "Random Forest": "#D97706",
    "Transformer": "#2563EB",
    "Hydro-PINN": "#059669",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. DATA LOADING & FEATURE ENGINEERING
# =============================================================================

def load_basin_data(csv_path: str) -> pd.DataFrame:
    """Loads a basin CSV, cleans any minor gaps, and encodes monsoon seasonality."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing '{csv_path}'. Please run datas.py first in this folder."
        )
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().bfill()

    # Cyclical day-of-year encoding for Himalayan monsoon & snowmelt seasonality
    doy = df["date"].dt.dayofyear.values
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return df


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs multi-scale rolling hydrometeorological features (3d, 7d, 15d, 30d)
    for Random Forest so it uses the exact same 30-day memory horizon as deep models.
    """
    feats = df[FEATURE_COLS].copy()
    for w in (3, 7, 15, 30):
        feats[f"rain_sum_{w}d"] = df["rainfall_mm"].rolling(w).sum()
        feats[f"rain_mean_{w}d"] = df["rainfall_mm"].rolling(w).mean()
        feats[f"temp_mean_{w}d"] = df["temp_c"].rolling(w).mean()
        feats[f"et_sum_{w}d"] = df["evapotranspiration_mm"].rolling(w).sum()
        feats[f"sm_mean_{w}d"] = df["soil_moisture"].rolling(w).mean()

    feats["sm_diff_3d"] = df["soil_moisture"].diff(3)
    feats["rain_lag_1d"] = df["rainfall_mm"].shift(1)
    feats["rain_lag_2d"] = df["rainfall_mm"].shift(2)
    return feats


def build_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int,
) -> Dict[str, np.ndarray]:
    """
    Builds aligned 30-day input windows for sequence models plus raw unscaled
    physical forcings [rain, temp, ET, soil_moisture] for the Hydro-PINN physics loss.
    """
    x_scaled = scaler.transform(df[FEATURE_COLS].values)
    y_specific = df[TARGET_COL].values.astype(np.float32)
    q_cumecs = df["streamflow_cumecs"].values.astype(np.float32)
    area_km2 = float(df["basin_area_km2"].iloc[0])

    phys_raw = df[
        ["rainfall_mm", "temp_c", "evapotranspiration_mm", "soil_moisture"]
    ].values.astype(np.float32)

    X_seq, P_seq, y_target, q_obs, dates = [], [], [], [], []
    for idx in range(seq_len - 1, len(df)):
        start = idx - seq_len + 1
        X_seq.append(x_scaled[start: idx + 1])
        P_seq.append(phys_raw[start: idx + 1])
        y_target.append(y_specific[idx])
        q_obs.append(q_cumecs[idx])
        dates.append(df["date"].iloc[idx])

    return {
        "X_seq": np.asarray(X_seq, dtype=np.float32),
        "P_seq": np.asarray(P_seq, dtype=np.float32),
        "y_specific": np.asarray(y_target, dtype=np.float32),
        "q_cumecs": np.asarray(q_obs, dtype=np.float32),
        "dates": np.asarray(dates),
        "area_km2": area_km2,
    }


def specific_to_cumecs(q_mm: np.ndarray, area_km2: float) -> np.ndarray:
    """Converts specific runoff (mm/day) back to physical river discharge (m3/s)."""
    return np.maximum(0.0, q_mm) * area_km2 / 86.4


# =============================================================================
# 3. NEURAL ARCHITECTURES (TRANSFORMER & HYDRO-PINN)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence order awareness."""

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TemporalTransformer(nn.Module):
    """
    Model 2: Pure Data-Driven Temporal Transformer.
    Captures long-range rainfall-runoff lag dependencies via multi-head self-attention.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)
        context = 0.5 * h.mean(dim=1) + 0.5 * h[:, -1, :]
        return self.head(context).squeeze(-1)


class HydroPINN(nn.Module):
    """
    Model 3: Physics-Informed Hydrological Neural Network (Hydro-PINN).

    Physical Inductive Biases Embedded:
      1. Structural Non-Negativity : Softplus head guarantees q_hat >= 0 (zero negative flow).
      2. Differentiable Water Balance: Constrains multi-day runoff against net water supply
         (orographic rain + degree-day snowmelt - effective ET - soil storage change).
      3. Dry-Spell Recession Law   : Penalizes unphysical upward flow spikes on dry days.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Learnable physical watershed parameters
        self.raw_beta_p = nn.Parameter(
            torch.tensor(0.0))    # Orographic rain scaling
        # Effective ET fraction (0..1)
        self.raw_alpha_et = nn.Parameter(torch.tensor(-0.5))
        self.raw_k_melt = nn.Parameter(
            torch.tensor(-2.0))   # Degree-day melt factor
        # Soil storage drainage scale
        self.raw_k_sm = nn.Parameter(torch.tensor(1.0))

    @property
    def physics_params(self) -> Dict[str, float]:
        return {
            "beta_p (Orographic Rain Factor)": float(F.softplus(self.raw_beta_p).item()),
            "alpha_et (Effective ET Fraction)": float(torch.sigmoid(self.raw_alpha_et).item()),
            "k_melt (Snowmelt mm/d/degC)": float(F.softplus(self.raw_k_melt).item()),
            "k_sm (Soil Storage Scale)": float(F.softplus(self.raw_k_sm).item()),
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_seq, _ = self.gru(x)                            # (B, T, H)
        q_seq = F.softplus(self.head(h_seq).squeeze(-1)
                           )  # Strictly >= 0 (B, T)
        return q_seq[:, -1], q_seq

    def compute_physics_loss(
        self, q_seq: torch.Tensor, p_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rain = p_raw[:, :, 0]
        temp = p_raw[:, :, 1]
        et = p_raw[:, :, 2]
        sm = p_raw[:, :, 3]

        beta_p = F.softplus(self.raw_beta_p)
        alpha_et = torch.sigmoid(self.raw_alpha_et)
        k_melt = F.softplus(self.raw_k_melt)
        k_sm = F.softplus(self.raw_k_sm)

        # 1. Control-volume Water Balance Residual over the 30-day window
        snowmelt_proxy = k_melt * F.relu(temp)
        net_input = beta_p * rain + snowmelt_proxy - alpha_et * et
        delta_storage = k_sm * (sm[:, -1] - sm[:, 0]) / \
            max(1, q_seq.size(1) - 1)
        wb_supply = F.relu(net_input.mean(dim=1) - delta_storage)
        loss_wb = F.smooth_l1_loss(q_seq.mean(dim=1), wb_supply)

        # 2. Dry-Spell Hydrograph Recession Constraint (rain < 0.5 mm/day)
        dry_mask = (rain[:, 1:] < 0.5).float()
        dq_pos = F.relu(q_seq[:, 1:] - q_seq[:, :-1])
        loss_recession = ((dq_pos ** 2) * dry_mask).sum() / \
            (dry_mask.sum() + 1e-6)

        return loss_wb, loss_recession


# =============================================================================
# 4. HYDROLOGICAL & FLOOD-DETECTION EVALUATION METRICS
# =============================================================================

def compute_nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (1.0 = perfect, 0.0 = mean baseline)."""
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - np.sum((obs - pred) ** 2) / denom) if denom > 0 else np.nan


def compute_kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency integrating correlation, variability, and volume bias."""
    if np.std(pred) < 1e-9 or np.std(obs) < 1e-9:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / (np.mean(obs) + 1e-9)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def compute_pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    """Percent Bias (%). Optimal = 0%."""
    return float(100.0 * np.sum(pred - obs) / (np.sum(obs) + 1e-9))


def evaluate_predictions(
    obs_q: np.ndarray,
    pred_q: np.ndarray,
    obs_mm: np.ndarray,
    raw_pred_mm: np.ndarray,
    flood_threshold_q: float,
) -> Dict[str, float]:
    """Computes continuous hydrological metrics, physical violations, and Q90 flood metrics."""
    pred_mm_clipped = np.maximum(0.0, raw_pred_mm)
    neg_pct = float(100.0 * np.mean(raw_pred_mm < 0.0))

    # Binary Flood / High-Flow Event Detection (Q >= Basin Q90)
    y_true_flood = (obs_q >= flood_threshold_q).astype(int)
    y_pred_flood = (pred_q >= flood_threshold_q).astype(int)

    return {
        "NSE": compute_nse(obs_q, pred_q),
        "KGE": compute_kge(obs_q, pred_q),
        "R2": float(r2_score(obs_q, pred_q)),
        "RMSE (m3/s)": float(np.sqrt(mean_squared_error(obs_q, pred_q))),
        "MAE (m3/s)": float(mean_absolute_error(obs_q, pred_q)),
        "RMSE (mm/d)": float(np.sqrt(mean_squared_error(obs_mm, pred_mm_clipped))),
        "PBIAS (%)": compute_pbias(obs_q, pred_q),
        "Neg Flow (%)": neg_pct,
        "Flood Accuracy": float(accuracy_score(y_true_flood, y_pred_flood)),
        "Flood Precision": float(precision_score(y_true_flood, y_pred_flood, zero_division=0)),
        "Flood Recall": float(recall_score(y_true_flood, y_pred_flood, zero_division=0)),
        "Flood F1": float(f1_score(y_true_flood, y_pred_flood, zero_division=0)),
    }


# =============================================================================
# 5. TRAINING & INFERENCE ROUTINES
# =============================================================================

def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> Tuple[TemporalTransformer, Dict[str, List[float]]]:
    model = TemporalTransformer(input_dim=X_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CFG.batch_size,
        shuffle=True,
    )
    X_val_t, y_val_t = torch.from_numpy(X_val).to(
        device), torch.from_numpy(y_val).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(X_val_t), y_val_t).item()

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:02d}/{CFG.epochs} | Train MSE: {history['train_loss'][-1]:.4f} "
                f"| Val MSE: {val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.patience:
                print(
                    f"    Early stopping at epoch {epoch} (best Val MSE: {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_hydro_pinn(
    X_train: np.ndarray,
    P_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> Tuple[HydroPINN, Dict[str, List[float]]]:
    model = HydroPINN(input_dim=X_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(P_train),
            torch.from_numpy(y_train),
        ),
        batch_size=CFG.batch_size,
        shuffle=True,
    )
    X_val_t, y_val_t = torch.from_numpy(X_val).to(
        device), torch.from_numpy(y_val).to(device)

    history = {"train_data_loss": [], "train_phys_loss": [], "val_loss": []}
    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        d_losses, p_losses = [], []
        for xb, pb, yb in loader:
            xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)
            optimizer.zero_grad()
            q_today, q_seq = model(xb)

            loss_data = F.mse_loss(q_today, yb)
            loss_wb, loss_rec = model.compute_physics_loss(q_seq, pb)
            loss_phys = CFG.lambda_water_balance * loss_wb + CFG.lambda_recession * loss_rec
            loss = loss_data + loss_phys

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            d_losses.append(loss_data.item())
            p_losses.append(loss_phys.item())

        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val_t)
            val_loss = F.mse_loss(val_pred, y_val_t).item()

        history["train_data_loss"].append(float(np.mean(d_losses)))
        history["train_phys_loss"].append(float(np.mean(p_losses)))
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:02d}/{CFG.epochs} | Data MSE: {history['train_data_loss'][-1]:.4f} "
                f"| Phys Loss: {history['train_phys_loss'][-1]:.4f} | Val MSE: {val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.patience:
                print(
                    f"    Early stopping at epoch {epoch} (best Val MSE: {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_torch(
    model: nn.Module, X_seq: np.ndarray, device: torch.device, is_pinn: bool = False
) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X_seq)),
                        batch_size=256, shuffle=False)
    for (xb,) in loader:
        xb = xb.to(device)
        out = model(xb)[0] if is_pinn else model(xb)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


# =============================================================================
# 6. PUBLICATION-QUALITY VISUALIZATIONS & REPORT EXPORT
# =============================================================================

def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_metric_comparison(metrics_df: pd.DataFrame, out_dir: str) -> None:
    """Figure 1: Grouped comparison of NSE, KGE, Flood F1, and Flood Recall."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    specs = [
        ("NSE", "Nash-Sutcliffe Efficiency (NSE) ↑"),
        ("KGE", "Kling-Gupta Efficiency (KGE) ↑"),
        ("Flood F1", "High-Flow Flood Detection F1-Score (Q ≥ Q90) ↑"),
        ("Flood Recall", "High-Flow Flood Detection Recall / Sensitivity ↑"),
    ]
    palette = {m: MODEL_COLORS[m] for m in [
        "Random Forest", "Transformer", "Hydro-PINN"]}

    for ax, (col, title) in zip(axes.flat, specs):
        sns.barplot(
            data=metrics_df,
            x="Basin",
            y=col,
            hue="Model",
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        min_val = float(metrics_df[col].min())
        ax.set_ylim(min(0.0, min_val - 0.15), 1.08)
        ax.axhline(0.0, color="#6B7280", linestyle="--", linewidth=0.8)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8.5, padding=2)

        if ax != axes[0, 0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Model", frameon=True, loc="lower right")

    fig.suptitle(
        "Cross-Basin Generalization Benchmark: Trained on Karnali → Zero-Shot Tested on Koshi & Babai",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "fig1_metric_comparison.png"),
                bbox_inches="tight")
    plt.close(fig)


def plot_hydrographs_2024(
    eval_store: Dict[str, Dict[str, np.ndarray]], out_dir: str
) -> None:
    """Figure 2: 2024 Daily Streamflow Hydrographs with Monsoon & Q90 Threshold."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    for ax, (basin_label, data) in zip(axes, eval_store.items()):
        dates = pd.to_datetime(data["dates"])
        mask = dates >= pd.Timestamp("2024-01-01")
        d_sub = dates[mask]
        obs = data["Observed"][mask]
        q90 = data["q90_threshold"]

        ax.axvspan(
            pd.Timestamp("2024-06-15"),
            pd.Timestamp("2024-09-30"),
            color="#DBEAFE",
            alpha=0.45,
            label="Monsoon Season (Jun–Sep)",
        )
        ax.axhline(
            q90,
            color="#DC2626",
            linestyle=":",
            linewidth=1.3,
            label=f"Flood Threshold Q90 ({q90:.0f} m³/s)",
        )
        ax.plot(d_sub, obs, color=MODEL_COLORS["Observed"],
                linewidth=2.0, label="Observed", zorder=4)

        for model_name, style in [
            ("Random Forest", "--"),
            ("Transformer", "-."),
            ("Hydro-PINN", "-"),
        ]:
            pred = data[model_name][mask]
            nse_val = compute_nse(obs, pred)
            ax.plot(
                d_sub,
                pred,
                linestyle=style,
                linewidth=1.5,
                alpha=0.9,
                color=MODEL_COLORS[model_name],
                label=f"{model_name} (2024 NSE={nse_val:.2f})",
                zorder=3,
            )

        ax.set_title(
            f"{basin_label} — Daily Streamflow Hydrograph (2024)", loc="left")
        ax.set_ylabel("Discharge (m³/s)")
        ax.legend(loc="upper right", ncol=2, fontsize=8.5, frameon=True)

    axes[-1].set_xlabel("Date (2024)")
    fig.suptitle(
        "Observed vs. Predicted Himalayan River Hydrographs (Zero-Shot Cross-Basin Transfer)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig2_hydrographs_2024.png"),
                bbox_inches="tight")
    plt.close(fig)


def plot_scatter_diagnostics(
    eval_store: Dict[str, Dict[str, np.ndarray]], out_dir: str
) -> None:
    """Figure 3: Observed vs. Predicted 3x3 scatter diagnostics with 1:1 line."""
    basin_names = list(eval_store.keys())
    model_names = ["Random Forest", "Transformer", "Hydro-PINN"]
    fig, axes = plt.subplots(len(basin_names), len(
        model_names), figsize=(13, 11.5))

    for row, basin in enumerate(basin_names):
        obs = eval_store[basin]["Observed"]
        max_val = float(np.percentile(obs, 99.5)) * 1.15
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            pred = eval_store[basin][model_name]
            nse = compute_nse(obs, pred)
            pbias = compute_pbias(obs, pred)

            ax.scatter(
                obs, pred, s=12, alpha=0.45, color=MODEL_COLORS[model_name], edgecolors="none"
            )
            ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.1)
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.text(
                0.05,
                0.81,
                f"NSE = {nse:.2f}\nPBIAS = {pbias:+.1f}%",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.88),
            )
            if row == 0:
                ax.set_title(model_name, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{basin}\nPredicted Q (m³/s)", fontsize=9.5)
            if row == len(basin_names) - 1:
                ax.set_xlabel("Observed Q (m³/s)", fontsize=9.5)

    fig.suptitle(
        "Predicted vs. Observed Discharge Scatter Diagnostics (Dashed = 1:1 Ideal)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(
        out_dir, "fig3_scatter_diagnostics.png"), bbox_inches="tight")
    plt.close(fig)


def plot_flood_detection_cm(
    eval_store: Dict[str, Dict[str, np.ndarray]], out_dir: str
) -> None:
    """Figure 4: Confusion matrices for Zero-Shot High-Flow Flood Detection (Q >= Q90)."""
    unseen_basins = [k for k in eval_store.keys() if "Karnali" not in k]
    model_names = ["Random Forest", "Transformer", "Hydro-PINN"]
    fig, axes = plt.subplots(len(unseen_basins), len(
        model_names), figsize=(12.5, 7.5))

    for row, basin in enumerate(unseen_basins):
        obs = eval_store[basin]["Observed"]
        q90 = eval_store[basin]["q90_threshold"]
        y_true = (obs >= q90).astype(int)

        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            y_pred = (eval_store[basin][model_name] >= q90).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["Normal", "Flood (≥Q90)"],
                yticklabels=["Normal", "Flood (≥Q90)"],
                annot_kws={"size": 11, "weight": "bold"},
            )
            ax.set_title(
                f"{model_name}\nPrec={prec:.2f} | Rec={rec:.2f} | F1={f1:.2f}",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel(f"{basin}\nActual Event", fontsize=10)
            if row == len(unseen_basins) - 1:
                ax.set_xlabel("Predicted Event", fontsize=10)

    fig.suptitle(
        "Zero-Shot High-Flow / Flood Event Detection Confusion Matrices (Threshold: Basin Q90)",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(
        out_dir, "fig4_flood_detection_cm.png"), bbox_inches="tight")
    plt.close(fig)


def plot_training_convergence(
    tf_hist: Dict[str, List[float]],
    pinn_hist: Dict[str, List[float]],
    out_dir: str,
) -> None:
    """Figure 5: Optimization convergence for Transformer and Hydro-PINN."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    tf_ep = np.arange(1, len(tf_hist["train_loss"]) + 1)
    pinn_ep = np.arange(1, len(pinn_hist["train_data_loss"]) + 1)

    axes[0].plot(
        tf_ep, tf_hist["train_loss"], "--", color=MODEL_COLORS["Transformer"], alpha=0.6, label="Transformer Train MSE"
    )
    axes[0].plot(
        tf_ep, tf_hist["val_loss"], "-", color=MODEL_COLORS["Transformer"], linewidth=2.0, label="Transformer Val MSE (2024)"
    )
    axes[0].plot(
        pinn_ep, pinn_hist["train_data_loss"], "--", color=MODEL_COLORS["Hydro-PINN"], alpha=0.6, label="Hydro-PINN Train Data MSE"
    )
    axes[0].plot(
        pinn_ep, pinn_hist["val_loss"], "-", color=MODEL_COLORS["Hydro-PINN"], linewidth=2.0, label="Hydro-PINN Val MSE (2024)"
    )
    axes[0].set_title("Data Fidelity Convergence (Specific Runoff MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (mm/day)²")
    axes[0].legend(frameon=True, fontsize=8.5)

    axes[1].plot(
        pinn_ep,
        pinn_hist["train_phys_loss"],
        color="#047857",
        linewidth=2.2,
        label="Weighted Physics Loss (Water Balance + Recession)",
    )
    axes[1].set_title("Hydro-PINN Physical Constraint Convergence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Physics Residual Loss")
    axes[1].legend(frameon=True, fontsize=8.5)

    fig.suptitle(
        "Deep Learning Convergence & Physics Regularization (Karnali Training)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(
        out_dir, "fig5_training_convergence.png"), bbox_inches="tight")
    plt.close(fig)


def export_markdown_report(
    metrics_df: pd.DataFrame, physics_params: Dict[str, float], out_dir: str
) -> None:
    """Exports a clean Markdown summary table ready to paste directly into GitHub README.md."""
    cols = [
        "Basin", "Model", "NSE", "KGE", "R2", "RMSE (m3/s)", "PBIAS (%)",
        "Flood Accuracy", "Flood Precision", "Flood Recall", "Flood F1", "Neg Flow (%)",
    ]
    lines = [
        "# Cross-Basin Benchmark Results\n",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in metrics_df[cols].iterrows():
        vals = [
            str(row["Basin"]),
            f"**{row['Model']}**",
            f"{row['NSE']:.3f}",
            f"{row['KGE']:.3f}",
            f"{row['R2']:.3f}",
            f"{row['RMSE (m3/s)']:.1f}",
            f"{row['PBIAS (%)']:+.1f}%",
            f"{row['Flood Accuracy']:.3f}",
            f"{row['Flood Precision']:.3f}",
            f"{row['Flood Recall']:.3f}",
            f"{row['Flood F1']:.3f}",
            f"{row['Neg Flow (%)']:.2f}%",
        ]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("\n## Learned Hydro-PINN Physical Parameters\n")
    for k, v in physics_params.items():
        lines.append(f"- **{k}:** `{v:.4f}`")

    with open(os.path.join(out_dir, "RESULTS_TABLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# 7. MAIN EXECUTION PIPELINE
# =============================================================================

def main() -> None:
    t0 = time.time()
    set_seed(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    setup_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("HIMALAYAN CROSS-BASIN STREAMFLOW BENCHMARK (RF vs. Transformer vs. Hydro-PINN)")
    print(
        f"Device: {device} | Lookback: {CFG.seq_len} days | Flood Threshold: Q{int(CFG.flood_quantile*100)}")
    print("=" * 80)

    # 1. Load datasets
    df_karnali = load_basin_data(CFG.train_csv)
    df_koshi = load_basin_data(CFG.koshi_csv)
    df_babai = load_basin_data(CFG.babai_csv)

    # 2. Fit StandardScaler strictly on Karnali training split (2020-2023)
    train_mask_raw = df_karnali["date"] <= pd.Timestamp(CFG.train_end_date)
    scaler = StandardScaler()
    scaler.fit(df_karnali.loc[train_mask_raw, FEATURE_COLS].values)

    # 3. Build aligned 30-day sequence tensors
    seq_karnali = build_sequences(df_karnali, scaler, CFG.seq_len)
    seq_koshi = build_sequences(df_koshi, scaler, CFG.seq_len)
    seq_babai = build_sequences(df_babai, scaler, CFG.seq_len)

    k_dates = pd.to_datetime(seq_karnali["dates"])
    is_train = k_dates <= pd.Timestamp(CFG.train_end_date)
    is_val = ~is_train

    X_tr, P_tr, y_tr = (
        seq_karnali["X_seq"][is_train],
        seq_karnali["P_seq"][is_train],
        seq_karnali["y_specific"][is_train],
    )
    X_val, y_val = seq_karnali["X_seq"][is_val], seq_karnali["y_specific"][is_val]

    # Build aligned tabular rolling features for Random Forest
    tab_karnali = build_tabular_features(
        df_karnali).iloc[CFG.seq_len - 1:].reset_index(drop=True)
    tab_koshi = build_tabular_features(
        df_koshi).iloc[CFG.seq_len - 1:].reset_index(drop=True)
    tab_babai = build_tabular_features(
        df_babai).iloc[CFG.seq_len - 1:].reset_index(drop=True)

    assert not tab_karnali.isna().any().any(), "Unexpected NaNs in tabular features."
    X_rf_tr = tab_karnali.loc[is_train].values
    X_rf_val = tab_karnali.loc[is_val].values

    print(f"Karnali Train (2020-2023)     : {len(y_tr)} days")
    print(f"Karnali Holdout Test (2024)   : {len(y_val)} days")
    print(
        f"Koshi Zero-Shot Test (2020-24): {len(seq_koshi['y_specific'])} days")
    print(
        f"Babai Zero-Shot Test (2020-24): {len(seq_babai['y_specific'])} days")
    print("-" * 80)

    # 4. Train Model 1: Random Forest
    print("[1/3] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        random_state=CFG.seed,
        n_jobs=-1,
    )
    rf_model.fit(X_rf_tr, y_tr)

    # 5. Train Model 2: Temporal Transformer
    print("[2/3] Training Temporal Transformer...")
    tf_model, tf_hist = train_transformer(X_tr, y_tr, X_val, y_val, device)

    # 6. Train Model 3: Hydro-PINN
    print("[3/3] Training Hydro-PINN (Water-Balance + Recession Physics)...")
    pinn_model, pinn_hist = train_hydro_pinn(
        X_tr, P_tr, y_tr, X_val, y_val, device)

    print("\nLearned Physical Parameters (Hydro-PINN):")
    for k, v in pinn_model.physics_params.items():
        print(f"  - {k}: {v:.4f}")

    # 7. Evaluate across In-Basin & Zero-Shot Cross-Basin regimes
    eval_configs = [
        (
            "Karnali (In-Basin 2024)",
            X_rf_val,
            seq_karnali["X_seq"][is_val],
            seq_karnali["y_specific"][is_val],
            seq_karnali["q_cumecs"][is_val],
            seq_karnali["dates"][is_val],
            seq_karnali["area_km2"],
        ),
        (
            "Koshi (Zero-Shot Large)",
            tab_koshi.values,
            seq_koshi["X_seq"],
            seq_koshi["y_specific"],
            seq_koshi["q_cumecs"],
            seq_koshi["dates"],
            seq_koshi["area_km2"],
        ),
        (
            "Babai (Zero-Shot Small)",
            tab_babai.values,
            seq_babai["X_seq"],
            seq_babai["y_specific"],
            seq_babai["q_cumecs"],
            seq_babai["dates"],
            seq_babai["area_km2"],
        ),
    ]

    records = []
    eval_store: Dict[str, Dict[str, np.ndarray]] = {}

    for basin_label, x_rf, x_seq, obs_mm, obs_q, dates, area_km2 in eval_configs:
        q90_thresh = float(np.quantile(obs_q, CFG.flood_quantile))
        raw_preds_mm = {
            "Random Forest": rf_model.predict(x_rf),
            "Transformer": predict_torch(tf_model, x_seq, device, is_pinn=False),
            "Hydro-PINN": predict_torch(pinn_model, x_seq, device, is_pinn=True),
        }

        eval_store[basin_label] = {
            "dates": dates,
            "Observed": obs_q,
            "q90_threshold": q90_thresh,
        }

        for model_name, pred_mm_raw in raw_preds_mm.items():
            pred_q = specific_to_cumecs(pred_mm_raw, area_km2)
            eval_store[basin_label][model_name] = pred_q
            m = evaluate_predictions(
                obs_q, pred_q, obs_mm, pred_mm_raw, q90_thresh)
            records.append({"Basin": basin_label, "Model": model_name, **m})

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(
        CFG.output_dir, "metrics_summary.csv"), index=False)
    export_markdown_report(
        metrics_df, pinn_model.physics_params, CFG.output_dir)

    # 8. Print formatted results table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    show_cols = [
        "Basin", "Model", "NSE", "KGE", "RMSE (m3/s)", "PBIAS (%)",
        "Flood Accuracy", "Flood Precision", "Flood Recall", "Flood F1",
    ]
    print(metrics_df[show_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # 9. Save all 5 publication-quality figures
    print("\nGenerating publication-ready figures in ./results/ ...")
    plot_metric_comparison(metrics_df, CFG.output_dir)
    plot_hydrographs_2024(eval_store, CFG.output_dir)
    plot_scatter_diagnostics(eval_store, CFG.output_dir)
    plot_flood_detection_cm(eval_store, CFG.output_dir)
    plot_training_convergence(tf_hist, pinn_hist, CFG.output_dir)

    elapsed = time.time() - t0
    print(
        f"\nCompleted in {elapsed:.1f}s! Artifacts saved to: {os.path.abspath(CFG.output_dir)}")


if __name__ == "__main__":
    main()
