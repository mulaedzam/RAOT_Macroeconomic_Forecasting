from pathlib import Path
import copy
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_data"
RESULT_DIR = PROJECT_ROOT / "results" / "standard_transformer"
MODEL_DIR = RESULT_DIR / "models"
FIGURE_DIR = PROJECT_ROOT / "figures" / "standard_transformer"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_METRICS_OUT = RESULT_DIR / "standard_transformer_run_metrics.csv"
SUMMARY_OUT = RESULT_DIR / "standard_transformer_summary.csv"
PREDICTIONS_OUT = RESULT_DIR / "standard_transformer_test_predictions.csv"
HISTORY_OUT = RESULT_DIR / "standard_transformer_training_history.csv"
CONFIG_OUT = RESULT_DIR / "standard_transformer_config.json"
FIGURE_DPI = 300

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
# T1 = Standard Transformer using the compact/basic feature set.
# T2 = Same Standard Transformer using the richer expanded feature set.
# T2 will later be compared directly with the proposed feature-aware model.
FEATURE_SETS = ["A_basic", "B_expanded_refined"]
MODEL_LABELS = {
    "A_basic": "T1_standard_basic",
    "B_expanded_refined": "T2_standard_expanded",
}

# Multiple random seeds quantify training variability rather than reporting a
# potentially lucky single neural-network run.
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

# Deliberately small architecture because only 71 training sequences are available.
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
D_FF = 64
DROPOUT = 0.20

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 40
MIN_DELTA = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    """Control Python, NumPy and PyTorch randomness for reproducible repeated runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================
# STANDARD TRANSFORMER COMPONENTS
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Attention itself has no knowledge of sequence order, so deterministic
    position information is added to each quarterly embedding.
    """
    def __init__(self, d_model, max_len=128):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StandardTransformer(nn.Module):
    """
    Vanilla Transformer encoder adapted to sequence-to-one GDP forecasting.

    Pipeline:
      macro features -> linear embedding -> positional encoding
      -> Transformer encoder -> final time-step representation -> GDP forecast

    No explicit feature-weighting mechanism is included here. This is therefore
    the architectural baseline against which the proposed model will be tested.
    """
    def __init__(self, n_features):
        super().__init__()
        self.input_embedding = nn.Linear(n_features, D_MODEL)
        self.position = SinusoidalPositionalEncoding(D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.forecast_head = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        # Scale embeddings by sqrt(d_model), following standard Transformer practice.
        x = self.input_embedding(x) * math.sqrt(D_MODEL)
        x = self.position(x)
        x = self.encoder(x)

        # The final encoded quarter summarises the preceding 8-quarter context.
        x = self.output_norm(x[:, -1, :])
        return self.forecast_head(x).squeeze(-1)


# ============================================================
# DATA LOADING
# ============================================================
def load_split(feature_set, split):
    """Load one scaled sequence split prepared by Script 08."""
    path = MODEL_DATA_DIR / f"{feature_set}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing model-data file: {path}")

    data = np.load(path)
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "y_raw": data["y_raw"].astype(np.float64),
        "target_dates": pd.to_datetime(data["target_dates"]),
    }


def load_target_scaler(feature_set):
    """Read training-only target scaling parameters created by Script 08."""
    path = MODEL_DATA_DIR / f"{feature_set}_target_scaler.csv"
    scaler = pd.read_csv(path).iloc[0]
    return float(scaler["training_mean"]), float(scaler["training_std"])


def make_loader(X, y, shuffle, seed):
    """Create a small deterministic DataLoader for quarterly sequences."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


# ============================================================
# TRAINING + PREDICTION
# ============================================================
def evaluate_loss(model, X, y, criterion):
    """Calculate MSE without changing model weights."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X).to(DEVICE)
        y_tensor = torch.from_numpy(y).to(DEVICE)
        return float(criterion(model(x_tensor), y_tensor).item())


def predict_scaled(model, X):
    """Return predictions in scaled target units."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X).to(DEVICE)
        return model(x_tensor).cpu().numpy()


def train_one_run(feature_set, seed, train, val):
    """
    Train one Standard Transformer run.

    Early stopping monitors validation MSE only. The test set is never consulted
    during fitting or epoch selection.
    """
    set_seed(seed)
    model = StandardTransformer(n_features=train["X"].shape[2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(train["X"], train["y"], shuffle=True, seed=seed)
    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()

            # Gradient clipping improves stability on the small, volatile GDP sample.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_loss(model, val["X"], val["y"], criterion)

        history.append({
            "feature_set": feature_set,
            "model_label": MODEL_LABELS[feature_set],
            "seed": seed,
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "validation_mse_scaled": val_loss,
        })

        # Save only genuine validation improvement; test results never influence this.
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No valid model state produced for {feature_set}, seed={seed}")

    model.load_state_dict(best_state)
    return model, best_epoch, best_val, history


# ============================================================
# METRICS
# ============================================================
def regression_metrics(y_true, y_pred):
    """
    Report metrics in original GDP-growth percentage units.

    MAPE is intentionally excluded because quarterly GDP growth can be close to
    zero, making percentage errors unstable and potentially misleading.
    """
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.where(
        denominator > 1e-8,
        200 * np.abs(error) / denominator,
        0.0,
    )
    smape = float(np.mean(smape_terms))

    direction = float(100 * np.mean(np.sign(y_true) == np.sign(y_pred)))
    bias = float(np.mean(error))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "sMAPE": smape,
        "direction_accuracy_pct": direction,
        "mean_error_bias": bias,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================
run_rows = []
history_rows = []
prediction_frames = []
parameter_rows = []

print("\n" + "=" * 75)
print("09 - STANDARD TRANSFORMER")
print("=" * 75)
print(f"\nDevice: {DEVICE}")
print(f"Repeated runs per experiment: {len(SEEDS)}")

for feature_set in FEATURE_SETS:
    print("\n" + "-" * 75)
    print(f"{MODEL_LABELS[feature_set]} | {feature_set}")
    print("-" * 75)

    train = load_split(feature_set, "train")
    val = load_split(feature_set, "validation")
    test = load_split(feature_set, "test")
    target_mean, target_std = load_target_scaler(feature_set)

    # Record parameter count once per feature set.
    temp_model = StandardTransformer(n_features=train["X"].shape[2])
    parameter_rows.append({
        "feature_set": feature_set,
        "model_label": MODEL_LABELS[feature_set],
        "input_channels": train["X"].shape[2],
        "lookback_quarters": train["X"].shape[1],
        "trainable_parameters": sum(p.numel() for p in temp_model.parameters() if p.requires_grad),
    })
    del temp_model

    set_predictions = []

    for run_number, seed in enumerate(SEEDS, start=1):
        model, best_epoch, best_val, history = train_one_run(
            feature_set, seed, train, val
        )
        history_rows.extend(history)

        val_scaled = predict_scaled(model, val["X"])
        test_scaled = predict_scaled(model, test["X"])

        # Inverse-transform forecasts to interpretable GDP growth units.
        val_pred = val_scaled * target_std + target_mean
        test_pred = test_scaled * target_std + target_mean

        val_metrics = regression_metrics(val["y_raw"], val_pred)
        test_metrics = regression_metrics(test["y_raw"], test_pred)

        row = {
            "feature_set": feature_set,
            "model_label": MODEL_LABELS[feature_set],
            "run": run_number,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_mse_scaled": best_val,
        }
        row.update({f"validation_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        run_rows.append(row)

        set_predictions.append(test_pred)

        # Save the best validation checkpoint for every seed for reproducibility.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_set": feature_set,
                "model_label": MODEL_LABELS[feature_set],
                "seed": seed,
                "input_channels": train["X"].shape[2],
                "config": {
                    "d_model": D_MODEL,
                    "n_heads": N_HEADS,
                    "n_layers": N_LAYERS,
                    "d_ff": D_FF,
                    "dropout": DROPOUT,
                },
            },
            MODEL_DIR / f"{MODEL_LABELS[feature_set]}_seed_{seed}.pt",
        )

        print(
            f"Run {run_number:02d}/{len(SEEDS)} | seed={seed:>3} | "
            f"best epoch={best_epoch:>3} | "
            f"test RMSE={test_metrics['RMSE']:.4f} | "
            f"MAE={test_metrics['MAE']:.4f}"
        )

    # Aggregate predictions across repeated runs; this is for stability reporting,
    # not a separately tuned ensemble model.
    prediction_matrix = np.vstack(set_predictions)
    pred_mean = prediction_matrix.mean(axis=0)
    pred_std = prediction_matrix.std(axis=0, ddof=0)

    prediction_frames.append(pd.DataFrame({
        "feature_set": feature_set,
        "model_label": MODEL_LABELS[feature_set],
        "target_date": test["target_dates"],
        "actual_gdp_growth": test["y_raw"],
        "prediction_mean": pred_mean,
        "prediction_std": pred_std,
    }))

    # test forecast figure for this feature set.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test["target_dates"], test["y_raw"], marker="o", label="Actual")
    ax.plot(test["target_dates"], pred_mean, marker="o", label="Mean prediction")
    ax.fill_between(
        test["target_dates"],
        pred_mean - pred_std,
        pred_mean + pred_std,
        alpha=0.2,
        label="±1 run SD",
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_title(f"{MODEL_LABELS[feature_set]}: One-Quarter-Ahead GDP Forecasts")
    ax.set_xlabel("Target quarter")
    ax.set_ylabel("GDP QoQ growth (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{MODEL_LABELS[feature_set]}_test_predictions.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

# ============================================================
# SAVE RESULTS
# ============================================================
run_df = pd.DataFrame(run_rows)
history_df = pd.DataFrame(history_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)
parameter_df = pd.DataFrame(parameter_rows)

run_df.to_csv(RUN_METRICS_OUT, index=False)
history_df.to_csv(HISTORY_OUT, index=False)
predictions_df.to_csv(PREDICTIONS_OUT, index=False, date_format="%Y-%m-%d")
parameter_df.to_csv(RESULT_DIR / "standard_transformer_parameter_counts.csv", index=False)

# Summarise repeated-run performance using mean and standard deviation.
metric_cols = [
    "test_RMSE", "test_MAE", "test_sMAPE",
    "test_direction_accuracy_pct", "test_mean_error_bias",
]
summary_rows = []

for feature_set, group in run_df.groupby("feature_set"):
    row = {
        "feature_set": feature_set,
        "model_label": MODEL_LABELS[feature_set],
        "runs": len(group),
        "mean_best_epoch": group["best_epoch"].mean(),
    }
    for metric in metric_cols:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)

# Save configuration so dissertation experiments can be exactly reproduced.
config = {
    "feature_sets": FEATURE_SETS,
    "model_labels": MODEL_LABELS,
    "seeds": SEEDS,
    "architecture": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "feedforward_dimension": D_FF,
        "dropout": DROPOUT,
        "positional_encoding": "sinusoidal",
        "sequence_summary": "final_encoder_time_step",
    },
    "training": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "early_stopping_min_delta": MIN_DELTA,
        "loss": "MSE",
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
    },
}
with open(CONFIG_OUT, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# comparison of repeated-run test RMSE.
fig, ax = plt.subplots(figsize=(8, 5))
rmse_data = [
    run_df.loc[run_df["feature_set"] == feature_set, "test_RMSE"].to_numpy()
    for feature_set in FEATURE_SETS
]
ax.boxplot(rmse_data, tick_labels=[MODEL_LABELS[s] for s in FEATURE_SETS])
ax.set_title("Standard Transformer Test RMSE Across Repeated Runs")
ax.set_ylabel("RMSE")
ax.tick_params(axis="x", rotation=10)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "standard_transformer_rmse_across_runs.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("STANDARD TRANSFORMER EXPERIMENT COMPLETE")
print("=" * 75)

print("\nArchitecture:")
print(f"  d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, "
      f"feedforward={D_FF}, dropout={DROPOUT}")
print(f"  Runs per feature set: {len(SEEDS)}")
print("\nParameter counts:\n")
print(parameter_df.to_string(index=False))

print("\nRepeated-run test summary:\n")
display_cols = [
    "model_label", "runs",
    "test_RMSE_mean", "test_RMSE_std",
    "test_MAE_mean", "test_MAE_std",
    "test_sMAPE_mean", "test_direction_accuracy_pct_mean",
]
print(summary_df[display_cols].to_string(index=False))

print("\nImportant:")
print("  Test data were not used for training, early stopping or model selection.")
print("  MAPE is omitted because GDP growth can be close to zero.")
print("  The proposed feature-aware Transformer must use the same B-expanded")
print("  data, seeds, training procedure and core Transformer dimensions.")

print(f"\nResults:\n  {RESULT_DIR}")
print(f"\n300-DPI figures:\n  {FIGURE_DIR}")


if __name__ == "__main__":
    pass
