from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
DESIGN_DIR = PROJECT_ROOT / "data" / "processed" / "experiment_design"
MODEL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_data"
FIGURE_DIR = PROJECT_ROOT / "figures" / "model_data"

DATA_PATH = FEATURE_DIR / "engineered_features_target_window.csv"
SET_PATH = DESIGN_DIR / "experiment_feature_sets.csv"
AUDIT_OUT = MODEL_DATA_DIR / "model_data_preparation_audit.csv"

MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "gdp_growth_qoq"
LOOKBACK = 8
FORECAST_HORIZON = 1
FIGURE_DPI = 300

# Target-date split boundaries are kept identical to Script 07.
TRAIN_END = pd.Timestamp("2013-10-01")
VAL_START = pd.Timestamp("2014-01-01")
VAL_END = pd.Timestamp("2018-10-01")
TEST_START = pd.Timestamp("2019-01-01")

PRIMARY_SETS = ["A_basic", "B_expanded_refined"]

# ============================================================
# LOAD DATA AND LOCKED FEATURE SETS
# ============================================================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

set_df = pd.read_csv(SET_PATH)
feature_sets = {}
for set_name in PRIMARY_SETS:
    sub = set_df[set_df["feature_set"] == set_name].sort_values("feature_order")
    if sub.empty:
        raise ValueError(f"Feature set '{set_name}' not found in {SET_PATH}")
    feature_sets[set_name] = sub["feature_name"].tolist()

# A and B must use exactly the same quarterly observations for a fair ablation.
# Because B contains A plus additional features, joint completeness is defined
# using the union of all primary predictors and the GDP target.
joint_features = sorted(set(feature_sets["A_basic"] + feature_sets["B_expanded_refined"]))
primary = df[["date", TARGET] + joint_features].dropna().copy().reset_index(drop=True)

if primary.empty:
    raise ValueError("No joint complete-case sample exists for A and B.")

# ============================================================
# HELPERS
# ============================================================
def assign_split(target_date):
    """Assign a forecast target date to the fixed chronological split."""
    if target_date <= TRAIN_END:
        return "train"
    if VAL_START <= target_date <= VAL_END:
        return "validation"
    if target_date >= TEST_START:
        return "test"
    return "unassigned"


def build_sequences(data, input_cols):
    """
    Build leakage-safe one-quarter-ahead sequences.

    Historical GDP growth is included as an INPUT channel because past GDP is
    known at the forecast origin and provides an autoregressive signal. The model
    still predicts GDP at t+1, so future target information is never included.
    """
    X, y, rows = [], [], []

    for target_idx in range(LOOKBACK + FORECAST_HORIZON - 1, len(data)):
        input_end_idx = target_idx - FORECAST_HORIZON
        input_start_idx = input_end_idx - LOOKBACK + 1
        if input_start_idx < 0:
            continue

        # Require a strictly consecutive quarterly window; no gap may be treated
        # as though two non-adjacent quarters were consecutive observations.
        dates = data.loc[input_start_idx:target_idx, "date"].dt.to_period("Q")
        expected = pd.period_range(dates.iloc[0], dates.iloc[-1], freq="Q")
        if len(dates) != len(expected) or not np.array_equal(
            dates.astype(str).to_numpy(), expected.astype(str).to_numpy()
        ):
            continue

        x_values = data.loc[input_start_idx:input_end_idx, input_cols].to_numpy(dtype=float)
        target_value = float(data.loc[target_idx, TARGET])

        # Missing values are not imputed here. Any missing sequence is rejected so
        # later model comparisons are based on identical observed information.
        if not np.isfinite(x_values).all() or not np.isfinite(target_value):
            continue

        X.append(x_values)
        y.append(target_value)
        rows.append({
            "input_start": data.loc[input_start_idx, "date"],
            "input_end": data.loc[input_end_idx, "date"],
            "target_date": data.loc[target_idx, "date"],
            "split": assign_split(data.loc[target_idx, "date"]),
        })

    return np.asarray(X), np.asarray(y), pd.DataFrame(rows)


def training_scaler(data, input_cols, metadata):
    """
    Fit feature scaling ONLY on unique quarterly observations used by training inputs.

    Fitting on the full dataset would leak validation/test distribution information.
    Using unique quarters also avoids overweighting observations repeated across
    overlapping sequence windows.
    """
    train_meta = metadata[metadata["split"] == "train"]
    if train_meta.empty:
        raise ValueError("No training sequences available.")

    scaler_start = train_meta["input_start"].min()
    scaler_end = train_meta["input_end"].max()
    train_rows = data[
        (data["date"] >= scaler_start) &
        (data["date"] <= scaler_end)
    ][input_cols]

    mean = train_rows.mean()
    std = train_rows.std(ddof=0)

    # A zero-variance feature cannot be standardised; using 1 keeps it centred
    # without creating division-by-zero or infinite model inputs.
    std = std.mask(std == 0, 1.0)
    return mean, std, scaler_start, scaler_end


def scale_X(X, mean, std):
    """Apply training-only z-score parameters to a 3-D sequence tensor."""
    return (X - mean.to_numpy()[None, None, :]) / std.to_numpy()[None, None, :]


# ============================================================
# PREPARE MODEL DATA
# ============================================================
audit_rows = []
target_dates_by_set = {}

for set_name, predictors in feature_sets.items():
    # Lagged GDP is added as the first input channel; the remaining channels are
    # the locked exogenous/macroeconomic predictors from Script 07.
    input_cols = [TARGET] + predictors
    X_raw, y_raw, meta = build_sequences(primary, input_cols)

    if len(meta) == 0:
        raise ValueError(f"No valid sequences produced for {set_name}.")

    # Fit input scaling only on quarters that appear in training input windows.
    x_mean, x_std, scaler_start, scaler_end = training_scaler(primary, input_cols, meta)
    X_scaled = scale_X(X_raw, x_mean, x_std)

    # Scale the target using TRAINING TARGETS only; predictions can later be
    # inverse-transformed to GDP growth percentage units for evaluation.
    train_mask = meta["split"].eq("train").to_numpy()
    y_train_mean = float(y_raw[train_mask].mean())
    y_train_std = float(y_raw[train_mask].std(ddof=0))
    if y_train_std == 0:
        y_train_std = 1.0
    y_scaled = (y_raw - y_train_mean) / y_train_std

    # Save ordered input channels because feature order must be identical when
    # the arrays are loaded by LSTM/Transformer training scripts.
    pd.DataFrame({
        "input_order": np.arange(1, len(input_cols) + 1),
        "input_feature": input_cols,
        "role": ["lagged_target"] + ["predictor"] * len(predictors),
    }).to_csv(MODEL_DATA_DIR / f"{set_name}_input_features.csv", index=False)

    # Store scaler parameters in a transparent CSV rather than hiding them in
    # a binary object; this makes the preprocessing easy to audit in the thesis.
    pd.DataFrame({
        "input_feature": input_cols,
        "training_mean": x_mean.to_numpy(),
        "training_std": x_std.to_numpy(),
    }).to_csv(MODEL_DATA_DIR / f"{set_name}_input_scaler.csv", index=False)

    pd.DataFrame([{
        "target": TARGET,
        "training_mean": y_train_mean,
        "training_std": y_train_std,
    }]).to_csv(MODEL_DATA_DIR / f"{set_name}_target_scaler.csv", index=False)

    # Save one NPZ per split. Both scaled and raw targets are retained so model
    # training can use scaled y while final metrics are reported in original units.
    split_counts = {}
    for split_name in ["train", "validation", "test"]:
        mask = meta["split"].eq(split_name).to_numpy()
        split_counts[split_name] = int(mask.sum())

        np.savez_compressed(
            MODEL_DATA_DIR / f"{set_name}_{split_name}.npz",
            X=X_scaled[mask],
            y=y_scaled[mask],
            y_raw=y_raw[mask],
            target_dates=meta.loc[mask, "target_date"].to_numpy(dtype="datetime64[D]"),
            input_start_dates=meta.loc[mask, "input_start"].to_numpy(dtype="datetime64[D]"),
            input_end_dates=meta.loc[mask, "input_end"].to_numpy(dtype="datetime64[D]"),
        )

    # Sequence-date metadata provides a human-readable audit trail.
    meta.to_csv(
        MODEL_DATA_DIR / f"{set_name}_sequence_dates.csv",
        index=False,
        date_format="%Y-%m-%d"
    )
    target_dates_by_set[set_name] = meta["target_date"].to_numpy()

    audit_rows.append({
        "feature_set": set_name,
        "exogenous_predictors": len(predictors),
        "input_channels_including_lagged_gdp": len(input_cols),
        "lookback_quarters": LOOKBACK,
        "forecast_horizon_quarters": FORECAST_HORIZON,
        "total_sequences": len(meta),
        "train_sequences": split_counts["train"],
        "validation_sequences": split_counts["validation"],
        "test_sequences": split_counts["test"],
        "scaler_fit_start": scaler_start.date(),
        "scaler_fit_end": scaler_end.date(),
        "first_target": meta["target_date"].min().date(),
        "last_target": meta["target_date"].max().date(),
    })

# A and B must have exactly the same forecast target dates; otherwise the feature
# ablation and architecture comparisons would not be directly comparable.
if not np.array_equal(
    target_dates_by_set["A_basic"],
    target_dates_by_set["B_expanded_refined"]
):
    raise ValueError("A_basic and B_expanded_refined do not have identical target dates.")

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_OUT, index=False)

# ============================================================
# FIGURE: MODEL SAMPLE COUNTS
# ============================================================
plot_df = audit_df.set_index("feature_set")[["train_sequences", "validation_sequences", "test_sequences"]]
fig, ax = plt.subplots(figsize=(9, 5))
plot_df.plot(kind="bar", ax=ax)
ax.set_title("Prepared Forecast Sequences by Data Split")
ax.set_xlabel("Feature set")
ax.set_ylabel("Number of sequences")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "prepared_sequence_split_counts.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)
plt.close(fig)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("08 - MODEL DATA PREPARATION COMPLETE")
print("=" * 75)

print(f"\nCommon quarterly data: {primary['date'].min().date()} to {primary['date'].max().date()}")
print(f"Lookback: {LOOKBACK} quarters")
print(f"Forecast horizon: {FORECAST_HORIZON} quarter")
print("\nImportant design choice:")
print("  Historical GDP growth is included as an input channel.")
print("  GDP at the forecast target quarter is never included in the input.")

print("\nPrepared datasets:\n")
print(audit_df.to_string(index=False))

print("\nA/B target-date equality check: PASSED")
print("\nScaling:")
print("  Input scalers fitted only on unique training-input quarters.")
print("  Target scalers fitted only on training targets.")
print("  No validation/test observations were used to estimate scaling parameters.")

print(f"\nOutputs:\n  {MODEL_DATA_DIR}")
print(f"\nAudit:\n  {AUDIT_OUT}")
print(f"\n300-DPI figure:\n  {FIGURE_DIR / 'prepared_sequence_split_counts.png'}")


if __name__ == "__main__":
    pass
