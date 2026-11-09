from pathlib import Path
import json

import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

ALIGNED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "brazil_release_alignment"
    / "brazil_common_release_aligned.csv"
)

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "brazil_features"
    / "brazil_engineered_features.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "brazil_model_data"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NPZ_OUT = (
    OUT_DIR
    / "brazil_q0_o1_model_data.npz"
)

TRAIN_INDEX_OUT = (
    OUT_DIR
    / "brazil_train_sequence_index.csv"
)

TEST_INDEX_OUT = (
    OUT_DIR
    / "brazil_test_sequence_index.csv"
)

SCALER_OUT = (
    OUT_DIR
    / "brazil_training_scalers.json"
)

CONFIG_OUT = (
    OUT_DIR
    / "brazil_model_data_config.json"
)

SUMMARY_OUT = (
    OUT_DIR
    / "brazil_model_data_summary.txt"
)

# ============================================================
# FROZEN CROSS-COUNTRY DESIGN
# ============================================================
LOOKBACK = 8
HORIZON = 1

# Keep lagged GDP growth as the FIRST input channel, matching the frozen
# South African Q0/O1 representation used in Scripts 10I and 11B.
# This does not change the data or sample size; it only fixes channel order
# so the cross-country representation is consistent.
FEATURES = [
    "gdp_growth_qoq",
    "headline_inflation_yoy",
    "policy_rate_change_pp",
    "m3_growth_qoq",
    "usd_brl_change_qoq",
    "imports_growth_qoq",
    "investment_growth_qoq",
    "private_consumption_growth_qoq",
    "manufacturing_growth",
    "share_price_return_qoq",
    "reer_change_qoq",
    "capacity_utilisation",
]

TARGET = "gdp_growth_qoq"

# Same calendar evaluation period used in the South African final evaluation.
TRAIN_TARGET_END = pd.Timestamp("2018-10-01")
TEST_TARGET_START = pd.Timestamp("2019-01-01")
TEST_TARGET_END = pd.Timestamp("2025-10-01")

# Frozen final-evaluation training settings from the South African O1/Q0 phase.
SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]

Q0_FIXED_EPOCHS = 4
O1_FIXED_EPOCHS = 10

# ============================================================
# HELPERS
# ============================================================
def quarter_diff_is_one(a, b):
    """
    True when b is exactly one quarter after a.
    """
    return (
        b.to_period("Q")
        == a.to_period("Q") + 1
    )


def is_consecutive_quarter_window(dates):
    dates = list(
        pd.to_datetime(dates)
    )

    if len(dates) <= 1:
        return True

    return all(
        quarter_diff_is_one(
            dates[i],
            dates[i + 1],
        )
        for i in range(
            len(dates) - 1
        )
    )


def fit_standard_scaler_from_rows(df, feature_cols):
    """
    Fit feature-wise mean/std using UNIQUE training input quarters only.
    This avoids overweighting quarters simply because rolling windows overlap.
    """
    means = {}
    stds = {}

    for col in feature_cols:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        mean = float(
            values.mean()
        )

        std = float(
            values.std(
                ddof=0
            )
        )

        if not np.isfinite(std) or std == 0:
            std = 1.0

        means[col] = mean
        stds[col] = std

    return means, stds


def scale_X(X, feature_cols, means, stds):
    X_scaled = X.astype(
        np.float32
    ).copy()

    for j, col in enumerate(
        feature_cols
    ):
        X_scaled[:, :, j] = (
            X_scaled[:, :, j]
            - means[col]
        ) / stds[col]

    return X_scaled


def target_scaler(y):
    mean = float(
        np.mean(y)
    )

    std = float(
        np.std(
            y,
            ddof=0,
        )
    )

    if not np.isfinite(std) or std == 0:
        std = 1.0

    return mean, std


# ============================================================
# LOAD DATA
# ============================================================
if not ALIGNED_FILE.exists():
    raise FileNotFoundError(
        f"Missing release-aligned Brazil file:\n{ALIGNED_FILE}\n"
        "Run Script 17 first."
    )

if not TARGET_FILE.exists():
    raise FileNotFoundError(
        f"Missing engineered Brazil target file:\n{TARGET_FILE}\n"
        "Run Script 16 first."
    )

aligned = pd.read_csv(
    ALIGNED_FILE
)

targets = pd.read_csv(
    TARGET_FILE
)

aligned["date"] = pd.to_datetime(
    aligned["date"],
    errors="coerce",
)

targets["date"] = pd.to_datetime(
    targets["date"],
    errors="coerce",
)

aligned = (
    aligned.dropna(
        subset=["date"]
    )
    .sort_values("date")
    .drop_duplicates(
        subset=["date"],
        keep="last",
    )
    .reset_index(drop=True)
)

targets = (
    targets.dropna(
        subset=["date"]
    )
    .sort_values("date")
    .drop_duplicates(
        subset=["date"],
        keep="last",
    )
    .reset_index(drop=True)
)

missing_features = [
    col
    for col in FEATURES
    if col not in aligned.columns
]

if missing_features:
    raise KeyError(
        "Missing aligned model features:\n"
        + "\n".join(
            missing_features
        )
    )

if FEATURES[0] != TARGET:
    raise ValueError(
        "Frozen Q0/O1 representation requires lagged GDP growth "
        "as the first input channel."
    )

if TARGET not in targets.columns:
    raise KeyError(
        f"Target {TARGET!r} missing from {TARGET_FILE.name}"
    )

# ######
# Inputs come from the RELEASE-ALIGNED table.
# Targets come from the UNALIGNED engineered GDP-growth table.
#
# This prevents the release-aligned historical GDP input channel from being
# accidentally used as the forecast target.
target_lookup = (
    targets.set_index(
        "date"
    )[TARGET]
)

# ============================================================
# BUILD 8-QUARTER -> NEXT-QUARTER SEQUENCES
# ============================================================
X_rows = []
y_rows = []
meta_rows = []

for end_idx in range(
    LOOKBACK - 1,
    len(aligned),
):
    start_idx = (
        end_idx
        - LOOKBACK
        + 1
    )

    window = aligned.iloc[
        start_idx:
        end_idx + 1
    ].copy()

    input_dates = window[
        "date"
    ].tolist()

    if not is_consecutive_quarter_window(
        input_dates
    ):
        continue

    forecast_origin = (
        input_dates[-1]
    )

    target_date = (
        forecast_origin
        .to_period("Q")
        + HORIZON
    ).start_time

    # Keep the final cross-country evaluation within 2025Q4.
    if target_date > TEST_TARGET_END:
        continue

    if target_date not in target_lookup.index:
        continue

    x = (
        window[
            FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    y = pd.to_numeric(
        pd.Series(
            [
                target_lookup.loc[
                    target_date
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if (
        not np.isfinite(x).all()
        or not np.isfinite(y)
    ):
        continue

    X_rows.append(
        x
    )

    y_rows.append(
        float(y)
    )

    meta_rows.append({
        "sequence_start":
            input_dates[0],
        "sequence_end":
            forecast_origin,
        "target_date":
            target_date,
        "target_gdp_growth_qoq":
            float(y),
    })

if not X_rows:
    raise RuntimeError(
        "No valid Brazil modelling sequences were created."
    )

X_all = np.stack(
    X_rows
).astype(
    np.float32
)

y_all = np.asarray(
    y_rows,
    dtype=np.float32,
)

meta = pd.DataFrame(
    meta_rows
)

# ============================================================
# CHRONOLOGICAL FINAL-EVALUATION SPLIT
# ============================================================
train_mask = (
    meta["target_date"]
    <= TRAIN_TARGET_END
)

test_mask = (
    (meta["target_date"] >= TEST_TARGET_START)
    & (meta["target_date"] <= TEST_TARGET_END)
)

# No validation subset is created:
# architecture, hyperparameters and fixed epoch counts were frozen in South Africa.
if (
    train_mask.sum() == 0
    or test_mask.sum() == 0
):
    raise RuntimeError(
        "Train/test split produced an empty partition."
    )

# Ensure every eligible sequence belongs to exactly one of the intended periods.
unexpected = (
    ~(train_mask | test_mask)
)

if unexpected.any():
    unexpected_dates = (
        meta.loc[
            unexpected,
            "target_date",
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )

    raise RuntimeError(
        "Found sequences outside the intended train/test periods:\n"
        + "\n".join(
            unexpected_dates
        )
    )

X_train_raw = (
    X_all[
        train_mask.to_numpy()
    ]
)

y_train_raw = (
    y_all[
        train_mask.to_numpy()
    ]
)

X_test_raw = (
    X_all[
        test_mask.to_numpy()
    ]
)

y_test_raw = (
    y_all[
        test_mask.to_numpy()
    ]
)

train_meta = (
    meta.loc[
        train_mask
    ]
    .reset_index(
        drop=True
    )
)

test_meta = (
    meta.loc[
        test_mask
    ]
    .reset_index(
        drop=True
    )
)

# ============================================================
# FIT INPUT SCALERS USING UNIQUE TRAINING INPUT QUARTERS ONLY
# ============================================================
train_input_dates = pd.DatetimeIndex(
    sorted(
        set(
            date
            for _, row in train_meta.iterrows()
            for date in pd.period_range(
                start=row[
                    "sequence_start"
                ],
                end=row[
                    "sequence_end"
                ],
                freq="Q",
            ).start_time
        )
    )
)

train_input_rows = (
    aligned.set_index(
        "date"
    )
    .reindex(
        train_input_dates
    )
)

if (
    train_input_rows[
        FEATURES
    ]
    .isna()
    .any()
    .any()
):
    raise RuntimeError(
        "Unexpected missing values in unique training input quarters."
    )

feature_means, feature_stds = (
    fit_standard_scaler_from_rows(
        train_input_rows,
        FEATURES,
    )
)

target_mean, target_std = (
    target_scaler(
        y_train_raw
    )
)

X_train = scale_X(
    X_train_raw,
    FEATURES,
    feature_means,
    feature_stds,
)

X_test = scale_X(
    X_test_raw,
    FEATURES,
    feature_means,
    feature_stds,
)

y_train = (
    (
        y_train_raw
        - target_mean
    )
    / target_std
).astype(
    np.float32
)

y_test = (
    (
        y_test_raw
        - target_mean
    )
    / target_std
).astype(
    np.float32
)

# ============================================================
# LEAKAGE / CONSISTENCY ASSERTIONS
# ============================================================
assert (
    train_meta[
        "target_date"
    ].max()
    <= TRAIN_TARGET_END
)

assert (
    test_meta[
        "target_date"
    ].min()
    >= TEST_TARGET_START
)

assert (
    train_meta[
        "target_date"
    ].max()
    < test_meta[
        "target_date"
    ].min()
)

assert (
    train_input_dates.max()
    < test_meta[
        "target_date"
    ].min()
)

assert (
    X_train.shape[1]
    == LOOKBACK
)

assert (
    X_train.shape[2]
    == len(FEATURES)
)

assert (
    X_test.shape[1:]
    == X_train.shape[1:]
)

# ============================================================
# SAVE ARRAYS
# ============================================================
np.savez_compressed(
    NPZ_OUT,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_train_raw=X_train_raw,
    y_train_raw=y_train_raw,
    X_test_raw=X_test_raw,
    y_test_raw=y_test_raw,
    train_target_dates=(
        train_meta[
            "target_date"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    ),
    test_target_dates=(
        test_meta[
            "target_date"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    ),
    feature_names=np.asarray(
        FEATURES,
        dtype=str,
    ),
)

train_meta.to_csv(
    TRAIN_INDEX_OUT,
    index=False,
)

test_meta.to_csv(
    TEST_INDEX_OUT,
    index=False,
)

scaler_payload = {
    "input_scaler": {
        "fitted_on":
            "unique release-aligned quarters appearing in training input windows only",
        "train_input_start":
            str(
                train_input_dates.min()
            ),
        "train_input_end":
            str(
                train_input_dates.max()
            ),
        "n_unique_train_input_quarters":
            len(
                train_input_dates
            ),
        "feature_means":
            feature_means,
        "feature_stds":
            feature_stds,
    },
    "target_scaler": {
        "fitted_on":
            "training GDP-growth targets only",
        "target_mean":
            target_mean,
        "target_std":
            target_std,
    },
}

with open(
    SCALER_OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        scaler_payload,
        f,
        indent=2,
    )

config = {
    "country":
        "Brazil",
    "primary_cross_country_feature_set":
        "A_common_strict_11 + historical GDP growth",
    "feature_order":
        FEATURES,
    "exogenous_features":
        11,
    "total_input_channels":
        len(FEATURES),
    "lookback_quarters":
        LOOKBACK,
    "forecast_horizon":
        HORIZON,
    "target":
        TARGET,
    "target_definition":
        "unshifted next-quarter real GDP quarter-on-quarter growth",
    "input_definition":
        "release-aligned quarterly information set",
    "train_target_end":
        str(
            TRAIN_TARGET_END.date()
        ),
    "test_target_start":
        str(
            TEST_TARGET_START.date()
        ),
    "test_target_end":
        str(
            TEST_TARGET_END.date()
        ),
    "validation_split_used":
        False,
    "reason_no_validation_split":
        (
            "Q0/O1 architecture, optimizer settings, seeds and fixed epoch "
            "counts are frozen from South African development."
        ),
    "seeds":
        SEEDS,
    "q0_fixed_epochs":
        Q0_FIXED_EPOCHS,
    "o1_fixed_epochs":
        O1_FIXED_EPOCHS,
    "important_cross_country_requirement":
        (
            "For the primary cross-country comparison, South Africa must also "
            "be evaluated with the same common feature concepts and feature count. "
            "The existing 23-channel South African final result should not be "
            "treated as directly comparable to this 12-channel Brazil experiment."
        ),
}

with open(
    CONFIG_OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )

# ============================================================
# SUMMARY
# ============================================================
summary_lines = [
    "=" * 78,
    "BRAZIL MODEL DATA PREPARATION",
    "=" * 78,
    "",
    f"Input channels: {len(FEATURES)}",
    f"Lookback: {LOOKBACK} quarters",
    f"Forecast horizon: {HORIZON} quarter",
    "",
    f"All valid sequences through 2025Q4: {len(meta)}",
    "",
    f"Training sequences: {len(train_meta)}",
    f"Training target dates: {train_meta['target_date'].min().date()} -> {train_meta['target_date'].max().date()}",
    "",
    f"Test sequences: {len(test_meta)}",
    f"Test target dates: {test_meta['target_date'].min().date()} -> {test_meta['target_date'].max().date()}",
    "",
    f"Unique training input quarters used for scaling: {len(train_input_dates)}",
    f"Input-scaler period: {train_input_dates.min().date()} -> {train_input_dates.max().date()}",
    "",
    f"Target scaler mean: {target_mean:.8f}",
    f"Target scaler std: {target_std:.8f}",
    "",
    "Important:",
    "  - Inputs are release-aligned.",
    "  - Forecast targets are unshifted next-quarter GDP growth.",
    "  - Test targets are never used in scaling.",
    "  - No validation set or Brazil-specific hyperparameter tuning is used.",
    "  - Q0 fixed epochs = 4; O1 fixed epochs = 10.",
    "  - Primary cross-country comparison requires rerunning South Africa",
    "    with the same common 12-channel information set.",
]

SUMMARY_OUT.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)

# ============================================================
# CONSOLE OUTPUT
# ============================================================
print("\n" + "=" * 78)
print("18 - BRAZIL MODEL DATA PREPARATION")
print("=" * 78)

print(
    f"\nFeature channels: {len(FEATURES)}"
)

for i, feature in enumerate(
    FEATURES,
    start=1,
):
    print(
        f"  {i:>2}. {feature}"
    )

print(
    f"\nSequence shape: "
    f"{LOOKBACK} quarters x "
    f"{len(FEATURES)} channels"
)

print(
    f"\nAll valid sequences through 2025Q4: "
    f"{len(meta)}"
)

print("\nTraining:")
print(
    f"  sequences:    {len(train_meta)}"
)
print(
    f"  target dates: "
    f"{train_meta['target_date'].min().date()} -> "
    f"{train_meta['target_date'].max().date()}"
)

print("\nTest:")
print(
    f"  sequences:    {len(test_meta)}"
)
print(
    f"  target dates: "
    f"{test_meta['target_date'].min().date()} -> "
    f"{test_meta['target_date'].max().date()}"
)

print("\nScaling:")
print(
    f"  unique training input quarters: "
    f"{len(train_input_dates)}"
)
print(
    f"  input-scaler period: "
    f"{train_input_dates.min().date()} -> "
    f"{train_input_dates.max().date()}"
)
print(
    f"  target mean/std: "
    f"{target_mean:.6f} / {target_std:.6f}"
)

print("\nArray shapes:")
print(
    f"  X_train: {X_train.shape}"
)
print(
    f"  y_train: {y_train.shape}"
)
print(
    f"  X_test:  {X_test.shape}"
)
print(
    f"  y_test:  {y_test.shape}"
)

print("\nFrozen final-evaluation settings:")
print(
    f"  seeds: {SEEDS}"
)
print(
    f"  Q0 epochs: {Q0_FIXED_EPOCHS}"
)
print(
    f"  O1 epochs: {O1_FIXED_EPOCHS}"
)
print(
    "  validation/early stopping: NONE"
)

print("\nLeakage checks: PASSED")

print("\nOutputs:")
print(f"  {NPZ_OUT}")
print(f"  {TRAIN_INDEX_OUT}")
print(f"  {TEST_INDEX_OUT}")
print(f"  {SCALER_OUT}")
print(f"  {CONFIG_OUT}")
print(f"  {SUMMARY_OUT}")

print("\nCross-country methodological note:")
print(
    "  Existing South African final O1/Q0 results used 23 input channels."
)
print(
    "  Brazil now uses the strict common 12-channel set."
)
print(
    "  Therefore, before making the PRIMARY SA-vs-Brazil generalisability "
    "comparison, rerun frozen Q0/O1 on South Africa using these same common "
    "feature concepts. The 23-channel SA result remains valid as the original "
    "South African country-rich experiment."
)

if __name__ == "__main__":
    pass
