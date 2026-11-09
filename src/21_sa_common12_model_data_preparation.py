from pathlib import Path
import json

import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "engineered_features_target_window.csv"
)

RELEASE_METADATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "availability"
    / "feature_release_metadata.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sa_common_model_data"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALIGNED_OUT = (
    OUT_DIR
    / "south_africa_common12_release_aligned.csv"
)

RELEASE_MAP_OUT = (
    OUT_DIR
    / "south_africa_common12_release_map.csv"
)

NPZ_OUT = (
    OUT_DIR
    / "south_africa_common12_q0_o1_model_data.npz"
)

TRAIN_INDEX_OUT = (
    OUT_DIR
    / "south_africa_common12_train_sequence_index.csv"
)

TEST_INDEX_OUT = (
    OUT_DIR
    / "south_africa_common12_test_sequence_index.csv"
)

SCALER_OUT = (
    OUT_DIR
    / "south_africa_common12_training_scalers.json"
)

CONFIG_OUT = (
    OUT_DIR
    / "south_africa_common12_model_data_config.json"
)

SUMMARY_OUT = (
    OUT_DIR
    / "south_africa_common12_model_data_summary.txt"
)

# ============================================================
# FROZEN CROSS-COUNTRY DESIGN
# ============================================================
LOOKBACK = 8
HORIZON = 1

TARGET_SOURCE = "gdp_growth_qoq"

TRAIN_TARGET_END = pd.Timestamp("2018-10-01")
TEST_TARGET_START = pd.Timestamp("2019-01-01")
TEST_TARGET_END = pd.Timestamp("2025-10-01")

SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]

Q0_FIXED_EPOCHS = 4
O1_FIXED_EPOCHS = 10

# ============================================================
# STRICT COMMON FEATURE MAP
# ============================================================
COMMON_FEATURE_MAP = [
    {
        "input_order": 1,
        "canonical_name": "gdp_growth_qoq",
        "source_feature": "gdp_growth_qoq",
        "concept": "historical real GDP growth",
    },
    {
        "input_order": 2,
        "canonical_name": "headline_inflation_yoy",
        "source_feature": "headline_inflation_yoy",
        "concept": "headline inflation",
    },
    {
        "input_order": 3,
        "canonical_name": "policy_rate_change_pp",
        "source_feature": "repo_rate_change",
        "concept": "policy-rate change",
    },
    {
        "input_order": 4,
        "canonical_name": "m3_growth_qoq",
        "source_feature": "m3_growth_qoq",
        "concept": "broad-money growth",
    },
    {
        "input_order": 5,
        "canonical_name": "usd_local_change_qoq",
        "source_feature": "usd_zar_change_qoq",
        "concept": "USD/local-currency exchange-rate change",
    },
    {
        "input_order": 6,
        "canonical_name": "imports_growth_qoq",
        "source_feature": "imports_growth_qoq",
        "concept": "real imports growth",
    },
    {
        "input_order": 7,
        "canonical_name": "investment_growth_qoq",
        "source_feature": "investment_growth_qoq",
        "concept": "real investment growth",
    },
    {
        "input_order": 8,
        "canonical_name": "private_consumption_growth_qoq",
        "source_feature": "private_consumption_growth_qoq",
        "concept": "real private-consumption growth",
    },
    {
        "input_order": 9,
        "canonical_name": "manufacturing_growth",
        "source_feature": "manufacturing_growth_qoq",
        "concept": "manufacturing production growth",
    },
    {
        "input_order": 10,
        "canonical_name": "share_price_return_qoq",
        "source_feature": "share_price_return_qoq",
        "concept": "share-price return",
    },
    {
        "input_order": 11,
        "canonical_name": "reer_change_qoq",
        "source_feature": "reer_change_qoq",
        "concept": "real effective exchange-rate change",
    },
    {
        "input_order": 12,
        "canonical_name": "capacity_utilisation",
        "source_feature": "capacity_utilisation",
        "concept": "manufacturing capacity utilisation",
    },
]

# ============================================================
# HELPERS
# ============================================================
def quarter_is_next(a, b):
    return (
        b.to_period("Q")
        == a.to_period("Q") + 1
    )


def consecutive_quarters(dates):
    dates = list(
        pd.to_datetime(
            dates
        )
    )

    return all(
        quarter_is_next(
            dates[i],
            dates[i + 1],
        )
        for i in range(
            len(dates) - 1
        )
    )


def fit_feature_scaler(df, columns):
    means = {}
    stds = {}

    for col in columns:
        s = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        mean = float(
            s.mean()
        )

        std = float(
            s.std(
                ddof=0
            )
        )

        if (
            not np.isfinite(std)
            or std == 0
        ):
            std = 1.0

        means[col] = mean
        stds[col] = std

    return means, stds


def scale_X(
    X,
    columns,
    means,
    stds,
):
    out = (
        X.astype(
            np.float32
        )
        .copy()
    )

    for j, col in enumerate(
        columns
    ):
        out[
            :,
            :,
            j,
        ] = (
            out[
                :,
                :,
                j,
            ]
            - means[col]
        ) / stds[col]

    return out


# ============================================================
# LOAD SOUTH AFRICAN FEATURES + RELEASE METADATA
# ============================================================
if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Missing South African engineered features:\n{FEATURE_PATH}"
    )

if not RELEASE_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing South African release metadata:\n{RELEASE_METADATA_PATH}\n"
        "Run Script 10g first."
    )

df = pd.read_csv(
    FEATURE_PATH
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df = (
    df.dropna(
        subset=["date"]
    )
    .sort_values("date")
    .drop_duplicates(
        subset=["date"],
        keep="last",
    )
    .reset_index(drop=True)
)

release_meta = pd.read_csv(
    RELEASE_METADATA_PATH
)

required_release_cols = {
    "feature_name",
    "approx_release_delay_days",
    "strict_quarter_end_shift_quarters",
}

missing_release_cols = (
    required_release_cols
    - set(
        release_meta.columns
    )
)

if missing_release_cols:
    raise ValueError(
        "Release metadata missing required columns: "
        f"{sorted(missing_release_cols)}"
    )

required_source_features = [
    row[
        "source_feature"
    ]
    for row in COMMON_FEATURE_MAP
]

missing_features = [
    f
    for f in required_source_features
    if f not in df.columns
]

if missing_features:
    raise KeyError(
        "South African engineered feature file is missing:\n"
        + "\n".join(
            missing_features
        )
    )

# ============================================================
# BUILD RELEASE MAP FROM EXISTING SA METADATA
# ============================================================
release_rows = []

for item in COMMON_FEATURE_MAP:
    source_feature = item[
        "source_feature"
    ]

    matched = release_meta[
        release_meta[
            "feature_name"
        ] == source_feature
    ]

    if len(matched) != 1:
        raise RuntimeError(
            f"Expected exactly one release-metadata row for "
            f"{source_feature!r}; found {len(matched)}."
        )

    meta = matched.iloc[0]

    release_rows.append({
        **item,
        "approx_release_delay_days":
            int(
                meta[
                    "approx_release_delay_days"
                ]
            ),
        "strict_quarter_shift":
            int(
                meta[
                    "strict_quarter_end_shift_quarters"
                ]
            ),
        "metadata_confidence":
            (
                meta[
                    "metadata_confidence"
                ]
                if "metadata_confidence"
                in release_meta.columns
                else ""
            ),
        "delay_basis":
            (
                meta[
                    "delay_basis"
                ]
                if "delay_basis"
                in release_meta.columns
                else ""
            ),
    })

release_map = (
    pd.DataFrame(
        release_rows
    )
    .sort_values(
        "input_order"
    )
    .reset_index(
        drop=True
    )
)

release_map.to_csv(
    RELEASE_MAP_OUT,
    index=False,
)

# ============================================================
# APPLY SAME STRICT QUARTER-END AVAILABILITY ALIGNMENT
# ============================================================
aligned = pd.DataFrame({
    "date":
        df["date"],
    # Forecast target remains the actual UNALIGNED GDP-growth observation.
    "target_gdp_growth_qoq":
        pd.to_numeric(
            df[
                TARGET_SOURCE
            ],
            errors="coerce",
        ),
})

FEATURE_NAMES = []

for _, row in release_map.iterrows():
    canonical = row[
        "canonical_name"
    ]

    source_feature = row[
        "source_feature"
    ]

    shift_q = int(
        row[
            "strict_quarter_shift"
        ]
    )

    FEATURE_NAMES.append(
        canonical
    )

    aligned[
        canonical
    ] = pd.to_numeric(
        df[
            source_feature
        ],
        errors="coerce",
    ).shift(
        shift_q
    )

if FEATURE_NAMES[0] != "gdp_growth_qoq":
    raise ValueError(
        "Lagged GDP growth must be the first frozen input channel."
    )

aligned.to_csv(
    ALIGNED_OUT,
    index=False,
)

# ============================================================
# BUILD STRICT 8Q > 1Q SEQUENCES
# ============================================================
X_rows = []
y_rows = []
meta_rows = []

target_lookup = (
    aligned.set_index(
        "date"
    )[
        "target_gdp_growth_qoq"
    ]
)

for end_idx in range(
    LOOKBACK - 1,
    len(
        aligned
    ),
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

    input_dates = (
        window[
            "date"
        ]
        .tolist()
    )

    if not consecutive_quarters(
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

    if target_date > TEST_TARGET_END:
        continue

    if (
        target_date
        not in target_lookup.index
    ):
        continue

    x = (
        window[
            FEATURE_NAMES
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
        pd.Series([
            target_lookup.loc[
                target_date
            ]
        ]),
        errors="coerce",
    ).iloc[0]

    if (
        not np.isfinite(
            x
        ).all()
        or not np.isfinite(
            y
        )
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
        "No South African common-12 sequences were created."
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
# SAME FINAL EVALUATION PERIOD AS BRAZIL
# ============================================================
train_mask = (
    meta[
        "target_date"
    ]
    <= TRAIN_TARGET_END
)

test_mask = (
    meta[
        "target_date"
    ]
    .between(
        TEST_TARGET_START,
        TEST_TARGET_END,
    )
)

unexpected = (
    ~(
        train_mask
        | test_mask
    )
)

if unexpected.any():
    raise RuntimeError(
        "Found sequence targets outside frozen train/test periods."
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

if len(
    test_meta
) != 28:
    raise RuntimeError(
        "Expected exactly 28 South African test quarters "
        f"(2019Q1-2025Q4), found {len(test_meta)}."
    )

# ============================================================
# FIT SCALERS ON UNIQUE TRAINING INPUT QUARTERS ONLY
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
        FEATURE_NAMES
    ]
    .isna()
    .any()
    .any()
):
    raise RuntimeError(
        "Unexpected missing values in South African training input quarters."
    )

feature_means, feature_stds = (
    fit_feature_scaler(
        train_input_rows,
        FEATURE_NAMES,
    )
)

target_mean = float(
    y_train_raw.mean()
)

target_std = float(
    y_train_raw.std(
        ddof=0
    )
)

if (
    not np.isfinite(
        target_std
    )
    or target_std == 0
):
    target_std = 1.0

X_train = scale_X(
    X_train_raw,
    FEATURE_NAMES,
    feature_means,
    feature_stds,
)

X_test = scale_X(
    X_test_raw,
    FEATURE_NAMES,
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
# AUDIT ASSERTIONS
# ============================================================
assert (
    X_train.shape[1]
    == LOOKBACK
)

assert (
    X_train.shape[2]
    == 12
)

assert (
    X_test.shape[1:]
    == X_train.shape[1:]
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
    test_meta[
        "target_date"
    ].min()
    == TEST_TARGET_START
)

assert (
    test_meta[
        "target_date"
    ].max()
    == TEST_TARGET_END
)

# ============================================================
# SAVE
# ============================================================
np.savez_compressed(
    NPZ_OUT,
    X_train=
        X_train,
    y_train=
        y_train,
    X_test=
        X_test,
    y_test=
        y_test,
    X_train_raw=
        X_train_raw,
    y_train_raw=
        y_train_raw,
    X_test_raw=
        X_test_raw,
    y_test_raw=
        y_test_raw,
    train_target_dates=
        train_meta[
            "target_date"
        ].to_numpy(
            dtype="datetime64[ns]"
        ),
    test_target_dates=
        test_meta[
            "target_date"
        ].to_numpy(
            dtype="datetime64[ns]"
        ),
    feature_names=
        np.asarray(
            FEATURE_NAMES,
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
            "unique release-aligned South African quarters appearing in training windows only",
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
            "South African training GDP-growth targets only",
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
        "South Africa",
    "purpose":
        "Strict common-12 cross-country rerun for direct comparison with Brazil.",
    "source_feature_file":
        str(
            FEATURE_PATH
        ),
    "release_metadata_file":
        str(
            RELEASE_METADATA_PATH
        ),
    "feature_order":
        FEATURE_NAMES,
    "source_feature_map":
        COMMON_FEATURE_MAP,
    "lookback_quarters":
        LOOKBACK,
    "forecast_horizon":
        HORIZON,
    "target":
        TARGET_SOURCE,
    "target_definition":
        "unshifted next-quarter real GDP QoQ growth",
    "input_definition":
        "release-aligned strict common 12-channel information set",
    "test_period":
        "2019Q1-2025Q4",
    "validation_split_used":
        False,
    "seeds":
        SEEDS,
    "q0_fixed_epochs":
        Q0_FIXED_EPOCHS,
    "o1_fixed_epochs":
        O1_FIXED_EPOCHS,
    "important_note":
        (
            "This rerun is for fair SA-vs-Brazil generalisability comparison. "
            "It does not replace the original 23-channel South African country-rich experiment."
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

summary_lines = [
    "=" * 78,
    "SOUTH AFRICA STRICT COMMON-12 MODEL DATA PREPARATION",
    "=" * 78,
    "",
    "Purpose:",
    "  Build the same conceptual 12-channel release-aligned information set",
    "  used in the Brazil cross-country experiment.",
    "",
    f"All valid sequences through 2025Q4: {len(meta)}",
    f"Training sequences: {len(train_meta)}",
    f"Training target dates: {train_meta['target_date'].min().date()} -> {train_meta['target_date'].max().date()}",
    f"Test sequences: {len(test_meta)}",
    f"Test target dates: {test_meta['target_date'].min().date()} -> {test_meta['target_date'].max().date()}",
    "",
    f"Unique training input quarters: {len(train_input_dates)}",
    f"Input scaler period: {train_input_dates.min().date()} -> {train_input_dates.max().date()}",
    f"Target mean/std: {target_mean:.8f} / {target_std:.8f}",
    "",
    "The original 23-channel South African result remains a separate country-rich experiment.",
]

SUMMARY_OUT.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)

# ============================================================
# CONSOLE
# ============================================================
print(
    "\n"
    + "=" * 78
)

print(
    "21 - SOUTH AFRICA STRICT COMMON-12 MODEL DATA PREPARATION"
)

print(
    "=" * 78
)

print(
    "\nCommon input order:"
)

for _, row in release_map.iterrows():
    print(
        f"  {int(row['input_order']):>2}. "
        f"{row['canonical_name']:<34} "
        f"<- {row['source_feature']:<32} "
        f"shift={int(row['strict_quarter_shift'])}"
    )

print(
    f"\nTotal channels: {len(FEATURE_NAMES)}"
)

print(
    f"Sequence shape: {LOOKBACK} quarters x {len(FEATURE_NAMES)} channels"
)

print(
    f"\nAll valid sequences through 2025Q4: {len(meta)}"
)

print(
    "\nTraining:"
)

print(
    f"  sequences:    {len(train_meta)}"
)

print(
    "  target dates: "
    f"{train_meta['target_date'].min().date()} -> "
    f"{train_meta['target_date'].max().date()}"
)

print(
    "\nTest:"
)

print(
    f"  sequences:    {len(test_meta)}"
)

print(
    "  target dates: "
    f"{test_meta['target_date'].min().date()} -> "
    f"{test_meta['target_date'].max().date()}"
)

print(
    "\nScaling:"
)

print(
    f"  unique training input quarters: {len(train_input_dates)}"
)

print(
    "  input-scaler period: "
    f"{train_input_dates.min().date()} -> "
    f"{train_input_dates.max().date()}"
)

print(
    f"  target mean/std: {target_mean:.6f} / {target_std:.6f}"
)

print(
    "\nArray shapes:"
)

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

print(
    "\nLeakage / date checks: PASSED"
)

print(
    "\nOutputs:"
)

print(
    f"  {ALIGNED_OUT}"
)

print(
    f"  {RELEASE_MAP_OUT}"
)

print(
    f"  {NPZ_OUT}"
)

print(
    f"  {TRAIN_INDEX_OUT}"
)

print(
    f"  {TEST_INDEX_OUT}"
)

print(
    f"  {SCALER_OUT}"
)

print(
    f"  {CONFIG_OUT}"
)

print(
    f"  {SUMMARY_OUT}"
)

if __name__ == "__main__":
    pass
