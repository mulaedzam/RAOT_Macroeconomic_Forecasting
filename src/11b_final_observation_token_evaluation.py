from pathlib import Path
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
ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT / "data" / "processed" / "features"
    / "engineered_features_target_window.csv"
)

INPUT_FEATURE_PATH = (
    ROOT / "data" / "processed" / "model_data"
    / "B_expanded_refined_input_features.csv"
)

RELEASE_METADATA_PATH = (
    ROOT / "data" / "processed" / "availability"
    / "feature_release_metadata.csv"
)

# Pre-test architecture-development results from Script 10I.
PRETEST_RUN_PATH = (
    ROOT / "results" / "tokenisation_ablation"
    / "10i_run_metrics.csv"
)

RESULT_DIR = (
    ROOT / "results" / "final_observation_token_evaluation"
)

FIGURE_DIR = (
    ROOT / "figures" / "final_observation_token_evaluation"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_OUT = RESULT_DIR / "11b_run_metrics.csv"
PREDICTION_OUT = RESULT_DIR / "11b_test_predictions_by_seed.csv"
MEAN_PREDICTION_OUT = RESULT_DIR / "11b_mean_test_predictions.csv"
SUMMARY_OUT = RESULT_DIR / "11b_model_summary.csv"
EPOCH_OUT = RESULT_DIR / "11b_fixed_epoch_selection.csv"
CONFIG_OUT = RESULT_DIR / "11b_config.json"

FIGURE_DPI = 300

# ============================================================
# FINAL FROZEN DESIGN
# ============================================================
TARGET = "gdp_growth_qoq"
TARGET_COL = "target_gdp_growth_qoq"

LOOKBACK = 8
FORECAST_HORIZON = 1

# Final test period is fixed and used once after architecture freeze.
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2025-10-01")  # 2025 Q4 quarter-start timestamp

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

# EXACT architecture used in Script 10I.
D_MODEL = 16
N_HEADS = 2
D_FF = 32
DROPOUT = 0.20

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CANDIDATES = {
    "Q0_aligned_quarter_token": "quarter",
    "O1_aligned_observation_token": "observation",
}

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    """Control Python, NumPy and PyTorch randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

# ============================================================
# LOAD PRE-TEST EPOCH SELECTION
# ============================================================
if not PRETEST_RUN_PATH.exists():
    raise FileNotFoundError(
        "Script 10I run metrics were not found. Expected:\n"
        f"{PRETEST_RUN_PATH}\n"
        "Run 10I first because final epoch counts must be selected "
        "from pre-2019 development results only."
    )

pretest_runs = pd.read_csv(
    PRETEST_RUN_PATH
)

required_pretest_cols = {
    "candidate",
    "best_epoch",
}

missing_pretest_cols = (
    required_pretest_cols
    - set(pretest_runs.columns)
)

if missing_pretest_cols:
    raise ValueError(
        "10I run metrics are missing columns: "
        f"{sorted(missing_pretest_cols)}"
    )

epoch_rows = []

for candidate in CANDIDATES:
    candidate_epochs = (
        pretest_runs.loc[
            pretest_runs["candidate"] == candidate,
            "best_epoch",
        ]
        .astype(float)
    )

    if candidate_epochs.empty:
        raise ValueError(
            f"No pre-test epochs found for {candidate}."
        )

    # Fixed final epoch count = rounded median best epoch across all
    # 40 pre-test fold-seed development runs for that architecture.
    fixed_epochs = int(
        max(
            1,
            round(
                float(
                    candidate_epochs.median()
                )
            ),
        )
    )

    epoch_rows.append({
        "candidate": candidate,
        "pretest_runs": int(
            len(candidate_epochs)
        ),
        "median_best_epoch": float(
            candidate_epochs.median()
        ),
        "mean_best_epoch": float(
            candidate_epochs.mean()
        ),
        "fixed_final_epochs": fixed_epochs,
    })

epoch_df = pd.DataFrame(
    epoch_rows
)

epoch_df.to_csv(
    EPOCH_OUT,
    index=False,
)

FIXED_EPOCHS = dict(
    zip(
        epoch_df["candidate"],
        epoch_df["fixed_final_epochs"],
    )
)

# ============================================================
# LOAD DATA + APPLY SAME AVAILABILITY ALIGNMENT AS 10H/10I/10J
# ============================================================
df = pd.read_csv(
    FEATURE_PATH
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df = (
    df.dropna(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

feature_order = (
    pd.read_csv(INPUT_FEATURE_PATH)
    .sort_values("input_order")
    .reset_index(drop=True)
)

INPUT_FEATURES = feature_order[
    "input_feature"
].tolist()

if (
    not INPUT_FEATURES
    or INPUT_FEATURES[0] != TARGET
):
    raise ValueError(
        "Expected lagged GDP growth as first input channel."
    )

release_meta = (
    pd.read_csv(
        RELEASE_METADATA_PATH
    )
    .sort_values("input_order")
    .reset_index(drop=True)
)

if (
    release_meta["feature_name"].tolist()
    != INPUT_FEATURES
):
    raise ValueError(
        "Release metadata order does not match model input order."
    )

SHIFT_QUARTERS = (
    release_meta[
        "strict_quarter_end_shift_quarters"
    ]
    .astype(int)
    .to_numpy()
)

aligned = pd.DataFrame({
    "date": df["date"],
    TARGET_COL: df[TARGET],
})

ALIGNED_INPUT_COLS = []

for feature, shift_q in zip(
    INPUT_FEATURES,
    SHIFT_QUARTERS,
):
    col = f"input__{feature}"
    ALIGNED_INPUT_COLS.append(col)

    aligned[col] = df[feature].shift(
        int(shift_q)
    )

aligned = (
    aligned[
        ["date", TARGET_COL]
        + ALIGNED_INPUT_COLS
    ]
    .dropna()
    .reset_index(drop=True)
)

# ============================================================
# BUILD STRICT 8Q -> 1Q SEQUENCES
# ============================================================
def build_sequences(data):
    """
    Input window: availability-aligned information from t-7 ... t
    Target: actual GDP growth at t+1
    """
    X, y, rows = [], [], []

    for target_idx in range(
        LOOKBACK + FORECAST_HORIZON - 1,
        len(data),
    ):
        input_end_idx = (
            target_idx
            - FORECAST_HORIZON
        )

        input_start_idx = (
            input_end_idx
            - LOOKBACK
            + 1
        )

        dates = (
            data.loc[
                input_start_idx:target_idx,
                "date",
            ]
            .dt.to_period("Q")
        )

        expected = pd.period_range(
            dates.iloc[0],
            dates.iloc[-1],
            freq="Q",
        )

        if (
            len(dates) != len(expected)
            or not np.array_equal(
                dates.astype(str).to_numpy(),
                expected.astype(str).to_numpy(),
            )
        ):
            continue

        X.append(
            data.loc[
                input_start_idx:input_end_idx,
                ALIGNED_INPUT_COLS,
            ]
            .to_numpy(dtype=np.float32)
        )

        y.append(
            float(
                data.loc[
                    target_idx,
                    TARGET_COL,
                ]
            )
        )

        rows.append({
            "input_start":
                data.loc[
                    input_start_idx,
                    "date",
                ],
            "input_end":
                data.loc[
                    input_end_idx,
                    "date",
                ],
            "target_date":
                data.loc[
                    target_idx,
                    "date",
                ],
        })

    return (
        np.asarray(X),
        np.asarray(
            y,
            dtype=np.float32,
        ),
        pd.DataFrame(rows),
    )


X_ALL, Y_ALL, META_ALL = (
    build_sequences(aligned)
)

TRAIN_MASK = (
    META_ALL["target_date"]
    < TEST_START
)

TEST_MASK = (
    META_ALL["target_date"]
    .between(
        TEST_START,
        TEST_END,
    )
)

if TRAIN_MASK.sum() == 0:
    raise RuntimeError(
        "Final pre-test training sample is empty."
    )

if TEST_MASK.sum() != 28:
    raise RuntimeError(
        "Expected exactly 28 final test quarters "
        f"(2019Q1-2025Q4), found {int(TEST_MASK.sum())}."
    )

X_TRAIN_RAW = X_ALL[
    TRAIN_MASK.to_numpy()
]

Y_TRAIN_RAW = Y_ALL[
    TRAIN_MASK.to_numpy()
]

X_TEST_RAW = X_ALL[
    TEST_MASK.to_numpy()
]

Y_TEST_RAW = Y_ALL[
    TEST_MASK.to_numpy()
]

TEST_META = (
    META_ALL.loc[
        TEST_MASK
    ]
    .reset_index(drop=True)
)

if (
    TEST_META["target_date"].min()
    != TEST_START
    or TEST_META["target_date"].max()
    != TEST_END
):
    raise RuntimeError(
        "Final test date boundaries do not match 2019Q1-2025Q4."
    )

# ============================================================
# FINAL TRAIN-ONLY SCALING
# ============================================================
# Fit feature scaling only on observations appearing in pre-2019
# training sequences. No test observation contributes to scaling.
feature_mean = (
    X_TRAIN_RAW
    .reshape(
        -1,
        len(INPUT_FEATURES),
    )
    .mean(axis=0)
    .astype(np.float32)
)

feature_std = (
    X_TRAIN_RAW
    .reshape(
        -1,
        len(INPUT_FEATURES),
    )
    .std(
        axis=0,
        ddof=0,
    )
    .astype(np.float32)
)

feature_std[
    feature_std == 0
] = 1.0

X_TRAIN = (
    X_TRAIN_RAW
    - feature_mean[
        None, None, :
    ]
) / feature_std[
    None, None, :
]

X_TEST = (
    X_TEST_RAW
    - feature_mean[
        None, None, :
    ]
) / feature_std[
    None, None, :
]

target_mean = float(
    Y_TRAIN_RAW.mean()
)

target_std = float(
    Y_TRAIN_RAW.std(
        ddof=0
    )
)

if target_std == 0:
    target_std = 1.0

Y_TRAIN = (
    Y_TRAIN_RAW
    - target_mean
) / target_std

# ============================================================
# POSITIONAL ENCODING
# ============================================================
def make_sinusoidal_encoding(
    length,
    d_model,
):
    position = torch.arange(
        length,
        dtype=torch.float32,
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float32,
        )
        * (
            -math.log(10000.0)
            / d_model
        )
    )

    pe = torch.zeros(
        length,
        d_model,
    )

    pe[:, 0::2] = torch.sin(
        position * div_term
    )

    pe[:, 1::2] = torch.cos(
        position * div_term
    )

    return pe

# ============================================================
# FROZEN TOKENISERS
# ============================================================
class QuarterTokenizer(nn.Module):
    """One 23-variable quarter vector = one Transformer token."""
    def __init__(
        self,
        n_features,
    ):
        super().__init__()

        self.projection = nn.Linear(
            n_features,
            D_MODEL,
        )

        self.register_buffer(
            "position",
            make_sinusoidal_encoding(
                LOOKBACK,
                D_MODEL,
            ),
        )

    def forward(self, x):
        return (
            self.projection(x)
            + self.position[
                None, :, :
            ]
        )


class ObservationTokenizer(nn.Module):
    """
    One scalar feature-time observation = one Transformer token.
    8 quarters x 23 features = 184 tokens.
    """
    def __init__(
        self,
        n_features,
    ):
        super().__init__()

        self.n_features = (
            n_features
        )

        self.value_projection = nn.Linear(
            1,
            D_MODEL,
        )

        self.feature_embedding = nn.Embedding(
            n_features,
            D_MODEL,
        )

        self.register_buffer(
            "time_encoding",
            make_sinusoidal_encoding(
                LOOKBACK,
                D_MODEL,
            ),
        )

    def forward(self, x):
        batch, steps, features = x.shape

        if (
            steps != LOOKBACK
            or features != self.n_features
        ):
            raise ValueError(
                "Unexpected observation-token input shape."
            )

        value_tokens = (
            self.value_projection(
                x.unsqueeze(-1)
            )
        )

        feature_ids = torch.arange(
            features,
            device=x.device,
        )

        feature_tokens = (
            self.feature_embedding(
                feature_ids
            )
            [None, None, :, :]
        )

        time_tokens = (
            self.time_encoding[
                None, :, None, :
            ]
        )

        tokens = (
            value_tokens
            + feature_tokens
            + time_tokens
        )

        return tokens.reshape(
            batch,
            steps * features,
            D_MODEL,
        )

# ============================================================
# FROZEN STANDARD SELF-ATTENTION
# ============================================================
class StandardSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        if D_MODEL % N_HEADS != 0:
            raise ValueError(
                "D_MODEL must be divisible by N_HEADS."
            )

        self.head_dim = (
            D_MODEL // N_HEADS
        )

        self.qkv = nn.Linear(
            D_MODEL,
            3 * D_MODEL,
        )

        self.out = nn.Linear(
            D_MODEL,
            D_MODEL,
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

    def forward(self, x):
        batch, length, _ = x.shape

        q, k, v = self.qkv(x).chunk(
            3,
            dim=-1,
        )

        def split(z):
            return (
                z.view(
                    batch,
                    length,
                    N_HEADS,
                    self.head_dim,
                )
                .transpose(1, 2)
            )

        q = split(q)
        k = split(k)
        v = split(v)

        scores = (
            torch.matmul(
                q,
                k.transpose(-2, -1),
            )
            / math.sqrt(
                self.head_dim
            )
        )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention
        )

        out = torch.matmul(
            attention,
            v,
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(
                batch,
                length,
                D_MODEL,
            )
        )

        return self.out(out)

# ============================================================
# SAME ENCODER + MEAN POOLING FOR Q0 AND O1
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = StandardSelfAttention()

        self.norm1 = nn.LayerNorm(
            D_MODEL
        )

        self.norm2 = nn.LayerNorm(
            D_MODEL
        )

        self.dropout1 = nn.Dropout(
            DROPOUT
        )

        self.dropout2 = nn.Dropout(
            DROPOUT
        )

        self.ff_dropout = nn.Dropout(
            DROPOUT
        )

        self.ff1 = nn.Linear(
            D_MODEL,
            D_FF,
        )

        self.ff2 = nn.Linear(
            D_FF,
            D_MODEL,
        )

    def forward(self, x):
        attn = self.attention(x)

        x = self.norm1(
            x
            + self.dropout1(attn)
        )

        ff = self.ff2(
            self.ff_dropout(
                torch.relu(
                    self.ff1(x)
                )
            )
        )

        return self.norm2(
            x
            + self.dropout2(ff)
        )


class FinalTransformer(nn.Module):
    """Frozen Q0/O1 architecture from Script 10I."""
    def __init__(
        self,
        tokenisation,
    ):
        super().__init__()

        if tokenisation == "quarter":
            self.tokenizer = QuarterTokenizer(
                len(INPUT_FEATURES)
            )
        elif tokenisation == "observation":
            self.tokenizer = ObservationTokenizer(
                len(INPUT_FEATURES)
            )
        else:
            raise ValueError(
                f"Unknown tokenisation: {tokenisation}"
            )

        self.encoder = EncoderLayer()

        self.forecast_head = nn.Linear(
            D_MODEL,
            1,
        )

    def forward(self, x):
        tokens = self.tokenizer(x)

        encoded = self.encoder(
            tokens
        )

        pooled = encoded.mean(
            dim=1
        )

        return (
            self.forecast_head(
                pooled
            )
            .squeeze(-1)
        )

# ============================================================
# FINAL FIXED-EPOCH TRAINING
# ============================================================
def make_loader(
    X,
    y,
    seed,
):
    dataset = TensorDataset(
        torch.from_numpy(
            X.astype(np.float32)
        ),
        torch.from_numpy(
            y.astype(np.float32)
        ),
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    return DataLoader(
        dataset,
        batch_size=min(
            BATCH_SIZE,
            len(dataset),
        ),
        shuffle=True,
        generator=generator,
    )


def train_fixed_epochs(
    candidate,
    seed,
):
    """
    Final refit:
      - all pre-2019 sequences;
      - no validation early stopping;
      - architecture-specific epoch count fixed from Script 10I only.
    """
    set_seed(seed)

    model = FinalTransformer(
        CANDIDATES[candidate]
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loader = make_loader(
        X_TRAIN,
        Y_TRAIN,
        seed,
    )

    epochs = int(
        FIXED_EPOCHS[candidate]
    )

    for _ in range(epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=
                    GRADIENT_CLIP,
            )

            optimizer.step()

    return model

# ============================================================
# METRICS
# ============================================================
def metrics(
    y_true,
    y_pred,
):
    error = (
        y_pred - y_true
    )

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    smape_terms = np.zeros_like(
        denominator,
        dtype=float,
    )

    valid = (
        denominator > 1e-8
    )

    smape_terms[valid] = (
        200
        * np.abs(
            error[valid]
        )
        / denominator[valid]
    )

    return {
        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
        "MAE":
            float(
                np.mean(
                    np.abs(error)
                )
            ),
        "sMAPE":
            float(
                smape_terms.mean()
            ),
        "direction_accuracy_pct":
            float(
                100
                * np.mean(
                    np.sign(y_true)
                    == np.sign(y_pred)
                )
            ),
        "mean_error_bias":
            float(
                np.mean(error)
            ),
    }

# ============================================================
# FINAL 2019Q1-2025Q4 EVALUATION
# ============================================================
run_rows = []
prediction_rows = []
parameter_rows = []

print("\n" + "=" * 78)
print("11B - FINAL OBSERVATION-TOKEN TRANSFORMER EVALUATION")
print("=" * 78)

print(f"\nDevice: {DEVICE}")
print(f"Pre-test training sequences: {len(X_TRAIN)}")
print(f"Final test sequences:        {len(X_TEST)}")
print(
    "Final test period:           "
    f"{TEST_META['target_date'].min().date()} "
    f"to {TEST_META['target_date'].max().date()}"
)

print("\nFixed epochs selected ONLY from Script 10I:")
print(
    epoch_df.to_string(
        index=False
    )
)

for candidate, tokenisation in CANDIDATES.items():

    temp = FinalTransformer(
        tokenisation
    )

    parameter_rows.append({
        "candidate":
            candidate,
        "tokenisation":
            tokenisation,
        "trainable_parameters":
            sum(
                p.numel()
                for p in temp.parameters()
                if p.requires_grad
            ),
        "fixed_epochs":
            int(
                FIXED_EPOCHS[
                    candidate
                ]
            ),
    })

    del temp

    print("\n" + "-" * 78)
    print(candidate)
    print("-" * 78)

    candidate_scores = []

    for seed in SEEDS:
        model = train_fixed_epochs(
            candidate,
            seed,
        )

        model.eval()

        with torch.no_grad():
            pred_scaled = (
                model(
                    torch.from_numpy(
                        X_TEST.astype(
                            np.float32
                        )
                    )
                    .to(DEVICE)
                )
                .cpu()
                .numpy()
            )

        pred_raw = (
            pred_scaled
            * target_std
            + target_mean
        )

        score = metrics(
            Y_TEST_RAW,
            pred_raw,
        )

        candidate_scores.append(
            score
        )

        run_rows.append({
            "candidate":
                candidate,
            "seed":
                seed,
            "fixed_epochs":
                int(
                    FIXED_EPOCHS[
                        candidate
                    ]
                ),
            "train_sequences":
                len(X_TRAIN),
            "test_sequences":
                len(X_TEST),
            **score,
        })

        for i, target_date in enumerate(
            TEST_META["target_date"]
        ):
            actual = float(
                Y_TEST_RAW[i]
            )

            prediction = float(
                pred_raw[i]
            )

            error = (
                prediction - actual
            )

            prediction_rows.append({
                "candidate":
                    candidate,
                "seed":
                    seed,
                "target_date":
                    target_date,
                "actual":
                    actual,
                "prediction":
                    prediction,
                "error":
                    error,
                "absolute_error":
                    abs(error),
                "squared_error":
                    error ** 2,
            })

    print(
        "RMSE="
        f"{np.mean([s['RMSE'] for s in candidate_scores]):.6f} "
        "± "
        f"{np.std([s['RMSE'] for s in candidate_scores], ddof=1):.6f}"
    )

    print(
        "MAE ="
        f"{np.mean([s['MAE'] for s in candidate_scores]):.6f} "
        "± "
        f"{np.std([s['MAE'] for s in candidate_scores], ddof=1):.6f}"
    )

# ============================================================
# SAVE FINAL RESULTS
# ============================================================
run_df = pd.DataFrame(
    run_rows
)

run_df.to_csv(
    RUN_OUT,
    index=False,
)

prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_df.to_csv(
    PREDICTION_OUT,
    index=False,
)

parameter_df = pd.DataFrame(
    parameter_rows
)

summary_df = (
    run_df
    .groupby(
        "candidate",
        as_index=False,
    )
    .agg(
        RMSE_mean=
            ("RMSE", "mean"),
        RMSE_std=
            ("RMSE", "std"),
        MAE_mean=
            ("MAE", "mean"),
        MAE_std=
            ("MAE", "std"),
        sMAPE_mean=
            ("sMAPE", "mean"),
        sMAPE_std=
            ("sMAPE", "std"),
        direction_accuracy_pct_mean=
            (
                "direction_accuracy_pct",
                "mean",
            ),
        direction_accuracy_pct_std=
            (
                "direction_accuracy_pct",
                "std",
            ),
        mean_error_bias_mean=
            (
                "mean_error_bias",
                "mean",
            ),
        mean_error_bias_std=
            (
                "mean_error_bias",
                "std",
            ),
    )
    .merge(
        parameter_df,
        on="candidate",
        how="left",
    )
    .sort_values(
        "RMSE_mean"
    )
    .reset_index(drop=True)
)

summary_df.insert(
    0,
    "rank_by_RMSE",
    np.arange(
        1,
        len(summary_df) + 1,
    ),
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)

# Mean prediction across 10 seeds for each model and quarter.
mean_prediction_df = (
    prediction_df
    .groupby(
        [
            "candidate",
            "target_date",
        ],
        as_index=False,
    )
    .agg(
        actual=
            ("actual", "first"),
        mean_prediction=
            ("prediction", "mean"),
        prediction_std=
            ("prediction", "std"),
    )
)

mean_prediction_df[
    "error"
] = (
    mean_prediction_df[
        "mean_prediction"
    ]
    - mean_prediction_df[
        "actual"
    ]
)

mean_prediction_df[
    "absolute_error"
] = np.abs(
    mean_prediction_df[
        "error"
    ]
)

mean_prediction_df[
    "squared_error"
] = (
    mean_prediction_df[
        "error"
    ] ** 2
)

mean_prediction_df.to_csv(
    MEAN_PREDICTION_OUT,
    index=False,
)

# ============================================================
# FIGURES
# ============================================================
fig, ax = plt.subplots(
    figsize=(10, 5)
)

actual_series = (
    mean_prediction_df[
        mean_prediction_df[
            "candidate"
        ]
        == "Q0_aligned_quarter_token"
    ]
    .sort_values(
        "target_date"
    )
)

ax.plot(
    pd.to_datetime(
        actual_series[
            "target_date"
        ]
    ),
    actual_series[
        "actual"
    ],
    label="Actual GDP growth",
)

for candidate in CANDIDATES:
    sub = (
        mean_prediction_df[
            mean_prediction_df[
                "candidate"
            ]
            == candidate
        ]
        .sort_values(
            "target_date"
        )
    )

    ax.plot(
        pd.to_datetime(
            sub["target_date"]
        ),
        sub[
            "mean_prediction"
        ],
        label=candidate,
    )

ax.set_title(
    "Final 2019Q1-2025Q4 GDP Growth Forecasts"
)

ax.set_xlabel(
    "Target quarter"
)

ax.set_ylabel(
    "GDP growth QoQ"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "11b_final_forecasts.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    summary_df[
        "candidate"
    ],
    summary_df[
        "RMSE_mean"
    ],
)

ax.set_title(
    "Final Test RMSE: Quarter Tokens vs Observation Tokens"
)

ax.set_xlabel(
    "Frozen architecture"
)

ax.set_ylabel(
    "Mean RMSE across 10 seeds"
)

ax.tick_params(
    axis="x",
    rotation=10,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "11b_final_RMSE_comparison.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# CONFIGURATION / AUDIT
# ============================================================
config = {
    "architecture_frozen_before_test":
        True,
    "test_period":
        {
            "start":
                str(
                    TEST_START.date()
                ),
            "end":
                str(
                    TEST_END.date()
                ),
            "quarters":
                int(
                    len(X_TEST)
                ),
        },
    "training_period_rule":
        "all sequences with target_date before 2019-01-01",
    "fixed_epoch_source":
        str(
            PRETEST_RUN_PATH
        ),
    "fixed_epoch_rule":
        "rounded median best_epoch across 40 pre-test Script 10I runs per architecture",
    "availability_alignment":
        "same strict quarter-end feature shifts as Scripts 10H-10J",
    "Q0":
        "availability-aligned quarter-token standard-attention Transformer",
    "O1":
        "availability-aligned feature-time observation-token standard-attention Transformer",
    "architecture": {
        "lookback_quarters":
            LOOKBACK,
        "forecast_horizon_quarters":
            FORECAST_HORIZON,
        "input_features":
            len(INPUT_FEATURES),
        "d_model":
            D_MODEL,
        "n_heads":
            N_HEADS,
        "feedforward_dimension":
            D_FF,
        "dropout":
            DROPOUT,
        "pooling":
            "mean pooling",
    },
    "seeds":
        SEEDS,
    "post_test_rule":
        "Do not alter architecture or hyperparameters after observing final test results.",
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
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 78)
print("FINAL OBSERVATION-TOKEN EVALUATION COMPLETE")
print("=" * 78)

print("\nFinal 2019Q1-2025Q4 ranking:\n")

print(
    summary_df[
        [
            "rank_by_RMSE",
            "candidate",
            "RMSE_mean",
            "RMSE_std",
            "MAE_mean",
            "MAE_std",
            "sMAPE_mean",
            "direction_accuracy_pct_mean",
            "mean_error_bias_mean",
            "trainable_parameters",
            "fixed_epochs",
        ]
    ].to_string(
        index=False
    )
)

q0_rmse = float(
    summary_df.loc[
        summary_df["candidate"]
        == "Q0_aligned_quarter_token",
        "RMSE_mean",
    ].iloc[0]
)

o1_rmse = float(
    summary_df.loc[
        summary_df["candidate"]
        == "O1_aligned_observation_token",
        "RMSE_mean",
    ].iloc[0]
)

q0_mae = float(
    summary_df.loc[
        summary_df["candidate"]
        == "Q0_aligned_quarter_token",
        "MAE_mean",
    ].iloc[0]
)

o1_mae = float(
    summary_df.loc[
        summary_df["candidate"]
        == "O1_aligned_observation_token",
        "MAE_mean",
    ].iloc[0]
)

print("\nO1 relative to Q0:")
print(
    f"  RMSE improvement: "
    f"{100 * (q0_rmse - o1_rmse) / q0_rmse:+.2f}%"
)

print(
    f"  MAE improvement:  "
    f"{100 * (q0_mae - o1_mae) / q0_mae:+.2f}%"
)

print("\nSafeguards:")
print("  - Architecture was frozen before final test evaluation.")
print("  - Fixed epochs came only from pre-2019 Script 10I results.")
print("  - Test data were not used for scaling or training.")
print("  - No post-test architecture tuning is permitted.")

print(f"\nResults:\n  {RESULT_DIR}")
print(f"\n300-DPI figures:\n  {FIGURE_DIR}")

if __name__ == "__main__":
    pass
