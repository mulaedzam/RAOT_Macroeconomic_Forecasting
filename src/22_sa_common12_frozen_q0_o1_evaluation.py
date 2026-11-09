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

MODEL_DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sa_common_model_data"
)

NPZ_PATH = (
    MODEL_DATA_DIR
    / "south_africa_common12_q0_o1_model_data.npz"
)

SCALER_PATH = (
    MODEL_DATA_DIR
    / "south_africa_common12_training_scalers.json"
)

TEST_INDEX_PATH = (
    MODEL_DATA_DIR
    / "south_africa_common12_test_sequence_index.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "sa_common12_final_q0_o1"
)

FIGURE_DIR = (
    ROOT
    / "figures"
    / "sa_common12_final_q0_o1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_OUT = (
    RESULT_DIR
    / "22_sa_common12_run_metrics.csv"
)

PREDICTION_OUT = (
    RESULT_DIR
    / "22_sa_common12_test_predictions_by_seed.csv"
)

MEAN_PREDICTION_OUT = (
    RESULT_DIR
    / "22_sa_common12_mean_test_predictions.csv"
)

SUMMARY_OUT = (
    RESULT_DIR
    / "22_sa_common12_model_summary.csv"
)

CONFIG_OUT = (
    RESULT_DIR
    / "22_sa_common12_config.json"
)

FIGURE_DPI = 300

# ============================================================
# FROZEN CROSS-COUNTRY MODEL DESIGN
# ============================================================
LOOKBACK = 8

SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]

D_MODEL = 16
N_HEADS = 2
D_FF = 32
DROPOUT = 0.20

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

FIXED_EPOCHS = {
    "Q0_aligned_quarter_token":
        4,
    "O1_aligned_observation_token":
        10,
}

CANDIDATES = {
    "Q0_aligned_quarter_token":
        "quarter",
    "O1_aligned_observation_token":
        "observation",
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
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
# LOAD PREPARED SOUTH AFRICAN COMMON-12 ARRAYS
# ============================================================
for path in [
    NPZ_PATH,
    SCALER_PATH,
    TEST_INDEX_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Run Script 21 first."
        )

data = np.load(
    NPZ_PATH,
    allow_pickle=False,
)

X_TRAIN = data[
    "X_train"
].astype(
    np.float32
)

Y_TRAIN = data[
    "y_train"
].astype(
    np.float32
)

X_TEST = data[
    "X_test"
].astype(
    np.float32
)

Y_TEST_RAW = data[
    "y_test_raw"
].astype(
    np.float32
)

FEATURE_NAMES = (
    data[
        "feature_names"
    ]
    .astype(str)
    .tolist()
)

with open(
    SCALER_PATH,
    "r",
    encoding="utf-8",
) as f:
    scaler_config = json.load(f)

target_mean = float(
    scaler_config[
        "target_scaler"
    ][
        "target_mean"
    ]
)

target_std = float(
    scaler_config[
        "target_scaler"
    ][
        "target_std"
    ]
)

TEST_META = pd.read_csv(
    TEST_INDEX_PATH
)

TEST_META[
    "target_date"
] = pd.to_datetime(
    TEST_META[
        "target_date"
    ],
    errors="coerce",
)

# ============================================================
# CONSISTENCY CHECKS
# ============================================================
if X_TRAIN.ndim != 3:
    raise ValueError(
        f"Expected 3D X_train, found {X_TRAIN.shape}"
    )

if X_TEST.ndim != 3:
    raise ValueError(
        f"Expected 3D X_test, found {X_TEST.shape}"
    )

if X_TRAIN.shape[1] != LOOKBACK:
    raise ValueError(
        "South Africa lookback does not match frozen 8-quarter design."
    )

if len(FEATURE_NAMES) != X_TRAIN.shape[2]:
    raise ValueError(
        "Feature-name count does not match input channels."
    )

if FEATURE_NAMES[0] != "gdp_growth_qoq":
    raise ValueError(
        "Frozen cross-country representation requires lagged GDP "
        "growth as the first input channel."
    )

if X_TRAIN.shape[2] != 12:
    raise ValueError(
        f"Expected 12 common channels, found {X_TRAIN.shape[2]}."
    )

if len(TEST_META) != len(X_TEST):
    raise ValueError(
        "Test index length does not match X_test."
    )

if len(X_TEST) != 28:
    raise ValueError(
        f"Expected 28 final test quarters, found {len(X_TEST)}."
    )

if (
    TEST_META[
        "target_date"
    ].min()
    != pd.Timestamp("2019-01-01")
):
    raise ValueError(
        "South African common-12 test must start at 2019Q1."
    )

if (
    TEST_META[
        "target_date"
    ].max()
    != pd.Timestamp("2025-10-01")
):
    raise ValueError(
        "South African common-12 test must end at 2025Q4."
    )

N_FEATURES = X_TRAIN.shape[2]

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

    def forward(
        self,
        x,
    ):
        return (
            self.projection(x)
            + self.position[
                None, :, :
            ]
        )


class ObservationTokenizer(nn.Module):
    def __init__(
        self,
        n_features,
    ):
        super().__init__()

        self.n_features = n_features

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

    def forward(
        self,
        x,
    ):
        batch, steps, features = (
            x.shape
        )

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
            )[
                None, None, :, :
            ]
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

    def forward(
        self,
        x,
    ):
        batch, length, _ = (
            x.shape
        )

        q, k, v = (
            self.qkv(x)
            .chunk(
                3,
                dim=-1,
            )
        )

        def split(z):
            return (
                z.view(
                    batch,
                    length,
                    N_HEADS,
                    self.head_dim,
                )
                .transpose(
                    1,
                    2,
                )
            )

        q = split(q)
        k = split(k)
        v = split(v)

        scores = (
            torch.matmul(
                q,
                k.transpose(
                    -2,
                    -1,
                ),
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
            out.transpose(
                1,
                2,
            )
            .contiguous()
            .view(
                batch,
                length,
                D_MODEL,
            )
        )

        return self.out(
            out
        )

# ============================================================
# SAME ENCODER + MEAN POOLING FOR Q0 AND O1
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = (
            StandardSelfAttention()
        )

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

    def forward(
        self,
        x,
    ):
        attn = self.attention(
            x
        )

        x = self.norm1(
            x
            + self.dropout1(
                attn
            )
        )

        ff = self.ff2(
            self.ff_dropout(
                torch.relu(
                    self.ff1(
                        x
                    )
                )
            )
        )

        return self.norm2(
            x
            + self.dropout2(
                ff
            )
        )


class FinalTransformer(nn.Module):
    def __init__(
        self,
        tokenisation,
    ):
        super().__init__()

        if tokenisation == "quarter":
            self.tokenizer = (
                QuarterTokenizer(
                    N_FEATURES
                )
            )

        elif tokenisation == "observation":
            self.tokenizer = (
                ObservationTokenizer(
                    N_FEATURES
                )
            )

        else:
            raise ValueError(
                f"Unknown tokenisation: {tokenisation}"
            )

        self.encoder = (
            EncoderLayer()
        )

        self.forecast_head = (
            nn.Linear(
                D_MODEL,
                1,
            )
        )

    def forward(
        self,
        x,
    ):
        tokens = self.tokenizer(
            x
        )

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
# FIXED-EPOCH TRAINING
# ============================================================
def make_loader(
    X,
    y,
    seed,
):
    dataset = TensorDataset(
        torch.from_numpy(
            X.astype(
                np.float32
            )
        ),
        torch.from_numpy(
            y.astype(
                np.float32
            )
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
    set_seed(seed)

    model = (
        FinalTransformer(
            CANDIDATES[
                candidate
            ]
        )
        .to(
            DEVICE
        )
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    loader = make_loader(
        X_TRAIN,
        Y_TRAIN,
        seed,
    )

    epochs = int(
        FIXED_EPOCHS[
            candidate
        ]
    )

    for _ in range(
        epochs
    ):
        model.train()

        for xb, yb in loader:
            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                xb
            )

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
        y_pred
        - y_true
    )

    denominator = (
        np.abs(
            y_true
        )
        + np.abs(
            y_pred
        )
    )

    smape_terms = np.zeros_like(
        denominator,
        dtype=float,
    )

    valid = (
        denominator
        > 1e-8
    )

    smape_terms[
        valid
    ] = (
        200.0
        * np.abs(
            error[
                valid
            ]
        )
        / denominator[
            valid
        ]
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
                    np.abs(
                        error
                    )
                )
            ),
        "sMAPE":
            float(
                smape_terms.mean()
            ),
        "direction_accuracy_pct":
            float(
                100.0
                * np.mean(
                    np.sign(
                        y_true
                    )
                    == np.sign(
                        y_pred
                    )
                )
            ),
        "mean_error_bias":
            float(
                np.mean(
                    error
                )
            ),
    }

# ============================================================
# FINAL SOUTH AFRICAN COMMON-12 EVALUATION
# ============================================================
run_rows = []
prediction_rows = []
parameter_rows = []

print(
    "\n"
    + "=" * 78
)

print(
    "22 - SOUTH AFRICA COMMON-12 FROZEN Q0 VS RAOT/O1"
)

print(
    "=" * 78
)

print(
    f"\nDevice: {DEVICE}"
)

print(
    f"Training sequences: {len(X_TRAIN)}"
)

print(
    f"Test sequences:     {len(X_TEST)}"
)

print(
    "Test period:        "
    f"{TEST_META['target_date'].min().date()} "
    "-> "
    f"{TEST_META['target_date'].max().date()}"
)

print(
    f"Input shape:         "
    f"{LOOKBACK} quarters x "
    f"{N_FEATURES} channels"
)

print(
    f"Q0 tokens:           {LOOKBACK}"
)

print(
    f"O1/RAOT tokens:      {LOOKBACK * N_FEATURES}"
)

print(
    "\nFrozen settings:"
)

print(
    f"  d_model={D_MODEL}"
)

print(
    f"  heads={N_HEADS}"
)

print(
    f"  FF={D_FF}"
)

print(
    f"  dropout={DROPOUT}"
)

print(
    f"  batch={BATCH_SIZE}"
)

print(
    f"  lr={LEARNING_RATE}"
)

print(
    f"  weight_decay={WEIGHT_DECAY}"
)

print(
    f"  seeds={SEEDS}"
)

for candidate, tokenisation in (
    CANDIDATES.items()
):
    temp = FinalTransformer(
        tokenisation
    )

    parameter_count = sum(
        p.numel()
        for p in temp.parameters()
        if p.requires_grad
    )

    parameter_rows.append({
        "candidate":
            candidate,
        "tokenisation":
            tokenisation,
        "trainable_parameters":
            parameter_count,
        "fixed_epochs":
            int(
                FIXED_EPOCHS[
                    candidate
                ]
            ),
        "tokens_per_sequence":
            (
                LOOKBACK
                if tokenisation
                == "quarter"
                else LOOKBACK
                * N_FEATURES
            ),
    })

    del temp

    print(
        "\n"
        + "-" * 78
    )

    print(
        candidate
    )

    print(
        "-" * 78
    )

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
                        X_TEST
                    )
                    .to(
                        DEVICE
                    )
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
            "country":
                "South Africa",
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
                len(
                    X_TRAIN
                ),
            "test_sequences":
                len(
                    X_TEST
                ),
            "input_channels":
                N_FEATURES,
            **score,
        })

        for i, target_date in enumerate(
            TEST_META[
                "target_date"
            ]
        ):
            actual = float(
                Y_TEST_RAW[
                    i
                ]
            )

            prediction = float(
                pred_raw[
                    i
                ]
            )

            error = (
                prediction
                - actual
            )

            prediction_rows.append({
                "country":
                    "South Africa",
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
                    abs(
                        error
                    ),
                "squared_error":
                    error ** 2,
            })

    rmse_values = [
        s["RMSE"]
        for s in candidate_scores
    ]

    mae_values = [
        s["MAE"]
        for s in candidate_scores
    ]

    print(
        "RMSE="
        f"{np.mean(rmse_values):.6f} "
        "± "
        f"{np.std(rmse_values, ddof=1):.6f}"
    )

    print(
        "MAE ="
        f"{np.mean(mae_values):.6f} "
        "± "
        f"{np.std(mae_values, ddof=1):.6f}"
    )

# ============================================================
# SAVE RESULTS
# ============================================================
run_df = pd.DataFrame(
    run_rows
)

prediction_df = pd.DataFrame(
    prediction_rows
)

parameter_df = pd.DataFrame(
    parameter_rows
)

run_df.to_csv(
    RUN_OUT,
    index=False,
)

prediction_df.to_csv(
    PREDICTION_OUT,
    index=False,
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
    .reset_index(
        drop=True
    )
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

# ============================================================
# MEAN PREDICTIONS
# ============================================================
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
# DIRECT O1 - Q0 DIFFERENCES
# ============================================================
q0 = summary_df[
    summary_df[
        "candidate"
    ]
    == "Q0_aligned_quarter_token"
].iloc[0]

o1 = summary_df[
    summary_df[
        "candidate"
    ]
    == "O1_aligned_observation_token"
].iloc[0]

rmse_diff = float(
    o1[
        "RMSE_mean"
    ]
    - q0[
        "RMSE_mean"
    ]
)

mae_diff = float(
    o1[
        "MAE_mean"
    ]
    - q0[
        "MAE_mean"
    ]
)

rmse_improvement_pct = float(
    100.0
    * (
        q0[
            "RMSE_mean"
        ]
        - o1[
            "RMSE_mean"
        ]
    )
    / q0[
        "RMSE_mean"
    ]
)

mae_improvement_pct = float(
    100.0
    * (
        q0[
            "MAE_mean"
        ]
        - o1[
            "MAE_mean"
        ]
    )
    / q0[
        "MAE_mean"
    ]
)

# ============================================================
# FORECAST FIGURE
# ============================================================
fig, ax = plt.subplots(
    figsize=(
        10,
        5,
    )
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
    label=
        "Actual GDP growth",
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

    label = (
        "Q0 quarter-token"
        if candidate
        == "Q0_aligned_quarter_token"
        else "RAOT/O1 observation-token"
    )

    ax.plot(
        pd.to_datetime(
            sub[
                "target_date"
            ]
        ),
        sub[
            "mean_prediction"
        ],
        label=label,
    )

ax.set_title(
    "South Africa Common-12: Frozen Q0 vs RAOT/O1"
)

ax.set_xlabel(
    "Quarter"
)

ax.set_ylabel(
    "Real GDP growth (QoQ, %)"
)

ax.legend()

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "22_sa_common12_q0_o1_test_forecasts.png"
)

fig.savefig(
    figure_path,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(
    fig
)

# ============================================================
# CONFIG
# ============================================================
config = {
    "country":
        "South Africa",
    "purpose":
        (
            "Direct common-12 cross-country comparison with Brazil."
        ),
    "feature_set":
        (
            "same strict common 12 conceptual channels as Brazil"
        ),
    "feature_order":
        FEATURE_NAMES,
    "training_sequences":
        len(
            X_TRAIN
        ),
    "test_sequences":
        len(
            X_TEST
        ),
    "test_period":
        "2019Q1-2025Q4",
    "seeds":
        SEEDS,
    "architecture": {
        "d_model":
            D_MODEL,
        "n_heads":
            N_HEADS,
        "feedforward_dimension":
            D_FF,
        "dropout":
            DROPOUT,
        "attention":
            "standard scaled dot-product",
        "pooling":
            "mean pooling",
    },
    "training": {
        "optimizer":
            "AdamW",
        "loss":
            "MSE",
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "batch_size":
            BATCH_SIZE,
        "gradient_clip":
            GRADIENT_CLIP,
        "q0_epochs":
            FIXED_EPOCHS[
                "Q0_aligned_quarter_token"
            ],
        "o1_epochs":
            FIXED_EPOCHS[
                "O1_aligned_observation_token"
            ],
        "validation":
            None,
        "early_stopping":
            False,
    },
    "important_note":
        (
            "This is a fair common-feature rerun and does not replace the "
            "original 23-channel South African country-rich result."
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
# CONSOLE SUMMARY
# ============================================================
print(
    "\n"
    + "=" * 78
)

print(
    "SOUTH AFRICA COMMON-12 FINAL MODEL SUMMARY"
)

print(
    "=" * 78
)

print(
    "\n"
    + summary_df[
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
            "tokens_per_sequence",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nO1/RAOT relative to Q0:"
)

print(
    f"  RMSE difference (O1-Q0): "
    f"{rmse_diff:+.6f}"
)

print(
    f"  MAE difference  (O1-Q0): "
    f"{mae_diff:+.6f}"
)

print(
    f"  RMSE improvement: "
    f"{rmse_improvement_pct:+.2f}%"
)

print(
    f"  MAE improvement:  "
    f"{mae_improvement_pct:+.2f}%"
)

print(
    "\nInterpretation rule:"
)

print(
    "  Positive improvement % means O1/RAOT is better; "
    "negative means Q0 is better."
)

print(
    "\nOutputs:"
)

print(
    f"  {RUN_OUT}"
)

print(
    f"  {PREDICTION_OUT}"
)

print(
    f"  {MEAN_PREDICTION_OUT}"
)

print(
    f"  {SUMMARY_OUT}"
)

print(
    f"  {CONFIG_OUT}"
)

print(
    "\n300-DPI figure:"
)

print(
    f"  {figure_path}"
)

if __name__ == "__main__":
    pass
