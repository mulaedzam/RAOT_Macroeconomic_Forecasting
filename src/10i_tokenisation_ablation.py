from pathlib import Path
import copy, json, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "data" / "processed" / "features" / "engineered_features_target_window.csv"
INPUT_FEATURE_PATH = ROOT / "data" / "processed" / "model_data" / "B_expanded_refined_input_features.csv"
META_PATH = ROOT / "data" / "processed" / "availability" / "feature_release_metadata.csv"
RESULT_DIR = ROOT / "results" / "tokenisation_ablation"
FIGURE_DIR = ROOT / "figures" / "tokenisation_ablation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_OUT = RESULT_DIR / "10i_run_metrics.csv"
FOLD_OUT = RESULT_DIR / "10i_fold_summary.csv"
SUMMARY_OUT = RESULT_DIR / "10i_architecture_summary.csv"
PAIRED_OUT = RESULT_DIR / "10i_paired_run_comparison.csv"
PAIRED_FOLD_OUT = RESULT_DIR / "10i_paired_fold_summary.csv"
BOOT_OUT = RESULT_DIR / "10i_paired_difference_bootstrap.csv"
CONFIG_OUT = RESULT_DIR / "10i_config.json"
FIGURE_DPI = 300

# ============================================================
# LOCKED DESIGN
# ============================================================
TARGET = "gdp_growth_qoq"
TARGET_COL = "target_gdp_growth_qoq"
LOOKBACK = 8
HORIZON = 1
FOLDS = [
    {"fold": 1, "val_start": "2010-01-01", "val_end": "2011-10-01"},
    {"fold": 2, "val_start": "2012-01-01", "val_end": "2013-10-01"},
    {"fold": 3, "val_start": "2014-01-01", "val_end": "2015-10-01"},
    {"fold": 4, "val_start": "2016-01-01", "val_end": "2018-10-01"},
]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

D_MODEL = 16
N_HEADS = 2
D_FF = 32
DROPOUT = 0.20
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 350
PATIENCE = 30
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
BOOT_REPS = 10000
BOOT_SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATES = {
    "Q0_aligned_quarter_token": "quarter",
    "O1_aligned_observation_token": "observation",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

# ============================================================
# LOAD + APPLY EXACT SAME AVAILABILITY ALIGNMENT AS 10H
# ============================================================
df = pd.read_csv(FEATURE_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

feature_order = pd.read_csv(INPUT_FEATURE_PATH).sort_values("input_order").reset_index(drop=True)
INPUT_FEATURES = feature_order["input_feature"].tolist()
if not INPUT_FEATURES or INPUT_FEATURES[0] != TARGET:
    raise ValueError("Expected lagged GDP growth as first input channel.")

release_meta = pd.read_csv(META_PATH).sort_values("input_order").reset_index(drop=True)
if release_meta["feature_name"].tolist() != INPUT_FEATURES:
    raise ValueError("Release metadata does not match model input order.")
SHIFT_Q = release_meta["strict_quarter_end_shift_quarters"].astype(int).to_numpy()

aligned = pd.DataFrame({"date": df["date"], TARGET_COL: df[TARGET]})
ALIGNED_COLS = []
for feature, shift in zip(INPUT_FEATURES, SHIFT_Q):
    col = f"input__{feature}"
    ALIGNED_COLS.append(col)
    aligned[col] = df[feature].shift(int(shift))
aligned = aligned[["date", TARGET_COL] + ALIGNED_COLS].dropna().reset_index(drop=True)

# ============================================================
# STRICT 8Q > 1Q SEQUENCES
# ============================================================
def build_sequences(data):
    X, y, rows = [], [], []
    for target_idx in range(LOOKBACK + HORIZON - 1, len(data)):
        input_end = target_idx - HORIZON
        input_start = input_end - LOOKBACK + 1
        dates = data.loc[input_start:target_idx, "date"].dt.to_period("Q")
        expected = pd.period_range(dates.iloc[0], dates.iloc[-1], freq="Q")
        if len(dates) != len(expected) or not np.array_equal(
            dates.astype(str).to_numpy(), expected.astype(str).to_numpy()
        ):
            continue
        X.append(data.loc[input_start:input_end, ALIGNED_COLS].to_numpy(dtype=np.float32))
        y.append(float(data.loc[target_idx, TARGET_COL]))
        rows.append({
            "input_start": data.loc[input_start, "date"],
            "input_end": data.loc[input_end, "date"],
            "target_date": data.loc[target_idx, "date"],
        })
    return np.asarray(X), np.asarray(y, dtype=np.float32), pd.DataFrame(rows)

X_RAW, Y_RAW, META = build_sequences(aligned)
pretest_mask = META["target_date"] < pd.Timestamp("2019-01-01")
X_RAW = X_RAW[pretest_mask.to_numpy()]
Y_RAW = Y_RAW[pretest_mask.to_numpy()]
META = META.loc[pretest_mask].reset_index(drop=True)
if META.empty or META["target_date"].max() >= pd.Timestamp("2019-01-01"):
    raise RuntimeError("Pre-test safeguard failed.")

# ============================================================
# TOKENISERS
# ============================================================
def sinusoidal_encoding(length, d_model):
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class QuarterTokenizer(nn.Module):
    """Each quarter's 23-dimensional vector becomes one token."""
    def __init__(self, n_features):
        super().__init__()
        self.projection = nn.Linear(n_features, D_MODEL)
        self.register_buffer("position", sinusoidal_encoding(LOOKBACK, D_MODEL))

    def forward(self, x):
        return self.projection(x) + self.position[None, :, :]


class ObservationTokenizer(nn.Module):
    """Each scalar feature-time observation becomes one token: 8 x 23 = 184 tokens."""
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.value_projection = nn.Linear(1, D_MODEL)
        self.feature_embedding = nn.Embedding(n_features, D_MODEL)
        self.register_buffer("time_encoding", sinusoidal_encoding(LOOKBACK, D_MODEL))

    def forward(self, x):
        batch, steps, features = x.shape
        if steps != LOOKBACK or features != self.n_features:
            raise ValueError("Unexpected observation-token input shape.")
        value_tokens = self.value_projection(x.unsqueeze(-1))
        feature_ids = torch.arange(features, device=x.device)
        feature_tokens = self.feature_embedding(feature_ids)[None, None, :, :]
        time_tokens = self.time_encoding[None, :, None, :]
        return (value_tokens + feature_tokens + time_tokens).reshape(
            batch, steps * features, D_MODEL
        )

# ============================================================
# IDENTICAL STANDARD ATTENTION / ENCODER FOR BOTH MODELS
# ============================================================
class StandardSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        if D_MODEL % N_HEADS != 0:
            raise ValueError("D_MODEL must be divisible by N_HEADS.")
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(z):
            return z.view(batch, length, N_HEADS, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = StandardSelfAttention()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.drop1 = nn.Dropout(DROPOUT)
        self.drop2 = nn.Dropout(DROPOUT)
        self.ffdrop = nn.Dropout(DROPOUT)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self.attention(x)))
        ff = self.ff2(self.ffdrop(torch.relu(self.ff1(x))))
        return self.norm2(x + self.drop2(ff))


class TokenisationAblationTransformer(nn.Module):
    def __init__(self, tokenisation):
        super().__init__()
        if tokenisation == "quarter":
            self.tokenizer = QuarterTokenizer(len(INPUT_FEATURES))
        elif tokenisation == "observation":
            self.tokenizer = ObservationTokenizer(len(INPUT_FEATURES))
        else:
            raise ValueError(f"Unknown tokenisation: {tokenisation}")
        self.encoder = EncoderLayer()
        self.forecast_head = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        encoded = self.encoder(self.tokenizer(x))
        pooled = encoded.mean(dim=1)
        return self.forecast_head(pooled).squeeze(-1)

# ============================================================
# FOLD-SPECIFIC SCALING
# ============================================================
def prepare_fold(fold):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])
    train_mask = META["target_date"] < val_start
    val_mask = META["target_date"].between(val_start, val_end)
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(f"Fold {fold['fold']} has an empty sample.")

    Xtr_raw = X_RAW[train_mask.to_numpy()]
    Xva_raw = X_RAW[val_mask.to_numpy()]
    mu = Xtr_raw.reshape(-1, len(INPUT_FEATURES)).mean(axis=0).astype(np.float32)
    sd = Xtr_raw.reshape(-1, len(INPUT_FEATURES)).std(axis=0, ddof=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xtr = (Xtr_raw - mu[None, None, :]) / sd[None, None, :]
    Xva = (Xva_raw - mu[None, None, :]) / sd[None, None, :]

    ytr_raw = Y_RAW[train_mask.to_numpy()]
    yva_raw = Y_RAW[val_mask.to_numpy()]
    ymu = float(ytr_raw.mean())
    ysd = float(ytr_raw.std(ddof=0))
    if ysd == 0:
        ysd = 1.0

    return {
        "X_train": Xtr.astype(np.float32),
        "y_train": ((ytr_raw - ymu) / ysd).astype(np.float32),
        "X_val": Xva.astype(np.float32),
        "y_val": ((yva_raw - ymu) / ysd).astype(np.float32),
        "y_val_raw": yva_raw.astype(np.float64),
        "y_mean": ymu,
        "y_std": ysd,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
    }

# ============================================================
# TRAINING
# ============================================================
def make_loader(X, y, seed):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=min(BATCH_SIZE, len(ds)),
        shuffle=True,
        generator=gen,
    )


def validation_loss(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        return float(
            criterion(
                model(torch.from_numpy(X).to(DEVICE)),
                torch.from_numpy(y).to(DEVICE),
            ).item()
        )


def train_model(candidate, fd, seed):
    set_seed(seed)
    model = TokenisationAblationTransformer(CANDIDATES[candidate]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = make_loader(fd["X_train"], fd["y_train"], seed)

    best_state, best_val, best_epoch, wait = None, np.inf, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        vl = validation_loss(model, fd["X_val"], fd["y_val"], criterion)
        if vl < best_val - MIN_DELTA:
            best_state = copy.deepcopy(model.state_dict())
            best_val, best_epoch, wait = vl, epoch, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No valid state for {candidate}, seed={seed}.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_val

# ============================================================
# METRICS
# ============================================================
def metrics(y_true, y_pred):
    error = y_pred - y_true
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.zeros_like(denom, dtype=float)
    valid = denom > 1e-8
    smape[valid] = 200 * np.abs(error[valid]) / denom[valid]
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "sMAPE": float(smape.mean()),
        "direction_accuracy_pct": float(
            100 * np.mean(np.sign(y_true) == np.sign(y_pred))
        ),
    }

# ============================================================
# RUN 40 PAIRED PRE-TEST EXPERIMENTS PER ARCHITECTURE
# ============================================================
run_rows, parameter_rows = [], []

print("\n" + "=" * 78)
print("10I - TOKENISATION ABLATION")
print("=" * 78)
print(f"\nDevice: {DEVICE}")
print("2019-2025 test targets used: NO")
print(f"Availability-complete rows: {len(aligned)}")
print(f"Pre-2019 forecast sequences: {len(META)}")
print(f"Quarter-token sequence length: {LOOKBACK}")
print(f"Observation-token sequence length: {LOOKBACK * len(INPUT_FEATURES)}")
print(f"Runs per architecture: {len(FOLDS) * len(SEEDS)}")

for candidate, tokenisation in CANDIDATES.items():
    temp = TokenisationAblationTransformer(tokenisation)
    parameter_rows.append({
        "candidate": candidate,
        "tokenisation": tokenisation,
        "tokens_per_sequence": LOOKBACK if tokenisation == "quarter" else LOOKBACK * len(INPUT_FEATURES),
        "trainable_parameters": sum(p.numel() for p in temp.parameters() if p.requires_grad),
    })
    del temp

    print("\n" + "-" * 78)
    print(candidate)
    print("-" * 78)

    for fold in FOLDS:
        fd = prepare_fold(fold)
        fold_rows = []

        for seed in SEEDS:
            model, best_epoch, best_val = train_model(candidate, fd, seed)
            model.eval()
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(fd["X_val"]).to(DEVICE)).cpu().numpy()
            pred_raw = pred_scaled * fd["y_std"] + fd["y_mean"]
            sc = metrics(fd["y_val_raw"], pred_raw)
            fold_rows.append(sc)
            run_rows.append({
                "candidate": candidate,
                "fold": fold["fold"],
                "seed": seed,
                "train_sequences": fd["train_count"],
                "validation_sequences": fd["val_count"],
                "best_epoch": best_epoch,
                "best_validation_mse_scaled": best_val,
                **sc,
            })

        print(
            f"Fold {fold['fold']} | {fd['train_count']:>2} train / {fd['val_count']:>2} val | "
            f"RMSE={np.mean([r['RMSE'] for r in fold_rows]):.4f} | "
            f"MAE={np.mean([r['MAE'] for r in fold_rows]):.4f}"
        )

# ============================================================
# SUMMARIES
# ============================================================
run_df = pd.DataFrame(run_rows)
run_df.to_csv(RUN_OUT, index=False)
parameter_df = pd.DataFrame(parameter_rows)

fold_df = (
    run_df.groupby(["candidate", "fold"], as_index=False)
    .agg(
        train_sequences=("train_sequences", "first"),
        validation_sequences=("validation_sequences", "first"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        sMAPE_mean=("sMAPE", "mean"),
        direction_accuracy_pct_mean=("direction_accuracy_pct", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
    )
)
fold_df.to_csv(FOLD_OUT, index=False)

summary_df = (
    fold_df.groupby("candidate", as_index=False)
    .agg(
        rolling_RMSE_mean=("RMSE_mean", "mean"),
        rolling_RMSE_fold_std=("RMSE_mean", "std"),
        rolling_MAE_mean=("MAE_mean", "mean"),
        rolling_MAE_fold_std=("MAE_mean", "std"),
        rolling_sMAPE_mean=("sMAPE_mean", "mean"),
        rolling_direction_accuracy_pct_mean=("direction_accuracy_pct_mean", "mean"),
        mean_best_epoch=("mean_best_epoch", "mean"),
    )
    .merge(parameter_df, on="candidate", how="left")
    .sort_values("rolling_RMSE_mean")
    .reset_index(drop=True)
)
summary_df.insert(0, "rank_by_RMSE", np.arange(1, len(summary_df) + 1))
summary_df.to_csv(SUMMARY_OUT, index=False)

# ============================================================
# PAIRED O1 vs Q0
# ============================================================
q0 = run_df[run_df["candidate"] == "Q0_aligned_quarter_token"][
    ["fold", "seed", "RMSE", "MAE", "sMAPE"]
].rename(columns={"RMSE": "Q0_RMSE", "MAE": "Q0_MAE", "sMAPE": "Q0_sMAPE"})

o1 = run_df[run_df["candidate"] == "O1_aligned_observation_token"][
    ["fold", "seed", "RMSE", "MAE", "sMAPE"]
].rename(columns={"RMSE": "O1_RMSE", "MAE": "O1_MAE", "sMAPE": "O1_sMAPE"})

paired = q0.merge(o1, on=["fold", "seed"], how="inner")
for metric in ["RMSE", "MAE", "sMAPE"]:
    paired[f"O1_minus_Q0_{metric}"] = paired[f"O1_{metric}"] - paired[f"Q0_{metric}"]
paired["RMSE_winner"] = np.where(paired["O1_RMSE"] < paired["Q0_RMSE"], "O1", "Q0")
paired["MAE_winner"] = np.where(paired["O1_MAE"] < paired["Q0_MAE"], "O1", "Q0")
paired.to_csv(PAIRED_OUT, index=False)

paired_fold = (
    paired.groupby("fold", as_index=False)
    .agg(
        mean_RMSE_difference=("O1_minus_Q0_RMSE", "mean"),
        median_RMSE_difference=("O1_minus_Q0_RMSE", "median"),
        mean_MAE_difference=("O1_minus_Q0_MAE", "mean"),
        median_MAE_difference=("O1_minus_Q0_MAE", "median"),
        O1_RMSE_wins=("RMSE_winner", lambda s: int((s == "O1").sum())),
        O1_MAE_wins=("MAE_winner", lambda s: int((s == "O1").sum())),
        paired_runs=("seed", "count"),
    )
)
paired_fold.to_csv(PAIRED_FOLD_OUT, index=False)

# ============================================================
# BOOTSTRAP
# ============================================================
def bootstrap_mean_difference(values, reps=BOOT_REPS, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "observed_mean": float(values.mean()),
        "ci_2_5": float(np.percentile(means, 2.5)),
        "ci_97_5": float(np.percentile(means, 97.5)),
        "bootstrap_probability_less_than_zero": float(np.mean(means < 0)),
    }

boot_rows = []
for metric in ["RMSE", "MAE", "sMAPE"]:
    boot_rows.append({
        "metric": metric,
        "interpretation": "negative favours observation-token O1",
        "paired_runs": len(paired),
        "bootstrap_repetitions": BOOT_REPS,
        **bootstrap_mean_difference(paired[f"O1_minus_Q0_{metric}"]),
    })
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(BOOT_OUT, index=False)

# ============================================================
# FIGURES
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(summary_df["candidate"], summary_df["rolling_RMSE_mean"])
ax.set_title("Pre-2019 Tokenisation Ablation")
ax.set_xlabel("Tokenisation")
ax.set_ylabel("Mean rolling-validation RMSE")
ax.tick_params(axis="x", rotation=10)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "10i_tokenisation_RMSE.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for fold in sorted(paired["fold"].unique()):
    sub = paired[paired["fold"] == fold]
    ax.scatter([fold] * len(sub), sub["O1_minus_Q0_RMSE"])
ax.axhline(0, linewidth=0.8)
ax.set_title("Paired RMSE: Observation Token - Quarter Token")
ax.set_xlabel("Rolling-validation fold")
ax.set_ylabel("O1 RMSE - Q0 RMSE")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "10i_paired_RMSE_differences.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# CONFIG + CONSOLE
# ============================================================
config = {
    "purpose": "Isolate observation-level tokenisation using identical availability-aligned data.",
    "test_period_excluded": "2019-01-01 onward",
    "availability_alignment": "identical Q0/O1; Script 10g strict quarter-end shifts",
    "Q0": "8 quarter-level tokens",
    "O1": "184 feature-time observation tokens",
    "shared_architecture": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "feedforward_dimension": D_FF,
        "dropout": DROPOUT,
        "attention": "standard scaled dot-product",
        "pooling": "mean pooling",
        "lookback": LOOKBACK,
        "horizon": HORIZON,
    },
    "folds": FOLDS,
    "seeds": SEEDS,
}
with open(CONFIG_OUT, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 78)
print("TOKENISATION ABLATION COMPLETE")
print("=" * 78)

print("\n10-seed pre-2019 ranking:\n")
print(summary_df[
    [
        "rank_by_RMSE", "candidate", "tokens_per_sequence",
        "rolling_RMSE_mean", "rolling_RMSE_fold_std",
        "rolling_MAE_mean", "rolling_sMAPE_mean",
        "rolling_direction_accuracy_pct_mean", "trainable_parameters",
    ]
].to_string(index=False))

print("\nPaired O1 vs Q0 across 40 fold-seed runs:")
print(f"  Mean RMSE difference (O1-Q0): {paired['O1_minus_Q0_RMSE'].mean():+.6f}")
print(f"  Median RMSE difference:       {paired['O1_minus_Q0_RMSE'].median():+.6f}")
print(f"  Mean MAE difference (O1-Q0):  {paired['O1_minus_Q0_MAE'].mean():+.6f}")
print(f"  Median MAE difference:        {paired['O1_minus_Q0_MAE'].median():+.6f}")
print(f"  O1 RMSE wins: {(paired['RMSE_winner'] == 'O1').sum()}/{len(paired)}")
print(f"  O1 MAE wins:  {(paired['MAE_winner'] == 'O1').sum()}/{len(paired)}")

print("\nPaired results by fold:\n")
print(paired_fold.to_string(index=False))

print("\nBootstrap 95% CIs:\n")
print(boot_df.to_string(index=False))

print("\nDecision rule:")
print("  Carry O1 forward only if it consistently beats the matched quarter-token Q0.")
print("  If O1 is not robustly better, stop architecture tuning before final test use.")

print("\nSafeguard:")
print("  No target observation from 2019 onward was used.")

print(f"\nResults:\n  {RESULT_DIR}")
print(f"\n300-DPI figures:\n  {FIGURE_DIR}")

if __name__ == "__main__":
    pass
