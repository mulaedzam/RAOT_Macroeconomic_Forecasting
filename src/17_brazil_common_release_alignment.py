from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
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
    / "brazil_release_alignment"
)

FIG_DIR = (
    ROOT
    / "figures"
    / "brazil_release_alignment"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_OUT = (
    OUT_DIR
    / "brazil_common_release_metadata.csv"
)

ALIGNED_OUT = (
    OUT_DIR
    / "brazil_common_release_aligned.csv"
)

UNALIGNED_OUT = (
    OUT_DIR
    / "brazil_common_unaligned.csv"
)

COVERAGE_OUT = (
    OUT_DIR
    / "brazil_release_aligned_coverage.csv"
)

CONFIG_OUT = (
    OUT_DIR
    / "brazil_release_alignment_config.json"
)

SUMMARY_OUT = (
    OUT_DIR
    / "brazil_release_alignment_summary.txt"
)

FIGURE_DPI = 300

LOOKBACK_QUARTERS = 8
FORECAST_HORIZON = 1

# ============================================================
# FROZEN PRIMARY CROSS-COUNTRY FEATURE SET
# ============================================================

COMMON_EXOGENOUS = [
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

HISTORICAL_GDP_CHANNEL = "gdp_growth_qoq"

MODEL_INPUT_CHANNELS = (
    COMMON_EXOGENOUS
    + [HISTORICAL_GDP_CHANNEL]
)

# ============================================================
# OPERATIONAL RELEASE-LAG METADATA
# ============================================================
RELEASE_SPECS = [
    {
        "feature": "headline_inflation_yoy",
        "concept": "headline CPI inflation",
        "operational_release_lag_days": 20,
        "rationale": "Monthly CPI is published after the reference month; use a conservative post-quarter release proxy.",
    },
    {
        "feature": "policy_rate_change_pp",
        "concept": "policy interest rate",
        "operational_release_lag_days": 0,
        "rationale": "SELIC decisions are known when announced.",
    },
    {
        "feature": "m3_growth_qoq",
        "concept": "broad money",
        "operational_release_lag_days": 30,
        "rationale": "Monetary aggregates are released after the reference month/quarter.",
    },
    {
        "feature": "usd_brl_change_qoq",
        "concept": "nominal exchange rate",
        "operational_release_lag_days": 0,
        "rationale": "Market exchange rates are observed contemporaneously.",
    },
    {
        "feature": "imports_growth_qoq",
        "concept": "real imports from national accounts",
        "operational_release_lag_days": 75,
        "rationale": "National-account expenditure components are available with the quarterly GDP release.",
    },
    {
        "feature": "investment_growth_qoq",
        "concept": "real gross fixed capital formation",
        "operational_release_lag_days": 75,
        "rationale": "National-account expenditure components are available with the quarterly GDP release.",
    },
    {
        "feature": "private_consumption_growth_qoq",
        "concept": "real private consumption",
        "operational_release_lag_days": 75,
        "rationale": "National-account expenditure components are available with the quarterly GDP release.",
    },
    {
        "feature": "manufacturing_growth",
        "concept": "manufacturing production",
        "operational_release_lag_days": 42,
        "rationale": "Production indicators are released after the reference period.",
    },
    {
        "feature": "share_price_return_qoq",
        "concept": "share prices",
        "operational_release_lag_days": 0,
        "rationale": "Market prices are observed contemporaneously.",
    },
    {
        "feature": "reer_change_qoq",
        "concept": "real effective exchange rate",
        "operational_release_lag_days": 20,
        "rationale": "REER index is published after the reference month.",
    },
    {
        "feature": "capacity_utilisation",
        "concept": "manufacturing capacity utilisation",
        "operational_release_lag_days": 65,
        "rationale": "Survey/activity measure is treated conservatively as post-quarter information.",
    },
    {
        "feature": "gdp_growth_qoq",
        "concept": "historical real GDP growth input",
        "operational_release_lag_days": 75,
        "rationale": "Quarterly GDP is only available after the reference quarter; historical GDP input must therefore also be release-aligned.",
    },
]


def strict_quarter_shift(lag_days):
    if lag_days <= 0:
        return 0

    return int(
        math.ceil(
            lag_days / 90.0
        )
    )


def longest_complete_block(df, columns):
    work = (
        df[
            ["date"] + columns
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    complete = (
        work[columns]
        .notna()
        .all(axis=1)
    )

    best_start = pd.NaT
    best_end = pd.NaT
    best_len = 0

    current_start = None
    current_end = None
    current_len = 0
    previous_period = None

    for i, row in work.iterrows():
        if not bool(complete.iloc[i]):
            current_start = None
            current_end = None
            current_len = 0
            previous_period = None
            continue

        period = (
            row["date"]
            .to_period("Q")
        )

        if (
            previous_period is not None
            and period == previous_period + 1
        ):
            current_len += 1
            current_end = row["date"]
        else:
            current_start = row["date"]
            current_end = row["date"]
            current_len = 1

        if current_len > best_len:
            best_len = current_len
            best_start = current_start
            best_end = current_end

        previous_period = period

    return {
        "start": best_start,
        "end": best_end,
        "quarters": best_len,
    }


def potential_sequence_count(n_quarters):
    return max(
        0,
        n_quarters
        - LOOKBACK_QUARTERS
        - FORECAST_HORIZON
        + 1,
    )


# ============================================================
# LOAD FEATURES
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing engineered Brazil features:\n{INPUT_FILE}\n"
        "Run Script 16 first."
    )

df = pd.read_csv(
    INPUT_FILE
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

missing = [
    col
    for col in MODEL_INPUT_CHANNELS
    if col not in df.columns
]

if missing:
    raise KeyError(
        "Missing required frozen common features:\n"
        + "\n".join(missing)
    )

# ============================================================
# BUILD METADATA
# ============================================================
metadata = pd.DataFrame(
    RELEASE_SPECS
)

metadata[
    "strict_quarter_shift"
] = metadata[
    "operational_release_lag_days"
].apply(
    strict_quarter_shift
)

metadata[
    "availability_class"
] = np.where(
    metadata[
        "strict_quarter_shift"
    ] == 0,
    "contemporaneous",
    np.where(
        metadata[
            "strict_quarter_shift"
        ] == 1,
        "one_quarter_delayed",
        "two_or_more_quarters_delayed",
    ),
)

metadata.to_csv(
    METADATA_OUT,
    index=False,
)

# ============================================================
# UNALIGNED PRIMARY COMMON DATA
# ============================================================
unaligned = df[
    ["date"]
    + MODEL_INPUT_CHANNELS
].copy()

unaligned.to_csv(
    UNALIGNED_OUT,
    index=False,
)

# ============================================================
# APPLY RELEASE ALIGNMENT
# ============================================================
aligned = pd.DataFrame({
    "date":
        df["date"],
})

for _, row in metadata.iterrows():
    feature = row[
        "feature"
    ]

    shift_q = int(
        row[
            "strict_quarter_shift"
        ]
    )

    aligned[
        feature
    ] = df[
        feature
    ].shift(
        shift_q
    )

aligned.to_csv(
    ALIGNED_OUT,
    index=False,
)

# ============================================================
# COVERAGE + COMPLETE-BLOCK AUDIT
# ============================================================
coverage_rows = []

for feature in MODEL_INPUT_CHANNELS:
    original = (
        unaligned[
            feature
        ]
    )

    release_aligned = (
        aligned[
            feature
        ]
    )

    valid = (
        release_aligned
        .dropna()
    )

    coverage_rows.append({
        "feature":
            feature,
        "strict_quarter_shift":
            int(
                metadata.loc[
                    metadata["feature"] == feature,
                    "strict_quarter_shift",
                ].iloc[0]
            ),
        "unaligned_non_missing":
            int(
                original.notna().sum()
            ),
        "aligned_non_missing":
            int(
                release_aligned.notna().sum()
            ),
        "aligned_first_available":
            (
                aligned.loc[
                    release_aligned.notna(),
                    "date",
                ].min()
                if release_aligned.notna().any()
                else pd.NaT
            ),
        "aligned_last_available":
            (
                aligned.loc[
                    release_aligned.notna(),
                    "date",
                ].max()
                if release_aligned.notna().any()
                else pd.NaT
            ),
    })

coverage_df = pd.DataFrame(
    coverage_rows
)

coverage_df.to_csv(
    COVERAGE_OUT,
    index=False,
)

unaligned_block = longest_complete_block(
    unaligned,
    MODEL_INPUT_CHANNELS,
)

aligned_block = longest_complete_block(
    aligned,
    MODEL_INPUT_CHANNELS,
)

unaligned_sequences = potential_sequence_count(
    unaligned_block[
        "quarters"
    ]
)

aligned_sequences = potential_sequence_count(
    aligned_block[
        "quarters"
    ]
)

# ============================================================
# FIGURE
# ============================================================
plot_df = metadata[
    [
        "feature",
        "operational_release_lag_days",
    ]
].sort_values(
    "operational_release_lag_days",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(
        11,
        7,
    )
)

ax.barh(
    plot_df["feature"],
    plot_df[
        "operational_release_lag_days"
    ],
)

ax.set_title(
    "Brazil Operational Publication-Lag Proxies"
)

ax.set_xlabel(
    "Operational release lag (days)"
)

ax.set_ylabel(
    "Frozen common feature"
)

fig.tight_layout()

figure_path = (
    FIG_DIR
    / "brazil_common_release_lag_proxies.png"
)

fig.savefig(
    figure_path,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# CONFIG / SUMMARY
# ============================================================
config = {
    "country":
        "Brazil",
    "primary_cross_country_feature_set":
        "A_common_strict_11",
    "exogenous_feature_count":
        len(COMMON_EXOGENOUS),
    "historical_gdp_channels":
        1,
    "total_model_input_channels":
        len(MODEL_INPUT_CHANNELS),
    "lookback_quarters":
        LOOKBACK_QUARTERS,
    "forecast_horizon":
        FORECAST_HORIZON,
    "release_alignment_method":
        "strict quarter-end operational publication-lag proxies",
    "historical_vintage_database_used":
        False,
    "important_interpretation":
        (
            "Release lags are operational proxies and must not be described "
            "as exact historical vintage availability dates."
        ),
    "unaligned_complete_block": {
        "start":
            str(
                unaligned_block["start"]
            ),
        "end":
            str(
                unaligned_block["end"]
            ),
        "quarters":
            unaligned_block["quarters"],
        "potential_sequences":
            unaligned_sequences,
    },
    "release_aligned_complete_block": {
        "start":
            str(
                aligned_block["start"]
            ),
        "end":
            str(
                aligned_block["end"]
            ),
        "quarters":
            aligned_block["quarters"],
        "potential_sequences":
            aligned_sequences,
    },
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
    "BRAZIL PRIMARY COMMON FEATURE SET + RELEASE ALIGNMENT",
    "=" * 78,
    "",
    "Frozen primary cross-country set: A_common_strict_11",
    f"Exogenous features: {len(COMMON_EXOGENOUS)}",
    f"+ historical GDP growth input: 1",
    f"Total Transformer input channels: {len(MODEL_INPUT_CHANNELS)}",
    "",
    "Why A_common_strict_11 was selected:",
    "  - longest clean common sample;",
    "  - all 11 predictor concepts are genuinely comparable with South Africa;",
    "  - avoids relabelling Brazil nominal exports as real exports;",
    "  - avoids relabelling Brazil private-credit level growth as credit/GDP change;",
    "  - avoids the 2007+ 10Y yield sample restriction;",
    "  - retains late-2025 quarters that would be lost by requiring government consumption.",
    "",
    "Release-alignment interpretation:",
    "  The lag values are operational publication-lag proxies, not exact historical vintages.",
    "  A strict quarter-end rule is used: any positive post-quarter publication lag",
    "  prevents the same-quarter observation from entering the forecasting information set.",
    "",
    f"Unaligned longest complete block: {unaligned_block['start']} -> {unaligned_block['end']} "
    f"({unaligned_block['quarters']} quarters; {unaligned_sequences} potential sequences)",
    f"Release-aligned longest complete block: {aligned_block['start']} -> {aligned_block['end']} "
    f"({aligned_block['quarters']} quarters; {aligned_sequences} potential sequences)",
    "",
    "No model has been trained.",
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
print("17 - BRAZIL COMMON FEATURE FREEZE + RELEASE ALIGNMENT")
print("=" * 78)

print("\nFrozen primary feature set:")
for feature in COMMON_EXOGENOUS:
    print(f"  - {feature}")

print(
    f"\nHistorical GDP input channel:\n"
    f"  - {HISTORICAL_GDP_CHANNEL}"
)

print(
    f"\nTotal model input channels: "
    f"{len(MODEL_INPUT_CHANNELS)}"
)

print("\nRelease metadata:")
print(
    metadata[
        [
            "feature",
            "operational_release_lag_days",
            "strict_quarter_shift",
            "availability_class",
        ]
    ].to_string(
        index=False
    )
)

print("\nComplete-block feasibility:")
print(
    f"  Unaligned:       "
    f"{unaligned_block['start']} -> "
    f"{unaligned_block['end']} | "
    f"{unaligned_block['quarters']} quarters | "
    f"{unaligned_sequences} potential sequences"
)

print(
    f"  Release-aligned: "
    f"{aligned_block['start']} -> "
    f"{aligned_block['end']} | "
    f"{aligned_block['quarters']} quarters | "
    f"{aligned_sequences} potential sequences"
)

print("\nAligned coverage:")
print(
    coverage_df.to_string(
        index=False
    )
)

print("\nOutputs:")
print(f"  {METADATA_OUT}")
print(f"  {UNALIGNED_OUT}")
print(f"  {ALIGNED_OUT}")
print(f"  {COVERAGE_OUT}")
print(f"  {CONFIG_OUT}")
print(f"  {SUMMARY_OUT}")

print("\n300-DPI figure:")
print(f"  {figure_path}")

if __name__ == "__main__":
    pass
