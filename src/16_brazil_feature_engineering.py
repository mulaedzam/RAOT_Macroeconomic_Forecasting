from pathlib import Path
import json

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
    / "brazil_harmonised"
    / "brazil_quarterly_harmonised.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "brazil_features"
)

FIG_DIR = (
    ROOT
    / "figures"
    / "brazil_features"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURES_OUT = (
    OUT_DIR
    / "brazil_engineered_features.csv"
)

DICTIONARY_OUT = (
    OUT_DIR
    / "brazil_feature_dictionary.csv"
)

COVERAGE_OUT = (
    OUT_DIR
    / "brazil_engineered_feature_coverage.csv"
)

FEASIBILITY_OUT = (
    OUT_DIR
    / "brazil_feature_set_feasibility.csv"
)

FEATURE_SETS_OUT = (
    OUT_DIR
    / "brazil_feature_sets.json"
)

SUMMARY_OUT = (
    OUT_DIR
    / "brazil_feature_engineering_summary.txt"
)

FIGURE_DPI = 300

# Fixed design already established in South Africa.
LOOKBACK_QUARTERS = 8
FORECAST_HORIZON = 1

# Diagnostic period. It is not a training/test split.
DIAGNOSTIC_START = pd.Timestamp("1996-01-01")
DIAGNOSTIC_END = pd.Timestamp("2025-10-01")

# ============================================================
# HELPERS
# ============================================================
def pct_change(series):
    """
    Standard percentage change without silently forward-filling missing values.
    """
    return (
        series.pct_change(
            fill_method=None
        )
        * 100.0
    )


def safe_log_return_pct(series):
    """
    Log return in percentage points.
    Returns NaN where either current or previous value is non-positive.
    """
    s = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = (
        (s > 0)
        & (s.shift(1) > 0)
    )

    out = pd.Series(
        np.nan,
        index=s.index,
        dtype=float,
    )

    out.loc[valid] = (
        100.0
        * np.log(
            s.loc[valid]
            / s.shift(1).loc[valid]
        )
    )

    return out


def coverage_stats(
    df,
    columns,
    start,
    end,
):
    idx = pd.period_range(
        start=start,
        end=end,
        freq="Q",
    ).start_time

    work = (
        df.set_index("date")
        .reindex(idx)
    )

    rows = []

    for col in columns:
        s = work[col]

        valid = s.dropna()

        rows.append({
            "feature":
                col,
            "expected_quarters":
                len(idx),
            "non_missing_quarters":
                int(
                    s.notna().sum()
                ),
            "coverage_pct":
                float(
                    100
                    * s.notna().mean()
                ),
            "first_available":
                (
                    valid.index.min()
                    if len(valid)
                    else pd.NaT
                ),
            "last_available":
                (
                    valid.index.max()
                    if len(valid)
                    else pd.NaT
                ),
        })

    return pd.DataFrame(
        rows
    )


def longest_consecutive_complete_block(
    df,
    columns,
):
    """
    Find the longest block of consecutive quarterly rows for which all
    requested columns are non-missing.
    """
    work = (
        df[
            ["date"]
            + columns
        ]
        .copy()
        .sort_values("date")
    )

    complete = work[
        columns
    ].notna().all(
        axis=1
    )

    blocks = []
    current = []

    prev_period = None

    for i, row in work.iterrows():
        if not complete.loc[i]:
            if current:
                blocks.append(
                    current
                )
                current = []

            prev_period = None
            continue

        current_period = (
            row["date"]
            .to_period("Q")
        )

        if (
            prev_period is not None
            and current_period
            != prev_period + 1
        ):
            if current:
                blocks.append(
                    current
                )

            current = []

        current.append(
            row["date"]
        )

        prev_period = (
            current_period
        )

    if current:
        blocks.append(
            current
        )

    if not blocks:
        return {
            "start":
                pd.NaT,
            "end":
                pd.NaT,
            "quarters":
                0,
        }

    best = max(
        blocks,
        key=len,
    )

    return {
        "start":
            min(best),
        "end":
            max(best),
        "quarters":
            len(best),
    }


def sequence_count(
    n_complete_quarters,
    lookback=LOOKBACK_QUARTERS,
    horizon=FORECAST_HORIZON,
):
    """
    Number of rolling sequences available from one consecutive complete block
    when T lookback quarters predict the next-quarter target.
    """
    return max(
        0,
        n_complete_quarters
        - lookback
        - horizon
        + 1,
    )


# ============================================================
# LOAD HARMONISED QUARTERLY DATA
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing harmonised Brazil file:\n{INPUT_FILE}\n"
        "Run the corrected Script 15 first."
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

# ============================================================
# FEATURE ENGINEERING
# ============================================================
features = pd.DataFrame({
    "date":
        df["date"],
})

# ------------------------------------------------------------
# TARGET / HISTORICAL GDP CHANNEL
# ------------------------------------------------------------
# Real GDP is a level. The modeling target is quarterly real GDP growth.
features[
    "gdp_growth_qoq"
] = pct_change(
    df["br_real_gdp"]
)

# ------------------------------------------------------------
# PRICES
# ------------------------------------------------------------
# IPCA is already a rolling 12-month inflation rate.
features[
    "headline_inflation_yoy"
] = df[
    "br_headline_inflation_yoy"
]

# ------------------------------------------------------------
# MONETARY POLICY / INTEREST RATES
# ------------------------------------------------------------
# Match the South African repo-rate-change concept:
# change in the quarterly average policy rate, in percentage points.
features[
    "policy_rate_change_pp"
] = (
    df[
        "br_selic_target"
    ]
    .diff()
)

# Short-term Treasury bill rate is retained as a secondary level/change signal.
features[
    "short_rate_level"
] = df[
    "br_treasury_bill_yield"
]

features[
    "short_rate_change_pp"
] = (
    df[
        "br_treasury_bill_yield"
    ]
    .diff()
)

# The 10Y series starts only in 2007, so the yield spread is SECONDARY.
features[
    "yield_spread_10y_short"
] = (
    df[
        "br_long_term_10y_yield"
    ]
    - df[
        "br_treasury_bill_yield"
    ]
)

# ------------------------------------------------------------
# MONEY / CREDIT
# ------------------------------------------------------------
# M3 is a quarter-end nominal stock -> quarterly growth.
features[
    "m3_growth_qoq"
] = pct_change(
    df["br_m3_level"]
)

# Brazil credit source is a USD-denominated LEVEL, not credit/GDP.
# Keep it country-specific and do not treat it as equivalent to SA credit/GDP.
features[
    "private_credit_level_growth_qoq"
] = pct_change(
    df[
        "br_private_credit_level_usd"
    ]
)

# Credit gap is already expressed as a gap to trend.
features[
    "credit_gap"
] = df[
    "br_credit_gap"
]

# ------------------------------------------------------------
# EXCHANGE RATE / EXTERNAL PRICES
# ------------------------------------------------------------
# BRL per USD. Positive growth means BRL depreciation against USD.
features[
    "usd_brl_change_qoq"
] = pct_change(
    df[
        "br_usd_brl_level"
    ]
)

# REER is an index; use quarterly percentage change.
features[
    "reer_change_qoq"
] = pct_change(
    df[
        "br_reer"
    ]
)

# ------------------------------------------------------------
# TRADE / DEMAND
# ------------------------------------------------------------
# Imports, investment and consumption are real quarterly national-account levels.
features[
    "imports_growth_qoq"
] = pct_change(
    df[
        "br_real_imports"
    ]
)

features[
    "investment_growth_qoq"
] = pct_change(
    df[
        "br_real_investment"
    ]
)

features[
    "private_consumption_growth_qoq"
] = pct_change(
    df[
        "br_real_private_consumption"
    ]
)

features[
    "government_consumption_growth_qoq"
] = pct_change(
    df[
        "br_real_government_consumption"
    ]
)

# Current Brazil export source is nominal merchandise exports.
# It is a valid Brazil predictor, but NOT strictly equivalent to SA real exports.
features[
    "nominal_exports_growth_qoq"
] = pct_change(
    df[
        "br_nominal_merchandise_exports"
    ]
)

# ------------------------------------------------------------
# REAL-ECONOMY ACTIVITY
# ------------------------------------------------------------
# FRED/OECD code ending 657S is already a growth-rate-from-previous-period
# type series. Do NOT percentage-change it again.
features[
    "manufacturing_growth"
] = df[
    "br_manufacturing_growth"
]

features[
    "retail_sales_growth"
] = df[
    "br_retail_sales_growth"
]

# Capacity utilisation is a rate/index level.
features[
    "capacity_utilisation"
] = df[
    "br_capacity_utilisation"
]

# Business confidence is an index level, but shorter (2008+).
features[
    "business_confidence"
] = df[
    "br_business_confidence"
]

# Electricity is a quarterly total GWh after Script 15 aggregation.
features[
    "electricity_growth_qoq"
] = pct_change(
    df[
        "br_electricity_generation"
    ]
)

# ------------------------------------------------------------
# FINANCIAL / RESERVES
# ------------------------------------------------------------
# Share-price index -> quarterly return.
features[
    "share_price_return_qoq"
] = pct_change(
    df[
        "br_share_price_index"
    ]
)

# Reserves are quarter-end stock -> growth.
features[
    "reserves_growth_qoq"
] = pct_change(
    df[
        "br_reserves_ex_gold"
    ]
)

# ============================================================
# SANITY CLEANING
# ============================================================
# Replace infinities generated by any zero denominators with NaN.
numeric_cols = [
    c
    for c in features.columns
    if c != "date"
]

features[
    numeric_cols
] = features[
    numeric_cols
].replace(
    [np.inf, -np.inf],
    np.nan,
)

# ============================================================
# FEATURE DICTIONARY
# ============================================================
dictionary_rows = [
    {
        "feature": "gdp_growth_qoq",
        "category": "target/history",
        "transformation": "100 * (GDP_t / GDP_t-1 - 1)",
        "cross_country_status": "target + lagged history channel",
        "reason": "Same next-quarter GDP-growth forecasting problem as South Africa.",
    },
    {
        "feature": "headline_inflation_yoy",
        "category": "prices",
        "transformation": "none; already 12-month IPCA rate",
        "cross_country_status": "strict_common",
        "reason": "Direct analogue of headline inflation used in South Africa.",
    },
    {
        "feature": "policy_rate_change_pp",
        "category": "monetary policy",
        "transformation": "quarterly SELIC average first difference",
        "cross_country_status": "strict_common",
        "reason": "Direct analogue of South African repo-rate change.",
    },
    {
        "feature": "m3_growth_qoq",
        "category": "money",
        "transformation": "quarter-on-quarter percentage growth of quarter-end M3",
        "cross_country_status": "strict_common",
        "reason": "Comparable broad-money growth concept.",
    },
    {
        "feature": "usd_brl_change_qoq",
        "category": "exchange rate",
        "transformation": "quarter-on-quarter percentage change in BRL per USD",
        "cross_country_status": "strict_common",
        "reason": "Direct analogue of USD/ZAR change; positive values indicate domestic-currency depreciation.",
    },
    {
        "feature": "imports_growth_qoq",
        "category": "external/demand",
        "transformation": "quarter-on-quarter percentage growth of real imports",
        "cross_country_status": "strict_common",
        "reason": "Comparable real-import growth measure.",
    },
    {
        "feature": "investment_growth_qoq",
        "category": "demand",
        "transformation": "quarter-on-quarter percentage growth of real gross fixed capital formation",
        "cross_country_status": "strict_common",
        "reason": "Comparable real-investment growth measure.",
    },
    {
        "feature": "private_consumption_growth_qoq",
        "category": "demand",
        "transformation": "quarter-on-quarter percentage growth of real private consumption",
        "cross_country_status": "strict_common",
        "reason": "Comparable real private-consumption growth measure.",
    },
    {
        "feature": "manufacturing_growth",
        "category": "activity",
        "transformation": "none; source already growth rate from previous period",
        "cross_country_status": "strict_common",
        "reason": "Same activity concept available in South African expanded set.",
    },
    {
        "feature": "share_price_return_qoq",
        "category": "financial",
        "transformation": "quarter-on-quarter percentage change in share-price index",
        "cross_country_status": "strict_common",
        "reason": "Same share-price-return concept available in South Africa.",
    },
    {
        "feature": "reer_change_qoq",
        "category": "external/competitiveness",
        "transformation": "quarter-on-quarter percentage change in REER",
        "cross_country_status": "strict_common",
        "reason": "Same REER-change concept available in South Africa.",
    },
    {
        "feature": "government_consumption_growth_qoq",
        "category": "fiscal/demand",
        "transformation": "quarter-on-quarter percentage growth of real government consumption",
        "cross_country_status": "strict_common_optional",
        "reason": "Comparable to South African government-consumption growth, but Brazil source currently ends 2025Q1.",
    },
    {
        "feature": "capacity_utilisation",
        "category": "activity",
        "transformation": "none; quarterly utilisation rate/index",
        "cross_country_status": "strict_common",
        "reason": "Same capacity-utilisation concept available in South Africa.",
    },
    {
        "feature": "nominal_exports_growth_qoq",
        "category": "external",
        "transformation": "quarter-on-quarter percentage growth of nominal merchandise exports",
        "cross_country_status": "brazil_specific",
        "reason": "Not strictly equivalent to South African real exports of goods/services.",
    },
    {
        "feature": "private_credit_level_growth_qoq",
        "category": "credit",
        "transformation": "quarter-on-quarter percentage growth of BIS/FRED USD credit level",
        "cross_country_status": "brazil_specific",
        "reason": "Not equivalent to South African private-credit/GDP change and partly affected by currency valuation.",
    },
    {
        "feature": "reserves_growth_qoq",
        "category": "external buffers",
        "transformation": "quarter-on-quarter percentage growth of quarter-end reserves ex gold",
        "cross_country_status": "brazil_specific_optional",
        "reason": "Useful Brazil predictor but not part of the frozen SA common feature intersection.",
    },
    {
        "feature": "short_rate_change_pp",
        "category": "interest rates",
        "transformation": "quarterly short-rate first difference",
        "cross_country_status": "brazil_specific_optional",
        "reason": "SELIC change already represents the primary policy-rate concept.",
    },
    {
        "feature": "yield_spread_10y_short",
        "category": "interest rates",
        "transformation": "10Y yield minus Treasury-bill yield",
        "cross_country_status": "secondary_recent",
        "reason": "Conceptually comparable but starts only in 2007 and would materially shorten the sample.",
    },
    {
        "feature": "retail_sales_growth",
        "category": "activity",
        "transformation": "none; source already growth rate from previous period",
        "cross_country_status": "secondary_recent",
        "reason": "Starts in 2000; useful expanded-set candidate.",
    },
    {
        "feature": "business_confidence",
        "category": "survey",
        "transformation": "quarterly average index",
        "cross_country_status": "secondary_recent",
        "reason": "Starts in 2008.",
    },
    {
        "feature": "credit_gap",
        "category": "financial cycle",
        "transformation": "none; BIS gap to trend",
        "cross_country_status": "secondary_recent",
        "reason": "Starts in 2006 and is not the same concept as credit/GDP.",
    },
    {
        "feature": "electricity_growth_qoq",
        "category": "activity",
        "transformation": "quarter-on-quarter percentage growth of quarterly electricity generation",
        "cross_country_status": "secondary_recent",
        "reason": "Starts in 2015 and is too short for the long common model.",
    },
]

dictionary_df = pd.DataFrame(
    dictionary_rows
)

dictionary_df.to_csv(
    DICTIONARY_OUT,
    index=False,
)

# ============================================================
# FEATURE SETS
# ============================================================
COMMON_STRICT_11 = [
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

COMMON_PLUS_FISCAL_12 = (
    COMMON_STRICT_11
    + [
        "government_consumption_growth_qoq",
    ]
)

# Brazil-specific long set. This is NOT the primary cross-country comparison.
BRAZIL_LONG_RICH = (
    COMMON_PLUS_FISCAL_12
    + [
        "nominal_exports_growth_qoq",
        "private_credit_level_growth_qoq",
        "reserves_growth_qoq",
    ]
)

# Secondary expanded set to show how much sample is lost by recent variables.
BRAZIL_RECENT_RICH = (
    BRAZIL_LONG_RICH
    + [
        "retail_sales_growth",
        "yield_spread_10y_short",
        "business_confidence",
        "credit_gap",
        "electricity_growth_qoq",
    ]
)

FEATURE_SETS = {
    "A_common_strict_11":
        COMMON_STRICT_11,
    "B_common_plus_fiscal_12":
        COMMON_PLUS_FISCAL_12,
    "C_brazil_long_rich_15":
        BRAZIL_LONG_RICH,
    "D_brazil_recent_rich_20":
        BRAZIL_RECENT_RICH,
}

with open(
    FEATURE_SETS_OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        FEATURE_SETS,
        f,
        indent=2,
    )

# ============================================================
# FEATURE COVERAGE
# ============================================================
coverage_df = coverage_stats(
    features,
    numeric_cols,
    DIAGNOSTIC_START,
    DIAGNOSTIC_END,
).sort_values(
    [
        "coverage_pct",
        "feature",
    ],
    ascending=[
        False,
        True,
    ],
)

coverage_df.to_csv(
    COVERAGE_OUT,
    index=False,
)

# ============================================================
# FEATURE-SET FEASIBILITY
# ============================================================
feasibility_rows = []

for set_name, cols in (
    FEATURE_SETS.items()
):
    required_cols = (
        [
            "gdp_growth_qoq",
        ]
        + cols
    )

    block = (
        longest_consecutive_complete_block(
            features,
            required_cols,
        )
    )

    seqs = sequence_count(
        block["quarters"]
    )

    feasibility_rows.append({
        "feature_set":
            set_name,
        "exogenous_feature_count":
            len(cols),
        "plus_lagged_gdp_channel":
            1,
        "total_model_input_channels":
            len(cols) + 1,
        "longest_complete_start":
            block["start"],
        "longest_complete_end":
            block["end"],
        "longest_complete_quarters":
            block["quarters"],
        "lookback_quarters":
            LOOKBACK_QUARTERS,
        "forecast_horizon":
            FORECAST_HORIZON,
        "potential_sequences_before_split":
            seqs,
        "features":
            " | ".join(cols),
    })

feasibility_df = pd.DataFrame(
    feasibility_rows
)

feasibility_df.to_csv(
    FEASIBILITY_OUT,
    index=False,
)

# ============================================================
# SAVE ENGINEERED FEATURES
# ============================================================
features.to_csv(
    FEATURES_OUT,
    index=False,
)

# ============================================================
# FIGURE
# ============================================================
plot_df = (
    coverage_df
    .sort_values(
        "coverage_pct",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(
        11,
        max(
            7,
            0.34
            * len(plot_df),
        ),
    )
)

ax.barh(
    plot_df["feature"],
    plot_df["coverage_pct"],
)

ax.set_title(
    "Brazil Engineered-Feature Coverage, 1996Q1-2025Q4"
)

ax.set_xlabel(
    "Coverage (%)"
)

ax.set_ylabel(
    "Engineered feature"
)

ax.set_xlim(
    0,
    105,
)

fig.tight_layout()

figure_path = (
    FIG_DIR
    / "brazil_engineered_feature_coverage.png"
)

fig.savefig(
    figure_path,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# SUMMARY
# ============================================================
summary_lines = [
    "=" * 78,
    "BRAZIL FEATURE ENGINEERING + COMMON-SET FEASIBILITY",
    "=" * 78,
    "",
    f"Quarterly rows in harmonised input: {len(df)}",
    f"Engineered numeric features (including GDP growth): {len(numeric_cols)}",
    "",
    "Primary cross-country principle:",
    "  Use only genuinely comparable feature concepts across countries.",
    "  Do not relabel Brazil nominal exports as real exports.",
    "  Do not relabel Brazil private-credit level growth as private-credit/GDP change.",
    "",
    "Feature-set feasibility:",
]

for _, row in (
    feasibility_df.iterrows()
):
    start = (
        pd.to_datetime(
            row[
                "longest_complete_start"
            ]
        ).date()
        if pd.notna(
            row[
                "longest_complete_start"
            ]
        )
        else "NA"
    )

    end = (
        pd.to_datetime(
            row[
                "longest_complete_end"
            ]
        ).date()
        if pd.notna(
            row[
                "longest_complete_end"
            ]
        )
        else "NA"
    )

    summary_lines.append(
        f"  {row['feature_set']}: "
        f"{int(row['exogenous_feature_count'])} exogenous + GDP history, "
        f"{int(row['longest_complete_quarters'])} consecutive quarters "
        f"({start} -> {end}), "
        f"{int(row['potential_sequences_before_split'])} potential 8q->1q sequences."
    )

summary_lines.extend([
    "",
    "Interpretation:",
    "  - A_common_strict_11 is the cleanest long cross-country candidate.",
    "  - B_common_plus_fiscal_12 tests whether keeping government consumption is worth losing the late-2025 Brazil quarters.",
    "  - C_brazil_long_rich_15 is useful only as a Brazil-specific sensitivity analysis.",
    "  - D_brazil_recent_rich_20 is exploratory and is expected to be far too short for the primary experiment.",
    "  - US GDP growth is a shared global control already used in the South African project and can be joined later if retained in the final common specification.",
    "  - No model is trained and no final train/validation/test split is created in this script.",
])

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
print("16 - BRAZIL FEATURE ENGINEERING + COMMON-SET FEASIBILITY")
print("=" * 78)

print("\nEngineered feature coverage, 1996Q1-2025Q4:")
print(
    coverage_df[
        [
            "feature",
            "non_missing_quarters",
            "expected_quarters",
            "coverage_pct",
            "first_available",
            "last_available",
        ]
    ].to_string(
        index=False
    )
)

print("\nFeature-set feasibility:")
print(
    feasibility_df[
        [
            "feature_set",
            "exogenous_feature_count",
            "total_model_input_channels",
            "longest_complete_start",
            "longest_complete_end",
            "longest_complete_quarters",
            "potential_sequences_before_split",
        ]
    ].to_string(
        index=False
    )
)

print("\nPrimary candidate feature sets:")
print("\nA_common_strict_11:")
for feature in COMMON_STRICT_11:
    print(f"  - {feature}")

print("\nB_common_plus_fiscal_12:")
for feature in COMMON_PLUS_FISCAL_12:
    print(f"  - {feature}")

print("\nImportant equivalence exclusions:")
print(
    "  - nominal_exports_growth_qoq is NOT part of the strict common set."
)
print(
    "  - private_credit_level_growth_qoq is NOT part of the strict common set."
)
print(
    "  - yield_spread_10y_short is NOT part of the long common set because Brazil begins only in 2007."
)
print(
    "  - FX volatility is NOT engineered because the available daily Brazil history begins only in 2021."
)

print("\nOutputs:")
print(f"  {FEATURES_OUT}")
print(f"  {DICTIONARY_OUT}")
print(f"  {COVERAGE_OUT}")
print(f"  {FEASIBILITY_OUT}")
print(f"  {FEATURE_SETS_OUT}")
print(f"  {SUMMARY_OUT}")

print("\n300-DPI figure:")
print(f"  {figure_path}")


if __name__ == "__main__":
    pass
