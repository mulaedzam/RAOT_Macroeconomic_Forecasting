from pathlib import Path
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "feature_analysis"
FIGURE_DIR = PROJECT_ROOT / "figures" / "feature_analysis"

DATA_PATH = FEATURE_DIR / "engineered_features_target_window.csv"
REGISTRY_PATH = FEATURE_DIR / "feature_engineering_registry.csv"
AUDIT_PATH = FEATURE_DIR / "feature_engineering_audit.csv"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DESC_OUT = ANALYSIS_DIR / "descriptive_statistics.csv"
STATIONARITY_OUT = ANALYSIS_DIR / "stationarity_tests.csv"
TARGET_CORR_OUT = ANALYSIS_DIR / "target_correlations.csv"
HIGH_CORR_OUT = ANALYSIS_DIR / "high_correlation_pairs.csv"
FEATURE_SETS_OUT = ANALYSIS_DIR / "candidate_feature_sets.csv"
FEATURE_SET_SUMMARY_OUT = ANALYSIS_DIR / "candidate_feature_set_summary.csv"

FIGURE_DPI = 300
TARGET = "gdp_growth_qoq"
HIGH_CORR_THRESHOLD = 0.90
MIN_STATIONARITY_OBS = 24
MIN_PAIR_OBS = 24

# ============================================================
# CANDIDATE FEATURE SETS
# ============================================================
# These are candidate experiment pools, not irreversible final selections.
# Set A contains broad, longer-history macroeconomic drivers.
# Set B adds richer domestic, financial and external activity indicators.
# Set C adds shorter-history variables for a recent-data robustness experiment.
FEATURE_SETS = {
    "A_basic": [
        "headline_inflation_yoy", "repo_rate_change", "yield_spread_10y_3m",
        "m3_growth_qoq", "private_credit_gdp_change", "usd_zar_change_qoq",
        "exports_growth_qoq", "imports_growth_qoq", "investment_growth_qoq",
        "private_consumption_growth_qoq", "manufacturing_growth_qoq",
        "us_gdp_growth_qoq",
    ],
    "B_expanded": [
        "headline_inflation_yoy", "repo_rate_change", "yield_spread_10y_3m",
        "m3_growth_qoq", "private_credit_gdp_change", "usd_zar_change_qoq",
        "exports_growth_qoq", "imports_growth_qoq", "investment_growth_qoq",
        "private_consumption_growth_qoq", "manufacturing_growth_qoq",
        "us_gdp_growth_qoq", "share_price_return_qoq", "reer_change_qoq",
        "sa_usd_zar_volatility", "reserves_growth_qoq", "current_account_change",
        "government_consumption_growth_qoq", "retail_sales_growth_yoy",
        "capacity_utilisation", "business_confidence", "electricity_growth_qoq",
        "building_plans_growth_qoq", "wholesale_growth_qoq",
        "liquidations_change_yoy",
    ],
    "C_recent_rich": [
        "headline_inflation_yoy", "repo_rate_change", "yield_spread_10y_3m",
        "m3_growth_qoq", "private_credit_gdp_change", "usd_zar_change_qoq",
        "exports_growth_qoq", "imports_growth_qoq", "investment_growth_qoq",
        "private_consumption_growth_qoq", "manufacturing_growth_qoq",
        "us_gdp_growth_qoq", "share_price_return_qoq", "reer_change_qoq",
        "sa_usd_zar_volatility", "reserves_growth_qoq", "current_account_change",
        "government_consumption_growth_qoq", "retail_sales_growth_yoy",
        "capacity_utilisation", "business_confidence", "electricity_growth_qoq",
        "building_plans_growth_qoq", "wholesale_growth_qoq",
        "liquidations_change_yoy", "unemployment_change_pp",
        "labour_force_participation_change_pp", "household_debt_gdp_change",
        "tourism_growth_qoq", "food_beverage_growth_qoq", "freight_growth_qoq",
        "ppi_inflation_yoy", "export_uvi_growth_qoq", "import_uvi_growth_qoq",
        "trade_price_ratio_change", "motor_trade_growth_qoq",
    ],
}

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
registry = pd.read_csv(REGISTRY_PATH)
audit = pd.read_csv(AUDIT_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found in {DATA_PATH}")

predictors = [c for c in df.columns if c not in {"date", TARGET}]
missing_from_data = sorted({f for items in FEATURE_SETS.values() for f in items} - set(df.columns))
if missing_from_data:
    raise ValueError(f"Candidate features not found in engineered dataset: {missing_from_data}")

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================
# Descriptive statistics help identify extreme scale, skewness and remaining missingness.
desc_rows = []
for feature in [TARGET] + predictors:
    x = pd.to_numeric(df[feature], errors="coerce")
    valid = x.dropna()
    desc_rows.append({
        "feature_name": feature,
        "observations": int(valid.count()),
        "missing_values": int(x.isna().sum()),
        "coverage_pct": round(100 * x.notna().mean(), 2),
        "mean": valid.mean(),
        "std": valid.std(),
        "median": valid.median(),
        "q1": valid.quantile(0.25),
        "q3": valid.quantile(0.75),
        "minimum": valid.min(),
        "maximum": valid.max(),
        "skewness": valid.skew(),
        "kurtosis": valid.kurt(),
    })

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(DESC_OUT, index=False)

# ============================================================
# STATIONARITY TESTS
# ============================================================
def stationarity_test(series):
    """
    Run ADF and KPSS as complementary stationarity diagnostics.

    ADF H0: unit root/non-stationary.
    KPSS H0: stationary.
    Agreement between ADF p<0.05 and KPSS p>0.05 gives stronger evidence of
    stationarity than relying on only one test.
    """
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < MIN_STATIONARITY_OBS or x.nunique() < 3:
        return {"n": len(x), "adf_stat": np.nan, "adf_p": np.nan,
                "kpss_stat": np.nan, "kpss_p": np.nan, "stationarity": "insufficient_data"}

    try:
        adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
    except Exception:
        adf_stat, adf_p = np.nan, np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan

    if pd.notna(adf_p) and pd.notna(kpss_p):
        if adf_p < 0.05 and kpss_p > 0.05:
            status = "likely_stationary"
        elif adf_p >= 0.05 and kpss_p <= 0.05:
            status = "likely_nonstationary"
        else:
            status = "mixed_evidence"
    else:
        status = "test_failed"

    return {"n": len(x), "adf_stat": adf_stat, "adf_p": adf_p,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p, "stationarity": status}


stationarity_rows = []
for feature in [TARGET] + predictors:
    result = stationarity_test(df[feature])
    result["feature_name"] = feature
    stationarity_rows.append(result)

stationarity_df = pd.DataFrame(stationarity_rows)[
    ["feature_name", "n", "adf_stat", "adf_p", "kpss_stat", "kpss_p", "stationarity"]
]
stationarity_df.to_csv(STATIONARITY_OUT, index=False)

# ============================================================
# TARGET CORRELATIONS
# ============================================================
# Correlations are descriptive only. They are NOT used here to select final
# predictors, because selecting features using the full sample could leak
# information from the eventual validation/test period.
corr_rows = []
for feature in predictors:
    pair = df[[TARGET, feature]].dropna()
    if len(pair) < MIN_PAIR_OBS:
        pearson = spearman = np.nan
    else:
        pearson = pair[TARGET].corr(pair[feature], method="pearson")
        spearman = pair[TARGET].corr(pair[feature], method="spearman")

    corr_rows.append({
        "feature_name": feature,
        "paired_observations": len(pair),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "absolute_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
    })

target_corr_df = pd.DataFrame(corr_rows).sort_values("absolute_pearson", ascending=False)
target_corr_df.to_csv(TARGET_CORR_OUT, index=False)

# ============================================================
# PAIRWISE REDUNDANCY / MULTICOLLINEARITY SCREEN
# ============================================================
# Very highly correlated predictors can provide largely duplicated information.
# This is a diagnostic only; economic meaning and training-only analysis will
# determine whether either variable should eventually be removed.
high_corr_rows = []
for a, b in combinations(predictors, 2):
    pair = df[[a, b]].dropna()
    if len(pair) < MIN_PAIR_OBS:
        continue
    corr = pair[a].corr(pair[b])
    if pd.notna(corr) and abs(corr) >= HIGH_CORR_THRESHOLD:
        high_corr_rows.append({
            "feature_1": a,
            "feature_2": b,
            "paired_observations": len(pair),
            "pearson_correlation": corr,
            "absolute_correlation": abs(corr),
        })

high_corr_df = pd.DataFrame(high_corr_rows)
if len(high_corr_df):
    high_corr_df = high_corr_df.sort_values("absolute_correlation", ascending=False)
else:
    high_corr_df = pd.DataFrame(columns=[
        "feature_1", "feature_2", "paired_observations",
        "pearson_correlation", "absolute_correlation"
    ])
high_corr_df.to_csv(HIGH_CORR_OUT, index=False)

# ============================================================
# CANDIDATE FEATURE SET FEASIBILITY
# ============================================================
# Each experiment must compare models on identical dates. Therefore, the script
# reports the complete-case window produced by each candidate feature set rather
# than silently dropping different rows for different models.
set_rows = []
set_summary_rows = []

for set_name, feature_list in FEATURE_SETS.items():
    for order, feature in enumerate(feature_list, start=1):
        meta = audit[audit["feature_name"] == feature]
        set_rows.append({
            "feature_set": set_name,
            "feature_order": order,
            "feature_name": feature,
            "suggested_tier": meta["suggested_tier"].iloc[0] if len(meta) else None,
            "individual_coverage_pct": meta["coverage_pct"].iloc[0] if len(meta) else None,
        })

    cols = ["date", TARGET] + feature_list
    complete = df[cols].dropna()

    set_summary_rows.append({
        "feature_set": set_name,
        "n_predictors": len(feature_list),
        "total_target_rows": int(df[TARGET].notna().sum()),
        "complete_case_rows": len(complete),
        "complete_case_pct": round(100 * len(complete) / df[TARGET].notna().sum(), 2),
        "start_date": complete["date"].min().date() if len(complete) else None,
        "end_date": complete["date"].max().date() if len(complete) else None,
    })

feature_sets_df = pd.DataFrame(set_rows)
feature_set_summary_df = pd.DataFrame(set_summary_rows)
feature_sets_df.to_csv(FEATURE_SETS_OUT, index=False)
feature_set_summary_df.to_csv(FEATURE_SET_SUMMARY_OUT, index=False)

# ============================================================
# FIGURE 1: GDP GROWTH TARGET
# ============================================================
# The target plot provides a first visual check of volatility, crises and extreme
# observations that will influence model training and error metrics.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df[TARGET], linewidth=1.2)
ax.axhline(0, linewidth=0.8)
ax.set_title("South African Real GDP Quarter-on-Quarter Growth")
ax.set_xlabel("Date")
ax.set_ylabel("QoQ log growth (%)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "gdp_growth_target.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 2: BASIC FEATURE CORRELATION HEATMAP
# ============================================================
# The heatmap is limited to Set A plus the target so it remains readable.
# Missing values are handled pairwise by pandas correlation estimation.
heat_cols = [TARGET] + FEATURE_SETS["A_basic"]
corr_matrix = df[heat_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
image = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(heat_cols)))
ax.set_yticks(np.arange(len(heat_cols)))
ax.set_xticklabels(heat_cols, rotation=90)
ax.set_yticklabels(heat_cols)
ax.set_title("Correlation Matrix: Basic Candidate Feature Set")
fig.colorbar(image, ax=ax, label="Pearson correlation")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "basic_feature_correlation_heatmap.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 3: DESCRIPTIVE TARGET CORRELATIONS
# ============================================================
# Show only the 20 strongest absolute Pearson correlations for readability.
plot_corr = target_corr_df.dropna(subset=["pearson_correlation"]).head(20).sort_values(
    "pearson_correlation"
)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(plot_corr["feature_name"], plot_corr["pearson_correlation"])
ax.axvline(0, linewidth=0.8)
ax.set_title("Top Descriptive Correlations with GDP Growth")
ax.set_xlabel("Pearson correlation")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "top_target_correlations.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 4: CANDIDATE FEATURE-SET SAMPLE SIZE
# ============================================================
# This figure makes the trade-off between richer feature sets and shorter
# complete-case histories explicit before modelling begins.
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(feature_set_summary_df["feature_set"], feature_set_summary_df["complete_case_rows"])
ax.set_title("Complete Quarterly Observations by Candidate Feature Set")
ax.set_xlabel("Candidate feature set")
ax.set_ylabel("Complete quarterly observations")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "candidate_feature_set_sample_size.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("06 - EXPLORATORY FEATURE ANALYSIS COMPLETE")
print("=" * 75)
print(f"\nTarget rows: {int(df[TARGET].notna().sum())}")
print(f"Target period: {df.loc[df[TARGET].notna(), 'date'].min().date()} to "
      f"{df.loc[df[TARGET].notna(), 'date'].max().date()}")

print("\nStationarity summary:\n")
print(stationarity_df["stationarity"].value_counts(dropna=False).to_string())

print("\nTop 10 descriptive correlations with GDP growth:\n")
print(target_corr_df[
    ["feature_name", "paired_observations", "pearson_correlation", "spearman_correlation"]
].head(10).to_string(index=False))

print(f"\nHighly correlated predictor pairs (|r| >= {HIGH_CORR_THRESHOLD:.2f}): {len(high_corr_df)}")
if len(high_corr_df):
    print(high_corr_df.head(15).to_string(index=False))

print("\nCandidate feature-set feasibility:\n")
print(feature_set_summary_df.to_string(index=False))

print("\nOutputs:")
print(f"  {DESC_OUT}")
print(f"  {STATIONARITY_OUT}")
print(f"  {TARGET_CORR_OUT}")
print(f"  {HIGH_CORR_OUT}")
print(f"  {FEATURE_SETS_OUT}")
print(f"  {FEATURE_SET_SUMMARY_OUT}")
print(f"\n300-DPI figures:\n  {FIGURE_DIR}")


if __name__ == "__main__":
    pass
