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
FIGURE_DIR = PROJECT_ROOT / "figures" / "experiment_design"

DATA_PATH = FEATURE_DIR / "engineered_features_target_window.csv"
DESIGN_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SET_MEMBERSHIP_OUT = DESIGN_DIR / "experiment_feature_sets.csv"
SET_FEASIBILITY_OUT = DESIGN_DIR / "experiment_feature_set_feasibility.csv"
SEQUENCE_AUDIT_OUT = DESIGN_DIR / "sequence_design_audit.csv"
SPLIT_OUT = DESIGN_DIR / "chronological_split_assignment.csv"
FIGURE_DPI = 300

TARGET = "gdp_growth_qoq"
FORECAST_HORIZON = 1
LOOKBACKS = [4, 8, 12]

# ============================================================
# LOCKED CANDIDATE EXPERIMENT SETS
# ============================================================
# A_BASIC is the compact long-history benchmark feature set.
A_BASIC = [
    "headline_inflation_yoy", "repo_rate_change", "yield_spread_10y_3m",
    "m3_growth_qoq", "private_credit_gdp_change", "usd_zar_change_qoq",
    "exports_growth_qoq", "imports_growth_qoq", "investment_growth_qoq",
    "private_consumption_growth_qoq", "manufacturing_growth_qoq",
    "us_gdp_growth_qoq",
]

# B_EXPANDED_REFINED adds broader macro-financial and real-activity information
# while deliberately excluding variables that would unnecessarily shorten or break
# the common sample (e.g. liquidations/current account/wholesale/reserves) or add
# avoidable redundancy. A and B can therefore be compared over the SAME dates.
B_EXPANDED_REFINED = A_BASIC + [
    "share_price_return_qoq",
    "reer_change_qoq",
    "sa_usd_zar_volatility",
    "reserves_growth_qoq",
    "government_consumption_growth_qoq",
    "retail_sales_growth_yoy",
    "capacity_utilisation",
    "business_confidence",
    "electricity_growth_qoq",
    "building_plans_growth_qoq",
]

# C_RECENT is retained only as an exploratory robustness pool. It is not intended
# to support the primary Transformer claim because its usable history is short.
C_RECENT = B_EXPANDED_REFINED + [
    "unemployment_change_pp",
    "labour_force_participation_change_pp",
    "household_debt_gdp_change",
    "tourism_growth_qoq",
    "food_beverage_growth_qoq",
    "freight_growth_qoq",
    "ppi_inflation_yoy",
    "export_uvi_growth_qoq",
    "import_uvi_growth_qoq",
    "trade_price_ratio_change",
    "motor_trade_growth_qoq",
]

FEATURE_SETS = {
    "A_basic": A_BASIC,
    "B_expanded_refined": B_EXPANDED_REFINED,
    "C_recent_exploratory": C_RECENT,
}

# ============================================================
# CHRONOLOGICAL SPLIT DATES
# ============================================================
# Splits are defined by TARGET date, never randomly.
# Training ends before validation; validation ends before test.
# The test period begins in 2019 so it contains the COVID shock and subsequent
# recovery, giving a demanding genuinely unseen evaluation period.
TRAIN_END = pd.Timestamp("2013-10-01")   # 2013 Q4
VAL_START = pd.Timestamp("2014-01-01")   # 2014 Q1
VAL_END = pd.Timestamp("2018-10-01")     # 2018 Q4
TEST_START = pd.Timestamp("2019-01-01")  # 2019 Q1

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

required = {TARGET}
for feature_list in FEATURE_SETS.values():
    required.update(feature_list)

missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Required engineered variables are missing: {missing}")

# ============================================================
# FEATURE-SET FEASIBILITY
# ============================================================
membership_rows = []
feasibility_rows = []

for set_name, feature_list in FEATURE_SETS.items():
    for order, feature in enumerate(feature_list, start=1):
        membership_rows.append({
            "feature_set": set_name,
            "feature_order": order,
            "feature_name": feature,
        })

    complete = df[["date", TARGET] + feature_list].dropna()
    feasibility_rows.append({
        "feature_set": set_name,
        "n_predictors": len(feature_list),
        "complete_quarters": len(complete),
        "start_date": complete["date"].min().date() if len(complete) else None,
        "end_date": complete["date"].max().date() if len(complete) else None,
    })

membership_df = pd.DataFrame(membership_rows)
feasibility_df = pd.DataFrame(feasibility_rows)
membership_df.to_csv(SET_MEMBERSHIP_OUT, index=False)
feasibility_df.to_csv(SET_FEASIBILITY_OUT, index=False)

# ============================================================
# COMMON PRIMARY SAMPLE FOR A vs B
# ============================================================
# The principal feature-ablation comparison must use identical dates. We therefore
# construct one joint complete-case sample for A_BASIC and B_EXPANDED_REFINED.
primary_features = sorted(set(A_BASIC + B_EXPANDED_REFINED))
primary = df[["date", TARGET] + primary_features].dropna().copy().reset_index(drop=True)

if primary.empty:
    raise ValueError("No common complete-case sample exists for the primary feature sets.")

PRIMARY_START = primary["date"].min()
PRIMARY_END = primary["date"].max()

# ============================================================
# LEAKAGE-SAFE TARGET ALIGNMENT
# ============================================================
# For each sequence ending at quarter t, the target is GDP growth at t+1.
# Thus the model never uses quarter t+1 predictors to forecast quarter t+1 GDP.
def build_sequence_metadata(data, feature_list, lookback, horizon=1):
    """
    Return metadata for leakage-safe, strictly consecutive quarterly sequences.

    Input window: t-lookback+1 ... t
    Forecast target: t+horizon
    """
    work = data[["date", TARGET] + feature_list].dropna().reset_index(drop=True)
    rows = []

    for target_idx in range(lookback + horizon - 1, len(work)):
        input_end_idx = target_idx - horizon
        input_start_idx = input_end_idx - lookback + 1
        if input_start_idx < 0:
            continue

        # A valid sequence must contain every consecutive calendar quarter.
        window_dates = work.loc[input_start_idx:target_idx, "date"].dt.to_period("Q")
        expected = pd.period_range(window_dates.iloc[0], window_dates.iloc[-1], freq="Q")
        if len(window_dates) != len(expected) or not np.array_equal(
            window_dates.astype(str).to_numpy(), expected.astype(str).to_numpy()
        ):
            continue

        rows.append({
            "input_start": work.loc[input_start_idx, "date"],
            "input_end": work.loc[input_end_idx, "date"],
            "target_date": work.loc[target_idx, "date"],
            "target_value": work.loc[target_idx, TARGET],
        })

    return pd.DataFrame(rows)


def assign_split(target_date):
    """Assign each forecast target to a chronological train/validation/test block."""
    if target_date <= TRAIN_END:
        return "train"
    if VAL_START <= target_date <= VAL_END:
        return "validation"
    if target_date >= TEST_START:
        return "test"
    return "unassigned"


# ============================================================
# SEQUENCE SAMPLE AUDIT
# ============================================================
sequence_rows = []
split_rows = []

for set_name in ["A_basic", "B_expanded_refined"]:
    feature_list = FEATURE_SETS[set_name]

    # Use the exact same joint-complete dates for A and B so model comparisons
    # cannot be driven by different economic periods or missing-data patterns.
    set_data = primary[["date", TARGET] + feature_list].copy()

    for lookback in LOOKBACKS:
        meta = build_sequence_metadata(set_data, feature_list, lookback, FORECAST_HORIZON)
        if meta.empty:
            continue

        meta["split"] = meta["target_date"].apply(assign_split)
        counts = meta["split"].value_counts()

        sequence_rows.append({
            "feature_set": set_name,
            "lookback_quarters": lookback,
            "forecast_horizon_quarters": FORECAST_HORIZON,
            "predictors": len(feature_list),
            "common_data_start": set_data["date"].min().date(),
            "common_data_end": set_data["date"].max().date(),
            "total_sequences": len(meta),
            "train_sequences": int(counts.get("train", 0)),
            "validation_sequences": int(counts.get("validation", 0)),
            "test_sequences": int(counts.get("test", 0)),
            "first_target_date": meta["target_date"].min().date(),
            "last_target_date": meta["target_date"].max().date(),
        })

        # Save detailed assignment for the preferred 8-quarter design.
        if lookback == 8:
            temp = meta.copy()
            temp.insert(0, "feature_set", set_name)
            temp.insert(1, "lookback_quarters", lookback)
            split_rows.append(temp)

sequence_audit_df = pd.DataFrame(sequence_rows)
sequence_audit_df.to_csv(SEQUENCE_AUDIT_OUT, index=False)

split_df = pd.concat(split_rows, ignore_index=True) if split_rows else pd.DataFrame()
split_df.to_csv(SPLIT_OUT, index=False, date_format="%Y-%m-%d")

# ============================================================
# FIGURE 1: CHRONOLOGICAL SPLIT
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(primary["date"], primary[TARGET], linewidth=1.1)
ax.axvline(VAL_START, linestyle="--", linewidth=1, label="Validation begins")
ax.axvline(TEST_START, linestyle="--", linewidth=1, label="Test begins")
ax.axhline(0, linewidth=0.8)
ax.set_title("Chronological Train, Validation and Test Design")
ax.set_xlabel("Target quarter")
ax.set_ylabel("GDP QoQ growth (%)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "chronological_split_design.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 2: AVAILABLE SEQUENCES BY LOOKBACK
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
for set_name, group in sequence_audit_df.groupby("feature_set"):
    ax.plot(group["lookback_quarters"], group["total_sequences"],
            marker="o", label=set_name)
ax.set_xticks(LOOKBACKS)
ax.set_title("Available One-Quarter-Ahead Sequences by Lookback")
ax.set_xlabel("Lookback window (quarters)")
ax.set_ylabel("Number of forecast sequences")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sequence_counts_by_lookback.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("07 - EXPERIMENT DESIGN COMPLETE")
print("=" * 75)

print("\nCandidate feature-set feasibility:\n")
print(feasibility_df.to_string(index=False))

print("\nPrimary A vs B common sample:")
print(f"  Start: {PRIMARY_START.date()}")
print(f"  End:   {PRIMARY_END.date()}")
print(f"  Quarters: {len(primary)}")

print("\nChronological target splits:")
print(f"  Train:      <= {TRAIN_END.date()}")
print(f"  Validation: {VAL_START.date()} to {VAL_END.date()}")
print(f"  Test:       >= {TEST_START.date()}")

print("\nLeakage-safe sequence counts:\n")
print(sequence_audit_df.to_string(index=False))

print("\nPreferred initial lookback:")
preferred = sequence_audit_df[sequence_audit_df["lookback_quarters"] == 8]
print(preferred[
    ["feature_set", "predictors", "total_sequences",
     "train_sequences", "validation_sequences", "test_sequences"]
].to_string(index=False))

print("\nOutputs:")
print(f"  {SET_MEMBERSHIP_OUT}")
print(f"  {SET_FEASIBILITY_OUT}")
print(f"  {SEQUENCE_AUDIT_OUT}")
print(f"  {SPLIT_OUT}")
print(f"\n300-DPI figures:\n  {FIGURE_DIR}")


if __name__ == "__main__":
    pass
