from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_data"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "availability"
FIGURE_DIR = PROJECT_ROOT / "figures" / "availability"

INPUT_FEATURE_PATH = MODEL_DATA_DIR / "B_expanded_refined_input_features.csv"
OUT_PATH = OUT_DIR / "feature_release_metadata.csv"
FIGURE_PATH = FIGURE_DIR / "feature_release_delays.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DPI = 300

# ============================================================
# RELEASE-AVAILABILITY METADATA
# ============================================================
# The private-credit/GDP series is deliberately assigned a longer delay because
# the project uses the BIS quarterly credit-to-GDP series, not a contemporaneous
# daily/monthly market variable.
METADATA = {
    "gdp_growth_qoq": {
        "source_family": "IMF IFS / South African national accounts",
        "approx_release_delay_days": 75,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "South African quarterly GDP is released roughly 2-2.5 months after quarter-end; IMF/FRED series updates follow the national-accounts release.",
    },
    "headline_inflation_yoy": {
        "source_family": "Statistics South Africa CPI",
        "approx_release_delay_days": 20,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Monthly CPI is normally published around the third week of the following month.",
    },
    "repo_rate_change": {
        "source_family": "South African Reserve Bank MPC",
        "approx_release_delay_days": 0,
        "strict_quarter_end_shift_quarters": 0,
        "confidence": "high",
        "basis": "Policy-rate decisions are public on the MPC announcement date.",
    },
    "yield_spread_10y_3m": {
        "source_family": "Financial-market interest rates",
        "approx_release_delay_days": 0,
        "strict_quarter_end_shift_quarters": 0,
        "confidence": "medium",
        "basis": "Underlying bond/interbank rates are observable in financial markets contemporaneously, although the OECD monthly series used in the raw dataset is updated later.",
    },
    "m3_growth_qoq": {
        "source_family": "SARB money and banking / OECD series",
        "approx_release_delay_days": 30,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "medium",
        "basis": "SARB selected monthly monetary data are released around month-end for the preceding month; the project series is an OECD quarterly transformation.",
    },
    "private_credit_gdp_change": {
        "source_family": "BIS credit-to-GDP",
        "approx_release_delay_days": 165,
        "strict_quarter_end_shift_quarters": 2,
        "confidence": "high",
        "basis": "The project uses the BIS quarterly credit-to-GDP series, which is materially delayed relative to monthly bank-credit indicators.",
    },
    "usd_zar_change_qoq": {
        "source_family": "Daily USD/ZAR market rate",
        "approx_release_delay_days": 0,
        "strict_quarter_end_shift_quarters": 0,
        "confidence": "high",
        "basis": "Daily FX observations are contemporaneously observable.",
    },
    "exports_growth_qoq": {
        "source_family": "IMF IFS / South African national accounts",
        "approx_release_delay_days": 75,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "The project uses quarterly real national-accounts exports rather than faster customs trade values.",
    },
    "imports_growth_qoq": {
        "source_family": "IMF IFS / South African national accounts",
        "approx_release_delay_days": 75,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "The project uses quarterly real national-accounts imports rather than faster customs trade values.",
    },
    "investment_growth_qoq": {
        "source_family": "IMF IFS / South African national accounts",
        "approx_release_delay_days": 75,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Gross fixed capital formation is released with the quarterly national accounts.",
    },
    "private_consumption_growth_qoq": {
        "source_family": "IMF IFS / South African national accounts",
        "approx_release_delay_days": 75,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Private consumption is released with the quarterly national accounts.",
    },
    "manufacturing_growth_qoq": {
        "source_family": "Statistics South Africa manufacturing / OECD",
        "approx_release_delay_days": 42,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Stats SA manufacturing production is typically released about six weeks after the reference month.",
    },
    "us_gdp_growth_qoq": {
        "source_family": "US BEA real GDP",
        "approx_release_delay_days": 30,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "BEA advance GDP is normally released about one month after quarter-end.",
    },
    "share_price_return_qoq": {
        "source_family": "Equity-market prices",
        "approx_release_delay_days": 0,
        "strict_quarter_end_shift_quarters": 0,
        "confidence": "high",
        "basis": "Underlying share prices are observable contemporaneously even though the OECD monthly index used in the dataset is published later.",
    },
    "reer_change_qoq": {
        "source_family": "BIS real effective exchange rate",
        "approx_release_delay_days": 20,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "medium",
        "basis": "BIS monthly REER updates are generally available during the following month.",
    },
    "sa_usd_zar_volatility": {
        "source_family": "Daily USD/ZAR market rate",
        "approx_release_delay_days": 0,
        "strict_quarter_end_shift_quarters": 0,
        "confidence": "high",
        "basis": "Volatility is computed from contemporaneously observed daily FX returns.",
    },
    "reserves_growth_qoq": {
        "source_family": "SARB official gold and foreign-exchange reserves / IMF",
        "approx_release_delay_days": 7,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "SARB reserve notices are generally released in the first week after month-end.",
    },
    "government_consumption_growth_qoq": {
        "source_family": "IMF IFS / South African national accounts",
        "approx_release_delay_days": 75,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Government consumption is released with the quarterly national accounts.",
    },
    "retail_sales_growth_yoy": {
        "source_family": "Statistics South Africa retail trade / OECD",
        "approx_release_delay_days": 50,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Retail trade is generally released around seven weeks after the reference month.",
    },
    "capacity_utilisation": {
        "source_family": "Manufacturing capacity utilisation",
        "approx_release_delay_days": 65,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Capacity-utilisation releases are relatively delayed; current Stats SA/OECD schedules imply roughly two months.",
    },
    "business_confidence": {
        "source_family": "Business confidence survey",
        "approx_release_delay_days": 0,
        "strict_quarter_end_shift_quarters": 0,
        "confidence": "medium",
        "basis": "Business-confidence survey results are generally available during the reference quarter; exact historical release timing varies by provider.",
    },
    "electricity_growth_qoq": {
        "source_family": "Statistics South Africa electricity P4141",
        "approx_release_delay_days": 37,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Monthly electricity data are typically published around five to six weeks after the reference month.",
    },
    "building_plans_growth_qoq": {
        "source_family": "Statistics South Africa building statistics P5041.1",
        "approx_release_delay_days": 50,
        "strict_quarter_end_shift_quarters": 1,
        "confidence": "high",
        "basis": "Monthly building statistics are typically published about seven weeks after the reference month.",
    },
}

# ============================================================
# LOAD THE ACTUAL INPUT ORDER FROM SCRIPT 08
# ============================================================
inputs = pd.read_csv(INPUT_FEATURE_PATH).sort_values("input_order").reset_index(drop=True)

required_cols = {"input_order", "input_feature"}
missing_cols = required_cols - set(inputs.columns)
if missing_cols:
    raise ValueError(f"Missing columns in input-feature file: {sorted(missing_cols)}")

input_features = inputs["input_feature"].tolist()

missing_metadata = [f for f in input_features if f not in METADATA]
extra_metadata = [f for f in METADATA if f not in input_features]

if missing_metadata:
    raise ValueError(
        "Release metadata missing for model inputs: "
        + ", ".join(missing_metadata)
    )

# ============================================================
# BUILD OUTPUT TABLE
# ============================================================
rows = []

for _, row in inputs.iterrows():
    feature = row["input_feature"]
    meta = METADATA[feature]

    delay_days = int(meta["approx_release_delay_days"])
    shift_q = int(meta["strict_quarter_end_shift_quarters"])

    rows.append({
        "input_order": int(row["input_order"]),
        "feature_name": feature,
        "input_role": row["role"] if "role" in inputs.columns else "predictor",
        "source_family": meta["source_family"],
        "approx_release_delay_days": delay_days,
        "release_delay_months_approx": round(delay_days / 30.4375, 2),
        "release_delay_score_0_1": round(min(delay_days / 180.0, 1.0), 6),
        "strict_quarter_end_shift_quarters": shift_q,
        "current_quarter_available_at_quarter_end": shift_q == 0,
        "metadata_confidence": meta["confidence"],
        "delay_basis": meta["basis"],
    })

metadata_df = pd.DataFrame(rows).sort_values("input_order").reset_index(drop=True)
metadata_df.to_csv(OUT_PATH, index=False)

# ============================================================
# AUDIT CHECKS
# ============================================================
if len(metadata_df) != len(inputs):
    raise RuntimeError("Metadata row count does not match the model input count.")

if metadata_df["feature_name"].duplicated().any():
    raise RuntimeError("Duplicate feature names found in release metadata.")

# ============================================================
# FIGURE
# ============================================================
plot_df = metadata_df.sort_values("approx_release_delay_days")

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    plot_df["feature_name"],
    plot_df["approx_release_delay_days"],
)
ax.set_title("Approximate Publication Delay of Transformer Inputs")
ax.set_xlabel("Approximate release delay after reference period (days)")
ax.set_ylabel("Input feature")
fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 78)
print("10G PREPARATION - RELEASE AVAILABILITY METADATA COMPLETE")
print("=" * 78)

print(f"\nModel input channels: {len(metadata_df)}")
print(f"Metadata rows:        {len(metadata_df)}")

print("\nStrict quarter-end availability:")
availability_summary = (
    metadata_df
    .groupby("strict_quarter_end_shift_quarters")
    .size()
    .rename("features")
    .reset_index()
)
print(availability_summary.to_string(index=False))

print("\nInputs available contemporaneously at quarter-end:")
print(
    metadata_df.loc[
        metadata_df["strict_quarter_end_shift_quarters"] == 0,
        ["input_order", "feature_name", "approx_release_delay_days"]
    ].to_string(index=False)
)

print("\nMost delayed inputs:")
print(
    metadata_df.nlargest(
        8,
        "approx_release_delay_days"
    )[
        [
            "feature_name",
            "approx_release_delay_days",
            "strict_quarter_end_shift_quarters",
            "metadata_confidence",
        ]
    ].to_string(index=False)
)

if extra_metadata:
    print("\nUnused metadata definitions:")
    print("  " + ", ".join(extra_metadata))

print("\nImportant methodological rule:")
print("  Release-aware attention must not merely down-weight information that")
print("  was unavailable at the forecast origin. Delayed features should first")
print("  be shifted to the latest genuinely available quarter, then release-delay")
print("  metadata can be supplied to the attention mechanism.")

print(f"\nMetadata:\n  {OUT_PATH}")
print(f"\n300-DPI figure:\n  {FIGURE_PATH}")


if __name__ == "__main__":
    pass
