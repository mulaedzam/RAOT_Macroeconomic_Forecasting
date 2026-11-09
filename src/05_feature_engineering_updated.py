from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUARTERLY_WIDE = PROJECT_ROOT / "data" / "interim" / "quarterly_harmonised_wide.csv"
DAILY_FX = PROJECT_ROOT / "data" / "interim" / "series" / "sa_usd_zar.csv"
MONTHLY_CPI = PROJECT_ROOT / "data" / "interim" / "series" / "sa_cpi_total_index.csv"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
FIGURE_DIR = PROJECT_ROOT / "figures" / "feature_engineering"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FULL_OUT = FEATURE_DIR / "engineered_features_full.csv"
TARGET_OUT = FEATURE_DIR / "engineered_features_target_window.csv"
REGISTRY_OUT = FEATURE_DIR / "feature_engineering_registry.csv"
AUDIT_OUT = FEATURE_DIR / "feature_engineering_audit.csv"
CPI_VALIDATION_OUT = FEATURE_DIR / "cpi_extension_validation.csv"
FIGURE_DPI = 300

# ============================================================
# HELPERS
# ============================================================
def log_change(series, periods=1):
    """Return 100 × log change; non-positive observations remain missing."""
    x = pd.to_numeric(series, errors="coerce")
    x = x.where(x > 0)
    return 100 * (np.log(x) - np.log(x.shift(periods)))


def pp_change(series, periods=1):
    """Return percentage-point / native-unit change without dividing by the old level."""
    return pd.to_numeric(series, errors="coerce").diff(periods)


def log1p_change(series, periods=1):
    """Log1p change is suitable for non-negative count data that may contain zeros."""
    x = pd.to_numeric(series, errors="coerce")
    x = x.where(x >= 0)
    return 100 * (np.log1p(x) - np.log1p(x.shift(periods)))


def quarterly_cpi_inflation():
    """
    Convert the official monthly Stats SA CPI index to quarterly YoY inflation.

    YoY CPI inflation uses the standard percentage-change formula against the
    same month a year earlier. The three monthly YoY rates are then averaged
    within each quarter; incomplete quarters are left missing.
    """
    cpi = pd.read_csv(MONTHLY_CPI)
    cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
    cpi["sa_cpi_total_index"] = pd.to_numeric(cpi["sa_cpi_total_index"], errors="coerce")
    cpi = cpi.dropna(subset=["date"]).sort_values("date")

    # Published CPI inflation is a standard 12-month percentage change, not a log change.
    cpi["stats_sa_cpi_inflation_yoy"] = (
        100 * (cpi["sa_cpi_total_index"] / cpi["sa_cpi_total_index"].shift(12) - 1)
    )
    cpi["quarter"] = cpi["date"].dt.to_period("Q")

    grouped = cpi.groupby("quarter")["stats_sa_cpi_inflation_yoy"].agg(["mean", "count"])
    grouped["stats_sa_cpi_inflation_yoy"] = grouped["mean"].where(grouped["count"] == 3)
    grouped["date"] = grouped.index.to_timestamp(how="start")
    return grouped[["date", "stats_sa_cpi_inflation_yoy"]].reset_index(drop=True)


def quarterly_fx_volatility():
    """
    Calculate within-quarter volatility from daily USD/ZAR log returns.

    Daily data are used instead of the quarterly mean because volatility describes
    uncertainty within the quarter, which would disappear after simple averaging.
    """
    df = pd.read_csv(DAILY_FX)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sa_usd_zar"] = pd.to_numeric(df["sa_usd_zar"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["fx_log_return"] = np.log(df["sa_usd_zar"].where(df["sa_usd_zar"] > 0)).diff()
    df["quarter"] = df["date"].dt.to_period("Q")

    grouped = df.groupby("quarter")["fx_log_return"].agg(["std", "count"])
    # Require enough trading-day returns so an unusually sparse quarter is not treated as reliable.
    grouped["sa_usd_zar_volatility"] = 100 * grouped["std"].where(grouped["count"] >= 40)
    grouped["date"] = grouped.index.to_timestamp(how="start")
    return grouped[["date", "sa_usd_zar_volatility"]].reset_index(drop=True)


def register(registry, name, group, sources, transformation, reason, tier):
    """Store the methodological justification for every engineered feature."""
    registry.append({
        "feature_name": name,
        "feature_group": group,
        "source_series": sources,
        "transformation": transformation,
        "reason": reason,
        "suggested_tier": tier,
    })


# ============================================================
# LOAD QUARTERLY DATA
# ============================================================
df = pd.read_csv(QUARTERLY_WIDE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

features = pd.DataFrame({"date": df["date"]})
registry = []

# ============================================================
# TARGET
# ============================================================
# GDP is transformed from a seasonally adjusted real level to QoQ log growth
# because the research forecasts quarterly economic growth rather than the GDP level.
features["gdp_growth_qoq"] = log_change(df["sa_real_gdp"])
register(registry, "gdp_growth_qoq", "target", "sa_real_gdp",
         "100 × quarterly log difference",
         "Creates the quarterly real GDP growth target from the seasonally adjusted GDP level.",
         "target")

# ============================================================
# PRICES
# ============================================================
# The historical YoY inflation series is retained wherever it already has a complete
# quarterly value. Stats SA CPI-derived YoY inflation fills only the missing tail,
# extending the feature without replacing the validated historical observations.
stats_sa_cpi_q = quarterly_cpi_inflation()
stats_sa_cpi_map = stats_sa_cpi_q.set_index("date")["stats_sa_cpi_inflation_yoy"]
stats_sa_cpi = df["date"].map(stats_sa_cpi_map)
historical_cpi = df["sa_headline_inflation_yoy"]

# Validate the overlapping period before using Stats SA to extend the series.
cpi_validation = pd.DataFrame({
    "date": df["date"],
    "historical_inflation_yoy": historical_cpi,
    "stats_sa_inflation_yoy": stats_sa_cpi,
}).dropna()
cpi_validation["difference"] = (
    cpi_validation["stats_sa_inflation_yoy"] - cpi_validation["historical_inflation_yoy"]
)
cpi_validation.to_csv(CPI_VALIDATION_OUT, index=False, date_format="%Y-%m-%d")

features["headline_inflation_yoy"] = historical_cpi.combine_first(stats_sa_cpi)
register(registry, "headline_inflation_yoy", "prices",
         "sa_headline_inflation_yoy; monthly sa_cpi_total_index",
         "historical published YoY rate; missing tail extended using quarterly average of monthly Stats SA 12-month CPI inflation",
         "Preserves the validated historical inflation series and uses official Stats SA CPI only to extend unavailable later quarters.",
         "core")

# Core CPI YoY is retained as an alternative measure of underlying inflation pressure.
features["core_inflation_yoy"] = df["sa_core_inflation_yoy"]
register(registry, "core_inflation_yoy", "prices", "sa_core_inflation_yoy",
         "none; already YoY rate",
         "Captures underlying inflation excluding food and energy without applying a second growth transformation.",
         "extended")

# PPI is an index; YoY log growth measures producer-price inflation while reducing seasonal sensitivity.
features["ppi_inflation_yoy"] = log_change(df["sa_ppi_final_manufactured_goods"], periods=4)
register(registry, "ppi_inflation_yoy", "prices", "sa_ppi_final_manufactured_goods",
         "100 × 4-quarter log difference",
         "Converts the producer-price index into annual producer inflation.",
         "recent")

# ============================================================
# MONETARY / INTEREST-RATE CONDITIONS
# ============================================================
# Rate levels are economically meaningful, while their first differences capture policy/market tightening or easing.
features["repo_rate"] = df["sa_repo_rate"]
features["repo_rate_change"] = pp_change(df["sa_repo_rate"])
register(registry, "repo_rate", "monetary", "sa_repo_rate", "quarterly mean level",
         "Represents the prevailing monetary-policy stance.", "core")
register(registry, "repo_rate_change", "monetary", "sa_repo_rate", "first difference in percentage points",
         "Captures the direction and magnitude of monetary-policy changes.", "core")

features["interbank_3m_rate"] = df["sa_interbank_3m_rate"]
features["interbank_3m_change"] = pp_change(df["sa_interbank_3m_rate"])
register(registry, "interbank_3m_rate", "monetary", "sa_interbank_3m_rate", "quarterly mean level",
         "Represents short-term money-market financing conditions.", "core")
register(registry, "interbank_3m_change", "monetary", "sa_interbank_3m_rate", "first difference in percentage points",
         "Captures changes in short-term market interest-rate conditions.", "core")

features["bond_yield_10y"] = df["sa_bond_yield_10y"]
features["bond_yield_10y_change"] = pp_change(df["sa_bond_yield_10y"])
register(registry, "bond_yield_10y", "monetary", "sa_bond_yield_10y", "quarterly mean level",
         "Represents long-term borrowing costs and market expectations.", "core")
register(registry, "bond_yield_10y_change", "monetary", "sa_bond_yield_10y", "first difference in percentage points",
         "Captures changes in long-term financial conditions.", "core")

# The yield spread summarises the slope of the yield curve and may contain forward-looking cycle information.
features["yield_spread_10y_3m"] = df["sa_bond_yield_10y"] - df["sa_interbank_3m_rate"]
register(registry, "yield_spread_10y_3m", "monetary", "sa_bond_yield_10y; sa_interbank_3m_rate",
         "10-year yield minus 3-month interbank rate",
         "Measures the yield-curve slope as a compact indicator of monetary and cyclical expectations.",
         "core")

# The real policy rate uses the extended inflation feature created above so its
# coverage is not unnecessarily truncated by the older CPI source.
features["real_policy_rate"] = df["sa_repo_rate"] - features["headline_inflation_yoy"]
register(registry, "real_policy_rate", "monetary", "sa_repo_rate; headline_inflation_yoy",
         "repo rate minus extended headline YoY inflation",
         "Approximates the inflation-adjusted policy stance using the continuous CPI inflation feature.",
         "extended")

# M3 is already supplied as QoQ growth and must not be percentage-changed again.
features["m3_growth_qoq"] = df["sa_m3_growth_qoq"]
register(registry, "m3_growth_qoq", "monetary", "sa_m3_growth_qoq",
         "none; already QoQ growth",
         "Preserves the published broad-money growth measure without double transformation.",
         "core")

# Credit is already expressed relative to GDP; a percentage-point change captures credit deepening/tightening.
features["private_credit_gdp_change"] = pp_change(df["sa_private_credit_gdp"])
register(registry, "private_credit_gdp_change", "monetary", "sa_private_credit_gdp",
         "first difference in percentage points",
         "Measures changes in bank credit relative to the size of the economy.",
         "core")

# ============================================================
# FINANCIAL MARKETS
# ============================================================
# Log changes convert index levels into scale-independent quarterly returns/changes.
features["share_price_return_qoq"] = log_change(df["sa_share_price_index"])
register(registry, "share_price_return_qoq", "financial_markets", "sa_share_price_index",
         "100 × quarterly log difference",
         "Transforms the share-price index into a quarterly market return.",
         "core")

features["reer_change_qoq"] = log_change(df["sa_reer"])
register(registry, "reer_change_qoq", "financial_markets", "sa_reer",
         "100 × quarterly log difference",
         "Measures quarterly movement in South Africa's real effective exchange rate.",
         "core")

features["house_price_growth_qoq"] = log_change(df["sa_real_house_price_index"])
register(registry, "house_price_growth_qoq", "financial_markets", "sa_real_house_price_index",
         "100 × quarterly log difference",
         "Converts the real residential property-price index into quarterly real house-price growth.",
         "extended")

# Household debt is a GDP ratio, so percentage-point changes are preferable to percentage growth of the ratio.
features["household_debt_gdp_change"] = pp_change(df["sa_household_debt_gdp"])
register(registry, "household_debt_gdp_change", "financial_markets", "sa_household_debt_gdp",
         "first difference in percentage points",
         "Measures changes in household leverage relative to GDP.",
         "recent")

# ============================================================
# EXTERNAL SECTOR
# ============================================================
# Quarterly USD/ZAR log change captures exchange-rate movement; daily volatility captures within-quarter uncertainty.
features["usd_zar_change_qoq"] = log_change(df["sa_usd_zar"])
register(registry, "usd_zar_change_qoq", "external", "sa_usd_zar",
         "100 × quarterly log difference of quarterly mean",
         "Measures the scale-independent quarterly movement in the rand/dollar exchange rate.",
         "core")

fx_vol = quarterly_fx_volatility()
features = features.merge(fx_vol, on="date", how="left")
register(registry, "sa_usd_zar_volatility", "external", "daily sa_usd_zar",
         "100 × standard deviation of daily log returns within quarter",
         "Captures exchange-rate uncertainty that is lost when daily observations are averaged to quarterly frequency.",
         "core")

# Reserves are a positive stock; log growth measures proportional accumulation or drawdown.
features["reserves_growth_qoq"] = log_change(df["sa_reserves_ex_gold"])
register(registry, "reserves_growth_qoq", "external", "sa_reserves_ex_gold",
         "100 × quarterly log difference",
         "Measures proportional changes in reserves after quarter-end stock harmonisation.",
         "core")

# Real trade levels are converted to quarterly growth to remove scale and trend.
features["exports_growth_qoq"] = log_change(df["sa_real_exports_goods"])
register(registry, "exports_growth_qoq", "external", "sa_real_exports_goods",
         "100 × quarterly log difference",
         "Measures quarterly growth in real exports of goods.",
         "core")

features["imports_growth_qoq"] = log_change(df["sa_real_imports_goods_services"])
register(registry, "imports_growth_qoq", "external", "sa_real_imports_goods_services",
         "100 × quarterly log difference",
         "Measures quarterly growth in real imports of goods and services.",
         "core")

# Current-account values can be positive or negative, so a log transform is inappropriate.
# A first difference captures movement without constructing an invalid ratio against real GDP.
features["current_account_change"] = pp_change(df["sa_current_account_balance"])
register(registry, "current_account_change", "external", "sa_current_account_balance",
         "first difference in source units",
         "Captures quarter-to-quarter movement while allowing both positive and negative balances.",
         "core")

# Old/new UVI levels use different bases. Growth rates are scale-invariant, so each version is
# transformed separately and the newer methodology is preferred where a valid growth rate exists.
export_old = log_change(df["sa_export_uvi_base2020"])
export_new = log_change(df["sa_export_uvi_base2024"])
import_old = log_change(df["sa_import_uvi_base2020"])
import_new = log_change(df["sa_import_uvi_base2024"])

features["export_uvi_growth_qoq"] = export_new.combine_first(export_old)
features["import_uvi_growth_qoq"] = import_new.combine_first(import_old)
register(registry, "export_uvi_growth_qoq", "external_prices",
         "sa_export_uvi_base2020; sa_export_uvi_base2024",
         "quarterly log growth calculated separately by base; newer series preferred in overlap",
         "Avoids directly joining incompatible index levels while retaining scale-independent export-price movements.",
         "recent")
register(registry, "import_uvi_growth_qoq", "external_prices",
         "sa_import_uvi_base2020; sa_import_uvi_base2024",
         "quarterly log growth calculated separately by base; newer series preferred in overlap",
         "Avoids directly joining incompatible index levels while retaining scale-independent import-price movements.",
         "recent")

# Difference between export- and import-price log growth approximates the change in their price ratio.
features["trade_price_ratio_change"] = features["export_uvi_growth_qoq"] - features["import_uvi_growth_qoq"]
register(registry, "trade_price_ratio_change", "external_prices",
         "export_uvi_growth_qoq; import_uvi_growth_qoq",
         "export UVI log growth minus import UVI log growth",
         "Approximates quarterly improvement/deterioration in the relative price of exports to imports.",
         "recent")

# ============================================================
# DOMESTIC DEMAND
# ============================================================
# Real expenditure levels are transformed to QoQ log growth to measure changes in real demand.
features["investment_growth_qoq"] = log_change(df["sa_real_investment"])
register(registry, "investment_growth_qoq", "domestic_demand", "sa_real_investment",
         "100 × quarterly log difference",
         "Measures quarterly growth in real gross fixed capital formation.",
         "core")

features["private_consumption_growth_qoq"] = log_change(df["sa_real_private_consumption"])
register(registry, "private_consumption_growth_qoq", "domestic_demand", "sa_real_private_consumption",
         "100 × quarterly log difference",
         "Measures quarterly growth in real household/private consumption demand.",
         "core")

features["government_consumption_growth_qoq"] = log_change(df["sa_real_government_consumption"])
register(registry, "government_consumption_growth_qoq", "domestic_demand", "sa_real_government_consumption",
         "100 × quarterly log difference",
         "Measures quarterly growth in real government consumption demand.",
         "core")

# ============================================================
# DOMESTIC / BUSINESS ACTIVITY
# ============================================================
# Retail and manufacturing variables are already published growth rates.
features["retail_sales_growth_yoy"] = df["sa_retail_sales_growth_yoy"]
features["manufacturing_growth_qoq"] = df["sa_manufacturing_growth_qoq"]
register(registry, "retail_sales_growth_yoy", "domestic_activity", "sa_retail_sales_growth_yoy",
         "none; already YoY growth",
         "Preserves the published real retail-sales growth rate without double transformation.",
         "core")
register(registry, "manufacturing_growth_qoq", "domestic_activity", "sa_manufacturing_growth_qoq",
         "none; already QoQ growth",
         "Preserves the published manufacturing growth rate without double transformation.",
         "core")

# Mining is supplied as an index, so it is converted to quarterly growth.
features["mining_growth_qoq"] = log_change(df["sa_mining_production_index"])
register(registry, "mining_growth_qoq", "domestic_activity", "sa_mining_production_index",
         "100 × quarterly log difference",
         "Converts the mining-production index into scale-independent quarterly growth.",
         "extended")

# Capacity utilisation and confidence are retained in level form because their index/rate levels
# directly represent the degree of spare capacity and prevailing business sentiment.
features["capacity_utilisation"] = df["sa_capacity_utilisation"]
features["business_confidence"] = df["sa_business_confidence"]
register(registry, "capacity_utilisation", "domestic_activity", "sa_capacity_utilisation",
         "quarterly level",
         "Retains the utilisation rate because its level directly measures production-capacity pressure.",
         "core")
register(registry, "business_confidence", "domestic_activity", "sa_business_confidence",
         "quarterly index level",
         "Retains the sentiment index because its level directly represents prevailing business expectations.",
         "core")

# Stats SA real-activity flows were summed to quarters in Script 04; QoQ log growth now measures
# changes in underlying activity without retaining scale differences across source series.
features["electricity_growth_qoq"] = log_change(df["sa_electricity_available_sa"])
register(registry, "electricity_growth_qoq", "domestic_activity", "sa_electricity_available_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures changes in electricity available for distribution as a broad real-activity proxy.",
         "core")

features["building_plans_growth_qoq"] = log_change(df["sa_building_plans_real_sa"])
register(registry, "building_plans_growth_qoq", "domestic_activity", "sa_building_plans_real_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures growth in real building plans passed, a potentially forward-looking construction indicator.",
         "core")

features["buildings_completed_growth_qoq"] = log_change(df["sa_buildings_completed_real_sa"])
register(registry, "buildings_completed_growth_qoq", "domestic_activity", "sa_buildings_completed_real_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures realised construction activity through completed buildings.",
         "extended")

features["wholesale_growth_qoq"] = log_change(df["sa_wholesale_trade_real_sa"])
register(registry, "wholesale_growth_qoq", "domestic_activity", "sa_wholesale_trade_real_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures growth in real wholesale trade as a broad goods-distribution activity indicator.",
         "extended")

# Liquidations are volatile non-negative counts; YoY log1p change reduces scale sensitivity and
# compares the quarter with the same quarter a year earlier.
features["liquidations_change_yoy"] = log1p_change(df["sa_total_liquidations"], periods=4)
register(registry, "liquidations_change_yoy", "business_conditions", "sa_total_liquidations",
         "100 × 4-quarter log1p difference",
         "Measures annual change in business distress while accommodating count data and potential zero values.",
         "extended")

# Shorter-history services/activity indicators are retained for expanded/recent experiments.
features["tourism_growth_qoq"] = log_change(df["sa_tourist_nights_sa"])
register(registry, "tourism_growth_qoq", "services", "sa_tourist_nights_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures growth in tourist stay-unit nights as a service-sector activity indicator.",
         "extended")

features["food_beverage_growth_qoq"] = log_change(df["sa_food_beverage_income_real_sa"])
register(registry, "food_beverage_growth_qoq", "services", "sa_food_beverage_income_real_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures growth in real food-and-beverage industry income.",
         "extended")

features["freight_growth_qoq"] = log_change(df["sa_freight_payload_sa"])
register(registry, "freight_growth_qoq", "domestic_activity", "sa_freight_payload_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures growth in physical freight movement as a proxy for goods-producing and distribution activity.",
         "extended")

features["motor_trade_growth_qoq"] = log_change(df["sa_motor_trade_real_sa"])
register(registry, "motor_trade_growth_qoq", "domestic_activity", "sa_motor_trade_real_sa",
         "100 × quarterly log difference of quarterly total",
         "Measures real motor-trade activity but is reserved for recent-sample experiments because history begins in 2017.",
         "recent")

# ============================================================
# LABOUR MARKET
# ============================================================
# Labour-market rates are differenced in percentage points because changes in rates are more
# interpretable than percentage growth, and this avoids dividing by the previous rate level.
features["unemployment_change_pp"] = pp_change(df["sa_unemployment_rate"])
register(registry, "unemployment_change_pp", "labour", "sa_unemployment_rate",
         "first difference in percentage points",
         "Measures deterioration or improvement in unemployment while preserving the published rate interpretation.",
         "extended")

features["labour_force_participation_change_pp"] = pp_change(df["sa_labour_force_participation"])
register(registry, "labour_force_participation_change_pp", "labour", "sa_labour_force_participation",
         "first difference in percentage points",
         "Measures changes in labour-market participation without percentage-changing an already percentage-based rate.",
         "extended")

# ============================================================
# GLOBAL CONDITIONS
# ============================================================
# US real GDP is converted from a level to QoQ log growth to represent external demand conditions.
features["us_gdp_growth_qoq"] = log_change(df["us_real_gdp"])
register(registry, "us_gdp_growth_qoq", "global", "us_real_gdp",
         "100 × quarterly log difference",
         "Measures quarterly US real GDP growth as an external/global activity indicator.",
         "core")

# ============================================================
# CLEAN NUMERICAL OUTPUT
# ============================================================
# Infinite values are never valid model inputs; replace them with NaN while preserving the gap
# for later feature-selection/imputation decisions.
numeric_cols = features.columns.drop("date")
features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)

# Remove rows outside the actual quarterly data universe but do not impute any feature.
features = features.sort_values("date").reset_index(drop=True)

# Save the full engineered table for traceability.
features.to_csv(FULL_OUT, index=False, date_format="%Y-%m-%d")

# The model target begins only after the first GDP log difference and ends at the final observed target.
target_valid = features["gdp_growth_qoq"].notna()
TARGET_START = features.loc[target_valid, "date"].min()
TARGET_END = features.loc[target_valid, "date"].max()
target_window = features[(features["date"] >= TARGET_START) & (features["date"] <= TARGET_END)].copy()
target_window.to_csv(TARGET_OUT, index=False, date_format="%Y-%m-%d")

# ============================================================
# FEATURE REGISTRY + AUDIT
# ============================================================
registry_df = pd.DataFrame(registry)
registry_df.to_csv(REGISTRY_OUT, index=False)

audit_rows = []
for feature in numeric_cols:
    valid = target_window[feature].notna()
    audit_rows.append({
        "feature_name": feature,
        "rows_in_target_window": len(target_window),
        "valid_values": int(valid.sum()),
        "missing_values": int((~valid).sum()),
        "coverage_pct": round(100 * valid.mean(), 2),
        "start_date": target_window.loc[valid, "date"].min().date() if valid.any() else None,
        "end_date": target_window.loc[valid, "date"].max().date() if valid.any() else None,
        "minimum": target_window[feature].min(skipna=True),
        "maximum": target_window[feature].max(skipna=True),
    })

audit_df = pd.DataFrame(audit_rows)
audit_df = audit_df.merge(
    registry_df[["feature_name", "feature_group", "suggested_tier"]],
    on="feature_name", how="left"
)
audit_df.to_csv(AUDIT_OUT, index=False)

# ============================================================
# FIGURE: ENGINEERED FEATURE COVERAGE
# ============================================================
plot_df = audit_df.copy()
plot_df["start_date"] = pd.to_datetime(plot_df["start_date"])
plot_df["end_date"] = pd.to_datetime(plot_df["end_date"])
plot_df = plot_df.dropna(subset=["start_date", "end_date"]).sort_values("start_date")

fig, ax = plt.subplots(figsize=(14, 15))
left = mdates.date2num(plot_df["start_date"])
right = mdates.date2num(plot_df["end_date"])
ax.barh(plot_df["feature_name"], right - left, left=left)
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.axvline(TARGET_START, linestyle="--", linewidth=1, label="Target start")
ax.axvline(TARGET_END, linestyle="--", linewidth=1, label="Target end")
ax.set_title("Engineered Feature Coverage Within GDP Forecasting Window")
ax.set_xlabel("Date")
ax.set_ylabel("Feature")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "engineered_feature_coverage.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("05 - FEATURE ENGINEERING COMPLETE")
print("=" * 75)
print(f"\nEngineered variables: {len(numeric_cols)}")
print(f"Target modelling window: {TARGET_START.date()} to {TARGET_END.date()}")
print(f"\nFull engineered dataset:\n{FULL_OUT}")
print(f"\nTarget-window dataset:\n{TARGET_OUT}")
print(f"\nFeature registry:\n{REGISTRY_OUT}")
print(f"\nFeature audit:\n{AUDIT_OUT}")
print(f"\nCPI extension validation:\n{CPI_VALIDATION_OUT}")
print(f"\n300-DPI figure:\n{FIGURE_DIR / 'engineered_feature_coverage.png'}")

if len(cpi_validation):
    cpi_corr = cpi_validation["historical_inflation_yoy"].corr(
        cpi_validation["stats_sa_inflation_yoy"]
    )
    cpi_mae = cpi_validation["difference"].abs().mean()
    cpi_max_diff = cpi_validation["difference"].abs().max()
    print("\nCPI overlap validation:")
    print(f"  Overlap quarters: {len(cpi_validation)}")
    print(f"  Correlation: {cpi_corr:.6f}")
    print(f"  Mean absolute difference: {cpi_mae:.6f} percentage points")
    print(f"  Maximum absolute difference: {cpi_max_diff:.6f} percentage points")

print("\nFeatures with less than 95% coverage inside the target window:\n")
low_coverage = audit_df[audit_df["coverage_pct"] < 95].sort_values("coverage_pct")
if len(low_coverage):
    print(low_coverage[
        ["feature_name", "suggested_tier", "coverage_pct", "missing_values", "start_date", "end_date"]
    ].to_string(index=False))
else:
    print("None.")
