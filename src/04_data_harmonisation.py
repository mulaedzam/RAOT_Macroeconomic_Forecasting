from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = PROJECT_ROOT / "data" / "interim" / "series"
AUDIT_PATH = PROJECT_ROOT / "data" / "interim" / "extraction_audit.csv"
QUARTERLY_DIR = PROJECT_ROOT / "data" / "interim" / "quarterly"
HARMONISATION_DIR = PROJECT_ROOT / "data" / "processed" / "harmonisation"
FIGURE_DIR = PROJECT_ROOT / "figures" / "data_harmonisation"

QUARTERLY_DIR.mkdir(parents=True, exist_ok=True)
HARMONISATION_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

HARMONISATION_AUDIT = HARMONISATION_DIR / "harmonisation_audit.csv"
QUARTERLY_WIDE = PROJECT_ROOT / "data" / "interim" / "quarterly_harmonised_wide.csv"
FIGURE_DPI = 300

# ============================================================
# QUARTERLY HARMONISATION RULES
# ============================================================
# This script changes frequency only. It does NOT calculate growth rates,
# differences, spreads, volatility, scaling or missing-value imputations.
#
# Rule meanings:
#   keep       = source is already quarterly; preserve the observation.
#   mean       = average monthly/daily values within each quarter.
#   sum        = add monthly flow/count observations within each quarter.
#   last_valid = use the latest available monthly stock value in the quarter.
#
# min_obs prevents incomplete quarters from being treated as full-quarter data.
RULES = {
    # --- Already-quarterly series: no temporal aggregation is required. ---
    "sa_real_gdp": {"rule": "keep", "min_obs": 1},
    "sa_core_inflation_qoq": {"rule": "keep", "min_obs": 1},
    "sa_core_inflation_yoy": {"rule": "keep", "min_obs": 1},
    "sa_m3_growth_qoq": {"rule": "keep", "min_obs": 1},
    "sa_private_credit_gdp": {"rule": "keep", "min_obs": 1},
    "sa_real_house_price_index": {"rule": "keep", "min_obs": 1},
    "sa_household_debt_gdp": {"rule": "keep", "min_obs": 1},
    "sa_current_account_balance": {"rule": "keep", "min_obs": 1},
    "sa_real_exports_goods": {"rule": "keep", "min_obs": 1},
    "sa_real_imports_goods_services": {"rule": "keep", "min_obs": 1},
    "sa_real_investment": {"rule": "keep", "min_obs": 1},
    "sa_real_private_consumption": {"rule": "keep", "min_obs": 1},
    "sa_real_government_consumption": {"rule": "keep", "min_obs": 1},
    "sa_manufacturing_growth_qoq": {"rule": "keep", "min_obs": 1},
    "sa_mining_production_index": {"rule": "keep", "min_obs": 1},
    "sa_capacity_utilisation": {"rule": "keep", "min_obs": 1},
    "sa_business_confidence": {"rule": "keep", "min_obs": 1},
    "sa_unemployment_rate": {"rule": "keep", "min_obs": 1},
    "sa_labour_force_participation": {"rule": "keep", "min_obs": 1},
    "us_real_gdp": {"rule": "keep", "min_obs": 1},

    # --- Monthly rates/growth rates: quarterly mean represents average conditions. ---
    "sa_headline_inflation_yoy": {"rule": "mean", "min_obs": 3},
    "sa_interbank_3m_rate": {"rule": "mean", "min_obs": 3},
    "sa_bond_yield_10y": {"rule": "mean", "min_obs": 3},
    "sa_repo_rate": {"rule": "mean", "min_obs": 3},
    "sa_retail_sales_growth_yoy": {"rule": "mean", "min_obs": 3},

    # --- Monthly indices: quarterly mean reduces month-specific noise. ---
    "sa_share_price_index": {"rule": "mean", "min_obs": 3},
    "sa_reer": {"rule": "mean", "min_obs": 3},
    "sa_ppi_final_manufactured_goods": {"rule": "mean", "min_obs": 3},
    "sa_export_uvi_base2020": {"rule": "mean", "min_obs": 3},
    "sa_import_uvi_base2020": {"rule": "mean", "min_obs": 3},
    "sa_export_uvi_base2024": {"rule": "mean", "min_obs": 3},
    "sa_import_uvi_base2024": {"rule": "mean", "min_obs": 3},

    # --- Monthly flows/counts: quarterly totals preserve their flow interpretation. ---
    "sa_electricity_available_sa": {"rule": "sum", "min_obs": 3},
    "sa_building_plans_real_sa": {"rule": "sum", "min_obs": 3},
    "sa_buildings_completed_real_sa": {"rule": "sum", "min_obs": 3},
    "sa_wholesale_trade_real_sa": {"rule": "sum", "min_obs": 3},
    "sa_total_liquidations": {"rule": "sum", "min_obs": 3},
    "sa_tourist_nights_sa": {"rule": "sum", "min_obs": 3},
    "sa_food_beverage_income_real_sa": {"rule": "sum", "min_obs": 3},
    "sa_freight_payload_sa": {"rule": "sum", "min_obs": 3},
    "sa_motor_trade_real_sa": {"rule": "sum", "min_obs": 3},

    # Reserves are a stock, so the latest available value best represents quarter-end.
    "sa_reserves_ex_gold": {"rule": "last_valid", "min_obs": 1},

    # Daily USD/ZAR is averaged using valid trading-day observations only.
    # A 40-day minimum avoids treating weekends/public holidays as missing quarters.
    "sa_usd_zar": {"rule": "mean", "min_obs": 40},
}

# ============================================================
# HELPERS
# ============================================================
def load_series(feature):
    """Load one standardised series produced by Script 02."""
    path = SERIES_DIR / f"{feature}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing interim series: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns or feature not in df.columns:
        raise ValueError(f"Unexpected structure in {path.name}: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    return df[df["date"].notna()].sort_values("date").reset_index(drop=True)


def last_valid(values):
    """Return the final non-missing observation in a quarter without interpolation."""
    valid = values.dropna()
    return valid.iloc[-1] if len(valid) else np.nan


def harmonise_series(feature, rule, min_obs):
    """Convert one source series to quarter-start dates while preserving missingness."""
    df = load_series(feature)
    df["quarter"] = df["date"].dt.to_period("Q")

    # Existing quarterly observations are mapped to their quarter and left unchanged.
    if rule == "keep":
        grouped = df.groupby("quarter")[feature].agg(["size", "count", "first"]).reset_index()
        if (grouped["size"] > 1).any():
            bad = grouped.loc[grouped["size"] > 1, "quarter"].astype(str).tolist()
            raise ValueError(f"{feature} has multiple observations in quarterly periods: {bad}")
        grouped["value"] = grouped["first"]
    else:
        # Count supplied and valid observations before aggregation for auditability.
        grouped = df.groupby("quarter")[feature].agg(["size", "count"]).reset_index()
        if rule == "mean":
            values = df.groupby("quarter")[feature].mean()
        elif rule == "sum":
            values = df.groupby("quarter")[feature].sum(min_count=min_obs)
        elif rule == "last_valid":
            values = df.groupby("quarter")[feature].apply(last_valid)
        else:
            raise ValueError(f"Unknown harmonisation rule for {feature}: {rule}")
        grouped["value"] = grouped["quarter"].map(values)

        # Mean/last-valid outputs are suppressed when a quarter lacks enough valid data.
        grouped.loc[grouped["count"] < min_obs, "value"] = np.nan

    grouped["date"] = grouped["quarter"].dt.to_timestamp(how="start")
    grouped["meets_minimum"] = grouped["count"] >= min_obs
    out = grouped[["date", "value"]].rename(columns={"value": feature})

    # Keep the full observed quarterly span, including missing quarters, for transparency.
    full_q = pd.period_range(grouped["quarter"].min(), grouped["quarter"].max(), freq="Q")
    full = pd.DataFrame({"quarter": full_q})
    full["date"] = full["quarter"].dt.to_timestamp(how="start")
    out = full[["date"]].merge(out, on="date", how="left")

    details = grouped[["quarter", "size", "count", "meets_minimum"]].copy()
    details["feature_name"] = feature
    return out, details


# ============================================================
# MAIN HARMONISATION
# ============================================================
def main():
    audit = pd.read_csv(AUDIT_PATH)
    extracted_features = set(audit["feature_name"])
    configured_features = set(RULES)

    # Fail early if a new extracted series has no explicit economic aggregation rule.
    missing_rules = sorted(extracted_features - configured_features)
    extra_rules = sorted(configured_features - extracted_features)
    if missing_rules:
        raise ValueError(f"Extracted features without harmonisation rules: {missing_rules}")
    if extra_rules:
        raise ValueError(f"Harmonisation rules refer to missing extracted features: {extra_rules}")

    print("\n" + "=" * 75)
    print("04 - DATA HARMONISATION")
    print("=" * 75)
    print(f"\nSeries to harmonise: {len(audit)}\n")

    harmonised = {}
    audit_rows = []

    for i, feature in enumerate(audit["feature_name"], start=1):
        spec = RULES[feature]
        print(f"[{i:02d}/{len(audit):02d}] {feature:<40} rule={spec['rule']}")

        qdf, details = harmonise_series(feature, spec["rule"], spec["min_obs"])
        harmonised[feature] = qdf

        # Save each quarterly series separately so later feature engineering remains traceable.
        qdf.to_csv(QUARTERLY_DIR / f"{feature}.csv", index=False, date_format="%Y-%m-%d")

        valid = qdf[feature].notna()
        audit_rows.append({
            "feature_name": feature,
            "source_frequency": audit.loc[audit["feature_name"] == feature, "frequency"].iloc[0],
            "quarterly_rule": spec["rule"],
            "minimum_valid_observations": spec["min_obs"],
            "quarterly_rows": len(qdf),
            "valid_quarters": int(valid.sum()),
            "missing_quarters": int((~valid).sum()),
            "start_date": qdf.loc[valid, "date"].min().date() if valid.any() else None,
            "end_date": qdf.loc[valid, "date"].max().date() if valid.any() else None,
            "quarters_below_minimum": int((~details["meets_minimum"]).sum()),
        })

    # Outer-join all quarterly series; this is a harmonised data store, not a model dataset.
    wide = None
    for feature, qdf in harmonised.items():
        wide = qdf if wide is None else wide.merge(qdf, on="date", how="outer")
    wide = wide.sort_values("date").reset_index(drop=True)
    wide.to_csv(QUARTERLY_WIDE, index=False, date_format="%Y-%m-%d")

    harmonisation_audit = pd.DataFrame(audit_rows).sort_values("feature_name")
    harmonisation_audit.to_csv(HARMONISATION_AUDIT, index=False)

    # ========================================================
    # 300-DPI FIGURE: QUARTERLY COVERAGE AFTER HARMONISATION
    # ========================================================
    # Plot from 1990 onward so very old repo/reserve observations do not compress
    # the modelling-era coverage that matters for the South African experiment.
    plot_df = harmonisation_audit.copy()
    plot_df["start_date"] = pd.to_datetime(plot_df["start_date"])
    plot_df["end_date"] = pd.to_datetime(plot_df["end_date"])
    plot_df["plot_start"] = plot_df["start_date"].clip(lower=pd.Timestamp("1990-01-01"))
    plot_df = plot_df.sort_values("plot_start")

    fig, ax = plt.subplots(figsize=(14, 14))
    left = mdates.date2num(plot_df["plot_start"])
    right = mdates.date2num(plot_df["end_date"])
    ax.barh(plot_df["feature_name"], right - left, left=left)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Quarterly Feature Coverage After Harmonisation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "quarterly_harmonised_coverage.png",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================
    flagged = harmonisation_audit[harmonisation_audit["missing_quarters"] > 0]

    print("\n" + "=" * 75)
    print("DATA HARMONISATION COMPLETE")
    print("=" * 75)
    print(f"\nHarmonised series: {len(harmonisation_audit)}")
    print(f"Wide quarterly dataset:\n{QUARTERLY_WIDE}")
    print(f"\nHarmonisation audit:\n{HARMONISATION_AUDIT}")
    print(f"\nIndividual quarterly series:\n{QUARTERLY_DIR}")
    print(f"\n300-DPI figure:\n{FIGURE_DIR / 'quarterly_harmonised_coverage.png'}")

    print("\nSeries containing missing quarters after harmonisation:\n")
    if len(flagged):
        cols = ["feature_name", "quarterly_rule", "valid_quarters",
                "missing_quarters", "quarters_below_minimum", "start_date", "end_date"]
        print(flagged[cols].to_string(index=False))
    else:
        print("None.")


if __name__ == "__main__":
    main()
