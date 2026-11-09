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

RAW_DIR = ROOT / "data" / "raw" / "brazil"
EXTRACTED_DIR = ROOT / "data" / "interim" / "brazil_extracted"

OUT_DIR = ROOT / "data" / "processed" / "brazil_harmonised"
FIG_DIR = ROOT / "figures" / "brazil_harmonised"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

HARMONISED_OUT = OUT_DIR / "brazil_quarterly_harmonised.csv"
COVERAGE_OUT = OUT_DIR / "brazil_quarterly_coverage.csv"
VALIDATION_OUT = OUT_DIR / "brazil_source_validation.csv"
FX_AUDIT_OUT = OUT_DIR / "brazil_fx_source_audit.csv"
SUMMARY_OUT = OUT_DIR / "brazil_harmonisation_summary.txt"
CONFIG_OUT = OUT_DIR / "brazil_harmonisation_config.json"

FIGURE_DPI = 300

# Main coverage window used only for diagnostics.
# No model split is created in this script.
COVERAGE_START = pd.Timestamp("1996-01-01")
COVERAGE_END = pd.Timestamp("2025-10-01")

# ============================================================
# HELPERS
# ============================================================
def load_extracted(series_name):
    path = EXTRACTED_DIR / f"{series_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing extracted source-native series:\n{path}\n"
            "Run Script 14 first."
        )

    df = pd.read_csv(path)

    if not {"date", "value"}.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain date and value columns."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    return (
        df.dropna(
            subset=["date", "value"]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def qstart(dates):
    return (
        pd.to_datetime(dates)
        .dt.to_period("Q")
        .dt.start_time
    )


def quarterly_direct(df, name):
    out = df.copy()
    out["quarter"] = qstart(
        out["date"]
    )

    # Quarterly source should have one value per quarter.
    out = (
        out.groupby(
            "quarter",
            as_index=False,
        )["value"]
        .last()
        .rename(
            columns={
                "value": name,
            }
        )
    )

    return out


def quarterly_monthly_mean(
    df,
    name,
    require_three=True,
):
    out = df.copy()
    out["quarter"] = qstart(
        out["date"]
    )

    grouped = (
        out.groupby(
            "quarter"
        )["value"]
        .agg(
            ["mean", "count"]
        )
        .reset_index()
    )

    grouped[name] = grouped["mean"]

    if require_three:
        grouped.loc[
            grouped["count"] < 3,
            name,
        ] = np.nan

    return grouped[
        [
            "quarter",
            name,
        ]
    ]


def quarterly_monthly_sum(
    df,
    name,
    require_three=True,
):
    out = df.copy()
    out["quarter"] = qstart(
        out["date"]
    )

    grouped = (
        out.groupby(
            "quarter"
        )["value"]
        .agg(
            ["sum", "count"]
        )
        .reset_index()
    )

    grouped[name] = grouped["sum"]

    if require_three:
        grouped.loc[
            grouped["count"] < 3,
            name,
        ] = np.nan

    return grouped[
        [
            "quarter",
            name,
        ]
    ]


def quarterly_monthly_last(
    df,
    name,
    require_three=True,
):
    out = df.copy()
    out["quarter"] = qstart(
        out["date"]
    )

    grouped = (
        out.groupby(
            "quarter"
        )
        .agg(
            value=("value", "last"),
            count=("value", "count"),
        )
        .reset_index()
    )

    grouped[name] = grouped[
        "value"
    ]

    if require_three:
        grouped.loc[
            grouped["count"] < 3,
            name,
        ] = np.nan

    return grouped[
        [
            "quarter",
            name,
        ]
    ]


def quarterly_daily_mean(
    df,
    name,
    min_obs=40,
):
    out = df.copy()
    out["quarter"] = qstart(
        out["date"]
    )

    grouped = (
        out.groupby(
            "quarter"
        )["value"]
        .agg(
            ["mean", "count"]
        )
        .reset_index()
    )

    grouped[name] = grouped[
        "mean"
    ]

    grouped.loc[
        grouped["count"] < min_obs,
        name,
    ] = np.nan

    return grouped[
        [
            "quarter",
            name,
        ]
    ]


def coverage_row(
    series,
    name,
    window_index,
):
    s = (
        pd.Series(
            series.values,
            index=pd.to_datetime(
                series.index
            ),
        )
        .reindex(window_index)
    )

    valid = s.dropna()

    return {
        "series":
            name,
        "coverage_window_start":
            window_index.min(),
        "coverage_window_end":
            window_index.max(),
        "expected_quarters":
            len(window_index),
        "non_missing_quarters":
            int(
                s.notna().sum()
            ),
        "coverage_pct":
            float(
                100
                * s.notna().mean()
            ),
        "first_available_quarter":
            (
                valid.index.min()
                if len(valid)
                else pd.NaT
            ),
        "last_available_quarter":
            (
                valid.index.max()
                if len(valid)
                else pd.NaT
            ),
    }


def validation_row(
    check,
    result,
    metric,
    value,
    note,
):
    return {
        "validation_check":
            check,
        "result":
            result,
        "metric":
            metric,
        "value":
            value,
        "note":
            note,
    }

# ============================================================
# 1) CPI VALIDATION
# ============================================================
validation_rows = []

ipca = load_extracted(
    "br_headline_inflation_yoy"
)

cpi_index_path = (
    RAW_DIR
    / "Consumer Price Indices (CPIs, HICPs), COICOP 1999 Consumer Price Index Total for Brazil (BRACPIALLMINMEI).csv"
)

if not cpi_index_path.exists():
    raise FileNotFoundError(
        f"Missing CPI validation source:\n{cpi_index_path}"
    )

cpi_index = pd.read_csv(
    cpi_index_path
)

cpi_index["date"] = pd.to_datetime(
    cpi_index["observation_date"],
    errors="coerce",
)

cpi_index["index"] = pd.to_numeric(
    cpi_index["BRACPIALLMINMEI"],
    errors="coerce",
)

cpi_index = (
    cpi_index[
        ["date", "index"]
    ]
    .dropna()
    .sort_values("date")
)

cpi_index[
    "derived_yoy"
] = (
    100
    * (
        cpi_index["index"]
        / cpi_index["index"].shift(12)
        - 1
    )
)

cpi_overlap = (
    ipca.rename(
        columns={
            "value":
                "ibge_ipca_yoy",
        }
    )
    .merge(
        cpi_index[
            [
                "date",
                "derived_yoy",
            ]
        ],
        on="date",
        how="inner",
    )
    .dropna()
)

cpi_corr = float(
    cpi_overlap[
        "ibge_ipca_yoy"
    ].corr(
        cpi_overlap[
            "derived_yoy"
        ]
    )
)

cpi_mae = float(
    (
        cpi_overlap[
            "ibge_ipca_yoy"
        ]
        - cpi_overlap[
            "derived_yoy"
        ]
    )
    .abs()
    .mean()
)

cpi_max_diff = float(
    (
        cpi_overlap[
            "ibge_ipca_yoy"
        ]
        - cpi_overlap[
            "derived_yoy"
        ]
    )
    .abs()
    .max()
)

validation_rows.extend([
    validation_row(
        "CPI overlap",
        "PASS"
        if cpi_corr > 0.999
        else "REVIEW",
        "correlation",
        cpi_corr,
        "IBGE IPCA YoY versus 12-month inflation derived from the alternative monthly CPI index.",
    ),
    validation_row(
        "CPI overlap",
        "PASS"
        if cpi_mae < 0.05
        else "REVIEW",
        "MAE percentage points",
        cpi_mae,
        "Small differences may reflect rounding.",
    ),
    validation_row(
        "CPI overlap",
        "PASS"
        if cpi_max_diff < 0.10
        else "REVIEW",
        "maximum absolute difference percentage points",
        cpi_max_diff,
        "IBGE IPCA remains the selected inflation series.",
    ),
])

# ============================================================
# 2) M3 OVERLAP VALIDATION
# ============================================================
m3_selected = load_extracted(
    "br_m3_level"
).rename(
    columns={
        "value":
            "selected_m3",
    }
)

m3_validation_sources = [
    (
        "MYAGM3BRM189N",
        RAW_DIR
        / "M3 for Brazil (MYAGM3BRM189N).csv",
        "MYAGM3BRM189N",
    ),
    (
        "MABMM301BRM189N",
        RAW_DIR
        / "Monetary Aggregates and Their Components Broad Money and Components M3 for Brazil (MABMM301BRM189N).csv",
        "MABMM301BRM189N",
    ),
]

m3_recent_pass = False

for source_label, path, value_col in (
    m3_validation_sources
):
    if not path.exists():
        validation_rows.append(
            validation_row(
                f"M3 overlap {source_label}",
                "MISSING",
                "source_available",
                0,
                str(path),
            )
        )
        continue

    other = pd.read_csv(path)

    other["date"] = pd.to_datetime(
        other["observation_date"],
        errors="coerce",
    )

    other[
        "comparison_m3"
    ] = pd.to_numeric(
        other[value_col],
        errors="coerce",
    )

    other = (
        other[
            [
                "date",
                "comparison_m3",
            ]
        ]
        .dropna()
        .sort_values("date")
    )

    overlap = (
        m3_selected.merge(
            other,
            on="date",
            how="inner",
        )
        .dropna()
    )

    if len(overlap) < 12:
        validation_rows.append(
            validation_row(
                f"M3 overlap {source_label}",
                "REVIEW",
                "overlap observations",
                len(overlap),
                "Insufficient overlap for strong source validation.",
            )
        )
        continue

    level_corr = float(
        overlap[
            "selected_m3"
        ].corr(
            overlap[
                "comparison_m3"
            ]
        )
    )

    ratio = (
        overlap[
            "selected_m3"
        ]
        / overlap[
            "comparison_m3"
        ]
    )

    median_ratio = float(
        ratio.median()
    )

    selected_growth = (
        overlap[
            "selected_m3"
        ]
        .pct_change()
    )

    comparison_growth = (
        overlap[
            "comparison_m3"
        ]
        .pct_change()
    )

    valid_growth = (
        selected_growth.notna()
        & comparison_growth.notna()
    )

    growth_corr = float(
        selected_growth[
            valid_growth
        ].corr(
            comparison_growth[
                valid_growth
            ]
        )
    )

    growth_mae = float(
        (
            selected_growth[
                valid_growth
            ]
            - comparison_growth[
                valid_growth
            ]
        )
        .abs()
        .mean()
    )

    is_recent_reference = (
        source_label
        == "MABMM301BRM189N"
    )

    if is_recent_reference:
        m3_recent_pass = (
            level_corr > 0.99
            and 0.95
            <= median_ratio
            <= 1.05
            and growth_mae < 0.01
        )

    validation_rows.extend([
        validation_row(
            f"M3 overlap {source_label}",
            "PASS"
            if level_corr > 0.99
            else "REVIEW",
            "level correlation",
            level_corr,
            f"{len(overlap)} overlapping months.",
        ),
        validation_row(
            f"M3 overlap {source_label}",
            "PASS"
            if 0.95
            <= median_ratio
            <= 1.05
            else "REVIEW",
            "median level ratio selected/comparison",
            median_ratio,
            "A value near 1 indicates common scale over most of the overlap.",
        ),
        validation_row(
            f"M3 overlap {source_label}",
            "PASS"
            if growth_mae < 0.01
            else "REVIEW",
            "monthly growth MAE",
            growth_mae,
            f"Growth correlation={growth_corr:.6f}.",
        ),
    ])

validation_rows.append(
    validation_row(
        "M3 selected-source decision",
        "PASS"
        if m3_recent_pass
        else "REVIEW",
        "recent overlap accepted",
        int(
            m3_recent_pass
        ),
        "The recent MABMM301BRM189N overlap is the primary validation because it covers the modern modelling era.",
    )
)

# ============================================================
# 3) FX COVERAGE AUDIT + FALLBACK
# ============================================================
fx_daily = load_extracted(
    "br_usd_brl_daily"
)

fx_daily_start = (
    fx_daily["date"].min()
)

fx_daily_end = (
    fx_daily["date"].max()
)

fx_daily_obs = len(
    fx_daily
)

# For a 1996+ core model we require a genuinely long daily history.
fx_daily_sufficient_for_core = bool(
    fx_daily_start
    <= pd.Timestamp(
        "1996-01-01"
    )
    and fx_daily_obs >= 5000
)

fx_monthly_path = (
    RAW_DIR
    / "USD_BRL Historical Data.csv"
)

if not fx_monthly_path.exists():
    raise FileNotFoundError(
        f"Missing monthly FX fallback:\n{fx_monthly_path}"
    )

fx_monthly_raw = pd.read_csv(
    fx_monthly_path
)

fx_monthly = pd.DataFrame({
    "date":
        pd.to_datetime(
            fx_monthly_raw[
                "Date"
            ],
            errors="coerce",
        ),
    "value":
        pd.to_numeric(
            fx_monthly_raw[
                "Price"
            ],
            errors="coerce",
        ),
})

fx_monthly = (
    fx_monthly.dropna()
    .sort_values("date")
    .drop_duplicates(
        subset=["date"],
        keep="last",
    )
    .reset_index(drop=True)
)

if fx_daily_sufficient_for_core:
    fx_core_source = (
        "daily DEXBZUS"
    )

    fx_quarterly = (
        quarterly_daily_mean(
            fx_daily,
            "br_usd_brl_level",
            min_obs=40,
        )
    )
else:
    fx_core_source = (
        "monthly USD_BRL Historical Data fallback"
    )

    fx_quarterly = (
        quarterly_monthly_mean(
            fx_monthly,
            "br_usd_brl_level",
            require_three=True,
        )
    )

validation_rows.extend([
    validation_row(
        "FX daily-history coverage",
        "PASS"
        if fx_daily_sufficient_for_core
        else "FALLBACK",
        "daily observations",
        fx_daily_obs,
        f"Extracted daily series covers {fx_daily_start.date()} to {fx_daily_end.date()}.",
    ),
    validation_row(
        "FX core-source selection",
        "PASS",
        "core source",
        fx_core_source,
        "Monthly FX is acceptable for the long-history exchange-rate change feature when daily history is insufficient.",
    ),
])

fx_audit_df = pd.DataFrame([
    {
        "daily_source_start":
            fx_daily_start,
        "daily_source_end":
            fx_daily_end,
        "daily_observations":
            fx_daily_obs,
        "daily_sufficient_for_1996_core":
            fx_daily_sufficient_for_core,
        "monthly_fallback_start":
            fx_monthly["date"].min(),
        "monthly_fallback_end":
            fx_monthly["date"].max(),
        "monthly_observations":
            len(fx_monthly),
        "selected_core_fx_source":
            fx_core_source,
        "fx_volatility_status":
            (
                "main-model feasible"
                if fx_daily_sufficient_for_core
                else "secondary/recent only unless long daily DEXBZUS is restored"
            ),
    }
])

fx_audit_df.to_csv(
    FX_AUDIT_OUT,
    index=False,
)

# ============================================================
# 4) QUARTERLY HARMONISATION
# ============================================================
quarterly_parts = []

# Direct quarterly series.
direct_quarterly = {
    "br_real_gdp":
        "br_real_gdp",
    "br_private_credit_level_usd":
        "br_private_credit_level_usd",
    "br_real_imports":
        "br_real_imports",
    "br_real_investment":
        "br_real_investment",
    "br_real_private_consumption":
        "br_real_private_consumption",
    "br_real_government_consumption":
        "br_real_government_consumption",
    "br_manufacturing_growth":
        "br_manufacturing_growth",
    "br_retail_sales_growth":
        "br_retail_sales_growth",
    "br_capacity_utilisation":
        "br_capacity_utilisation",
    "br_credit_gap":
        "br_credit_gap",
}

for extracted_name, output_name in (
    direct_quarterly.items()
):
    quarterly_parts.append(
        quarterly_direct(
            load_extracted(
                extracted_name
            ),
            output_name,
        )
    )

# Monthly averages.
monthly_mean_series = {
    "br_headline_inflation_yoy":
        "br_headline_inflation_yoy",
    "br_treasury_bill_yield":
        "br_treasury_bill_yield",
    "br_reer":
        "br_reer",
    "br_share_price_index":
        "br_share_price_index",
    "br_business_confidence":
        "br_business_confidence",
    "br_long_term_10y_yield":
        "br_long_term_10y_yield",
}

for extracted_name, output_name in (
    monthly_mean_series.items()
):
    quarterly_parts.append(
        quarterly_monthly_mean(
            load_extracted(
                extracted_name
            ),
            output_name,
            require_three=True,
        )
    )

# Monthly stock variables -> quarter end.
quarterly_parts.append(
    quarterly_monthly_last(
        load_extracted(
            "br_m3_level"
        ),
        "br_m3_level",
        require_three=True,
    )
)

quarterly_parts.append(
    quarterly_monthly_last(
        load_extracted(
            "br_reserves_ex_gold"
        ),
        "br_reserves_ex_gold",
        require_three=True,
    )
)

# Monthly flow variable > quarterly sum.
quarterly_parts.append(
    quarterly_monthly_sum(
        load_extracted(
            "br_nominal_merchandise_exports"
        ),
        "br_nominal_merchandise_exports",
        require_three=True,
    )
)

# Electricity is a monthly flow, secondary only.
quarterly_parts.append(
    quarterly_monthly_sum(
        load_extracted(
            "br_electricity_generation"
        ),
        "br_electricity_generation",
        require_three=True,
    )
)

# Selected long-history FX level.
quarterly_parts.append(
    fx_quarterly
)

# ============================================================
# SELIC: RECONSTRUCT EFFECTIVE DAILY TARGET THEN QUARTERLY AVERAGE
# ============================================================
selic = load_extracted(
    "br_selic_target"
)

# BACEN timestamps were extracted with a 03:00 time component.
# Normalize them to midnight BEFORE reindexing to a daily calendar.
# Without this, 03:00 policy dates do not match the 00:00 daily index,
# causing the entire quarterly SELIC series to become NaN.
selic["date"] = selic["date"].dt.normalize()

selic = (
    selic.sort_values("date")
    .drop_duplicates(
        subset=["date"],
        keep="last",
    )
)

daily_index = pd.date_range(
    selic["date"].min(),
    selic["date"].max(),
    freq="D",
)

selic_daily = (
    selic.set_index(
        "date"
    )["value"]
    .reindex(
        daily_index
    )
    .ffill()
)

selic_daily_df = pd.DataFrame({
    "date":
        selic_daily.index,
    "value":
        selic_daily.values,
})

selic_quarterly = (
    quarterly_daily_mean(
        selic_daily_df,
        "br_selic_target",
        min_obs=80,
    )
)

quarterly_parts.append(
    selic_quarterly
)

# ============================================================
# MERGE TO COMMON QUARTER INDEX
# ============================================================
all_quarters = pd.period_range(
    start=min(
        p["quarter"].min()
        for p in quarterly_parts
        if len(p)
    ),
    end=max(
        p["quarter"].max()
        for p in quarterly_parts
        if len(p)
    ),
    freq="Q",
).start_time

harmonised = pd.DataFrame({
    "date":
        all_quarters,
})

for part in quarterly_parts:
    harmonised = harmonised.merge(
        part.rename(
            columns={
                "quarter":
                    "date",
            }
        ),
        on="date",
        how="left",
    )

harmonised = (
    harmonised.sort_values("date")
    .reset_index(drop=True)
)

harmonised.to_csv(
    HARMONISED_OUT,
    index=False,
)

# ============================================================
# 5) COVERAGE DIAGNOSTICS FOR 1996Q1-2025Q4
# ============================================================
coverage_index = (
    pd.period_range(
        start=COVERAGE_START,
        end=COVERAGE_END,
        freq="Q",
    )
    .start_time
)

coverage_rows = []

for col in harmonised.columns:
    if col == "date":
        continue

    s = (
        harmonised.set_index(
            "date"
        )[col]
    )

    coverage_rows.append(
        coverage_row(
            s,
            col,
            coverage_index,
        )
    )

coverage_df = (
    pd.DataFrame(
        coverage_rows
    )
    .sort_values(
        [
            "coverage_pct",
            "series",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

coverage_df.to_csv(
    COVERAGE_OUT,
    index=False,
)

# ============================================================
# 6) VALIDATION OUTPUT
# ============================================================
validation_df = pd.DataFrame(
    validation_rows
)

validation_df.to_csv(
    VALIDATION_OUT,
    index=False,
)

# ============================================================
# 7) COVERAGE FIGURE
# ============================================================
plot_df = coverage_df.copy()

fig, ax = plt.subplots(
    figsize=(
        11,
        max(
            7,
            0.38
            * len(plot_df),
        ),
    )
)

ax.barh(
    plot_df[
        "series"
    ],
    plot_df[
        "coverage_pct"
    ],
)

ax.set_title(
    "Brazil Quarterly Feature Coverage, 1996Q1-2025Q4"
)

ax.set_xlabel(
    "Coverage (%)"
)

ax.set_ylabel(
    "Harmonised series"
)

ax.set_xlim(
    0,
    105,
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "brazil_quarterly_feature_coverage.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# 8) CONFIG + SUMMARY
# ============================================================
config = {
    "country":
        "Brazil",
    "stage":
        "source validation and quarterly harmonisation",
    "model_used":
        False,
    "raw_files_modified":
        False,
    "coverage_diagnostic_window":
        "1996Q1-2025Q4",
    "aggregation_rules": {
        "quarterly sources":
            "retain source quarter directly",
        "monthly rates and indices":
            "quarterly mean, requiring all 3 months",
        "monthly flows":
            "quarterly sum, requiring all 3 months",
        "monthly stock variables":
            "last monthly value in quarter, requiring all 3 months",
        "SELIC irregular target":
            "calendar-day forward-filled effective policy target, then quarterly average",
        "daily FX":
            "quarterly daily mean if long history is sufficient",
        "monthly FX fallback":
            "quarterly mean if long daily history is unavailable",
    },
    "important_notes": [
        "No annual series were interpolated.",
        "Private credit remains a level series, not credit-to-GDP.",
        "Nominal merchandise exports are not treated as real total exports.",
        "FX volatility is not forced into the core set when daily history is too short.",
        "10Y yield, business confidence, credit gap and electricity remain secondary until sample feasibility is assessed.",
    ],
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
    "BRAZIL SOURCE VALIDATION + QUARTERLY HARMONISATION",
    "=" * 78,
    "",
    f"CPI overlap observations: {len(cpi_overlap)}",
    f"CPI correlation: {cpi_corr:.12f}",
    f"CPI MAE difference: {cpi_mae:.6f} percentage points",
    f"CPI maximum difference: {cpi_max_diff:.6f} percentage points",
    "",
    f"M3 recent-overlap validation passed: {m3_recent_pass}",
    "",
    f"Extracted daily FX: {fx_daily_start.date()} -> {fx_daily_end.date()} ({fx_daily_obs:,} obs)",
    f"Daily FX sufficient for 1996+ core: {fx_daily_sufficient_for_core}",
    f"Selected core FX source: {fx_core_source}",
    "",
    "Quarterly coverage, 1996Q1-2025Q4:",
]

for _, row in (
    coverage_df.iterrows()
):
    summary_lines.append(
        f"  {row['series']}: "
        f"{row['non_missing_quarters']}/{row['expected_quarters']} "
        f"({row['coverage_pct']:.2f}%)"
    )

summary_lines.extend([
    "",
    "Interpretation:",
    "  - This script validates sources and harmonises frequency only.",
    "  - No forecast target, training split or Transformer model is used here.",
    "  - Feature transformations such as GDP growth, M3 growth, FX change,",
    "    share-price return and REER change belong in the next feature-engineering stage.",
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
print("15 - BRAZIL VALIDATION + QUARTERLY HARMONISATION")
print("=" * 78)

print("\nCPI validation:")
print(
    f"  overlap observations: {len(cpi_overlap)}"
)
print(
    f"  correlation:          {cpi_corr:.12f}"
)
print(
    f"  MAE difference:       {cpi_mae:.6f} pp"
)
print(
    f"  max difference:       {cpi_max_diff:.6f} pp"
)

print("\nM3 validation:")
m3_print = validation_df[
    validation_df[
        "validation_check"
    ].str.contains(
        "M3",
        na=False,
    )
]
print(
    m3_print[
        [
            "validation_check",
            "result",
            "metric",
            "value",
        ]
    ].to_string(
        index=False
    )
)

print("\nFX audit:")
print(
    fx_audit_df.to_string(
        index=False
    )
)

print("\nQuarterly coverage, 1996Q1-2025Q4:")
print(
    coverage_df[
        [
            "series",
            "non_missing_quarters",
            "expected_quarters",
            "coverage_pct",
            "first_available_quarter",
            "last_available_quarter",
        ]
    ].to_string(
        index=False
    )
)

print("\nOutputs:")
print(f"  {HARMONISED_OUT}")
print(f"  {COVERAGE_OUT}")
print(f"  {VALIDATION_OUT}")
print(f"  {FX_AUDIT_OUT}")
print(f"  {SUMMARY_OUT}")
print(f"  {CONFIG_OUT}")

print("\n300-DPI figure:")
print(
    f"  {FIG_DIR / 'brazil_quarterly_feature_coverage.png'}"
)


if __name__ == "__main__":
    pass
