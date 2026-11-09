from pathlib import Path
import io
import json
import re

import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "brazil"

INVENTORY_DIR = ROOT / "data" / "processed" / "brazil_inventory"
INTERIM_DIR = ROOT / "data" / "interim" / "brazil_extracted"
OUT_DIR = ROOT / "data" / "processed" / "brazil_source_selection"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MANIFEST_OUT = OUT_DIR / "brazil_selected_source_manifest.csv"
EXCLUSION_OUT = OUT_DIR / "brazil_excluded_or_secondary_sources.csv"
EXTRACTED_LONG_OUT = INTERIM_DIR / "brazil_extracted_long.csv"
EXTRACTION_AUDIT_OUT = OUT_DIR / "brazil_extraction_audit.csv"
SUMMARY_OUT = OUT_DIR / "brazil_source_selection_summary.txt"
CONFIG_OUT = OUT_DIR / "brazil_source_selection_config.json"

# ============================================================
# SOURCE SELECTION PRINCIPLES
# ============================================================
# 1) Prefer long monthly/quarterly official/institutional series.
# 2) Preserve the potential ~1996-2025 modelling window where possible.
# 3) Do NOT use annual data by interpolating it to quarters.
# 4) Do NOT force short 2007+ / 2008+ / 2010+ series into the core set yet.
# 5) Extraction only: no modelling transformations happen in this script.
# 6) Raw files are never modified.

# ============================================================
# CANONICAL SOURCE MAP
# ============================================================

SOURCE_SPECS = [
    {
        "series_name": "br_real_gdp",
        "file_name": "Real Gross Domestic Product (GDP) for Brazil (NGDPRSAXDCBRQ).csv",
        "date_col": "observation_date",
        "value_col": "NGDPRSAXDCBRQ",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "real GDP level; seasonally adjusted constant-price series",
        "later_transformation": "quarter-on-quarter percentage growth; also lagged GDP input",
    },
    {
        "series_name": "br_headline_inflation_yoy",
        "file_name": "IBGE_IPCA.csv",
        "date_col": "Date",
        "value_col": "IPCA Rate (last 12 months)",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "IBGE",
        "unit_note": "IPCA inflation rate over previous 12 months, percent",
        "later_transformation": "quarterly average of monthly YoY inflation rates",
    },
    {
        "series_name": "br_selic_target",
        "file_name": "BACEN_SELIC.csv",
        "date_col": "DataInicioVigencia",
        "value_col": "MetaSelic",
        "native_frequency": "policy_decision_irregular",
        "role": "core_candidate",
        "source_family": "Banco Central do Brasil",
        "unit_note": "SELIC target, percent per annum",
        "later_transformation": "construct quarter-level policy rate from effective policy regime; then change in percentage points",
    },
    {
        "series_name": "br_treasury_bill_yield",
        "file_name": "Government Securities Treasury Bills Yields, Rate, Percent per Annum for Brazil (INTGSTBRM193N).csv",
        "date_col": "observation_date",
        "value_col": "INTGSTBRM193N",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "short-term government security yield, percent per annum",
        "later_transformation": "quarterly average; candidate short-rate feature",
    },
    {
        "series_name": "br_m3_level",
        "file_name": "BR Money Supply M3 - Money Supply M3 - Eulerpool - 2026-09-09.csv",
        "date_col": "Date",
        "value_col": "Money Supply M3",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "Eulerpool",
        "unit_note": "M3 level; long-history source selected for coverage",
        "later_transformation": "validate against overlapping FRED M3 series, then compute growth",
    },
    {
        "series_name": "br_private_credit_level_usd",
        "file_name": "Total Credit to Private Non-Financial Sector, Adjusted for Breaks, for Brazil (QBRPAMUSDA).csv",
        "date_col": "observation_date",
        "value_col": "QBRPAMUSDA",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "BIS/FRED",
        "unit_note": "total credit to private non-financial sector; adjusted for breaks; level series",
        "later_transformation": "growth/change candidate; DO NOT label as credit-to-GDP",
    },
    {
        "series_name": "br_usd_brl_daily",
        "file_name": "Brazilian Reals to U.S. Dollar Spot Exchange Rate (DEXBZUS)(1).csv",
        "date_col": "observation_date",
        "value_col": "DEXBZUS",
        "native_frequency": "daily",
        "role": "core_candidate",
        "source_family": "FRED",
        "unit_note": "Brazilian reais per U.S. dollar",
        "later_transformation": "quarter-end/quarterly average FX change and daily-log-return volatility",
    },
    {
        "series_name": "br_reer",
        "file_name": "Real Broad Effective Exchange Rate for Brazil (RBBRBIS).csv",
        "date_col": "observation_date",
        "value_col": "RBBRBIS",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "BIS/FRED",
        "unit_note": "real broad effective exchange-rate index",
        "later_transformation": "quarterly average then QoQ percentage change",
    },
    {
        "series_name": "br_share_price_index",
        "file_name": "Financial Market Share Prices for Brazil (SPASTT01BRM661N).csv",
        "date_col": "observation_date",
        "value_col": "SPASTT01BRM661N",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "financial market share-price index",
        "later_transformation": "quarterly average/end value then return",
    },
    {
        "series_name": "br_real_imports",
        "file_name": "Real Imports of Goods and Services for Brazil (NMRSAXDCBRQ).csv",
        "date_col": "observation_date",
        "value_col": "NMRSAXDCBRQ",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "real imports of goods and services",
        "later_transformation": "quarter-on-quarter percentage growth",
    },
    {
        "series_name": "br_nominal_merchandise_exports",
        "file_name": "International Merchandise Trade Statistics Exports Commodities for Brazil (BRAXTEXVA01NCMLM).csv",
        "date_col": "observation_date",
        "value_col": "BRAXTEXVA01NCMLM",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "nominal merchandise exports; not identical to real national-accounts exports",
        "later_transformation": "quarterly sum and growth; retain caveat until real quarterly exports are found",
    },
    {
        "series_name": "br_real_investment",
        "file_name": "Real Gross Fixed Capital Formation for Brazil (NFIRSAXDCBRQ).csv",
        "date_col": "observation_date",
        "value_col": "NFIRSAXDCBRQ",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "real gross fixed capital formation",
        "later_transformation": "quarter-on-quarter percentage growth",
    },
    {
        "series_name": "br_real_private_consumption",
        "file_name": "Real Final Consumption Expenditure, Private Sector for Brazil (NCPRSAXDCBRQ).csv",
        "date_col": "observation_date",
        "value_col": "NCPRSAXDCBRQ",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "real private final consumption expenditure",
        "later_transformation": "quarter-on-quarter percentage growth",
    },
    {
        "series_name": "br_real_government_consumption",
        "file_name": "Real General Government Final Consumption Expenditure for Brazil (NCGGRSAXDCBRQ).csv",
        "date_col": "observation_date",
        "value_col": "NCGGRSAXDCBRQ",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "real general government final consumption expenditure",
        "later_transformation": "quarter-on-quarter percentage growth",
    },
    {
        "series_name": "br_manufacturing_growth",
        "file_name": "Production Manufacturing Total Manufacturing for Brazil (PRMNTO01BRQ657S).csv",
        "date_col": "observation_date",
        "value_col": "PRMNTO01BRQ657S",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "manufacturing production series already expressed as growth-type series",
        "later_transformation": "inspect metadata/units before applying any additional percentage change",
    },
    {
        "series_name": "br_retail_sales_growth",
        "file_name": "Sales Retail Trade Total Retail Trade Volume for Brazil (SLRTTO01BRQ657S).csv",
        "date_col": "observation_date",
        "value_col": "SLRTTO01BRQ657S",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "retail trade volume growth-type series",
        "later_transformation": "inspect metadata/units before applying any additional percentage change",
    },
    {
        "series_name": "br_capacity_utilisation",
        "file_name": "Business Tendency Surveys Rate of Capacity Utilisation Economic Activity Manufacturing Current for Brazil (BRABSCURT02STSAQ).csv",
        "date_col": "observation_date",
        "value_col": "BRABSCURT02STSAQ",
        "native_frequency": "quarterly",
        "role": "core_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "manufacturing capacity utilisation rate",
        "later_transformation": "use level or change after exploratory analysis",
    },
    {
        "series_name": "br_reserves_ex_gold",
        "file_name": "Reserves Excluding Gold for Brazil (TRESEGBRM052N).csv",
        "date_col": "observation_date",
        "value_col": "TRESEGBRM052N",
        "native_frequency": "monthly",
        "role": "core_candidate",
        "source_family": "FRED/IMF",
        "unit_note": "reserves excluding gold; monthly level",
        "later_transformation": "quarter-end level and/or growth; assess missing early observations",
    },

    # -------------------- SECONDARY / SHORTER SERIES --------------------
    {
        "series_name": "br_business_confidence",
        "file_name": "Business Tendency Surveys Composite Business Confidence Economic Activity Services for Brazil (BRABVCICP02STSAM).csv",
        "date_col": "observation_date",
        "value_col": "BRABVCICP02STSAM",
        "native_frequency": "monthly",
        "role": "secondary_candidate",
        "source_family": "FRED/OECD",
        "unit_note": "business confidence, services",
        "later_transformation": "quarterly average; likely recent/expanded feature because starts 2008",
    },
    {
        "series_name": "br_long_term_10y_yield",
        "file_name": "Brazil 10-Year Bond Yield Historical Data.csv",
        "date_col": "Date",
        "value_col": "Price",
        "native_frequency": "monthly",
        "role": "secondary_candidate",
        "source_family": "Investing.com export",
        "unit_note": "10-year government bond yield, monthly historical data",
        "later_transformation": "quarterly average; use only in shorter expanded feature set due 2007 start",
    },
    {
        "series_name": "br_credit_gap",
        "file_name": "bis_dp_search_export_20260909-160048.csv",
        "date_col": "TIME_PERIOD:Period",
        "value_col": "OBS_VALUE:Value",
        "native_frequency": "quarterly",
        "role": "secondary_candidate",
        "source_family": "BIS",
        "unit_note": "credit-to-GDP GAP (actual minus trend), percentage of GDP; not the credit/GDP ratio itself",
        "later_transformation": "optional financial-cycle feature; starts 2006",
    },
    {
        "series_name": "br_electricity_generation",
        "file_name": "MES_0526.csv",
        "date_col": "Time",
        "value_col": "Value",
        "native_frequency": "monthly",
        "role": "secondary_candidate",
        "source_family": "IEA",
        "unit_note": "Brazil net electricity production, Electricity product, GWh",
        "later_transformation": "quarterly total and growth; short-history feature only",
    },
]

# Explicit non-use / duplicate notes.
EXCLUSIONS = [
    {
        "source": "Consumer Price Indices (CPIs, HICPs), COICOP 1999 Consumer Price Index Total for Brazil (BRACPIALLMINMEI).csv",
        "reason": "Alternative CPI index retained raw; direct official IBGE IPCA YoY series selected for the initial Brazil pipeline.",
    },
    {
        "source": "Consumer Price Index All Items Total for Brazil (CPALTT01BRQ657N).csv",
        "reason": "Quarterly CPI alternative ends earlier than IBGE IPCA; not selected as canonical inflation source.",
    },
    {
        "source": "M3 for Brazil (MYAGM3BRM189N).csv / MYAGM3BRM189N.csv",
        "reason": "Long historical M3 alternatives end in 2017; retained for overlap validation against selected long-history M3 source.",
    },
    {
        "source": "Monetary Aggregates and Their Components Broad Money and Components M3 for Brazil (MABMM301BRM189N).csv",
        "reason": "Useful overlap check but ends in 2023; not sufficient alone for full cross-country period.",
    },
    {
        "source": "credit_to_the_private_sector_percentage_of_gdp_end.csv",
        "reason": "Multi-country annual dataset. Inventory frequency was misleading because countries are stacked. Do not interpolate annual values to quarters.",
    },
    {
        "source": "international_reserves_percentage_of_gdp_period_av.csv",
        "reason": "Multi-country annual dataset. Monthly reserves-ex-gold series selected instead.",
    },
    {
        "source": "Total Credit to Private Non-Financial Sector, Adjusted for Breaks, for Brazil (QBRPAMUSDA).csv",
        "reason": "Selected, but explicitly as a CREDIT LEVEL series, not as credit-to-GDP.",
    },
    {
        "source": "Real Broad Effective Exchange Rate for Brazil (RBBRBIS) (2).csv",
        "reason": "Duplicate/alternative copy; canonical RBBRBIS file selected.",
    },
    {
        "source": "Real General Government Final Consumption Expenditure for Brazil (NCGGRSAXDCBRQ) (2).csv",
        "reason": "Duplicate/alternative copy; canonical non-(2) file selected.",
    },
    {
        "source": "TRESEGBRM052N.csv",
        "reason": "Duplicate/alternative copy; descriptive-named reserves file selected.",
    },
    {
        "source": "MSCI Brazil Historical Data.csv",
        "reason": "Useful shorter daily equity series, but long monthly OECD/FRED share-price series is preferred for main sample.",
    },
    {
        "source": "CBOE Brazil ETF Volatility Index (VXEWZCLS).csv / CBOE Brazil Etf Volatility Historical Data.csv",
        "reason": "Too short for the long common sample. FX volatility will instead be derived from daily USD/BRL.",
    },
    {
        "source": "Production Construction Total for Brazil (PRCNTO01BRA657S).csv",
        "reason": "Annual only. Not interpolated into quarterly observations.",
    },
    {
        "source": "target-for-federal-funds.csv",
        "reason": "Despite filename, contains SELIC; starts only 2016. BACEN SELIC history is selected.",
    },
    {
        "source": "USD_BRL Historical Data.csv",
        "reason": "Monthly commercial export retained as secondary cross-check. Long daily DEXBZUS(1) series is preferred for FX change and realised volatility.",
    },
]

# ============================================================
# READERS
# ============================================================
def require_file(file_name):
    path = RAW_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Required selected source is missing:\n{path}"
        )
    return path


def read_commented_csv(path):
    """Read CSVs that may begin with # comment metadata."""
    return pd.read_csv(
        path,
        comment="#",
    )


def read_bis_export(path):
    """Read BIS export beginning at KEY:Timeseries Key."""
    text = path.read_text(
        encoding="utf-8-sig"
    )
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("KEY:Timeseries Key"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not locate BIS timeseries header."
        )

    return pd.read_csv(
        io.StringIO(
            "\n".join(lines[header_idx:])
        )
    )


def read_iea_mes(path):
    """
    IEA MES file contains a legal notice before a semicolon-delimited table.
    Extract only Brazil / Net Electricity Production / Electricity.
    """
    text = path.read_text(
        encoding="cp1252"
    )
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Country;Time;Balance;Product;Value;Unit"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not locate IEA MES table header."
        )

    df = pd.read_csv(
        io.StringIO(
            "\n".join(lines[header_idx:])
        ),
        sep=";",
    )

    df = df[
        (df["Country"].astype(str).str.strip() == "Brazil")
        & (
            df["Balance"].astype(str).str.strip()
            == "Net Electricity Production"
        )
        & (
            df["Product"].astype(str).str.strip()
            == "Electricity"
        )
    ].copy()

    return df


def read_source(spec):
    path = require_file(
        spec["file_name"]
    )

    if spec["series_name"] == "br_credit_gap":
        return read_bis_export(path)

    if spec["series_name"] == "br_electricity_generation":
        return read_iea_mes(path)

    if spec["series_name"] == "br_m3_level":
        return read_commented_csv(path)

    return pd.read_csv(path)


def parse_dates(series, series_name):
    s = series.astype(str).str.strip()

    if series_name == "br_headline_inflation_yoy":
        return pd.to_datetime(
            s,
            format="%m/%Y",
            errors="coerce",
        )

    if series_name == "br_electricity_generation":
        return pd.to_datetime(
            s,
            format="%b-%y",
            errors="coerce",
        )

    # BACEN timestamps include Z.
    if series_name == "br_selic_target":
        return pd.to_datetime(
            s,
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)

    return pd.to_datetime(
        s,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)


def clean_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    s = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    return pd.to_numeric(
        s,
        errors="coerce",
    )

# ============================================================
# EXTRACTION
# ============================================================
if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Brazil raw directory does not exist:\n{RAW_DIR}"
    )

long_parts = []
audit_rows = []
manifest_rows = []

print("\n" + "=" * 78)
print("14 - BRAZIL SOURCE SELECTION + EXTRACTION")
print("=" * 78)

for spec in SOURCE_SPECS:
    df = read_source(spec)

    if spec["date_col"] not in df.columns:
        raise KeyError(
            f"{spec['series_name']}: date column "
            f"{spec['date_col']!r} not found in {spec['file_name']}.\n"
            f"Available columns: {list(df.columns)}"
        )

    if spec["value_col"] not in df.columns:
        raise KeyError(
            f"{spec['series_name']}: value column "
            f"{spec['value_col']!r} not found in {spec['file_name']}.\n"
            f"Available columns: {list(df.columns)}"
        )

    dates = parse_dates(
        df[spec["date_col"]],
        spec["series_name"],
    )

    values = clean_numeric(
        df[spec["value_col"]]
    )

    out = pd.DataFrame({
        "date": dates,
        "value": values,
    })

    # For the IEA extraction this is already filtered to Brazil total generation.
    out = (
        out.dropna(
            subset=["date", "value"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # BACEN file is reverse chronological; DataInicioVigencia identifies
    # when the target became effective. Keep the last record if duplicates occur.
    duplicate_dates_before = int(
        out["date"].duplicated(
            keep=False
        ).sum()
    )

    out = (
        out.drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    out["series_name"] = spec[
        "series_name"
    ]

    out["native_frequency"] = spec[
        "native_frequency"
    ]

    out["role"] = spec[
        "role"
    ]

    out["source_file"] = spec[
        "file_name"
    ]

    out["source_family"] = spec[
        "source_family"
    ]

    out["unit_note"] = spec[
        "unit_note"
    ]

    long_parts.append(
        out[
            [
                "series_name",
                "date",
                "value",
                "native_frequency",
                "role",
                "source_file",
                "source_family",
                "unit_note",
            ]
        ]
    )

    # Save one clean source-native file per selected series.
    series_path = (
        INTERIM_DIR
        / f"{spec['series_name']}.csv"
    )

    out[
        ["date", "value"]
    ].to_csv(
        series_path,
        index=False,
    )

    audit_rows.append({
        "series_name":
            spec["series_name"],
        "source_file":
            spec["file_name"],
        "source_family":
            spec["source_family"],
        "role":
            spec["role"],
        "native_frequency":
            spec["native_frequency"],
        "raw_rows_read":
            int(len(df)),
        "extracted_observations":
            int(len(out)),
        "start_date":
            out["date"].min(),
        "end_date":
            out["date"].max(),
        "duplicate_date_rows_before_dedup":
            duplicate_dates_before,
        "missing_values_after_extraction":
            int(out["value"].isna().sum()),
        "min_value":
            float(out["value"].min()),
        "max_value":
            float(out["value"].max()),
        "output_file":
            str(series_path),
    })

    manifest_rows.append({
        **spec,
        "selected":
            True,
        "extracted_observations":
            int(len(out)),
        "start_date":
            out["date"].min(),
        "end_date":
            out["date"].max(),
    })

    print(
        f"{spec['series_name']:<36} "
        f"{len(out):>6} obs | "
        f"{out['date'].min().date()} -> "
        f"{out['date'].max().date()} | "
        f"{spec['role']}"
    )

# ============================================================
# CONSOLIDATED OUTPUTS
# ============================================================
extracted_long = pd.concat(
    long_parts,
    ignore_index=True,
)

extracted_long.to_csv(
    EXTRACTED_LONG_OUT,
    index=False,
)

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    EXTRACTION_AUDIT_OUT,
    index=False,
)

manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    SOURCE_MANIFEST_OUT,
    index=False,
)

exclusion_df = pd.DataFrame(
    EXCLUSIONS
)

exclusion_df.to_csv(
    EXCLUSION_OUT,
    index=False,
)

# ============================================================
# VALIDATION FLAGS
# ============================================================
validation_notes = []

# M3 requires overlap validation because selected long source is not the same
# institutional source as the shorter FRED series.
validation_notes.append(
    "M3: selected Eulerpool long-history series must be validated against "
    "MYAGM3BRM189N and MABMM301BRM189N over overlapping periods before modelling."
)

# Credit warning.
validation_notes.append(
    "Private credit: QBRPAMUSDA is a quarterly level series, NOT credit-to-GDP. "
    "Do not reuse the South African private_credit_gdp_change label unless a genuine "
    "quarterly Brazil credit/GDP ratio is later obtained."
)

# Export warning.
validation_notes.append(
    "Exports: current selected source is nominal merchandise exports. It is not "
    "fully equivalent to quarterly real exports of goods and services. Preserve "
    "this distinction in cross-country feature mapping."
)

# Electricity warning.
validation_notes.append(
    "Electricity: IEA MES is extracted correctly only after filtering Country=Brazil, "
    "Balance=Net Electricity Production, Product=Electricity. It remains a shorter "
    "secondary feature."
)

# Long yield / business confidence sample warning.
validation_notes.append(
    "10Y yield (2007+) and business confidence (2008+) remain secondary because "
    "forcing them into the core set would materially shorten the ~1996+ sample."
)

# ============================================================
# CONFIG
# ============================================================
config = {
    "country":
        "Brazil",
    "stage":
        "source selection and source-native extraction",
    "raw_files_modified":
        False,
    "annual_interpolation_allowed":
        False,
    "core_design_goal":
        "Preserve a long quarterly-compatible sample beginning around 1996 where possible.",
    "selected_series_count":
        len(SOURCE_SPECS),
    "core_candidate_count":
        sum(
            s["role"] == "core_candidate"
            for s in SOURCE_SPECS
        ),
    "secondary_candidate_count":
        sum(
            s["role"] == "secondary_candidate"
            for s in SOURCE_SPECS
        ),
    "validation_notes":
        validation_notes,
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
# SUMMARY
# ============================================================
core = audit_df[
    audit_df["role"]
    == "core_candidate"
].copy()

secondary = audit_df[
    audit_df["role"]
    == "secondary_candidate"
].copy()

summary_lines = [
    "=" * 78,
    "BRAZIL SOURCE SELECTION + EXTRACTION SUMMARY",
    "=" * 78,
    "",
    f"Selected series: {len(audit_df)}",
    f"Core candidates: {len(core)}",
    f"Secondary candidates: {len(secondary)}",
    "",
    "Core candidate coverage:",
]

for _, row in core.iterrows():
    summary_lines.append(
        f"  {row['series_name']}: "
        f"{pd.to_datetime(row['start_date']).date()} -> "
        f"{pd.to_datetime(row['end_date']).date()} "
        f"({int(row['extracted_observations'])} obs)"
    )

summary_lines.extend([
    "",
    "Important unresolved equivalence issues:",
    "  1. Brazil private credit source is a LEVEL, not credit/GDP.",
    "  2. Brazil exports source is nominal merchandise exports, not real total exports.",
    "  3. Long-history M3 source requires overlap validation.",
    "  4. 10Y yield/business confidence/electricity are secondary due shorter coverage.",
    "",
    "No modelling transformation or quarterly harmonisation was performed here.",
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
print("SOURCE SELECTION + EXTRACTION COMPLETE")
print("=" * 78)

print(
    f"\nSelected series:      {len(audit_df)}"
)

print(
    f"Core candidates:      {len(core)}"
)

print(
    f"Secondary candidates: {len(secondary)}"
)

print("\nImportant validation notes:")
for note in validation_notes:
    print(f"  - {note}")

print("\nOutputs:")
print(f"  {SOURCE_MANIFEST_OUT}")
print(f"  {EXCLUSION_OUT}")
print(f"  {EXTRACTED_LONG_OUT}")
print(f"  {EXTRACTION_AUDIT_OUT}")
print(f"  {SUMMARY_OUT}")
print(f"  {CONFIG_OUT}")

if __name__ == "__main__":
    pass
