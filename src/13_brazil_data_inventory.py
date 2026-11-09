from pathlib import Path
import csv
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

# Put ALL Brazil raw files here without manually editing them.
RAW_DIR = ROOT / "data" / "raw" / "brazil"

OUT_DIR = ROOT / "data" / "processed" / "brazil_inventory"
FIG_DIR = ROOT / "figures" / "brazil_inventory"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FILE_INVENTORY_OUT = OUT_DIR / "brazil_file_inventory.csv"
SERIES_INVENTORY_OUT = OUT_DIR / "brazil_series_inventory.csv"
DUPLICATE_OUT = OUT_DIR / "brazil_exact_duplicate_groups.csv"
ZIP_CONTENTS_OUT = OUT_DIR / "brazil_zip_contents.csv"
SUMMARY_OUT = OUT_DIR / "brazil_inventory_summary.txt"
MANIFEST_OUT = OUT_DIR / "brazil_candidate_feature_manifest.csv"

FIGURE_DPI = 300

# ============================================================
# FILE TYPES
# ============================================================
TABULAR_EXTENSIONS = {
    ".csv",
    ".txt",
    ".tsv",
    ".xlsx",
    ".xls",
}

# ============================================================
# FEATURE CLASSIFICATION
# ============================================================
# These are only inventory labels. They do NOT transform the data.
FEATURE_RULES = [
    ("gdp_real", [
        r"real gross domestic product",
        r"ngdprsaxdcbrq",
    ]),
    ("private_consumption_real", [
        r"real final consumption expenditure.*private",
        r"ncprsaxdcbrq",
    ]),
    ("government_consumption_real", [
        r"real general government",
        r"real final consumption expenditure.*government",
        r"ncggrsaxdcbrq",
    ]),
    ("gross_fixed_capital_formation_real", [
        r"real gross fixed capital formation",
        r"nfirsaxdcbrq",
    ]),
    ("imports_real", [
        r"real imports of goods and services",
        r"nmrsaxdcbrq",
    ]),
    ("exports_real", [
        r"real exports of goods and services",
        r"nxrsaxdcbrq",
        r"nex.*brazil",
    ]),
    ("headline_cpi", [
        r"ipca",
        r"consumer price index.*brazil",
        r"cpi.*brazil",
        r"cpaltt01br",
        r"bracpiall",
    ]),
    ("policy_rate_selic", [
        r"selic",
        r"bacen_selic",
    ]),
    ("short_term_interest_rate", [
        r"treasury bills",
        r"intgstbrm193n",
    ]),
    ("long_term_bond_yield", [
        r"10-year bond",
        r"10 year bond",
    ]),
    ("m3", [
        r"\bm3\b",
        r"money supply m3",
        r"mabmm301br",
        r"myagm3br",
    ]),
    ("private_credit_gdp", [
        r"credit.*private.*gdp",
        r"credit to the private sector",
        r"qbrpam",
    ]),
    ("private_credit_level", [
        r"total credit to private non-financial",
    ]),
    ("usd_brl", [
        r"usd[_ -]?brl",
        r"brazilian reals to u\.s\. dollar",
        r"dexbzus",
    ]),
    ("reer", [
        r"real broad effective exchange rate",
        r"rbbrbis",
    ]),
    ("neer", [
        r"broad effective exchange rate",
        r"nbbrbis",
    ]),
    ("share_prices", [
        r"share prices.*brazil",
        r"spastt01br",
        r"msci brazil",
    ]),
    ("market_volatility", [
        r"volatility",
        r"vxewz",
    ]),
    ("manufacturing", [
        r"production manufacturing",
        r"prmnto01br",
    ]),
    ("retail_sales", [
        r"retail trade",
        r"slrtto01br",
    ]),
    ("capacity_utilisation", [
        r"capacity utilisation",
        r"bscurt",
    ]),
    ("business_confidence", [
        r"business confidence",
        r"bvcicp",
        r"business tendency.*confidence",
    ]),
    ("reserves", [
        r"reserves excluding gold",
        r"tresegbr",
        r"international_reserves",
        r"qsreservasi",
    ]),
    ("construction", [
        r"production construction",
        r"prcnto01bra",
        r"construction",
        r"building permits",
        r"building plans",
    ]),
    ("electricity", [
        r"electricity",
        r"mes_",
        r"eg\.elc",
    ]),
    ("trade_exports_nominal", [
        r"merchandise trade.*exports",
        r"braxtex",
    ]),
    ("trade_balance_nominal", [
        r"trade balance",
        r"braxtnt",
        r"xtntva",
    ]),
    ("world_bank_annual", [
        r"p_data_extract",
        r"api_",
    ]),
    ("bis_credit_gap", [
        r"bis_dp_search_export",
        r"credit gap",
    ]),
]

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def classify_feature(name):
    text = name.lower()
    for label, patterns in FEATURE_RULES:
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                return label
    return "unclassified"


def detect_separator_from_text(text):
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    ]

    if not lines:
        return ","

    sample = "\n".join(lines[:10])

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        )
        return dialect.delimiter
    except Exception:
        # IEA MES files are semicolon-delimited.
        counts = {
            ",": sample.count(","),
            ";": sample.count(";"),
            "\t": sample.count("\t"),
            "|": sample.count("|"),
        }
        return max(
            counts,
            key=counts.get,
        )


def first_probable_header_line(text):
    """
    Finds a likely header line in files containing comments or metadata rows.
    """
    lines = text.splitlines()

    for i, line in enumerate(lines[:80]):
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        lower = stripped.lower()

        # Strong known header patterns.
        known_header_terms = [
            "observation_date",
            "date,",
            "date;",
            "country;",
            "time series search export",
            "key:timeseries",
            "numeroreuniaocopom",
        ]

        if any(term in lower for term in known_header_terms):
            # BIS export has a metadata header before the actual timeseries header.
            if "time series search export" in lower:
                continue
            return i

        # Generic: at least two delimiters and contains letters.
        sep_counts = (
            stripped.count(",")
            + stripped.count(";")
            + stripped.count("\t")
        )

        if (
            sep_counts >= 1
            and re.search(r"[A-Za-z]", stripped)
        ):
            return i

    return 0


def read_text_guess(path):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    for encoding in encodings:
        try:
            return path.read_text(
                encoding=encoding
            ), encoding
        except Exception:
            pass

    return "", "unknown"


def read_csv_flexible(path):
    """
    Reads heterogeneous CSV exports without manually changing raw files.
    Returns (df, metadata).
    """
    text, encoding = read_text_guess(path)

    if not text:
        raise ValueError(
            "Could not decode file."
        )

    header_line = first_probable_header_line(
        text
    )

    remaining = "\n".join(
        text.splitlines()[
            header_line:
        ]
    )

    sep = detect_separator_from_text(
        remaining
    )

    # Special handling for BIS exports:
    # locate the row beginning KEY:Timeseries Key
    if "KEY:Timeseries Key" in text:
        lines = text.splitlines()

        for i, line in enumerate(lines):
            if line.startswith(
                "KEY:Timeseries Key"
            ):
                header_line = i
                remaining = "\n".join(
                    lines[i:]
                )
                sep = ","
                break

    df = pd.read_csv(
        io.StringIO(
            remaining
        ),
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df, {
        "encoding":
            encoding,
        "separator":
            repr(sep),
        "header_line_1_based":
            header_line + 1,
    }


def read_excel_flexible(path):
    book = pd.ExcelFile(
        path
    )

    frames = []

    for sheet in book.sheet_names:
        try:
            frame = pd.read_excel(
                path,
                sheet_name=sheet,
            )

            frame.columns = [
                str(c).strip()
                for c in frame.columns
            ]

            if len(frame):
                frame[
                    "__sheet_name__"
                ] = sheet

                frames.append(
                    frame
                )
        except Exception:
            continue

    if not frames:
        raise ValueError(
            "No readable Excel sheets."
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    ), {
        "encoding":
            "excel",
        "separator":
            "",
        "header_line_1_based":
            1,
        "sheet_count":
            len(book.sheet_names),
    }


def detect_date_column(df):
    """
    Returns the most plausible date/time column.
    """
    preferred_patterns = [
        r"^observation_date$",
        r"^date$",
        r"^time$",
        r"^time_period",
        r"^period$",
        r"datareuniaocopom",
        r"datainiciovigencia",
        r"year",
    ]

    for pattern in preferred_patterns:
        for col in df.columns:
            if re.search(
                pattern,
                str(col),
                flags=re.I,
            ):
                return col

    # Fall back to testing object columns.
    best_col = None
    best_ratio = 0.0

    for col in df.columns:
        series = df[col]

        if (
            pd.api.types.is_numeric_dtype(
                series
            )
        ):
            continue

        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

        ratio = float(
            parsed.notna().mean()
        )

        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    if best_ratio >= 0.70:
        return best_col

    return None


def parse_date_series(series):
    """
    Handles ordinary dates, year-only fields and Month-YY strings.
    """
    s = (
        series.astype(str)
        .str.strip()
    )

    # Numeric years.
    year_only = s.str.fullmatch(
        r"\d{4}"
    )

    if (
        len(s)
        and year_only.mean()
        >= 0.80
    ):
        return pd.to_datetime(
            s + "-01-01",
            errors="coerce",
        )

    # Month-YY 
    month_year = s.str.fullmatch(
        r"[A-Za-z]{3}-\d{2}"
    )

    if (
        len(s)
        and month_year.mean()
        >= 0.50
    ):
        return pd.to_datetime(
            s,
            format="%b-%y",
            errors="coerce",
        )

    return pd.to_datetime(
        s,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)


def infer_frequency(dates):
    dates = (
        pd.Series(dates)
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(dates) < 3:
        return "unknown"

    diffs = (
        dates.diff()
        .dropna()
        .dt.days
    )

    if diffs.empty:
        return "unknown"

    median_days = float(
        diffs.median()
    )

    if median_days <= 2:
        return "daily"

    if median_days <= 10:
        return "weekly"

    if 20 <= median_days <= 40:
        return "monthly"

    if 70 <= median_days <= 100:
        return "quarterly"

    if 150 <= median_days <= 220:
        return "semiannual"

    if 300 <= median_days <= 430:
        return "annual"

    return (
        f"irregular_"
        f"{median_days:.0f}d"
    )


def clean_numeric(series):
    """
    Converts common financial-export formats to numeric.
    """
    if pd.api.types.is_numeric_dtype(
        series
    ):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    s = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "":
                    np.nan,
                ".":
                    np.nan,
                "nan":
                    np.nan,
                "None":
                    np.nan,
            }
        )
    )

    # Remove thousands separators, percentage signs and common suffixes.
    s = (
        s.str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "%",
            "",
            regex=False,
        )
        .str.replace(
            "K",
            "e3",
            regex=False,
        )
        .str.replace(
            "M",
            "e6",
            regex=False,
        )
        .str.replace(
            "B",
            "e9",
            regex=False,
        )
    )

    return pd.to_numeric(
        s,
        errors="coerce",
    )


def candidate_numeric_columns(
    df,
    date_col,
):
    candidates = []

    excluded_name_patterns = [
        r"confidential",
        r"status",
        r"key",
        r"country",
        r"product",
        r"balance",
        r"unit",
        r"frequency",
        r"source",
        r"code",
        r"label",
        r"sheet",
    ]

    for col in df.columns:
        if col == date_col:
            continue

        if any(
            re.search(
                p,
                str(col),
                flags=re.I,
            )
            for p in excluded_name_patterns
        ):
            continue

        numeric = clean_numeric(
            df[col]
        )

        non_missing = int(
            numeric.notna().sum()
        )

        if non_missing >= max(
            3,
            int(
                0.20
                * max(
                    len(df),
                    1,
                )
            ),
        ):
            candidates.append(
                (
                    col,
                    numeric,
                )
            )

    return candidates


def completeness_pct(
    values,
):
    if len(values) == 0:
        return np.nan

    return float(
        100
        * pd.Series(
            values
        )
        .notna()
        .mean()
    )


def inspect_dataframe(
    df,
    source_name,
    source_type,
):
    """
    Returns series-level inventory rows for one parsed table.
    """
    rows = []

    date_col = detect_date_column(
        df
    )

    if date_col is None:
        return rows

    dates = parse_date_series(
        df[date_col]
    )

    valid_date_mask = (
        dates.notna()
    )

    if valid_date_mask.sum() < 3:
        return rows

    frequency = infer_frequency(
        dates
    )

    for col, numeric in (
        candidate_numeric_columns(
            df,
            date_col,
        )
    ):
        mask = (
            valid_date_mask
            & numeric.notna()
        )

        if mask.sum() < 3:
            continue

        series_dates = dates[
            mask
        ]

        values = numeric[
            mask
        ]

        rows.append({
            "source_name":
                source_name,
            "source_type":
                source_type,
            "candidate_feature":
                classify_feature(
                    source_name
                    + " "
                    + str(col)
                ),
            "date_column":
                date_col,
            "value_column":
                col,
            "inferred_frequency":
                frequency,
            "start_date":
                series_dates.min(),
            "end_date":
                series_dates.max(),
            "non_missing_observations":
                int(
                    values.notna().sum()
                ),
            "raw_rows":
                int(
                    len(df)
                ),
            "value_completeness_pct":
                completeness_pct(
                    numeric
                ),
            "duplicate_dates":
                int(
                    series_dates.duplicated().sum()
                ),
            "min_value":
                float(
                    values.min()
                ),
            "max_value":
                float(
                    values.max()
                ),
            "mean_value":
                float(
                    values.mean()
                ),
        })

    return rows


def inspect_zip(path):
    rows = []
    series_rows = []

    with zipfile.ZipFile(
        path,
        "r",
    ) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue

            suffix = Path(
                member
            ).suffix.lower()

            info = zf.getinfo(
                member
            )

            rows.append({
                "zip_file":
                    path.name,
                "member_name":
                    member,
                "member_extension":
                    suffix,
                "uncompressed_bytes":
                    int(
                        info.file_size
                    ),
                "compressed_bytes":
                    int(
                        info.compress_size
                    ),
            })

            if suffix not in {
                ".csv",
                ".txt",
                ".tsv",
            }:
                continue

            try:
                raw = zf.read(
                    member
                )

                text = None

                for encoding in [
                    "utf-8-sig",
                    "utf-8",
                    "cp1252",
                    "latin1",
                ]:
                    try:
                        text = raw.decode(
                            encoding
                        )
                        break
                    except Exception:
                        pass

                if text is None:
                    continue

                header_line = (
                    first_probable_header_line(
                        text
                    )
                )

                remaining = "\n".join(
                    text.splitlines()[
                        header_line:
                    ]
                )

                sep = (
                    detect_separator_from_text(
                        remaining
                    )
                )

                df = pd.read_csv(
                    io.StringIO(
                        remaining
                    ),
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )

                df.columns = [
                    str(c).strip()
                    for c in df.columns
                ]

                series_rows.extend(
                    inspect_dataframe(
                        df,
                        source_name=
                            f"{path.name}::{member}",
                        source_type=
                            "zip_member",
                    )
                )

            except Exception:
                continue

    return rows, series_rows

# ============================================================
# MAIN INVENTORY
# ============================================================
if not RAW_DIR.exists():
    raise FileNotFoundError(
        "Brazil raw-data folder does not exist:\n"
        f"{RAW_DIR}\n\n"
        "Create data\\raw\\brazil and copy the Brazil files into it "
        "without manually editing the raw files."
    )

all_files = sorted(
    [
        p
        for p in RAW_DIR.rglob("*")
        if p.is_file()
    ]
)

if not all_files:
    raise RuntimeError(
        f"No files found in {RAW_DIR}"
    )

file_rows = []
series_rows = []
zip_rows = []

print("\n" + "=" * 78)
print("13 - BRAZIL DATA INVENTORY")
print("=" * 78)

print(
    f"\nRaw folder: {RAW_DIR}"
)

print(
    f"Files found: {len(all_files)}"
)

for path in all_files:
    relative = str(
        path.relative_to(
            RAW_DIR
        )
    )

    suffix = (
        path.suffix.lower()
    )

    file_hash = sha256_file(
        path
    )

    row = {
        "relative_path":
            relative,
        "file_name":
            path.name,
        "extension":
            suffix,
        "size_bytes":
            int(
                path.stat().st_size
            ),
        "sha256":
            file_hash,
        "candidate_feature":
            classify_feature(
                path.name
            ),
        "parse_status":
            "",
        "parsed_rows":
            np.nan,
        "parsed_columns":
            np.nan,
        "detected_date_column":
            "",
        "detected_frequency":
            "",
        "start_date":
            pd.NaT,
        "end_date":
            pd.NaT,
        "encoding":
            "",
        "separator":
            "",
        "notes":
            "",
    }

    try:
        if suffix == ".zip":
            zip_info, zip_series = (
                inspect_zip(
                    path
                )
            )

            zip_rows.extend(
                zip_info
            )

            series_rows.extend(
                zip_series
            )

            row[
                "parse_status"
            ] = "zip_inspected"

            row[
                "notes"
            ] = (
                f"{len(zip_info)} archive members; "
                f"{len(zip_series)} parseable numeric series"
            )

        elif suffix in {
            ".csv",
            ".txt",
            ".tsv",
        }:
            df, meta = read_csv_flexible(
                path
            )

            row[
                "parse_status"
            ] = "parsed"

            row[
                "parsed_rows"
            ] = int(
                len(df)
            )

            row[
                "parsed_columns"
            ] = int(
                len(df.columns)
            )

            row[
                "encoding"
            ] = meta.get(
                "encoding",
                "",
            )

            row[
                "separator"
            ] = meta.get(
                "separator",
                "",
            )

            date_col = detect_date_column(
                df
            )

            if date_col is not None:
                dates = parse_date_series(
                    df[date_col]
                )

                if dates.notna().any():
                    row[
                        "detected_date_column"
                    ] = date_col

                    row[
                        "detected_frequency"
                    ] = infer_frequency(
                        dates
                    )

                    row[
                        "start_date"
                    ] = dates.min()

                    row[
                        "end_date"
                    ] = dates.max()

            series_rows.extend(
                inspect_dataframe(
                    df,
                    source_name=
                        relative,
                    source_type=
                        "csv",
                )
            )

        elif suffix in {
            ".xlsx",
            ".xls",
        }:
            df, meta = read_excel_flexible(
                path
            )

            row[
                "parse_status"
            ] = "parsed"

            row[
                "parsed_rows"
            ] = int(
                len(df)
            )

            row[
                "parsed_columns"
            ] = int(
                len(df.columns)
            )

            row[
                "notes"
            ] = (
                f"{meta.get('sheet_count', 0)} sheets"
            )

            series_rows.extend(
                inspect_dataframe(
                    df,
                    source_name=
                        relative,
                    source_type=
                        "excel",
                )
            )

        else:
            row[
                "parse_status"
            ] = "not_tabular"

    except Exception as exc:
        row[
            "parse_status"
        ] = "parse_error"

        row[
            "notes"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )[:500]

    file_rows.append(
        row
    )

file_df = pd.DataFrame(
    file_rows
)

series_df = pd.DataFrame(
    series_rows
)

zip_df = pd.DataFrame(
    zip_rows
)

# ============================================================
# EXACT DUPLICATES
# ============================================================
duplicate_rows = []

duplicate_groups = (
    file_df
    .groupby(
        "sha256",
        dropna=False,
    )
)

group_id = 0

for file_hash, group in (
    duplicate_groups
):
    if len(group) <= 1:
        continue

    group_id += 1

    for _, r in group.iterrows():
        duplicate_rows.append({
            "duplicate_group":
                group_id,
            "sha256":
                file_hash,
            "relative_path":
                r[
                    "relative_path"
                ],
            "file_name":
                r[
                    "file_name"
                ],
            "size_bytes":
                r[
                    "size_bytes"
                ],
        })

duplicate_df = pd.DataFrame(
    duplicate_rows
)

# ============================================================
# CANDIDATE FEATURE MANIFEST
# ============================================================
if not series_df.empty:
    manifest = (
        series_df
        .sort_values(
            [
                "candidate_feature",
                "start_date",
                "end_date",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # Initial usability flag only. Final selection happens later.
    manifest[
        "inventory_usability"
    ] = np.select(
        [
            manifest[
                "inferred_frequency"
            ].isin(
                [
                    "daily",
                    "weekly",
                    "monthly",
                    "quarterly",
                ]
            ),

            manifest[
                "inferred_frequency"
            ].eq(
                "annual"
            ),
        ],
        [
            "potential_main_model",
            "annual_last_resort",
        ],
        default=
            "review",
    )

else:
    manifest = pd.DataFrame()

# ============================================================
# SAVE OUTPUTS
# ============================================================
file_df.to_csv(
    FILE_INVENTORY_OUT,
    index=False,
)

series_df.to_csv(
    SERIES_INVENTORY_OUT,
    index=False,
)

duplicate_df.to_csv(
    DUPLICATE_OUT,
    index=False,
)

zip_df.to_csv(
    ZIP_CONTENTS_OUT,
    index=False,
)

manifest.to_csv(
    MANIFEST_OUT,
    index=False,
)

# ============================================================
# COVERAGE FIGURE
# ============================================================
if not manifest.empty:
    plot_df = (
        manifest[
            manifest[
                "start_date"
            ].notna()
            & manifest[
                "end_date"
            ].notna()
        ]
        .copy()
    )

    # Prefer one longest series per classified feature for readability.
    plot_df[
        "coverage_days"
    ] = (
        pd.to_datetime(
            plot_df[
                "end_date"
            ]
        )
        - pd.to_datetime(
            plot_df[
                "start_date"
            ]
        )
    ).dt.days

    plot_df = (
        plot_df
        .sort_values(
            "coverage_days",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "candidate_feature",
            ],
            keep="first",
        )
    )

    plot_df = plot_df[
        plot_df[
            "candidate_feature"
        ]
        != "unclassified"
    ]

    if len(plot_df):
        fig_height = max(
            6,
            0.42
            * len(plot_df),
        )

        fig, ax = plt.subplots(
            figsize=(
                11,
                fig_height,
            )
        )

        for y, (_, row) in enumerate(
            plot_df.iterrows()
        ):
            start = pd.to_datetime(
                row[
                    "start_date"
                ]
            )

            end = pd.to_datetime(
                row[
                    "end_date"
                ]
            )

            ax.plot(
                [start, end],
                [y, y],
                linewidth=4,
            )

        ax.set_yticks(
            np.arange(
                len(plot_df)
            )
        )

        ax.set_yticklabels(
            plot_df[
                "candidate_feature"
            ]
        )

        ax.set_title(
            "Brazil Candidate-Series Coverage"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Candidate feature"
        )

        fig.tight_layout()

        fig.savefig(
            FIG_DIR
            / "brazil_candidate_series_coverage.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

# ============================================================
# TEXT SUMMARY
# ============================================================
frequency_counts = (
    series_df[
        "inferred_frequency"
    ]
    .value_counts(
        dropna=False
    )
    if not series_df.empty
    else pd.Series(
        dtype=int
    )
)

feature_counts = (
    series_df[
        "candidate_feature"
    ]
    .value_counts(
        dropna=False
    )
    if not series_df.empty
    else pd.Series(
        dtype=int
    )
)

summary_lines = [
    "=" * 78,
    "BRAZIL DATA INVENTORY SUMMARY",
    "=" * 78,
    "",
    f"Raw directory: {RAW_DIR}",
    f"Files inspected: {len(file_df)}",
    f"Parseable numeric series found: {len(series_df)}",
    f"Exact duplicate file groups: {duplicate_df['duplicate_group'].nunique() if len(duplicate_df) else 0}",
    f"ZIP archive members listed: {len(zip_df)}",
    "",
    "Frequency counts:",
]

for frequency, count in (
    frequency_counts.items()
):
    summary_lines.append(
        f"  {frequency}: {count}"
    )

summary_lines.extend([
    "",
    "Candidate feature counts:",
])

for feature, count in (
    feature_counts.items()
):
    summary_lines.append(
        f"  {feature}: {count}"
    )

summary_lines.extend([
    "",
    "Important interpretation:",
    "  - Exact duplicates are identified by SHA-256 hash; raw files are NOT deleted.",
    "  - Annual series are inventoried but are not automatically used in the quarterly model.",
    "  - Frequency detection is diagnostic and must be reviewed before harmonisation.",
    "  - No raw file is modified by this script.",
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
print("\nInventory complete.")

print(
    f"Files inspected: {len(file_df)}"
)

print(
    f"Numeric series found: {len(series_df)}"
)

print(
    "Exact duplicate groups: "
    f"{duplicate_df['duplicate_group'].nunique() if len(duplicate_df) else 0}"
)

if not series_df.empty:
    print("\nDetected series by frequency:")
    print(
        frequency_counts.to_string()
    )

    print("\nCandidate feature coverage:")
    compact = (
        manifest[
            [
                "candidate_feature",
                "source_name",
                "value_column",
                "inferred_frequency",
                "start_date",
                "end_date",
                "non_missing_observations",
                "inventory_usability",
            ]
        ]
        .sort_values(
            [
                "candidate_feature",
                "start_date",
            ]
        )
    )

    print(
        compact.to_string(
            index=False,
        )
    )

print("\nOutputs:")
print(
    f"  {FILE_INVENTORY_OUT}"
)
print(
    f"  {SERIES_INVENTORY_OUT}"
)
print(
    f"  {DUPLICATE_OUT}"
)
print(
    f"  {ZIP_CONTENTS_OUT}"
)
print(
    f"  {MANIFEST_OUT}"
)
print(
    f"  {SUMMARY_OUT}"
)

print("\n300-DPI figure:")
print(
    f"  {FIG_DIR / 'brazil_candidate_series_coverage.png'}"
)

if __name__ == "__main__":
    pass
