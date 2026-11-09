from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "results"
    / "brazil_final_q0_o1"
)

RUN_PATH = (
    SOURCE_DIR
    / "19_brazil_run_metrics.csv"
)

PREDICTION_PATH = (
    SOURCE_DIR
    / "19_brazil_test_predictions_by_seed.csv"
)

MEAN_PREDICTION_PATH = (
    SOURCE_DIR
    / "19_brazil_mean_test_predictions.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "brazil_q0_o1_statistical_evaluation"
)

FIGURE_DIR = (
    ROOT
    / "figures"
    / "brazil_q0_o1_statistical_evaluation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PAIRED_RUN_OUT = (
    RESULT_DIR
    / "20_brazil_paired_seed_comparison.csv"
)

PAIRED_SUMMARY_OUT = (
    RESULT_DIR
    / "20_brazil_paired_seed_summary.csv"
)

QUARTER_LOSS_OUT = (
    RESULT_DIR
    / "20_brazil_quarter_level_loss_comparison.csv"
)

BOOTSTRAP_OUT = (
    RESULT_DIR
    / "20_brazil_moving_block_bootstrap.csv"
)

DIAGNOSTIC_OUT = (
    RESULT_DIR
    / "20_brazil_full_covid_diagnostic.csv"
)

SUMMARY_OUT = (
    RESULT_DIR
    / "20_brazil_statistical_summary.json"
)

FIGURE_DPI = 300

# ============================================================
# LOCKED STATISTICAL DESIGN
# ============================================================
Q0 = "Q0_aligned_quarter_token"
O1 = "O1_aligned_observation_token"

EXPECTED_TEST_QUARTERS = 28

BLOCK_LENGTH = 4
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 2026

# Same secondary crisis window used in the South African analysis.
COVID_START = pd.Timestamp("2020-04-01")
COVID_END = pd.Timestamp("2021-01-01")

# ============================================================
# LOAD SCRIPT 19 OUTPUTS
# ============================================================
for path in [
    RUN_PATH,
    PREDICTION_PATH,
    MEAN_PREDICTION_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            "Required Script 19 output not found:\n"
            f"{path}\n"
            "Run 19_brazil_frozen_q0_o1_evaluation.py first."
        )

run_df = pd.read_csv(
    RUN_PATH
)

prediction_df = pd.read_csv(
    PREDICTION_PATH
)

prediction_df[
    "target_date"
] = pd.to_datetime(
    prediction_df[
        "target_date"
    ],
    errors="coerce",
)

mean_prediction_df = pd.read_csv(
    MEAN_PREDICTION_PATH
)

mean_prediction_df[
    "target_date"
] = pd.to_datetime(
    mean_prediction_df[
        "target_date"
    ],
    errors="coerce",
)

# ============================================================
# AUDIT CHECKS
# ============================================================
required_models = {
    Q0,
    O1,
}

if (
    set(
        run_df[
            "candidate"
        ].unique()
    )
    != required_models
):
    raise ValueError(
        "Unexpected model set in Brazil Script 19 run metrics."
    )

test_dates = (
    mean_prediction_df[
        "target_date"
    ]
    .drop_duplicates()
    .sort_values()
)

if len(
    test_dates
) != EXPECTED_TEST_QUARTERS:
    raise RuntimeError(
        f"Expected {EXPECTED_TEST_QUARTERS} Brazil test quarters, "
        f"found {len(test_dates)}."
    )

# ============================================================
# PAIRED SEED-LEVEL COMPARISON
# ============================================================
q0_runs = (
    run_df[
        run_df[
            "candidate"
        ] == Q0
    ][
        [
            "seed",
            "RMSE",
            "MAE",
            "sMAPE",
            "direction_accuracy_pct",
            "mean_error_bias",
        ]
    ]
    .rename(
        columns={
            "RMSE":
                "Q0_RMSE",
            "MAE":
                "Q0_MAE",
            "sMAPE":
                "Q0_sMAPE",
            "direction_accuracy_pct":
                "Q0_direction_accuracy_pct",
            "mean_error_bias":
                "Q0_mean_error_bias",
        }
    )
)

o1_runs = (
    run_df[
        run_df[
            "candidate"
        ] == O1
    ][
        [
            "seed",
            "RMSE",
            "MAE",
            "sMAPE",
            "direction_accuracy_pct",
            "mean_error_bias",
        ]
    ]
    .rename(
        columns={
            "RMSE":
                "O1_RMSE",
            "MAE":
                "O1_MAE",
            "sMAPE":
                "O1_sMAPE",
            "direction_accuracy_pct":
                "O1_direction_accuracy_pct",
            "mean_error_bias":
                "O1_mean_error_bias",
        }
    )
)

paired_runs = q0_runs.merge(
    o1_runs,
    on="seed",
    how="inner",
)

if len(
    paired_runs
) != 10:
    raise RuntimeError(
        "Expected 10 matched Brazil seeds."
    )

for metric in [
    "RMSE",
    "MAE",
    "sMAPE",
    "direction_accuracy_pct",
    "mean_error_bias",
]:
    paired_runs[
        f"O1_minus_Q0_{metric}"
    ] = (
        paired_runs[
            f"O1_{metric}"
        ]
        - paired_runs[
            f"Q0_{metric}"
        ]
    )

paired_runs[
    "O1_RMSE_better"
] = (
    paired_runs[
        "O1_RMSE"
    ]
    < paired_runs[
        "Q0_RMSE"
    ]
)

paired_runs[
    "O1_MAE_better"
] = (
    paired_runs[
        "O1_MAE"
    ]
    < paired_runs[
        "Q0_MAE"
    ]
)

paired_runs.to_csv(
    PAIRED_RUN_OUT,
    index=False,
)

paired_seed_summary = pd.DataFrame([
    {
        "paired_seeds":
            len(
                paired_runs
            ),
        "mean_RMSE_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_RMSE"
            ].mean(),
        "median_RMSE_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_RMSE"
            ].median(),
        "O1_RMSE_wins":
            int(
                paired_runs[
                    "O1_RMSE_better"
                ].sum()
            ),
        "mean_MAE_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_MAE"
            ].mean(),
        "median_MAE_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_MAE"
            ].median(),
        "O1_MAE_wins":
            int(
                paired_runs[
                    "O1_MAE_better"
                ].sum()
            ),
        "mean_sMAPE_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_sMAPE"
            ].mean(),
        "mean_direction_accuracy_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_direction_accuracy_pct"
            ].mean(),
        "mean_bias_difference_O1_minus_Q0":
            paired_runs[
                "O1_minus_Q0_mean_error_bias"
            ].mean(),
    }
])

paired_seed_summary.to_csv(
    PAIRED_SUMMARY_OUT,
    index=False,
)

# ============================================================
# QUARTER-LEVEL COMPARISON USING 10-SEED MEAN FORECASTS
# ============================================================
q0_mean = (
    mean_prediction_df[
        mean_prediction_df[
            "candidate"
        ] == Q0
    ][
        [
            "target_date",
            "actual",
            "mean_prediction",
        ]
    ]
    .rename(
        columns={
            "mean_prediction":
                "Q0_prediction",
        }
    )
)

o1_mean = (
    mean_prediction_df[
        mean_prediction_df[
            "candidate"
        ] == O1
    ][
        [
            "target_date",
            "actual",
            "mean_prediction",
        ]
    ]
    .rename(
        columns={
            "actual":
                "O1_actual",
            "mean_prediction":
                "O1_prediction",
        }
    )
)

quarter_df = q0_mean.merge(
    o1_mean,
    on="target_date",
    how="inner",
)

if not np.allclose(
    quarter_df[
        "actual"
    ],
    quarter_df[
        "O1_actual"
    ],
):
    raise RuntimeError(
        "Q0 and O1 Brazil target values do not match."
    )

quarter_df = (
    quarter_df.drop(
        columns=[
            "O1_actual",
        ]
    )
    .sort_values(
        "target_date"
    )
    .reset_index(
        drop=True
    )
)

quarter_df[
    "Q0_error"
] = (
    quarter_df[
        "Q0_prediction"
    ]
    - quarter_df[
        "actual"
    ]
)

quarter_df[
    "O1_error"
] = (
    quarter_df[
        "O1_prediction"
    ]
    - quarter_df[
        "actual"
    ]
)

quarter_df[
    "Q0_squared_error"
] = (
    quarter_df[
        "Q0_error"
    ] ** 2
)

quarter_df[
    "O1_squared_error"
] = (
    quarter_df[
        "O1_error"
    ] ** 2
)

quarter_df[
    "Q0_absolute_error"
] = np.abs(
    quarter_df[
        "Q0_error"
    ]
)

quarter_df[
    "O1_absolute_error"
] = np.abs(
    quarter_df[
        "O1_error"
    ]
)

quarter_df[
    "squared_loss_difference_O1_minus_Q0"
] = (
    quarter_df[
        "O1_squared_error"
    ]
    - quarter_df[
        "Q0_squared_error"
    ]
)

quarter_df[
    "absolute_loss_difference_O1_minus_Q0"
] = (
    quarter_df[
        "O1_absolute_error"
    ]
    - quarter_df[
        "Q0_absolute_error"
    ]
)

quarter_df[
    "O1_squared_error_better"
] = (
    quarter_df[
        "O1_squared_error"
    ]
    < quarter_df[
        "Q0_squared_error"
    ]
)

quarter_df[
    "O1_absolute_error_better"
] = (
    quarter_df[
        "O1_absolute_error"
    ]
    < quarter_df[
        "Q0_absolute_error"
    ]
)

quarter_df.to_csv(
    QUARTER_LOSS_OUT,
    index=False,
)

# ============================================================
# MOVING-BLOCK BOOTSTRAP
# ============================================================
def moving_block_bootstrap_mean(
    values,
    block_length=4,
    repetitions=10000,
    seed=2026,
):
    """
    Circular moving-block bootstrap.

    Resamples contiguous blocks to preserve short-run serial dependence.
    Statistic = mean O1-Q0 loss difference.
    Negative values favour O1.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    if (
        block_length < 1
        or block_length > n
    ):
        raise ValueError(
            "Invalid block length."
        )

    rng = np.random.default_rng(
        seed
    )

    n_blocks = int(
        np.ceil(
            n
            / block_length
        )
    )

    boot_means = np.empty(
        repetitions,
        dtype=float,
    )

    base_index = np.arange(
        block_length
    )

    for b in range(
        repetitions
    ):
        sample_values = []

        starts = rng.integers(
            0,
            n,
            size=n_blocks,
        )

        for start in starts:
            idx = (
                start
                + base_index
            ) % n

            sample_values.extend(
                values[
                    idx
                ]
            )

        sample = np.asarray(
            sample_values[
                :n
            ],
            dtype=float,
        )

        boot_means[
            b
        ] = sample.mean()

    observed = float(
        values.mean()
    )

    return {
        "observed_mean_difference":
            observed,
        "ci_2_5":
            float(
                np.percentile(
                    boot_means,
                    2.5,
                )
            ),
        "ci_97_5":
            float(
                np.percentile(
                    boot_means,
                    97.5,
                )
            ),
        "bootstrap_probability_difference_less_than_zero":
            float(
                np.mean(
                    boot_means < 0
                )
            ),
        "bootstrap_probability_difference_greater_than_zero":
            float(
                np.mean(
                    boot_means > 0
                )
            ),
    }


bootstrap_rows = []

for loss_name, column in [
    (
        "squared_error",
        "squared_loss_difference_O1_minus_Q0",
    ),
    (
        "absolute_error",
        "absolute_loss_difference_O1_minus_Q0",
    ),
]:
    result = (
        moving_block_bootstrap_mean(
            quarter_df[
                column
            ],
            block_length=
                BLOCK_LENGTH,
            repetitions=
                BOOTSTRAP_REPS,
            seed=
                BOOTSTRAP_SEED,
        )
    )

    bootstrap_rows.append({
        "loss":
            loss_name,
        "interpretation":
            "negative favours O1",
        "test_quarters":
            len(
                quarter_df
            ),
        "block_length":
            BLOCK_LENGTH,
        "bootstrap_repetitions":
            BOOTSTRAP_REPS,
        **result,
    })

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    BOOTSTRAP_OUT,
    index=False,
)

# ============================================================
# FULL-PERIOD + COVID DIAGNOSTIC
# ============================================================
def forecast_metrics(
    actual,
    prediction,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = (
        prediction
        - actual
    )

    denominator = (
        np.abs(
            actual
        )
        + np.abs(
            prediction
        )
    )

    smape_terms = np.zeros_like(
        denominator,
        dtype=float,
    )

    valid = (
        denominator > 1e-8
    )

    smape_terms[
        valid
    ] = (
        200
        * np.abs(
            error[
                valid
            ]
        )
        / denominator[
            valid
        ]
    )

    return {
        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        error
                        ** 2
                    )
                )
            ),
        "MAE":
            float(
                np.mean(
                    np.abs(
                        error
                    )
                )
            ),
        "sMAPE":
            float(
                smape_terms.mean()
            ),
        "direction_accuracy_pct":
            float(
                100
                * np.mean(
                    np.sign(
                        actual
                    )
                    == np.sign(
                        prediction
                    )
                )
            ),
        "mean_error_bias":
            float(
                np.mean(
                    error
                )
            ),
    }


covid_mask = (
    quarter_df[
        "target_date"
    ]
    .between(
        COVID_START,
        COVID_END,
    )
)

non_covid_mask = (
    ~covid_mask
)

diagnostic_rows = []

for period_name, period_mask in [
    (
        "full_test",
        np.ones(
            len(
                quarter_df
            ),
            dtype=bool,
        ),
    ),
    (
        "covid_diagnostic",
        covid_mask.to_numpy(),
    ),
    (
        "non_covid_diagnostic",
        non_covid_mask.to_numpy(),
    ),
]:
    for candidate, pred_col in [
        (
            Q0,
            "Q0_prediction",
        ),
        (
            O1,
            "O1_prediction",
        ),
    ]:
        score = forecast_metrics(
            quarter_df.loc[
                period_mask,
                "actual",
            ],
            quarter_df.loc[
                period_mask,
                pred_col,
            ],
        )

        diagnostic_rows.append({
            "period":
                period_name,
            "candidate":
                candidate,
            "quarters":
                int(
                    np.sum(
                        period_mask
                    )
                ),
            "start_date":
                quarter_df.loc[
                    period_mask,
                    "target_date",
                ].min(),
            "end_date":
                quarter_df.loc[
                    period_mask,
                    "target_date",
                ].max(),
            **score,
        })

diagnostic_df = pd.DataFrame(
    diagnostic_rows
)

diagnostic_df.to_csv(
    DIAGNOSTIC_OUT,
    index=False,
)

# ============================================================
# SUMMARY
# ============================================================
quarter_squared_wins = int(
    quarter_df[
        "O1_squared_error_better"
    ].sum()
)

quarter_absolute_wins = int(
    quarter_df[
        "O1_absolute_error_better"
    ].sum()
)

summary = {
    "country":
        "Brazil",
    "primary_period":
        "2019Q1-2025Q4",
    "paired_seed_results": {
        "paired_seeds":
            int(
                len(
                    paired_runs
                )
            ),
        "mean_RMSE_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_RMSE"
                ].mean()
            ),
        "median_RMSE_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_RMSE"
                ].median()
            ),
        "O1_RMSE_wins":
            int(
                paired_runs[
                    "O1_RMSE_better"
                ].sum()
            ),
        "mean_MAE_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_MAE"
                ].mean()
            ),
        "median_MAE_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_MAE"
                ].median()
            ),
        "O1_MAE_wins":
            int(
                paired_runs[
                    "O1_MAE_better"
                ].sum()
            ),
        "mean_sMAPE_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_sMAPE"
                ].mean()
            ),
        "mean_direction_accuracy_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_direction_accuracy_pct"
                ].mean()
            ),
        "mean_bias_difference_O1_minus_Q0":
            float(
                paired_runs[
                    "O1_minus_Q0_mean_error_bias"
                ].mean()
            ),
    },
    "quarter_level_results": {
        "test_quarters":
            int(
                len(
                    quarter_df
                )
            ),
        "O1_squared_error_wins":
            quarter_squared_wins,
        "Q0_squared_error_wins":
            int(
                len(
                    quarter_df
                )
                - quarter_squared_wins
            ),
        "O1_absolute_error_wins":
            quarter_absolute_wins,
        "Q0_absolute_error_wins":
            int(
                len(
                    quarter_df
                )
                - quarter_absolute_wins
            ),
        "mean_squared_loss_difference_O1_minus_Q0":
            float(
                quarter_df[
                    "squared_loss_difference_O1_minus_Q0"
                ].mean()
            ),
        "mean_absolute_loss_difference_O1_minus_Q0":
            float(
                quarter_df[
                    "absolute_loss_difference_O1_minus_Q0"
                ].mean()
            ),
    },
    "moving_block_bootstrap": {
        "block_length":
            BLOCK_LENGTH,
        "repetitions":
            BOOTSTRAP_REPS,
        "note":
            "Negative O1-Q0 loss difference favours RAOT/O1.",
    },
    "covid_diagnostic": {
        "window_start":
            str(
                COVID_START.date()
            ),
        "window_end":
            str(
                COVID_END.date()
            ),
        "primary_result_remains_full_test":
            True,
    },
    "cross_country_interpretation_rule":
        (
            "Brazil is an external country-level generalisation test of the frozen "
            "representation. Do not make the primary SA-vs-Brazil comparison until "
            "South Africa is rerun on the same common 12-channel feature set."
        ),
}

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

# ============================================================
# FIGURES
# ============================================================
fig, ax = plt.subplots(
    figsize=(
        10,
        5,
    )
)

ax.plot(
    quarter_df[
        "target_date"
    ],
    quarter_df[
        "squared_loss_difference_O1_minus_Q0"
    ],
    marker="o",
)

ax.axhline(
    0,
    linewidth=0.8,
)

ax.set_title(
    "Brazil Quarter-Level Squared-Loss Difference: O1 - Q0"
)

ax.set_xlabel(
    "Target quarter"
)

ax.set_ylabel(
    "Squared-error difference"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "20_brazil_quarter_squared_loss_difference.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(
    fig
)

fig, ax = plt.subplots(
    figsize=(
        10,
        5,
    )
)

ax.plot(
    quarter_df[
        "target_date"
    ],
    quarter_df[
        "absolute_loss_difference_O1_minus_Q0"
    ],
    marker="o",
)

ax.axhline(
    0,
    linewidth=0.8,
)

ax.set_title(
    "Brazil Quarter-Level Absolute-Loss Difference: O1 - Q0"
)

ax.set_xlabel(
    "Target quarter"
)

ax.set_ylabel(
    "Absolute-error difference"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "20_brazil_quarter_absolute_loss_difference.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(
    fig
)

# ============================================================
# CONSOLE SUMMARY
# ============================================================
print(
    "\n"
    + "=" * 78
)

print(
    "20 - BRAZIL Q0 VS RAOT/O1 STATISTICAL EVALUATION"
)

print(
    "=" * 78
)

print(
    "\nPaired seed-level O1 vs Q0:\n"
)

print(
    paired_seed_summary.to_string(
        index=False
    )
)

print(
    "\nQuarter-level mean-forecast comparison:"
)

print(
    f"  Test quarters: {len(quarter_df)}"
)

print(
    f"  O1 squared-error wins: "
    f"{quarter_squared_wins}/{len(quarter_df)}"
)

print(
    f"  O1 absolute-error wins: "
    f"{quarter_absolute_wins}/{len(quarter_df)}"
)

print(
    "  Mean squared-loss difference (O1-Q0): "
    f"{quarter_df['squared_loss_difference_O1_minus_Q0'].mean():+.6f}"
)

print(
    "  Mean absolute-loss difference (O1-Q0): "
    f"{quarter_df['absolute_loss_difference_O1_minus_Q0'].mean():+.6f}"
)

print(
    "\nMoving-block bootstrap 95% CIs:\n"
)

print(
    bootstrap_df.to_string(
        index=False
    )
)

print(
    "\nFull-period and COVID diagnostic:\n"
)

print(
    diagnostic_df[
        [
            "period",
            "candidate",
            "quarters",
            "RMSE",
            "MAE",
            "sMAPE",
            "direction_accuracy_pct",
            "mean_error_bias",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nInterpretation rule:"
)

print(
    "  - Full 2019Q1-2025Q4 Brazil performance is the PRIMARY country result."
)

print(
    "  - COVID/non-COVID results are secondary diagnostics only."
)

print(
    "  - A bootstrap 95% CI entirely below zero supports lower O1 loss."
)

print(
    "  - If the CI crosses zero, report the O1 mean improvement as "
    "statistically inconclusive at the 95% level."
)

print(
    "\nOutputs:"
)

print(
    f"  {PAIRED_RUN_OUT}"
)

print(
    f"  {PAIRED_SUMMARY_OUT}"
)

print(
    f"  {QUARTER_LOSS_OUT}"
)

print(
    f"  {BOOTSTRAP_OUT}"
)

print(
    f"  {DIAGNOSTIC_OUT}"
)

print(
    f"  {SUMMARY_OUT}"
)

print(
    "\nfigures:"
)

print(
    f"  {FIGURE_DIR}"
)

if __name__ == "__main__":
    pass
