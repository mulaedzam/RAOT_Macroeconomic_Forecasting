from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

# Brazil
BR_MODEL_SUMMARY = (
    ROOT
    / "results"
    / "brazil_final_q0_o1"
    / "19_brazil_model_summary.csv"
)

BR_PAIRED = (
    ROOT
    / "results"
    / "brazil_q0_o1_statistical_evaluation"
    / "20_brazil_paired_seed_summary.csv"
)

BR_BOOTSTRAP = (
    ROOT
    / "results"
    / "brazil_q0_o1_statistical_evaluation"
    / "20_brazil_moving_block_bootstrap.csv"
)

BR_DIAGNOSTIC = (
    ROOT
    / "results"
    / "brazil_q0_o1_statistical_evaluation"
    / "20_brazil_full_covid_diagnostic.csv"
)

# South Africa common-12
SA_MODEL_SUMMARY = (
    ROOT
    / "results"
    / "sa_common12_final_q0_o1"
    / "22_sa_common12_model_summary.csv"
)

SA_PAIRED = (
    ROOT
    / "results"
    / "sa_common12_q0_o1_statistical_evaluation"
    / "23_sa_common12_paired_seed_summary.csv"
)

SA_BOOTSTRAP = (
    ROOT
    / "results"
    / "sa_common12_q0_o1_statistical_evaluation"
    / "23_sa_common12_moving_block_bootstrap.csv"
)

SA_DIAGNOSTIC = (
    ROOT
    / "results"
    / "sa_common12_q0_o1_statistical_evaluation"
    / "23_sa_common12_full_covid_diagnostic.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "cross_country_generalisability"
)

FIG_DIR = (
    ROOT
    / "figures"
    / "cross_country_generalisability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PERFORMANCE_OUT = (
    OUT_DIR
    / "24_cross_country_model_performance.csv"
)

STATISTICAL_OUT = (
    OUT_DIR
    / "24_cross_country_statistical_evidence.csv"
)

DIAGNOSTIC_OUT = (
    OUT_DIR
    / "24_cross_country_covid_diagnostic.csv"
)

SUMMARY_OUT = (
    OUT_DIR
    / "24_cross_country_generalisability_summary.csv"
)

SUMMARY_JSON_OUT = (
    OUT_DIR
    / "24_cross_country_generalisability_summary.json"
)

FIGURE_DPI = 300

Q0 = "Q0_aligned_quarter_token"
O1 = "O1_aligned_observation_token"

# ============================================================
# LOAD + AUDIT
# ============================================================
required_paths = [
    BR_MODEL_SUMMARY,
    BR_PAIRED,
    BR_BOOTSTRAP,
    BR_DIAGNOSTIC,
    SA_MODEL_SUMMARY,
    SA_PAIRED,
    SA_BOOTSTRAP,
    SA_DIAGNOSTIC,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )


def load_country(
    country,
    model_path,
    paired_path,
    bootstrap_path,
    diagnostic_path,
):
    model = pd.read_csv(
        model_path
    )

    paired = pd.read_csv(
        paired_path
    )

    bootstrap = pd.read_csv(
        bootstrap_path
    )

    diagnostic = pd.read_csv(
        diagnostic_path
    )

    if set(
        model["candidate"]
    ) != {
        Q0,
        O1,
    }:
        raise ValueError(
            f"{country}: unexpected model set."
        )

    if len(
        paired
    ) != 1:
        raise ValueError(
            f"{country}: expected one paired-seed summary row."
        )

    if set(
        bootstrap["loss"]
    ) != {
        "squared_error",
        "absolute_error",
    }:
        raise ValueError(
            f"{country}: unexpected bootstrap loss rows."
        )

    q0 = (
        model.loc[
            model["candidate"] == Q0
        ]
        .iloc[0]
    )

    o1 = (
        model.loc[
            model["candidate"] == O1
        ]
        .iloc[0]
    )

    paired_row = paired.iloc[
        0
    ]

    squared_boot = (
        bootstrap.loc[
            bootstrap["loss"]
            == "squared_error"
        ]
        .iloc[0]
    )

    absolute_boot = (
        bootstrap.loc[
            bootstrap["loss"]
            == "absolute_error"
        ]
        .iloc[0]
    )

    # Relative improvement:
    # positive = O1 better
    # negative = Q0 better
    rmse_improvement_pct = (
        100.0
        * (
            q0["RMSE_mean"]
            - o1["RMSE_mean"]
        )
        / q0["RMSE_mean"]
    )

    mae_improvement_pct = (
        100.0
        * (
            q0["MAE_mean"]
            - o1["MAE_mean"]
        )
        / q0["MAE_mean"]
    )

    smape_improvement_points = (
        q0["sMAPE_mean"]
        - o1["sMAPE_mean"]
    )

    direction_difference_pp = (
        o1[
            "direction_accuracy_pct_mean"
        ]
        - q0[
            "direction_accuracy_pct_mean"
        ]
    )

    squared_ci_excludes_zero = (
        (
            squared_boot["ci_2_5"] > 0
            and squared_boot["ci_97_5"] > 0
        )
        or (
            squared_boot["ci_2_5"] < 0
            and squared_boot["ci_97_5"] < 0
        )
    )

    absolute_ci_excludes_zero = (
        (
            absolute_boot["ci_2_5"] > 0
            and absolute_boot["ci_97_5"] > 0
        )
        or (
            absolute_boot["ci_2_5"] < 0
            and absolute_boot["ci_97_5"] < 0
        )
    )

    # Evidence classification deliberately avoids claiming superiority
    # when either bootstrap CI crosses zero.
    if (
        squared_boot["ci_97_5"] < 0
        and absolute_boot["ci_97_5"] < 0
    ):
        evidence_class = (
            "statistically supported O1 advantage"
        )
    elif (
        squared_boot["ci_2_5"] > 0
        and absolute_boot["ci_2_5"] > 0
    ):
        evidence_class = (
            "statistically supported Q0 advantage"
        )
    elif (
        rmse_improvement_pct > 0
        and mae_improvement_pct > 0
    ):
        evidence_class = (
            "numerical O1 advantage; statistically inconclusive"
        )
    elif (
        rmse_improvement_pct < 0
        and mae_improvement_pct < 0
    ):
        evidence_class = (
            "numerical Q0 advantage; statistically inconclusive"
        )
    else:
        evidence_class = (
            "mixed numerical evidence; statistically inconclusive"
        )

    performance_rows = []

    for _, row in model.iterrows():
        performance_rows.append({
            "country":
                country,
            "candidate":
                row[
                    "candidate"
                ],
            "RMSE_mean":
                row[
                    "RMSE_mean"
                ],
            "RMSE_std":
                row[
                    "RMSE_std"
                ],
            "MAE_mean":
                row[
                    "MAE_mean"
                ],
            "MAE_std":
                row[
                    "MAE_std"
                ],
            "sMAPE_mean":
                row[
                    "sMAPE_mean"
                ],
            "direction_accuracy_pct_mean":
                row[
                    "direction_accuracy_pct_mean"
                ],
            "mean_error_bias_mean":
                row[
                    "mean_error_bias_mean"
                ],
            "trainable_parameters":
                row[
                    "trainable_parameters"
                ],
            "fixed_epochs":
                row[
                    "fixed_epochs"
                ],
            "tokens_per_sequence":
                row[
                    "tokens_per_sequence"
                ],
        })

    statistical_row = {
        "country":
            country,
        "paired_seeds":
            int(
                paired_row[
                    "paired_seeds"
                ]
            ),
        "O1_RMSE_wins":
            int(
                paired_row[
                    "O1_RMSE_wins"
                ]
            ),
        "O1_MAE_wins":
            int(
                paired_row[
                    "O1_MAE_wins"
                ]
            ),
        "mean_RMSE_difference_O1_minus_Q0":
            paired_row[
                "mean_RMSE_difference_O1_minus_Q0"
            ],
        "mean_MAE_difference_O1_minus_Q0":
            paired_row[
                "mean_MAE_difference_O1_minus_Q0"
            ],
        "mean_sMAPE_difference_O1_minus_Q0":
            paired_row[
                "mean_sMAPE_difference_O1_minus_Q0"
            ],
        "squared_loss_mean_difference_O1_minus_Q0":
            squared_boot[
                "observed_mean_difference"
            ],
        "squared_loss_ci_2_5":
            squared_boot[
                "ci_2_5"
            ],
        "squared_loss_ci_97_5":
            squared_boot[
                "ci_97_5"
            ],
        "squared_loss_P_difference_lt_0":
            squared_boot[
                "bootstrap_probability_difference_less_than_zero"
            ],
        "absolute_loss_mean_difference_O1_minus_Q0":
            absolute_boot[
                "observed_mean_difference"
            ],
        "absolute_loss_ci_2_5":
            absolute_boot[
                "ci_2_5"
            ],
        "absolute_loss_ci_97_5":
            absolute_boot[
                "ci_97_5"
            ],
        "absolute_loss_P_difference_lt_0":
            absolute_boot[
                "bootstrap_probability_difference_less_than_zero"
            ],
        "squared_ci_excludes_zero":
            bool(
                squared_ci_excludes_zero
            ),
        "absolute_ci_excludes_zero":
            bool(
                absolute_ci_excludes_zero
            ),
    }

    summary_row = {
        "country":
            country,
        "Q0_RMSE":
            q0[
                "RMSE_mean"
            ],
        "O1_RMSE":
            o1[
                "RMSE_mean"
            ],
        "O1_RMSE_improvement_pct":
            rmse_improvement_pct,
        "Q0_MAE":
            q0[
                "MAE_mean"
            ],
        "O1_MAE":
            o1[
                "MAE_mean"
            ],
        "O1_MAE_improvement_pct":
            mae_improvement_pct,
        "Q0_sMAPE":
            q0[
                "sMAPE_mean"
            ],
        "O1_sMAPE":
            o1[
                "sMAPE_mean"
            ],
        "O1_sMAPE_improvement_points":
            smape_improvement_points,
        "direction_accuracy_difference_O1_minus_Q0_pp":
            direction_difference_pp,
        "O1_RMSE_seed_wins":
            int(
                paired_row[
                    "O1_RMSE_wins"
                ]
            ),
        "O1_MAE_seed_wins":
            int(
                paired_row[
                    "O1_MAE_wins"
                ]
            ),
        "squared_loss_bootstrap_CI":
            (
                f"[{squared_boot['ci_2_5']:.6f}, "
                f"{squared_boot['ci_97_5']:.6f}]"
            ),
        "absolute_loss_bootstrap_CI":
            (
                f"[{absolute_boot['ci_2_5']:.6f}, "
                f"{absolute_boot['ci_97_5']:.6f}]"
            ),
        "evidence_class":
            evidence_class,
    }

    diagnostic = (
        diagnostic.copy()
    )

    diagnostic.insert(
        0,
        "country",
        country,
    )

    return (
        performance_rows,
        statistical_row,
        summary_row,
        diagnostic,
    )


country_specs = [
    (
        "South Africa",
        SA_MODEL_SUMMARY,
        SA_PAIRED,
        SA_BOOTSTRAP,
        SA_DIAGNOSTIC,
    ),
    (
        "Brazil",
        BR_MODEL_SUMMARY,
        BR_PAIRED,
        BR_BOOTSTRAP,
        BR_DIAGNOSTIC,
    ),
]

all_performance = []
all_statistical = []
all_summary = []
all_diagnostics = []

for spec in country_specs:
    (
        performance_rows,
        statistical_row,
        summary_row,
        diagnostic,
    ) = load_country(
        *spec
    )

    all_performance.extend(
        performance_rows
    )

    all_statistical.append(
        statistical_row
    )

    all_summary.append(
        summary_row
    )

    all_diagnostics.append(
        diagnostic
    )

performance_df = pd.DataFrame(
    all_performance
)

statistical_df = pd.DataFrame(
    all_statistical
)

summary_df = pd.DataFrame(
    all_summary
)

diagnostic_df = pd.concat(
    all_diagnostics,
    ignore_index=True,
)

performance_df.to_csv(
    PERFORMANCE_OUT,
    index=False,
)

statistical_df.to_csv(
    STATISTICAL_OUT,
    index=False,
)

diagnostic_df.to_csv(
    DIAGNOSTIC_OUT,
    index=False,
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)

# ============================================================
# CROSS-COUNTRY INTERPRETATION
# ============================================================
both_smape_better = bool(
    (
        summary_df[
            "O1_sMAPE_improvement_points"
        ]
        > 0
    ).all()
)

all_rmse_better = bool(
    (
        summary_df[
            "O1_RMSE_improvement_pct"
        ]
        > 0
    ).all()
)

all_mae_better = bool(
    (
        summary_df[
            "O1_MAE_improvement_pct"
        ]
        > 0
    ).all()
)

any_statistically_supported = bool(
    (
        statistical_df[
            "squared_ci_excludes_zero"
        ]
        | statistical_df[
            "absolute_ci_excludes_zero"
        ]
    ).any()
)

if (
    all_rmse_better
    and all_mae_better
    and any_statistically_supported
):
    generalisability_conclusion = (
        "Evidence supports a consistent cross-country O1 advantage."
    )
elif (
    not all_rmse_better
    or not all_mae_better
):
    generalisability_conclusion = (
        "O1 does not show uniform RMSE/MAE superiority across countries; "
        "its benefit is country-dependent."
    )
else:
    generalisability_conclusion = (
        "O1 is numerically favourable across countries but the statistical "
        "evidence remains inconclusive."
    )

cross_country_summary = {
    "primary_experiment":
        (
            "Frozen Q0 versus release-aligned observation-token O1/RAOT, "
            "same 12 conceptual input channels, same 8-quarter lookback, "
            "same 2019Q1-2025Q4 test period."
        ),
    "countries":
        [
            "South Africa",
            "Brazil",
        ],
    "uniform_RMSE_improvement":
        all_rmse_better,
    "uniform_MAE_improvement":
        all_mae_better,
    "uniform_sMAPE_improvement":
        both_smape_better,
    "any_95pct_bootstrap_CI_excluding_zero":
        any_statistically_supported,
    "primary_conclusion":
        generalisability_conclusion,
    "recommended_wording":
        (
            "The release-aligned observation-token representation produced "
            "country-dependent effects. In Brazil it reduced average RMSE, MAE "
            "and sMAPE relative to the matched quarter-token Transformer, whereas "
            "in South Africa it slightly increased RMSE and MAE while reducing "
            "sMAPE. Moving-block bootstrap confidence intervals crossed zero in "
            "both countries, so neither advantage nor disadvantage was statistically "
            "conclusive at the 95% level. The evidence therefore does not support "
            "a claim of uniform cross-country superiority, although the representation "
            "remained competitive and improved relative percentage error in both settings."
        ),
}

with open(
    SUMMARY_JSON_OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        cross_country_summary,
        f,
        indent=2,
    )

# ============================================================
# 300-DPI FIGURE 1: RELATIVE RMSE / MAE IMPROVEMENT
# ============================================================
plot_df = summary_df[
    [
        "country",
        "O1_RMSE_improvement_pct",
        "O1_MAE_improvement_pct",
    ]
].copy()

x = np.arange(
    len(
        plot_df
    )
)

width = 0.35

fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)

ax.bar(
    x - width / 2,
    plot_df[
        "O1_RMSE_improvement_pct"
    ],
    width,
    label="RMSE improvement",
)

ax.bar(
    x + width / 2,
    plot_df[
        "O1_MAE_improvement_pct"
    ],
    width,
    label="MAE improvement",
)

ax.axhline(
    0,
    linewidth=0.8,
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    plot_df[
        "country"
    ]
)

ax.set_ylabel(
    "O1 improvement over Q0 (%)"
)

ax.set_title(
    "Cross-Country Relative Performance of RAOT/O1"
)

ax.legend()

fig.tight_layout()

improvement_figure = (
    FIG_DIR
    / "24_cross_country_relative_rmse_mae_improvement.png"
)

fig.savefig(
    improvement_figure,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

plt.close(
    fig
)

# ============================================================
# FIGURE 2: BOOTSTRAP LOSS DIFFERENCE CIs
# ============================================================
ci_rows = []

for _, row in statistical_df.iterrows():
    ci_rows.append({
        "label":
            f"{row['country']} - squared",
        "mean":
            row[
                "squared_loss_mean_difference_O1_minus_Q0"
            ],
        "low":
            row[
                "squared_loss_ci_2_5"
            ],
        "high":
            row[
                "squared_loss_ci_97_5"
            ],
    })

    ci_rows.append({
        "label":
            f"{row['country']} - absolute",
        "mean":
            row[
                "absolute_loss_mean_difference_O1_minus_Q0"
            ],
        "low":
            row[
                "absolute_loss_ci_2_5"
            ],
        "high":
            row[
                "absolute_loss_ci_97_5"
            ],
    })

ci_df = pd.DataFrame(
    ci_rows
)

y = np.arange(
    len(
        ci_df
    )
)

xerr = np.vstack([
    ci_df["mean"]
    - ci_df["low"],
    ci_df["high"]
    - ci_df["mean"],
])

fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)

ax.errorbar(
    ci_df["mean"],
    y,
    xerr=xerr,
    fmt="o",
    capsize=4,
)

ax.axvline(
    0,
    linewidth=0.8,
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    ci_df[
        "label"
    ]
)

ax.set_xlabel(
    "Mean O1 - Q0 loss difference"
)

ax.set_title(
    "Cross-Country Moving-Block Bootstrap 95% Confidence Intervals"
)

fig.tight_layout()

ci_figure = (
    FIG_DIR
    / "24_cross_country_bootstrap_confidence_intervals.png"
)

fig.savefig(
    ci_figure,
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
    "24 - SOUTH AFRICA VS BRAZIL COMMON-12 GENERALISABILITY"
)

print(
    "=" * 78
)

print(
    "\nPrimary cross-country summary:\n"
)

print(
    summary_df[
        [
            "country",
            "Q0_RMSE",
            "O1_RMSE",
            "O1_RMSE_improvement_pct",
            "Q0_MAE",
            "O1_MAE",
            "O1_MAE_improvement_pct",
            "Q0_sMAPE",
            "O1_sMAPE",
            "O1_sMAPE_improvement_points",
            "O1_RMSE_seed_wins",
            "O1_MAE_seed_wins",
            "evidence_class",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nBootstrap evidence:\n"
)

print(
    statistical_df[
        [
            "country",
            "squared_loss_mean_difference_O1_minus_Q0",
            "squared_loss_ci_2_5",
            "squared_loss_ci_97_5",
            "absolute_loss_mean_difference_O1_minus_Q0",
            "absolute_loss_ci_2_5",
            "absolute_loss_ci_97_5",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nCross-country decision:"
)

print(
    f"  Uniform RMSE improvement:  {all_rmse_better}"
)

print(
    f"  Uniform MAE improvement:   {all_mae_better}"
)

print(
    f"  Uniform sMAPE improvement: {both_smape_better}"
)

print(
    f"  Any 95% bootstrap CI excluding zero: "
    f"{any_statistically_supported}"
)

print(
    "\nPrimary conclusion:"
)

print(
    "  "
    + generalisability_conclusion
)

print(
    "\nRecommended dissertation wording:"
)

print(
    "  "
    + cross_country_summary[
        "recommended_wording"
    ]
)

print(
    "\nOutputs:"
)

print(
    f"  {PERFORMANCE_OUT}"
)

print(
    f"  {STATISTICAL_OUT}"
)

print(
    f"  {DIAGNOSTIC_OUT}"
)

print(
    f"  {SUMMARY_OUT}"
)

print(
    f"  {SUMMARY_JSON_OUT}"
)

print(
    "\n300-DPI figures:"
)

print(
    f"  {improvement_figure}"
)

print(
    f"  {ci_figure}"
)

if __name__ == "__main__":
    pass
