# Release-Aligned Observation Tokenisation (RAOT) for Macroeconomic Forecasting

Reproducibility package for the manuscript **“Release-Aligned Observation Tokenisation for Transformer-Based Macroeconomic Forecasting: Evidence from South Africa and Brazil.”**

## What this repository reproduces

The study forecasts one-quarter-ahead real GDP quarter-on-quarter growth with an eight-quarter lookback. It compares:

- **Q0:** a release-aligned quarter-token Transformer with one multivariate token per quarter; and
- **RAOT/O1:** a release-aligned observation-token Transformer where every feature-time observation becomes a token containing value, feature identity and temporal position.

South Africa is used for architecture development and final evaluation. Brazil is an independent cross-country evaluation. The final direct comparison uses the same common 12 conceptual channels, frozen architecture/training settings and 2019Q1–2025Q4 test period.

## Repository layout

```text
.
├── README.md
├── SCRIPT_ORDER.md
├── REPRODUCIBILITY_CHECKLIST.md
├── CITATION.cff
├── LICENSE_NOTE.md
├── run_final_pipeline.py
├── config/
│   └── final_model_config.json
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── south_africa/
│   │   └── brazil/
│   ├── interim/
│   ├── processed/
│   └── metadata/
├── environment/
│   ├── README.md
│   └── requirements.txt
├── figures/
├── results/
├── src/
└── tools/
    └── capture_environment.py
```

## Fixed final design

The frozen compact Transformer uses `d_model=16`, 2 attention heads, one encoder layer, feed-forward dimension 32, dropout 0.20, AdamW with learning rate `1e-3` and weight decay `1e-4`, batch size 16, gradient clipping at 1.0 and MSE training loss. Q0 is refit for 4 epochs and RAOT/O1 for 10 epochs. Final evaluation uses ten fixed seeds: `11, 22, 33, 44, 55, 66, 77, 88, 99, 111`.

See `config/final_model_config.json` for the machine-readable version.

## Data availability and release alignment

Underlying macroeconomic data come from public statistical/economic sources. Source portals and the Brazil raw-file manifest are in `data/metadata/`. Raw third-party downloads are not bundled automatically because redistribution conditions differ across providers.

Release alignment is implemented with conservative operational publication-lag proxies. It is **not** a historical real-time-vintage database. The exact South African and Brazilian release-shift assumptions are recorded in:

- `data/metadata/sa_release_shift_metadata.csv`
- `data/metadata/brazil_release_shift_metadata.csv`

## Quick start

Create a clean environment and install the imported packages:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r environment/requirements.txt
```

For a publication-grade reproduction, use the exact `requirements-lock.txt` captured from the original experiment environment rather than the unpinned file above.

After placing the archived processed/model-ready files in `data/processed/`, run:

```bash
python run_final_pipeline.py
```

The final pipeline runs the Brazil frozen evaluation, the matched South African common-12 evaluation and the cross-country synthesis.

## Reported checks

`results/reported_common12_model_performance.csv` and `results/reported_bootstrap_summary.csv` contain the values reported in the current manuscript and can be used as regression checks after rerunning the final pipeline.

## Important release status

This package contains the actual final analysis scripts available in the project archive and has been converted from a machine-specific Windows path to repository-relative paths. However, two items still need to be copied from the original experiment environment before the repository should be described as fully reproducible:

1. the exact generated processed/model-ready CSV/NPZ and original seed-level result files; and
2. the exact package/version lock from the environment that produced the reported results.

See `REPRODUCIBILITY_CHECKLIST.md` and `data/processed/README.md`.

## Citation

Use `CITATION.cff`. Replace the placeholder repository URL and publication DOI once available.

## Licence

A licence has deliberately not been selected automatically. See `LICENSE_NOTE.md` before making the repository public.
