# Reproducibility release checklist

Before making the GitHub repository public and inserting its URL into the MDPI manuscript:

- [ ] Copy the exact processed/model-ready files from the original experiment environment into `data/processed/`.
- [ ] Copy the original final seed-level result CSVs and quarter-level loss files into `results/original_outputs/`.
- [ ] Run `python tools/capture_environment.py` in the original experiment environment.
- [ ] Run `python -m pip freeze > environment/requirements-lock.txt` in that same environment.
- [ ] Confirm the ten fixed seeds are `11, 22, 33, 44, 55, 66, 77, 88, 99, 111`.
- [ ] Confirm Q0 uses 4 frozen epochs and RAOT/O1 uses 10 frozen epochs.
- [ ] Re-run the final pipeline and compare its generated summary to `results/reported_common12_model_performance.csv` and `results/reported_bootstrap_summary.csv`.
- [ ] Verify all figures are generated at 300 dpi.
- [ ] Review raw-data redistribution rights; do not commit third-party downloads unless permitted.
- [ ] Choose a software licence and replace `LICENSE_NOTE.md` with a `LICENSE` file.
- [ ] Create the GitHub repository and update `CITATION.cff` with the repository URL.
- [ ] Create a tagged release (for example `v1.0.0-paper`) after the manuscript analysis is frozen.
- [ ] Archive that release in a DOI-capable repository if desired and insert the version/DOI in the manuscript's Data Availability Statement.
- [ ] Ensure no API keys, passwords, personal paths or confidential files are present.
