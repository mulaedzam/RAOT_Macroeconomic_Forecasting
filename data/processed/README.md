# Processed data artefacts to add from the archived experiment

The saved ChatGPT/Library project files contained the analysis scripts but not the original generated model-ready CSV/NPZ outputs as standalone files. Copy the exact originals from the experiment environment before the repository is made public.

At minimum, preserve the final artefacts used by the frozen evaluations, including:

## South Africa
- `features/engineered_features_target_window.csv`
- `availability/feature_release_metadata.csv`
- `model_data/B_expanded_refined_input_features.csv`
- country-rich model-data artefacts required by scripts 10i/11b/12b
- `south_africa_common12/` outputs created by script 21 (aligned CSV, release map, NPZ, train/test indexes, scaler/config files)

## Brazil
- `brazil_harmonised/brazil_quarterly_harmonised.csv`
- `brazil_features/brazil_engineered_features.csv`
- `brazil_release_alignment/brazil_common_release_aligned.csv`
- `brazil_model_data/brazil_q0_o1_model_data.npz`
- `brazil_model_data/brazil_train_sequence_index.csv`
- `brazil_model_data/brazil_test_sequence_index.csv`
- `brazil_model_data/brazil_training_scalers.json`
- `brazil_model_data/brazil_model_data_config.json`

Do not regenerate and silently substitute different vintages if the purpose is to reproduce the published numbers. Archive the exact files used for the reported results.
