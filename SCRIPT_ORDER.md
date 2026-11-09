# Script order and role

## South Africa development
1. `04_data_harmonisation.py` — convert extracted source series to quarterly form.
2. `05_feature_engineering_updated.py` — construct the engineered macroeconomic feature matrix and target.
3. `06_exploratory_feature_analysis.py` — coverage, stationarity and correlation diagnostics.
4. `07_experiment_design.py` — chronological split and sequence feasibility.
5. `08_model_data_preparation.py` — training-only scaling and sequence construction.
6. `09_standard_transformer.py` — conventional Transformer baseline development.
7. `10g_release_availability_metadata.py` — operational publication-lag metadata.
8. `10i_tokenisation_ablation.py` — matched pre-test Q0 versus observation-token comparison.
9. `11b_final_observation_token_evaluation.py` — frozen final South African country-rich Q0/RAOT evaluation.
10. `12b_observation_token_statistical_evaluation.py` — seed-level and quarter-level statistical evaluation.

## Brazil independent evaluation
11. `13_brazil_data_inventory.py`
12. `14_brazil_source_selection_extraction.py`
13. `15_brazil_validation_harmonisation_FIXED.py`
14. `16_brazil_feature_engineering.py`
15. `17_brazil_common_release_alignment.py`
16. `18_brazil_model_data_preparation_FIXED_ORDER.py`
17. `19_brazil_frozen_q0_o1_evaluation.py`
18. `20_brazil_q0_o1_statistical_evaluation.py`

## Matched common-12 South Africa / cross-country comparison
19. `21_sa_common12_model_data_preparation.py`
20. `22_sa_common12_frozen_q0_o1_evaluation.py`
21. `23_sa_common12_q0_o1_statistical_evaluation.py`
22. `24_cross_country_generalisability.py`

The repository includes the final/relevant development scripts rather than every rejected architecture experiment. Rejected variants can be archived in a separate `experiments_archive/` release if required for dissertation provenance, but they are not necessary to reproduce the paper's final claims.
