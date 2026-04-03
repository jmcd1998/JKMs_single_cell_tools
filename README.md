# GitHub Bundle
Authorship: 
- Code was conceptualized and largely written by Jack McDonald. Code was refactored, and at times written following verbal instructions by ChatGPT codex models. All code checked by human Author and emphasis placed on thorough bug checks in CSV outputs. 

Variant note:
- This cloned bundle is identical to `_github` except that Chapter 3, PKM2, and pPKM2 cluster-composition inference uses biological-replicate weighted cluster fractions with repeated-measures ANOVA instead of chi-square tests on pooled cell counts.
- The saved post-hoc table in this RM-ANOVA bundle is restricted to the pre-planned Vehicle comparison for the MBP-high cluster:
  - Chapter 3: `MBP`
  - Chapter 4 PKM2 / pPKM2: `MBP/O4`
- For these planned contrast tables, raw `p_value` is the primary inferential result and `fdr_p_value` is retained as a secondary sensitivity check.
- Family sizes are small:
  - Chapter 3 MBP planned contrasts: 3 tests total (`Pranlukast`, `MDL29951`, `HAMI3379` vs `Vehicle`)
  - Chapter 4 PKM2 / pPKM2 MBP-high planned contrasts: 5 tests total (all non-Vehicle treatments vs `Vehicle`)

Contents:
- `clustered_stripped_lmm_analysis.py`: shared Chapter 3 / PKM2 / pPKM2 LMM runner.
- `build_parsed_data_from_original_data.py`: rebuilds the examiner-facing `parsed_data/` tables from the bundled source files in `original_data/`.
- `run_chapter_3_github.py`: Chapter 3 raw-scale stripped pipeline on `parsed_data/chapter_3/per_cell_stats_all.csv`.
- `run_pkm2_github.py`: Chapter 4 PKM2 raw-scale stripped pipeline on `parsed_data/pkm2/per_cell_stats_all.csv`.
- `run_ppkm2_github.py`: Chapter 4 pPKM2 raw-scale stripped pipeline on `parsed_data/ppkm2/per_cell_stats_all.csv`.
- `run_calcium_github.py`: full calcium clustering, plotting, trace, and nested raw-AUC LMM pipeline on bundled original calcium CSVs, including `stats/` model-report CSV exports.
- `calcium_auc_lmm_observation_summary_github.py`: calcium nested AUC-only reviewer summary on the clustered table produced by `run_calcium_github.py`.
- `calcium_auc_lmm_observation_summary_github.ipynb`: notebook front-end for the calcium AUC-only summary.
- `make_calcium_thesis_tables.py`: builds the thesis-ready calcium MixedLM effect table nested by stimulation window and effect section.
- `run_caspase_biolrep_stats_github.py`: caspase replicate-aware repeated-measures stats on `parsed_data/caspase/caspase_summary_table.csv`.
- `validate_original_data_bundle.py`: optional legacy validation helper for checking the bundled `original_data/` against a local `example_data/` reference set when available.
- `smoke_test_calcium_github_full.py`: calcium full-pipeline smoke test against the original local reference outputs.

Parsed data built from source files:
- `parsed_data/chapter_3/per_cell_stats_all.csv`
- `parsed_data/pkm2/per_cell_stats_all.csv`
- `parsed_data/ppkm2/per_cell_stats_all.csv`
- `parsed_data/caspase/caspase_cell_data_legacy_filtered.csv`
- `parsed_data/caspase/caspase_summary_table.csv`
- `parsed_data/parsed_data_build_summary.csv`

Original raw data:
- `original_data/chapter_3/csvs/*.csv`
- `original_data/pkm2/csvs/*.csv`
- `original_data/ppkm2/csvs/*.csv`
- `original_data/calcium/out_registration_batch/verify_inputs.csv`
- `original_data/calcium/out_registration_batch/*/matched_cells_with_stain_metrics.csv`
- `original_data/calcium/out_registration_batch/*/calcium_metrics.csv`
- `original_data/calcium/out_registration_batch/*/calcium_timeseries_long.csv`
- `original_data/caspase/csvs/*.csv`

Outputs are written into the local `outputs/` subfolders beside these scripts.
Parsed-data rebuild outputs are written into `parsed_data/`.
Validation outputs are written into `original_data/validation/`.
These generated folders are not tracked in git.

From-scratch workflow from inside this folder:
- `python build_parsed_data_from_original_data.py`
- `python run_chapter_3_github.py`
- `python run_pkm2_github.py`
- `python run_ppkm2_github.py`
- `python run_caspase_biolrep_stats_github.py`
- `python run_calcium_github.py`
- `python calcium_auc_lmm_observation_summary_github.py`
- `python make_calcium_thesis_tables.py`
- `python smoke_test_calcium_github_full.py`
- `python make_calcium_auc_lmm_observation_summary_github_notebook.py`

Notes:
- Run `build_parsed_data_from_original_data.py` before Chapter 3, PKM2, pPKM2, or caspase analyses.
- Calcium does not use `parsed_data`; it reads directly from `original_data/calcium/out_registration_batch`.
- `calcium_auc_lmm_observation_summary_github.py` expects the clustered calcium table written by `run_calcium_github.py`.
- `run_calcium_github.py` now writes MixedLM report CSVs into `outputs/calcium_full_output_V1/stats`.
- `make_calcium_thesis_tables.py` expects those calcium `stats/` outputs and writes into `outputs/thesis_tables`.
- `example_data/` is optional, not tracked in git, and is not used by the main from-scratch runners.
- `validate_original_data_bundle.py` is therefore optional and only relevant if you have a local `example_data/` reference bundle available.

The generated notebook is:
- `calcium_auc_lmm_observation_summary_github.ipynb`
