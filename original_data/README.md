# Original Data

This folder contains the original pre-concatenation CSV bundles used by the legacy analyses.

Contents:
- `chapter_3/csvs`: 16 Chapter 3 per-image CSVs from the legacy Chapter 3 pipeline.
- `pkm2/csvs`: the 5 non-`partial` PKM2 per-cell CSVs used by the Chapter 4 PKM2 notebooks.
- `ppkm2/csvs`: the 5 non-`partial` pPKM2 per-cell CSVs used by the Chapter 4 pPKM2 notebooks.
- `calcium/out_registration_batch/verify_inputs.csv`: original bundle-to-source-file map from the calcium registration batch run.
- `calcium/out_registration_batch/*/matched_cells_with_stain_metrics.csv`: 24 bundle-level merged calcium/stain CSVs before concatenation.
- `calcium/out_registration_batch/*/calcium_metrics.csv`: 24 per-bundle calcium summary metric tables.
- `calcium/out_registration_batch/*/calcium_timeseries_long.csv`: 24 per-bundle long-format dF/F0 time-series tables.
- `caspase/csvs`: raw caspase stain-area CSVs from the legacy caspase workflow.

Notes:
- Chapter 3 includes legacy convexity-column mismatches across files. Validation uses the same `harmonise_convexity_columns` logic as the old Chapter 3 scripts before checking agreement.
- PKM2 and pPKM2 intentionally include only the non-`partial` CSVs because the legacy notebooks filtered out `*_partial.csv`.
- The packaged runnable caspase example is a reviewer-facing summary table. The raw caspase CSVs here are the upstream source files for that workflow.
- The calcium bundle now includes the full CSV inputs needed to rerun clustering, figures, dF/F0 trace summaries, and the raw nested AUC LMM workflow directly from `run_calcium_github.py`.
- Use `python build_parsed_data_from_original_data.py` from the `_github` folder to rebuild the non-calcium parsed tables used by the examiner-facing Chapter 3, PKM2, pPKM2, and caspase scripts.

Validation outputs are written to:
- `validation/original_data_validation_summary.csv`
- `validation/original_data_validation_report.md`
