# Original Data

This folder contains the source CSV bundles used to rebuild the Chapter 3 and Chapter 4 analyses in this repository.

Contents:
- `chapter_3/csvs`: 16 Chapter 3 per-image CSVs for the 40x differentiation assay.
- `pkm2/csvs`: the non-`partial` PKM2 per-cell CSV exports used by the Chapter 4 PKM2 workflow.
- `ppkm2/csvs`: the non-`partial` pPKM2 per-cell CSV exports used by the Chapter 4 pPKM2 workflow.
- `calcium/out_registration_batch/verify_inputs.csv`: file manifest for the calcium registration batch.
- `calcium/out_registration_batch/*/matched_cells_with_stain_metrics.csv`: bundle-level merged calcium and stain tables.
- `calcium/out_registration_batch/*/calcium_metrics.csv`: per-bundle calcium summary metric tables.
- `calcium/out_registration_batch/*/calcium_timeseries_long.csv`: per-bundle long-format dF/F0 time-series tables.
- `caspase/csvs`: raw caspase stain-area CSVs used to rebuild the caspase phenotype summaries.

Notes:
- The Chapter 3 source files use two equivalent convexity column names; the rebuild scripts standardize them before concatenation.
- PKM2 and pPKM2 include the non-`partial` CSV exports used in the final analysis bundle.
- The calcium bundle includes the matched-cell, summary-metric, and time-series CSVs required by `run_calcium_github.py`.
- The caspase CSVs are the per-cell source tables used to rebuild `parsed_data/caspase/caspase_summary_table.csv`.
- You can rebuild the non-calcium parsed tables in batch with `python build_parsed_data_from_original_data.py`, or simply run the analysis scripts directly and let them rebuild the required inputs when needed.

Validation outputs are written to:
- `validation/original_data_validation_summary.csv`
- `validation/original_data_validation_report.md`
