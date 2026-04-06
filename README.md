# JKM Thesis

Authorship:
- Code was conceptualized and largely written by Jack McDonald. Code was refactored, and at times written following verbal instructions by ChatGPT Codex models. All code was checked by the human author, with particular emphasis on verifying the generated CSV outputs.

Overview:
- This repository lives in `jkm_thesis` and reproduces the same final analysis outputs as `_github_clustercomp_rm_anova`.
- The main analysis entry points are written as self-contained, heavily commented scripts so each workflow can be reviewed in one place.
- The active cluster-composition workflow for Chapter 3, PKM2, and pPKM2 uses biological-replicate weighted fractions analysed with repeated-measures ANOVA.
- See `THESIS_CHAPTER_3_4_OUTPUT_MAP.md` for the Chapter 3 / Chapter 4 figure and table mapping.
- Use `compare_outputs_against_reference_bundle.py` to confirm that this repository reproduces the same public output set as the reference RM-ANOVA bundle.

Main analysis scripts:
- `run_chapter_3_github.py`: Chapter 3 40x differentiation assay used for Chapter 3 Figure 4 and Tables 2-4.
- `run_pkm2_github.py`: Chapter 4 PKM2 companion differentiation assay used alongside the Figure 5 analysis bundle.
- `run_ppkm2_github.py`: Chapter 4 pPKM2 differentiation assay used for Figure 5, Table 3, and related supplementary outputs.
- `run_calcium_github.py`: Chapter 4 calcium clustering, plotting, trace, and nested raw-AUC mixed-model workflow used for Figure 3 and Table 2.
- `run_caspase_biolrep_stats_github.py`: Chapter 4 caspase replicate-aware repeated-measures workflow used for Figure 6 and the related supplementary tables.

Image quantification pipelines:
- `image_quantification/run_chapter3.py`: original 40x Chapter 3 image-processing pipeline from LIF files through Ilastik and Cellpose to per-cell morphology and intensity CSVs.
- `image_quantification/chapter_4_PKM2_v6.py`: original Chapter 4 PKM2 single-cell image-quantification pipeline for the 5-channel signalling assay.
- `image_quantification/chapter_4_pPKM2_v6.py`: original Chapter 4 pPKM2 single-cell image-quantification pipeline for the 5-channel signalling assay.
- `image_quantification/full_pipeline_batch_v1.py`: original batch calcium registration, segmentation, and metric-extraction pipeline that builds the matched-cell and calcium summary tables used downstream.
- `image_quantification/ensheathment_get_ilastik_outputs.py`: ensheathment crop-to-ilastik export pipeline that writes segmentation TIFFs, manifests, and preview images.
- `image_quantification/5c_getmasks_caspase.ipynb`: caspase mask-generation notebook from the original image-processing workflow.

Ensheathment workflow:
- `quantify_ensheathment.py`: quantifies per-cell MBP area, soma/process split, and nanofiber colocalization from ensheathment TIFFs, ilastik outputs, and Fiji ROI zips.
- `run_ensheathment_mixedlm.py`: fits one-term treatment MixedLMs with biological replicate and field-of-view random effects for the ensheathment metrics.
- `plot_ensheathment_graphs.py`: produces treatment-ordered normalized ensheathment box plots.
- `make_ensheathment_thesis_tables.py`: writes thesis-ready ensheathment model and contrast tables from the MixedLM exports.

Supporting scripts:
- `build_parsed_data_from_original_data.py`: batch rebuild helper for the non-calcium parsed tables in `parsed_data/`.
- `make_cluster_composition_thesis_tables.py`: thesis-ready cluster-composition tables across Chapter 3, PKM2, and pPKM2.
- `make_calcium_thesis_tables.py`: thesis-ready calcium mixed-model table exports.
- `make_caspase_thesis_tables.py`: thesis-ready caspase repeated-measures and post-hoc table exports.
- `compare_outputs_against_reference_bundle.py`: checks that this repository reproduces the same public output set as the reference RM-ANOVA bundle.

Derived model terms:
- `std_effect`: fixed-effect estimate divided by the model residual SD, `sqrt(scale)`. This is a standardized effect in residual-SD units and is added to the one-term and two-term MixedLM term tables.
- `std_effect_vs_vehicle`: treatment-vs-reference contrast estimate divided by the residual SD. This is added to the one-term treatment contrast tables.
- `emm`, `ci_low`, `ci_high`: estimated marginal mean and its 95% confidence interval reconstructed from the fitted fixed effects and covariance matrix for a specified treatment or treatment-by-cluster combination. These are written to `legacy_one_term_norm_plot_emms.csv` and `two_term_lmm_emms.csv`.
- `emm_std_units`: estimated marginal mean divided by the residual SD, included alongside the EMM tables.
- `within_estimate`, `within_SE`, `within_z`, `within_p_value`: within-cluster treatment-versus-reference contrasts derived from the fitted `treatment * cluster` model by combining the treatment main effect with the matching interaction term when required.
- `std_effect_within_cluster`: `within_estimate` divided by the residual SD. This is the value shown in the within-cluster heatmaps.
- `p_stars`, `within_p_stars`: significance labels derived from raw p values.
- `fdr_p_value`, `fdr_p_stars`, `within_fdr_p_value`, `within_fdr_p_stars`: Benjamini-Hochberg multiple-testing adjustments added after model fitting to the exported contrast and term tables.
- `weighted_fraction`: cluster-composition value calculated at biological-replicate level as the cell-number-weighted mean of FOV-level cluster fractions within each treatment and replicate. These values are then used as the input to the repeated-measures ANOVA cluster-composition analysis.
- `partial_eta_sq`: effect size derived from repeated-measures ANOVA outputs as `(F * numerator_df) / ((F * numerator_df) + denominator_df)` in the thesis-table exports.
- `rm_anova`, `paired_t`: formatted text summaries of the test statistic and degrees of freedom created for the thesis-table exports; these are display columns rather than direct model outputs.

Source data:
- `original_data/chapter_3/csvs/*.csv`
- `original_data/pkm2/csvs/*.csv`
- `original_data/ppkm2/csvs/*.csv`
- `original_data/calcium/out_registration_batch/verify_inputs.csv`
- `original_data/calcium/out_registration_batch/*/matched_cells_with_stain_metrics.csv`
- `original_data/calcium/out_registration_batch/*/calcium_metrics.csv`
- `original_data/calcium/out_registration_batch/*/calcium_timeseries_long.csv`
- `original_data/caspase/csvs/*.csv`
- additional image-quantification inputs for the copied upstream pipelines are defined inside the scripts themselves and include local LIF, TIFF, Cellpose, and ilastik model paths.

Generated data:
- `parsed_data/chapter_3/per_cell_stats_all.csv`
- `parsed_data/pkm2/per_cell_stats_all.csv`
- `parsed_data/ppkm2/per_cell_stats_all.csv`
- `parsed_data/caspase/*.csv`
- `parsed_data/parsed_data_build_summary.csv`
- `outputs/*`
- `quantification/*` for the ensheathment metrics, MixedLM outputs, graphs, and thesis-table exports when the ensheathment workflow is run.

From-scratch workflow from inside this folder:
- `python build_parsed_data_from_original_data.py`
- `python run_chapter_3_github.py`
- `python run_pkm2_github.py`
- `python run_ppkm2_github.py`
- `python run_caspase_biolrep_stats_github.py`
- `python run_calcium_github.py`
- `python image_quantification/ensheathment_get_ilastik_outputs.py`
- `python quantify_ensheathment.py`
- `python run_ensheathment_mixedlm.py`
- `python plot_ensheathment_graphs.py`
- `python make_ensheathment_thesis_tables.py`
- `python make_cluster_composition_thesis_tables.py`
- `python make_calcium_thesis_tables.py`
- `python make_caspase_thesis_tables.py`
- `python compare_outputs_against_reference_bundle.py`

Notes:
- Chapter 3, PKM2, pPKM2, and caspase auto-build their parsed inputs from `original_data/` when the required parsed table is missing.
- Calcium reads directly from `original_data/calcium/out_registration_batch` and does not use `parsed_data`.
- The copied `image_quantification/` pipelines are preserved as the original upstream image-processing scripts; some use acquisition-specific local path settings and may need path edits or command-line overrides before rerunning on a new machine.
- The ensheathment scripts form a separate workflow from the Chapter 3 / Chapter 4 analysis bundle and write into `quantification/`.
- `make_cluster_composition_thesis_tables.py`, `make_calcium_thesis_tables.py`, and `make_caspase_thesis_tables.py` write their outputs into `outputs/thesis_tables`.
- Generated `parsed_data/`, `outputs/`, and `original_data/validation/` folders are local analysis artifacts and are not tracked in git.
