# Thesis Chapter 3 and Chapter 4 Output Map

This repository covers the quantitative analysis bundles used in the Chapter 3 and Chapter 4 thesis sections listed below.

Chapter 3
- Figure 4 and Tables 2-4: 40x neonatal mouse OPC differentiation assay.
  Script: `run_chapter_3_github.py`
  Output folder: `outputs/chapter_3_output_V1`
  Key figure files: `graphs/pdg_int_ratio_violin.png`, `graphs/cnp_int_ratio_violin.png`, `graphs/mbp_int_ratio_violin.png`, `graphs/clusters_3d_plot.png`, `graphs/cluster_composition_by_treatment.png`, `graphs/cell_counts_cells_per_fov_boxplot_all_treatments.png`, plus the morphology plots in `graphs/`.
  Key statistics files: `stats/one_term_lmm_term_effects.csv`, `stats/one_term_lmm_vs_vehicle.csv`, `stats/two_term_lmm_term_effects.csv`, `stats/two_term_lmm_within_cluster_contrasts.csv`, `stats/cluster_composition_global_repeated_measures_tests.csv`, `stats/cluster_composition_weighted_fraction_pairwise_vs_vehicle.csv`.
  Supporting clustering diagnostics: `clustering/elbow_vehicle_only.png`, `clustering/silhouette_vehicle_only.png`.
- Figures 1-3 and 5-6 from Chapter 3 are outside the scope of this repository.

Chapter 4
- Figure 3 and Table 2: multiplexed cluster-resolved calcium assay.
  Script: `run_calcium_github.py`
  Output folder: `outputs/calcium_full_output_V1`
  Key figure files: `figures/traces_mean_ci_*.png`, `figures/auc_violin_*.png`, `figures/auc_rt_grid_w*.png`, `figures/mfi_clusters_3d.png`.
  Key statistics files: `stats/two_term_lmm_term_effects.csv`, `stats/two_term_lmm_within_cluster_contrasts.csv`.
  Thesis-ready table export: `outputs/thesis_tables/calcium_auc_nested_mixedlm_effects_thesis_table.csv`.
  Supporting clustering diagnostic: `figures/mfi_elbow_silhouette.png`.
- Figure 5 and Table 3: differentiation state-resolved PKM2 and pPKM2 analyses.
  Primary pPKM2 bundle:
  Script: `run_ppkm2_github.py`
  Output folder: `outputs/ppkm2_output_V1`
  Companion PKM2 bundle:
  Script: `run_pkm2_github.py`
  Output folder: `outputs/pkm2_output_V1`
  Key figure files across these bundles: marker-ratio violin plots, `clusters_3d_plot.png`, `cluster_composition_by_treatment.png`, `cell_counts_cells_per_fov_boxplot_all_treatments.png`, and the morphology plots in each `graphs/` folder.
  Key statistics files: `stats/one_term_lmm_term_effects.csv`, `stats/two_term_lmm_term_effects.csv`, `stats/two_term_lmm_within_cluster_contrasts.csv`, `stats/cluster_composition_global_repeated_measures_tests.csv`, `stats/cluster_composition_weighted_fraction_pairwise_vs_vehicle.csv`.
  Thesis-ready cluster-composition exports: `outputs/thesis_tables/cluster_composition_global_rm_anova_thesis_table.csv` and `outputs/thesis_tables/cluster_composition_planned_posthoc_thesis_table.csv`.
  Supporting clustering diagnostics: each bundle's `clustering/elbow_vehicle_only.png` and `clustering/silhouette_vehicle_only.png`.
- Figure 6 and related supplementary tables: cleaved-caspase-3 phenotype analysis.
  Script: `run_caspase_biolrep_stats_github.py`
  Output folder: `outputs/caspase_output_V1`
  Key figure files: `graphs/replicate_level_caspase_pos_by_treatment_phenotype.png`, `graphs/treatment_vs_vehicle_odds_ratio_forest_by_phenotype.png`.
  Key statistics files: `stats/global_repeated_measures_tests.csv`, `stats/overall_treatment_vs_vehicle_stats.csv`, `stats/treatment_vs_vehicle_by_phenotype_stats.csv`.
  Thesis-ready table exports: `outputs/thesis_tables/caspase_global_rm_anova_thesis_table.csv`, `outputs/thesis_tables/caspase_overall_vs_vehicle_thesis_table.csv`, `outputs/thesis_tables/caspase_by_phenotype_vs_vehicle_thesis_table.csv`.
- Figures 1, 2, and 4 from Chapter 4 are outside the scope of this repository.
