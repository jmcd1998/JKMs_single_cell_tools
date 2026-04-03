from pathlib import Path

from clustered_stripped_lmm_analysis import (
    ClusteredAnalysisConfig,
    inspect_dataset,
    run_clustered_analysis,
)


ROOT = Path(__file__).resolve().parent


def build_config() -> ClusteredAnalysisConfig:
    combined_csv = ROOT / "parsed_data" / "ppkm2" / "per_cell_stats_all.csv"
    return ClusteredAnalysisConfig(
        chapter_label="Chapter 4",
        dataset_label="pPKM2",
        analysis_name="chapter_4_ppkm2_github_parsed_clustercomp_rm_anova",
        input_mode="chapter4_combined",
        input_candidates=(combined_csv,),
        analysis_dir=ROOT / "outputs" / "ppkm2_output_V1",
        treatment_order=("Vehicle", "MDL29951", "HAMI3379", "Pranlukast", "RWT9996", "Clemastine"),
        cluster_feature_cols=("mbp_int_ratio", "o4_int_ratio", "pdg_int_ratio"),
        one_term_response_cols=("mbp_int_ratio", "o4_int_ratio", "pdg_int_ratio", "ppkm2_int_ratio"),
        two_term_response_cols=(
            "cell_area_px",
            "morph_all_AUC",
            "morph_all_convexity_ratio",
            "morph_all_Rmax_px",
            "morph_all_Imax",
            "morph_all_CriticalValue",
            "ppkm2_int_ratio",
            "ppkm2_nuc_int_ratio",
        ),
        cluster_k=4,
        row_tokens=("7", "8"),
        input_export_name="per_cell_stats_parsed.csv",
        treatment_col="treatment",
        group_col="N",
        fov_col="FOV_ID",
        cluster_col="cluster",
        treatment_reference="Vehicle",
        optimizer_sequence=("lbfgs", "bfgs", "cg"),
        min_unique_values=5,
        cluster_composition_inference="rm_anova_weighted_fraction",
        cluster_composition_posthoc_scope="planned_mbp_high_vs_vehicle",
        legacy_mfi_point_every=25,
        legacy_mfi_point_seed=1,
    )


def main() -> None:
    config = build_config()
    print(inspect_dataset(config).to_string(index=False))
    results = run_clustered_analysis(config)
    print("analysis_dir:", results["analysis_dir"])
    print("stats_dir:", results["stats_dir"])
    print("graphs_dir:", results["graphs_dir"])
    print("clustering_dir:", results["clustering_dir"])
    print("supplementary_summary:", config.stats_dir / "lmm_supplementary_model_observation_summary.csv")


if __name__ == "__main__":
    main()
