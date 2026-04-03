from pathlib import Path

from clustered_stripped_lmm_analysis import (
    ClusteredAnalysisConfig,
    inspect_dataset,
    run_clustered_analysis,
)


ROOT = Path(__file__).resolve().parent


def build_config() -> ClusteredAnalysisConfig:
    return ClusteredAnalysisConfig(
        chapter_label="Chapter 3",
        dataset_label="MBP/CNP/PDGFRa",
        analysis_name="chapter_3_github_parsed_clustercomp_rm_anova",
        input_mode="chapter3_combined",
        input_candidates=(
            ROOT / "parsed_data" / "chapter_3" / "per_cell_stats_all.csv",
        ),
        analysis_dir=ROOT / "outputs" / "chapter_3_output_V1",
        treatment_order=("Vehicle", "MDL29951", "HAMI3379", "Pranlukast"),
        cluster_feature_cols=("mbp_int_ratio", "cnp_int_ratio", "pdg_int_ratio"),
        one_term_response_cols=("mbp_int_ratio", "cnp_int_ratio", "pdg_int_ratio"),
        two_term_response_cols=("cell_area_px", "AUC", "area_ratio_union", "Rmax_px", "Imax", "CriticalValue"),
        cluster_k=3,
        input_export_name="per_cell_stats_all.csv",
        treatment_col="treatment",
        group_col="N",
        fov_col="FOV_ID",
        cluster_col="cluster",
        treatment_reference="Vehicle",
        optimizer_sequence=("lbfgs", "bfgs", "cg"),
        min_unique_values=5,
        cluster_composition_inference="rm_anova_weighted_fraction",
        cluster_composition_posthoc_scope="planned_mbp_high_vs_vehicle",
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
