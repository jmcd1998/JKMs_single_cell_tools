from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

import run_calcium_github as analysis


REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = Path(r"C:\Users\JackM\calcium\out_registration_batch")


def sort_frame(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    usable = [column for column in sort_cols if column in df.columns]
    if usable:
        return df.sort_values(usable).reset_index(drop=True)
    return df.reset_index(drop=True)


def compare_frames(
    generated_path: Path,
    reference_path: Path,
    *,
    sort_cols: list[str],
    allow_generated_superset: bool = False,
    atol: float = 1e-8,
) -> None:
    generated = pd.read_csv(generated_path)
    reference = pd.read_csv(reference_path)

    if allow_generated_superset:
        missing_reference_cols = [column for column in reference.columns if column not in generated.columns]
        if missing_reference_cols:
            raise AssertionError(
                f"Generated output is missing reference columns for {generated_path.name}: "
                f"{missing_reference_cols}"
            )
        generated = generated[reference.columns.tolist()].copy()
    elif list(generated.columns) != list(reference.columns):
        raise AssertionError(
            f"Column mismatch for {generated_path.name}: "
            f"{list(generated.columns)} != {list(reference.columns)}"
        )

    generated = sort_frame(generated, sort_cols)
    reference = sort_frame(reference, sort_cols)

    if generated.shape != reference.shape:
        raise AssertionError(
            f"Shape mismatch for {generated_path.name}: "
            f"{generated.shape} != {reference.shape}"
        )

    for column in generated.columns:
        gen_col = generated[column]
        ref_col = reference[column]
        if pd.api.types.is_numeric_dtype(gen_col) and pd.api.types.is_numeric_dtype(ref_col):
            if not np.allclose(gen_col.to_numpy(), ref_col.to_numpy(), atol=atol, equal_nan=True):
                raise AssertionError(f"Numeric mismatch in {generated_path.name} column '{column}'")
        else:
            gen_vals = gen_col.fillna("<NA>").astype(str).to_numpy()
            ref_vals = ref_col.fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(gen_vals, ref_vals):
                raise AssertionError(f"Value mismatch in {generated_path.name} column '{column}'")


def expected_figure_paths(fig_dir: Path) -> list[Path]:
    figures: list[Path] = [
        fig_dir / "mfi_elbow_silhouette.png",
        fig_dir / "mfi_clusters_3d.png",
        fig_dir / "auc_mixedlm_residual_diagnostics_raw_shifted.png",
        fig_dir / "auc_mixedlm_transform_sweep_residuals_raw_w2.png",
        fig_dir / "auc_mixedlm_transform_sweep_residuals_raw_w3.png",
    ]
    for treatment in analysis.TREAT_ORDER:
        figures.append(fig_dir / f"rt_scatter_{treatment}.png")
        figures.append(fig_dir / f"traces_mean_ci_{analysis.safe_name(treatment)}.png")
        for window_label in analysis.WINDOWS:
            figures.append(fig_dir / f"auc_violin_{treatment}_{window_label}.png")
            figures.append(fig_dir / f"auc_violin_{treatment}_{window_label}_norm.png")
    for window_label in analysis.WINDOWS:
        figures.append(fig_dir / f"auc_rt_grid_{window_label}.png")
    return figures


def expected_clustering_paths(clustering_dir: Path) -> list[Path]:
    return [
        clustering_dir / "mfi_kmeans_curve_metrics.csv",
        clustering_dir / "mfi_cluster_feature_means.csv",
        clustering_dir / "mfi_cluster_label_map.csv",
        clustering_dir / "mfi_cluster_centers_scaled.csv",
    ]


def expected_stats_paths(stats_dir: Path) -> list[Path]:
    paths: list[Path] = [
        stats_dir / "two_term_lmm_term_effects.csv",
        stats_dir / "two_term_lmm_emms.csv",
        stats_dir / "two_term_lmm_within_cluster_contrasts.csv",
        stats_dir / "two_term_lmm_model_fit_summary.csv",
        stats_dir / "lmm_supplementary_model_observation_summary.csv",
    ]
    report_dir = stats_dir / "mixedlm_full_reports"
    for window_label in analysis.WINDOWS:
        response_var = analysis.auc_col_for(window_label, "raw")
        paths.append(stats_dir / f"two_term_lmm_term_zscores_{response_var}.csv")
        paths.append(report_dir / f"two_term_lmm_{response_var}.csv")
        paths.append(report_dir / f"two_term_lmm_{response_var}.txt")
    return paths


def main() -> None:
    if not analysis.DEFAULT_DATA_DIR.exists():
        raise FileNotFoundError(f"Bundled calcium original data not found at {analysis.DEFAULT_DATA_DIR}")
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Reference analysis directory not found at {REFERENCE_DIR}")

    output_dir = REPO_ROOT / "outputs" / f"smoke_calcium_{uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Smoke-test output directory: {output_dir}")

    outputs = analysis.run_analysis(master_dir=analysis.DEFAULT_DATA_DIR, output_dir=output_dir)

    compare_frames(
        output_dir / "all_matched_cells_with_stain_metrics.csv",
        REFERENCE_DIR / "all_matched_cells_with_stain_metrics.csv",
        sort_cols=["bundle", "stain_label", "calcium_label"],
    )
    compare_frames(
        output_dir / "DF1_cells_with_cluster_assignment_present.csv",
        REFERENCE_DIR / "DF1_cells_with_cluster_assignment_present.csv",
        sort_cols=["bundle", "stain_label", "calcium_label"],
    )

    for window_label in analysis.WINDOWS:
        compare_frames(
            output_dir / f"auc_mixedlm_pairs_raw_{window_label}.csv",
            REFERENCE_DIR / f"auc_mixedlm_pairs_raw_{window_label}.csv",
            sort_cols=["window", "comparison"],
            allow_generated_superset=True,
            atol=1e-6,
        )
        compare_frames(
            output_dir / f"auc_mixedlm_interactions_raw_{window_label}.csv",
            REFERENCE_DIR / f"auc_mixedlm_interactions_raw_{window_label}.csv",
            sort_cols=["window", "term"],
            allow_generated_superset=True,
            atol=1e-6,
        )

    compare_frames(
        output_dir / "auc_mixedlm_pairs_raw_all_windows.csv",
        REFERENCE_DIR / "auc_mixedlm_pairs_raw_all_windows.csv",
        sort_cols=["window", "comparison"],
        allow_generated_superset=True,
        atol=1e-6,
    )
    compare_frames(
        output_dir / "auc_mixedlm_interactions_raw_all_windows.csv",
        REFERENCE_DIR / "auc_mixedlm_interactions_raw_all_windows.csv",
        sort_cols=["window", "term"],
        allow_generated_superset=True,
        atol=1e-6,
    )

    missing_figures = [path for path in expected_figure_paths(outputs["fig_dir"]) if not path.exists()]
    if missing_figures:
        raise AssertionError(f"Missing expected figures: {missing_figures}")

    missing_clustering = [path for path in expected_clustering_paths(outputs["clustering_dir"]) if not path.exists()]
    if missing_clustering:
        raise AssertionError(f"Missing expected clustering outputs: {missing_clustering}")

    missing_stats = [path for path in expected_stats_paths(outputs["stats_dir"]) if not path.exists()]
    if missing_stats:
        raise AssertionError(f"Missing expected stats outputs: {missing_stats}")

    fit_summary_df = pd.read_csv(outputs["stats_dir"] / "two_term_lmm_model_fit_summary.csv")
    if set(fit_summary_df["window_label"].dropna().astype(str)) != set(analysis.WINDOWS):
        raise AssertionError("Calcium stats fit summary is missing one or more stimulation windows")

    term_df = pd.read_csv(outputs["stats_dir"] / "two_term_lmm_term_effects.csv")
    required_term_cols = {"window_label", "response_var", "term", "estimate", "p_value", "fdr_p_value"}
    if not required_term_cols.issubset(term_df.columns):
        raise AssertionError("two_term_lmm_term_effects.csv is missing expected report columns")

    within_df = pd.read_csv(outputs["stats_dir"] / "two_term_lmm_within_cluster_contrasts.csv")
    required_within_cols = {"window_label", "treatment", "cluster", "within_estimate", "within_p_value", "within_fdr_p_value"}
    if not required_within_cols.issubset(within_df.columns):
        raise AssertionError("two_term_lmm_within_cluster_contrasts.csv is missing expected within-cluster columns")

    summary_df = pd.read_csv(output_dir / "analysis_summary.csv")
    expected_artifacts = {
        "mfi_kmeans_curve_metrics",
        "mfi_cluster_feature_means",
        "mfi_cluster_label_map",
        "auc_mixedlm_nested_raw",
        "auc_two_term_lmm_term_effects",
        "auc_two_term_lmm_within_cluster_contrasts",
    }
    if not expected_artifacts.issubset(set(summary_df["artifact_name"])):
        raise AssertionError("analysis_summary.csv is missing expected clustering or raw AUC artifacts")

    print("Smoke test passed.")
    print(f"Validated raw nested AUC outputs against: {REFERENCE_DIR}")
    print(f"Generated figures in: {outputs['fig_dir']}")
    print(f"Generated clustering outputs in: {outputs['clustering_dir']}")


if __name__ == "__main__":
    main()
