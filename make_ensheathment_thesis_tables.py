from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
STATS_DIR = ROOT / "quantification" / "mixedlm"
THESIS_DIR = ROOT / "quantification" / "thesis_tables"

METRIC_ORDER = {
    "mbp_total_area_px": 0,
    "mbp_soma_area_px": 1,
    "process_to_total_ratio": 2,
    "mbp_process_area_px": 3,
    "pct_mbp_nanofiber_colocalized": 4,
    "pct_process_nanofiber_colocalized": 5,
}
TREATMENT_ORDER = {
    "pranlukast": 0,
    "HAMI3379": 1,
}
METRIC_DISPLAY = {
    "Total MBP Area": "Total MBP area (px^2)",
    "MBP Soma Area": "MBP soma area (px^2)",
    "Process:Total MBP Ratio": "Process:total MBP ratio",
    "MBP Process Area": "MBP process area (px^2)",
    "Percent MBP Colocalized With Nanofiber": "% MBP colocalized with nanofiber",
    "Percent Process Colocalized With Nanofiber": "% process colocalized with nanofiber",
}
TREATMENT_DISPLAY = {
    "pranlukast": "Pranlukast",
    "HAMI3379": "HAMI3379",
    "vehicle": "vehicle",
}
RATIO_PERCENT_METRICS = {
    "process_to_total_ratio",
    "pct_mbp_nanofiber_colocalized",
    "pct_process_nanofiber_colocalized",
}


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def p_to_stars(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def write_table(df: pd.DataFrame, stem: str) -> None:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(THESIS_DIR / f"{stem}.csv", index=False)
    df.to_csv(THESIS_DIR / f"{stem}.tsv", index=False, sep="\t")


def append_fdr_columns(
    df: pd.DataFrame,
    *,
    p_value_col: str = "p_value",
    output_col: str = "fdr_p_value",
    stars_output_col: str = "fdr_p_stars",
) -> pd.DataFrame:
    out = df.copy()
    if "p_stars" not in out.columns and p_value_col in out.columns:
        out["p_stars"] = out[p_value_col].map(p_to_stars)
    mask = out[p_value_col].notna()
    corrected = [float("nan")] * len(out)
    if mask.any():
        adj = multipletests(out.loc[mask, p_value_col].astype(float).to_numpy(), method="fdr_bh")[1]
        adj_iter = iter(adj.tolist())
        corrected = [next(adj_iter) if bool(flag) else float("nan") for flag in mask.to_list()]
    out[output_col] = corrected
    out[stars_output_col] = out[output_col].map(p_to_stars)
    return out


def metric_sort_key(value: object) -> int:
    return METRIC_ORDER.get(str(value), 999)


def treatment_sort_key(value: object) -> int:
    return TREATMENT_ORDER.get(str(value), 999)


def metric_display(value: object) -> str:
    return METRIC_DISPLAY.get(str(value), str(value))


def treatment_display(value: object) -> str:
    return TREATMENT_DISPLAY.get(str(value), str(value))


def round_metric_value(metric: object, value: object) -> object:
    if pd.isna(value):
        return value
    metric_name = str(metric)
    decimals = 3 if metric_name in RATIO_PERCENT_METRICS else 1
    return round(float(value), decimals)


def build_global_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = STATS_DIR / "mixedlm_model_summary.csv"
    if not model_path.exists():
        raise FileNotFoundError(
            "MixedLM model summary not found. Run python run_ensheathment_mixedlm.py first."
        )

    df = pd.read_csv(model_path).copy()
    df = df.loc[df["status"].astype(str) == "ok"].copy()
    df = append_fdr_columns(df, p_value_col="omnibus_treatment_pvalue")
    df["metric_order"] = df["metric"].map(metric_sort_key)
    df["metric_display"] = df["metric_label"].map(metric_display)
    df["nested_mixedlm"] = df.apply(
        lambda row: f"chi2({int(row['omnibus_treatment_df'])}) = {float(row['omnibus_treatment_chi2']):.2f}",
        axis=1,
    )
    df.sort_values(["metric_order", "metric_display"], inplace=True, na_position="last")
    df.reset_index(drop=True, inplace=True)
    df["p_value_display"] = df["omnibus_treatment_pvalue"].map(format_p)
    df["fdr_p_value_display"] = df["fdr_p_value"].map(format_p)

    full = df[
        [
            "metric",
            "metric_label",
            "metric_display",
            "nested_mixedlm",
            "omnibus_treatment_chi2",
            "omnibus_treatment_df",
            "omnibus_treatment_pvalue",
            "p_value_display",
            "fdr_p_value",
            "fdr_p_value_display",
            "residual_sd",
            "n_cells_used",
            "n_biol_reps",
            "n_fields",
            "converged",
            "optimizer",
            "warning_count",
            "warning_messages",
            "random_structure",
        ]
    ].copy()

    thesis = df[
        [
            "metric_display",
            "nested_mixedlm",
            "omnibus_treatment_pvalue",
            "fdr_p_value",
        ]
    ].copy()
    thesis = thesis.rename(
        columns={
            "metric_display": "Metric",
            "nested_mixedlm": "Nested MixedLM",
            "omnibus_treatment_pvalue": "p value",
            "fdr_p_value": "p value (FDR)",
        }
    )
    thesis["p value"] = thesis["p value"].map(format_p)
    thesis["p value (FDR)"] = thesis["p value (FDR)"].map(format_p)
    return full, thesis


def build_vs_vehicle_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_path = STATS_DIR / "mixedlm_vehicle_contrasts.csv"
    if not contrast_path.exists():
        raise FileNotFoundError(
            "MixedLM vehicle contrasts not found. Run python run_ensheathment_mixedlm.py first."
        )

    df = pd.read_csv(contrast_path).copy()
    df = append_fdr_columns(df, p_value_col="p_value")
    df["metric_order"] = df["metric"].map(metric_sort_key)
    df["treatment_order"] = df["treatment"].map(treatment_sort_key)
    df["metric_display"] = df["metric_label"].map(metric_display)
    df["treatment_display"] = df["treatment"].map(treatment_display)
    df["contrast"] = df["treatment_display"].astype(str) + " vs vehicle"
    df.sort_values(["metric_order", "treatment_order", "metric_display"], inplace=True, na_position="last")
    df.reset_index(drop=True, inplace=True)
    df["p_value_display"] = df["p_value"].map(format_p)
    df["fdr_p_value_display"] = df["fdr_p_value"].map(format_p)

    full = df[
        [
            "metric",
            "metric_label",
            "metric_display",
            "treatment",
            "treatment_display",
            "contrast",
            "vehicle_mean_raw",
            "treatment_mean_raw",
            "estimate_vs_vehicle",
            "SE",
            "z",
            "std_effect_vs_vehicle",
            "ci_low",
            "ci_high",
            "p_value",
            "p_stars",
            "p_value_display",
            "fdr_p_value",
            "fdr_p_stars",
            "fdr_p_value_display",
            "residual_sd",
            "n_cells",
            "n_groups",
            "n_fovs",
            "avg_group_size",
            "avg_fov_size",
            "avg_fovs_per_group",
            "converged",
            "optimizer",
            "warning_count",
            "random_structure",
        ]
    ].copy()

    thesis = df[
        [
            "metric",
            "metric_display",
            "contrast",
            "vehicle_mean_raw",
            "treatment_mean_raw",
            "estimate_vs_vehicle",
            "SE",
            "z",
            "std_effect_vs_vehicle",
            "p_value",
            "fdr_p_value",
        ]
    ].copy()
    thesis["vehicle_mean_raw"] = thesis.apply(
        lambda row: round_metric_value(row["metric"], row["vehicle_mean_raw"]),
        axis=1,
    )
    thesis["treatment_mean_raw"] = thesis.apply(
        lambda row: round_metric_value(row["metric"], row["treatment_mean_raw"]),
        axis=1,
    )
    thesis["estimate_vs_vehicle"] = thesis.apply(
        lambda row: round_metric_value(row["metric"], row["estimate_vs_vehicle"]),
        axis=1,
    )
    thesis["SE"] = thesis.apply(
        lambda row: round_metric_value(row["metric"], row["SE"]),
        axis=1,
    )
    thesis["z"] = thesis["z"].map(lambda value: round(float(value), 3) if pd.notna(value) else value)
    thesis["std_effect_vs_vehicle"] = thesis["std_effect_vs_vehicle"].map(
        lambda value: round(float(value), 3) if pd.notna(value) else value
    )
    thesis = thesis.rename(
        columns={
            "metric_display": "Metric",
            "contrast": "Contrast",
            "vehicle_mean_raw": "Vehicle mean",
            "treatment_mean_raw": "Treatment mean",
            "estimate_vs_vehicle": "Estimate",
            "SE": "SE",
            "z": "z",
            "std_effect_vs_vehicle": "Std effect",
            "p_value": "p value",
            "fdr_p_value": "p value (FDR)",
        }
    )
    thesis["p value"] = thesis["p value"].map(format_p)
    thesis["p value (FDR)"] = thesis["p value (FDR)"].map(format_p)
    thesis = thesis[
        [
            "Metric",
            "Contrast",
            "Vehicle mean",
            "Treatment mean",
            "Estimate",
            "SE",
            "z",
            "Std effect",
            "p value",
            "p value (FDR)",
        ]
    ].copy()
    return full, thesis


def main() -> None:
    global_full, global_thesis = build_global_tables()
    contrast_full, contrast_thesis = build_vs_vehicle_tables()

    write_table(global_full, "ensheathment_nested_mixedlm_global_full")
    write_table(global_thesis, "ensheathment_nested_mixedlm_global_thesis_table")
    write_table(contrast_full, "ensheathment_nested_mixedlm_vs_vehicle_full")
    write_table(contrast_thesis, "ensheathment_nested_mixedlm_vs_vehicle_thesis_table")

    print("Wrote:")
    print(THESIS_DIR / "ensheathment_nested_mixedlm_global_thesis_table.csv")
    print(THESIS_DIR / "ensheathment_nested_mixedlm_global_thesis_table.tsv")
    print(THESIS_DIR / "ensheathment_nested_mixedlm_vs_vehicle_thesis_table.csv")
    print(THESIS_DIR / "ensheathment_nested_mixedlm_vs_vehicle_thesis_table.tsv")


if __name__ == "__main__":
    main()
