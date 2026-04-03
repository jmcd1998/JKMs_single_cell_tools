from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
THESIS_DIR = OUTPUTS / "thesis_tables"


DATASETS = [
    {
        "dataset": "Chapter 3",
        "chapter": "Chapter 3",
        "output_dir": OUTPUTS / "chapter_3_output_V1" / "stats",
    },
    {
        "dataset": "PKM2",
        "chapter": "Chapter 4",
        "output_dir": OUTPUTS / "pkm2_output_V1" / "stats",
    },
    {
        "dataset": "pPKM2",
        "chapter": "Chapter 4",
        "output_dir": OUTPUTS / "ppkm2_output_V1" / "stats",
    },
]


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def partial_eta_squared(*, f_value: float, num_df: float, den_df: float) -> float:
    if pd.isna(f_value) or pd.isna(num_df) or pd.isna(den_df):
        return float("nan")
    f_value = float(f_value)
    num_df = float(num_df)
    den_df = float(den_df)
    denom = (f_value * num_df) + den_df
    if denom <= 0:
        return float("nan")
    return (f_value * num_df) / denom


def load_global_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for info in DATASETS:
        path = info["output_dir"] / "cluster_composition_global_repeated_measures_tests.csv"
        df = pd.read_csv(path).copy()
        df.insert(0, "dataset", info["dataset"])
        df.insert(1, "chapter", info["chapter"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["cluster"] = pd.to_numeric(combined["cluster"], errors="coerce")
    combined.sort_values(["chapter", "dataset", "cluster"], inplace=True, na_position="last")
    combined.reset_index(drop=True, inplace=True)

    thesis = combined[
        [
            "chapter",
            "dataset",
            "cluster_label",
            "n_biological_replicates",
            "n_treatments",
            "num_df",
            "den_df",
            "F",
            "p_value",
        ]
    ].copy()
    thesis = thesis.rename(
        columns={
            "cluster_label": "cluster",
            "n_biological_replicates": "n_biol_rep",
        }
    )
    thesis["partial_eta_sq"] = thesis.apply(
        lambda row: partial_eta_squared(
            f_value=row["F"],
            num_df=row["num_df"],
            den_df=row["den_df"],
        ),
        axis=1,
    )
    thesis["rm_anova"] = thesis.apply(
        lambda row: f"F({int(row['num_df'])}, {int(row['den_df'])}) = {float(row['F']):.2f}"
        if pd.notna(row["num_df"]) and pd.notna(row["den_df"]) and pd.notna(row["F"])
        else "",
        axis=1,
    )
    thesis["p_value_display"] = thesis["p_value"].map(format_p)
    thesis["partial_eta_sq_display"] = thesis["partial_eta_sq"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
    )
    thesis = thesis[
        [
            "chapter",
            "dataset",
            "cluster",
            "n_biol_rep",
            "n_treatments",
            "rm_anova",
            "partial_eta_sq",
            "partial_eta_sq_display",
            "p_value",
            "p_value_display",
        ]
    ]
    return combined, thesis


def load_posthoc_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for info in DATASETS:
        path = info["output_dir"] / "cluster_composition_weighted_fraction_pairwise_vs_vehicle.csv"
        df = pd.read_csv(path).copy()
        df.insert(0, "dataset", info["dataset"])
        df.insert(1, "chapter", info["chapter"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["cluster"] = pd.to_numeric(combined["cluster"], errors="coerce")
    combined.sort_values(["chapter", "dataset", "cluster", "p_value", "treatment"], inplace=True, na_position="last")
    combined.reset_index(drop=True, inplace=True)

    thesis = combined[
        [
            "chapter",
            "dataset",
            "cluster_label",
            "treatment",
            "n_biol_rep",
            "vehicle_mean_fraction",
            "treatment_mean_fraction",
            "mean_diff_treatment_minus_vehicle",
            "t_stat",
            "p_value",
            "fdr_p_value",
        ]
    ].copy()
    thesis = thesis.rename(columns={"cluster_label": "cluster"})
    thesis["contrast"] = thesis["treatment"].astype(str) + " vs Vehicle"
    thesis["paired_t"] = thesis.apply(
        lambda row: f"t({int(row['n_biol_rep']) - 1}) = {float(row['t_stat']):.2f}"
        if pd.notna(row["n_biol_rep"]) and pd.notna(row["t_stat"])
        else "",
        axis=1,
    )
    thesis["vehicle_mean_fraction"] = thesis["vehicle_mean_fraction"].map(lambda value: round(float(value), 6))
    thesis["treatment_mean_fraction"] = thesis["treatment_mean_fraction"].map(lambda value: round(float(value), 6))
    thesis["mean_diff_treatment_minus_vehicle"] = thesis["mean_diff_treatment_minus_vehicle"].map(
        lambda value: round(float(value), 6)
    )
    thesis["p_value_display"] = thesis["p_value"].map(format_p)
    thesis["fdr_p_value_display"] = thesis["fdr_p_value"].map(format_p)
    thesis = thesis[
        [
            "chapter",
            "dataset",
            "cluster",
            "contrast",
            "n_biol_rep",
            "vehicle_mean_fraction",
            "treatment_mean_fraction",
            "mean_diff_treatment_minus_vehicle",
            "paired_t",
            "p_value",
            "p_value_display",
            "fdr_p_value",
            "fdr_p_value_display",
        ]
    ]
    return combined, thesis


def write_table(df: pd.DataFrame, stem: str) -> None:
    csv_path = THESIS_DIR / f"{stem}.csv"
    tsv_path = THESIS_DIR / f"{stem}.tsv"
    df.to_csv(csv_path, index=False)
    df.to_csv(tsv_path, index=False, sep="\t")


def main() -> None:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    global_full, global_thesis = load_global_tables()
    posthoc_full, posthoc_thesis = load_posthoc_tables()

    write_table(global_full, "cluster_composition_global_rm_anova_combined_full")
    write_table(global_thesis, "cluster_composition_global_rm_anova_thesis_table")
    write_table(posthoc_full, "cluster_composition_planned_posthoc_combined_full")
    write_table(posthoc_thesis, "cluster_composition_planned_posthoc_thesis_table")

    print("Wrote:")
    for path in sorted(THESIS_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
