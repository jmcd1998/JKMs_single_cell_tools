from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STATS_DIR = ROOT / "outputs" / "caspase_output_V1" / "stats"
THESIS_DIR = ROOT / "outputs" / "thesis_tables"


# Builds thesis-ready repeated-measures and post-hoc tables for the Chapter 4
# caspase phenotype analysis.
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


def write_table(df: pd.DataFrame, stem: str) -> None:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(THESIS_DIR / f"{stem}.csv", index=False)
    df.to_csv(THESIS_DIR / f"{stem}.tsv", index=False, sep="\t")


def build_global_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(STATS_DIR / "global_repeated_measures_tests.csv").copy()
    df["partial_eta_sq"] = df.apply(
        lambda row: partial_eta_squared(
            f_value=row["f_value"],
            num_df=row["num_df"],
            den_df=row["den_df"],
        ),
        axis=1,
    )
    df["rm_anova"] = df.apply(
        lambda row: f"F({int(row['num_df'])}, {int(row['den_df'])}) = {float(row['f_value']):.2f}",
        axis=1,
    )

    thesis = df[["effect", "rm_anova", "partial_eta_sq", "p_value"]].copy()
    thesis["partial_eta_sq"] = thesis["partial_eta_sq"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
    )
    thesis["p_value"] = thesis["p_value"].map(format_p)
    thesis = thesis.rename(
        columns={
            "effect": "Effect",
            "rm_anova": "RM ANOVA",
            "partial_eta_sq": "Partial eta-squared",
            "p_value": "p value",
        }
    )
    return df, thesis


def build_overall_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(STATS_DIR / "overall_treatment_vs_vehicle_stats.csv").copy()
    df["contrast"] = df["treatment"].astype(str) + " vs vehicle"
    df["paired_t"] = df.apply(
        lambda row: f"t({int(row['n_biol_rep']) - 1}) = {float(row['t_stat']):.2f}",
        axis=1,
    )

    thesis = df[
        [
            "contrast",
            "vehicle_frac_mean",
            "treatment_frac_mean",
            "frac_diff_mean",
            "paired_t",
            "p_value",
            "p_adj_fdr_bh",
        ]
    ].copy()
    thesis["vehicle_frac_mean"] = thesis["vehicle_frac_mean"].map(lambda value: round(float(value), 3))
    thesis["treatment_frac_mean"] = thesis["treatment_frac_mean"].map(lambda value: round(float(value), 3))
    thesis["frac_diff_mean"] = thesis["frac_diff_mean"].map(lambda value: round(float(value), 3))
    thesis["p_value"] = thesis["p_value"].map(format_p)
    thesis["p_adj_fdr_bh"] = thesis["p_adj_fdr_bh"].map(format_p)
    thesis = thesis.rename(
        columns={
            "contrast": "Contrast",
            "vehicle_frac_mean": "Vehicle mean fraction",
            "treatment_frac_mean": "Treatment mean fraction",
            "frac_diff_mean": "Mean difference",
            "paired_t": "Paired t test",
            "p_value": "p value",
            "p_adj_fdr_bh": "p value (FDR)",
        }
    )
    return df, thesis


def build_by_phenotype_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(STATS_DIR / "treatment_vs_vehicle_by_phenotype_stats.csv").copy()
    df["contrast"] = df["treatment"].astype(str) + " vs vehicle"
    df["paired_t"] = df.apply(
        lambda row: f"t({int(row['n_biol_rep']) - 1}) = {float(row['t_stat']):.2f}",
        axis=1,
    )

    thesis = df[
        [
            "phenotype",
            "contrast",
            "vehicle_frac_mean",
            "treatment_frac_mean",
            "frac_diff_mean",
            "paired_t",
            "p_value",
            "p_adj_fdr_bh",
        ]
    ].copy()
    thesis["vehicle_frac_mean"] = thesis["vehicle_frac_mean"].map(lambda value: round(float(value), 3))
    thesis["treatment_frac_mean"] = thesis["treatment_frac_mean"].map(lambda value: round(float(value), 3))
    thesis["frac_diff_mean"] = thesis["frac_diff_mean"].map(lambda value: round(float(value), 3))
    thesis["p_value"] = thesis["p_value"].map(format_p)
    thesis["p_adj_fdr_bh"] = thesis["p_adj_fdr_bh"].map(format_p)
    thesis = thesis.rename(
        columns={
            "phenotype": "Phenotype",
            "contrast": "Contrast",
            "vehicle_frac_mean": "Vehicle mean fraction",
            "treatment_frac_mean": "Treatment mean fraction",
            "frac_diff_mean": "Mean difference",
            "paired_t": "Paired t test",
            "p_value": "p value",
            "p_adj_fdr_bh": "p value (FDR)",
        }
    )
    return df, thesis


def main() -> None:
    global_full, global_thesis = build_global_table()
    overall_full, overall_thesis = build_overall_table()
    by_pheno_full, by_pheno_thesis = build_by_phenotype_table()

    write_table(global_full, "caspase_global_rm_anova_full")
    write_table(global_thesis, "caspase_global_rm_anova_thesis_table")
    write_table(overall_full, "caspase_overall_vs_vehicle_full")
    write_table(overall_thesis, "caspase_overall_vs_vehicle_thesis_table")
    write_table(by_pheno_full, "caspase_by_phenotype_vs_vehicle_full")
    write_table(by_pheno_thesis, "caspase_by_phenotype_vs_vehicle_thesis_table")

    print("Wrote:")
    print(THESIS_DIR / "caspase_global_rm_anova_thesis_table.csv")
    print(THESIS_DIR / "caspase_global_rm_anova_thesis_table.tsv")
    print(THESIS_DIR / "caspase_overall_vs_vehicle_thesis_table.csv")
    print(THESIS_DIR / "caspase_overall_vs_vehicle_thesis_table.tsv")
    print(THESIS_DIR / "caspase_by_phenotype_vs_vehicle_thesis_table.csv")
    print(THESIS_DIR / "caspase_by_phenotype_vs_vehicle_thesis_table.tsv")


if __name__ == "__main__":
    main()
