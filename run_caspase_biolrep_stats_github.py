from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


# This file is intentionally self-contained:
# 1. If the parsed caspase summary table already exists, we use it directly.
# 2. If it does not exist yet, we rebuild it from the source caspase CSV bundle.
# 3. We then run the replicate-aware RM-ANOVA and treatment-versus-vehicle summaries.
#
# The script produces the Chapter 4 Figure 6 caspase bundle in one place so the
# full path from source data to final plots and tables is easy to inspect.
REPO_ROOT = Path(__file__).resolve().parent
ORIGINAL_CASPASE_DIR = REPO_ROOT / "original_data" / "caspase" / "csvs"
PARSED_CASPASE_DIR = REPO_ROOT / "parsed_data" / "caspase"
PARSED_CASPASE_SUMMARY = PARSED_CASPASE_DIR / "caspase_summary_table.csv"
PARSED_CASPASE_FILTERED = PARSED_CASPASE_DIR / "caspase_cell_data_legacy_filtered.csv"

CASPASE_TREATMENT_PATTERN = re.compile(
    r"(vehicle|pranlukast|mdl29951|hami3379|rwt9996|clemastine|h202)",
    re.IGNORECASE,
)
CASPASE_TREATMENT_MAP = {
    "vehicle": "vehicle",
    "pranlukast": "pranlukast",
    "mdl29951": "MDL29951",
    "hami3379": "HAMI3379",
    "rwt9996": "RWT9996",
    "clemastine": "clemastine",
    "h202": "H2O2",
}
DEFAULT_CASPASE_TREATMENT_ORDER = (
    "vehicle",
    "MDL29951",
    "HAMI3379",
    "pranlukast",
    "RWT9996",
    "clemastine",
)
DEFAULT_CASPASE_PHENOTYPE_ORDER = (
    "PDGFRa",
    "O4",
    "MBP/O4",
    "Marker Low",
)


@dataclass(frozen=True)
class CaspaseGithubConfig:
    input_candidates: tuple[Path, ...]
    output_dir: Path
    treatment_order: tuple[str, ...] = DEFAULT_CASPASE_TREATMENT_ORDER
    phenotype_order: tuple[str, ...] = DEFAULT_CASPASE_PHENOTYPE_ORDER
    phenotype_palette: dict[str, str] = None
    biol_rep_col: str = "biol_rep"
    treatment_col: str = "treatment"
    phenotype_col: str = "phenotype"
    frac_col: str = "frac_caspase_pos"
    count_col: str = "n_cells"

    @property
    def input_path(self) -> Path:
        # Prefer a prebuilt parsed table when present, but rebuild it locally from
        # `original_data/caspase/csvs` if the user has only uploaded the raw bundle.
        for path in self.input_candidates:
            if path.exists():
                return path
        rebuild_caspase_parsed_inputs()
        for path in self.input_candidates:
            if path.exists():
                return path
        missing = self.input_candidates[0]
        raise FileNotFoundError(
            f"Expected parsed caspase input at {missing}. "
            "Also tried rebuilding it from original_data/caspase/csvs."
        )

    @property
    def stats_dir(self) -> Path:
        return self.output_dir / "stats"

    @property
    def graphs_dir(self) -> Path:
        return self.output_dir / "graphs"

    def __post_init__(self) -> None:
        if self.phenotype_palette is None:
            object.__setattr__(
                self,
                "phenotype_palette",
                {
                    "PDGFRa": "#2E7D32",
                    "O4": "#F6A623",
                    "MBP/O4": "#C2185B",
                    "Marker Low": "#7F8C8D",
                },
            )


CASPASE_LEGACY_TO_STANDARD_PHENOTYPE = {
    "pdgfra like": "PDGFRa",
    "o4 like": "O4",
    "mbp like": "MBP/O4",
    "marker low": "Marker Low",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def rebuild_caspase_legacy_filtered_cells(raw_dir: Path) -> pd.DataFrame:
    # Start from raw per-cell caspase/stain CSVs, recover treatment and replicate
    # labels, derive caspase positivity, then assign a phenotype per cell before
    # replicate-level aggregation.
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No caspase CSVs found in {raw_dir}")
    df = pd.concat((pd.read_csv(path) for path in csvs), ignore_index=True)

    df["treatment_raw"] = (
        df["base_name"].astype(str).str.extract(CASPASE_TREATMENT_PATTERN, expand=False).str.lower()
    )
    df["treatment"] = df["treatment_raw"].map(CASPASE_TREATMENT_MAP).fillna("vehicle")
    df["N"] = df["base_name"].astype(str).str.extract(r"(N\d+)", expand=False)
    df["well"] = df["base_name"].astype(str).str.extract(r"_(\d+)_R\d+", expand=False)
    df["fov"] = df["base_name"]

    # Cells are classed as caspase positive when `caspase_area_px > 50`.
    df["caspase_pos"] = pd.to_numeric(df["caspase_area_px"], errors="coerce") > 50
    df["caspase_pos_bin"] = df["caspase_pos"].astype(int)

    for marker in ["pdgfra", "o4", "mbp"]:
        df[f"{marker}_frac"] = pd.to_numeric(df[f"{marker}_area_px"], errors="coerce") / pd.to_numeric(
            df["cell_area_px"],
            errors="coerce",
        )

    # The PDGFRa-like phenotype applies a size-prior adjustment, so that step is
    # written out explicitly here instead of being hidden inside a generic utility.
    log_cell_area = np.log1p(pd.to_numeric(df["cell_area_px"], errors="coerce"))
    median = log_cell_area.median()
    mad = float(np.median(np.abs(log_cell_area - median)))
    if mad == 0:
        mad = 1.0
    df["size_prior_pdgfra"] = -((log_cell_area - median) / mad)

    min_area = 150

    def phenotype_with_size(row: pd.Series) -> str:
        scores: dict[str, float] = {}
        pdg_ok = row["pdgfra_area_px"] > min_area
        o4_ok = row["o4_area_px"] > min_area
        mbp_ok = row["mbp_area_px"] > min_area
        scores["PDGFRa"] = row["pdgfra_frac"] + 0.3 * row["size_prior_pdgfra"] if pdg_ok else 0.0
        scores["O4"] = row["o4_frac"] if o4_ok else 0.0
        scores["MBP/O4"] = row["mbp_frac"] if mbp_ok else 0.0
        if max(scores.values()) <= 0:
            return "Marker Low"
        return max(scores, key=scores.get)

    df["phenotype"] = df.apply(phenotype_with_size, axis=1)
    df = df[
        df["treatment"].isin(DEFAULT_CASPASE_TREATMENT_ORDER)
        & df["phenotype"].isin(DEFAULT_CASPASE_PHENOTYPE_ORDER)
    ].copy()
    df["N_well"] = df["N"].astype(str) + "_W" + df["well"].astype(str)
    df["well_id"] = df["N_well"] + "_" + df["treatment"].astype(str)
    return df


def rebuild_caspase_parsed_inputs() -> Path:
    # This is the self-contained replacement for requiring a separate
    # `build_parsed_data_from_original_data.py` step before running caspase stats.
    print("[prep] Parsed caspase summary missing; rebuilding it from original_data/caspase/csvs")
    filtered = rebuild_caspase_legacy_filtered_cells(ORIGINAL_CASPASE_DIR)
    ensure_parent(PARSED_CASPASE_FILTERED)
    filtered.to_csv(PARSED_CASPASE_FILTERED, index=False)

    summary = (
        filtered.groupby(["N", "phenotype", "treatment"], observed=True)
        .agg(
            n_cells=("caspase_pos_bin", "size"),
            n_caspase_pos=("caspase_pos_bin", "sum"),
        )
        .reset_index()
        .rename(columns={"N": "biol_rep"})
    )
    summary["frac_caspase_pos"] = summary["n_caspase_pos"] / summary["n_cells"]
    summary = summary.drop(columns="n_caspase_pos")

    summary["treatment"] = pd.Categorical(
        summary["treatment"],
        categories=DEFAULT_CASPASE_TREATMENT_ORDER,
        ordered=True,
    )
    summary["phenotype"] = pd.Categorical(
        summary["phenotype"],
        categories=DEFAULT_CASPASE_PHENOTYPE_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["phenotype", "treatment", "biol_rep"]).reset_index(drop=True)

    ensure_parent(PARSED_CASPASE_SUMMARY)
    summary.to_csv(PARSED_CASPASE_SUMMARY, index=False)
    print(f"[prep] Wrote parsed caspase summary: {PARSED_CASPASE_SUMMARY}")
    return PARSED_CASPASE_SUMMARY


def ensure_output_dirs(config: CaspaseGithubConfig) -> None:
    config.stats_dir.mkdir(parents=True, exist_ok=True)
    config.graphs_dir.mkdir(parents=True, exist_ok=True)


def load_caspase_table(config: CaspaseGithubConfig) -> pd.DataFrame:
    df = pd.read_csv(config.input_path)
    required = {
        config.biol_rep_col,
        config.phenotype_col,
        config.treatment_col,
        config.frac_col,
        config.count_col,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {config.input_path}: {missing}")
    return df


def prepare_caspase_table(config: CaspaseGithubConfig) -> pd.DataFrame:
    df = load_caspase_table(config).copy()
    df[config.biol_rep_col] = df[config.biol_rep_col].astype(str)
    df[config.phenotype_col] = (
        df[config.phenotype_col].astype(str).replace(CASPASE_LEGACY_TO_STANDARD_PHENOTYPE)
    )
    df[config.treatment_col] = df[config.treatment_col].astype(str)
    df[config.count_col] = pd.to_numeric(df[config.count_col], errors="coerce")
    df[config.frac_col] = pd.to_numeric(df[config.frac_col], errors="coerce")
    df = df.dropna(
        subset=[
            config.biol_rep_col,
            config.phenotype_col,
            config.treatment_col,
            config.frac_col,
            config.count_col,
        ]
    ).copy()
    df = df[df[config.count_col] > 0].copy()

    df["caspase_pos_count"] = np.rint(df[config.frac_col] * df[config.count_col]).astype(int)
    df["caspase_pos_count"] = df["caspase_pos_count"].clip(lower=0, upper=df[config.count_col].astype(int))

    aggregated = (
        df.groupby(
            [config.biol_rep_col, config.treatment_col, config.phenotype_col],
            observed=True,
            as_index=False,
        )
        .agg(
            caspase_pos_count=("caspase_pos_count", "sum"),
            n_cells=(config.count_col, "sum"),
        )
    )
    aggregated[config.frac_col] = aggregated["caspase_pos_count"] / aggregated["n_cells"]
    boundary_mask = (
        aggregated[config.frac_col].notna()
        & ((aggregated[config.frac_col] <= 0.0) | (aggregated[config.frac_col] >= 1.0))
    )
    if boundary_mask.any():
        raise RuntimeError(
            "Replicate-level caspase fractions contained 0 or 1, so the plain logit transform is undefined."
        )
    aggregated["logit_frac"] = np.log(
        aggregated[config.frac_col] / (1.0 - aggregated[config.frac_col])
    )
    aggregated["percent_caspase_pos"] = 100.0 * aggregated[config.frac_col]

    observed_treatments = set(aggregated[config.treatment_col].unique())
    treatment_levels = [value for value in config.treatment_order if value in observed_treatments]
    treatment_levels += sorted(observed_treatments.difference(treatment_levels))

    observed_phenotypes = set(aggregated[config.phenotype_col].unique())
    phenotype_levels = [value for value in config.phenotype_order if value in observed_phenotypes]
    phenotype_levels += sorted(observed_phenotypes.difference(phenotype_levels))

    aggregated[config.treatment_col] = pd.Categorical(
        aggregated[config.treatment_col],
        categories=treatment_levels,
        ordered=True,
    )
    aggregated[config.phenotype_col] = pd.Categorical(
        aggregated[config.phenotype_col],
        categories=phenotype_levels,
        ordered=True,
    )
    return aggregated.sort_values(
        [config.phenotype_col, config.treatment_col, config.biol_rep_col]
    ).reset_index(drop=True)


def inspect_caspase_dataset(config: CaspaseGithubConfig) -> pd.DataFrame:
    df = prepare_caspase_table(config)
    rows = [
        {"metric": "data_csv", "value": str(config.input_path)},
        {"metric": "output_dir", "value": str(config.output_dir)},
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_biol_reps", "value": int(df[config.biol_rep_col].nunique())},
        {"metric": "n_treatments", "value": int(df[config.treatment_col].nunique())},
        {"metric": "n_phenotypes", "value": int(df[config.phenotype_col].nunique())},
        {"metric": "avg_cells_per_biol_rep_treatment_phenotype", "value": float(df["n_cells"].mean())},
        {"metric": "median_cells_per_biol_rep_treatment_phenotype", "value": float(df["n_cells"].median())},
    ]
    return pd.DataFrame(rows)


def run_global_rm_anova(df: pd.DataFrame, config: CaspaseGithubConfig) -> pd.DataFrame:
    anova = AnovaRM(
        data=df,
        depvar=config.frac_col,
        subject=config.biol_rep_col,
        within=[config.treatment_col, config.phenotype_col],
    ).fit()
    table = anova.anova_table.reset_index().rename(
        columns={
            "index": "effect",
            "F Value": "f_value",
            "Num DF": "num_df",
            "Den DF": "den_df",
            "Pr > F": "p_value",
        }
    )
    table.insert(0, "analysis", "caspase_raw_fraction_rm_anova")
    return table


def build_pairwise_vs_vehicle_by_phenotype(
    df: pd.DataFrame,
    config: CaspaseGithubConfig,
) -> pd.DataFrame:
    treatments = list(df[config.treatment_col].cat.categories)
    phenotypes = list(df[config.phenotype_col].cat.categories)
    rows: list[dict[str, object]] = []

    for phenotype in phenotypes:
        sub = df[df[config.phenotype_col] == phenotype].copy()
        wide_frac = sub.pivot(index=config.biol_rep_col, columns=config.treatment_col, values=config.frac_col)
        if "vehicle" not in wide_frac.columns:
            continue
        for treatment in [value for value in treatments if value != "vehicle"]:
            if treatment not in wide_frac.columns:
                continue
            paired_frac = wide_frac[[treatment, "vehicle"]].dropna()
            if paired_frac.empty:
                continue
            diff_frac = paired_frac[treatment] - paired_frac["vehicle"]
            n = int(diff_frac.notna().sum())
            t_stat, p_value = ttest_rel(paired_frac[treatment], paired_frac["vehicle"])

            mean_frac_diff = float(diff_frac.mean())
            if n >= 2:
                sd_diff = float(diff_frac.std(ddof=1))
                sem_diff = sd_diff / np.sqrt(n)
                t_crit = float(student_t.ppf(0.975, df=n - 1))
                ci_low = mean_frac_diff - t_crit * sem_diff
                ci_high = mean_frac_diff + t_crit * sem_diff
            else:
                ci_low = np.nan
                ci_high = np.nan

            rows.append(
                {
                    "phenotype": str(phenotype),
                    "treatment": treatment,
                    "n_biol_rep": n,
                    "vehicle_frac_mean": float(paired_frac["vehicle"].mean()),
                    "treatment_frac_mean": float(paired_frac[treatment].mean()),
                    "frac_diff_mean": mean_frac_diff,
                    "frac_diff_ci_low": ci_low,
                    "frac_diff_ci_high": ci_high,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                }
            )

    contrast_df = pd.DataFrame(rows)
    if contrast_df.empty:
        return contrast_df
    contrast_df["p_adj_fdr_bh"] = multipletests(contrast_df["p_value"], method="fdr_bh")[1]
    return contrast_df.sort_values(["phenotype", "p_adj_fdr_bh", "treatment"]).reset_index(drop=True)


def build_overall_pairwise_vs_vehicle(
    df: pd.DataFrame,
    config: CaspaseGithubConfig,
) -> pd.DataFrame:
    collapsed = (
        df.groupby([config.biol_rep_col, config.treatment_col], observed=True)
        .agg(
            frac_caspase_pos=(config.frac_col, "mean"),
        )
        .reset_index()
    )
    wide_frac = collapsed.pivot(index=config.biol_rep_col, columns=config.treatment_col, values="frac_caspase_pos")
    rows: list[dict[str, object]] = []

    for treatment in [value for value in wide_frac.columns if value != "vehicle"]:
        paired_frac = wide_frac[[treatment, "vehicle"]].dropna()
        if paired_frac.empty:
            continue
        diff_frac = paired_frac[treatment] - paired_frac["vehicle"]
        n = int(diff_frac.notna().sum())
        t_stat, p_value = ttest_rel(paired_frac[treatment], paired_frac["vehicle"])

        mean_frac_diff = float(diff_frac.mean())
        if n >= 2:
            sd_diff = float(diff_frac.std(ddof=1))
            sem_diff = sd_diff / np.sqrt(n)
            t_crit = float(student_t.ppf(0.975, df=n - 1))
            ci_low = mean_frac_diff - t_crit * sem_diff
            ci_high = mean_frac_diff + t_crit * sem_diff
        else:
            ci_low = np.nan
            ci_high = np.nan

        rows.append(
            {
                "treatment": treatment,
                "n_biol_rep": n,
                "vehicle_frac_mean": float(paired_frac["vehicle"].mean()),
                "treatment_frac_mean": float(paired_frac[treatment].mean()),
                "frac_diff_mean": mean_frac_diff,
                "frac_diff_ci_low": ci_low,
                "frac_diff_ci_high": ci_high,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
        )

    overall_df = pd.DataFrame(rows)
    if overall_df.empty:
        return overall_df
    overall_df["p_adj_fdr_bh"] = multipletests(overall_df["p_value"], method="fdr_bh")[1]
    return overall_df.sort_values(["p_adj_fdr_bh", "treatment"]).reset_index(drop=True)


def write_dataset_summary(df: pd.DataFrame, config: CaspaseGithubConfig) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "data_csv": str(config.input_path),
                "n_rows": int(len(df)),
                "n_biol_reps": int(df[config.biol_rep_col].nunique()),
                "n_treatments": int(df[config.treatment_col].nunique()),
                "n_phenotypes": int(df[config.phenotype_col].nunique()),
                "avg_cells_per_biol_rep_treatment_phenotype": float(df["n_cells"].mean()),
                "median_cells_per_biol_rep_treatment_phenotype": float(df["n_cells"].median()),
            }
        ]
    )
    summary.to_csv(config.stats_dir / "caspase_dataset_summary.csv", index=False)
    return summary


def plot_replicate_level_percent_positive(df: pd.DataFrame, config: CaspaseGithubConfig) -> Path:
    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(
        data=df,
        x=config.treatment_col,
        y="percent_caspase_pos",
        hue=config.phenotype_col,
        showfliers=False,
        palette=config.phenotype_palette,
    )
    sns.stripplot(
        data=df,
        x=config.treatment_col,
        y="percent_caspase_pos",
        hue=config.phenotype_col,
        palette=config.phenotype_palette,
        dodge=True,
        alpha=0.7,
        linewidth=0.6,
        edgecolor="black",
    )
    handles, labels = ax.get_legend_handles_labels()
    keep = len(df[config.phenotype_col].cat.categories)
    ax.legend(handles[:keep], labels[:keep], title="Phenotype", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Caspase-positive cells (%)")
    ax.set_title("Replicate-level caspase-positive fraction by treatment and phenotype")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = config.graphs_dir / "replicate_level_caspase_pos_by_treatment_phenotype.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_forest_by_phenotype(contrast_df: pd.DataFrame, config: CaspaseGithubConfig) -> Path | None:
    if contrast_df.empty:
        return None

    phenotypes = contrast_df["phenotype"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(phenotypes), figsize=(4.8 * len(phenotypes), 5.5), sharex=True)
    if len(phenotypes) == 1:
        axes = [axes]

    for ax, phenotype in zip(axes, phenotypes):
        sub = contrast_df[contrast_df["phenotype"] == phenotype].copy()
        y = np.arange(len(sub))
        ax.errorbar(
            x=sub["frac_diff_mean"],
            y=y,
            xerr=[
                sub["frac_diff_mean"] - sub["frac_diff_ci_low"],
                sub["frac_diff_ci_high"] - sub["frac_diff_mean"],
            ],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=3,
        )
        ax.axvline(0.0, color="firebrick", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["treatment"])
        ax.set_title(str(phenotype))
        ax.set_xlabel("Raw fraction difference vs vehicle")
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Treatment")
    fig.suptitle("Treatment vs vehicle raw-fraction differences within phenotype", y=1.02)
    plt.tight_layout()
    path = config.graphs_dir / "treatment_vs_vehicle_odds_ratio_forest_by_phenotype.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run_caspase_biolrep_stats(config: CaspaseGithubConfig) -> dict[str, object]:
    # Keep the runtime flow obvious and linear:
    # 1. resolve or rebuild the input table,
    # 2. prepare replicate-level data,
    # 3. write the main statistical tables,
    # 4. write the summary figures.
    ensure_output_dirs(config)
    df = prepare_caspase_table(config)
    dataset_summary = write_dataset_summary(df, config)
    global_tests = run_global_rm_anova(df, config)
    by_phenotype = build_pairwise_vs_vehicle_by_phenotype(df, config)
    overall = build_overall_pairwise_vs_vehicle(df, config)

    global_path = config.stats_dir / "global_repeated_measures_tests.csv"
    phenotype_path = config.stats_dir / "treatment_vs_vehicle_by_phenotype_stats.csv"
    overall_path = config.stats_dir / "overall_treatment_vs_vehicle_stats.csv"
    input_path = config.stats_dir / "replicate_level_analysis_input.csv"

    global_tests.to_csv(global_path, index=False)
    by_phenotype.to_csv(phenotype_path, index=False)
    overall.to_csv(overall_path, index=False)
    df.to_csv(input_path, index=False)

    percent_plot = plot_replicate_level_percent_positive(df, config)
    forest_plot = plot_forest_by_phenotype(by_phenotype, config)

    return {
        "config": config,
        "dataset_summary": dataset_summary,
        "global_tests": global_tests,
        "by_phenotype": by_phenotype,
        "overall": overall,
        "replicate_input": df,
        "dataset_summary_path": config.stats_dir / "caspase_dataset_summary.csv",
        "global_tests_path": global_path,
        "by_phenotype_path": phenotype_path,
        "overall_path": overall_path,
        "input_path": input_path,
        "percent_plot_path": percent_plot,
        "forest_plot_path": forest_plot,
    }


def build_default_config() -> CaspaseGithubConfig:
    return CaspaseGithubConfig(
        input_candidates=(PARSED_CASPASE_SUMMARY,),
        output_dir=REPO_ROOT / "outputs" / "caspase_output_V1",
    )


def main() -> None:
    config = build_default_config()
    print(inspect_caspase_dataset(config).to_string(index=False))
    results = run_caspase_biolrep_stats(config)
    print("global_tests_path:", results["global_tests_path"])
    print("by_phenotype_path:", results["by_phenotype_path"])
    print("overall_path:", results["overall_path"])


if __name__ == "__main__":
    main()
