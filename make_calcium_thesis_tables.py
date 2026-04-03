from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STATS_DIR = ROOT / "outputs" / "calcium_full_output_V1" / "stats"
THESIS_DIR = ROOT / "outputs" / "thesis_tables"

REFERENCE_TREATMENT = "MDL29951"
REFERENCE_CLUSTER = 0
WINDOW_ORDER = {"w1": 1, "w2": 2, "w3": 3}
WINDOW_DISPLAY = {
    "w1": "Stimulation window W1",
    "w2": "Stimulation window W2",
    "w3": "Stimulation window W3",
}
TREATMENT_ORDER = {"MDL29951": 0, "pranlukast": 1, "HAMI3379": 2}
TREATMENT_DISPLAY = {
    "MDL29951": "MDL29951",
    "pranlukast": "Pranlukast",
    "HAMI3379": "HAMI3379",
}


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def write_table(df: pd.DataFrame, stem: str) -> None:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(THESIS_DIR / f"{stem}.csv", index=False)
    df.to_csv(THESIS_DIR / f"{stem}.tsv", index=False, sep="\t")


def coerce_cluster(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(numeric) if numeric.is_integer() else numeric


def cluster_text(value: object) -> str:
    cluster = coerce_cluster(value)
    if pd.isna(cluster):
        return ""
    return str(cluster)


def cluster_sort_key(value: object) -> tuple[int, object]:
    cluster = coerce_cluster(value)
    if pd.isna(cluster):
        return (2, "")
    if isinstance(cluster, (int, float)):
        return (0, float(cluster))
    return (1, str(cluster))


def parse_cluster_term(term: object) -> object:
    match = re.search(r"C\(cluster_k3_present\)\[T\.([^\]]+)\]", str(term))
    if not match:
        return pd.NA
    return coerce_cluster(match.group(1))


def parse_treatment_term(term: object) -> str:
    match = re.search(
        r'C\(treatment, Treatment\(reference="MDL29951"\)\)\[T\.([^\]]+)\]',
        str(term),
    )
    return match.group(1) if match else ""


def display_treatment(value: object) -> str:
    return TREATMENT_DISPLAY.get(str(value), str(value))


def standardize_effect_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["window_order"] = out["window_label"].map(WINDOW_ORDER)
    out["model_label"] = out["window_label"].map(WINDOW_DISPLAY).fillna(out["window_label"].astype(str))
    out["cluster_sort"] = out["cluster"].map(cluster_sort_key)
    out["treatment_sort"] = out["treatment"].map(lambda value: TREATMENT_ORDER.get(str(value), 99))
    out.sort_values(
        ["window_order", "section_order", "cluster_sort", "treatment_sort", "effect_label"],
        inplace=True,
        na_position="last",
    )
    out.reset_index(drop=True, inplace=True)
    out["p_value_display"] = out["p_value"].map(format_p)
    out["fdr_p_value_display"] = out["fdr_p_value"].map(format_p)
    return out


def build_cluster_comparison_rows(term_df: pd.DataFrame) -> pd.DataFrame:
    mask = (~term_df["is_interaction"].fillna(False)) & term_df["term"].astype(str).str.contains("C\\(cluster_k3_present\\)")
    df = term_df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame()

    df["cluster"] = df["term"].map(parse_cluster_term)
    df["treatment"] = pd.NA
    df["section"] = "Cluster comparison"
    df["section_order"] = 1
    df["effect_label"] = df["cluster"].map(
        lambda cluster: f"Cluster {cluster_text(cluster)} vs Cluster {REFERENCE_CLUSTER} at {REFERENCE_TREATMENT}"
    )
    return df[
        [
            "window_label",
            "section",
            "section_order",
            "effect_label",
            "response_var",
            "term",
            "treatment",
            "cluster",
            "estimate",
            "SE",
            "z",
            "std_effect",
            "p_value",
            "fdr_p_value",
        ]
    ].copy()


def build_interaction_rows(term_df: pd.DataFrame) -> pd.DataFrame:
    df = term_df.loc[term_df["is_interaction"].fillna(False)].copy()
    if df.empty:
        return pd.DataFrame()

    df["treatment"] = df["term"].map(parse_treatment_term)
    df["cluster"] = df["term"].map(parse_cluster_term)
    df["section"] = "Interaction term"
    df["section_order"] = 2
    df["effect_label"] = df.apply(
        lambda row: (
            f"{display_treatment(row['treatment'])} vs {REFERENCE_TREATMENT} interaction "
            f"in Cluster {cluster_text(row['cluster'])} relative to Cluster {REFERENCE_CLUSTER}"
        ),
        axis=1,
    )
    return df[
        [
            "window_label",
            "section",
            "section_order",
            "effect_label",
            "response_var",
            "term",
            "treatment",
            "cluster",
            "estimate",
            "SE",
            "z",
            "std_effect",
            "p_value",
            "fdr_p_value",
        ]
    ].copy()


def build_within_cluster_rows(within_df: pd.DataFrame) -> pd.DataFrame:
    df = within_df.copy()
    if df.empty:
        return pd.DataFrame()

    df["cluster"] = df["cluster"].map(coerce_cluster)
    df["section"] = "Within-cluster treatment effects"
    df["section_order"] = 3
    df["effect_label"] = df.apply(
        lambda row: (
            f"{display_treatment(row['treatment'])} vs {REFERENCE_TREATMENT} "
            f"within Cluster {cluster_text(row['cluster'])}"
        ),
        axis=1,
    )
    df["term"] = pd.NA
    df = df.rename(
        columns={
            "within_estimate": "estimate",
            "within_SE": "SE",
            "within_z": "z",
            "std_effect_within_cluster": "std_effect",
            "within_p_value": "p_value",
            "within_fdr_p_value": "fdr_p_value",
        }
    )
    return df[
        [
            "window_label",
            "section",
            "section_order",
            "effect_label",
            "response_var",
            "term",
            "treatment",
            "cluster",
            "estimate",
            "SE",
            "z",
            "std_effect",
            "p_value",
            "fdr_p_value",
        ]
    ].copy()


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    term_path = STATS_DIR / "two_term_lmm_term_effects.csv"
    within_path = STATS_DIR / "two_term_lmm_within_cluster_contrasts.csv"
    if not term_path.exists() or not within_path.exists():
        raise FileNotFoundError(
            "Calcium stats outputs were not found. "
            "Run python run_calcium_github.py first."
        )

    term_df = pd.read_csv(term_path)
    within_df = pd.read_csv(within_path)

    frames = [
        build_cluster_comparison_rows(term_df),
        build_interaction_rows(term_df),
        build_within_cluster_rows(within_df),
    ]
    combined = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    combined = standardize_effect_frame(combined)

    full = combined[
        [
            "window_label",
            "model_label",
            "section",
            "effect_label",
            "response_var",
            "term",
            "treatment",
            "cluster",
            "estimate",
            "SE",
            "z",
            "std_effect",
            "p_value",
            "p_value_display",
            "fdr_p_value",
            "fdr_p_value_display",
        ]
    ].copy()

    thesis = combined[
        [
            "model_label",
            "section",
            "effect_label",
            "estimate",
            "SE",
            "z",
            "std_effect",
            "p_value_display",
            "fdr_p_value_display",
        ]
    ].copy()
    for column in ["estimate", "SE", "z", "std_effect"]:
        thesis[column] = thesis[column].map(lambda value: round(float(value), 3) if pd.notna(value) else value)
    thesis = thesis.rename(
        columns={
            "model_label": "Stimulation window",
            "section": "Section",
            "effect_label": "Effect",
            "estimate": "Estimate",
            "SE": "SE",
            "z": "z",
            "std_effect": "Std effect",
            "p_value_display": "p value",
            "fdr_p_value_display": "p value (FDR)",
        }
    )
    return full, thesis


def main() -> None:
    full, thesis = build_tables()
    write_table(full, "calcium_auc_nested_mixedlm_effects_full")
    write_table(thesis, "calcium_auc_nested_mixedlm_effects_thesis_table")

    print("Wrote:")
    print(THESIS_DIR / "calcium_auc_nested_mixedlm_effects_full.csv")
    print(THESIS_DIR / "calcium_auc_nested_mixedlm_effects_full.tsv")
    print(THESIS_DIR / "calcium_auc_nested_mixedlm_effects_thesis_table.csv")
    print(THESIS_DIR / "calcium_auc_nested_mixedlm_effects_thesis_table.tsv")


if __name__ == "__main__":
    main()
