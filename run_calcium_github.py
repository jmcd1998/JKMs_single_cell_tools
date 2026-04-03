from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from clustered_stripped_lmm_analysis import (
    append_fdr_columns,
    append_term_fdr_columns,
    contrast_from_vector,
    fixed_effect_parts,
    mixedlm_summary_to_csv_frame,
    p_to_stars,
    summarise_group_structure,
    write_csv_with_lock_fallback,
)
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "original_data" / "calcium" / "out_registration_batch"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "calcium_full_output_V1"

TREAT_ORDER = ("MDL29951", "pranlukast", "HAMI3379")
TREAT_DISPLAY = {
    "MDL29951": "MDL29951",
    "pranlukast": "Pranlukast",
    "HAMI3379": "HAMI3379",
}
CLUSTERS = (0, 1, 2)
WINDOWS = ("w1", "w2", "w3")
MFI_COLS = ("PDGFRa_mfi_norm", "MBP_mfi_norm", "O4_mfi_norm")

INCLUDE_NONPOSITIVE_AUC = True
FRAME_SEC = 1.3

CLUSTER_COLORS = {
    0: "#28e561b4",
    1: "#ffc70e",
    2: "#d835bf",
}
PALETTE = [CLUSTER_COLORS[k] for k in CLUSTERS]
JITTER_RNG = np.random.default_rng(42)


def log_df(df: pd.DataFrame, name: str) -> None:
    print(f"{name}: shape={df.shape}, columns={list(df.columns)}")


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")


def resolve_data_dir(requested: Path | None = None) -> Path:
    candidates = [requested, DEFAULT_DATA_DIR]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"Could not find analysis data directory. Searched: {searched}")


def remove_outliers_iqr(
    df: pd.DataFrame,
    col: str,
    *,
    group_cols: list[str] | tuple[str, ...] | None = None,
    k: float = 1.5,
) -> pd.DataFrame:
    if group_cols is None:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        return df[(df[col] >= lo) & (df[col] <= hi)].copy()

    def _filt(group: pd.DataFrame) -> pd.DataFrame:
        q1 = group[col].quantile(0.25)
        q3 = group[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        return group[(group[col] >= lo) & (group[col] <= hi)]

    return (
        df.groupby(list(group_cols), group_keys=False, observed=True)
        .apply(_filt)
        .copy()
    )


def normalize_by_bio_mean(
    df_in: pd.DataFrame,
    cols: tuple[str, ...],
    *,
    group_col: str = "bio_rep",
) -> pd.DataFrame:
    out = df_in.copy()
    for col in cols:
        out[f"{col}_bio"] = out[col] / out.groupby(group_col)[col].transform("mean")
    return out


def load_bundle_tables(master_dir: Path, filename: str) -> pd.DataFrame:
    files = sorted(master_dir.rglob(filename), key=lambda path: (path.parent.name, path.name))
    if not files:
        raise FileNotFoundError(f"No '{filename}' files found under {master_dir}")
    return pd.concat(
        [pd.read_csv(path).assign(bundle=path.parent.name) for path in files],
        ignore_index=True,
    )


def prepare_combined_matched_cells(master_dir: Path, output_dir: Path) -> pd.DataFrame:
    df_concat_raw = load_bundle_tables(master_dir, "matched_cells_with_stain_metrics.csv")
    df_with_meta = df_concat_raw.copy()
    df_with_meta = df_with_meta.assign(
        bio_rep=df_with_meta["bundle"].str.extract(r"(N\d+)", expand=False),
        treatment=df_with_meta["bundle"].str.extract(r"(HAMI3379|MDL29951|pranlukast)", expand=False),
        tech_rep=df_with_meta["bundle"].str.extract(r"(T\d+)", expand=False),
    )
    df_with_meta["tech_rep_id"] = (
        df_with_meta["bio_rep"].astype(str) + "_" + df_with_meta["tech_rep"].astype(str)
    )
    df_with_meta.loc[
        df_with_meta["bio_rep"].isna() | df_with_meta["tech_rep"].isna(),
        "tech_rep_id",
    ] = pd.NA
    log_df(df_with_meta, "df_with_meta")

    out_csv = output_dir / "all_matched_cells_with_stain_metrics.csv"
    df_with_meta.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df_with_meta


def run_kmeans_curves(
    x_values: np.ndarray,
    *,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> tuple[list[float], list[float]]:
    wcss: list[float] = []
    sil: list[float] = []
    for k_value in k_range:
        kmeans = KMeans(n_clusters=k_value, n_init="auto", random_state=random_state)
        labels = kmeans.fit_predict(x_values)
        wcss.append(kmeans.inertia_)
        sil.append(silhouette_score(x_values, labels))
    return wcss, sil


def plot_curves(k_range: range, wcss: list[float], sil: list[float], title_prefix: str) -> plt.Figure:
    k_values = list(k_range)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(k_values, wcss, marker="o")
    axes[0].set_title(f"{title_prefix} Elbow")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("WCSS")
    axes[1].plot(k_values, sil, marker="o")
    axes[1].set_title(f"{title_prefix} Silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    fig.tight_layout()
    return fig


def cluster_cells(
    df_all: pd.DataFrame,
    output_dir: Path,
    fig_dir: Path,
    clustering_dir: Path,
) -> pd.DataFrame:
    df1 = df_all.copy()

    _tmp_mfi_raw = df1[["bio_rep", *MFI_COLS]].dropna().copy()
    _tmp_mfi_norm = normalize_by_bio_mean(_tmp_mfi_raw, MFI_COLS)

    x_mfi = _tmp_mfi_norm[[f"{col}_bio" for col in MFI_COLS]].to_numpy()
    x_mfi_scaled = StandardScaler().fit_transform(x_mfi)

    wcss, sil = run_kmeans_curves(x_mfi_scaled)
    kmeans_curve_df = pd.DataFrame(
        {
            "k": list(range(2, 11)),
            "wcss": wcss,
            "silhouette_score": sil,
        }
    )
    kmeans_curve_path = clustering_dir / "mfi_kmeans_curve_metrics.csv"
    kmeans_curve_df.to_csv(kmeans_curve_path, index=False)
    print(f"Saved: {kmeans_curve_path}")

    fig = plot_curves(range(2, 11), wcss, sil, "MFI")
    save_fig(fig, fig_dir / "mfi_elbow_silhouette.png")
    plt.close(fig)

    kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)
    _tmp_mfi_clustered = _tmp_mfi_norm.copy()
    _tmp_mfi_clustered["cluster_k3"] = kmeans.fit_predict(x_mfi_scaled)

    cluster_means = (
        _tmp_mfi_clustered.groupby("cluster_k3")[
            ["PDGFRa_mfi_norm_bio", "O4_mfi_norm_bio", "MBP_mfi_norm_bio"]
        ]
        .mean()
    )
    marker_of_cluster = cluster_means.idxmax(axis=1)
    cluster_map = {
        int(cluster_id): {"PDGFRa_mfi_norm_bio": 0, "O4_mfi_norm_bio": 1, "MBP_mfi_norm_bio": 2}[marker]
        for cluster_id, marker in marker_of_cluster.items()
    }
    cluster_label_map_df = pd.DataFrame(
        {
            "cluster_k3": [int(cluster_id) for cluster_id in cluster_means.index],
            "dominant_marker": [marker_of_cluster.loc[cluster_id] for cluster_id in cluster_means.index],
            "cluster_k3_present": [cluster_map[int(cluster_id)] for cluster_id in cluster_means.index],
        }
    )
    cluster_label_map_path = clustering_dir / "mfi_cluster_label_map.csv"
    cluster_label_map_df.to_csv(cluster_label_map_path, index=False)
    print(f"Saved: {cluster_label_map_path}")

    cluster_feature_means_df = cluster_means.reset_index().rename(columns={"cluster_k3": "cluster_k3"})
    cluster_feature_means_path = clustering_dir / "mfi_cluster_feature_means.csv"
    cluster_feature_means_df.to_csv(cluster_feature_means_path, index=False)
    print(f"Saved: {cluster_feature_means_path}")

    scaled_centers_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=[f"{col}_scaled_center" for col in [f"{feature}_bio" for feature in MFI_COLS]],
    ).reset_index(names="cluster_k3")
    scaled_centers_path = clustering_dir / "mfi_cluster_centers_scaled.csv"
    scaled_centers_df.to_csv(scaled_centers_path, index=False)
    print(f"Saved: {scaled_centers_path}")

    df1_clustered = df1.copy()
    df1_clustered.loc[_tmp_mfi_clustered.index, "cluster_k3"] = _tmp_mfi_clustered["cluster_k3"]

    df1_present = df1_clustered.copy()
    df1_present["cluster_k3_present"] = pd.to_numeric(
        df1_present["cluster_k3"],
        errors="coerce",
    ).map(cluster_map)
    log_df(df1_present, "df1_present")

    _tmp_present = _tmp_mfi_clustered.copy()
    _tmp_present["cluster_k3_present"] = _tmp_present["cluster_k3"].map(cluster_map)

    fig = plt.figure(figsize=(7, 5))
    axis = fig.add_subplot(111, projection="3d")
    z_values = _tmp_present["PDGFRa_mfi_norm_bio"]
    x_values = _tmp_present["MBP_mfi_norm_bio"]
    y_values = _tmp_present["O4_mfi_norm_bio"]
    colors = _tmp_present["cluster_k3_present"].map(CLUSTER_COLORS)
    axis.scatter(
        x_values,
        y_values,
        z_values,
        c=colors,
        s=10,
        alpha=0.8,
        edgecolor="0.1",
        linewidth=0.4,
    )
    axis.set_zlabel("PDGFRa (bio-norm)")
    axis.set_xlabel("MBP (bio-norm)")
    axis.set_ylabel("O4 (bio-norm)")
    axis.set_title("MFI Clusters (k=3) - Presentation Labels")
    save_fig(fig, fig_dir / "mfi_clusters_3d.png")
    plt.close(fig)

    clustered_path = output_dir / "DF1_cells_with_cluster_assignment.csv"
    present_path = output_dir / "DF1_cells_with_cluster_assignment_present.csv"
    df1_clustered.to_csv(clustered_path, index=False)
    df1_present.to_csv(present_path, index=False)
    print(f"Saved: {clustered_path}")
    print(f"Saved: {present_path}")

    return df1_present


def build_auc_clean_table(df1_present: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df2_present = df1_present.copy()

    df2_auc_base = df2_present.dropna(subset=["calcium AUC"]).copy()
    if INCLUDE_NONPOSITIVE_AUC:
        df2_auc_filtered = df2_auc_base.copy()
    else:
        df2_auc_filtered = df2_auc_base[df2_auc_base["calcium AUC"] > 0].copy()

    df2_auc_clean = remove_outliers_iqr(
        df2_auc_filtered,
        "calcium AUC",
        group_cols=["treatment", "cluster_k3_present"],
        k=1.5,
    )
    log_df(df2_auc_clean, "df2_auc_clean")
    print(
        f"{len(df2_present)} -> {len(df2_auc_clean)} "
        f"(include nonpositive AUC: {INCLUDE_NONPOSITIVE_AUC})"
    )
    return df2_present, df2_auc_clean


def auc_col_for(window_label: str, mode: str) -> str:
    if mode == "norm":
        return f"calcium_auc_norm_{window_label}"
    if mode == "raw":
        return f"calcium_auc_{window_label}"
    raise ValueError("mode must be 'raw' or 'norm'")


def auc_label(mode: str) -> str:
    return "Norm AUC" if mode == "norm" else "AUC"


def auc_ylim(mode: str) -> tuple[float, float]:
    return (0, 3.5) if mode == "norm" else (0, 250)


def plot_rt_scatter_per_treatment(df2_present: pd.DataFrame, fig_dir: Path) -> None:
    for treatment in TREAT_ORDER:
        df_treat_rt = df2_present[df2_present["treatment"].str.lower() == treatment.lower()].copy()
        df_rt_num = df_treat_rt.assign(
            cluster_k3_present=pd.to_numeric(df_treat_rt["cluster_k3_present"], errors="coerce")
        )
        df_rt_clean = df_rt_num.dropna(subset=["calcium response time", "cluster_k3_present"]).copy()
        df_rt_clean["cluster_k3_present"] = df_rt_clean["cluster_k3_present"].astype(int)

        fig, axis = plt.subplots(1, 1, figsize=(5, 4))
        for cluster in CLUSTERS:
            subset = df_rt_clean[df_rt_clean["cluster_k3_present"] == cluster]
            x_values = np.full(len(subset), cluster) + JITTER_RNG.uniform(-0.15, 0.15, size=len(subset))
            axis.scatter(
                x_values,
                subset["calcium response time"],
                s=12,
                alpha=0.7,
                color=CLUSTER_COLORS[cluster],
            )
        axis.set_title(f"{treatment} - Calcium Response Time")
        axis.set_xlabel("Cluster")
        axis.set_xticks(CLUSTERS)
        axis.set_xticklabels([str(cluster) for cluster in CLUSTERS])
        axis.set_ylabel("calcium response time")
        save_fig(fig, fig_dir / f"rt_scatter_{treatment}.png")
        plt.close(fig)


def plot_auc_violin_per_treatment(df2_auc_clean: pd.DataFrame, fig_dir: Path) -> None:
    for mode in ("raw", "norm"):
        suffix = "_norm" if mode == "norm" else ""
        for treatment in TREAT_ORDER:
            for window_label in WINDOWS:
                auc_col = auc_col_for(window_label, mode)
                df_treat_auc = df2_auc_clean[df2_auc_clean["treatment"].str.lower() == treatment.lower()].copy()
                df_auc_num = df_treat_auc.assign(
                    cluster_k3_present=pd.to_numeric(df_treat_auc["cluster_k3_present"], errors="coerce")
                )
                df_auc_clean2 = df_auc_num.dropna(subset=[auc_col, "cluster_k3_present"]).copy()
                df_auc_clean2["cluster_k3_present"] = df_auc_clean2["cluster_k3_present"].astype(int)

                fig, axis = plt.subplots(1, 1, figsize=(5, 4))
                sns.violinplot(
                    data=df_auc_clean2,
                    x="cluster_k3_present",
                    y=auc_col,
                    order=CLUSTERS,
                    palette=PALETTE,
                    inner="box",
                    cut=0,
                    ax=axis,
                )
                axis.set_title(f"{treatment} - {auc_label(mode)} {window_label.upper()}")
                axis.set_xlabel("Cluster")
                axis.set_ylabel(auc_col)
                axis.set_ylim(auc_ylim(mode))
                save_fig(fig, fig_dir / f"auc_violin_{treatment}_{window_label}{suffix}.png")
                plt.close(fig)


def plot_auc_rt_grid(df2_present: pd.DataFrame, df2_auc_clean: pd.DataFrame, fig_dir: Path) -> None:
    for window_label in WINDOWS:
        auc_col = auc_col_for(window_label, "raw")
        rt_col = f"calcium_response_time_{window_label}"

        fig, axes = plt.subplots(2, len(TREAT_ORDER), figsize=(5 * len(TREAT_ORDER), 8), sharex=False, sharey="row")
        if len(TREAT_ORDER) == 1:
            axes = np.array(axes).reshape(2, 1)

        for column_index, treatment in enumerate(TREAT_ORDER):
            df_treat_auc = df2_auc_clean[df2_auc_clean["treatment"].str.lower() == treatment.lower()].copy()
            df_auc_num = df_treat_auc.assign(
                cluster_k3_present=pd.to_numeric(df_treat_auc["cluster_k3_present"], errors="coerce")
            )
            df_auc_clean2 = df_auc_num.dropna(subset=[auc_col, "cluster_k3_present"]).copy()
            df_auc_clean2["cluster_k3_present"] = df_auc_clean2["cluster_k3_present"].astype(int)

            sns.violinplot(
                data=df_auc_clean2,
                x="cluster_k3_present",
                y=auc_col,
                order=CLUSTERS,
                palette=PALETTE,
                inner="box",
                cut=0,
                ax=axes[0, column_index],
            )
            axes[0, column_index].set_title(f"{treatment} - AUC {window_label.upper()}")
            axes[0, column_index].set_xlabel("")
            axes[0, column_index].set_ylabel(auc_col if column_index == 0 else "")
            axes[0, column_index].set_xticks(CLUSTERS)
            axes[0, column_index].set_xticklabels([])
            axes[0, column_index].set_ylim(auc_ylim("raw"))

            df_treat_rt = df2_present[df2_present["treatment"].str.lower() == treatment.lower()].copy()
            df_rt_num = df_treat_rt.assign(
                cluster_k3_present=pd.to_numeric(df_treat_rt["cluster_k3_present"], errors="coerce")
            )
            df_rt_clean = df_rt_num.dropna(subset=[rt_col, "cluster_k3_present"]).copy()
            df_rt_clean["cluster_k3_present"] = df_rt_clean["cluster_k3_present"].astype(int)

            axis = axes[1, column_index]
            for cluster in CLUSTERS:
                subset = df_rt_clean[df_rt_clean["cluster_k3_present"] == cluster]
                x_values = np.full(len(subset), cluster) + JITTER_RNG.uniform(-0.15, 0.15, size=len(subset))
                axis.scatter(x_values, subset[rt_col], s=12, alpha=0.7, color=CLUSTER_COLORS[cluster])

            axis.set_title(f"{treatment} - Response Time {window_label.upper()}")
            axis.set_xlabel("Cluster")
            axis.set_xticks(CLUSTERS)
            axis.set_xticklabels([str(cluster) for cluster in CLUSTERS])
            axis.set_ylabel(rt_col if column_index == 0 else "")

        fig.tight_layout()
        save_fig(fig, fig_dir / f"auc_rt_grid_{window_label}.png")
        plt.close(fig)


def prepare_auc_model_df(df2_auc_clean: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    model_df = df2_auc_clean.copy()
    model_df = model_df.assign(
        cluster_k3_present=pd.to_numeric(model_df["cluster_k3_present"], errors="coerce")
    )
    model_df = model_df.dropna(
        subset=[auc_col, "cluster_k3_present", "bio_rep", "tech_rep_id", "treatment"]
    ).copy()
    model_df["cluster_k3_present"] = pd.Categorical(
        model_df["cluster_k3_present"].astype(int),
        categories=list(CLUSTERS),
        ordered=True,
    )
    model_df["treatment"] = pd.Categorical(
        model_df["treatment"],
        categories=list(TREAT_ORDER),
        ordered=True,
    )
    model_df["tech_rep_id"] = model_df["tech_rep_id"].astype(str)
    model_df["bio_rep"] = model_df["bio_rep"].astype(str)
    return model_df


def build_auc_formula(y_col: str) -> str:
    return (
        f'Q("{y_col}") ~ '
        'C(treatment, Treatment(reference="MDL29951")) * C(cluster_k3_present)'
    )


def collect_warning_messages(caught_warnings: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for warning in caught_warnings:
        message = str(warning.message).strip()
        if message and message not in messages:
            messages.append(message)
    return messages


def fit_auc_mixedlm(model_df: pd.DataFrame, y_col: str):
    formula = build_auc_formula(y_col)
    last_exception: Exception | None = None

    def _fit_with_method(method: str | None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df["bio_rep"],
                vc_formula={"tech_rep_id": "0 + C(tech_rep_id)"},
            )
            if method is None:
                fit = model.fit(reml=True)
                optimizer = "default"
            else:
                fit = model.fit(reml=True, method=method, disp=False)
                optimizer = method

        setattr(fit, "_codex_formula", formula)
        setattr(fit, "_codex_optimizer", optimizer)
        setattr(fit, "_codex_random_structure", "bio_rep + tech_rep_id")
        setattr(fit, "_codex_warning_messages", collect_warning_messages(caught))
        return fit

    try:
        return _fit_with_method(None)
    except Exception as exc:
        last_exception = exc

    for method in ("lbfgs", "bfgs", "cg"):
        try:
            return _fit_with_method(method)
        except Exception as exc:
            last_exception = exc

    raise RuntimeError(f"Unable to fit MixedLM for {y_col}") from last_exception


def residual_sigma(model_fit) -> float:
    scale = getattr(model_fit, "scale", np.nan)
    return float(np.sqrt(scale)) if np.isfinite(scale) and scale > 0 else np.nan


def build_auc_fit_summary_row(
    *,
    model_df: pd.DataFrame,
    data_csv: Path,
    window_label: str,
    auc_col: str,
    model_fit,
) -> dict[str, object]:
    row = {
        "window_label": window_label,
        "response_var": auc_col,
        "model_kind": "two_term",
        "formula": getattr(model_fit, "_codex_formula", build_auc_formula(auc_col)),
        "random_structure": getattr(model_fit, "_codex_random_structure", "bio_rep + tech_rep_id"),
        "field_unit_col": "tech_rep_id",
        "data_csv": str(data_csv),
        "status": "ok",
        "converged": bool(getattr(model_fit, "converged", False)),
        "optimizer": getattr(model_fit, "_codex_optimizer", "unknown"),
        "warning_count": len(getattr(model_fit, "_codex_warning_messages", [])),
        "warning_messages": " | ".join(getattr(model_fit, "_codex_warning_messages", [])),
    }
    row.update(
        summarise_group_structure(
            model_df,
            group_col="bio_rep",
            fov_col="tech_rep_id",
            treatment_col="treatment",
        )
    )
    return row


def write_calcium_mixedlm_full_report(
    model_fit,
    stats_dir: Path,
    *,
    report_name: str,
    window_label: str,
    formula: str,
    response_var: str,
    model_kind: str,
    random_structure: str,
    data: pd.DataFrame,
) -> tuple[Path, Path]:
    report_dir = stats_dir / "mixedlm_full_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_stem = safe_name(report_name)
    report_path = report_dir / f"{report_stem}.txt"
    report_csv_path = report_dir / f"{report_stem}.csv"

    metadata = {
        "report_name": report_name,
        "window_label": window_label,
        "response_var": response_var,
        "model_kind": model_kind,
        "formula": formula,
        "optimizer": getattr(model_fit, "_codex_optimizer", "unknown"),
        "random_structure": random_structure,
        "n_rows": len(data),
        "n_biological_replicates": data["bio_rep"].nunique() if "bio_rep" in data.columns else np.nan,
        "n_fields_or_tech_reps": data["tech_rep_id"].nunique() if "tech_rep_id" in data.columns else np.nan,
        "converged": bool(getattr(model_fit, "converged", False)),
        "warning_messages": " | ".join(getattr(model_fit, "_codex_warning_messages", [])),
    }

    summary = model_fit.summary()
    rows = mixedlm_summary_to_csv_frame(summary, metadata)
    header_lines = [f"{key}: {value}" for key, value in metadata.items()]
    report_path.write_text(
        "\n".join(header_lines) + "\n\n" + summary.as_text() + "\n",
        encoding="utf-8",
    )
    rows.to_csv(report_csv_path, index=False)
    return report_path, report_csv_path


def build_auc_term_effects_df(
    model_fit,
    *,
    window_label: str,
    auc_col: str,
    fit_summary_row: dict[str, object],
) -> pd.DataFrame:
    params, bse, pvalues, _ = fixed_effect_parts(model_fit)
    sigma = residual_sigma(model_fit)
    return pd.DataFrame(
        {
            "window_label": window_label,
            "response_var": auc_col,
            "model_kind": "two_term",
            "term": params.index.astype(str),
            "estimate": params.values,
            "SE": bse.values,
            "z": params.values / bse.values,
            "p_value": pvalues.values,
            "std_effect": params.values / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
            "is_interaction": params.index.astype(str).str.contains(":C\\(cluster_k3_present\\)"),
            "optimizer": getattr(model_fit, "_codex_optimizer", "unknown"),
            "converged": bool(getattr(model_fit, "converged", False)),
            "avg_group_size": fit_summary_row["avg_group_size"],
            "avg_fov_size": fit_summary_row["avg_fov_size"],
            "warning_count": fit_summary_row["warning_count"],
            "random_structure": fit_summary_row["random_structure"],
        }
    )


def build_auc_within_cluster_contrasts_df(
    model_fit,
    model_df: pd.DataFrame,
    *,
    window_label: str,
    auc_col: str,
    fit_summary_row: dict[str, object],
) -> pd.DataFrame:
    params, _, _, cov = fixed_effect_parts(model_fit)
    names = list(params.index)
    sigma = residual_sigma(model_fit)

    treatment_levels = list(model_df["treatment"].cat.categories)
    cluster_levels = list(model_df["cluster_k3_present"].cat.categories)
    rows: list[dict[str, object]] = []
    for treatment in treatment_levels:
        if treatment == TREAT_ORDER[0]:
            continue
        treatment_name = f'C(treatment, Treatment(reference="MDL29951"))[T.{treatment}]'
        if treatment_name not in names:
            raise RuntimeError(
                f"Expected treatment coefficient '{treatment_name}' for '{auc_col}', but it was absent."
            )
        for cluster in cluster_levels:
            vector = np.zeros(len(names), dtype=float)
            vector[names.index(treatment_name)] = 1.0
            interaction_name = (
                f'C(treatment, Treatment(reference="MDL29951"))[T.{treatment}]'
                f":C(cluster_k3_present)[T.{cluster}]"
            )
            if interaction_name in names:
                vector[names.index(interaction_name)] = 1.0
            estimate, se, z_value, p_value = contrast_from_vector(params, cov, vector)
            rows.append(
                {
                    "window_label": window_label,
                    "response_var": auc_col,
                    "model_kind": "two_term",
                    "treatment": treatment,
                    "cluster": cluster,
                    "within_estimate": estimate,
                    "within_SE": se,
                    "within_z": z_value,
                    "within_p_value": p_value,
                    "within_p_stars": p_to_stars(p_value),
                    "std_effect_within_cluster": estimate / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                    "scale": "raw",
                    "avg_group_size": fit_summary_row["avg_group_size"],
                    "avg_fov_size": fit_summary_row["avg_fov_size"],
                    "converged": fit_summary_row["converged"],
                    "optimizer": fit_summary_row["optimizer"],
                    "warning_count": fit_summary_row["warning_count"],
                    "random_structure": fit_summary_row["random_structure"],
                }
            )
    return pd.DataFrame(rows)


def build_auc_emm_grid(
    model_fit,
    model_df: pd.DataFrame,
    *,
    window_label: str,
    auc_col: str,
    fit_summary_row: dict[str, object],
) -> pd.DataFrame:
    params, _, _, cov = fixed_effect_parts(model_fit)
    names = list(params.index)
    sigma = residual_sigma(model_fit)
    if "Intercept" not in names:
        raise RuntimeError(f"Expected an Intercept term for '{auc_col}', but it was absent.")

    treatment_levels = list(model_df["treatment"].cat.categories)
    cluster_levels = list(model_df["cluster_k3_present"].cat.categories)
    rows: list[dict[str, object]] = []
    for treatment in treatment_levels:
        for cluster in cluster_levels:
            vector = np.zeros(len(names), dtype=float)
            vector[names.index("Intercept")] = 1.0

            treatment_name = f'C(treatment, Treatment(reference="MDL29951"))[T.{treatment}]'
            cluster_name = f"C(cluster_k3_present)[T.{cluster}]"
            interaction_name = (
                f'C(treatment, Treatment(reference="MDL29951"))[T.{treatment}]'
                f":C(cluster_k3_present)[T.{cluster}]"
            )
            if treatment_name in names:
                vector[names.index(treatment_name)] = 1.0
            if cluster_name in names:
                vector[names.index(cluster_name)] = 1.0
            if interaction_name in names:
                vector[names.index(interaction_name)] = 1.0

            estimate, se, _, _ = contrast_from_vector(params, cov, vector)
            rows.append(
                {
                    "window_label": window_label,
                    "response_var": auc_col,
                    "model_kind": "two_term",
                    "treatment": treatment,
                    "cluster": cluster,
                    "emm": estimate,
                    "SE": se,
                    "CI_lower": estimate - 1.96 * se if np.isfinite(se) else np.nan,
                    "CI_upper": estimate + 1.96 * se if np.isfinite(se) else np.nan,
                    "emm_std_units": estimate / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                    "avg_group_size": fit_summary_row["avg_group_size"],
                    "avg_fov_size": fit_summary_row["avg_fov_size"],
                    "converged": fit_summary_row["converged"],
                    "optimizer": fit_summary_row["optimizer"],
                    "warning_count": fit_summary_row["warning_count"],
                    "random_structure": fit_summary_row["random_structure"],
                }
            )
    return pd.DataFrame(rows)


def interaction_terms_df(model_fit, window_label: str, auc_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sigma = residual_sigma(model_fit)
    for term in model_fit.params.index:
        if "C(treatment, Treatment(reference=\"MDL29951\"))" in term and ":C(cluster_k3_present)" in term:
            effect = float(model_fit.params[term])
            rows.append(
                {
                    "window": window_label,
                    "auc_col": auc_col,
                    "term": term,
                    "effect": effect,
                    "std_effect": effect / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                    "SE": float(model_fit.bse[term]),
                    "t": float(model_fit.tvalues[term]),
                    "p_uncorrected": float(model_fit.pvalues[term]),
                    "residual_sigma": sigma,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = multipletests(df["p_uncorrected"], method="holm")[1]
    return df


def write_calcium_lmm_supplementary_summary(fit_summary_df: pd.DataFrame, stats_dir: Path) -> Path:
    combined = fit_summary_df.copy()
    combined["dataset"] = "calcium_multiplex"
    combined["analysis_name"] = "calcium_auc_nested_raw"

    rename_map = {
        "n_rows": "n_cells_used",
        "n_biological_replicates": "n_biol_reps",
        "n_fovs": "n_fields_or_tech_reps_used",
        "n_group_treatment_combos": "n_biol_rep_treatment_groups",
        "avg_cells_per_biological_replicate_total": "avg_cells_per_biol_rep_total",
        "median_cells_per_biological_replicate_total": "median_cells_per_biol_rep_total",
        "avg_cells_per_biological_replicate_treatment": "avg_cells_per_biol_rep_treatment_group",
        "median_cells_per_biological_replicate_treatment": "median_cells_per_biol_rep_treatment_group",
        "avg_cells_per_fov": "avg_cells_per_field_or_tech_rep",
        "median_cells_per_fov": "median_cells_per_field_or_tech_rep",
        "avg_fovs_per_biological_replicate": "avg_fields_or_tech_reps_per_biol_rep",
        "median_fovs_per_biological_replicate": "median_fields_or_tech_reps_per_biol_rep",
    }
    for source_col, target_col in rename_map.items():
        if source_col in combined.columns and target_col not in combined.columns:
            combined[target_col] = combined[source_col]

    ordered_cols = [
        "dataset",
        "analysis_name",
        "model_kind",
        "window_label",
        "response_var",
        "formula",
        "random_structure",
        "field_unit_col",
        "status",
        "converged",
        "optimizer",
        "warning_count",
        "warning_messages",
        "data_csv",
        "n_cells_used",
        "n_biol_reps",
        "n_fields_or_tech_reps_used",
        "n_biol_rep_treatment_groups",
        "avg_cells_per_biol_rep_total",
        "median_cells_per_biol_rep_total",
        "avg_cells_per_biol_rep_treatment_group",
        "median_cells_per_biol_rep_treatment_group",
        "avg_cells_per_field_or_tech_rep",
        "median_cells_per_field_or_tech_rep",
        "avg_fields_or_tech_reps_per_biol_rep",
        "median_fields_or_tech_reps_per_biol_rep",
    ]
    remaining_cols = [col for col in combined.columns if col not in ordered_cols]
    combined = combined[[col for col in ordered_cols if col in combined.columns] + remaining_cols].copy()
    combined.sort_values(["window_label", "response_var"], inplace=True, na_position="last")
    combined.reset_index(drop=True, inplace=True)

    out_path = stats_dir / "lmm_supplementary_model_observation_summary.csv"
    write_csv_with_lock_fallback(combined, out_path)
    return out_path


def run_auc_mixedlm_for_window(
    df2_auc_clean: pd.DataFrame,
    auc_col: str,
    window_label: str,
    *,
    data_csv: Path,
) -> dict[str, object]:
    model_df = prepare_auc_model_df(df2_auc_clean, auc_col)
    log_df(model_df, f"df_auc_model_cat_{window_label}")
    model_fit = fit_auc_mixedlm(model_df, auc_col)
    sigma = residual_sigma(model_fit)
    fit_summary_row = build_auc_fit_summary_row(
        model_df=model_df,
        data_csv=data_csv,
        window_label=window_label,
        auc_col=auc_col,
        model_fit=model_fit,
    )

    print(f"\n=== Nested MixedLM raw AUC window {window_label} ===")
    print(model_fit.summary())

    fixed_effects = model_fit.model.exog_names

    def contrast_vector(weights: dict[str, float]) -> np.ndarray:
        vector = np.zeros(len(fixed_effects))
        for name, weight in weights.items():
            vector[fixed_effects.index(name)] = weight
        return np.atleast_2d(vector)

    rows: list[dict[str, object]] = []
    for cluster in CLUSTERS:
        for i, treatment_a in enumerate(TREAT_ORDER):
            for treatment_b in TREAT_ORDER[i + 1 :]:
                weights: dict[str, float] = {}
                if treatment_a != TREAT_ORDER[0]:
                    weights[f"C(treatment, Treatment(reference=\"MDL29951\"))[T.{treatment_a}]"] = -1
                if treatment_b != TREAT_ORDER[0]:
                    weights[f"C(treatment, Treatment(reference=\"MDL29951\"))[T.{treatment_b}]"] = 1
                if cluster != 0:
                    if treatment_a != TREAT_ORDER[0]:
                        weights[
                            f"C(treatment, Treatment(reference=\"MDL29951\"))[T.{treatment_a}]"
                            f":C(cluster_k3_present)[T.{cluster}]"
                        ] = -1
                    if treatment_b != TREAT_ORDER[0]:
                        weights[
                            f"C(treatment, Treatment(reference=\"MDL29951\"))[T.{treatment_b}]"
                            f":C(cluster_k3_present)[T.{cluster}]"
                        ] = 1

                test_result = model_fit.t_test(contrast_vector(weights))
                effect = float(test_result.effect)
                rows.append(
                    {
                        "window": window_label,
                        "auc_col": auc_col,
                        "comparison": f"{treatment_b} vs {treatment_a} @ cluster {cluster}",
                        "effect": effect,
                        "std_effect": effect / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                        "SE": float(test_result.sd),
                        "t": float(test_result.tvalue),
                        "p_uncorrected": float(test_result.pvalue),
                        "residual_sigma": sigma,
                    }
                )

    pairs_df = pd.DataFrame(rows)
    pairs_df["p_holm"] = multipletests(pairs_df["p_uncorrected"], method="holm")[1]
    interactions_df = interaction_terms_df(model_fit, window_label, auc_col)
    return {
        "window_label": window_label,
        "response_var": auc_col,
        "model_df": model_df,
        "model_fit": model_fit,
        "formula": getattr(model_fit, "_codex_formula", build_auc_formula(auc_col)),
        "random_structure": getattr(model_fit, "_codex_random_structure", "bio_rep + tech_rep_id"),
        "fit_summary_row": fit_summary_row,
        "pairs_df": pairs_df,
        "interactions_df": interactions_df,
        "term_df": build_auc_term_effects_df(
            model_fit,
            window_label=window_label,
            auc_col=auc_col,
            fit_summary_row=fit_summary_row,
        ),
        "within_df": build_auc_within_cluster_contrasts_df(
            model_fit,
            model_df,
            window_label=window_label,
            auc_col=auc_col,
            fit_summary_row=fit_summary_row,
        ),
        "emm_df": build_auc_emm_grid(
            model_fit,
            model_df,
            window_label=window_label,
            auc_col=auc_col,
            fit_summary_row=fit_summary_row,
        ),
    }


def run_auc_models(df2_auc_clean: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    data_csv = output_dir / "DF1_cells_with_cluster_assignment_present.csv"

    pairwise_frames: list[pd.DataFrame] = []
    interaction_frames: list[pd.DataFrame] = []
    term_frames: list[pd.DataFrame] = []
    emm_frames: list[pd.DataFrame] = []
    within_frames: list[pd.DataFrame] = []
    fit_summary_rows: list[dict[str, object]] = []
    saved_paths: dict[str, Path] = {}

    for window_label in WINDOWS:
        auc_col = auc_col_for(window_label, "raw")
        result = run_auc_mixedlm_for_window(
            df2_auc_clean,
            auc_col,
            window_label,
            data_csv=data_csv,
        )
        pairs_df = result["pairs_df"]
        interactions_df = result["interactions_df"]
        pairwise_frames.append(pairs_df)
        interaction_frames.append(interactions_df)
        term_frames.append(result["term_df"])
        emm_frames.append(result["emm_df"])
        within_frames.append(result["within_df"])
        fit_summary_rows.append(result["fit_summary_row"])

        report_name = f"two_term_lmm_{result['response_var']}"
        report_txt_path, report_csv_path = write_calcium_mixedlm_full_report(
            result["model_fit"],
            stats_dir,
            report_name=report_name,
            window_label=window_label,
            formula=result["formula"],
            response_var=result["response_var"],
            model_kind="two_term",
            random_structure=result["random_structure"],
            data=result["model_df"],
        )
        saved_paths[f"full_report_txt_{window_label}"] = report_txt_path
        saved_paths[f"full_report_csv_{window_label}"] = report_csv_path

        pair_path = output_dir / f"auc_mixedlm_pairs_raw_{window_label}.csv"
        interaction_path = output_dir / f"auc_mixedlm_interactions_raw_{window_label}.csv"
        write_csv_with_lock_fallback(pairs_df, pair_path)
        write_csv_with_lock_fallback(interactions_df, interaction_path)
        print(f"Saved: {pair_path}")
        print(f"Saved: {interaction_path}")
        saved_paths[f"pairs_{window_label}"] = pair_path
        saved_paths[f"interactions_{window_label}"] = interaction_path

    all_pairs = pd.concat(pairwise_frames, ignore_index=True)
    all_interactions = pd.concat(interaction_frames, ignore_index=True)
    all_pairs_path = output_dir / "auc_mixedlm_pairs_raw_all_windows.csv"
    all_interactions_path = output_dir / "auc_mixedlm_interactions_raw_all_windows.csv"
    write_csv_with_lock_fallback(all_pairs, all_pairs_path)
    write_csv_with_lock_fallback(all_interactions, all_interactions_path)
    print(f"Saved: {all_pairs_path}")
    print(f"Saved: {all_interactions_path}")
    saved_paths["pairs_all"] = all_pairs_path
    saved_paths["interactions_all"] = all_interactions_path

    term_df = pd.concat(term_frames, ignore_index=True) if term_frames else pd.DataFrame()
    if not term_df.empty:
        term_df = append_term_fdr_columns(term_df)
        term_df.sort_values(["window_label", "p_value", "term"], inplace=True, na_position="last")
        term_df.reset_index(drop=True, inplace=True)
        term_path = stats_dir / "two_term_lmm_term_effects.csv"
        write_csv_with_lock_fallback(term_df, term_path)
        saved_paths["two_term_term_effects"] = term_path
        for response_var, response_term_df in term_df.groupby("response_var", sort=False):
            zscore_path = stats_dir / f"two_term_lmm_term_zscores_{response_var}.csv"
            write_csv_with_lock_fallback(response_term_df, zscore_path)
            saved_paths[f"two_term_term_zscores_{response_var}"] = zscore_path

    emm_df = pd.concat(emm_frames, ignore_index=True) if emm_frames else pd.DataFrame()
    if not emm_df.empty:
        emm_df.sort_values(["window_label", "cluster", "treatment"], inplace=True, na_position="last")
        emm_df.reset_index(drop=True, inplace=True)
        emm_path = stats_dir / "two_term_lmm_emms.csv"
        write_csv_with_lock_fallback(emm_df, emm_path)
        saved_paths["two_term_emms"] = emm_path

    within_df = pd.concat(within_frames, ignore_index=True) if within_frames else pd.DataFrame()
    if not within_df.empty:
        within_df = append_fdr_columns(
            within_df,
            p_value_col="within_p_value",
            output_col="within_fdr_p_value",
            stars_output_col="within_fdr_p_stars",
        )
        within_df.sort_values(["window_label", "cluster", "treatment"], inplace=True, na_position="last")
        within_df.reset_index(drop=True, inplace=True)
        within_path = stats_dir / "two_term_lmm_within_cluster_contrasts.csv"
        write_csv_with_lock_fallback(within_df, within_path)
        saved_paths["two_term_within_cluster_contrasts"] = within_path

    fit_summary_df = pd.DataFrame(fit_summary_rows)
    if not fit_summary_df.empty:
        fit_summary_df.sort_values(["window_label", "response_var"], inplace=True, na_position="last")
        fit_summary_df.reset_index(drop=True, inplace=True)
    fit_summary_path = stats_dir / "two_term_lmm_model_fit_summary.csv"
    write_csv_with_lock_fallback(fit_summary_df, fit_summary_path)
    saved_paths["two_term_model_fit_summary"] = fit_summary_path

    supplementary_path = write_calcium_lmm_supplementary_summary(fit_summary_df, stats_dir)
    saved_paths["lmm_supplementary_summary"] = supplementary_path
    saved_paths["stats_dir"] = stats_dir
    return saved_paths


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def plot_average_traces(master_dir: Path, df1_present: pd.DataFrame, fig_dir: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
        }
    )

    df_ts = load_bundle_tables(master_dir, "calcium_timeseries_long.csv")
    log_df(df_ts, "df_ts")

    df_cells = df1_present.copy()
    df_cells["calcium_label"] = pd.to_numeric(df_cells["calcium_label"], errors="coerce")
    df_cells["cluster_k3_present"] = pd.to_numeric(df_cells["cluster_k3_present"], errors="coerce")
    df_cells = df_cells.dropna(subset=["calcium_label", "cluster_k3_present", "treatment"]).copy()
    if "overlap_px" in df_cells.columns:
        df_cells = df_cells.sort_values("overlap_px", ascending=False)
    df_cells = df_cells.drop_duplicates(subset=["bundle", "calcium_label"]).copy()
    df_cells["calcium_label"] = df_cells["calcium_label"].astype(int)
    df_cells["cluster_k3_present"] = df_cells["cluster_k3_present"].astype(int)

    df_ts["cell_id"] = pd.to_numeric(df_ts["cell_id"], errors="coerce")
    df_ts = df_ts.dropna(subset=["cell_id"]).copy()
    df_ts["cell_id"] = df_ts["cell_id"].astype(int)
    if "t" in df_ts.columns:
        df_ts["t"] = pd.to_numeric(df_ts["t"], errors="coerce")

    df_ts_meta = df_ts.merge(
        df_cells[["bundle", "calcium_label", "cluster_k3_present", "treatment"]],
        left_on=["bundle", "cell_id"],
        right_on=["bundle", "calcium_label"],
        how="inner",
    )
    log_df(df_ts_meta, "df_ts_meta")
    df_ts_meta = df_ts_meta.dropna(subset=["t"]).copy()
    df_ts_meta["t_sec"] = df_ts_meta["t"] * FRAME_SEC

    def trace_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        summary = (
            df.groupby(group_cols + ["t_sec"])["dff"]
            .agg(["mean", "count", "std"])
            .reset_index()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        summary["ci95"] = 1.96 * summary["sem"]
        return summary

    stats_treat = trace_stats(df_ts_meta, ["treatment"])
    stats_treat_cluster = trace_stats(df_ts_meta, ["treatment", "cluster_k3_present"])

    y_low = min(
        (stats_treat["mean"] - stats_treat["ci95"]).min(),
        (stats_treat_cluster["mean"] - stats_treat_cluster["ci95"]).min(),
    )
    y_high = max(
        (stats_treat["mean"] + stats_treat["ci95"]).max(),
        (stats_treat_cluster["mean"] + stats_treat_cluster["ci95"]).max(),
    )
    if np.isfinite(y_low) and np.isfinite(y_high):
        pad = 0.05 * (y_high - y_low) if y_high > y_low else 0.1
        y_lim = (y_low - pad, y_high + pad)
    else:
        y_lim = None

    x_low = min(stats_treat["t_sec"].min(), stats_treat_cluster["t_sec"].min())
    x_high = max(stats_treat["t_sec"].max(), stats_treat_cluster["t_sec"].max())
    if np.isfinite(x_low) and np.isfinite(x_high):
        x_lim = (x_low, x_high)
    else:
        x_lim = None

    for treatment in TREAT_ORDER:
        fig, axis = plt.subplots(figsize=(5, 4))
        subset_all = stats_treat[stats_treat["treatment"].str.lower() == treatment.lower()]
        if not subset_all.empty:
            axis.plot(subset_all["t_sec"], subset_all["mean"], color="black", lw=2.5)
            axis.fill_between(
                subset_all["t_sec"],
                subset_all["mean"] - subset_all["ci95"],
                subset_all["mean"] + subset_all["ci95"],
                color="black",
                alpha=0.12,
            )

        for cluster in CLUSTERS:
            subset_cluster = stats_treat_cluster[
                (stats_treat_cluster["treatment"].str.lower() == treatment.lower())
                & (stats_treat_cluster["cluster_k3_present"] == cluster)
            ]
            if subset_cluster.empty:
                continue
            color = CLUSTER_COLORS[cluster]
            axis.plot(subset_cluster["t_sec"], subset_cluster["mean"], color=color, lw=2.0)
            axis.fill_between(
                subset_cluster["t_sec"],
                subset_cluster["mean"] - subset_cluster["ci95"],
                subset_cluster["mean"] + subset_cluster["ci95"],
                color=color,
                alpha=0.16,
            )

        axis.set_title(f"{treatment} - Mean dF/F0 Trace")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("dF/F0")
        if y_lim is not None:
            axis.set_ylim(y_lim)
        if x_lim is not None:
            axis.set_xlim(x_lim)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        fig.tight_layout()
        save_fig(fig, fig_dir / f"traces_mean_ci_{safe_name(treatment)}.png")
        plt.close(fig)


def qq_plot(ax: plt.Axes, resid: pd.Series, title: str, point_alpha: float, point_size: int) -> float:
    (osm, osr), (slope, intercept, r_value) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=point_size, alpha=point_alpha)
    ax.plot(osm, slope * osm + intercept, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    return float(r_value)


def run_residual_diagnostics(df2_auc_clean: pd.DataFrame, output_dir: Path, fig_dir: Path) -> None:
    diag_rows: list[dict[str, object]] = []
    point_alpha = 0.25
    point_size = 8
    fig, axes = plt.subplots(len(WINDOWS), 4, figsize=(18, 4.5 * len(WINDOWS)))
    if len(WINDOWS) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, window_label in enumerate(WINDOWS):
        auc_col = auc_col_for(window_label, "raw")
        df_raw = prepare_auc_model_df(df2_auc_clean, auc_col)

        raw_fit = fit_auc_mixedlm(df_raw, auc_col)
        resid_raw = pd.Series(raw_fit.resid).astype(float)
        fitted_raw = pd.Series(raw_fit.fittedvalues).astype(float)

        log_col = f"{auc_col}_log"
        min_auc = float(df_raw[auc_col].min())
        n_nonpositive = int((df_raw[auc_col] <= 0).sum())
        shift_used = (-min_auc + 1e-6) if min_auc <= 0 else 0.0

        df_log = df_raw.copy()
        df_log[log_col] = np.log(df_log[auc_col] + shift_used)
        log_fit = fit_auc_mixedlm(df_log, log_col)
        resid_log = pd.Series(log_fit.resid).astype(float)
        fitted_log = pd.Series(log_fit.fittedvalues).astype(float)

        axis = axes[row_index, 0]
        axis.scatter(fitted_raw, resid_raw, s=point_size, alpha=point_alpha)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{window_label.upper()} raw: residuals vs fitted")
        axis.set_xlabel("Fitted")
        axis.set_ylabel("Residual")

        qq_r_raw = qq_plot(
            axes[row_index, 1],
            resid_raw,
            f"{window_label.upper()} raw: QQ",
            point_alpha,
            point_size,
        )
        axes[row_index, 1].text(
            0.02,
            0.98,
            f"r={qq_r_raw:.3f}",
            transform=axes[row_index, 1].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        axis = axes[row_index, 2]
        axis.scatter(fitted_log, resid_log, s=point_size, alpha=point_alpha)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{window_label.upper()} log: residuals vs fitted")
        axis.set_xlabel("Fitted")
        axis.set_ylabel("Residual")

        qq_r_log = qq_plot(
            axes[row_index, 3],
            resid_log,
            f"{window_label.upper()} log: QQ",
            point_alpha,
            point_size,
        )
        axes[row_index, 3].text(
            0.02,
            0.98,
            f"r={qq_r_log:.3f}",
            transform=axes[row_index, 3].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        diag_rows.append(
            {
                "window": window_label,
                "auc_col": auc_col,
                "n_raw": int(df_raw.shape[0]),
                "n_nonpositive_raw": n_nonpositive,
                "log_method": "shifted",
                "log_shift_used": shift_used,
                "n_log": int(df_log.shape[0]),
                "raw_resid_sd": float(resid_raw.std(ddof=1)),
                "raw_resid_skew": float(resid_raw.skew()),
                "raw_resid_kurtosis": float(resid_raw.kurt()),
                "log_resid_sd": float(resid_log.std(ddof=1)),
                "log_resid_skew": float(resid_log.skew()),
                "log_resid_kurtosis": float(resid_log.kurt()),
            }
        )

    fig.suptitle("AUC MixedLM residual diagnostics (raw)", y=1.01)
    fig.tight_layout()
    save_fig(fig, fig_dir / "auc_mixedlm_residual_diagnostics_raw_shifted.png")
    plt.close(fig)

    csv_path = output_dir / "auc_mixedlm_residual_diagnostics_raw_shifted.csv"
    pd.DataFrame(diag_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def run_transform_sweep(df2_auc_clean: pd.DataFrame, output_dir: Path, fig_dir: Path) -> None:
    sweep_rows: list[dict[str, object]] = []
    point_alpha = 0.25
    point_size = 8

    for window_label in ("w2", "w3"):
        auc_col = auc_col_for(window_label, "raw")
        df_base = prepare_auc_model_df(df2_auc_clean, auc_col)
        min_auc = float(df_base[auc_col].min())
        n_nonpositive = int((df_base[auc_col] <= 0).sum())
        log_shift = (-min_auc + 1e-6) if min_auc <= 0 else 0.0

        transforms: list[dict[str, object]] = []

        df_raw = df_base.copy()
        y_raw = f"{auc_col}_raw_for_diag"
        df_raw[y_raw] = df_raw[auc_col]
        transforms.append({"name": "raw", "df": df_raw, "y_col": y_raw, "log_shift": 0.0, "yeojohnson_lambda": np.nan})

        df_log = df_base.copy()
        y_log = f"{auc_col}_log_shift_for_diag"
        df_log[y_log] = np.log(df_log[auc_col] + log_shift)
        transforms.append(
            {
                "name": "log_shift",
                "df": df_log,
                "y_col": y_log,
                "log_shift": float(log_shift),
                "yeojohnson_lambda": np.nan,
            }
        )

        df_yj = df_base.copy()
        y_yj = f"{auc_col}_yeojohnson_for_diag"
        yj_vals, yj_lambda = stats.yeojohnson(df_yj[auc_col].to_numpy())
        df_yj[y_yj] = yj_vals
        transforms.append(
            {
                "name": "yeojohnson",
                "df": df_yj,
                "y_col": y_yj,
                "log_shift": np.nan,
                "yeojohnson_lambda": float(yj_lambda),
            }
        )

        df_asinh = df_base.copy()
        y_asinh = f"{auc_col}_asinh_for_diag"
        df_asinh[y_asinh] = np.arcsinh(df_asinh[auc_col])
        transforms.append(
            {
                "name": "asinh",
                "df": df_asinh,
                "y_col": y_asinh,
                "log_shift": np.nan,
                "yeojohnson_lambda": np.nan,
            }
        )

        fig, axes = plt.subplots(2, len(transforms), figsize=(5 * len(transforms), 8))
        if len(transforms) == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for column_index, transform_info in enumerate(transforms):
            model_fit = fit_auc_mixedlm(transform_info["df"], str(transform_info["y_col"]))
            residuals = pd.Series(model_fit.resid).astype(float)
            fitted = pd.Series(model_fit.fittedvalues).astype(float)

            axis = axes[0, column_index]
            axis.scatter(fitted, residuals, s=point_size, alpha=point_alpha)
            axis.axhline(0, color="black", linestyle="--", linewidth=1)
            axis.set_title(f"{window_label.upper()} {transform_info['name']}: residuals vs fitted")
            axis.set_xlabel("Fitted")
            axis.set_ylabel("Residual")

            qq_r = qq_plot(
                axes[1, column_index],
                residuals,
                f"{window_label.upper()} {transform_info['name']}: QQ",
                point_alpha,
                point_size,
            )
            axes[1, column_index].set_title(
                f"{window_label.upper()} {transform_info['name']}: QQ (r={qq_r:.3f})"
            )

            sweep_rows.append(
                {
                    "window": window_label,
                    "auc_col": auc_col,
                    "transform": transform_info["name"],
                    "n": int(transform_info["df"].shape[0]),
                    "n_nonpositive_base": n_nonpositive,
                    "log_shift": transform_info["log_shift"],
                    "yeojohnson_lambda": transform_info["yeojohnson_lambda"],
                    "qq_r": qq_r,
                    "resid_sd": float(residuals.std(ddof=1)),
                    "resid_skew": float(residuals.skew()),
                    "resid_kurtosis": float(residuals.kurt()),
                    "aic": float(model_fit.aic),
                    "bic": float(model_fit.bic),
                }
            )

        fig.suptitle(f"AUC MixedLM transform sweep residual diagnostics (raw, {window_label.upper()})", y=1.02)
        fig.tight_layout()
        save_fig(fig, fig_dir / f"auc_mixedlm_transform_sweep_residuals_raw_{window_label}.png")
        plt.close(fig)

    csv_path = output_dir / "auc_mixedlm_transform_sweep_residuals_raw_w2_w3.csv"
    pd.DataFrame(sweep_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def write_analysis_summary(fig_dir: Path, output_dir: Path, stats_dir: Path) -> Path:
    summary_rows = [
        {
            "artifact_type": "table",
            "artifact_name": "mfi_kmeans_curve_metrics",
            "dataframe": "kmeans_curve_df",
            "output_file": str(output_dir / "clustering" / "mfi_kmeans_curve_metrics.csv"),
            "manipulations": "k=2..10 on bio-replicate normalized and standardized marker MFIs",
            "notes": "KMeans WCSS and silhouette values used to choose k=3",
        },
        {
            "artifact_type": "table",
            "artifact_name": "mfi_cluster_feature_means",
            "dataframe": "cluster_means",
            "output_file": str(output_dir / "clustering" / "mfi_cluster_feature_means.csv"),
            "manipulations": "mean bio-normalized marker MFIs by raw k-means cluster",
            "notes": "Cluster feature profiles before presentation-label remap",
        },
        {
            "artifact_type": "table",
            "artifact_name": "mfi_cluster_label_map",
            "dataframe": "cluster_label_map_df",
            "output_file": str(output_dir / "clustering" / "mfi_cluster_label_map.csv"),
            "manipulations": "dominant-marker rule PDGFRa->0, O4->1, MBP->2",
            "notes": "Mapping from raw k-means cluster IDs to presentation cluster labels",
        },
        {
            "artifact_type": "table",
            "artifact_name": "mfi_cluster_centers_scaled",
            "dataframe": "scaled_centers_df",
            "output_file": str(output_dir / "clustering" / "mfi_cluster_centers_scaled.csv"),
            "manipulations": "cluster centers in standardized bio-normalized MFI feature space",
            "notes": "Numerical k-means cluster centers from the fitted k=3 model",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "mfi_elbow_silhouette",
            "dataframe": "_tmp_mfi_norm",
            "output_file": str(fig_dir / "mfi_elbow_silhouette.png"),
            "manipulations": "bio_rep mean-normalization; standard scaling",
            "notes": "KMeans elbow and silhouette curves (MFI)",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "mfi_clusters_3d",
            "dataframe": "_tmp_present",
            "output_file": str(fig_dir / "mfi_clusters_3d.png"),
            "manipulations": "cluster remap for presentation; bio_rep mean-normalization",
            "notes": "3D MFI scatter with presentation cluster labels",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "auc_violin_per_treatment_windows",
            "dataframe": "df2_auc_clean",
            "output_file": str(fig_dir / "auc_violin_<treatment>_<window>.png"),
            "manipulations": "include nonpositive AUC; drop NaNs; IQR outlier removal by treatment x cluster",
            "notes": "Raw and normalized AUC violin plots by treatment and cluster",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "rt_scatter_per_treatment",
            "dataframe": "df2_present",
            "output_file": str(fig_dir / "rt_scatter_<treatment>.png"),
            "manipulations": "cluster remap for presentation; drop NaNs",
            "notes": "Response-time scatter plots retained as descriptive figures only",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "auc_rt_grid",
            "dataframe": "df2_auc_clean (AUC) + df2_present (RT)",
            "output_file": str(fig_dir / "auc_rt_grid_<window>.png"),
            "manipulations": "raw AUC cleaned as above; RT uses unfiltered response-time data",
            "notes": "Top row raw AUC violins, bottom row response-time scatter",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "mean_dff_traces",
            "dataframe": "df_ts_meta",
            "output_file": str(fig_dir / "traces_mean_ci_<treatment>.png"),
            "manipulations": "merge per-cell time series with cluster labels; mean +/- 95% CI",
            "notes": "Average dF/F0 traces by treatment and cluster",
        },
        {
            "artifact_type": "model",
            "artifact_name": "auc_mixedlm_nested_raw",
            "dataframe": "df_auc_model_cat",
            "output_file": str(output_dir / "auc_mixedlm_pairs_raw_all_windows.csv"),
            "manipulations": "drop NaNs; categorical ordering; raw AUC per window; random intercepts bio_rep + tech_rep_id (nested)",
            "notes": "Two-term MixedLM raw AUC ~ treatment * cluster + (1|bio_rep) + (1|tech_rep_id)",
        },
        {
            "artifact_type": "table",
            "artifact_name": "auc_two_term_lmm_term_effects",
            "dataframe": "two_term_term_df",
            "output_file": str(stats_dir / "two_term_lmm_term_effects.csv"),
            "manipulations": "fixed-effect coefficients exported from the nested raw-AUC MixedLM across all stimulation windows",
            "notes": "Includes main effects, interaction terms, z statistics, and FDR-adjusted p values",
        },
        {
            "artifact_type": "table",
            "artifact_name": "auc_two_term_lmm_within_cluster_contrasts",
            "dataframe": "two_term_within_df",
            "output_file": str(stats_dir / "two_term_lmm_within_cluster_contrasts.csv"),
            "manipulations": "treatment-vs-MDL29951 contrasts computed within each cluster and stimulation window",
            "notes": "Within-cluster treatment effects derived from the same nested raw-AUC MixedLM fits",
        },
        {
            "artifact_type": "table",
            "artifact_name": "auc_two_term_lmm_emms",
            "dataframe": "two_term_emm_df",
            "output_file": str(stats_dir / "two_term_lmm_emms.csv"),
            "manipulations": "estimated marginal means for every treatment x cluster combination in each stimulation window",
            "notes": "Model-based marginal means grid for the nested raw-AUC MixedLM",
        },
        {
            "artifact_type": "table",
            "artifact_name": "auc_two_term_lmm_model_fit_summary",
            "dataframe": "two_term_fit_summary_df",
            "output_file": str(stats_dir / "two_term_lmm_model_fit_summary.csv"),
            "manipulations": "window-level fit metadata and observation counts",
            "notes": "Reviewer-facing summary of rows, biological replicates, tech reps, optimizer, and warnings per stimulation window",
        },
        {
            "artifact_type": "table",
            "artifact_name": "auc_two_term_lmm_full_reports",
            "dataframe": "MixedLM summary tables",
            "output_file": str(stats_dir / "mixedlm_full_reports" / "two_term_lmm_calcium_auc_w1.csv"),
            "manipulations": "full statsmodels summary export per stimulation window",
            "notes": "CSV and TXT full-report dumps for each nested raw-AUC MixedLM fit",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "auc_mixedlm_residual_diagnostics",
            "dataframe": "df_auc_model_cat",
            "output_file": str(fig_dir / "auc_mixedlm_residual_diagnostics_raw_shifted.png"),
            "manipulations": "compare raw vs shifted-log residual behavior",
            "notes": "Supplementary residual diagnostics for the raw nested AUC MixedLM",
        },
        {
            "artifact_type": "plot",
            "artifact_name": "auc_mixedlm_transform_sweep",
            "dataframe": "df_auc_model_cat",
            "output_file": str(fig_dir / "auc_mixedlm_transform_sweep_residuals_raw_w2.png"),
            "manipulations": "compare raw, shifted log, Yeo-Johnson, and asinh residual behavior for W2/W3",
            "notes": "Supplementary transform sweep diagnostics",
        },
    ]

    summary_path = output_dir / "analysis_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    return summary_path


def run_analysis(master_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Path]:
    master_dir = resolve_data_dir(master_dir)
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir = output_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data directory: {master_dir}")
    print(f"Writing outputs to: {output_dir}")

    df_all = prepare_combined_matched_cells(master_dir, output_dir)
    df1_present = cluster_cells(df_all, output_dir, fig_dir, clustering_dir)
    df2_present, df2_auc_clean = build_auc_clean_table(df1_present)

    plot_rt_scatter_per_treatment(df2_present, fig_dir)
    plot_auc_violin_per_treatment(df2_auc_clean, fig_dir)
    plot_auc_rt_grid(df2_present, df2_auc_clean, fig_dir)
    model_paths = run_auc_models(df2_auc_clean, output_dir)
    plot_average_traces(master_dir, df1_present, fig_dir)
    run_residual_diagnostics(df2_auc_clean, output_dir, fig_dir)
    run_transform_sweep(df2_auc_clean, output_dir, fig_dir)
    summary_path = write_analysis_summary(fig_dir, output_dir, output_dir / "stats")

    return {
        "master_dir": master_dir,
        "output_dir": output_dir,
        "fig_dir": fig_dir,
        "clustering_dir": clustering_dir,
        "stats_dir": output_dir / "stats",
        "summary_path": summary_path,
        **model_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the calcium multiplex plotting and raw AUC nested MixedLM workflow.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the vendored out_registration_batch-style bundle CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated plots and result tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(master_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
