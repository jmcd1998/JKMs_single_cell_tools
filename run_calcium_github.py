from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, t, ttest_rel
import seaborn as sns

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

LEGACY_REPLICATE_COLORS = (
    "#3b4cc0",
    "#c51b7d",
    "#7e1e9c",
    "#1f78b4",
    "#f781bf",
    "#984ea3",
)


@dataclass(frozen=True)
class ClusteredAnalysisConfig:
    chapter_label: str
    dataset_label: str
    analysis_name: str
    input_mode: str
    input_candidates: tuple[Path, ...]
    analysis_dir: Path
    treatment_order: tuple[str, ...]
    cluster_feature_cols: tuple[str, ...]
    one_term_response_cols: tuple[str, ...]
    two_term_response_cols: tuple[str, ...]
    cluster_k: int
    row_tokens: tuple[str, ...] = ()
    csv_dir_candidates: tuple[Path, ...] = ()
    input_export_name: str = "per_cell_stats_input.csv"
    treatment_col: str = "treatment"
    group_col: str = "N"
    fov_col: str = "FOV_ID"
    cluster_col: str = "cluster"
    treatment_reference: str = "Vehicle"
    optimizer_sequence: tuple[str, ...] = ("lbfgs", "bfgs", "cg")
    min_unique_values: int = 5
    # Cluster composition defaults to biological-replicate weighted fractions
    # analysed with repeated-measures ANOVA. This is the active inference path
    # used by the Chapter 3 and Chapter 4 cluster-composition outputs.
    cluster_composition_inference: str = "rm_anova_weighted_fraction"
    cluster_composition_posthoc_scope: str = "all_clusters_vs_vehicle"
    cell_count_inference: str = "rm_anova"
    legacy_mfi_point_every: int | None = 10
    legacy_mfi_point_seed: int = 1

    @property
    def stats_dir(self) -> Path:
        return self.analysis_dir / "stats"

    @property
    def graphs_dir(self) -> Path:
        return self.analysis_dir / "graphs"

    @property
    def clustering_dir(self) -> Path:
        return self.analysis_dir / "clustering"

    @property
    def input_path(self) -> Path:
        for path in self.input_candidates:
            if path.exists():
                return path
        missing = self.input_candidates[0]
        raise FileNotFoundError(
            f"Expected parsed analysis input at {missing}. "
            "Run build_parsed_data_from_original_data.py first."
        )

    @property
    def csv_dir(self) -> Path | None:
        for path in self.csv_dir_candidates:
            if path.exists():
                return path
        return None


def ensure_output_dirs(config: ClusteredAnalysisConfig) -> None:
    for path in [config.analysis_dir, config.stats_dir, config.graphs_dir, config.clustering_dir]:
        path.mkdir(parents=True, exist_ok=True)


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return (
        "ns" if p >= 0.05 else
        "*" if p >= 0.01 else
        "**" if p >= 0.001 else
        "***"
    )


def build_rep_palette(rep_order: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    if not rep_order:
        return {}
    repeats = (len(rep_order) // len(LEGACY_REPLICATE_COLORS)) + 1
    colors = (list(LEGACY_REPLICATE_COLORS) * repeats)[:len(rep_order)]
    return {label: color for label, color in zip(rep_order, colors)}


def build_simplified_replicate_map(rep_order: Sequence[str]) -> dict[str, str]:
    return {label: f"N{idx}" for idx, label in enumerate(rep_order, start=1)}


def write_standalone_replicate_legend(
    rep_order: Sequence[str],
    *,
    rep_palette: dict[str, tuple[float, float, float, float]],
    out_path: Path,
) -> Path | None:
    if not rep_order:
        return None

    display_map = build_simplified_replicate_map(rep_order)
    fig_height = max(1.4, 0.45 * len(rep_order) + 0.35)
    fig, ax = plt.subplots(figsize=(2.2, fig_height))
    y_positions = np.arange(len(rep_order))[::-1]
    for ypos, label in zip(y_positions, rep_order):
        ax.scatter(
            [0.0],
            [ypos],
            s=85,
            color=rep_palette[label],
            edgecolors="white",
            linewidths=0.6,
        )
        ax.text(0.22, ypos, display_map[label], va="center", ha="left", fontsize=11)
    ax.set_xlim(-0.12, 0.95)
    ax.set_ylim(-0.7, len(rep_order) - 0.3)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


def slugify_filename_part(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not slug:
        raise ValueError(f"Unable to create filename slug from value: {value!r}")
    return slug


LEGACY_CHAPTER3_PREFIX_PAT = re.compile(r"(?i)^(vehicle|mdl|hami|pranlukast)_([0-9]+)")
LEGACY_CHAPTER3_ALIASES = {
    "vehicle": "Vehicle",
    "mdl": "MDL29951",
    "mdl29951": "MDL29951",
    "hami": "HAMI3379",
    "hami3379": "HAMI3379",
    "pranlukast": "Pranlukast",
}
GENERIC_TREATMENT_LOOKUP = {
    **LEGACY_CHAPTER3_ALIASES,
    "rwt9996": "RWT9996",
    "clemastine": "Clemastine",
}
RX_0929_PREFIX = re.compile(r"^(?!N4_0929)0929(?=[_ ]|$)", flags=re.IGNORECASE)


def ensure_fov_id(df: pd.DataFrame, config: ClusteredAnalysisConfig) -> pd.DataFrame:
    df = df.copy()
    if config.fov_col in df.columns:
        return df

    fov_source = None
    for candidate in ["FOV", "series_name", "series name", "Series Name", "Series_Name", "scene_idx"]:
        if candidate in df.columns:
            fov_source = candidate
            break
    if fov_source is None:
        return df

    if "FOV" not in df.columns:
        df["FOV"] = df[fov_source]

    if config.group_col in df.columns:
        base = df[config.group_col].astype(str)
    elif "file" in df.columns:
        base = df["file"].astype(str)
    else:
        base = pd.Series("group", index=df.index, dtype=object)

    df[config.fov_col] = base + "__" + df["FOV"].astype(str).fillna("FOV?")
    return df


def legacy_chapter3_parse_filename(value: object) -> tuple[str, str]:
    text = str(value).strip()
    stem = Path(text).stem
    match = LEGACY_CHAPTER3_PREFIX_PAT.match(stem)
    if not match:
        return "Unknown", "UNK"
    treatment_key = match.group(1).lower()
    rep_id = str(int(match.group(2))) if match.group(2).isdigit() else match.group(2)
    return LEGACY_CHAPTER3_ALIASES.get(treatment_key, treatment_key.capitalize()), rep_id


def harmonise_convexity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_conv = "convexity_ratio" in out.columns
    has_area = "area_ratio_union" in out.columns
    if has_conv and not has_area:
        out["area_ratio_union"] = out["convexity_ratio"]
    elif has_area and not has_conv:
        out["convexity_ratio"] = out["area_ratio_union"]
    elif has_area and has_conv:
        out["convexity_ratio"] = pd.to_numeric(out["convexity_ratio"], errors="coerce").fillna(
            pd.to_numeric(out["area_ratio_union"], errors="coerce")
        )
    return out


def invert_convexity_measurements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    convexity_cols = [
        column
        for column in [
            "convexity_ratio",
            "area_ratio_union",
            "scene_area_ratio_union",
            "morph_mbppdg_convexity_ratio",
            "morph_all_convexity_ratio",
        ]
        if column in out.columns
    ]
    for column in convexity_cols:
        values = pd.to_numeric(out[column], errors="coerce")
        out[column] = np.where(
            np.isfinite(values) & (values != 0),
            1.0 / values,
            np.nan,
        )
    return out


def combine_csv_dir(csv_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(path for path in csv_dir.glob("*.csv") if "partial" not in path.name.lower())
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")
    df = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True, axis=0)
    if "file" not in df.columns:
        if "base_name" not in df.columns:
            raise RuntimeError("Expected either 'file' or 'base_name' while combining CSVs.")
        df["file"] = df["base_name"].astype(str)
    if "cell_id" not in df.columns:
        df["cell_id"] = np.arange(len(df))
    return df


def _fix_0929_prefix(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    return RX_0929_PREFIX.sub("N4_0929", str(value).strip())


def fix_0929_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("lif_name", "base_name", "file"):
        if col in out.columns:
            out[col] = out[col].map(_fix_0929_prefix)
    if "lif_path" in out.columns:
        out["lif_path"] = out["lif_path"].astype(str).str.replace(
            r"(?i)(?<!N4_)0929(?=_(?:PKM2|pPKM2)\.lif\b)",
            "N4_0929",
            regex=True,
        )
    return out


def load_chapter3_combined_input(config: ClusteredAnalysisConfig) -> pd.DataFrame:
    csv_dir = config.csv_dir
    if csv_dir is not None:
        df = combine_csv_dir(csv_dir)
    else:
        df = pd.read_csv(config.input_path)
        if "file" not in df.columns:
            if "base_name" not in df.columns:
                raise RuntimeError("Expected either 'file' or 'base_name' in the Chapter 3 combined input.")
            df["file"] = df["base_name"].astype(str)
        if "cell_id" not in df.columns:
            df["cell_id"] = np.arange(len(df))

    out = df.copy()
    parsed = out["file"].apply(legacy_chapter3_parse_filename)
    out[["treatment", "N"]] = pd.DataFrame(parsed.tolist(), index=out.index)
    out["N"] = out["N"].astype(str)

    raw_ids = [value for value in sorted(out["N"].dropna().unique().tolist(), key=sort_key) if value != "UNK"]
    label_map = {rid: f"N{i + 1}" for i, rid in enumerate(raw_ids)}
    out["N_label"] = out["N"].map(label_map).fillna("N?")

    out = harmonise_convexity_columns(out)

    if "FOV" not in out.columns:
        fov_source = None
        for candidate in ["series_name", "series name", "Series Name", "Series_Name", "scene_idx"]:
            if candidate in out.columns:
                fov_source = candidate
                break
        if fov_source is not None:
            out["FOV"] = out[fov_source].astype(str)
    out = ensure_fov_id(out, config)
    return out


def parse_chapter4_basename(value: object, row_tokens: Sequence[str]) -> tuple[str, str, str, str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N?", "S?", "?", "Unknown"

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "N?", "S?", "?", "Unknown"

    parts = [part for part in re.split(r"[_\-]+", text) if part]
    n = "N?"
    fov = "S?"
    row = "?"
    treatment = "Unknown"

    for part in parts:
        n_match = re.fullmatch(r"(?i)N(\d+)", part)
        if n_match:
            n = f"N{n_match.group(1)}"
            continue

        fov_match = re.fullmatch(r"(?i)S(\d+)", part)
        if fov_match:
            fov = f"S{fov_match.group(1)}"
            continue

        if part in row_tokens:
            row = part
            continue

        g_row_match = re.fullmatch(r"(?i)G(\d+)", part)
        if g_row_match and g_row_match.group(1) in row_tokens:
            row = g_row_match.group(1)
            continue

        treatment_key = part.lower()
        if treatment_key in GENERIC_TREATMENT_LOOKUP:
            treatment = GENERIC_TREATMENT_LOOKUP[treatment_key]

    return n, fov, row, treatment


def standardize_chapter4_fov(df: pd.DataFrame, config: ClusteredAnalysisConfig) -> pd.DataFrame:
    out = df.copy()
    out["FOV"] = out["FOV"].fillna("S?").astype(str).str.strip()
    out["Well"] = out["Well"].fillna("?").astype(str).str.strip()
    out["Well_ID"] = (
        out[config.treatment_col].astype(str)
        + "__"
        + out[config.group_col].astype(str)
        + "__"
        + out["Well"].astype(str)
    )
    out[config.fov_col] = out["Well_ID"] + "__" + out["FOV"].astype(str)
    return out


def load_chapter4_csv_input(config: ClusteredAnalysisConfig) -> pd.DataFrame:
    csv_dir = config.csv_dir
    if csv_dir is None:
        raise FileNotFoundError("Chapter 4 raw-input mode requires a CSV directory.")

    out = combine_csv_dir(csv_dir)
    return prepare_chapter4_input_frame(out, config)


def prepare_chapter4_input_frame(
    df: pd.DataFrame,
    config: ClusteredAnalysisConfig,
) -> pd.DataFrame:
    out = df.copy()
    out = fix_0929_names(out)
    if "base_name" not in out.columns:
        if "file" not in out.columns:
            raise RuntimeError("Expected 'base_name' or 'file' column for Chapter 4 input parsing.")
        out["base_name"] = out["file"].astype(str)
    if "file" not in out.columns:
        out["file"] = out["base_name"].astype(str)
    if "cell_id" not in out.columns:
        out["cell_id"] = np.arange(len(out))

    parsed = out["base_name"].apply(lambda value: parse_chapter4_basename(value, config.row_tokens))
    out[[config.group_col, "FOV", "row", config.treatment_col]] = pd.DataFrame(parsed.tolist(), index=out.index)
    out["biol rep"] = out[config.group_col]
    out["Well"] = out["row"]
    out["well tag"] = (
        out[config.treatment_col].astype(str)
        + "_"
        + out[config.group_col].astype(str)
        + "_"
        + out["row"].astype(str)
    )
    out = standardize_chapter4_fov(out, config)

    raw_ids = sorted(out[config.group_col].dropna().unique().tolist(), key=sort_key)
    clean_ids = [rid for rid in raw_ids if isinstance(rid, str) and rid != "N?"]
    has_real_n = len(clean_ids) > 0 and all(re.match(r"^N\d+$", rid) for rid in clean_ids)
    if has_real_n:
        out["N_label"] = out[config.group_col]
    else:
        label_map = {rid: f"N{i + 1}" for i, rid in enumerate(clean_ids)}
        out["N_label"] = out[config.group_col].map(label_map).fillna("N?")
    return out


def load_chapter4_combined_input(config: ClusteredAnalysisConfig) -> pd.DataFrame:
    return prepare_chapter4_input_frame(pd.read_csv(config.input_path), config)


def load_analysis_table(config: ClusteredAnalysisConfig) -> pd.DataFrame:
    if config.input_mode == "chapter3_combined":
        df = load_chapter3_combined_input(config)
    elif config.input_mode == "chapter4_csv_dir":
        df = load_chapter4_csv_input(config)
    elif config.input_mode == "chapter4_combined":
        df = load_chapter4_combined_input(config)
    elif config.input_mode == "clustered":
        path = config.input_path
        df = pd.read_csv(path)
        df = ensure_fov_id(df, config)
    else:
        raise ValueError(f"Unsupported input_mode: {config.input_mode}")

    missing = [
        column
        for column in [config.treatment_col, config.group_col, config.fov_col]
        if column not in df.columns
    ]
    if missing:
        raise RuntimeError(f"Missing required columns in input table: {missing}")
    df = invert_convexity_measurements(df)
    return df


def load_clustered_analysis_table(config: ClusteredAnalysisConfig) -> pd.DataFrame:
    path = config.input_path
    df = pd.read_csv(path)
    df = ensure_fov_id(df, config)

    missing = [
        column
        for column in [config.treatment_col, config.group_col, config.fov_col, config.cluster_col]
        if column not in df.columns
    ]
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")

    df = invert_convexity_measurements(df)
    return df


def input_source_label(config: ClusteredAnalysisConfig) -> str:
    if config.csv_dir is not None and config.input_mode in {"chapter3_combined", "chapter4_csv_dir"}:
        return str(config.csv_dir)
    return str(config.input_path)


def resolve_response_cols(df: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    return [column for column in requested if column in df.columns]


def inspect_dataset(config: ClusteredAnalysisConfig) -> pd.DataFrame:
    df = load_analysis_table(config)
    one_term_present = resolve_response_cols(df, config.one_term_response_cols)
    two_term_present = resolve_response_cols(df, config.two_term_response_cols)

    rows = [
        {"metric": "chapter", "value": config.chapter_label},
        {"metric": "dataset", "value": config.dataset_label},
        {"metric": "data_csv", "value": input_source_label(config)},
        {"metric": "analysis_dir", "value": str(config.analysis_dir)},
        {"metric": "cluster_composition_inference", "value": config.cluster_composition_inference},
        {"metric": "cluster_composition_posthoc_scope", "value": config.cluster_composition_posthoc_scope},
        {"metric": "cell_count_inference", "value": config.cell_count_inference},
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_treatments", "value": int(df[config.treatment_col].dropna().astype(str).nunique())},
        {"metric": "n_biological_replicates", "value": int(df[config.group_col].dropna().astype(str).nunique())},
        {"metric": "n_fovs", "value": int(df[config.fov_col].dropna().astype(str).nunique())},
        {"metric": "n_clusters", "value": int(df[config.cluster_col].dropna().astype(str).nunique()) if config.cluster_col in df.columns else 0},
        {"metric": "one_term_responses_found", "value": ", ".join(one_term_present)},
        {"metric": "two_term_responses_found", "value": ", ".join(two_term_present)},
        {
            "metric": "treatments_seen",
            "value": ", ".join(
                sorted(df[config.treatment_col].dropna().astype(str).unique().tolist())
            ),
        },
    ]
    return pd.DataFrame(rows)


def strip_measurement_label(name: str) -> str:
    text = name.replace("_int_ratio", "").replace("ppkm2", "pPKM2").replace("_", " ")
    return text


def response_axis_label(response_col: str, config: ClusteredAnalysisConfig) -> str:
    if config.chapter_label == "Chapter 3":
        label_map = chapter_3_heatmap_measurement_label_map()
    elif config.chapter_label == "Chapter 4":
        label_map = chapter_4_heatmap_measurement_label_map(config)
    else:
        label_map = {}
    if response_col in label_map:
        return label_map[response_col]
    if response_col == "AUC" or response_col.endswith("_AUC"):
        return "AUC"
    return strip_measurement_label(response_col)


def chapter_3_heatmap_measurement_label_map() -> dict[str, str]:
    return {
        "cell_area_px": "Cell area",
        "AUC": "AUC",
        "area_ratio_union": "Convexity",
        "Rmax_px": "Rmax",
        "Imax": "Imax",
        "CriticalValue": "Critical value",
    }


def chapter_4_heatmap_measurement_label_map(config: ClusteredAnalysisConfig) -> dict[str, str]:
    return {
        "cell_area_px": "Cell area",
        "morph_all_AUC": "AUC",
        "morph_all_convexity_ratio": "Convexity",
        "morph_all_Rmax_px": "Rmax",
        "morph_all_Imax": "Imax",
        "morph_all_CriticalValue": "Critical value",
    }


def expected_heatmap_cluster_labels(config: ClusteredAnalysisConfig) -> dict[int, str]:
    if config.chapter_label == "Chapter 3":
        return {
            0: "PDGFRa",
            1: "CNP",
            2: "MBP",
        }
    if config.chapter_label == "Chapter 4":
        if config.cluster_k == 3:
            return {
                0: "PDGFRa",
                1: "O4",
                2: "MBP/O4",
            }
        if config.cluster_k == 4:
            return {
                0: "PDGFRa",
                1: "O4",
                2: "MBP/O4",
                3: "Marker Low",
            }
    return {}


def expected_heatmap_cluster_colors(config: ClusteredAnalysisConfig) -> dict[int, str]:
    if config.chapter_label == "Chapter 3":
        return {
            0: "#2E7D32",
            1: "#EF6C00",
            2: "#C2185B",
        }
    if config.chapter_label == "Chapter 4":
        return {
            0: "#2E7D32",
            1: "#F6A623",
            2: "#C2185B",
            3: "#7F8C8D",
        }
    return {}


def expected_heatmap_cluster_order(config: ClusteredAnalysisConfig) -> list[int]:
    if config.chapter_label == "Chapter 3":
        return [0, 1, 2]
    if config.chapter_label == "Chapter 4":
        return [0, 1, 2, 3]
    return []


def expected_heatmap_measurement_order(
    config: ClusteredAnalysisConfig,
    response_cols: Sequence[str],
) -> list[str]:
    if config.chapter_label == "Chapter 3":
        preferred = list(chapter_3_heatmap_measurement_label_map().keys())
    elif config.chapter_label == "Chapter 4":
        preferred = list(chapter_4_heatmap_measurement_label_map(config).keys())
    else:
        preferred = list(response_cols)
    return [response for response in preferred if response in response_cols]


def norm_long_id_columns(df: pd.DataFrame, config: ClusteredAnalysisConfig) -> list[str]:
    id_candidates = [
        "file",
        "base_name",
        "lif_name",
        "lif_path",
        "series_name",
        "scene_idx",
        config.treatment_col,
        config.group_col,
        "N_label",
        "biol rep",
        "row",
        "cell_id",
        "Well",
        "well tag",
        "Well_ID",
        "FOV",
        config.fov_col,
        config.cluster_col,
        "cluster_raw",
        "cluster_name",
    ]
    return [column for column in id_candidates if column in df.columns]


def select_numeric_measurements(
    df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
) -> list[str]:
    excluded = set(id_cols)
    numeric_cols = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]
    numeric_cols = [
        column
        for column in numeric_cols
        if column not in {config_col for config_col in ["cluster_raw"]}
        and not column.endswith("_norm")
    ]
    return numeric_cols


def remove_outliers_raw(group: pd.DataFrame) -> pd.DataFrame:
    if group["value"].notna().sum() < 4:
        return group
    q1 = group["value"].quantile(0.1)
    q3 = group["value"].quantile(0.9)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return group[(group["value"] >= lo) & (group["value"] <= hi)]


def cluster_label_for_value(value: object, config: ClusteredAnalysisConfig) -> str:
    label_map = expected_heatmap_cluster_labels(config)
    level_int = _cluster_level_as_int(value)
    if level_int is not None and level_int in label_map:
        return label_map[level_int]
    return str(value)


def cluster_composition_posthoc_target_labels(config: ClusteredAnalysisConfig) -> set[str] | None:
    if config.cluster_composition_posthoc_scope != "planned_mbp_high_vs_vehicle":
        return None
    if config.chapter_label == "Chapter 3":
        return {"MBP"}
    if config.chapter_label == "Chapter 4":
        return {"MBP/O4"}
    return set()


def build_norm_long(df: pd.DataFrame, config: ClusteredAnalysisConfig) -> pd.DataFrame:
    id_cols = norm_long_id_columns(df, config)
    numeric_cols = select_numeric_measurements(df, id_cols=id_cols)
    if not numeric_cols:
        raise RuntimeError("No numeric measurement columns were found in the input table.")

    long_df = (
        df[id_cols + numeric_cols]
        .copy()
        .melt(
            id_vars=id_cols,
            value_vars=numeric_cols,
            var_name="measurement",
            value_name="value",
        )
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    long_df = (
        long_df.groupby([config.group_col, "measurement"], group_keys=False)
        .apply(remove_outliers_raw)
        .reset_index(drop=True)
    )
    if long_df.empty:
        raise RuntimeError("All rows were removed by the legacy outlier filter.")

    veh_rep = (
        long_df[long_df[config.treatment_col].astype(str) == config.treatment_reference]
        .groupby([config.group_col, "measurement"], observed=True)["value"]
        .mean()
        .rename("vehicle_mean_group")
    )
    veh_global = (
        long_df[long_df[config.treatment_col].astype(str) == config.treatment_reference]
        .groupby("measurement", observed=True)["value"]
        .mean()
        .rename("vehicle_mean_global")
    )

    norm_long = (
        long_df
        .merge(veh_rep, on=[config.group_col, "measurement"], how="left")
        .merge(veh_global, on="measurement", how="left")
    )
    ref = np.where(
        norm_long["vehicle_mean_group"].notna() & (norm_long["vehicle_mean_group"] != 0),
        norm_long["vehicle_mean_group"],
        np.where(
            norm_long["vehicle_mean_global"].notna() & (norm_long["vehicle_mean_global"] != 0),
            norm_long["vehicle_mean_global"],
            np.nan,
        ),
    )
    norm_long["value_norm"] = np.where(np.isfinite(ref), norm_long["value"] / ref, norm_long["value"])
    norm_long = norm_long.drop(columns=["vehicle_mean_group", "vehicle_mean_global"], errors="ignore")
    return norm_long


def write_pipeline_tables(
    df: pd.DataFrame,
    norm_long: pd.DataFrame,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_copy_path = config.analysis_dir / config.input_export_name
    write_csv_with_lock_fallback(df, input_copy_path)

    norm_long_path = config.analysis_dir / "per_cell_stats_norm_long.csv"
    write_csv_with_lock_fallback(norm_long, norm_long_path)

    wide_index_candidates = [
        "cell_id",
        "file",
        config.treatment_col,
        config.group_col,
        "N_label",
        "Well_ID",
        "FOV",
        config.fov_col,
        config.cluster_col,
        "cluster_name",
    ]
    wide_index = [column for column in wide_index_candidates if column in norm_long.columns]

    norm_long_for_wide = norm_long.assign(feature_norm=norm_long["measurement"].astype(str) + "_norm")
    norm_wide = (
        norm_long_for_wide.pivot_table(
            index=wide_index,
            columns="feature_norm",
            values="value_norm",
            aggfunc="first",
            observed=True,
        )
        .reset_index()
    )
    if isinstance(norm_wide.columns, pd.MultiIndex):
        norm_wide.columns = [column[0] if isinstance(column, tuple) else column for column in norm_wide.columns]
    norm_wide.columns.name = None

    cluster_info_cols = [column for column in ["cluster_raw", config.cluster_col, "cluster_name"] if column in df.columns]
    if cluster_info_cols and config.cluster_col not in norm_wide.columns:
        cluster_merge_keys = [
            column
            for column in ["file", "cell_id", config.fov_col]
            if column in df.columns and column in norm_wide.columns
        ]
        if cluster_merge_keys:
            cluster_lookup = df[cluster_merge_keys + cluster_info_cols].drop_duplicates()
            norm_wide = norm_wide.merge(cluster_lookup, on=cluster_merge_keys, how="left", validate="many_to_one")

    merge_keys = [column for column in ["file", "cell_id", config.fov_col] if column in df.columns and column in norm_wide.columns]
    norm_cols = [f"{column}_norm" for column in config.cluster_feature_cols if f"{column}_norm" in norm_wide.columns]
    if merge_keys and norm_cols:
        analysis_df = df.drop(columns=norm_cols, errors="ignore").merge(
            norm_wide[merge_keys + norm_cols],
            on=merge_keys,
            how="left",
            validate="many_to_one",
        )
    else:
        analysis_df = df.copy()

    if "cluster_name" not in analysis_df.columns and config.cluster_col in analysis_df.columns:
        analysis_df["cluster_name"] = analysis_df[config.cluster_col].map(
            lambda value: cluster_label_for_value(value, config) if pd.notna(value) else np.nan
        )

    if merge_keys and norm_cols and config.cluster_col in analysis_df.columns:
        cluster_lookup_cols = [column for column in merge_keys + ["cluster_raw", config.cluster_col, "cluster_name"] if column in analysis_df.columns]
        summary_clustered_df = (
            norm_wide[
                [
                    column
                    for column in [
                        "cell_id",
                        "file",
                        config.treatment_col,
                        config.group_col,
                        "N_label",
                        "Well_ID",
                        "FOV",
                        config.fov_col,
                        *norm_cols,
                    ]
                    if column in norm_wide.columns
                ]
            ]
            .merge(
                analysis_df[cluster_lookup_cols].drop_duplicates(),
                on=merge_keys,
                how="left",
                validate="many_to_one",
            )
        )
        summary_clustered_df = summary_clustered_df.dropna(subset=[config.cluster_col]).copy()
        rename_norm_cols = {
            f"{column}_norm": column
            for column in config.cluster_feature_cols
            if f"{column}_norm" in summary_clustered_df.columns
        }
        summary_clustered_df = summary_clustered_df.rename(columns=rename_norm_cols)
        summary_clustered_df = summary_clustered_df[
            [
                column
                for column in [
                    "cell_id",
                    "file",
                    config.treatment_col,
                    config.group_col,
                    "N_label",
                    "Well_ID",
                    "FOV",
                    config.fov_col,
                    *config.cluster_feature_cols,
                    "cluster_raw",
                    config.cluster_col,
                    "cluster_name",
                ]
                if column in summary_clustered_df.columns
            ]
        ].copy()
    else:
        clustered_cols = [
            column
            for column in [
                "cell_id",
                "file",
                config.treatment_col,
                config.group_col,
                "N_label",
                "Well_ID",
                "FOV",
                config.fov_col,
                *config.cluster_feature_cols,
                *norm_cols,
                "cluster_raw",
                config.cluster_col,
                "cluster_name",
            ]
            if column in analysis_df.columns
        ]
        summary_clustered_df = analysis_df[clustered_cols].copy()

    write_csv_with_lock_fallback(summary_clustered_df, config.clustering_dir / "summary_table_clustered.csv")
    write_csv_with_lock_fallback(analysis_df, config.clustering_dir / "summary_table_with_clusters.csv")
    write_csv_with_lock_fallback(analysis_df, config.clustering_dir / "analysis_cells_used_columns_with_clusters.csv")

    profile_cols = [column for column in config.cluster_feature_cols if column in summary_clustered_df.columns]
    if profile_cols and config.cluster_col in summary_clustered_df.columns:
        profile_df = summary_clustered_df.groupby(config.cluster_col, observed=True)[profile_cols].mean().reset_index()
        if "cluster_name" in summary_clustered_df.columns:
            labels = (
                summary_clustered_df[[config.cluster_col, "cluster_name"]]
                .drop_duplicates()
                .sort_values(config.cluster_col)
            )
            profile_df = profile_df.merge(labels, on=config.cluster_col, how="left")
            profile_df = profile_df.rename(columns={"cluster_name": "cluster_label"})
        write_csv_with_lock_fallback(profile_df, config.clustering_dir / "cluster_feature_profiles.csv")

    return analysis_df, norm_wide


def _cluster_level_as_int(level: object) -> int | None:
    try:
        numeric = float(level)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    integer = int(numeric)
    if float(integer) != numeric:
        return None
    return integer


def make_cluster_palette(
    cluster_levels: Sequence[object],
    config: ClusteredAnalysisConfig | None = None,
) -> dict[str, str | tuple[float, float, float, float]]:
    if config is not None:
        configured_colors = expected_heatmap_cluster_colors(config)
        if configured_colors:
            palette: dict[str, str | tuple[float, float, float, float]] = {}
            for level in cluster_levels:
                key = str(level)
                level_int = _cluster_level_as_int(level)
                if level_int is not None and level_int in configured_colors:
                    palette[key] = configured_colors[level_int]
            if len(palette) == len(cluster_levels):
                return palette

    cmap = plt.cm.get_cmap("tab10", max(len(cluster_levels), 3))
    return {str(level): cmap(idx % cmap.N) for idx, level in enumerate(cluster_levels)}


def prepare_cluster_plot_table(
    norm_wide: pd.DataFrame,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, list[str]]:
    norm_cols = [f"{column}_norm" for column in config.cluster_feature_cols if f"{column}_norm" in norm_wide.columns]
    if config.cluster_col not in norm_wide.columns:
        raise RuntimeError(f"Missing '{config.cluster_col}' in normalized wide table for plotting.")
    required = [config.treatment_col, config.cluster_col, *norm_cols]
    cluster_df = norm_wide.dropna(subset=[column for column in required if column in norm_wide.columns]).copy()
    if cluster_df.empty:
        raise RuntimeError("Unable to prepare clustering plots because no rows had all normalized feature columns.")
    cluster_df[config.treatment_col] = cluster_df[config.treatment_col].astype(str)
    cluster_df[config.cluster_col] = cluster_df[config.cluster_col].astype(str)
    return cluster_df, norm_cols


def plot_cluster_selection_diagnostics(cluster_df: pd.DataFrame, norm_cols: Sequence[str], config: ClusteredAnalysisConfig) -> None:
    vehicle_df = cluster_df[cluster_df[config.treatment_col] == config.treatment_reference].copy()
    if vehicle_df.shape[0] <= max(2, config.cluster_k):
        return

    scaler = StandardScaler()
    x_train = scaler.fit_transform(vehicle_df[list(norm_cols)])

    max_k = min(9, max(2, len(x_train) - 1))
    usable_k = [k for k in range(2, max_k + 1) if k < len(x_train)]
    if not usable_k:
        return

    inertia = []
    silhouette_rows = []
    for k in usable_k:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(x_train)
        inertia.append({"k": k, "inertia": float(km.inertia_)})
        if len(np.unique(labels)) > 1 and len(x_train) > k:
            silhouette_rows.append({"k": k, "silhouette": float(silhouette_score(x_train, labels))})

    inertia_df = pd.DataFrame(inertia)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(inertia_df["k"], inertia_df["inertia"], marker="o")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Within-cluster SSE")
    ax.set_title("Elbow Method (Vehicle only)")
    fig.tight_layout()
    fig.savefig(config.clustering_dir / "elbow_vehicle_only.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if silhouette_rows:
        silhouette_df = pd.DataFrame(silhouette_rows)
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.plot(silhouette_df["k"], silhouette_df["silhouette"], marker="o")
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette vs K (Vehicle only)")
        fig.tight_layout()
        fig.savefig(config.clustering_dir / "silhouette_vehicle_only.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_cluster_scatter_3d(cluster_df: pd.DataFrame, norm_cols: Sequence[str], config: ClusteredAnalysisConfig) -> None:
    if len(norm_cols) < 3:
        return

    vehicle_df = cluster_df[cluster_df[config.treatment_col].astype(str) == config.treatment_reference].copy()
    scaler = StandardScaler()
    if vehicle_df.empty:
        scaled = scaler.fit_transform(cluster_df[list(norm_cols)])
    else:
        scaler.fit(vehicle_df[list(norm_cols)])
        scaled = scaler.transform(cluster_df[list(norm_cols)])
    present_levels = cluster_df[config.cluster_col].astype(str).unique().tolist()
    preferred_levels = [
        str(level)
        for level in expected_heatmap_cluster_order(config)
        if str(level) in present_levels
    ]
    levels = preferred_levels or sorted(present_levels, key=sort_key)
    palette = make_cluster_palette(levels, config)

    fig = plt.figure(figsize=(8.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    label_map = expected_heatmap_cluster_labels(config)
    for level in levels:
        mask = cluster_df[config.cluster_col].astype(str) == str(level)
        legend_label = cluster_label_for_value(level, config) if label_map else str(level)
        ax.scatter(
            scaled[mask.to_numpy(), 0],
            scaled[mask.to_numpy(), 1],
            scaled[mask.to_numpy(), 2],
            s=25,
            alpha=0.7,
            label=legend_label,
            color=palette[str(level)],
            edgecolor="0.1",
            linewidth=0.4,
        )
    labels = [strip_measurement_label(column) for column in config.cluster_feature_cols[:3]]
    ax.set_xlabel(f"{labels[0]} (scaled)")
    ax.set_ylabel(f"{labels[1]} (scaled)")
    ax.set_zlabel(f"{labels[2]} (scaled)")
    ax.set_title(f"Cell clusters (KMeans, k={config.cluster_k}, trained on {config.treatment_reference})")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(config.graphs_dir / "clusters_3d_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_clustering_from_norm_long(
    norm_long: pd.DataFrame,
    df: pd.DataFrame,
    config: ClusteredAnalysisConfig,
) -> pd.DataFrame:
    feature_cols = [column for column in config.cluster_feature_cols if column in norm_long["measurement"].astype(str).unique().tolist()]
    if len(feature_cols) != len(config.cluster_feature_cols):
        missing = [column for column in config.cluster_feature_cols if column not in feature_cols]
        raise RuntimeError(f"Missing clustering features in normalized long table: {missing}")

    index_candidates = [
        "cell_id",
        "file",
        config.treatment_col,
        config.group_col,
        "N_label",
        "Well_ID",
        "FOV",
        config.fov_col,
    ]
    index_cols = [column for column in index_candidates if column in norm_long.columns]
    wide = (
        norm_long.loc[norm_long["measurement"].isin(feature_cols), index_cols + ["measurement", "value_norm"]]
        .pivot_table(
            index=index_cols,
            columns="measurement",
            values="value_norm",
            aggfunc="first",
            observed=True,
        )
        .reset_index()
    )
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [column[0] if isinstance(column, tuple) else column for column in wide.columns]
    wide.columns.name = None

    x_df_all = wide.dropna(subset=feature_cols).copy()
    x_train = x_df_all[x_df_all[config.treatment_col].astype(str) == config.treatment_reference].copy()
    if x_train.empty:
        raise RuntimeError("No Vehicle rows found to train clustering.")
    if len(x_train) <= config.cluster_k:
        raise RuntimeError(
            f"Need more than {config.cluster_k} Vehicle observations to fit KMeans; found {len(x_train)}."
        )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train[feature_cols])
    x_all_scaled = scaler.transform(x_df_all[feature_cols])

    kmeans = KMeans(n_clusters=config.cluster_k, random_state=42, n_init="auto")
    kmeans.fit(x_train_scaled)
    labels_all = np.asarray(kmeans.predict(x_all_scaled), dtype=int)
    x_df_all["cluster_raw"] = labels_all

    cluster_means = x_df_all.groupby("cluster_raw", observed=True)[feature_cols].mean()
    third_cluster = int(cluster_means[feature_cols[2]].idxmax())
    remaining = [idx for idx in cluster_means.index.tolist() if idx != third_cluster]
    second_scores = cluster_means.loc[remaining, feature_cols[1]] - cluster_means.loc[remaining, feature_cols[0]]
    second_cluster = int(second_scores.idxmax())

    if config.cluster_k == 3:
        last_cluster = int([idx for idx in cluster_means.index.tolist() if idx not in {third_cluster, second_cluster}][0])
        remap = {third_cluster: 0, second_cluster: 1, last_cluster: 2}
    elif config.cluster_k == 4:
        remaining = [idx for idx in remaining if idx != second_cluster]
        combo_scores = cluster_means.loc[remaining, feature_cols[0]] + cluster_means.loc[remaining, feature_cols[1]]
        combo_cluster = int(combo_scores.idxmax())
        marker_low_cluster = int([idx for idx in remaining if idx != combo_cluster][0])
        remap = {third_cluster: 0, second_cluster: 1, combo_cluster: 2, marker_low_cluster: 3}
    else:
        raise RuntimeError(f"Unsupported cluster_k={config.cluster_k}")

    x_df_all[config.cluster_col] = x_df_all["cluster_raw"].map(remap).astype(int)
    x_df_all["cluster_name"] = x_df_all[config.cluster_col].map(lambda value: cluster_label_for_value(value, config))

    merge_keys = [column for column in ["file", "cell_id", config.fov_col] if column in df.columns and column in x_df_all.columns]
    if not merge_keys:
        raise RuntimeError("Unable to merge cluster assignments back onto the per-cell table.")
    clustered_df = df.drop(columns=[config.cluster_col], errors="ignore").merge(
        x_df_all[merge_keys + ["cluster_raw", config.cluster_col, "cluster_name"]],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    return clustered_df


def plot_cluster_composition(df: pd.DataFrame, config: ClusteredAnalysisConfig) -> pd.DataFrame:
    comp = (
        df.dropna(subset=[config.treatment_col, config.cluster_col])
        .assign(
            **{
                config.treatment_col: lambda x: x[config.treatment_col].astype(str),
            }
        )
        .groupby([config.treatment_col, config.cluster_col], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    comp["prop"] = comp.groupby(config.treatment_col, observed=True)["count"].transform(lambda x: x / x.sum())
    comp["cluster_label"] = comp[config.cluster_col].map(lambda value: cluster_label_for_value(value, config))
    treatment_order = [
        treatment for treatment in config.treatment_order if treatment in comp[config.treatment_col].unique().tolist()
    ]
    cluster_order = [
        level
        for level in expected_heatmap_cluster_order(config)
        if level in comp[config.cluster_col].dropna().astype(int).unique().tolist()
    ]
    if treatment_order:
        comp[config.treatment_col] = pd.Categorical(comp[config.treatment_col], categories=treatment_order, ordered=True)
    if cluster_order:
        comp[config.cluster_col] = pd.Categorical(comp[config.cluster_col], categories=cluster_order, ordered=True)
    comp = comp.sort_values([config.treatment_col, config.cluster_col]).reset_index(drop=True)
    comp.to_csv(config.clustering_dir / "cluster_composition_by_treatment.csv", index=False)

    plot_comp = comp.copy()
    plot_comp[config.treatment_col] = plot_comp[config.treatment_col].astype(str)
    plot_comp[config.cluster_col] = plot_comp[config.cluster_col].astype(str)
    present_levels = plot_comp[config.cluster_col].unique().tolist()
    preferred_levels = [
        str(level)
        for level in expected_heatmap_cluster_order(config)
        if str(level) in present_levels
    ]
    levels = preferred_levels or sorted(present_levels, key=sort_key)
    palette = make_cluster_palette(levels, config)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.barplot(
        data=plot_comp,
        x=config.treatment_col,
        y="prop",
        hue=config.cluster_col,
        order=treatment_order or None,
        hue_order=levels,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Proportion of cells")
    ax.tick_params(axis="x", rotation=28)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(config.graphs_dir / "clusters_barplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(config.graphs_dir / "cluster_composition_by_treatment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return comp


def plot_two_term_response_distributions(
    df: pd.DataFrame,
    *,
    response_cols: Sequence[str],
    config: ClusteredAnalysisConfig,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    present_levels = df[config.cluster_col].dropna().astype(str).unique().tolist()
    preferred_levels = [
        str(level)
        for level in expected_heatmap_cluster_order(config)
        if str(level) in present_levels
    ]
    levels = preferred_levels or sorted(present_levels, key=sort_key)
    palette = make_cluster_palette(levels, config)

    for response_col in response_cols:
        needed = [response_col, config.treatment_col, config.cluster_col]
        subset = df.dropna(subset=[column for column in needed if column in df.columns]).copy()
        if subset.empty:
            continue

        subset[config.treatment_col] = subset[config.treatment_col].astype(str)
        subset[config.cluster_col] = subset[config.cluster_col].astype(str)
        treatment_order = [
            treatment
            for treatment in config.treatment_order
            if treatment in subset[config.treatment_col].unique().tolist()
        ]
        axis_label = response_axis_label(response_col, config)
        is_auc = response_col == "AUC" or response_col.endswith("_AUC")

        if not is_auc:
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            sns.violinplot(
                data=subset,
                x=config.treatment_col,
                y=response_col,
                hue=config.cluster_col,
                order=treatment_order or None,
                hue_order=levels,
                palette=palette,
                cut=0,
                inner="quartile",
                ax=ax,
            )
            ax.set_xlabel("Treatment")
            ax.set_ylabel(axis_label)
            ax.tick_params(axis="x", rotation=28)
            ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(config.graphs_dir / f"violin_{response_col}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        sns.boxplot(
            data=subset,
            x=config.treatment_col,
            y=response_col,
            hue=config.cluster_col,
            order=treatment_order or None,
            hue_order=levels,
            palette=palette,
            showfliers=False,
            ax=ax,
        )
        ax.set_xlabel("Treatment")
        ax.set_ylabel(axis_label)
        ax.tick_params(axis="x", rotation=28)
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(config.graphs_dir / f"boxplot_{response_col}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def subsample_points_legacy(
    df: pd.DataFrame,
    *,
    every: int | None,
    seed: int,
) -> pd.DataFrame:
    if df.empty or every is None or every <= 1:
        return df.copy()
    return df.sample(frac=1, random_state=seed).iloc[::every].copy()


def emm_from_mixedlm(result, levels: Sequence[str], *, ref: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params, _, _, cov = fixed_effect_parts(result)
    names = list(params.index)
    if "Intercept" not in names:
        raise RuntimeError("Intercept not found in mixed model parameters.")

    intercept_idx = names.index("Intercept")
    means: list[float] = []
    ses: list[float] = []
    for level in levels:
        vector = np.zeros(len(params), dtype=float)
        vector[intercept_idx] = 1.0
        treatment_name = f"treatment[T.{level}]"
        if level != ref and treatment_name in names:
            vector[names.index(treatment_name)] = 1.0
        means.append(float(vector @ params))
        variance = float(vector @ cov @ vector)
        ses.append(float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan)

    means_arr = np.asarray(means, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    z_crit = 1.96
    return means_arr, means_arr - z_crit * ses_arr, means_arr + z_crit * ses_arr


def build_legacy_mfi_model_overlay(
    sub: pd.DataFrame,
    *,
    measurement: str,
    x_order: Sequence[str],
    config: ClusteredAnalysisConfig,
) -> pd.DataFrame:
    formula = build_model_formula("one_term")
    model_df, treatment_levels, _ = prepare_model_df(
        sub,
        response_col="value_norm",
        model_kind="one_term",
        config=config,
    )
    fit_summary_row = base_model_summary_row(
        response_var=measurement,
        model_kind="one_term_norm_plot",
        formula=formula,
        model_df=model_df,
        config=config,
    )
    fit, random_structure, warning_messages = fit_mixedlm_nested(
        formula,
        model_df,
        group_col=config.group_col,
        fov_col=config.fov_col,
        optimizer_sequence=config.optimizer_sequence,
    )
    means, ci_low, ci_high = emm_from_mixedlm(fit, treatment_levels, ref=config.treatment_reference)
    overlay_df = pd.DataFrame(
        {
            "response_var": measurement,
            "model_kind": "one_term_norm_plot",
            "treatment": treatment_levels,
            "emm": means,
            "CI_lower": ci_low,
            "CI_upper": ci_high,
            "converged": bool(getattr(fit, "converged", False)),
            "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
            "warning_count": len(warning_messages),
            "warning_messages": " | ".join(warning_messages),
            "random_structure": random_structure,
            "avg_group_size": fit_summary_row["avg_group_size"],
            "avg_fov_size": fit_summary_row["avg_fov_size"],
        }
    )
    if x_order:
        overlay_df["treatment"] = pd.Categorical(
            overlay_df["treatment"].astype(str),
            categories=list(x_order),
            ordered=True,
        )
        overlay_df = overlay_df.sort_values("treatment", na_position="last").reset_index(drop=True)
        overlay_df["treatment"] = overlay_df["treatment"].astype(str)
    return overlay_df


def add_model_ci_overlay(
    ax,
    *,
    overlay_df: pd.DataFrame,
    x_order: Sequence[str],
) -> None:
    if overlay_df.empty:
        return

    stats_df = overlay_df.set_index("treatment").reindex(x_order)
    y_mean = stats_df["emm"].to_numpy(dtype=float)
    y_lo = stats_df["CI_lower"].to_numpy(dtype=float)
    y_hi = stats_df["CI_upper"].to_numpy(dtype=float)
    if not np.isfinite(y_mean).any():
        return

    x_pos = np.arange(len(x_order))

    yerr = np.vstack(
        [
            np.where(np.isfinite(y_mean) & np.isfinite(y_lo), y_mean - y_lo, 0.0),
            np.where(np.isfinite(y_mean) & np.isfinite(y_hi), y_hi - y_mean, 0.0),
        ]
    )
    ax.errorbar(
        x_pos,
        y_mean,
        yerr=yerr,
        fmt="D",
        ms=6,
        lw=1.8,
        capsize=6,
        color="black",
        zorder=5,
    )
    for xpos, mean_value in zip(x_pos, y_mean):
        if np.isfinite(mean_value):
            ax.hlines(mean_value, xpos - 0.15, xpos + 0.15, colors="black", linewidth=2.2, zorder=6)


def plot_legacy_marker_mfi_violins(
    norm_long: pd.DataFrame,
    *,
    config: ClusteredAnalysisConfig,
) -> None:
    if norm_long.empty:
        return

    measurement_values = norm_long["measurement"].astype(str).unique().tolist()
    measurements = [measurement for measurement in config.cluster_feature_cols if measurement in measurement_values]
    if not measurements:
        return

    replicate_col = "N_label" if "N_label" in norm_long.columns else config.group_col
    if replicate_col not in norm_long.columns:
        return

    treatment_values = norm_long[config.treatment_col].dropna().astype(str).unique().tolist()
    treatment_order = [treatment for treatment in config.treatment_order if treatment in treatment_values]
    if not treatment_order:
        treatment_order = sorted(treatment_values, key=sort_key)
    if not treatment_order:
        return

    rep_order = sorted(norm_long[replicate_col].dropna().astype(str).unique().tolist(), key=sort_key)
    rep_palette = build_rep_palette(rep_order)
    rep_display_map = build_simplified_replicate_map(rep_order)
    rep_display_palette = {rep_display_map[label]: rep_palette[label] for label in rep_order}
    write_standalone_replicate_legend(
        rep_order,
        rep_palette=rep_palette,
        out_path=config.graphs_dir / "replicate_legend_one_term_mfi_violin.png",
    )

    sns.set_theme(style="whitegrid", context="talk")
    overlay_rows: list[pd.DataFrame] = []
    for measurement in measurements:
        subset = norm_long[norm_long["measurement"].astype(str) == measurement].copy()
        subset = subset.dropna(subset=[config.treatment_col, "value_norm", replicate_col])
        if subset.empty:
            continue

        subset[config.treatment_col] = subset[config.treatment_col].astype(str)
        subset[replicate_col] = subset[replicate_col].astype(str)
        subset["replicate_display"] = subset[replicate_col].map(rep_display_map).fillna(subset[replicate_col])
        sampled_points = subsample_points_legacy(
            subset,
            every=config.legacy_mfi_point_every,
            seed=config.legacy_mfi_point_seed,
        )

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        sns.violinplot(
            data=subset,
            x=config.treatment_col,
            y="value_norm",
            order=treatment_order,
            inner="box",
            cut=0,
            linewidth=1,
            color="lightgray",
            ax=ax,
        )
        sns.stripplot(
            data=sampled_points,
            x=config.treatment_col,
            y="value_norm",
            hue="replicate_display",
            order=treatment_order,
            hue_order=[rep_display_map[label] for label in rep_order],
            palette=rep_display_palette,
            dodge=False,
            jitter=0.15,
            alpha=0.8,
            size=2.6,
            edgecolor="white",
            linewidth=0.3,
            ax=ax,
            zorder=3,
        )
        overlay_df = build_legacy_mfi_model_overlay(
            subset,
            measurement=measurement,
            x_order=treatment_order,
            config=config,
        )
        add_model_ci_overlay(ax, overlay_df=overlay_df, x_order=treatment_order)
        overlay_rows.append(overlay_df)

        ax.set_title(measurement)
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Norm. value (Vehicle = 1)")
        ax.tick_params(axis="x", labelsize=11, rotation=28)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        fig.tight_layout()
        fig.savefig(
            config.graphs_dir / f"{slugify_filename_part(measurement)}_violin.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)

    if overlay_rows:
        overlay_out = pd.concat(overlay_rows, ignore_index=True)
        write_csv_with_lock_fallback(overlay_out, config.stats_dir / "legacy_one_term_norm_plot_emms.csv")


def build_cell_count_tables(
    df: pd.DataFrame,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [config.treatment_col, config.group_col, config.fov_col]
    subset = df.dropna(subset=[column for column in required if column in df.columns]).copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_fov_keys = [config.treatment_col, config.group_col, config.fov_col]
    per_fov_extra = [column for column in ["N_label", "Well_ID", "FOV"] if column in subset.columns]

    if "num_nuclei" in subset.columns:
        subset["num_nuclei"] = pd.to_numeric(subset["num_nuclei"], errors="coerce")
        per_fov = (
            subset.dropna(subset=["num_nuclei"])
            .groupby(per_fov_keys + per_fov_extra, observed=True)["num_nuclei"]
            .sum()
            .rename("n_cells")
            .reset_index()
        )
    else:
        per_fov = (
            subset.groupby(per_fov_keys + per_fov_extra, observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )

    if per_fov.empty:
        return per_fov, pd.DataFrame()

    per_fov["cells_per_fov"] = per_fov["n_cells"].astype(float)

    per_rep_keys = [config.group_col, config.treatment_col]
    per_rep_extra = [column for column in ["N_label"] if column in per_fov.columns]
    per_replicate = (
        per_fov.groupby(per_rep_keys + per_rep_extra, observed=True)
        .agg(
            n_cells=("n_cells", "sum"),
            n_fovs=(config.fov_col, "nunique"),
        )
        .reset_index()
    )
    if "Well_ID" in per_fov.columns:
        n_wells = (
            per_fov.groupby(per_rep_keys + per_rep_extra, observed=True)["Well_ID"]
            .nunique()
            .rename("n_wells")
            .reset_index()
        )
        per_replicate = per_replicate.merge(
            n_wells,
            on=per_rep_keys + per_rep_extra,
            how="left",
            validate="one_to_one",
        )

    per_replicate["cells_per_fov"] = per_replicate["n_cells"] / per_replicate["n_fovs"]
    per_replicate[config.treatment_col] = per_replicate[config.treatment_col].astype(str)

    vehicle_group_mean = (
        per_replicate[per_replicate[config.treatment_col] == config.treatment_reference]
        .groupby(config.group_col, observed=True)["cells_per_fov"]
        .mean()
        .rename("vehicle_mean_group")
    )
    vehicle_global_mean = float(
        per_replicate.loc[
            per_replicate[config.treatment_col] == config.treatment_reference,
            "cells_per_fov",
        ].mean()
    )
    per_replicate = per_replicate.merge(vehicle_group_mean, on=config.group_col, how="left")
    ref = np.where(
        per_replicate["vehicle_mean_group"].notna() & (per_replicate["vehicle_mean_group"] != 0),
        per_replicate["vehicle_mean_group"],
        vehicle_global_mean if np.isfinite(vehicle_global_mean) and vehicle_global_mean != 0 else np.nan,
    )
    per_replicate["cells_per_fov_norm"] = np.where(
        np.isfinite(ref),
        per_replicate["cells_per_fov"] / ref,
        per_replicate["cells_per_fov"],
    )

    treatment_rank = {treatment: idx for idx, treatment in enumerate(config.treatment_order)}
    for out_df in (per_fov, per_replicate):
        out_df["_treatment_rank"] = out_df[config.treatment_col].astype(str).map(treatment_rank).fillna(len(treatment_rank))
        sort_cols = ["_treatment_rank", config.group_col]
        if config.fov_col in out_df.columns:
            sort_cols.append(config.fov_col)
        out_df.sort_values(sort_cols, inplace=True)
        out_df.drop(columns="_treatment_rank", inplace=True)
        out_df.reset_index(drop=True, inplace=True)

    return per_fov, per_replicate


def run_cell_count_rm_anova(
    per_replicate: pd.DataFrame,
    *,
    out_stats_dir: Path,
    config: ClusteredAnalysisConfig,
) -> dict[str, pd.DataFrame]:
    analysis_cols = [
        column
        for column in [
            config.group_col,
            "N_label",
            config.treatment_col,
            "n_cells",
            "n_fovs",
            "n_wells",
            "vehicle_mean_group",
            "cells_per_fov",
            "cells_per_fov_norm",
        ]
        if column in per_replicate.columns
    ]
    analysis_input = per_replicate[analysis_cols].copy() if not per_replicate.empty else pd.DataFrame(columns=analysis_cols)
    if analysis_input.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_stats_dir / "cell_counts_replicate_level_analysis_input.csv", index=False)
        empty.to_csv(out_stats_dir / "cell_counts_global_repeated_measures_tests.csv", index=False)
        empty.to_csv(out_stats_dir / "cell_counts_cells_per_fov_pairwise_vs_vehicle.csv", index=False)
        return {"analysis_input": empty, "anova": empty, "pairwise": empty}

    analysis_input[config.treatment_col] = analysis_input[config.treatment_col].astype(str)
    present_treatments = [
        treatment
        for treatment in config.treatment_order
        if treatment in analysis_input[config.treatment_col].unique().tolist()
    ]
    analysis_input = analysis_input[analysis_input[config.treatment_col].isin(present_treatments)].copy()

    wide = (
        analysis_input.pivot(
            index=config.group_col,
            columns=config.treatment_col,
            values="cells_per_fov",
        )
        .reindex(columns=present_treatments)
    )
    complete_subjects = [str(value) for value in wide.dropna().index.tolist()]
    all_subjects = [str(value) for value in wide.index.tolist()]
    excluded_subjects = [value for value in all_subjects if value not in complete_subjects]

    analysis_input = analysis_input[analysis_input[config.group_col].astype(str).isin(complete_subjects)].copy()
    treatment_rank = {treatment: idx for idx, treatment in enumerate(present_treatments)}
    analysis_input["_treatment_rank"] = analysis_input[config.treatment_col].map(treatment_rank).fillna(len(treatment_rank))
    analysis_input.sort_values([config.group_col, "_treatment_rank"], inplace=True)
    analysis_input.drop(columns="_treatment_rank", inplace=True)
    analysis_input.reset_index(drop=True, inplace=True)
    analysis_input.to_csv(out_stats_dir / "cell_counts_replicate_level_analysis_input.csv", index=False)

    if len(complete_subjects) >= 2 and len(present_treatments) >= 2:
        try:
            anova = AnovaRM(
                data=analysis_input,
                depvar="cells_per_fov",
                subject=config.group_col,
                within=[config.treatment_col],
            ).fit()
            anova_df = (
                anova.anova_table
                .reset_index()
                .rename(
                    columns={
                        "index": "effect",
                        "F Value": "F",
                        "Num DF": "num_df",
                        "Den DF": "den_df",
                        "Pr > F": "p_value",
                    }
                )
            )
            anova_df["status"] = "ok"
            anova_df["error"] = ""
        except Exception as exc:
            anova_df = pd.DataFrame(
                [
                    {
                        "effect": config.treatment_col,
                        "F": np.nan,
                        "num_df": np.nan,
                        "den_df": np.nan,
                        "p_value": np.nan,
                        "status": "failed",
                        "error": str(exc),
                    }
                ]
            )
    else:
        anova_df = pd.DataFrame(
            [
                {
                    "effect": config.treatment_col,
                    "F": np.nan,
                    "num_df": np.nan,
                    "den_df": np.nan,
                    "p_value": np.nan,
                    "status": "insufficient_data",
                    "error": "Need at least 2 complete biological replicates and 2 treatments.",
                }
            ]
        )

    anova_df["metric"] = "cells_per_fov"
    anova_df["test"] = "repeated_measures_anova"
    anova_df["analysis_level"] = "biological_replicate"
    anova_df["reference_treatment"] = config.treatment_reference
    anova_df["n_biological_replicates"] = len(complete_subjects)
    anova_df["n_treatments"] = len(present_treatments)
    anova_df["complete_biological_replicates"] = "|".join(complete_subjects)
    anova_df["excluded_biological_replicates"] = "|".join(excluded_subjects)
    anova_df.to_csv(out_stats_dir / "cell_counts_global_repeated_measures_tests.csv", index=False)

    pairwise_rows: list[dict[str, object]] = []
    if config.treatment_reference in present_treatments and complete_subjects:
        wide_complete = wide.loc[complete_subjects]
        for treatment in present_treatments:
            if treatment == config.treatment_reference:
                continue

            paired = wide_complete[[config.treatment_reference, treatment]].dropna()
            n_pairs = int(len(paired))
            vehicle_mean = float(paired[config.treatment_reference].mean()) if n_pairs else np.nan
            treatment_mean = float(paired[treatment].mean()) if n_pairs else np.nan
            mean_ratio = treatment_mean / vehicle_mean if n_pairs and np.isfinite(vehicle_mean) and vehicle_mean != 0 else np.nan

            if n_pairs >= 2:
                diffs = paired[treatment] - paired[config.treatment_reference]
                t_stat, p_value = ttest_rel(paired[treatment], paired[config.treatment_reference])
                mean_diff = float(diffs.mean())
                sd_diff = float(diffs.std(ddof=1))
                se_diff = sd_diff / np.sqrt(n_pairs) if np.isfinite(sd_diff) else np.nan
                t_crit = float(t.ppf(0.975, df=n_pairs - 1))
                ci_low = mean_diff - t_crit * se_diff if np.isfinite(se_diff) else np.nan
                ci_high = mean_diff + t_crit * se_diff if np.isfinite(se_diff) else np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
                mean_diff = np.nan
                ci_low = np.nan
                ci_high = np.nan

            pairwise_rows.append(
                {
                    "metric": "cells_per_fov",
                    "test": "paired_t_test",
                    "analysis_level": "biological_replicate",
                    "reference_treatment": config.treatment_reference,
                    "treatment": treatment,
                    "n_biol_rep": n_pairs,
                    "vehicle_mean": vehicle_mean,
                    "treatment_mean": treatment_mean,
                    "mean_diff_treatment_minus_vehicle": mean_diff,
                    "diff_ci_low": ci_low,
                    "diff_ci_high": ci_high,
                    "mean_ratio_treatment_over_vehicle": mean_ratio,
                    "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["p_stars"] = pairwise_df["p_value"].map(p_to_stars)
        pairwise_df = append_fdr_columns(pairwise_df)
        pairwise_df.sort_values(["fdr_p_value", "treatment"], inplace=True, na_position="last")
        pairwise_df.reset_index(drop=True, inplace=True)
    pairwise_df.to_csv(out_stats_dir / "cell_counts_cells_per_fov_pairwise_vs_vehicle.csv", index=False)

    return {
        "analysis_input": analysis_input,
        "anova": anova_df,
        "pairwise": pairwise_df,
    }


def write_cell_count_outputs(df: pd.DataFrame, config: ClusteredAnalysisConfig) -> dict[str, pd.DataFrame]:
    per_fov, per_replicate = build_cell_count_tables(df, config)
    if per_fov.empty or per_replicate.empty:
        empty = pd.DataFrame()
        empty.to_csv(config.stats_dir / "cell_counts_per_fov.csv", index=False)
        empty.to_csv(config.stats_dir / "cell_counts_cells_per_fov_all_treatments.csv", index=False)
        empty.to_csv(config.stats_dir / "cell_counts_replicate_level_analysis_input.csv", index=False)
        empty.to_csv(config.stats_dir / "cell_counts_global_repeated_measures_tests.csv", index=False)
        empty.to_csv(config.stats_dir / "cell_counts_cells_per_fov_pairwise_vs_vehicle.csv", index=False)
        return {
            "per_fov": empty,
            "per_replicate": empty,
            "analysis_input": empty,
            "rm_anova": empty,
            "pairwise": empty,
        }

    per_fov.to_csv(config.stats_dir / "cell_counts_per_fov.csv", index=False)
    per_replicate.to_csv(config.stats_dir / "cell_counts_cells_per_fov_all_treatments.csv", index=False)

    present_treatments = [
        treatment
        for treatment in config.treatment_order
        if treatment in per_replicate[config.treatment_col].astype(str).unique().tolist()
    ]
    rep_levels = (
        sorted(per_replicate["N_label"].dropna().astype(str).unique().tolist(), key=sort_key)
        if "N_label" in per_replicate.columns
        else []
    )
    rep_palette = build_rep_palette(rep_levels) if rep_levels else {}

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(
        data=per_replicate,
        x=config.treatment_col,
        y="cells_per_fov",
        order=present_treatments,
        ax=ax,
        showfliers=False,
        color="lightgray",
    )
    if rep_levels:
        sns.stripplot(
            data=per_replicate,
            x=config.treatment_col,
            y="cells_per_fov",
            order=present_treatments,
            hue="N_label",
            hue_order=rep_levels,
            palette=rep_palette,
            dodge=False,
            jitter=0.15,
            size=6,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
            zorder=3,
        )
        legend = ax.get_legend()
        if legend is not None:
            ax.legend(title="Replicate (N)", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    else:
        sns.stripplot(
            data=per_replicate,
            x=config.treatment_col,
            y="cells_per_fov",
            order=present_treatments,
            ax=ax,
            color="black",
            alpha=0.55,
            size=4,
        )
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Cells per FOV")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(config.graphs_dir / "cell_counts_cells_per_fov_boxplot_all_treatments.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if config.cell_count_inference != "rm_anova":
        raise RuntimeError(
            "This rm_anova bundle only supports cell_count_inference='rm_anova'."
        )
    inference_results = run_cell_count_rm_anova(
        per_replicate,
        out_stats_dir=config.stats_dir,
        config=config,
    )

    return {
        "per_fov": per_fov,
        "per_replicate": per_replicate,
        "analysis_input": inference_results["analysis_input"],
        "rm_anova": inference_results["anova"],
        "pairwise": inference_results["pairwise"],
    }


def mixedlm_summary_to_csv_frame(summary, metadata: dict[str, object]) -> pd.DataFrame:
    def _table_rows(table) -> list[list[object]]:
        if isinstance(table, pd.DataFrame):
            frame = table.reset_index()
            rows = [frame.columns.astype(str).tolist()]
            rows.extend(frame.astype(object).values.tolist())
            return rows
        if hasattr(table, "data"):
            return list(table.data)
        raise TypeError(f"Unsupported summary table type: {type(table)!r}")

    rows = []
    for table_index, table in enumerate(summary.tables):
        for row_index, row in enumerate(_table_rows(table)):
            out_row = dict(metadata)
            out_row["table_index"] = table_index
            out_row["row_index"] = row_index
            for col_index, value in enumerate(row):
                out_row[f"col_{col_index}"] = value
            rows.append(out_row)
    if not rows:
        raise RuntimeError("MixedLM summary had no table rows to export.")
    return pd.DataFrame(rows)


def write_mixedlm_full_report(
    fit,
    out_stats_dir: Path,
    *,
    report_name: str,
    formula: str,
    response_var: str,
    model_kind: str,
    random_structure: str,
    data: pd.DataFrame,
    warning_messages: Sequence[str],
) -> Path:
    report_dir = out_stats_dir / "mixedlm_full_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_stem = slugify_filename_part(report_name)
    report_path = report_dir / f"{report_stem}.txt"
    report_csv_path = report_dir / f"{report_stem}.csv"

    metadata = {
        "report_name": report_name,
        "response_var": response_var,
        "model_kind": model_kind,
        "formula": formula,
        "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
        "random_structure": random_structure,
        "n_rows": len(data),
        "n_biological_replicates": data["N"].nunique() if "N" in data.columns else np.nan,
        "n_fovs": data["FOV_ID"].nunique() if "FOV_ID" in data.columns else np.nan,
        "converged": bool(getattr(fit, "converged", False)),
        "warning_messages": " | ".join(warning_messages),
    }

    summary = fit.summary()
    rows = mixedlm_summary_to_csv_frame(summary, metadata)
    header_lines = [f"{key}: {value}" for key, value in metadata.items()]

    report_path.write_text(
        "\n".join(header_lines) + "\n\n" + summary.as_text() + "\n",
        encoding="utf-8",
    )
    rows.to_csv(report_csv_path, index=False)
    return report_path


def fixed_effect_parts(fit) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    fe_names = list(fit.fe_params.index)
    params = fit.fe_params.loc[fe_names].astype(float)
    bse = fit.bse.loc[fe_names].astype(float)
    pvalues = fit.pvalues.loc[fe_names].astype(float)
    cov = fit.cov_params().loc[fe_names, fe_names]
    return params, bse, pvalues, cov


def contrast_from_vector(
    params: pd.Series,
    cov: pd.DataFrame,
    vector: np.ndarray,
) -> tuple[float, float, float, float]:
    estimate = float(vector @ params.values)
    variance = float(vector @ cov.values @ vector)
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance)) if np.isfinite(variance) else np.nan
    z_value = estimate / se if np.isfinite(se) and se > 0 else np.nan
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return estimate, se, z_value, p_value


def sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def summarise_group_structure(
    data: pd.DataFrame,
    *,
    group_col: str,
    fov_col: str,
    treatment_col: str | None = None,
) -> dict[str, float]:
    group_sizes = data.groupby(group_col, observed=True).size()
    fov_sizes = data.groupby(fov_col, observed=True).size()
    fovs_per_group = data.groupby(group_col, observed=True)[fov_col].nunique()
    group_treatment_sizes = pd.Series(dtype=float)
    if treatment_col is not None and treatment_col in data.columns:
        group_treatment_sizes = data.groupby([group_col, treatment_col], observed=True).size()
    return {
        "n_rows": int(len(data)),
        "n_groups": int(group_sizes.shape[0]),
        "n_fovs": int(fov_sizes.shape[0]),
        "n_group_treatment_combos": int(group_treatment_sizes.shape[0]) if not group_treatment_sizes.empty else np.nan,
        "avg_group_size": float(group_sizes.mean()) if not group_sizes.empty else np.nan,
        "median_group_size": float(group_sizes.median()) if not group_sizes.empty else np.nan,
        "avg_group_treatment_size": float(group_treatment_sizes.mean()) if not group_treatment_sizes.empty else np.nan,
        "median_group_treatment_size": float(group_treatment_sizes.median()) if not group_treatment_sizes.empty else np.nan,
        "avg_fov_size": float(fov_sizes.mean()) if not fov_sizes.empty else np.nan,
        "median_fov_size": float(fov_sizes.median()) if not fov_sizes.empty else np.nan,
        "avg_fovs_per_group": float(fovs_per_group.mean()) if not fovs_per_group.empty else np.nan,
        "median_fovs_per_group": float(fovs_per_group.median()) if not fovs_per_group.empty else np.nan,
        "n_biological_replicates": int(group_sizes.shape[0]),
        "avg_cells_per_biological_replicate_total": float(group_sizes.mean()) if not group_sizes.empty else np.nan,
        "median_cells_per_biological_replicate_total": float(group_sizes.median()) if not group_sizes.empty else np.nan,
        "avg_cells_per_biological_replicate_treatment": float(group_treatment_sizes.mean()) if not group_treatment_sizes.empty else np.nan,
        "median_cells_per_biological_replicate_treatment": float(group_treatment_sizes.median()) if not group_treatment_sizes.empty else np.nan,
        "avg_cells_per_fov": float(fov_sizes.mean()) if not fov_sizes.empty else np.nan,
        "median_cells_per_fov": float(fov_sizes.median()) if not fov_sizes.empty else np.nan,
        "avg_fovs_per_biological_replicate": float(fovs_per_group.mean()) if not fovs_per_group.empty else np.nan,
        "median_fovs_per_biological_replicate": float(fovs_per_group.median()) if not fovs_per_group.empty else np.nan,
    }


def fit_mixedlm_nested(
    formula: str,
    data: pd.DataFrame,
    *,
    group_col: str,
    fov_col: str,
    optimizer_sequence: Sequence[str],
) -> tuple[object, str, list[str]]:
    if group_col not in data.columns or data[group_col].nunique() < 2:
        raise RuntimeError(f"Need at least two biological replicate levels in '{group_col}'.")
    if fov_col not in data.columns or data[fov_col].nunique() < 2:
        raise RuntimeError(f"Need at least two FOV levels in '{fov_col}'.")

    vc_formula = {"FOV": f"0 + C({fov_col})"}
    last_exception: Exception | None = None
    last_fit = None
    last_warnings: list[str] = []

    for method in optimizer_sequence:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data,
                    groups=data[group_col],
                    vc_formula=vc_formula,
                    re_formula="1",
                )
                fit = model.fit(reml=True, method=method, disp=False)

            warning_messages = []
            for warning in caught:
                message = str(warning.message).strip()
                if message and message not in warning_messages:
                    warning_messages.append(message)

            setattr(fit, "_codex_optimizer", method)
            setattr(fit, "_codex_warning_messages", warning_messages)

            last_fit = fit
            last_warnings = warning_messages
            if bool(getattr(fit, "converged", False)):
                return fit, f"{group_col} + {fov_col}", warning_messages
        except Exception as exc:
            last_exception = exc

    if last_fit is not None:
        return last_fit, f"{group_col} + {fov_col}", last_warnings

    raise RuntimeError(
        f"MixedLM failed for formula '{formula}' with optimizers {list(optimizer_sequence)}: {last_exception}"
    ) from last_exception


def prepare_model_df(
    df: pd.DataFrame,
    *,
    response_col: str,
    model_kind: str,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    keep_cols = [response_col, config.treatment_col, config.group_col, config.fov_col]
    if model_kind == "two_term":
        keep_cols.append(config.cluster_col)

    model_df = df[keep_cols].copy()
    model_df["response"] = pd.to_numeric(model_df[response_col], errors="coerce")

    needed = ["response", config.treatment_col, config.group_col, config.fov_col]
    if model_kind == "two_term":
        needed.append(config.cluster_col)
    model_df = model_df.dropna(subset=needed).copy()

    present_treatments = [
        treatment
        for treatment in config.treatment_order
        if treatment in model_df[config.treatment_col].astype(str).unique().tolist()
    ]
    if config.treatment_reference not in present_treatments:
        raise RuntimeError(
            f"Reference treatment '{config.treatment_reference}' was not found for '{response_col}'."
        )
    if len(present_treatments) < 2:
        raise RuntimeError(
            f"Need at least two treatment levels for '{response_col}'; found {present_treatments}."
        )

    other_treatments = sorted(
        [
            treatment
            for treatment in model_df[config.treatment_col].astype(str).unique().tolist()
            if treatment not in present_treatments
        ]
    )
    category_order = [config.treatment_reference]
    category_order.extend([t for t in present_treatments if t != config.treatment_reference])
    category_order.extend(other_treatments)
    model_df[config.treatment_col] = pd.Categorical(
        model_df[config.treatment_col].astype(str),
        categories=category_order,
        ordered=True,
    )

    cluster_levels: list[str] = []
    if model_kind == "two_term":
        cluster_levels = sorted(
            model_df[config.cluster_col].astype(str).unique().tolist(),
            key=sort_key,
        )
        if len(cluster_levels) < 2:
            raise RuntimeError(
                f"Need at least two cluster levels for '{response_col}'; found {cluster_levels}."
            )
        model_df[config.cluster_col] = pd.Categorical(
            model_df[config.cluster_col].astype(str),
            categories=cluster_levels,
            ordered=True,
        )

    model_df[config.group_col] = model_df[config.group_col].astype(str)
    model_df[config.fov_col] = model_df[config.fov_col].astype(str)

    unique_values = model_df["response"].nunique()
    if unique_values < config.min_unique_values:
        raise RuntimeError(
            f"MixedLM for '{response_col}' needs at least {config.min_unique_values} unique values; found {unique_values}."
        )

    return model_df, category_order, cluster_levels


def build_model_formula(model_kind: str) -> str:
    if model_kind == "one_term":
        return "response ~ treatment"
    if model_kind == "two_term":
        return "response ~ treatment * cluster"
    raise ValueError(f"Unknown model kind: {model_kind}")


def base_model_summary_row(
    *,
    response_var: str,
    model_kind: str,
    formula: str,
    model_df: pd.DataFrame,
    config: ClusteredAnalysisConfig,
) -> dict[str, object]:
    row = {
        "response_var": response_var,
        "model_kind": model_kind,
        "formula": formula,
        "random_structure": f"{config.group_col} + {config.fov_col}",
        "data_csv": input_source_label(config),
    }
    row.update(
        summarise_group_structure(
            model_df,
            group_col=config.group_col,
            fov_col=config.fov_col,
            treatment_col=config.treatment_col,
        )
    )
    return row


def append_fdr_columns(
    df: pd.DataFrame,
    *,
    p_value_col: str = "p_value",
    output_col: str = "fdr_p_value",
    stars_output_col: str = "fdr_p_stars",
) -> pd.DataFrame:
    df = df.copy()
    mask = df[p_value_col].notna()
    corrected = np.full(len(df), np.nan, dtype=float)
    if mask.any():
        corrected[mask.to_numpy()] = multipletests(
            df.loc[mask, p_value_col].astype(float).to_numpy(),
            method="fdr_bh",
        )[1]
    df[output_col] = corrected
    df[stars_output_col] = df[output_col].map(p_to_stars)
    return df


def append_term_fdr_columns(
    df: pd.DataFrame,
    *,
    p_value_col: str = "p_value",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    if "p_stars" not in out.columns and p_value_col in out.columns:
        out["p_stars"] = out[p_value_col].map(p_to_stars)
    return append_fdr_columns(out, p_value_col=p_value_col)


def write_csv_with_lock_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def cramers_v_from_table(table: pd.DataFrame | np.ndarray, chi2: float) -> float:
    values = table.to_numpy(dtype=float) if isinstance(table, pd.DataFrame) else np.asarray(table, dtype=float)
    if values.size == 0:
        return np.nan
    n = float(values.sum())
    if not np.isfinite(n) or n <= 0:
        return np.nan
    rows, cols = values.shape
    denom = min(rows - 1, cols - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(float(chi2) / (n * denom)))


def run_one_term_lmm(
    df: pd.DataFrame,
    *,
    response_cols: Sequence[str],
    out_stats_dir: Path,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrast_rows: list[dict[str, object]] = []
    term_rows: list[pd.DataFrame] = []
    fit_summary_rows: list[dict[str, object]] = []

    for response_col in response_cols:
        formula = build_model_formula("one_term")
        try:
            model_df, treatment_levels, _ = prepare_model_df(
                df,
                response_col=response_col,
                model_kind="one_term",
                config=config,
            )
            fit_summary_row = base_model_summary_row(
                response_var=response_col,
                model_kind="one_term",
                formula=formula,
                model_df=model_df,
                config=config,
            )

            fit, random_structure, warning_messages = fit_mixedlm_nested(
                formula,
                model_df,
                group_col=config.group_col,
                fov_col=config.fov_col,
                optimizer_sequence=config.optimizer_sequence,
            )
            fit_summary_row.update(
                {
                    "status": "ok",
                    "converged": bool(getattr(fit, "converged", False)),
                    "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
                    "warning_count": len(warning_messages),
                    "warning_messages": " | ".join(warning_messages),
                }
            )
            fit_summary_rows.append(fit_summary_row)

            write_mixedlm_full_report(
                fit,
                out_stats_dir,
                report_name=f"one_term_lmm_{response_col}",
                formula=formula,
                response_var=response_col,
                model_kind="one_term",
                random_structure=random_structure,
                data=model_df,
                warning_messages=warning_messages,
            )

            params, bse, pvalues, cov = fixed_effect_parts(fit)
            sigma = float(np.sqrt(fit.scale)) if np.isfinite(fit.scale) and fit.scale > 0 else np.nan
            term_df = pd.DataFrame(
                {
                    "response_var": response_col,
                    "model_kind": "one_term",
                    "term": params.index.astype(str),
                    "estimate": params.values,
                    "SE": bse.values,
                    "z": params.values / bse.values,
                    "p_value": pvalues.values,
                    "std_effect": params.values / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                    "residual_sd": sigma,
                    "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
                    "converged": bool(getattr(fit, "converged", False)),
                    "avg_group_size": fit_summary_row["avg_group_size"],
                    "avg_fov_size": fit_summary_row["avg_fov_size"],
                    "warning_count": len(warning_messages),
                    "random_structure": random_structure,
                }
            )
            term_rows.append(term_df)

            for treatment in treatment_levels:
                if treatment == config.treatment_reference:
                    continue

                coef_name = f"treatment[T.{treatment}]"
                if coef_name not in params.index or coef_name not in cov.index:
                    raise RuntimeError(
                        f"Expected coefficient '{coef_name}' in one-term MixedLM for '{response_col}', but it was absent."
                    )

                estimate = float(params.loc[coef_name])
                variance = float(cov.loc[coef_name, coef_name])
                se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
                z_value = estimate / se if np.isfinite(se) and se > 0 else np.nan
                p_value = float(pvalues.loc[coef_name]) if coef_name in pvalues.index else np.nan

                vehicle_mean = float(
                    model_df.loc[
                        model_df[config.treatment_col].astype(str) == config.treatment_reference,
                        "response",
                    ].mean()
                )
                treatment_mean = float(
                    model_df.loc[
                        model_df[config.treatment_col].astype(str) == treatment,
                        "response",
                    ].mean()
                )

                contrast_rows.append(
                    {
                        "response_var": response_col,
                        "model_kind": "one_term",
                        "treatment": treatment,
                        "estimate_vs_vehicle": estimate,
                        "SE": se,
                        "z": z_value,
                        "p_value": p_value,
                        "p_stars": p_to_stars(p_value),
                        "std_effect_vs_vehicle": estimate / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                        "residual_sd": sigma,
                        "vehicle_mean_raw": vehicle_mean,
                        "treatment_mean_raw": treatment_mean,
                        "n_cells": int(len(model_df)),
                        "n_groups": int(model_df[config.group_col].nunique()),
                        "n_fovs": int(model_df[config.fov_col].nunique()),
                        "avg_group_size": fit_summary_row["avg_group_size"],
                        "avg_fov_size": fit_summary_row["avg_fov_size"],
                        "avg_fovs_per_group": fit_summary_row["avg_fovs_per_group"],
                        "converged": bool(getattr(fit, "converged", False)),
                        "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
                        "warning_count": len(warning_messages),
                        "random_structure": random_structure,
                    }
                )
        except Exception as exc:
            fit_summary_rows.append(
                {
                    "response_var": response_col,
                    "model_kind": "one_term",
                    "formula": formula,
                    "random_structure": f"{config.group_col} + {config.fov_col}",
                    "status": "failed",
                    "converged": np.nan,
                    "optimizer": "",
                    "warning_count": 0,
                    "warning_messages": "",
                    "error": str(exc),
                    "data_csv": input_source_label(config),
                    "n_rows": np.nan,
                    "n_groups": np.nan,
                    "n_fovs": np.nan,
                    "n_group_treatment_combos": np.nan,
                    "avg_group_size": np.nan,
                    "median_group_size": np.nan,
                    "avg_group_treatment_size": np.nan,
                    "median_group_treatment_size": np.nan,
                    "avg_fov_size": np.nan,
                    "median_fov_size": np.nan,
                    "avg_fovs_per_group": np.nan,
                    "median_fovs_per_group": np.nan,
                    "n_biological_replicates": np.nan,
                    "avg_cells_per_biological_replicate_total": np.nan,
                    "median_cells_per_biological_replicate_total": np.nan,
                    "avg_cells_per_biological_replicate_treatment": np.nan,
                    "median_cells_per_biological_replicate_treatment": np.nan,
                    "avg_cells_per_fov": np.nan,
                    "median_cells_per_fov": np.nan,
                    "avg_fovs_per_biological_replicate": np.nan,
                    "median_fovs_per_biological_replicate": np.nan,
                }
            )
            print(f"[warn] one-term LMM failed for {response_col}: {exc}")

    contrast_df = pd.DataFrame(contrast_rows)
    if not contrast_df.empty:
        contrast_df = append_fdr_columns(contrast_df)
        write_csv_with_lock_fallback(contrast_df, out_stats_dir / "one_term_lmm_vs_vehicle.csv")

    term_df = pd.concat(term_rows, ignore_index=True) if term_rows else pd.DataFrame()
    if not term_df.empty:
        term_df = append_term_fdr_columns(term_df)
        write_csv_with_lock_fallback(term_df, out_stats_dir / "one_term_lmm_term_effects.csv")
        for response_var, response_term_df in term_df.groupby("response_var", sort=False):
            write_csv_with_lock_fallback(
                response_term_df,
                out_stats_dir / f"one_term_lmm_term_zscores_{response_var}.csv",
            )

    fit_summary_df = pd.DataFrame(fit_summary_rows)
    write_csv_with_lock_fallback(fit_summary_df, out_stats_dir / "one_term_lmm_model_fit_summary.csv")
    return contrast_df, term_df, fit_summary_df


def within_cluster_contrasts(
    fit,
    sub: pd.DataFrame,
    *,
    response_var: str,
    config: ClusteredAnalysisConfig,
    fit_summary_row: dict[str, object],
) -> pd.DataFrame:
    params, _, _, cov = fixed_effect_parts(fit)
    names = list(params.index)
    treatment_levels = list(sub[config.treatment_col].cat.categories)
    cluster_levels = list(sub[config.cluster_col].cat.categories)
    sigma = float(np.sqrt(fit.scale)) if np.isfinite(fit.scale) and fit.scale > 0 else np.nan

    rows = []
    for treatment in treatment_levels:
        if treatment == config.treatment_reference:
            continue
        treatment_name = f"treatment[T.{treatment}]"
        if treatment_name not in names:
            raise RuntimeError(
                f"Expected treatment coefficient '{treatment_name}' in two-term MixedLM for '{response_var}', but it was absent."
            )
        for cluster in cluster_levels:
            vector = np.zeros(len(names), dtype=float)
            vector[names.index(treatment_name)] = 1.0
            interaction_name = f"treatment[T.{treatment}]:cluster[T.{cluster}]"
            if interaction_name in names:
                vector[names.index(interaction_name)] = 1.0

            estimate, se, z_value, p_value = contrast_from_vector(params, cov, vector)
            rows.append(
                {
                    "response_var": response_var,
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


def write_two_term_within_effect_heatmap(
    within_df: pd.DataFrame,
    *,
    out_dir: Path,
    response_cols: Sequence[str],
    config: ClusteredAnalysisConfig,
) -> dict[str, object]:
    if within_df is None or within_df.empty:
        return {}

    plot_df = within_df.copy()
    plot_df["cluster"] = pd.to_numeric(plot_df["cluster"], errors="coerce")
    plot_df = plot_df.dropna(subset=["cluster", "std_effect_within_cluster"]).copy()
    if plot_df.empty:
        return {}

    plot_df["cluster"] = plot_df["cluster"].astype(int)
    present_responses = plot_df["response_var"].astype(str).unique().tolist()
    row_order = [
        response
        for response in expected_heatmap_measurement_order(config, response_cols)
        if response in present_responses
    ]
    if not row_order:
        row_order = sorted(present_responses)

    expected_cluster_order = expected_heatmap_cluster_order(config)
    if expected_cluster_order:
        cluster_order = [cluster for cluster in expected_cluster_order if cluster in plot_df["cluster"].unique().tolist()]
    else:
        cluster_order = sorted(plot_df["cluster"].unique().tolist(), key=sort_key)
    treatment_order = [
        treatment
        for treatment in config.treatment_order
        if treatment != config.treatment_reference
        and treatment in plot_df["treatment"].astype(str).unique().tolist()
    ]
    if not treatment_order:
        treatment_order = sorted(plot_df["treatment"].astype(str).unique().tolist())

    col_order = [(response, cluster) for response in row_order for cluster in cluster_order]
    indexed = plot_df.set_index(["treatment", "response_var", "cluster"]).sort_index()

    value_rows: list[list[float]] = []
    star_rows: list[list[str]] = []
    for treatment in treatment_order:
        value_row: list[float] = []
        star_row: list[str] = []
        for response, cluster in col_order:
            key = (treatment, response, cluster)
            if key not in indexed.index:
                value_row.append(np.nan)
                star_row.append("")
                continue
            record = indexed.loc[key]
            value_row.append(float(record["std_effect_within_cluster"]))
            star_row.append(str(record.get("within_fdr_p_stars", record.get("within_p_stars", ""))))
        value_rows.append(value_row)
        star_rows.append(star_row)

    values_df = pd.DataFrame(value_rows, index=treatment_order, columns=pd.MultiIndex.from_tuples(col_order))
    stars_df = pd.DataFrame(star_rows, index=treatment_order, columns=values_df.columns)
    finite_values = values_df.to_numpy(dtype=float)
    if not np.isfinite(finite_values).any():
        return {}

    vmax = float(np.nanmax(np.abs(finite_values)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    measurement_label_map = (
        chapter_3_heatmap_measurement_label_map()
        if config.chapter_label == "Chapter 3"
        else chapter_4_heatmap_measurement_label_map(config)
        if config.chapter_label == "Chapter 4"
        else {}
    )
    xlabels = [
        f"{measurement_label_map.get(response, strip_measurement_label(response))}\n"
        f"{cluster}"
        for response, cluster in col_order
    ]
    width = max(10.0, 0.55 * len(col_order) + 3.0)
    height = max(3.5, 0.6 * len(treatment_order) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        values_df,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Within-cluster std effect (beta / residual SD)", "shrink": 0.9},
        annot=False,
    )
    ax.set_title("Within-cluster standardised effects (raw)", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(xlabels)) + 0.5)
    ax.set_xticklabels(xlabels, rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for row_idx in range(values_df.shape[0]):
        for col_idx in range(values_df.shape[1]):
            star = stars_df.iat[row_idx, col_idx]
            if star and star != "ns":
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    star,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                )

    for response_idx in range(1, len(row_order)):
        ax.axvline(response_idx * len(cluster_order), color="white", linewidth=2.0)

    fig.subplots_adjust(bottom=0.32, left=0.10, right=0.98, top=0.88)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "heatmap_within_std_effect_raw.png"
    out_pdf = out_dir / "heatmap_within_std_effect_raw.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return {
        "png": out_png,
        "pdf": out_pdf,
        "n_treatments": len(treatment_order),
        "n_measurements": len(row_order),
        "n_clusters": len(cluster_order),
    }


def build_emm_grid(
    fit,
    sub: pd.DataFrame,
    *,
    response_var: str,
    fit_summary_row: dict[str, object],
    config: ClusteredAnalysisConfig,
) -> pd.DataFrame:
    params, _, _, cov = fixed_effect_parts(fit)
    names = list(params.index)
    if "Intercept" not in names:
        raise RuntimeError(f"Expected an Intercept term in two-term MixedLM for '{response_var}', but it was absent.")

    treatment_levels = list(sub[config.treatment_col].cat.categories)
    cluster_levels = list(sub[config.cluster_col].cat.categories)
    sigma = float(np.sqrt(fit.scale)) if np.isfinite(fit.scale) and fit.scale > 0 else np.nan

    rows = []
    for treatment in treatment_levels:
        for cluster in cluster_levels:
            vector = np.zeros(len(names), dtype=float)
            vector[names.index("Intercept")] = 1.0

            treatment_name = f"treatment[T.{treatment}]"
            cluster_name = f"cluster[T.{cluster}]"
            interaction_name = f"treatment[T.{treatment}]:cluster[T.{cluster}]"

            if treatment_name in names:
                vector[names.index(treatment_name)] = 1.0
            if cluster_name in names:
                vector[names.index(cluster_name)] = 1.0
            if interaction_name in names:
                vector[names.index(interaction_name)] = 1.0

            estimate, se, _, _ = contrast_from_vector(params, cov, vector)
            rows.append(
                {
                    "response_var": response_var,
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


def run_two_term_lmm(
    df: pd.DataFrame,
    *,
    response_cols: Sequence[str],
    out_stats_dir: Path,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    term_rows: list[pd.DataFrame] = []
    emm_rows: list[pd.DataFrame] = []
    within_rows: list[pd.DataFrame] = []
    fit_summary_rows: list[dict[str, object]] = []

    for response_col in response_cols:
        formula = build_model_formula("two_term")
        try:
            model_df, _, _ = prepare_model_df(
                df,
                response_col=response_col,
                model_kind="two_term",
                config=config,
            )
            fit_summary_row = base_model_summary_row(
                response_var=response_col,
                model_kind="two_term",
                formula=formula,
                model_df=model_df,
                config=config,
            )

            fit, random_structure, warning_messages = fit_mixedlm_nested(
                formula,
                model_df,
                group_col=config.group_col,
                fov_col=config.fov_col,
                optimizer_sequence=config.optimizer_sequence,
            )
            fit_summary_row.update(
                {
                    "status": "ok",
                    "converged": bool(getattr(fit, "converged", False)),
                    "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
                    "warning_count": len(warning_messages),
                    "warning_messages": " | ".join(warning_messages),
                }
            )
            fit_summary_rows.append(fit_summary_row)

            write_mixedlm_full_report(
                fit,
                out_stats_dir,
                report_name=f"two_term_lmm_{response_col}",
                formula=formula,
                response_var=response_col,
                model_kind="two_term",
                random_structure=random_structure,
                data=model_df,
                warning_messages=warning_messages,
            )

            params, bse, pvalues, _ = fixed_effect_parts(fit)
            sigma = float(np.sqrt(fit.scale)) if np.isfinite(fit.scale) and fit.scale > 0 else np.nan
            term_df = pd.DataFrame(
                {
                    "response_var": response_col,
                    "model_kind": "two_term",
                    "term": params.index.astype(str),
                    "estimate": params.values,
                    "SE": bse.values,
                    "z": params.values / bse.values,
                    "p_value": pvalues.values,
                    "std_effect": params.values / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
                    "is_interaction": params.index.astype(str).str.contains("treatment.*:cluster"),
                    "optimizer": getattr(fit, "_codex_optimizer", "unknown"),
                    "converged": bool(getattr(fit, "converged", False)),
                    "avg_group_size": fit_summary_row["avg_group_size"],
                    "avg_fov_size": fit_summary_row["avg_fov_size"],
                    "warning_count": len(warning_messages),
                    "random_structure": random_structure,
                }
            )
            term_rows.append(term_df)

            within_rows.append(
                within_cluster_contrasts(
                    fit,
                    model_df,
                    response_var=response_col,
                    config=config,
                    fit_summary_row=fit_summary_row,
                )
            )
            emm_rows.append(
                build_emm_grid(
                    fit,
                    model_df,
                    response_var=response_col,
                    fit_summary_row=fit_summary_row,
                    config=config,
                )
            )
        except Exception as exc:
            fit_summary_rows.append(
                {
                    "response_var": response_col,
                    "model_kind": "two_term",
                    "formula": formula,
                    "random_structure": f"{config.group_col} + {config.fov_col}",
                    "status": "failed",
                    "converged": np.nan,
                    "optimizer": "",
                    "warning_count": 0,
                    "warning_messages": "",
                    "error": str(exc),
                    "data_csv": input_source_label(config),
                    "n_rows": np.nan,
                    "n_groups": np.nan,
                    "n_fovs": np.nan,
                    "n_group_treatment_combos": np.nan,
                    "avg_group_size": np.nan,
                    "median_group_size": np.nan,
                    "avg_group_treatment_size": np.nan,
                    "median_group_treatment_size": np.nan,
                    "avg_fov_size": np.nan,
                    "median_fov_size": np.nan,
                    "avg_fovs_per_group": np.nan,
                    "median_fovs_per_group": np.nan,
                    "n_biological_replicates": np.nan,
                    "avg_cells_per_biological_replicate_total": np.nan,
                    "median_cells_per_biological_replicate_total": np.nan,
                    "avg_cells_per_biological_replicate_treatment": np.nan,
                    "median_cells_per_biological_replicate_treatment": np.nan,
                    "avg_cells_per_fov": np.nan,
                    "median_cells_per_fov": np.nan,
                    "avg_fovs_per_biological_replicate": np.nan,
                    "median_fovs_per_biological_replicate": np.nan,
                }
            )
            print(f"[warn] two-term LMM failed for {response_col}: {exc}")

    term_df = pd.concat(term_rows, ignore_index=True) if term_rows else pd.DataFrame()
    if not term_df.empty:
        term_df = append_term_fdr_columns(term_df)
        write_csv_with_lock_fallback(term_df, out_stats_dir / "two_term_lmm_term_effects.csv")
        for response_var, response_term_df in term_df.groupby("response_var", sort=False):
            write_csv_with_lock_fallback(
                response_term_df,
                out_stats_dir / f"two_term_lmm_term_zscores_{response_var}.csv",
            )

    emm_df = pd.concat(emm_rows, ignore_index=True) if emm_rows else pd.DataFrame()
    if not emm_df.empty:
        write_csv_with_lock_fallback(emm_df, out_stats_dir / "two_term_lmm_emms.csv")

    within_df = pd.concat(within_rows, ignore_index=True) if within_rows else pd.DataFrame()
    if not within_df.empty:
        within_df = append_fdr_columns(
            within_df,
            p_value_col="within_p_value",
            output_col="within_fdr_p_value",
            stars_output_col="within_fdr_p_stars",
        )
        write_csv_with_lock_fallback(within_df, out_stats_dir / "two_term_lmm_within_cluster_contrasts.csv")

    fit_summary_df = pd.DataFrame(fit_summary_rows)
    write_csv_with_lock_fallback(fit_summary_df, out_stats_dir / "two_term_lmm_model_fit_summary.csv")
    return term_df, emm_df, within_df, fit_summary_df


def run_cluster_chi_square(
    df: pd.DataFrame,
    *,
    out_stats_dir: Path,
    config: ClusteredAnalysisConfig,
) -> dict[str, pd.DataFrame]:
    subset = df.dropna(subset=[config.treatment_col, config.cluster_col]).copy()
    subset[config.treatment_col] = subset[config.treatment_col].astype(str)
    subset[config.cluster_col] = subset[config.cluster_col].astype(str)

    present_treatments = [
        treatment
        for treatment in config.treatment_order
        if treatment in subset[config.treatment_col].unique().tolist()
    ]
    if config.treatment_reference not in present_treatments:
        raise RuntimeError(
            f"Reference treatment '{config.treatment_reference}' was not found for chi-square comparisons."
        )

    subset = subset[subset[config.treatment_col].isin(present_treatments)].copy()
    cluster_levels = sorted(subset[config.cluster_col].unique().tolist(), key=sort_key)

    composition_df = (
        subset.groupby([config.treatment_col, config.cluster_col], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    composition_df["proportion"] = composition_df.groupby(config.treatment_col, observed=True)["n_cells"].transform(
        lambda values: values / values.sum()
    )
    composition_df.to_csv(out_stats_dir / "cluster_composition_by_treatment.csv", index=False)

    global_table = pd.crosstab(subset[config.treatment_col], subset[config.cluster_col]).reindex(
        index=present_treatments,
        columns=cluster_levels,
        fill_value=0,
    )
    chi2, p_value, dof, expected = chi2_contingency(global_table, correction=False)
    global_df = pd.DataFrame(
        [
            {
                "comparison": "all_treatments",
                "reference_treatment": config.treatment_reference,
                "chi2": float(chi2),
                "cramers_v": cramers_v_from_table(global_table, chi2),
                "p_value": float(p_value),
                "p_stars": p_to_stars(float(p_value)),
                "dof": int(dof),
                "n_treatments": int(global_table.shape[0]),
                "n_clusters": int(global_table.shape[1]),
                "total_cells": int(global_table.to_numpy().sum()),
                "min_expected_count": float(np.min(expected)),
            }
        ]
    )
    global_df.to_csv(out_stats_dir / "cluster_composition_global_chi_square.csv", index=False)

    pairwise_rows: list[dict[str, object]] = []
    per_cluster_rows: list[dict[str, object]] = []
    for treatment in present_treatments:
        if treatment == config.treatment_reference:
            continue

        pair = subset[subset[config.treatment_col].isin([config.treatment_reference, treatment])].copy()
        contingency = pd.crosstab(pair[config.treatment_col], pair[config.cluster_col]).reindex(
            index=[config.treatment_reference, treatment],
            columns=cluster_levels,
            fill_value=0,
        )
        chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)
        pairwise_rows.append(
            {
                "treatment": treatment,
                "reference_treatment": config.treatment_reference,
                "chi2": float(chi2),
                "cramers_v": cramers_v_from_table(contingency, chi2),
                "p_value": float(p_value),
                "p_stars": p_to_stars(float(p_value)),
                "dof": int(dof),
                "n_clusters": int(contingency.shape[1]),
                "vehicle_n_cells": int(contingency.loc[config.treatment_reference].sum()),
                "treatment_n_cells": int(contingency.loc[treatment].sum()),
                "total_cells": int(contingency.to_numpy().sum()),
                "min_expected_count": float(np.min(expected)),
            }
        )

        for cluster in cluster_levels:
            vehicle_in_cluster = int(contingency.loc[config.treatment_reference, cluster])
            treatment_in_cluster = int(contingency.loc[treatment, cluster])
            vehicle_out_cluster = int(contingency.loc[config.treatment_reference].sum() - vehicle_in_cluster)
            treatment_out_cluster = int(contingency.loc[treatment].sum() - treatment_in_cluster)
            table_2x2 = np.array(
                [
                    [vehicle_in_cluster, vehicle_out_cluster],
                    [treatment_in_cluster, treatment_out_cluster],
                ],
                dtype=float,
            )
            chi2_cl, p_value_cl, dof_cl, expected_cl = chi2_contingency(table_2x2, correction=False)
            vehicle_total = contingency.loc[config.treatment_reference].sum()
            treatment_total = contingency.loc[treatment].sum()

            per_cluster_rows.append(
                {
                    "treatment": treatment,
                    "reference_treatment": config.treatment_reference,
                    "cluster": cluster,
                    "chi2": float(chi2_cl),
                    "cramers_v": cramers_v_from_table(table_2x2, chi2_cl),
                    "p_value": float(p_value_cl),
                    "p_stars": p_to_stars(float(p_value_cl)),
                    "dof": int(dof_cl),
                    "vehicle_in_cluster": vehicle_in_cluster,
                    "vehicle_out_cluster": vehicle_out_cluster,
                    "treatment_in_cluster": treatment_in_cluster,
                    "treatment_out_cluster": treatment_out_cluster,
                    "vehicle_prop": float(vehicle_in_cluster / vehicle_total) if vehicle_total else np.nan,
                    "treatment_prop": float(treatment_in_cluster / treatment_total) if treatment_total else np.nan,
                    "prop_diff_treatment_minus_vehicle": (
                        float(treatment_in_cluster / treatment_total) - float(vehicle_in_cluster / vehicle_total)
                        if vehicle_total and treatment_total
                        else np.nan
                    ),
                    "min_expected_count": float(np.min(expected_cl)),
                }
            )

    pairwise_df = append_fdr_columns(pd.DataFrame(pairwise_rows))
    pairwise_df.to_csv(out_stats_dir / "cluster_composition_pairwise_chi_square_vs_vehicle.csv", index=False)

    per_cluster_df = append_fdr_columns(pd.DataFrame(per_cluster_rows))
    per_cluster_df.to_csv(
        out_stats_dir / "cluster_composition_pairwise_per_cluster_chi_square_vs_vehicle.csv",
        index=False,
    )

    return {
        "composition": composition_df,
        "global": global_df,
        "pairwise": pairwise_df,
        "per_cluster": per_cluster_df,
    }


def build_cluster_composition_weighted_fraction_table(
    df: pd.DataFrame,
    *,
    config: ClusteredAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    subset = df.dropna(subset=[config.treatment_col, config.cluster_col, config.group_col]).copy()
    subset[config.treatment_col] = subset[config.treatment_col].astype(str)
    subset[config.cluster_col] = subset[config.cluster_col].astype(str)
    subset[config.group_col] = subset[config.group_col].astype(str)

    present_treatments = [
        treatment
        for treatment in config.treatment_order
        if treatment in subset[config.treatment_col].unique().tolist()
    ]
    if config.treatment_reference not in present_treatments:
        raise RuntimeError(
            f"Reference treatment '{config.treatment_reference}' was not found for cluster-composition RM ANOVA."
        )

    subset = subset[subset[config.treatment_col].isin(present_treatments)].copy()

    expected_levels = [str(level) for level in expected_heatmap_cluster_order(config)]
    observed_levels = sorted(subset[config.cluster_col].unique().tolist(), key=sort_key)
    cluster_levels = [level for level in expected_levels if level in observed_levels]
    cluster_levels += [level for level in observed_levels if level not in cluster_levels]

    composition_df = (
        subset.groupby([config.treatment_col, config.cluster_col], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    composition_df["proportion"] = composition_df.groupby(config.treatment_col, observed=True)["n_cells"].transform(
        lambda values: values / values.sum()
    )
    composition_df["cluster_label"] = composition_df[config.cluster_col].map(
        lambda value: cluster_label_for_value(value, config)
    )

    group_cols = [config.group_col, config.treatment_col]
    if "N_label" in subset.columns:
        group_cols.append("N_label")
    total_counts = (
        subset.groupby(group_cols, observed=True)
        .size()
        .rename("n_cells_total")
        .reset_index()
    )
    cluster_counts = (
        subset.groupby(group_cols + [config.cluster_col], observed=True)
        .size()
        .rename("n_cells_cluster")
        .reset_index()
    )
    cluster_frame = pd.DataFrame({config.cluster_col: cluster_levels})
    replicate_df = total_counts.merge(cluster_frame, how="cross")
    replicate_df = replicate_df.merge(
        cluster_counts,
        on=group_cols + [config.cluster_col],
        how="left",
        validate="one_to_one",
    )
    replicate_df["n_cells_cluster"] = replicate_df["n_cells_cluster"].fillna(0).astype(int)
    replicate_df["weighted_fraction"] = np.where(
        replicate_df["n_cells_total"] > 0,
        replicate_df["n_cells_cluster"] / replicate_df["n_cells_total"],
        np.nan,
    )
    replicate_df["cluster_label"] = replicate_df[config.cluster_col].map(
        lambda value: cluster_label_for_value(value, config)
    )
    return composition_df, replicate_df, present_treatments, cluster_levels


def run_cluster_rm_anova_weighted_fraction(
    df: pd.DataFrame,
    *,
    out_stats_dir: Path,
    config: ClusteredAnalysisConfig,
) -> dict[str, pd.DataFrame]:
    composition_df, replicate_df, present_treatments, cluster_levels = build_cluster_composition_weighted_fraction_table(
        df,
        config=config,
    )
    composition_df.to_csv(out_stats_dir / "cluster_composition_by_treatment.csv", index=False)
    replicate_df.to_csv(
        out_stats_dir / "cluster_composition_weighted_fraction_by_biological_replicate.csv",
        index=False,
    )

    global_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    all_subjects = sorted(replicate_df[config.group_col].dropna().astype(str).unique().tolist(), key=sort_key)
    target_labels = cluster_composition_posthoc_target_labels(config)

    for cluster in cluster_levels:
        cluster_label = cluster_label_for_value(cluster, config)
        cluster_df = replicate_df[replicate_df[config.cluster_col] == cluster].copy()
        wide = (
            cluster_df.pivot(
                index=config.group_col,
                columns=config.treatment_col,
                values="weighted_fraction",
            )
            .reindex(columns=present_treatments)
        )
        complete_subjects = [str(value) for value in wide.dropna().index.tolist()]
        excluded_subjects = [value for value in all_subjects if value not in complete_subjects]

        analysis_input = cluster_df[cluster_df[config.group_col].isin(complete_subjects)].copy()
        if len(complete_subjects) >= 2 and len(present_treatments) >= 2:
            try:
                anova = AnovaRM(
                    data=analysis_input,
                    depvar="weighted_fraction",
                    subject=config.group_col,
                    within=[config.treatment_col],
                ).fit()
                anova_table = (
                    anova.anova_table
                    .reset_index()
                    .rename(
                        columns={
                            "index": "effect",
                            "F Value": "F",
                            "Num DF": "num_df",
                            "Den DF": "den_df",
                            "Pr > F": "p_value",
                        }
                    )
                )
                anova_row = anova_table.iloc[0].to_dict()
                status = "ok"
                error = ""
            except Exception as exc:
                anova_row = {
                    "effect": config.treatment_col,
                    "F": np.nan,
                    "num_df": np.nan,
                    "den_df": np.nan,
                    "p_value": np.nan,
                }
                status = "failed"
                error = str(exc)
        else:
            anova_row = {
                "effect": config.treatment_col,
                "F": np.nan,
                "num_df": np.nan,
                "den_df": np.nan,
                "p_value": np.nan,
            }
            status = "insufficient_data"
            error = "Need at least 2 complete biological replicates and 2 treatments."

        global_rows.append(
            {
                "metric": "cluster_composition_weighted_fraction",
                "cluster": cluster,
                "cluster_label": cluster_label,
                **anova_row,
                "test": "repeated_measures_anova",
                "analysis_level": "biological_replicate_weighted_fraction",
                "reference_treatment": config.treatment_reference,
                "n_biological_replicates": len(complete_subjects),
                "n_treatments": len(present_treatments),
                "complete_biological_replicates": "|".join(complete_subjects),
                "excluded_biological_replicates": "|".join(excluded_subjects),
                "status": status,
                "error": error,
            }
        )

        if config.treatment_reference not in present_treatments or not complete_subjects:
            continue
        if target_labels is not None and cluster_label not in target_labels:
            continue
        wide_complete = wide.loc[complete_subjects]
        for treatment in present_treatments:
            if treatment == config.treatment_reference:
                continue
            paired = wide_complete[[config.treatment_reference, treatment]].dropna()
            n_pairs = int(len(paired))
            vehicle_mean = float(paired[config.treatment_reference].mean()) if n_pairs else np.nan
            treatment_mean = float(paired[treatment].mean()) if n_pairs else np.nan

            if n_pairs >= 2:
                diffs = paired[treatment] - paired[config.treatment_reference]
                t_stat, p_value = ttest_rel(paired[treatment], paired[config.treatment_reference])
                mean_diff = float(diffs.mean())
                sd_diff = float(diffs.std(ddof=1))
                se_diff = sd_diff / np.sqrt(n_pairs) if np.isfinite(sd_diff) else np.nan
                t_crit = float(t.ppf(0.975, df=n_pairs - 1))
                ci_low = mean_diff - t_crit * se_diff if np.isfinite(se_diff) else np.nan
                ci_high = mean_diff + t_crit * se_diff if np.isfinite(se_diff) else np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
                mean_diff = np.nan
                ci_low = np.nan
                ci_high = np.nan

            pairwise_rows.append(
                {
                    "metric": "cluster_composition_weighted_fraction",
                    "cluster": cluster,
                    "cluster_label": cluster_label,
                    "test": "paired_t_test",
                    "analysis_level": "biological_replicate_weighted_fraction",
                    "posthoc_scope": config.cluster_composition_posthoc_scope,
                    "reference_treatment": config.treatment_reference,
                    "treatment": treatment,
                    "n_biol_rep": n_pairs,
                    "vehicle_mean_fraction": vehicle_mean,
                    "treatment_mean_fraction": treatment_mean,
                    "mean_diff_treatment_minus_vehicle": mean_diff,
                    "diff_ci_low": ci_low,
                    "diff_ci_high": ci_high,
                    "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(out_stats_dir / "cluster_composition_global_repeated_measures_tests.csv", index=False)

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["p_stars"] = pairwise_df["p_value"].map(p_to_stars)
        pairwise_df = append_fdr_columns(pairwise_df)
        if config.cluster_composition_posthoc_scope == "planned_mbp_high_vs_vehicle":
            pairwise_df.sort_values(["cluster", "p_value", "treatment"], inplace=True, na_position="last")
        else:
            pairwise_df.sort_values(["cluster", "fdr_p_value", "treatment"], inplace=True, na_position="last")
        pairwise_df.reset_index(drop=True, inplace=True)
    pairwise_df.to_csv(
        out_stats_dir / "cluster_composition_weighted_fraction_pairwise_vs_vehicle.csv",
        index=False,
    )

    return {
        "composition": composition_df,
        "global": global_df,
        "pairwise": pairwise_df,
        "per_cluster": pairwise_df,
    }


def write_lmm_supplementary_summary(
    one_term_fit_df: pd.DataFrame,
    two_term_fit_df: pd.DataFrame,
    *,
    out_stats_dir: Path,
    config: ClusteredAnalysisConfig,
) -> pd.DataFrame:
    frames = [df.copy() for df in [one_term_fit_df, two_term_fit_df] if df is not None and not df.empty]
    if not frames:
        empty = pd.DataFrame()
        empty.to_csv(out_stats_dir / "lmm_supplementary_model_observation_summary.csv", index=False)
        return empty

    combined = pd.concat(frames, ignore_index=True)
    combined["chapter"] = config.chapter_label
    combined["dataset"] = config.dataset_label
    combined["analysis_name"] = config.analysis_name

    rename_map = {
        "n_rows": "n_cells_used",
        "n_biological_replicates": "n_biol_reps",
        "n_fovs": "n_fovs_used",
        "n_group_treatment_combos": "n_biol_rep_treatment_groups",
        "avg_cells_per_biological_replicate_total": "avg_cells_per_biol_rep_total",
        "median_cells_per_biological_replicate_total": "median_cells_per_biol_rep_total",
        "avg_cells_per_biological_replicate_treatment": "avg_cells_per_biol_rep_treatment_group",
        "median_cells_per_biological_replicate_treatment": "median_cells_per_biol_rep_treatment_group",
        "avg_cells_per_fov": "avg_cells_per_fov",
        "median_cells_per_fov": "median_cells_per_fov",
        "avg_fovs_per_biological_replicate": "avg_fovs_per_biol_rep",
        "median_fovs_per_biological_replicate": "median_fovs_per_biol_rep",
    }
    for source_col, target_col in rename_map.items():
        if source_col in combined.columns and target_col not in combined.columns:
            combined[target_col] = combined[source_col]

    ordered_cols = [
        "chapter",
        "dataset",
        "analysis_name",
        "model_kind",
        "response_var",
        "formula",
        "random_structure",
        "status",
        "converged",
        "optimizer",
        "warning_count",
        "warning_messages",
        "error",
        "data_csv",
        "n_cells_used",
        "n_biol_reps",
        "n_fovs_used",
        "n_biol_rep_treatment_groups",
        "avg_cells_per_biol_rep_total",
        "median_cells_per_biol_rep_total",
        "avg_cells_per_biol_rep_treatment_group",
        "median_cells_per_biol_rep_treatment_group",
        "avg_cells_per_fov",
        "median_cells_per_fov",
        "avg_fovs_per_biol_rep",
        "median_fovs_per_biol_rep",
    ]
    remaining_cols = [col for col in combined.columns if col not in ordered_cols]
    combined = combined[[col for col in ordered_cols if col in combined.columns] + remaining_cols].copy()
    combined.sort_values(["model_kind", "response_var"], inplace=True, na_position="last")
    combined.reset_index(drop=True, inplace=True)
    combined.to_csv(out_stats_dir / "lmm_supplementary_model_observation_summary.csv", index=False)
    return combined


def run_clustered_analysis(config: ClusteredAnalysisConfig) -> dict[str, object]:
    ensure_output_dirs(config)
    input_df = load_analysis_table(config)
    norm_long = build_norm_long(input_df, config)
    if config.cluster_col not in input_df.columns:
        working_df = run_clustering_from_norm_long(norm_long, input_df, config)
    else:
        working_df = input_df

    analysis_df, norm_wide = write_pipeline_tables(working_df, norm_long, config)
    cluster_df, norm_cols = prepare_cluster_plot_table(norm_wide, config)
    plot_cluster_selection_diagnostics(cluster_df, norm_cols, config)
    plot_cluster_scatter_3d(cluster_df, norm_cols, config)
    cluster_composition_plot_df = plot_cluster_composition(analysis_df, config)
    plot_legacy_marker_mfi_violins(norm_long, config=config)
    plot_two_term_response_distributions(
        analysis_df,
        response_cols=[column for column in config.two_term_response_cols if column in analysis_df.columns],
        config=config,
    )
    cell_count_results = write_cell_count_outputs(analysis_df, config)

    one_term_responses = resolve_response_cols(analysis_df, config.one_term_response_cols)
    two_term_responses = resolve_response_cols(analysis_df, config.two_term_response_cols)
    if not one_term_responses:
        raise RuntimeError("No configured one-term responses were found in the clustered table.")
    if not two_term_responses:
        raise RuntimeError("No configured two-term responses were found in the clustered table.")

    one_term_df, one_term_terms_df, one_term_fit_df = run_one_term_lmm(
        analysis_df,
        response_cols=one_term_responses,
        out_stats_dir=config.stats_dir,
        config=config,
    )
    two_term_terms_df, two_term_emm_df, two_term_within_df, two_term_fit_df = run_two_term_lmm(
        analysis_df,
        response_cols=two_term_responses,
        out_stats_dir=config.stats_dir,
        config=config,
    )
    lmm_supplementary_df = write_lmm_supplementary_summary(
        one_term_fit_df,
        two_term_fit_df,
        out_stats_dir=config.stats_dir,
        config=config,
    )
    within_effect_heatmap = write_two_term_within_effect_heatmap(
        two_term_within_df,
        out_dir=config.graphs_dir,
        response_cols=two_term_responses,
        config=config,
    )
    if config.cluster_composition_inference == "chi_square":
        cluster_inference_results = run_cluster_chi_square(
            analysis_df,
            out_stats_dir=config.stats_dir,
            config=config,
        )
    elif config.cluster_composition_inference == "rm_anova_weighted_fraction":
        cluster_inference_results = run_cluster_rm_anova_weighted_fraction(
            analysis_df,
            out_stats_dir=config.stats_dir,
            config=config,
        )
    else:
        raise RuntimeError(
            f"Unsupported cluster_composition_inference: {config.cluster_composition_inference}"
        )

    return {
        "config": config,
        "data_csv": input_source_label(config),
        "analysis_dir": config.analysis_dir,
        "stats_dir": config.stats_dir,
        "graphs_dir": config.graphs_dir,
        "clustering_dir": config.clustering_dir,
        "df": analysis_df,
        "norm_long": norm_long,
        "cluster_plot_df": cluster_df,
        "cluster_composition_plot": cluster_composition_plot_df,
        "cell_count_df": cell_count_results["per_replicate"],
        "cell_count_per_fov": cell_count_results["per_fov"],
        "cell_count_replicate": cell_count_results["per_replicate"],
        "cell_count_analysis_input": cell_count_results["analysis_input"],
        "cell_count_rm_anova": cell_count_results["rm_anova"],
        "cell_count_pairwise": cell_count_results["pairwise"],
        "one_term_results": one_term_df,
        "one_term_term_effects": one_term_terms_df,
        "one_term_fit_summary": one_term_fit_df,
        "two_term_term_effects": two_term_terms_df,
        "two_term_emms": two_term_emm_df,
        "two_term_within_cluster": two_term_within_df,
        "two_term_fit_summary": two_term_fit_df,
        "within_effect_heatmap": within_effect_heatmap,
        "lmm_supplementary_summary": lmm_supplementary_df,
        "cluster_inference_composition": cluster_inference_results["composition"],
        "cluster_inference_global": cluster_inference_results["global"],
        "cluster_inference_pairwise": cluster_inference_results["pairwise"],
        "cluster_inference_per_cluster": cluster_inference_results["per_cluster"],
        # The results dictionary keeps the chi_square_* slots empty so the
        # output schema stays consistent across analysis bundles. The active
        # cluster-composition outputs for this path are cluster_inference_*
        # and cluster_composition_rm_anova*.
        "chi_square_composition": (
            cluster_inference_results["composition"]
            if config.cluster_composition_inference == "chi_square"
            else pd.DataFrame()
        ),
        "chi_square_global": (
            cluster_inference_results["global"]
            if config.cluster_composition_inference == "chi_square"
            else pd.DataFrame()
        ),
        "chi_square_pairwise": (
            cluster_inference_results["pairwise"]
            if config.cluster_composition_inference == "chi_square"
            else pd.DataFrame()
        ),
        "chi_square_per_cluster": (
            cluster_inference_results["per_cluster"]
            if config.cluster_composition_inference == "chi_square"
            else pd.DataFrame()
        ),
        "cluster_composition_rm_anova": (
            cluster_inference_results["global"]
            if config.cluster_composition_inference == "rm_anova_weighted_fraction"
            else pd.DataFrame()
        ),
        "cluster_composition_rm_anova_pairwise": (
            cluster_inference_results["pairwise"]
            if config.cluster_composition_inference == "rm_anova_weighted_fraction"
            else pd.DataFrame()
        ),
    }


import argparse

import matplotlib

matplotlib.use("Agg")

from scipy import stats
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "original_data" / "calcium" / "out_registration_batch"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "calcium_full_output_V1"

# This script is intentionally self-contained.
# It rebuilds the calcium analysis bundle used for Chapter 4 Figure 3 and
# Table 2 directly from `original_data/calcium/out_registration_batch`.
# The whole workflow stays in one file: loading, clustering, AUC cleaning,
# figure generation, mixed-model fitting, residual checks, and the final
# artifact summary.

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
    # 1. Resolve the uploaded raw-data bundle location and create the output folders.
    master_dir = resolve_data_dir(master_dir)
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir = output_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data directory: {master_dir}")
    print(f"Writing outputs to: {output_dir}")

    # 2. Concatenate the raw matched-cell tables, recover replicate/treatment metadata,
    #    and rebuild the stain-based clustering tables.
    df_all = prepare_combined_matched_cells(master_dir, output_dir)
    df1_present = cluster_cells(df_all, output_dir, fig_dir, clustering_dir)

    # 3. Build the cleaned AUC analysis table used by the downstream nested MixedLMs.
    df2_present, df2_auc_clean = build_auc_clean_table(df1_present)

    # 4. Generate the trace, AUC, and clustering figures used for the Chapter 4
    #    calcium results and the accompanying QC views.
    plot_rt_scatter_per_treatment(df2_present, fig_dir)
    plot_auc_violin_per_treatment(df2_auc_clean, fig_dir)
    plot_auc_rt_grid(df2_present, df2_auc_clean, fig_dir)

    # 5. Run the nested raw-AUC MixedLM analysis and the model diagnostics.
    model_paths = run_auc_models(df2_auc_clean, output_dir)
    plot_average_traces(master_dir, df1_present, fig_dir)
    run_residual_diagnostics(df2_auc_clean, output_dir, fig_dir)
    run_transform_sweep(df2_auc_clean, output_dir, fig_dir)

    # 6. Write the artifact manifest last so the output folder itself documents what
    #    was created and which intermediate table each figure/report came from.
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

