from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ORIGINAL_DATA_DIR = ROOT / "original_data"
PARSED_DATA_DIR = ROOT / "parsed_data"

RX_0929_PREFIX = re.compile(r"^(?!N4_0929)0929(?=[_ ]|$)", flags=re.IGNORECASE)
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
CASPASE_TREATMENT_ORDER = ["vehicle", "MDL29951", "HAMI3379", "pranlukast", "RWT9996", "clemastine"]
CASPASE_PHENOTYPE_ORDER = ["PDGFRa", "O4", "MBP/O4"]
CASPASE_SUMMARY_PHENOTYPE_ORDER = ["PDGFRa", "O4", "MBP/O4", "Marker Low"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def combine_csv_dir(csv_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(path for path in csv_dir.glob("*.csv") if "partial" not in path.name.lower())
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")
    df = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True, axis=0)
    if "file" not in df.columns and "base_name" in df.columns:
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
    for column in ("lif_name", "base_name", "file"):
        if column in out.columns:
            out[column] = out[column].map(_fix_0929_prefix)
    if "lif_path" in out.columns:
        out["lif_path"] = out["lif_path"].astype(str).str.replace(
            r"(?i)(?<!N4_)0929(?=_(?:PKM2|pPKM2)\.lif\b)",
            "N4_0929",
            regex=True,
        )
    return out


def build_chapter_3_parsed() -> dict[str, object]:
    raw_dir = ORIGINAL_DATA_DIR / "chapter_3" / "csvs"
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Chapter 3 CSVs found in {raw_dir}")
    df = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True, axis=0)
    df = harmonise_convexity_columns(df)
    if "file" not in df.columns and "base_name" in df.columns:
        df["file"] = df["base_name"].astype(str)
    out_path = PARSED_DATA_DIR / "chapter_3" / "per_cell_stats_all.csv"
    ensure_parent(out_path)
    df.to_csv(out_path, index=False)
    return {
        "dataset": "chapter_3",
        "source": str(raw_dir),
        "output": str(out_path),
        "n_source_csvs": len(csv_paths),
        "n_rows": len(df),
    }


def build_chapter_4_parsed(dataset: str) -> dict[str, object]:
    raw_dir = ORIGINAL_DATA_DIR / dataset / "csvs"
    df = combine_csv_dir(raw_dir)
    df = fix_0929_names(df)
    out_path = PARSED_DATA_DIR / dataset / "per_cell_stats_all.csv"
    ensure_parent(out_path)
    df.to_csv(out_path, index=False)
    return {
        "dataset": dataset,
        "source": str(raw_dir),
        "output": str(out_path),
        "n_source_csvs": len(sorted(path for path in raw_dir.glob("*.csv") if "partial" not in path.name.lower())),
        "n_rows": len(df),
    }


def rebuild_caspase_legacy_filtered_cells(raw_dir: Path) -> pd.DataFrame:
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
    df["caspase_pos"] = pd.to_numeric(df["caspase_area_px"], errors="coerce") > 50
    df["caspase_pos_bin"] = df["caspase_pos"].astype(int)

    for marker in ["pdgfra", "o4", "mbp"]:
        df[f"{marker}_frac"] = pd.to_numeric(df[f"{marker}_area_px"], errors="coerce") / pd.to_numeric(
            df["cell_area_px"],
            errors="coerce",
        )

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
    df = df[df["treatment"].isin(CASPASE_TREATMENT_ORDER) & df["phenotype"].isin(CASPASE_SUMMARY_PHENOTYPE_ORDER)].copy()
    df["N_well"] = df["N"].astype(str) + "_W" + df["well"].astype(str)
    df["well_id"] = df["N_well"] + "_" + df["treatment"].astype(str)
    return df


def build_caspase_parsed() -> dict[str, object]:
    raw_dir = ORIGINAL_DATA_DIR / "caspase" / "csvs"
    filtered = rebuild_caspase_legacy_filtered_cells(raw_dir)
    filtered_path = PARSED_DATA_DIR / "caspase" / "caspase_cell_data_legacy_filtered.csv"
    ensure_parent(filtered_path)
    filtered.to_csv(filtered_path, index=False)

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
        categories=CASPASE_TREATMENT_ORDER,
        ordered=True,
    )
    summary["phenotype"] = pd.Categorical(
        summary["phenotype"],
        categories=CASPASE_SUMMARY_PHENOTYPE_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["phenotype", "treatment", "biol_rep"]).reset_index(drop=True)

    summary_path = PARSED_DATA_DIR / "caspase" / "caspase_summary_table.csv"
    ensure_parent(summary_path)
    summary.to_csv(summary_path, index=False)
    return {
        "dataset": "caspase",
        "source": str(raw_dir),
        "output": str(summary_path),
        "filtered_cells_output": str(filtered_path),
        "n_source_csvs": len(list(raw_dir.glob("*.csv"))),
        "n_filtered_rows": len(filtered),
        "n_summary_rows": len(summary),
    }


def main() -> None:
    PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        build_chapter_3_parsed(),
        build_chapter_4_parsed("pkm2"),
        build_chapter_4_parsed("ppkm2"),
        build_caspase_parsed(),
        {
            "dataset": "calcium",
            "source": str(ORIGINAL_DATA_DIR / "calcium" / "out_registration_batch"),
            "output": "run_calcium_github.py writes its own parsed/clustered tables directly into outputs/calcium_full_output_V1",
            "n_source_csvs": len(list((ORIGINAL_DATA_DIR / "calcium" / "out_registration_batch").rglob("matched_cells_with_stain_metrics.csv"))),
            "n_rows": "n/a",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_path = PARSED_DATA_DIR / "parsed_data_build_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved parsed-data summary to {summary_path}")


if __name__ == "__main__":
    main()
