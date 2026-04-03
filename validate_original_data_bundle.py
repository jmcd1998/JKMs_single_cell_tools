from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ORIGINAL_DATA_DIR = ROOT / "original_data"
EXAMPLE_DATA_DIR = ROOT / "example_data"
VALIDATION_DIR = ORIGINAL_DATA_DIR / "validation"


def harmonise_convexity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_conv = "convexity_ratio" in out.columns
    has_area = "area_ratio_union" in out.columns
    if has_conv and not has_area:
        out["area_ratio_union"] = out["convexity_ratio"]
    elif has_area and not has_conv:
        out["convexity_ratio"] = out["area_ratio_union"]
    elif has_conv and has_area:
        out["convexity_ratio"] = pd.to_numeric(
            out["convexity_ratio"], errors="coerce"
        ).fillna(pd.to_numeric(out["area_ratio_union"], errors="coerce"))
    return out


def hash_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_json(value: bool) -> str:
    return json.dumps(bool(value))


def grouped_numeric_summary(
    df: pd.DataFrame, group_col: str, numeric_cols: list[str]
) -> pd.DataFrame:
    out = df.groupby(group_col)[numeric_cols].agg(["count", "sum", "mean"]).sort_index()
    return out


def aligned_allclose(left: pd.DataFrame, right: pd.DataFrame, atol: float = 1e-10) -> bool:
    if not left.index.equals(right.index):
        return False
    if not left.columns.equals(right.columns):
        return False
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    return bool(np.allclose(left_values, right_values, rtol=0.0, atol=atol, equal_nan=True))


def compare_file_hash_sets(left_dir: Path, right_dir: Path) -> tuple[bool, str]:
    left_files = sorted(p.name for p in left_dir.glob("*.csv"))
    right_files = sorted(p.name for p in right_dir.glob("*.csv"))
    if left_files != right_files:
        return False, f"filename mismatch: {left_files} vs {right_files}"
    mismatched = []
    for name in left_files:
        left_hash = hash_file(left_dir / name)
        right_hash = hash_file(right_dir / name)
        if left_hash != right_hash:
            mismatched.append(name)
    if mismatched:
        return False, f"hash mismatch in {mismatched}"
    return True, f"{len(left_files)} files matched by filename and md5"


def rebuild_caspase_legacy_filtered_cells(raw_dir: Path) -> pd.DataFrame:
    csvs = sorted(raw_dir.glob("*.csv"))
    df = pd.concat([pd.read_csv(path) for path in csvs], ignore_index=True)

    treatment_pattern = re.compile(
        r"(vehicle|pranlukast|mdl29951|hami3379|rwt9996|clemastine|h202)",
        re.IGNORECASE,
    )
    mapping = {
        "vehicle": "vehicle",
        "pranlukast": "pranlukast",
        "mdl29951": "MDL29951",
        "hami3379": "HAMI3379",
        "rwt9996": "RWT9996",
        "clemastine": "clemastine",
        "h202": "H2O2",
    }
    phenotype_order = ["PDGFRa", "O4", "MBP/O4"]
    treatment_order = ["vehicle", "MDL29951", "pranlukast", "HAMI3379", "RWT9996", "clemastine"]

    df["treatment_raw"] = (
        df["base_name"].astype(str).str.extract(treatment_pattern, expand=False).str.lower()
    )
    df["treatment"] = df["treatment_raw"].map(mapping).fillna("vehicle")
    df["N"] = df["base_name"].astype(str).str.extract(r"(N\d+)", expand=False)
    df["well"] = df["base_name"].astype(str).str.extract(r"_(\d+)_R\d+", expand=False)
    df["fov"] = df["base_name"]
    df["caspase_pos"] = pd.to_numeric(df["caspase_area_px"], errors="coerce") > 50
    df["caspase_pos_bin"] = df["caspase_pos"].astype(int)

    for marker in ["pdgfra", "o4", "mbp"]:
        df[f"{marker}_frac"] = pd.to_numeric(
            df[f"{marker}_area_px"], errors="coerce"
        ) / pd.to_numeric(df["cell_area_px"], errors="coerce")

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
        scores["PDGFRa"] = (
            row["pdgfra_frac"] + 0.3 * row["size_prior_pdgfra"] if pdg_ok else 0.0
        )
        scores["O4"] = row["o4_frac"] if o4_ok else 0.0
        scores["MBP/O4"] = row["mbp_frac"] if mbp_ok else 0.0
        if max(scores.values()) <= 0:
            return "Marker Low"
        return max(scores, key=scores.get)

    df["phenotype"] = df.apply(phenotype_with_size, axis=1)
    df = df[
        df["treatment"].isin(treatment_order) & df["phenotype"].isin(phenotype_order)
    ].copy()
    df["N_well"] = df["N"].astype(str) + "_W" + df["well"].astype(str)
    df["well_id"] = df["N_well"] + "_" + df["treatment"].astype(str)
    return df


def normalise_caspase_phenotypes(df: pd.DataFrame, column: str = "phenotype") -> pd.DataFrame:
    out = df.copy()
    if column in out.columns:
        out[column] = out[column].astype(str).replace(
            {
                "pdgfra like": "PDGFRa",
                "o4 like": "O4",
                "mbp like": "MBP/O4",
                "marker low": "Marker Low",
            }
        )
    return out


def build_validation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    chapter3_raw_dir = ORIGINAL_DATA_DIR / "chapter_3" / "csvs"
    chapter3_example = EXAMPLE_DATA_DIR / "chapter_3" / "per_cell_stats_all.csv"
    chapter3_raw = pd.concat(
        [pd.read_csv(path) for path in sorted(chapter3_raw_dir.glob("*.csv"))],
        ignore_index=True,
    )
    chapter3_raw = harmonise_convexity_columns(chapter3_raw)
    chapter3_example_df = pd.read_csv(chapter3_example)
    shared_numeric = [
        column
        for column in chapter3_raw.columns
        if column in chapter3_example_df.columns
        and (
            pd.api.types.is_numeric_dtype(chapter3_raw[column])
            or pd.api.types.is_numeric_dtype(chapter3_example_df[column])
        )
    ]
    for df in (chapter3_raw, chapter3_example_df):
        for column in shared_numeric:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    chapter3_count_match = chapter3_raw.groupby("base_name").size().sort_index().equals(
        chapter3_example_df.groupby("file").size().sort_index()
    )
    chapter3_agg_match = aligned_allclose(
        grouped_numeric_summary(chapter3_raw, "base_name", shared_numeric),
        grouped_numeric_summary(chapter3_example_df, "file", shared_numeric),
        atol=1e-10,
    )
    rows.extend(
        [
            {
                "dataset": "chapter_3",
                "check": "raw_csv_count",
                "status": "pass",
                "details": f"{len(list(chapter3_raw_dir.glob('*.csv')))} raw CSVs copied",
            },
            {
                "dataset": "chapter_3",
                "check": "row_count_match_vs_example_data",
                "status": "pass" if len(chapter3_raw) == len(chapter3_example_df) else "fail",
                "details": f"raw_rows={len(chapter3_raw)} example_rows={len(chapter3_example_df)}",
            },
            {
                "dataset": "chapter_3",
                "check": "per_image_cell_count_match_after_legacy_convexity_harmonization",
                "status": "pass" if chapter3_count_match else "fail",
                "details": bool_json(chapter3_count_match),
            },
            {
                "dataset": "chapter_3",
                "check": "per_image_numeric_aggregate_match_vs_example_data",
                "status": "pass" if chapter3_agg_match else "fail",
                "details": f"{len(shared_numeric)} shared numeric columns checked with atol=1e-10",
            },
        ]
    )

    for dataset in ["pkm2", "ppkm2"]:
        matched, detail = compare_file_hash_sets(
            ORIGINAL_DATA_DIR / dataset / "csvs",
            EXAMPLE_DATA_DIR / dataset / "csvs",
        )
        rows.append(
            {
                "dataset": dataset,
                "check": "raw_file_hash_match_vs_example_data",
                "status": "pass" if matched else "fail",
                "details": detail,
            }
        )

    calcium_raw_dir = ORIGINAL_DATA_DIR / "calcium" / "out_registration_batch"
    calcium_example = EXAMPLE_DATA_DIR / "calcium" / "DF1_cells_with_cluster_assignment_present.csv"
    calcium_metrics_count = len(list(calcium_raw_dir.rglob("calcium_metrics.csv")))
    calcium_timeseries_count = len(list(calcium_raw_dir.rglob("calcium_timeseries_long.csv")))
    calcium_verify_exists = (calcium_raw_dir / "verify_inputs.csv").exists()
    calcium_raw = pd.concat(
        [
            pd.read_csv(path).assign(bundle=path.parent.name)
            for path in sorted(calcium_raw_dir.rglob("matched_cells_with_stain_metrics.csv"))
        ],
        ignore_index=True,
    )
    calcium_example_df = pd.read_csv(calcium_example)
    calcium_shared_numeric = [
        column
        for column in calcium_raw.columns
        if column in calcium_example_df.columns
        and (
            pd.api.types.is_numeric_dtype(calcium_raw[column])
            or pd.api.types.is_numeric_dtype(calcium_example_df[column])
        )
    ]
    for df in (calcium_raw, calcium_example_df):
        for column in calcium_shared_numeric:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    calcium_bundle_match = calcium_raw.groupby("bundle").size().sort_index().equals(
        calcium_example_df.groupby("bundle").size().sort_index()
    )
    calcium_agg_match = aligned_allclose(
        grouped_numeric_summary(calcium_raw, "bundle", calcium_shared_numeric),
        grouped_numeric_summary(calcium_example_df, "bundle", calcium_shared_numeric),
        atol=1e-10,
    )
    rows.extend(
        [
            {
                "dataset": "calcium",
                "check": "raw_bundle_count",
                "status": "pass",
                "details": f"{len(list(calcium_raw_dir.rglob('matched_cells_with_stain_metrics.csv')))} bundle CSVs copied",
            },
            {
                "dataset": "calcium",
                "check": "calcium_metrics_bundle_count",
                "status": "pass" if calcium_metrics_count == 24 else "fail",
                "details": f"{calcium_metrics_count} calcium_metrics.csv bundle files copied",
            },
            {
                "dataset": "calcium",
                "check": "calcium_timeseries_bundle_count",
                "status": "pass" if calcium_timeseries_count == 24 else "fail",
                "details": f"{calcium_timeseries_count} calcium_timeseries_long.csv bundle files copied",
            },
            {
                "dataset": "calcium",
                "check": "verify_inputs_csv_present",
                "status": "pass" if calcium_verify_exists else "fail",
                "details": bool_json(calcium_verify_exists),
            },
            {
                "dataset": "calcium",
                "check": "row_count_match_vs_example_data",
                "status": "pass" if len(calcium_raw) == len(calcium_example_df) else "fail",
                "details": f"raw_rows={len(calcium_raw)} example_rows={len(calcium_example_df)}",
            },
            {
                "dataset": "calcium",
                "check": "per_bundle_row_count_match_vs_example_data",
                "status": "pass" if calcium_bundle_match else "fail",
                "details": bool_json(calcium_bundle_match),
            },
            {
                "dataset": "calcium",
                "check": "per_bundle_numeric_aggregate_match_vs_example_data",
                "status": "pass" if calcium_agg_match else "fail",
                "details": f"{len(calcium_shared_numeric)} shared numeric columns checked with atol=1e-10",
            },
        ]
    )

    caspase_raw_dir = ORIGINAL_DATA_DIR / "caspase" / "csvs"
    caspase_filtered = rebuild_caspase_legacy_filtered_cells(caspase_raw_dir)
    rows.append(
        {
            "dataset": "caspase",
            "check": "raw_csv_count",
            "status": "pass",
            "details": f"{len(list(caspase_raw_dir.glob('*.csv')))} raw CSVs copied",
        }
    )
    rows.append(
        {
            "dataset": "caspase",
            "check": "legacy_filtered_cell_table_rows",
            "status": "pass",
            "details": f"filtered_rows={len(caspase_filtered)} after six-treatment + non-marker-low notebook filter",
        }
    )

    local_caspase_cell_ref = ROOT.parent / "caspase" / "plots" / "caspase_cell_data.csv"
    if local_caspase_cell_ref.exists():
        caspase_cell_ref = normalise_caspase_phenotypes(pd.read_csv(local_caspase_cell_ref))
        key_cols = ["N", "well", "fov", "treatment", "phenotype"]
        rebuilt_grouped = (
            caspase_filtered.groupby(key_cols, observed=True)
            .agg(
                n_cells=("caspase_pos_bin", "size"),
                n_caspase_pos=("caspase_pos_bin", "sum"),
            )
            .reset_index()
        )
        ref_grouped = (
            caspase_cell_ref.groupby(key_cols, observed=True)
            .agg(
                n_cells=("caspase_pos_bin", "size"),
                n_caspase_pos=("caspase_pos_bin", "sum"),
            )
            .reset_index()
        )
        for df in (rebuilt_grouped, ref_grouped):
            for column in key_cols:
                df[column] = df[column].astype(str)
            for column in ["n_cells", "n_caspase_pos"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df.sort_values(key_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)
        grouped_match = rebuilt_grouped.equals(ref_grouped)
        rows.append(
            {
                "dataset": "caspase",
                "check": "raw_rebuild_matches_legacy_filtered_cell_table",
                "status": "pass" if grouped_match else "fail",
                "details": f"grouped_fov_rows={len(rebuilt_grouped)} legacy_rows={len(ref_grouped)}",
            }
        )
    else:
        rows.append(
            {
                "dataset": "caspase",
                "check": "raw_rebuild_matches_legacy_filtered_cell_table",
                "status": "skipped",
                "details": f"reference not found: {local_caspase_cell_ref}",
            }
        )

    packaged_caspase_summary = EXAMPLE_DATA_DIR / "caspase" / "caspase_summary_table.csv"
    local_caspase_summary = ROOT.parent / "caspase" / "caspase_summary_table.csv"
    if local_caspase_summary.exists() and packaged_caspase_summary.exists():
        packaged_summary_df = normalise_caspase_phenotypes(pd.read_csv(packaged_caspase_summary))
        local_summary_df = normalise_caspase_phenotypes(pd.read_csv(local_caspase_summary))
        key_cols = ["biol_rep", "phenotype", "treatment"]
        for df in (packaged_summary_df, local_summary_df):
            for column in key_cols:
                df[column] = df[column].astype(str)
            for column in ["n_cells", "frac_caspase_pos"]:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
            df.sort_values(key_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)
        summary_match = packaged_summary_df.equals(local_summary_df)
        rows.append(
            {
                "dataset": "caspase",
                "check": "packaged_summary_matches_local_legacy_summary_export",
                "status": "pass" if summary_match else "fail",
                "details": f"local_summary={local_caspase_summary}",
            }
        )
    else:
        rows.append(
            {
                "dataset": "caspase",
                "check": "packaged_summary_matches_local_legacy_summary_export",
                "status": "skipped",
                "details": f"reference not found: {local_caspase_summary} or {packaged_caspase_summary}",
            }
        )

    return rows


def write_markdown_report(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Original Data Validation",
        "",
        "This report checks the packaged `_github/original_data` bundles against the bundled runnable `example_data` or, where applicable, against the local legacy reference exports that generated those reviewer-facing inputs.",
        "",
    ]
    for dataset, sub_df in df.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        lines.append("")
        for row in sub_df.itertuples(index=False):
            lines.append(f"- `{row.check}`: `{row.status}`. {row.details}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_validation_rows()
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(VALIDATION_DIR / "original_data_validation_summary.csv", index=False)
    write_markdown_report(summary_df, VALIDATION_DIR / "original_data_validation_report.md")
    print(summary_df.to_string(index=False))
    print(f"\nSaved validation outputs to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
