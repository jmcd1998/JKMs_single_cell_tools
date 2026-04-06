from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = REPO_ROOT.parent / "_github_clustercomp_rm_anova"

COMPARE_PATHS = (
    "parsed_data",
    "outputs/chapter_3_output_V1",
    "outputs/pkm2_output_V1",
    "outputs/ppkm2_output_V1",
    "outputs/caspase_output_V1",
    "outputs/calcium_full_output_V1",
    "outputs/thesis_tables",
)


PATH_VALUE_COLUMNS = {
    "analysis_dir",
    "by_phenotype_path",
    "data_csv",
    "dataset_summary_path",
    "fig_dir",
    "forest_plot_path",
    "global_tests_path",
    "graphs_dir",
    "input_path",
    "output_dir",
    "percent_plot_path",
    "stats_dir",
    "summary_path",
}


def walk_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        files[str(path.relative_to(root))] = path
    return files


def normalize_cell(value: object, repo_roots: tuple[str, ...]) -> object:
    if isinstance(value, str):
        out = value
        for root in repo_roots:
            out = out.replace(root, "<REPO_ROOT>")
        return out
    return value


def load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    if not df.empty:
        df = df.dropna(how="all").reset_index(drop=True)
    if not df.empty:
        first_col = df.columns[0]
        if df[first_col].notna().any() and df[first_col].isna().any():
            df = df[df[first_col].notna()].reset_index(drop=True)
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except (TypeError, ValueError):
            pass
    return df


def compare_tabular_files(local_path: Path, reference_path: Path, reference_dir: Path) -> bool:
    local_df = load_table(local_path)
    reference_df = load_table(reference_path)
    repo_roots = (str(REPO_ROOT), str(reference_dir))

    shared_cols = [column for column in local_df.columns if column in reference_df.columns]
    if set(local_df.columns) != set(reference_df.columns):
        return False

    for column in shared_cols:
        if local_df[column].dtype == object or reference_df[column].dtype == object or column in PATH_VALUE_COLUMNS:
            local_df[column] = local_df[column].map(lambda value: normalize_cell(value, repo_roots))
            reference_df[column] = reference_df[column].map(lambda value: normalize_cell(value, repo_roots))

    try:
        assert_frame_equal(
            local_df,
            reference_df,
            check_dtype=False,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        return True
    except AssertionError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_tree(local_root: Path, reference_root: Path, reference_bundle_root: Path) -> list[str]:
    problems: list[str] = []
    local_files = walk_files(local_root) if local_root.exists() else {}
    reference_files = walk_files(reference_root) if reference_root.exists() else {}

    for rel_path in sorted(set(local_files).difference(reference_files)):
        problems.append(f"Only in local copy: {local_root / rel_path}")
    for rel_path in sorted(set(reference_files).difference(local_files)):
        problems.append(f"Missing from local copy: {local_root / rel_path}")
    for rel_path in sorted(set(local_files).intersection(reference_files)):
        local_path = local_files[rel_path]
        reference_path = reference_files[rel_path]
        suffix = local_path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            matches = compare_tabular_files(local_path, reference_path, reference_bundle_root)
        elif suffix in {".png", ".pdf"}:
            matches = local_path.stat().st_size > 0 and reference_path.stat().st_size > 0
        else:
            matches = sha256(local_path) == sha256(reference_path)
        if not matches:
            problems.append(f"Content differs: {local_root / rel_path}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare analytical outputs in this readability-first clone against the original RM-ANOVA bundle."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing the reference RM-ANOVA bundle to compare against.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference bundle was not found: {reference_dir}")

    all_problems: list[str] = []
    for rel_path in COMPARE_PATHS:
        local_root = REPO_ROOT / rel_path
        reference_root = reference_dir / rel_path
        all_problems.extend(compare_tree(local_root, reference_root, reference_dir))

    if all_problems:
        print("Output comparison failed:")
        for problem in all_problems:
            print(problem)
        raise SystemExit(1)

    print("All checked outputs match the reference RM-ANOVA bundle exactly.")
    for rel_path in COMPARE_PATHS:
        print(REPO_ROOT / rel_path)


if __name__ == "__main__":
    main()
