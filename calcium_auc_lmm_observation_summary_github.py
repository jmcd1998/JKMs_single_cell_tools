from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class CalciumAucGithubConfig:
    input_candidates: tuple[Path, ...]
    output_dir: Path
    treatment_order: tuple[str, ...] = ("MDL29951", "pranlukast", "HAMI3379")
    cluster_col: str = "cluster_k3_present"
    group_col: str = "bio_rep"
    field_col: str = "tech_rep_id"
    treatment_col: str = "treatment"
    include_nonpositive_auc: bool = True
    optimizer_sequence: tuple[str, ...] = ("lbfgs", "bfgs", "cg")

    @property
    def input_path(self) -> Path:
        for path in self.input_candidates:
            if path.exists():
                return path
        missing = self.input_candidates[0]
        raise FileNotFoundError(
            f"Expected clustered calcium table at {missing}. "
            "Run run_calcium_github.py first."
        )


def ensure_output_dir(config: CalciumAucGithubConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def write_csv_with_lock_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def remove_outliers_iqr(
    df: pd.DataFrame,
    col: str,
    *,
    group_cols: Sequence[str] | None = None,
    k: float = 1.5,
) -> pd.DataFrame:
    if group_cols is None:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        return df[(df[col] >= lo) & (df[col] <= hi)].copy()

    grouped = df.groupby(list(group_cols), observed=True)[col]
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return df[(df[col] >= lo) & (df[col] <= hi)].copy()


def load_calcium_clustered_table(config: CalciumAucGithubConfig) -> pd.DataFrame:
    df = pd.read_csv(config.input_path)
    required = [
        config.group_col,
        config.field_col,
        config.treatment_col,
        config.cluster_col,
        "calcium AUC",
        "calcium_auc_w1",
        "calcium_auc_w2",
        "calcium_auc_w3",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {config.input_path}: {missing}")
    return df


def build_auc_clean_table(df: pd.DataFrame, config: CalciumAucGithubConfig) -> pd.DataFrame:
    auc_df = df.dropna(subset=["calcium AUC"]).copy()
    if not config.include_nonpositive_auc:
        auc_df = auc_df[auc_df["calcium AUC"] > 0].copy()
    return remove_outliers_iqr(
        auc_df,
        "calcium AUC",
        group_cols=[config.treatment_col, config.cluster_col],
        k=1.5,
    )


def prepare_auc_model_df(
    auc_clean_df: pd.DataFrame,
    *,
    auc_col: str,
    window_label: str,
    config: CalciumAucGithubConfig,
) -> pd.DataFrame:
    model_df = auc_clean_df.copy()
    model_df[config.cluster_col] = pd.to_numeric(model_df[config.cluster_col], errors="coerce")
    model_df = model_df.dropna(
        subset=[auc_col, config.cluster_col, config.group_col, config.field_col, config.treatment_col]
    ).copy()

    cluster_levels = sorted(model_df[config.cluster_col].astype(int).unique().tolist())
    model_df[config.cluster_col] = pd.Categorical(
        model_df[config.cluster_col].astype(int),
        categories=cluster_levels,
        ordered=True,
    )
    model_df[config.treatment_col] = pd.Categorical(
        model_df[config.treatment_col].astype(str),
        categories=list(config.treatment_order),
        ordered=True,
    )
    model_df = model_df.dropna(subset=[config.treatment_col]).copy()
    model_df[config.group_col] = model_df[config.group_col].astype(str)
    model_df[config.field_col] = model_df[config.field_col].astype(str)
    model_df["window_label"] = window_label
    model_df["model_response"] = auc_col
    model_df["transform"] = "raw"
    model_df["transform_param"] = np.nan
    return model_df


def fit_nested_mixedlm(
    formula: str,
    data: pd.DataFrame,
    *,
    group_col: str,
    field_col: str,
    optimizer_sequence: Sequence[str],
) -> tuple[object, str, list[str]]:
    last_exception: Exception | None = None
    last_fit = None
    last_warnings: list[str] = []

    for method in optimizer_sequence:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data[group_col],
                    vc_formula={"field_unit": f"0 + C({field_col})"},
                    re_formula="1",
                )
                fit = model.fit(reml=True, method=method, disp=False)

            warning_messages: list[str] = []
            for warning in caught:
                message = str(warning.message).strip()
                if message and message not in warning_messages:
                    warning_messages.append(message)

            setattr(fit, "_codex_optimizer", method)
            setattr(fit, "_codex_warning_messages", warning_messages)
            last_fit = fit
            last_warnings = warning_messages
            if bool(getattr(fit, "converged", False)):
                return fit, f"{group_col} + {field_col}", warning_messages
        except Exception as exc:
            last_exception = exc

    if last_fit is not None:
        return last_fit, f"{group_col} + {field_col}", last_warnings

    raise RuntimeError(
        f"MixedLM failed for formula '{formula}' with optimizers {list(optimizer_sequence)}: {last_exception}"
    ) from last_exception


def summarise_observation_structure(
    data: pd.DataFrame,
    *,
    group_col: str,
    field_col: str,
    treatment_col: str,
) -> dict[str, object]:
    group_sizes = data.groupby(group_col, observed=True).size()
    field_sizes = data.groupby(field_col, observed=True).size()
    group_treatment_sizes = data.groupby([group_col, treatment_col], observed=True).size()
    fields_per_group = data.groupby(group_col, observed=True)[field_col].nunique()
    return {
        "n_cells_used": int(len(data)),
        "n_biol_reps": int(group_sizes.shape[0]),
        "n_fields_or_tech_reps": int(field_sizes.shape[0]),
        "n_biol_rep_treatment_groups": int(group_treatment_sizes.shape[0]),
        "avg_cells_per_biol_rep_total": float(group_sizes.mean()) if not group_sizes.empty else np.nan,
        "median_cells_per_biol_rep_total": float(group_sizes.median()) if not group_sizes.empty else np.nan,
        "avg_cells_per_biol_rep_treatment_group": float(group_treatment_sizes.mean()) if not group_treatment_sizes.empty else np.nan,
        "median_cells_per_biol_rep_treatment_group": float(group_treatment_sizes.median()) if not group_treatment_sizes.empty else np.nan,
        "avg_cells_per_field": float(field_sizes.mean()) if not field_sizes.empty else np.nan,
        "median_cells_per_field": float(field_sizes.median()) if not field_sizes.empty else np.nan,
        "avg_fields_per_biol_rep": float(fields_per_group.mean()) if not fields_per_group.empty else np.nan,
        "median_fields_per_biol_rep": float(fields_per_group.median()) if not fields_per_group.empty else np.nan,
    }


def build_model_summary_rows(config: CalciumAucGithubConfig) -> pd.DataFrame:
    df = load_calcium_clustered_table(config)
    auc_clean_df = build_auc_clean_table(df, config)

    rows: list[dict[str, object]] = []
    auc_windows = [
        ("w1", "calcium_auc_w1"),
        ("w2", "calcium_auc_w2"),
        ("w3", "calcium_auc_w3"),
    ]
    for window_label, auc_col in auc_windows:
        model_df = prepare_auc_model_df(
            auc_clean_df,
            auc_col=auc_col,
            window_label=window_label,
            config=config,
        )
        formula = (
            f'Q("{auc_col}") ~ '
            f'C({config.treatment_col}, Treatment(reference="{config.treatment_order[0]}")) * '
            f'C({config.cluster_col})'
        )
        row = {
            "dataset": "calcium_multiplex",
            "model_family": "auc_mixedlm_raw",
            "window_label": window_label,
            "response_var": auc_col,
            "raw_response_var": auc_col,
            "transform": "raw",
            "transform_param": np.nan,
            "formula": formula,
            "random_structure": f"{config.group_col} + {config.field_col}",
            "field_unit_col": config.field_col,
            "data_csv": str(config.input_path),
            "status": "ok",
            "converged": np.nan,
            "optimizer": "",
            "warning_count": 0,
            "warning_messages": "",
            "error": "",
        }
        row.update(
            summarise_observation_structure(
                model_df,
                group_col=config.group_col,
                field_col=config.field_col,
                treatment_col=config.treatment_col,
            )
        )
        try:
            fit, random_structure, warning_messages = fit_nested_mixedlm(
                formula,
                model_df,
                group_col=config.group_col,
                field_col=config.field_col,
                optimizer_sequence=config.optimizer_sequence,
            )
            row["random_structure"] = random_structure
            row["converged"] = bool(getattr(fit, "converged", False))
            row["optimizer"] = getattr(fit, "_codex_optimizer", "unknown")
            row["warning_count"] = len(warning_messages)
            row["warning_messages"] = " | ".join(warning_messages)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values(["model_family", "window_label"], inplace=True, na_position="last")
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df


def write_calcium_auc_lmm_summary(config: CalciumAucGithubConfig) -> dict[str, object]:
    ensure_output_dir(config)
    summary_df = build_model_summary_rows(config)
    fit_path = config.output_dir / "calcium_auc_lmm_model_fit_summary.csv"
    summary_path = config.output_dir / "calcium_auc_lmm_supplementary_model_observation_summary.csv"
    fit_path = write_csv_with_lock_fallback(summary_df, fit_path)

    supplementary_cols = [
        "dataset",
        "model_family",
        "window_label",
        "response_var",
        "raw_response_var",
        "transform",
        "formula",
        "random_structure",
        "field_unit_col",
        "status",
        "converged",
        "optimizer",
        "warning_count",
        "data_csv",
        "n_cells_used",
        "n_biol_reps",
        "n_fields_or_tech_reps",
        "n_biol_rep_treatment_groups",
        "avg_cells_per_biol_rep_total",
        "median_cells_per_biol_rep_total",
        "avg_cells_per_biol_rep_treatment_group",
        "median_cells_per_biol_rep_treatment_group",
        "avg_cells_per_field",
        "median_cells_per_field",
        "avg_fields_per_biol_rep",
        "median_fields_per_biol_rep",
    ]
    supplementary_df = summary_df[[col for col in supplementary_cols if col in summary_df.columns]].copy()
    summary_path = write_csv_with_lock_fallback(supplementary_df, summary_path)
    return {
        "config": config,
        "summary": summary_df,
        "supplementary_summary": supplementary_df,
        "fit_summary_path": fit_path,
        "supplementary_summary_path": summary_path,
    }


def inspect_calcium_dataset(config: CalciumAucGithubConfig) -> pd.DataFrame:
    df = load_calcium_clustered_table(config)
    rows = [
        {"metric": "data_csv", "value": str(config.input_path)},
        {"metric": "output_dir", "value": str(config.output_dir)},
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_biol_reps", "value": int(df[config.group_col].dropna().astype(str).nunique())},
        {"metric": "n_fields_or_tech_reps", "value": int(df[config.field_col].dropna().astype(str).nunique())},
        {"metric": "n_treatments", "value": int(df[config.treatment_col].dropna().astype(str).nunique())},
        {"metric": "n_clusters", "value": int(pd.to_numeric(df[config.cluster_col], errors="coerce").dropna().nunique())},
    ]
    return pd.DataFrame(rows)


def build_default_config() -> CalciumAucGithubConfig:
    root = Path(__file__).resolve().parent
    return CalciumAucGithubConfig(
        input_candidates=(root / "outputs" / "calcium_full_output_V1" / "DF1_cells_with_cluster_assignment_present.csv",),
        output_dir=root / "outputs" / "calcium_auc_output_V1",
    )


def main() -> None:
    config = build_default_config()
    print(inspect_calcium_dataset(config).to_string(index=False))
    results = write_calcium_auc_lmm_summary(config)
    print("fit_summary_path:", results["fit_summary_path"])
    print("supplementary_summary_path:", results["supplementary_summary_path"])


if __name__ == "__main__":
    main()
