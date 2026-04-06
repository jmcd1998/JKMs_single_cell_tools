#!/usr/bin/env python3
"""
Fit one-term treatment MixedLMs for ensheathment metrics.

Model structure
---------------
- Fixed effect: treatment
- Random intercepts: bio_rep and field_of_view

Outputs
-------
- mixedlm_model_summary.csv
- mixedlm_fixed_effects.csv
- mixedlm_emmeans.csv
- mixedlm_vehicle_contrasts.csv
- mixedlm_pairwise_contrasts.csv
- per-metric statsmodels summary text files
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS_CSV = ROOT / "original_data" / "ensheathment" / "ensheathment_metrics.csv"
DEFAULT_OUTPUT_DIR = ROOT / "quantification" / "mixedlm"

TREATMENT_ORDER = ["vehicle", "pranlukast", "HAMI3379"]
REFERENCE_TREATMENT = TREATMENT_ORDER[0]
OPTIMIZER_SEQUENCE = ("lbfgs", "bfgs", "cg")
Z_975 = float(stats.norm.ppf(0.975))

METRIC_SPECS = [
    {"column": "mbp_total_area_px", "label": "Total MBP Area"},
    {"column": "mbp_soma_area_px", "label": "MBP Soma Area"},
    {"column": "process_to_total_ratio", "label": "Process:Total MBP Ratio"},
    {"column": "mbp_process_area_px", "label": "MBP Process Area"},
    {"column": "pct_mbp_nanofiber_colocalized", "label": "Percent MBP Colocalized With Nanofiber"},
    {"column": "pct_process_nanofiber_colocalized", "label": "Percent Process Colocalized With Nanofiber"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    required = {"treatment", "bio_rep", "field_of_view", "mbp_total_area_px", "mbp_process_area_px"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {sorted(missing)}")

    if "process_to_total_ratio" not in df.columns:
        df["process_to_total_ratio"] = np.where(
            df["mbp_total_area_px"] > 0,
            df["mbp_process_area_px"] / df["mbp_total_area_px"],
            np.nan,
        )

    df["treatment"] = pd.Categorical(df["treatment"], categories=TREATMENT_ORDER, ordered=True)
    df["bio_rep"] = df["bio_rep"].astype(str)
    df["field_of_view"] = df["field_of_view"].astype(str)
    return df


def summarise_structure(data: pd.DataFrame) -> dict[str, object]:
    group_sizes = data.groupby("bio_rep", observed=True).size()
    field_sizes = data.groupby("field_of_view", observed=True).size()
    fields_per_group = data.groupby("bio_rep", observed=True)["field_of_view"].nunique()
    return {
        "n_cells_used": int(len(data)),
        "n_biol_reps": int(group_sizes.shape[0]),
        "n_fields": int(field_sizes.shape[0]),
        "avg_cells_per_biol_rep": float(group_sizes.mean()) if not group_sizes.empty else np.nan,
        "median_cells_per_biol_rep": float(group_sizes.median()) if not group_sizes.empty else np.nan,
        "avg_cells_per_fov": float(field_sizes.mean()) if not field_sizes.empty else np.nan,
        "median_cells_per_fov": float(field_sizes.median()) if not field_sizes.empty else np.nan,
        "avg_fovs_per_biol_rep": float(fields_per_group.mean()) if not fields_per_group.empty else np.nan,
        "median_fovs_per_biol_rep": float(fields_per_group.median()) if not fields_per_group.empty else np.nan,
    }


def fit_nested_mixedlm(formula: str, data: pd.DataFrame):
    last_exception: Exception | None = None
    last_fit = None
    last_warnings: list[str] = []

    for method in OPTIMIZER_SEQUENCE:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data["bio_rep"],
                    vc_formula={"field_of_view": "0 + C(field_of_view)"},
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
                return fit, warning_messages
        except Exception as exc:
            last_exception = exc

    if last_fit is not None:
        return last_fit, last_warnings

    raise RuntimeError(
        f"MixedLM failed for formula '{formula}' with optimizers {list(OPTIMIZER_SEQUENCE)}: {last_exception}"
    ) from last_exception


def fixed_effect_vector(param_names: list[str], treatment: str) -> np.ndarray:
    vec = np.zeros(len(param_names), dtype=float)
    for idx, name in enumerate(param_names):
        if name == "Intercept":
            vec[idx] = 1.0
        elif name == f'C(treatment, Treatment(reference="{REFERENCE_TREATMENT}"))[T.{treatment}]':
            vec[idx] = 1.0
    return vec


def contrast_stats(
    estimate_vector: np.ndarray,
    fe_params: pd.Series,
    cov_fe: pd.DataFrame,
) -> dict[str, float]:
    beta = fe_params.to_numpy(dtype=float)
    cov = cov_fe.to_numpy(dtype=float)
    estimate = float(estimate_vector @ beta)
    variance = float(estimate_vector @ cov @ estimate_vector)
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance))
    z_value = float(estimate / se) if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "estimate": estimate,
        "se": se,
        "z": z_value,
        "pvalue": p_value,
        "ci_low": float(estimate - Z_975 * se) if np.isfinite(se) else np.nan,
        "ci_high": float(estimate + Z_975 * se) if np.isfinite(se) else np.nan,
    }


def compute_omnibus_treatment_test(fe_params: pd.Series, cov_fe: pd.DataFrame) -> tuple[float, int, float]:
    treat_names = [
        name
        for name in fe_params.index
        if name.startswith(f'C(treatment, Treatment(reference="{REFERENCE_TREATMENT}"))[')
    ]
    if not treat_names:
        return np.nan, 0, np.nan

    beta = fe_params.loc[treat_names].to_numpy(dtype=float)
    cov = cov_fe.loc[treat_names, treat_names].to_numpy(dtype=float)
    chi2_stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
    df_num = int(len(treat_names))
    p_value = float(stats.chi2.sf(chi2_stat, df_num))
    return chi2_stat, df_num, p_value


def compute_residual_sd(fit) -> float:
    scale = float(getattr(fit, "scale", np.nan))
    if np.isfinite(scale) and scale > 0:
        return float(np.sqrt(scale))
    return np.nan


def write_summary_text(output_dir: Path, metric: str, fit) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{metric}_mixedlm_summary.txt"
    text_path.write_text(str(fit.summary()), encoding="utf-8")


def run_metric(
    metric_spec: dict[str, str],
    df: pd.DataFrame,
    summary_dir: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    metric = metric_spec["column"]
    label = metric_spec["label"]
    data = df[["treatment", "bio_rep", "field_of_view", metric]].dropna().copy()

    row = {
        "metric": metric,
        "metric_label": label,
        "formula": f'{metric} ~ C(treatment, Treatment(reference="{REFERENCE_TREATMENT}"))',
        "random_structure": "bio_rep + field_of_view",
        "status": "ok",
        "converged": np.nan,
        "optimizer": "",
        "warning_count": 0,
        "warning_messages": "",
        "error": "",
    }
    row.update(summarise_structure(data))

    fixed_rows: list[dict[str, object]] = []
    emm_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    vehicle_rows: list[dict[str, object]] = []

    if data["treatment"].nunique() < 2:
        row["status"] = "failed"
        row["error"] = "Need at least two treatment levels to fit the model."
        return row, fixed_rows, emm_rows, contrast_rows, vehicle_rows

    fit, warning_messages = fit_nested_mixedlm(row["formula"], data)
    row["converged"] = bool(getattr(fit, "converged", False))
    row["optimizer"] = getattr(fit, "_codex_optimizer", "")
    row["warning_count"] = len(warning_messages)
    row["warning_messages"] = " | ".join(warning_messages)
    row["log_likelihood"] = float(fit.llf)
    row["aic"] = float(fit.aic) if np.isfinite(fit.aic) else np.nan
    row["bic"] = float(fit.bic) if np.isfinite(fit.bic) else np.nan
    row["residual_scale"] = float(fit.scale)
    row["residual_sd"] = compute_residual_sd(fit)
    treatment_means = data.groupby("treatment", observed=True)[metric].mean()

    fe_params = fit.fe_params.astype(float)
    fe_names = list(fe_params.index)
    cov_all = fit.cov_params()
    cov_fe = cov_all.loc[fe_names, fe_names]

    chi2_stat, df_num, p_value = compute_omnibus_treatment_test(fe_params, cov_fe)
    row["omnibus_treatment_chi2"] = chi2_stat
    row["omnibus_treatment_df"] = df_num
    row["omnibus_treatment_pvalue"] = p_value

    bse_fe = fit.bse_fe.reindex(fe_names).astype(float)
    for term in fe_names:
        estimate = float(fe_params[term])
        se = float(bse_fe[term])
        z_value = float(estimate / se) if se > 0 else np.nan
        p_term = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        fixed_rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "term": term,
                "estimate": estimate,
                "se": se,
                "z": z_value,
                "pvalue": p_term,
                "ci_low": float(estimate - Z_975 * se) if np.isfinite(se) else np.nan,
                "ci_high": float(estimate + Z_975 * se) if np.isfinite(se) else np.nan,
                "std_effect": estimate / row["residual_sd"] if np.isfinite(row["residual_sd"]) and row["residual_sd"] > 0 else np.nan,
                "residual_sd": row["residual_sd"],
            }
        )

    emm_vectors = {treatment: fixed_effect_vector(fe_names, treatment) for treatment in TREATMENT_ORDER}
    for treatment in TREATMENT_ORDER:
        stats_row = contrast_stats(emm_vectors[treatment], fe_params, cov_fe)
        emm_rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "treatment": treatment,
                **stats_row,
            }
        )

    for level_a, level_b in [
        ("pranlukast", "vehicle"),
        ("HAMI3379", "vehicle"),
        ("HAMI3379", "pranlukast"),
    ]:
        contrast_vec = emm_vectors[level_a] - emm_vectors[level_b]
        stats_row = contrast_stats(contrast_vec, fe_params, cov_fe)
        contrast_rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "contrast": f"{level_a} - {level_b}",
                "contrast_family": "vs_vehicle" if level_b == "vehicle" else "other_pairwise",
                "level_a": level_a,
                "level_b": level_b,
                **stats_row,
                "std_effect": (
                    stats_row["estimate"] / row["residual_sd"]
                    if np.isfinite(row["residual_sd"]) and row["residual_sd"] > 0
                    else np.nan
                ),
                "residual_sd": row["residual_sd"],
            }
        )

        if level_b == "vehicle":
            prefix = f"{level_a}_vs_vehicle"
            row[f"{prefix}_estimate"] = stats_row["estimate"]
            row[f"{prefix}_se"] = stats_row["se"]
            row[f"{prefix}_z"] = stats_row["z"]
            row[f"{prefix}_pvalue"] = stats_row["pvalue"]
            row[f"{prefix}_ci_low"] = stats_row["ci_low"]
            row[f"{prefix}_ci_high"] = stats_row["ci_high"]
            row[f"{prefix}_std_effect"] = (
                stats_row["estimate"] / row["residual_sd"]
                if np.isfinite(row["residual_sd"]) and row["residual_sd"] > 0
                else np.nan
            )
            vehicle_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "treatment": level_a,
                    "estimate_vs_vehicle": stats_row["estimate"],
                    "SE": stats_row["se"],
                    "z": stats_row["z"],
                    "p_value": stats_row["pvalue"],
                    "ci_low": stats_row["ci_low"],
                    "ci_high": stats_row["ci_high"],
                    "std_effect_vs_vehicle": (
                        stats_row["estimate"] / row["residual_sd"]
                        if np.isfinite(row["residual_sd"]) and row["residual_sd"] > 0
                        else np.nan
                    ),
                    "residual_sd": row["residual_sd"],
                    "vehicle_mean_raw": float(treatment_means.get(level_b, np.nan)),
                    "treatment_mean_raw": float(treatment_means.get(level_a, np.nan)),
                    "n_cells": row["n_cells_used"],
                    "n_groups": row["n_biol_reps"],
                    "n_fovs": row["n_fields"],
                    "avg_group_size": row["avg_cells_per_biol_rep"],
                    "avg_fov_size": row["avg_cells_per_fov"],
                    "avg_fovs_per_group": row["avg_fovs_per_biol_rep"],
                    "converged": row["converged"],
                    "optimizer": row["optimizer"],
                    "warning_count": row["warning_count"],
                    "random_structure": row["random_structure"],
                }
            )

    write_summary_text(summary_dir, metric, fit)
    return row, fixed_rows, emm_rows, contrast_rows, vehicle_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.metrics_csv)

    model_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    emm_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    vehicle_rows: list[dict[str, object]] = []

    for metric_spec in METRIC_SPECS:
        print(f"[mixedlm] {metric_spec['column']}")
        model_row, metric_fixed, metric_emm, metric_contrasts, metric_vehicle = run_metric(metric_spec, df, summary_dir)
        model_rows.append(model_row)
        fixed_rows.extend(metric_fixed)
        emm_rows.extend(metric_emm)
        contrast_rows.extend(metric_contrasts)
        vehicle_rows.extend(metric_vehicle)

    pd.DataFrame(model_rows).to_csv(args.output_dir / "mixedlm_model_summary.csv", index=False)
    pd.DataFrame(fixed_rows).to_csv(args.output_dir / "mixedlm_fixed_effects.csv", index=False)
    pd.DataFrame(emm_rows).to_csv(args.output_dir / "mixedlm_emmeans.csv", index=False)
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(args.output_dir / "mixedlm_pairwise_contrasts.csv", index=False)
    pd.DataFrame(vehicle_rows).to_csv(args.output_dir / "mixedlm_vehicle_contrasts.csv", index=False)
    print(f"[done] mixedlm outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
