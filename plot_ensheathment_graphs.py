#!/usr/bin/env python3
"""
Create treatment-ordered box plots for ensheathment quantification metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS_CSV = ROOT / "original_data" / "ensheathment" / "ensheathment_metrics.csv"
DEFAULT_OUTPUT_DIR = ROOT / "quantification" / "graphs"

TREATMENT_ORDER = ["vehicle", "pranlukast", "HAMI3379"]
BOX_COLOR = "0.92"
BOX_EDGE = "#4B5563"
NORMALIZED_YLABEL = "Relative to vehicle mean (=1)"

PLOT_SPECS = [
    {
        "column": "mbp_total_area_px",
        "slug": "mbp_total_area",
        "title": "Total MBP Area",
        "ylabel": NORMALIZED_YLABEL,
    },
    {
        "column": "mbp_soma_area_px",
        "slug": "mbp_soma_area",
        "title": "MBP Soma Area",
        "ylabel": NORMALIZED_YLABEL,
    },
    {
        "column": "process_to_total_ratio",
        "slug": "process_to_total_ratio",
        "title": "Process:Total MBP Ratio",
        "ylabel": NORMALIZED_YLABEL,
    },
    {
        "column": "mbp_process_area_px",
        "slug": "mbp_process_area",
        "title": "MBP Process Area",
        "ylabel": NORMALIZED_YLABEL,
    },
    {
        "column": "pct_mbp_nanofiber_colocalized",
        "slug": "pct_mbp_nanofiber_colocalized",
        "title": "Percent MBP Colocalized With Nanofiber",
        "ylabel": NORMALIZED_YLABEL,
    },
    {
        "column": "pct_process_nanofiber_colocalized",
        "slug": "pct_process_nanofiber_colocalized",
        "title": "Percent Process Colocalized With Nanofiber",
        "ylabel": NORMALIZED_YLABEL,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"treatment", "bio_rep"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["treatment"] = pd.Categorical(df["treatment"], categories=TREATMENT_ORDER, ordered=True)
    if "process_to_total_ratio" not in df.columns:
        df["process_to_total_ratio"] = np.where(
            df["mbp_total_area_px"] > 0,
            df["mbp_process_area_px"] / df["mbp_total_area_px"],
            np.nan,
        )
    unique_reps = sorted(df["bio_rep"].dropna().astype(str).unique())
    rep_map = {rep: rep.upper() for rep in unique_reps}
    rep_order = [rep_map[rep] for rep in unique_reps]
    df["bio_rep_label"] = pd.Categorical(df["bio_rep"].astype(str).map(rep_map), categories=rep_order, ordered=True)
    df = df[df["treatment"].notna()].copy()
    return df


def build_rep_palette(data: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    rep_order = list(data["bio_rep_label"].cat.categories)
    return dict(zip(rep_order, sns.color_palette("tab10", n_colors=max(1, len(rep_order)))))


def normalize_to_vehicle_mean(data: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    subset = data[["treatment", "bio_rep", "bio_rep_label", metric_col]].dropna().copy()
    parts: list[pd.DataFrame] = []
    for _, group in subset.groupby("bio_rep", observed=True):
        group = group.copy()
        vehicle_mean = group.loc[group["treatment"] == TREATMENT_ORDER[0], metric_col].mean()
        if pd.notna(vehicle_mean) and vehicle_mean != 0:
            group["plot_value"] = group[metric_col] / vehicle_mean
        else:
            group["plot_value"] = group[metric_col]
        parts.append(group)
    if not parts:
        return subset.assign(plot_value=np.nan)
    return pd.concat(parts, ignore_index=True)


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def add_rep_legend(ax: plt.Axes, rep_palette: dict[str, tuple[float, float, float]]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
            label=label,
        )
        for label, color in rep_palette.items()
    ]
    ax.legend(handles=handles, title="Biol. replicate", loc="upper left", frameon=True, fontsize=9, title_fontsize=9)


def build_rep_legend_handles(rep_palette: dict[str, tuple[float, float, float]]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label=label,
        )
        for label, color in rep_palette.items()
    ]


def save_rep_legend(output_dir: Path, rep_palette: dict[str, tuple[float, float, float]]) -> None:
    handles = build_rep_legend_handles(rep_palette)
    fig_height = max(1.8, 0.55 * len(handles) + 0.7)
    fig, ax = plt.subplots(figsize=(2.6, fig_height))
    ax.axis("off")
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        title="Biological replicate",
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=10,
        title_fontsize=10,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    fig.savefig(output_dir / "biol_rep_legend.png", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(output_dir / "biol_rep_legend.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    spec: dict[str, str],
    rep_palette: dict[str, tuple[float, float, float]],
) -> None:
    metric_col = spec["column"]
    subset = normalize_to_vehicle_mean(data, metric_col)
    if subset.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_axis(ax, spec["title"], spec["ylabel"])
        return

    sns.boxplot(
        data=subset,
        x="treatment",
        y="plot_value",
        order=TREATMENT_ORDER,
        color=BOX_COLOR,
        width=0.55,
        linewidth=1.4,
        fliersize=0,
        boxprops={"edgecolor": BOX_EDGE},
        medianprops={"color": "#111827", "linewidth": 1.6},
        whiskerprops={"color": BOX_EDGE, "linewidth": 1.2},
        capprops={"color": BOX_EDGE, "linewidth": 1.2},
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="treatment",
        y="plot_value",
        order=TREATMENT_ORDER,
        hue="bio_rep_label",
        hue_order=list(rep_palette.keys()),
        palette=rep_palette,
        alpha=0.95,
        size=4.5,
        jitter=0.18,
        dodge=False,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
        zorder=3,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    style_axis(ax, spec["title"], spec["ylabel"])


def save_single_metric_plot(data: pd.DataFrame, output_dir: Path, spec: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    rep_palette = build_rep_palette(data)
    plot_metric(ax, data, spec, rep_palette)
    fig.tight_layout()
    png_path = output_dir / f"{spec['slug']}.png"
    svg_path = output_dir / f"{spec['slug']}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def save_overview_panel(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()
    rep_palette = build_rep_palette(data)
    for ax, spec in zip(axes_flat, PLOT_SPECS):
        plot_metric(ax, data, spec, rep_palette)

    fig.suptitle("Ensheathment Metrics by Treatment", fontsize=18, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "ensheathment_metrics_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ensheathment_metrics_overview.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "DejaVu Sans"

    data = load_metrics(args.metrics_csv)
    rep_palette = build_rep_palette(data)
    for spec in PLOT_SPECS:
        save_single_metric_plot(data, args.output_dir, spec)
        print(f"[plot] {spec['slug']}")

    save_overview_panel(data, args.output_dir)
    save_rep_legend(args.output_dir, rep_palette)
    print(f"[done] graphs in {args.output_dir}")


if __name__ == "__main__":
    main()
