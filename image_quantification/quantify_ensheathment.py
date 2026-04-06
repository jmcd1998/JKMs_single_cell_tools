#!/usr/bin/env python3
"""
Quantify ensheathment metrics from ilastik segmentation outputs.

Inputs
------
- Manifest from `get_ilastik_outputs.py`
- Root traced TIFF exports in `Desktop/out`
- ROI zip files saved during the Fiji tracing step

Per-cell outputs
----------------
- total MBP-positive area
- MBP soma-positive area
- process:total ratio
- process area
- process area by raw subtraction (for transparency)
- MBP/C3 colocalized area
- percent MBP colocalized
- percent process colocalized

Also writes one QC PNG per cell with region categories in distinct colours.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries


DEFAULT_INPUT_ROOT = Path(r"C:\Users\JackM\Desktop\out")
DEFAULT_MANIFEST = Path(r"C:\Users\JackM\src2\ensheathment\ilastik_outputs\job_manifest.csv")
DEFAULT_SEG_ROOT = Path(r"C:\Users\JackM\src2\ensheathment\ilastik_outputs")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\JackM\src2\ensheathment\quantification")
DEFAULT_FFMPEG = Path(
    r"C:\Users\JackM\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
)

MBP_POSITIVE_LABEL = 1
C3_POSITIVE_LABELS = {1, 3}
NANOFIBER_DISPLAY_NAME = "Nanofiber"

FIGURE_BG = "#09131A"
PANEL_BG = "#101C26"
PANEL_EDGE = "#223240"
TEXT_PRIMARY = "#F5F8FB"
TEXT_MUTED = "#A8B5C2"
COLOR_MBP_SIGNAL = np.array([1.00, 0.28, 0.30])
COLOR_NANOFIBER_SIGNAL = np.array([0.34, 0.88, 0.42])
COLOR_MBP_SOMA = np.array([1.00, 0.50, 0.32])
COLOR_MBP_PROCESS = np.array([0.92, 0.17, 0.32])
COLOR_NANOFIBER_ONLY = np.array([0.34, 0.88, 0.42])
COLOR_COLOC_SOMA = np.array([0.98, 0.84, 0.26])
COLOR_COLOC_PROCESS = np.array([0.73, 0.48, 0.98])

DATE_TO_BIO_REP = {
    "2403": "n1",
    "2703": "n2",
    "0404": "n3",
    "1605": "n4",
}

PREFIX_TO_TREATMENT = {
    "X": "pranlukast",
    "Y": "vehicle",
    "Z": "HAMI3379",
}


@dataclass(frozen=True)
class ManifestRow:
    marker: str
    bio_rep: str
    prefix: str
    date_code: str
    scene: int
    cell_index: int
    region: str
    channel: str
    source_path: Path
    ilp_path: Path
    output_path: Path


@dataclass(frozen=True)
class CropGroup:
    key: str
    bio_rep: str
    prefix: str
    date_code: str
    scene: int
    cell_index: int
    original_image_name: str
    field_of_view: str
    mbp_cell: ManifestRow
    mbp_soma: ManifestRow
    c3_cell: ManifestRow


@dataclass(frozen=True)
class RoiRecord:
    name: str
    top: int
    left: int
    bottom: int
    right: int
    xs: np.ndarray
    ys: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--seg-root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffmpeg-exe", type=Path, default=DEFAULT_FFMPEG)
    return parser.parse_args()


def read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ManifestRow(
                    marker=row["marker"],
                    bio_rep=row["bio_rep"],
                    prefix=row["prefix"],
                    date_code=row["date_code"],
                    scene=int(row["scene"]),
                    cell_index=int(row["cell_index"]),
                    region=row["region"],
                    channel=row["channel"],
                    source_path=Path(row["source_path"]),
                    ilp_path=Path(row["ilp_path"]),
                    output_path=Path(row["output_path"]),
                )
            )
    return rows


def build_job_manifest_row(
    *,
    prefix: str,
    date_code: str,
    scene: int,
    cell_index: int,
    bio_rep: str,
    mbp_cell: ManifestRow | None,
    mbp_soma: ManifestRow | None,
    c3_cell: ManifestRow | None,
    status: str,
    issue_type: str,
    issue_detail: str,
    qc_path: Path | None = None,
    thesis_path: Path | None = None,
) -> dict[str, object]:
    original_stem = f"{prefix}{date_code}_{scene}"
    return {
        "cell_id": f"{original_stem}_cell{cell_index:02d}",
        "original_image_file_name": f"{original_stem}.tif",
        "field_of_view": original_stem,
        "bio_rep": bio_rep,
        "treatment_blinded": prefix,
        "treatment": PREFIX_TO_TREATMENT.get(prefix, "unknown"),
        "date_code": date_code,
        "prefix": prefix,
        "scene_index": scene,
        "cell_index": cell_index,
        "status": status,
        "issue_type": issue_type,
        "issue_detail": issue_detail,
        "mbp_cell_present": mbp_cell is not None,
        "mbp_soma_present": mbp_soma is not None,
        "c3_cell_present": c3_cell is not None,
        "mbp_cell_source_path": str(mbp_cell.source_path) if mbp_cell is not None else "",
        "mbp_soma_source_path": str(mbp_soma.source_path) if mbp_soma is not None else "",
        "c3_cell_source_path": str(c3_cell.source_path) if c3_cell is not None else "",
        "mbp_cell_seg_path": str(mbp_cell.output_path) if mbp_cell is not None else "",
        "mbp_soma_seg_path": str(mbp_soma.output_path) if mbp_soma is not None else "",
        "c3_cell_seg_path": str(c3_cell.output_path) if c3_cell is not None else "",
        "qc_image_path": str(qc_path) if qc_path is not None else "",
        "thesis_image_path": str(thesis_path) if thesis_path is not None else "",
    }


def group_manifest_rows(rows: list[ManifestRow]) -> tuple[list[CropGroup], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int, int], dict[tuple[str, str], ManifestRow]] = {}
    meta: dict[tuple[str, str, int, int], ManifestRow] = {}

    for row in rows:
        key = (row.prefix, row.date_code, row.scene, row.cell_index)
        grouped.setdefault(key, {})[(row.marker, row.region)] = row
        meta[key] = row

    out: list[CropGroup] = []
    job_manifest_rows: list[dict[str, object]] = []
    for key, bucket in grouped.items():
        required = [("mbp", "cell"), ("mbp", "soma"), ("c3", "cell")]
        absent = [pair for pair in required if pair not in bucket]
        prefix, date_code, scene, cell_index = key
        any_row = meta[key]
        original_stem = f"{prefix}{date_code}_{scene}"
        mbp_cell = bucket.get(("mbp", "cell"))
        mbp_soma = bucket.get(("mbp", "soma"))
        c3_cell = bucket.get(("c3", "cell"))

        if absent:
            job_manifest_rows.append(
                build_job_manifest_row(
                    prefix=prefix,
                    date_code=date_code,
                    scene=scene,
                    cell_index=cell_index,
                    bio_rep=any_row.bio_rep,
                    mbp_cell=mbp_cell,
                    mbp_soma=mbp_soma,
                    c3_cell=c3_cell,
                    status="skipped_missing_bundle",
                    issue_type="incomplete_bundle",
                    issue_detail=f"Missing manifest entries: {absent}",
                )
            )
            continue

        out.append(
            CropGroup(
                key=f"{original_stem}_cell{cell_index:02d}",
                bio_rep=any_row.bio_rep,
                prefix=prefix,
                date_code=date_code,
                scene=scene,
                cell_index=cell_index,
                original_image_name=f"{original_stem}.tif",
                field_of_view=original_stem,
                mbp_cell=mbp_cell,
                mbp_soma=mbp_soma,
                c3_cell=c3_cell,
            )
        )

    return (
        sorted(out, key=lambda g: (g.bio_rep, g.prefix, g.date_code, g.scene, g.cell_index)),
        job_manifest_rows,
    )


def parse_imagej_roi(data: bytes, name: str) -> RoiRecord:
    if data[:4] != b"Iout":
        raise ValueError(f"{name} is not an ImageJ ROI")
    top, left, bottom, right, n_coords = struct.unpack(">hhhhH", data[8:18])
    if n_coords <= 0:
        raise ValueError(f"{name} contains no coordinates")

    x_start = 64
    x_end = x_start + 2 * n_coords
    y_end = x_end + 2 * n_coords
    xs = np.frombuffer(data[x_start:x_end], dtype=">u2").astype(np.int32) + left
    ys = np.frombuffer(data[x_end:y_end], dtype=">u2").astype(np.int32) + top
    return RoiRecord(
        name=name,
        top=int(top),
        left=int(left),
        bottom=int(bottom),
        right=int(right),
        xs=xs,
        ys=ys,
    )


def build_mask_from_roi(roi: RoiRecord) -> np.ndarray:
    width = roi.right - roi.left
    height = roi.bottom - roi.top
    mask_img = Image.new("L", (width, height), 0)
    pts = [(int(x - roi.left), int(y - roi.top)) for x, y in zip(roi.xs, roi.ys)]
    ImageDraw.Draw(mask_img).polygon(pts, outline=1, fill=1)
    return np.asarray(mask_img, dtype=bool)


def load_roi_pair(group: CropGroup, input_root: Path) -> tuple[RoiRecord, RoiRecord, np.ndarray, np.ndarray]:
    roi_zip = input_root / f"{group.field_of_view}_manual_rois.zip"
    if not roi_zip.exists():
        raise FileNotFoundError(f"Missing ROI zip: {roi_zip}")

    cell_name = f"{group.field_of_view}_cell{group.cell_index:02d}_cell.roi"
    soma_name = f"{group.field_of_view}_cell{group.cell_index:02d}_soma.roi"

    with zipfile.ZipFile(roi_zip, "r") as zf:
        members = set(zf.namelist())
        if cell_name not in members or soma_name not in members:
            raise FileNotFoundError(
                f"ROI zip {roi_zip.name} missing {cell_name} or {soma_name}; has {sorted(members)}"
            )
        cell_roi = parse_imagej_roi(zf.read(cell_name), cell_name)
        soma_roi = parse_imagej_roi(zf.read(soma_name), soma_name)

    return cell_roi, soma_roi, build_mask_from_roi(cell_roi), build_mask_from_roi(soma_roi)


def read_raw_tiff(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D raw crop TIFF: {path}, got shape {arr.shape}")
    return arr


def read_lzw_segmentation(path: Path, shape: tuple[int, int], ffmpeg_exe: Path) -> np.ndarray:
    cmd = [
        str(ffmpeg_exe),
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    expected_size = shape[0] * shape[1]
    if len(result.stdout) != expected_size:
        raise ValueError(
            f"Unexpected segmentation byte count for {path.name}: "
            f"expected {expected_size}, got {len(result.stdout)}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(shape)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    positives = arr[arr > 0]
    if positives.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(positives, 1))
    hi = float(np.percentile(positives, 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def tint_grayscale(arr: np.ndarray, color: np.ndarray, gamma: float = 0.90) -> np.ndarray:
    norm = np.clip(arr.astype(np.float32), 0.0, 1.0) ** gamma
    return np.dstack([norm * color[0], norm * color[1], norm * color[2]])


def stylize_image_axis(ax: plt.Axes, title: str, panel_letter: str) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor(PANEL_EDGE)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, fontweight="semibold", pad=10)
    ax.text(
        0.03,
        0.97,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=TEXT_PRIMARY,
        path_effects=[pe.withStroke(linewidth=3, foreground=FIGURE_BG)],
    )


def stylize_thesis_axis(ax: plt.Axes, title: str, panel_letter: str) -> None:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#D0D6DD")

    ax.set_title(title, color="#111827", fontsize=12, fontweight="semibold", pad=10)
    ax.text(
        0.03,
        0.97,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="white",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )


def fmt_pct(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.1f}%"


def build_visual_layers(
    mbp_raw_cell: np.ndarray,
    c3_raw_cell: np.ndarray,
    soma_roi_mask_aligned: np.ndarray,
    mbp_soma_only: np.ndarray,
    mbp_process_only: np.ndarray,
    c3_only: np.ndarray,
    coloc_soma: np.ndarray,
    coloc_process: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mbp_norm = normalize_for_display(mbp_raw_cell)
    c3_norm = normalize_for_display(c3_raw_cell)
    soma_boundary = find_boundaries(soma_roi_mask_aligned, mode="outer")

    mbp_rgb = tint_grayscale(mbp_norm, COLOR_MBP_SIGNAL)
    nanofiber_rgb = tint_grayscale(c3_norm, COLOR_NANOFIBER_SIGNAL)
    composite_base = np.clip(mbp_rgb + nanofiber_rgb, 0.0, 1.0)
    overlay_rgb = np.clip(composite_base * 0.52 + 0.05, 0.0, 1.0)

    overlay_specs = [
        ("MBP soma", mbp_soma_only, COLOR_MBP_SOMA),
        ("MBP process", mbp_process_only, COLOR_MBP_PROCESS),
        (f"{NANOFIBER_DISPLAY_NAME} only", c3_only, COLOR_NANOFIBER_ONLY),
        ("Soma colocalized", coloc_soma, COLOR_COLOC_SOMA),
        ("Process colocalized", coloc_process, COLOR_COLOC_PROCESS),
    ]
    for _, mask, color in overlay_specs:
        if not np.any(mask):
            continue
        overlay_rgb[mask] = 0.18 * overlay_rgb[mask] + 0.82 * color

    mbp_rgb[soma_boundary] = np.array([1.0, 1.0, 1.0])
    nanofiber_rgb[soma_boundary] = np.array([1.0, 1.0, 1.0])
    overlay_rgb[soma_boundary] = np.array([1.0, 1.0, 1.0])
    return mbp_rgb, nanofiber_rgb, overlay_rgb


def aligned_mask_in_cell(
    small_mask: np.ndarray,
    cell_roi: RoiRecord,
    small_roi: RoiRecord,
    cell_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(cell_shape, dtype=bool)
    y0 = small_roi.top - cell_roi.top
    x0 = small_roi.left - cell_roi.left
    y1 = y0 + small_mask.shape[0]
    x1 = x0 + small_mask.shape[1]
    out[y0:y1, x0:x1] = small_mask
    return out


def save_qc_figure(
    qc_path: Path,
    cell_id: str,
    treatment: str,
    bio_rep: str,
    field_of_view: str,
    mbp_raw_cell: np.ndarray,
    c3_raw_cell: np.ndarray,
    soma_roi_mask_aligned: np.ndarray,
    mbp_soma_only: np.ndarray,
    mbp_process_only: np.ndarray,
    c3_only: np.ndarray,
    coloc_soma: np.ndarray,
    coloc_process: np.ndarray,
    total_mbp_area: int,
    soma_mbp_area: int,
    process_area: int,
    total_nanofiber_area: int,
    total_coloc_area: int,
    soma_to_total_ratio: float,
    process_to_total_ratio: float,
    pct_colocalized: float,
    pct_process_colocalized: float,
) -> None:
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    mbp_rgb, nanofiber_rgb, overlay_rgb = build_visual_layers(
        mbp_raw_cell=mbp_raw_cell,
        c3_raw_cell=c3_raw_cell,
        soma_roi_mask_aligned=soma_roi_mask_aligned,
        mbp_soma_only=mbp_soma_only,
        mbp_process_only=mbp_process_only,
        c3_only=c3_only,
        coloc_soma=coloc_soma,
        coloc_process=coloc_process,
    )

    fig = plt.figure(figsize=(12.5, 9.0), facecolor=FIGURE_BG)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.05],
        wspace=0.06,
        hspace=0.08,
    )
    ax_mbp = fig.add_subplot(gs[0, 0])
    ax_nanofiber = fig.add_subplot(gs[0, 1])
    ax_overlay = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    stylize_image_axis(ax_mbp, "MBP Signal", "A")
    stylize_image_axis(ax_nanofiber, f"{NANOFIBER_DISPLAY_NAME} Signal", "B")
    stylize_image_axis(ax_overlay, "Quantification Map", "C")

    ax_mbp.imshow(mbp_rgb, interpolation="nearest")
    ax_nanofiber.imshow(nanofiber_rgb, interpolation="nearest")
    ax_overlay.imshow(overlay_rgb, interpolation="nearest")

    ax_summary.set_facecolor(PANEL_BG)
    ax_summary.set_xticks([])
    ax_summary.set_yticks([])
    for spine in ax_summary.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor(PANEL_EDGE)

    ax_summary.text(
        0.04,
        0.95,
        "D",
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    ax_summary.text(
        0.09,
        0.95,
        "Cell Summary",
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TEXT_PRIMARY,
    )
    ax_summary.text(
        0.09,
        0.88,
        cell_id,
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    ax_summary.text(
        0.09,
        0.82,
        f"Treatment: {treatment}  |  Replicate: {bio_rep}  |  FOV: {field_of_view}",
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT_MUTED,
    )

    summary_lines = [
        ("MBP total area", f"{total_mbp_area:,} px"),
        ("MBP soma area", f"{soma_mbp_area:,} px"),
        ("MBP process area", f"{process_area:,} px"),
        ("Process:total ratio", "n/a" if np.isnan(process_to_total_ratio) else f"{process_to_total_ratio:.3f}"),
        (f"{NANOFIBER_DISPLAY_NAME} area", f"{total_nanofiber_area:,} px"),
        ("MBP + nanofiber area", f"{total_coloc_area:,} px"),
        ("% MBP colocalized", fmt_pct(pct_colocalized)),
        ("% process colocalized", fmt_pct(pct_process_colocalized)),
    ]

    y = 0.71
    for label, value in summary_lines:
        ax_summary.text(
            0.09,
            y,
            label,
            transform=ax_summary.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color=TEXT_MUTED,
        )
        ax_summary.text(
            0.92,
            y,
            value,
            transform=ax_summary.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            fontweight="semibold",
            color=TEXT_PRIMARY,
        )
        y -= 0.058

    handles = [
        Line2D([0], [0], color=(1, 1, 1), lw=2.5, label="Soma boundary"),
        Patch(color=COLOR_MBP_SOMA, label="MBP soma"),
        Patch(color=COLOR_MBP_PROCESS, label="MBP process"),
        Patch(color=COLOR_NANOFIBER_ONLY, label=f"{NANOFIBER_DISPLAY_NAME} only"),
        Patch(color=COLOR_COLOC_SOMA, label="Soma colocalized"),
        Patch(color=COLOR_COLOC_PROCESS, label="Process colocalized"),
    ]
    legend = ax_summary.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.03),
        frameon=False,
        fontsize=8.5,
        ncol=2,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.6,
        labelcolor=TEXT_PRIMARY,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.suptitle(
        f"Ensheathment Quantification QC | {treatment}",
        x=0.055,
        y=0.985,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.055,
        0.955,
        f"Raw marker signal and classified overlap map for {cell_id}",
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT_MUTED,
    )
    fig.savefig(qc_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_thesis_figure(
    thesis_path: Path,
    cell_id: str,
    treatment: str,
    bio_rep: str,
    field_of_view: str,
    mbp_raw_cell: np.ndarray,
    c3_raw_cell: np.ndarray,
    soma_roi_mask_aligned: np.ndarray,
    mbp_soma_only: np.ndarray,
    mbp_process_only: np.ndarray,
    c3_only: np.ndarray,
    coloc_soma: np.ndarray,
    coloc_process: np.ndarray,
) -> None:
    thesis_path.parent.mkdir(parents=True, exist_ok=True)

    mbp_rgb, nanofiber_rgb, overlay_rgb = build_visual_layers(
        mbp_raw_cell=mbp_raw_cell,
        c3_raw_cell=c3_raw_cell,
        soma_roi_mask_aligned=soma_roi_mask_aligned,
        mbp_soma_only=mbp_soma_only,
        mbp_process_only=mbp_process_only,
        c3_only=c3_only,
        coloc_soma=coloc_soma,
        coloc_process=coloc_process,
    )

    fig = plt.figure(figsize=(13.5, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.06], wspace=0.05)
    ax_mbp = fig.add_subplot(gs[0, 0])
    ax_nanofiber = fig.add_subplot(gs[0, 1])
    ax_overlay = fig.add_subplot(gs[0, 2])

    stylize_thesis_axis(ax_mbp, "MBP", "A")
    stylize_thesis_axis(ax_nanofiber, NANOFIBER_DISPLAY_NAME, "B")
    stylize_thesis_axis(ax_overlay, "Quantification Map", "C")

    ax_mbp.imshow(mbp_rgb, interpolation="nearest")
    ax_nanofiber.imshow(nanofiber_rgb, interpolation="nearest")
    ax_overlay.imshow(overlay_rgb, interpolation="nearest")

    fig.suptitle(
        f"{cell_id}  |  {treatment}  |  {bio_rep}  |  {field_of_view}",
        x=0.055,
        y=0.98,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )

    handles = [
        Line2D([0], [0], color=(1, 1, 1), lw=2.2, label="Soma boundary"),
        Patch(color=COLOR_MBP_SOMA, label="MBP soma"),
        Patch(color=COLOR_MBP_PROCESS, label="MBP process"),
        Patch(color=COLOR_NANOFIBER_ONLY, label=f"{NANOFIBER_DISPLAY_NAME} only"),
        Patch(color=COLOR_COLOC_SOMA, label="Soma colocalized"),
        Patch(color=COLOR_COLOC_PROCESS, label="Process colocalized"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.6,
        columnspacing=1.6,
    )
    fig.savefig(thesis_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def quantify_group(
    group: CropGroup,
    input_root: Path,
    ffmpeg_exe: Path,
    qc_dir: Path,
    thesis_dir: Path,
) -> dict[str, object]:
    cell_roi, soma_roi, cell_roi_mask, soma_roi_mask = load_roi_pair(group, input_root)

    mbp_raw_cell = read_raw_tiff(group.mbp_cell.source_path)
    mbp_raw_soma = read_raw_tiff(group.mbp_soma.source_path)
    c3_raw_cell = read_raw_tiff(group.c3_cell.source_path)

    if mbp_raw_cell.shape != cell_roi_mask.shape:
        raise ValueError(f"Cell ROI shape mismatch for {group.key}: raw {mbp_raw_cell.shape}, roi {cell_roi_mask.shape}")
    if mbp_raw_soma.shape != soma_roi_mask.shape:
        raise ValueError(f"Soma ROI shape mismatch for {group.key}: raw {mbp_raw_soma.shape}, roi {soma_roi_mask.shape}")
    if c3_raw_cell.shape != cell_roi_mask.shape:
        raise ValueError(f"C3 ROI shape mismatch for {group.key}: raw {c3_raw_cell.shape}, roi {cell_roi_mask.shape}")

    mbp_cell_seg = read_lzw_segmentation(group.mbp_cell.output_path, mbp_raw_cell.shape, ffmpeg_exe)
    mbp_soma_seg = read_lzw_segmentation(group.mbp_soma.output_path, mbp_raw_soma.shape, ffmpeg_exe)
    c3_cell_seg = read_lzw_segmentation(group.c3_cell.output_path, c3_raw_cell.shape, ffmpeg_exe)

    mbp_total_mask = (mbp_cell_seg == MBP_POSITIVE_LABEL) & cell_roi_mask
    mbp_soma_mask = (mbp_soma_seg == MBP_POSITIVE_LABEL) & soma_roi_mask
    c3_positive_mask = np.isin(c3_cell_seg, list(C3_POSITIVE_LABELS)) & cell_roi_mask

    soma_roi_mask_aligned = aligned_mask_in_cell(soma_roi_mask, cell_roi, soma_roi, mbp_raw_cell.shape)
    mbp_soma_mask_aligned = aligned_mask_in_cell(mbp_soma_mask, cell_roi, soma_roi, mbp_raw_cell.shape)

    mbp_soma_within_total = mbp_soma_mask_aligned & mbp_total_mask
    mbp_process_mask = mbp_total_mask & ~mbp_soma_within_total

    coloc_total_mask = mbp_total_mask & c3_positive_mask
    coloc_soma_mask = mbp_soma_within_total & c3_positive_mask
    coloc_process_mask = mbp_process_mask & c3_positive_mask

    total_mbp_area = int(mbp_total_mask.sum())
    soma_mbp_area = int(mbp_soma_mask.sum())
    soma_mbp_area_within_total = int(mbp_soma_within_total.sum())
    process_area = int(mbp_process_mask.sum())
    process_area_subtract = int(total_mbp_area - soma_mbp_area)
    total_c3_area = int(c3_positive_mask.sum())
    total_coloc_area = int(coloc_total_mask.sum())
    soma_coloc_area = int(coloc_soma_mask.sum())
    process_coloc_area = int(coloc_process_mask.sum())

    soma_to_total_ratio = float(soma_mbp_area / total_mbp_area) if total_mbp_area > 0 else np.nan
    process_to_total_ratio = float(process_area / total_mbp_area) if total_mbp_area > 0 else np.nan
    pct_colocalized = float((total_coloc_area / total_mbp_area) * 100.0) if total_mbp_area > 0 else np.nan
    pct_process_colocalized = (
        float((process_coloc_area / process_area) * 100.0) if process_area > 0 else np.nan
    )

    qc_path = qc_dir / f"{group.key}__qc.png"
    thesis_path = thesis_dir / f"{group.key}__thesis.png"
    save_qc_figure(
        qc_path=qc_path,
        cell_id=group.key,
        treatment=PREFIX_TO_TREATMENT.get(group.prefix, "unknown"),
        bio_rep=group.bio_rep,
        field_of_view=group.field_of_view,
        mbp_raw_cell=mbp_raw_cell,
        c3_raw_cell=c3_raw_cell,
        soma_roi_mask_aligned=soma_roi_mask_aligned,
        mbp_soma_only=mbp_soma_within_total & ~c3_positive_mask,
        mbp_process_only=mbp_process_mask & ~c3_positive_mask,
        c3_only=c3_positive_mask & ~mbp_total_mask,
        coloc_soma=coloc_soma_mask,
        coloc_process=coloc_process_mask,
        total_mbp_area=total_mbp_area,
        soma_mbp_area=soma_mbp_area,
        process_area=process_area,
        total_nanofiber_area=total_c3_area,
        total_coloc_area=total_coloc_area,
        soma_to_total_ratio=soma_to_total_ratio,
        process_to_total_ratio=process_to_total_ratio,
        pct_colocalized=pct_colocalized,
        pct_process_colocalized=pct_process_colocalized,
    )
    save_thesis_figure(
        thesis_path=thesis_path,
        cell_id=group.key,
        treatment=PREFIX_TO_TREATMENT.get(group.prefix, "unknown"),
        bio_rep=group.bio_rep,
        field_of_view=group.field_of_view,
        mbp_raw_cell=mbp_raw_cell,
        c3_raw_cell=c3_raw_cell,
        soma_roi_mask_aligned=soma_roi_mask_aligned,
        mbp_soma_only=mbp_soma_within_total & ~c3_positive_mask,
        mbp_process_only=mbp_process_mask & ~c3_positive_mask,
        c3_only=c3_positive_mask & ~mbp_total_mask,
        coloc_soma=coloc_soma_mask,
        coloc_process=coloc_process_mask,
    )

    return {
        "cell_id": group.key,
        "crop_file_name": group.mbp_cell.source_path.name,
        "original_image_file_name": group.original_image_name,
        "original_image_stem": group.field_of_view,
        "field_of_view": group.field_of_view,
        "bio_rep": group.bio_rep,
        "treatment_blinded": group.prefix,
        "treatment": PREFIX_TO_TREATMENT.get(group.prefix, "unknown"),
        "date_code": group.date_code,
        "prefix": group.prefix,
        "scene_index": group.scene,
        "cell_index": group.cell_index,
        "mbp_cell_source_path": str(group.mbp_cell.source_path),
        "mbp_soma_source_path": str(group.mbp_soma.source_path),
        "c3_cell_source_path": str(group.c3_cell.source_path),
        "mbp_cell_seg_path": str(group.mbp_cell.output_path),
        "mbp_soma_seg_path": str(group.mbp_soma.output_path),
        "c3_cell_seg_path": str(group.c3_cell.output_path),
        "qc_image_path": str(qc_path),
        "thesis_image_path": str(thesis_path),
        "mbp_total_area_px": total_mbp_area,
        "mbp_soma_area_px": soma_mbp_area,
        "mbp_soma_area_within_total_px": soma_mbp_area_within_total,
        "soma_to_total_ratio": soma_to_total_ratio,
        "process_to_total_ratio": process_to_total_ratio,
        "mbp_process_area_px": process_area,
        "mbp_process_area_subtract_px": process_area_subtract,
        "c3_total_area_px": total_c3_area,
        "nanofiber_total_area_px": total_c3_area,
        "mbp_c3_colocalized_area_px": total_coloc_area,
        "mbp_nanofiber_colocalized_area_px": total_coloc_area,
        "mbp_c3_soma_colocalized_area_px": soma_coloc_area,
        "mbp_nanofiber_soma_colocalized_area_px": soma_coloc_area,
        "mbp_c3_process_colocalized_area_px": process_coloc_area,
        "mbp_nanofiber_process_colocalized_area_px": process_coloc_area,
        "pct_mbp_colocalized": pct_colocalized,
        "pct_mbp_nanofiber_colocalized": pct_colocalized,
        "pct_process_colocalized": pct_process_colocalized,
        "pct_process_nanofiber_colocalized": pct_process_colocalized,
    }


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.ffmpeg_exe.exists():
        raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg_exe}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    qc_dir = args.output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir = args.output_root / "thesis"
    thesis_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    groups, job_manifest_rows = group_manifest_rows(rows)

    results = []
    for idx, group in enumerate(groups, start=1):
        print(f"[quantify] {idx}/{len(groups)} {group.key}")
        qc_path = qc_dir / f"{group.key}__qc.png"
        thesis_path = thesis_dir / f"{group.key}__thesis.png"
        try:
            result = quantify_group(group, args.input_root, args.ffmpeg_exe, qc_dir, thesis_dir)
            results.append(result)
            job_manifest_rows.append(
                build_job_manifest_row(
                    prefix=group.prefix,
                    date_code=group.date_code,
                    scene=group.scene,
                    cell_index=group.cell_index,
                    bio_rep=group.bio_rep,
                    mbp_cell=group.mbp_cell,
                    mbp_soma=group.mbp_soma,
                    c3_cell=group.c3_cell,
                    status="ok",
                    issue_type="",
                    issue_detail="",
                    qc_path=qc_path,
                    thesis_path=thesis_path,
                )
            )
        except FileNotFoundError as exc:
            print(f"[skip] {group.key} missing bundle: {exc}")
            job_manifest_rows.append(
                build_job_manifest_row(
                    prefix=group.prefix,
                    date_code=group.date_code,
                    scene=group.scene,
                    cell_index=group.cell_index,
                    bio_rep=group.bio_rep,
                    mbp_cell=group.mbp_cell,
                    mbp_soma=group.mbp_soma,
                    c3_cell=group.c3_cell,
                    status="skipped_missing_bundle",
                    issue_type="missing_file",
                    issue_detail=str(exc),
                    qc_path=qc_path if qc_path.exists() else None,
                    thesis_path=thesis_path if thesis_path.exists() else None,
                )
            )
        except Exception as exc:
            print(f"[warn] {group.key} quantification failed: {exc}")
            job_manifest_rows.append(
                build_job_manifest_row(
                    prefix=group.prefix,
                    date_code=group.date_code,
                    scene=group.scene,
                    cell_index=group.cell_index,
                    bio_rep=group.bio_rep,
                    mbp_cell=group.mbp_cell,
                    mbp_soma=group.mbp_soma,
                    c3_cell=group.c3_cell,
                    status="failed_quantification",
                    issue_type=type(exc).__name__,
                    issue_detail=str(exc),
                    qc_path=qc_path if qc_path.exists() else None,
                    thesis_path=thesis_path if thesis_path.exists() else None,
                )
            )

    out_csv = args.output_root / "ensheathment_metrics.csv"
    write_csv_rows(out_csv, results)
    job_manifest_csv = args.output_root / "quantification_job_manifest.csv"
    write_csv_rows(job_manifest_csv, job_manifest_rows)

    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {job_manifest_csv}")
    print(f"[done] qc images in {qc_dir}")
    print(f"[done] thesis images in {thesis_dir}")


if __name__ == "__main__":
    main()
