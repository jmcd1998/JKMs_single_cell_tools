"""
40x MBP/PDGFRa/CNP pipeline

LIF -> Ilastik (MBP/PDGFRa/CNP) + Cellpose (nuclei + cells)
 -> per-cell morphology + MFI-normalised intensities

Key rule:
    - Cellpose is used ONLY to define ROIs.
    - Actual cell mask for morphology & MFI:
        cell_mask = (MBP_mask ∪ PDGFRa_mask ∪ CNP_mask) ∩ (Cellpose ROI)

Background:
    - Per-scene, per-channel GLOBAL background, computed from pixels
      far from any cell/marker mask (robust median).

IMPORTANT:
    - Uses *raw* Ilastik masks (class == 1) with no binary reconstruction
      or extra filtering.

Requirements (conda env e.g. `cellpose3-gpu-python`):
    pip install readlif tifffile numpy scikit-image cellpose pandas scipy matplotlib
"""

import os
import pathlib
import subprocess
from typing import Dict, List
import gc  # for explicit garbage collection

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from readlif.reader import LifFile
from cellpose import models

from scipy import ndimage as ndi
from numpy import trapz

from skimage.morphology import (
    disk,
    binary_dilation,
    skeletonize,
    convex_hull_image,
    closing,
    remove_small_holes,  # used only inside Sholl smoothing
)
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


# ==========================
# CONFIG
# ==========================

# --- output root ---
OUTPUT_DIR = r"C:\Users\JackM\40xMBPCNP\python_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FULLMASK_DIR = os.path.join(OUTPUT_DIR, "full_masks")
QC_DIR       = os.path.join(OUTPUT_DIR, "QC")
CSV_DIR      = os.path.join(OUTPUT_DIR, "csvs")  # per-LIF CSVs go here

for _d in [FULLMASK_DIR, QC_DIR, CSV_DIR]:
    os.makedirs(_d, exist_ok=True)

# kept for backwards compatibility (not used now for writing)
OUT_CSV = os.path.join(OUTPUT_DIR, "per_cell_stats.csv")

# --- Ilastik executable ---
ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.0.post1\ilastik.exe"
if not os.path.exists(ILASTIK_EXE):
    raise FileNotFoundError(f"ILASTIK_EXE does not exist: {ILASTIK_EXE}")

# --- Cellpose models ---
NUC_MODEL_PATH  = r"C:\Users\JackM\40xMBPCNP\cellpose nuclei\models\Jack Nuclei"
CELL_MODEL_PATH = r"C:\Users\JackM\40xMBPCNP\cellpose\models\CP_20250902_074451"

# --- LIF + Ilastik configs (per N) ---
LIF_CONFIG: Dict[str, Dict[str, str]] = {
################################N1###################################

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Vehicle_2602.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_mbp\n1_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n1_pdgfra\n1_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_cnp\n1_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Pranlukast_2602.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_mbp\n1_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n1_pdgfra\n1_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_cnp\n1_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\MDL_2602.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_mbp\n1_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n1_pdgfra\n1_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_cnp\n1_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\HAMI_2602.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_mbp\n1_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n1_pdgfra\n1_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n1_cnp\n1_cnp.ilp",
    },  

#############################N2####################################


    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Vehicle_0404.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_mbp\n2_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n2_pdgfra\n2_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_cnp\n2_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Pranlukast_0404.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_mbp\n2_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n2_pdgfra\n2_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_cnp\n2_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\MDL_0404.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_mbp\n2_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n2_pdgfra\n2_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_cnp\n2_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\HAMI_0404.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_mbp\n2_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n2_pdgfra\n2_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n2_cnp\n2_cnp.ilp",
    },

#############################N3####################################


    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Vehicle_1104.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_mbp\n3_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n3_pdgfra\n3_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_cnp\n3_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Pranlukast_1104.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_mbp\n3_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n3_pdgfra\n3_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_cnp\n3_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\MDL_1104.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_mbp\n3_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n3_pdgfra\n3_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_cnp\n3_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\HAMI_1104.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_mbp\n3_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n3_pdgfra\n3_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n3_cnp\n3_cnp.ilp",
    },

    ##################################### N4 ##############################

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Vehicle_0606.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_mbp\n4_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n4_pdgfra\n4_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_cnp\n4_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\Pranlukast_0606.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_mbp\n4_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n4_pdgfra\n4_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_cnp\n4_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\MDL_0606.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_mbp\n4_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n4_pdgfra\n4_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_cnp\n4_cnp.ilp",
    },

    r"C:\Users\JackM\40xMBPCNP\40xN1N2N3N4lif\HAMI_0606.lif": {
        "mbp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_mbp\n4_mbp.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\40xMBPCNP\n4_pdgfra\n4_pdgfra.ilp",
        "cnp_ilp":    r"C:\Users\JackM\40xMBPCNP\n4_cnp\n4_cnp.ilp",
    },
}

# --- Channel indices (CYX array) ---
# 1 = DAPI, 2 = PDGFRa, 3 = MBP, 4 = CNP
CHAN_DAPI   = 0
CHAN_PDGFRa = 1
CHAN_MBP    = 2
CHAN_CNP    = 3

# --- Cellpose params ---
NUC_DIAMETER  = 70.0     # px
CELL_DIAMETER = 180.0    # px

# --- filtering thresholds ---
MIN_CELL_AREA_PX = 2000  # on marker-union mask; tweak if needed

# --- QC plotting options ---
PLOT_QC_CELLS = True
QC_PLOT_EVERY = 10       # plot every 10th cell


# ==========================
# HELPERS
# ==========================

def run_ilastik_on_array(
    img: np.ndarray,
    ilp_path: str,
    tmp_root: pathlib.Path,
    tag: str,
) -> np.ndarray:
    """
    Run Ilastik pixel classification 'Simple Segmentation' on a 2D image array.
    Writes tmp_root/<tag>_in.tiff, expects tmp_root/<tag>_seg.tiff back.
    """
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Clean any old junk for this tag
    for p in tmp_root.glob(f"{tag}_*.tif*"):
        try:
            p.unlink()
        except Exception:
            pass

    tmp_in  = tmp_root / f"{tag}_in.tiff"
    tmp_out = tmp_root / f"{tag}_seg.tiff"

    tiff.imwrite(tmp_in, img.astype(np.float64))

    cmd = [
        ILASTIK_EXE,
        "--headless",
        f"--project={ilp_path}",
        "--export_source=Simple Segmentation",
        "--output_format=tiff",
        f"--output_filename_format={tmp_out}",
        str(tmp_in),
    ]
    print(f"[ilastik] {tag}: running Ilastik...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Ilastik failed for tag '{tag}'.\n"
            f"Command:\n{' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not tmp_out.exists():
        existing = [p.name for p in tmp_root.glob("*.tif*")]
        raise FileNotFoundError(
            f"Expected Ilastik output '{tmp_out}' not found for tag '{tag}'.\n"
            f"Existing TIFFs in {tmp_root}:\n{existing}"
        )

    seg = tiff.imread(tmp_out)
    if seg.ndim > 2:
        seg = np.squeeze(seg)

    # Clean up input file to reduce clutter
    try:
        tmp_in.unlink(missing_ok=True)
    except Exception:
        pass

    # (We leave tmp_out on disk; removing it is optional and doesn't affect RAM.)

    return seg.astype(np.uint8)


def load_scene_as_cyx_and_name(lif: LifFile, scene_idx: int):
    """
    Load one scene from a LIF file as (C, Y, X) float64 + raw series name.
    """
    lif_img = lif.get_image(scene_idx)
    raw_name = str(lif_img.name)

    chans = []
    for ch_img in lif_img.get_iter_c(z=0, t=0, m=0):
        chans.append(np.array(ch_img))

    data = np.stack(chans, axis=0).astype(np.float64)
    return data, raw_name


def clean_series_name(lif_path: pathlib.Path, raw_series_name: str, scene_idx: int) -> str:
    """
    Construct a safe base_name keeping treatment info.
    """
    parts = raw_series_name.split(":", 1)
    if len(parts) > 1:
        series_desc = parts[1].strip()
    else:
        series_desc = raw_series_name.strip()

    series_desc_clean = (
        series_desc.replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\", "_")
                   .replace("µ", "u")
                   .replace("μ", "u")
                   .replace(":", "_")
    )

    base_name = (
        f"{lif_path.stem}_S{scene_idx+1}_{series_desc_clean}"
        .replace("-", "_")
        .replace("+", "_")
    )
    return base_name


def compute_global_background(
    channel_raw: np.ndarray,
    all_cells_mask: np.ndarray,
    all_marker_union: np.ndarray,
    margin_px: int = 5,
    bg_percentile: float = 50.0,
) -> float:
    """
    Compute a robust global background level for one scene + channel.

    - all_cells_mask: masks_cell > 0  (any Cellpose label)
    - all_marker_union: MBP ∪ PDGFRa ∪ CNP (full-image)
    """
    forbidden = all_cells_mask | all_marker_union
    forbidden = binary_dilation(forbidden, disk(margin_px))

    bg_region = ~forbidden
    bg_vals = channel_raw[bg_region]

    if bg_vals.size == 0:
        return np.nan

    bg_level = float(np.percentile(bg_vals, bg_percentile))
    return bg_level


def mfi_with_global_background(
    cell_mask: np.ndarray,
    channel_mask: np.ndarray,
    channel_raw: np.ndarray,
    global_bg: float,
):
    """
    MFI normalisation using a precomputed per-scene global background.

    - inside = channel-positive pixels within cell_mask
      (fallback: all cell_mask if that would be empty)
    - background = scalar global_bg
    """
    if not np.any(cell_mask) or np.isnan(global_bg):
        return np.nan, global_bg, np.nan

    inside_region = cell_mask & channel_mask
    if not np.any(inside_region):
        inside_region = cell_mask.copy()

    inside_vals = channel_raw[inside_region]
    if inside_vals.size == 0:
        return np.nan, global_bg, np.nan

    mean_in = float(inside_vals.mean())
    mean_bg = float(global_bg)
    ratio   = mean_in / mean_bg if mean_bg > 0 else np.nan
    return mean_in, mean_bg, ratio


def sholl_from_cell_mask(
    cell_mask: np.ndarray,
    r_min: int = 5,
    r_step: int = 5,
    band_width: int = 5,
):
    """
    Sholl analysis using thick annuli on a cleaned, padded cell mask.

    Returns:
        max_r, AUC, Imax, Rmax, CriticalValue, radii, intersections
    """
    sholl_mask = cell_mask.astype(bool).copy()
    sholl_mask = closing(sholl_mask, disk(1))
    sholl_mask = remove_small_holes(sholl_mask, area_threshold=150)

    if not np.any(sholl_mask):
        return 0, 0.0, 0.0, np.nan, np.nan, np.array([]), np.array([])

    lbl = label(sholl_mask)
    props = regionprops(lbl)
    if not props:
        return 0, 0.0, 0.0, np.nan, np.nan, np.array([]), np.array([])

    yc, xc = map(int, props[0].centroid)

    ys, xs = np.nonzero(sholl_mask)
    dists = np.sqrt((ys - yc) ** 2 + (xs - xc) ** 2)
    max_r = int(np.ceil(dists.max())) + 10

    pad = max_r + 5
    sholl_mask_pad = np.pad(
        sholl_mask, pad_width=pad, mode="constant", constant_values=0
    )
    yc_p, xc_p = yc + pad, xc + pad

    yy, xx = np.indices(sholl_mask_pad.shape)
    dist_im = np.sqrt((yy - yc_p) ** 2 + (xx - xc_p) ** 2)

    radii = np.arange(r_min, max_r + 1, r_step)
    intersections = []

    for r in radii:
        r_inner = r - band_width / 2.0
        r_outer = r + band_width / 2.0

        band = (dist_im >= r_inner) & (dist_im <= r_outer)
        band_intersection = band & sholl_mask_pad

        _, n_comp = ndi.label(band_intersection.astype(np.uint8))
        intersections.append(int(n_comp))

    intersections = np.array(intersections, dtype=float)

    if intersections.size > 0:
        AUC = float(trapz(intersections, radii))
        Imax = float(intersections.max())
        Rmax = float(radii[np.argmax(intersections)])
        if np.any(intersections > 0):
            CriticalValue = float(np.mean(radii[intersections > 0]))
        else:
            CriticalValue = np.nan
    else:
        AUC = 0.0
        Imax = 0.0
        Rmax = np.nan
        CriticalValue = np.nan

    return int(max_r), AUC, Imax, Rmax, CriticalValue, radii, intersections


def skeleton_morphology(cell_mask: np.ndarray):
    """
    Basic morphology metrics from a binary cell mask.
    """
    if not np.any(cell_mask):
        return 0, 0, 0, 0.0, np.nan

    cell_area = int(cell_mask.sum())
    hull = convex_hull_image(cell_mask)
    hull_area = int(hull.sum())
    convexity_ratio = hull_area / cell_area if cell_area > 0 else np.nan

    skel = skeletonize(cell_mask)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    nbr_count = ndi.convolve(skel.astype(int), kernel, mode="constant", cval=0)

    endpoints = skel & (nbr_count == 1)
    branch3   = skel & (nbr_count == 3)
    branch4   = skel & (nbr_count == 4)

    num_proc = int(endpoints.sum())
    num_b3   = int(branch3.sum())
    num_b4   = int(branch4.sum())

    segments_only = skel & ~(branch3 | branch4)
    seg_labels = label(segments_only, connectivity=2)
    seg_props = regionprops(seg_labels)
    lengths_px = [p.area for p in seg_props]
    avg_len = float(np.mean(lengths_px)) if lengths_px else 0.0

    return num_proc, num_b3, num_b4, avg_len, convexity_ratio


def save_full_cellpose_overlay(
    base_name: str,
    dapi_raw: np.ndarray,
    pdgfra_raw: np.ndarray,
    mbp_raw: np.ndarray,
    cnp_raw: np.ndarray,
    cell_labels: np.ndarray,
    out_dir: str,
):
    """
    Save a full-scene Cellpose overlay:
      - RGB composite background
      - white boundaries for all Cellpose cells
    """
    Y, X = dapi_raw.shape
    rgb = np.zeros((Y, X, 3), dtype=np.float32)

    rgb[..., 0] = pdgfra_raw          # R
    rgb[..., 1] = mbp_raw + cnp_raw   # G
    rgb[..., 2] = dapi_raw            # B

    for c in range(3):
        chan = rgb[..., c]
        if np.any(chan > 0):
            vmax = np.percentile(chan, 99)
            if vmax > 0:
                rgb[..., c] = np.clip(chan / vmax, 0, 1)

    boundaries = find_boundaries(cell_labels, mode="inner")
    rgb[boundaries, :] = 1.0

    out_img = (rgb * 255).astype(np.uint8)
    out_path = pathlib.Path(out_dir) / f"{base_name}_cellpose_overlay.png"

    plt.figure(figsize=(8, 8))
    plt.imshow(out_img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[QC] wrote cellpose overlay: {out_path}")


# ==========================
# MAIN PROCESSING
# ==========================

def process_lif(
    lif_path: str,
    cfg: Dict[str, str],
    nuc_model,
    cell_model,
    stats_records: List[Dict],
):
    """
    Process a single LIF file and append per-cell stats to stats_records.
    """
    lif_path = pathlib.Path(lif_path)
    print(f"\n=== Processing LIF: {lif_path} ===")

    lif = LifFile(str(lif_path))
    num_scenes = lif.num_images

    tmp_root = pathlib.Path(OUTPUT_DIR) / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for scene_idx in range(num_scenes):
        data, raw_series_name = load_scene_as_cyx_and_name(lif, scene_idx)
        C, Y, X = data.shape
        print(f"  Scene {scene_idx} / {num_scenes-1}: "
              f"name='{raw_series_name}', shape CYX={data.shape}")

        base_name = clean_series_name(lif_path, raw_series_name, scene_idx)

        # 0. Save full raw image (optional)
        full_out = pathlib.Path(FULLMASK_DIR) / f"{base_name}_full_raw.tif"
        tiff.imwrite(
            full_out,
            data.astype(np.float32),
            imagej=True,
            metadata={"axes": "CYX"},
        )

        # 1. Cellpose nuclei on DAPI
        dapi_raw = data[CHAN_DAPI]

        nuc_res = nuc_model.eval(
            dapi_raw,
            diameter=NUC_DIAMETER,
            channels=[0, 0],
            do_3D=False,
        )
        masks_nuc = nuc_res[0]
        dapi_mask = (masks_nuc > 0).astype(np.uint8)

        tiff.imwrite(
            pathlib.Path(QC_DIR) / f"{base_name}_DAPI_nuc_masks.tif",
            masks_nuc.astype(np.uint16),
            imagej=True,
        )

        try:
            # 2. Ilastik marker segmentations (RAW masks: class == 1)
            mbp_raw    = data[CHAN_MBP]
            pdgfra_raw = data[CHAN_PDGFRa]
            cnp_raw    = data[CHAN_CNP]

            # MBP
            mbp_seg_raw = run_ilastik_on_array(
                mbp_raw, cfg["mbp_ilp"], tmp_root, f"{base_name}_MBP"
            )
            mbp_mask = (mbp_seg_raw == 1)

            # PDGFRa
            pdgfra_seg_raw = run_ilastik_on_array(
                pdgfra_raw, cfg["pdgfra_ilp"], tmp_root, f"{base_name}_PDGF"
            )
            pdgfra_mask = (pdgfra_seg_raw == 1)

            # CNP
            cnp_seg_raw = run_ilastik_on_array(
                cnp_raw, cfg["cnp_ilp"], tmp_root, f"{base_name}_CNP"
            )
            cnp_mask = (cnp_seg_raw == 1)

            # Save raw masks
            tiff.imwrite(
                pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_MBP_raw.tif",
                (mbp_mask.astype(np.uint8) * 255),
                imagej=True,
            )
            tiff.imwrite(
                pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_PDGFRa_raw.tif",
                (pdgfra_mask.astype(np.uint8) * 255),
                imagej=True,
            )
            tiff.imwrite(
                pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_CNP_raw.tif",
                (cnp_mask.astype(np.uint8) * 255),
                imagej=True,
            )

            # 3. Cellpose cells on composite (for ROIs)
            c1 = dapi_raw
            c2 = pdgfra_raw
            c3 = mbp_raw
            c4 = cnp_raw

            comp_sum = c4 + c3 + c2
            comp_img = np.stack([c1, comp_sum], axis=-1)

            cell_res = cell_model.eval(
                comp_img,
                diameter=CELL_DIAMETER,
                channels=[2, 1],  # (chan2 = comp_sum, chan1 = DAPI)
                do_3D=False,
            )
            masks_cell = cell_res[0]

            max_label = int(masks_cell.max())
            labels_out = np.clip(masks_cell, 0, 65535).astype(np.uint16)

            tiff.imwrite(
                pathlib.Path(FULLMASK_DIR) / f"{base_name}_cellpose_labels.tif",
                labels_out,
                imagej=True,
            )

            # full-scene Cellpose overlay
            save_full_cellpose_overlay(
                base_name=base_name,
                dapi_raw=dapi_raw,
                pdgfra_raw=pdgfra_raw,
                mbp_raw=mbp_raw,
                cnp_raw=cnp_raw,
                cell_labels=masks_cell,
                out_dir=QC_DIR,
            )

            # 4. global background per channel (per scene)
            mbp_b    = mbp_mask.astype(bool)
            pdgfra_b = pdgfra_mask.astype(bool)
            cnp_b    = cnp_mask.astype(bool)

            marker_union_all = mbp_b | pdgfra_b | cnp_b
            all_cells_mask   = masks_cell > 0

            mbp_bg_global    = compute_global_background(mbp_raw,    all_cells_mask, marker_union_all)
            pdgfra_bg_global = compute_global_background(pdgfra_raw, all_cells_mask, marker_union_all)
            cnp_bg_global    = compute_global_background(cnp_raw,    all_cells_mask, marker_union_all)

            # 5. per-cell stats
            labels_unique = np.unique(masks_cell)
            labels_unique = labels_unique[labels_unique > 0]

            for cid in labels_unique:
                cellpose_roi = (masks_cell == cid)
                cellpose_area = int(cellpose_roi.sum())
                if cellpose_area == 0:
                    continue

                marker_union = marker_union_all & cellpose_roi
                cell_mask = marker_union
                cell_area = int(cell_mask.sum())
                if cell_area < MIN_CELL_AREA_PX:
                    continue

                # nuclei per ROI
                nuc_ids = np.unique(masks_nuc[cellpose_roi])
                nuc_ids = nuc_ids[nuc_ids > 0]
                num_nuc = int(len(nuc_ids))

                # morphology on marker-union mask
                num_proc, num_b3, num_b4, avg_len, convexity_ratio = skeleton_morphology(cell_mask)

                # Sholl on marker-union mask
                max_r, AUC, Imax, Rmax, crit_val, sholl_radii, sholl_inters = sholl_from_cell_mask(
                    cell_mask,
                    r_min=5,
                    r_step=5,
                    band_width=5,
                )

                # marker areas within ROI
                mbp_area    = int((mbp_b & cellpose_roi).sum())
                pdgfra_area = int((pdgfra_b & cellpose_roi).sum())
                cnp_area    = int((cnp_b & cellpose_roi).sum())

                # MFI normalisation with global background
                mbp_mean_in, mbp_mean_bg, mbp_ratio = mfi_with_global_background(
                    cell_mask, mbp_b, mbp_raw, mbp_bg_global
                )
                pdg_mean_in, pdg_mean_bg, pdg_ratio = mfi_with_global_background(
                    cell_mask, pdgfra_b, pdgfra_raw, pdgfra_bg_global
                )
                cnp_mean_in, cnp_mean_bg, cnp_ratio = mfi_with_global_background(
                    cell_mask, cnp_b, cnp_raw, cnp_bg_global
                )

                stats_records.append({
                    "lif_path":   str(lif_path),
                    "lif_name":   lif_path.stem,
                    "scene_idx":  scene_idx,
                    "series_name": raw_series_name,
                    "base_name":  base_name,
                    "cell_id":    int(cid),

                    # areas
                    "cellpose_area_px": cellpose_area,
                    "cell_area_px":     cell_area,
                    "num_nuclei":       num_nuc,
                    "mbp_area_px":      mbp_area,
                    "pdgfra_area_px":   pdgfra_area,
                    "cnp_area_px":      cnp_area,

                    # morphology
                    "num_proc":           num_proc,
                    "num_branch3_px":     num_b3,
                    "num_branch4_px":     num_b4,
                    "avg_segment_len_px": avg_len,
                    "convexity_ratio":    convexity_ratio,

                    # Sholl
                    "max_r_px":       max_r,
                    "AUC":            AUC,
                    "Imax":           Imax,
                    "Rmax_px":        Rmax,
                    "CriticalValue":  crit_val,

                    # MBP
                    "mbp_mean_in":    mbp_mean_in,
                    "mbp_mean_bg":    mbp_mean_bg,
                    "mbp_int_ratio":  mbp_ratio,

                    # PDGFRa
                    "pdg_mean_in":    pdg_mean_in,
                    "pdg_mean_bg":    pdg_mean_bg,
                    "pdg_int_ratio":  pdg_ratio,

                    # CNP
                    "cnp_mean_in":    cnp_mean_in,
                    "cnp_mean_bg":    cnp_mean_bg,
                    "cnp_int_ratio":  cnp_ratio,
                })

                # per-cell QC plotting
                if PLOT_QC_CELLS and QC_PLOT_EVERY > 0 and (cid % QC_PLOT_EVERY == 0):
                    cell_composite = np.stack([
                        pdgfra_raw,
                        mbp_raw + cnp_raw,
                        dapi_raw,
                    ], axis=-1)

                    cell_composite_display = cell_composite.copy()
                    cell_composite_display[~cellpose_roi] = 0

                    skel = skeletonize(cell_mask)
                    hull = convex_hull_image(cell_mask)

                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                    # composite
                    if np.any(cell_composite_display > 0):
                        vmax = np.percentile(
                            cell_composite_display[cell_composite_display > 0], 99
                        )
                    else:
                        vmax = 1.0
                    axes[0].imshow(
                        np.clip(cell_composite_display, 0, vmax).astype(np.float32) / float(vmax)
                    )
                    axes[0].set_title("Cell composite (within ROI)")
                    axes[0].axis("off")

                    # marker-union cell mask
                    axes[1].imshow(cell_mask, cmap="gray")
                    axes[1].set_title("Marker-union cell mask")
                    axes[1].axis("off")

                    # skeleton + hull
                    axes[2].imshow(hull, cmap="gray")
                    axes[2].contour(skel, colors="r", linewidths=0.7)
                    axes[2].set_title("Skeleton (red) on hull")
                    axes[2].axis("off")

                    # Sholl curve
                    if sholl_radii.size > 0:
                        axes[3].plot(sholl_radii, sholl_inters, marker="o")
                        axes[3].invert_xaxis()
                    axes[3].set_xlabel("Radius (px)")
                    axes[3].set_ylabel("Intersections")
                    axes[3].set_title("Sholl")
                    axes[3].grid(alpha=0.3)

                    plt.tight_layout()
                    qc_path = pathlib.Path(QC_DIR) / f"{base_name}_cell{cid}_QC.png"
                    plt.savefig(qc_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)

            # end for each cid

        except Exception as e:
            print(
                f"[ERROR] Skipping scene in {lif_path.name}, scene {scene_idx}, "
                f"base_name={base_name} due to: {e}"
            )
            log_path = pathlib.Path(QC_DIR) / "pipeline_failures.log"
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{lif_path.name}\tScene {scene_idx}\t{base_name}\t{repr(e)}\n")
            except Exception:
                pass

        # --- per-scene cleanup to reduce peak RAM ---
        del data
        del dapi_raw, masks_nuc, dapi_mask
        if "mbp_raw" in locals():
            del mbp_raw
        if "pdgfra_raw" in locals():
            del pdgfra_raw
        if "cnp_raw" in locals():
            del cnp_raw
        if "mbp_mask" in locals():
            del mbp_mask
        if "pdgfra_mask" in locals():
            del pdgfra_mask
        if "cnp_mask" in locals():
            del cnp_mask
        if "masks_cell" in locals():
            del masks_cell
        gc.collect()

    # --- end scenes loop ---

    # try to explicitly close LIF and free associated resources
    try:
        lif.close()
    except Exception:
        pass

    del lif
    gc.collect()


def main():
    print("[cellpose] loading models...")
    nuc_model  = models.CellposeModel(pretrained_model=NUC_MODEL_PATH,  gpu=True)
    cell_model = models.CellposeModel(pretrained_model=CELL_MODEL_PATH, gpu=True)

    any_cells_any_lif = False

    for lif_path, cfg in LIF_CONFIG.items():
        # fresh list for each LIF to avoid huge cumulative memory
        stats_records: List[Dict] = []

        process_lif(lif_path, cfg, nuc_model, cell_model, stats_records)

        if stats_records:
            any_cells_any_lif = True
            df = pd.DataFrame(stats_records)
            lif_stem = pathlib.Path(lif_path).stem
            csv_path = pathlib.Path(CSV_DIR) / f"{lif_stem}_per_cell_stats.csv"
            df.to_csv(csv_path, index=False)
            print(f"[stats] wrote {csv_path}")
        else:
            print(f"[stats] no cells found for {lif_path}")

        # clear list and force GC after each LIF
        del stats_records
        gc.collect()

    # cleanup models too
    del nuc_model, cell_model
    gc.collect()

    if not any_cells_any_lif:
        print("[stats] no cells found in any LIF; no CSVs written.")


if __name__ == "__main__":
    main()
