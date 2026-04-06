#!/usr/bin/env python3
"""
40x 5-channel pPKM2 signalling single-cell analysis
===================================================

Channel order (CYX):
  C1 DAPI   -> index 0
  C2 PDGFRa -> index 1
  C3 O4     -> index 2
  C4 pPKM2  -> index 3
  C5 MBP    -> index 4

Pipeline:
  LIF -> Ilastik (MBP / PDGFRa / O4 / pPKM2) + Cellpose (nuclei + cells)
      -> per-cell morphology + MFI-normalised intensities

Key rule:
  - Cellpose is used ONLY to define ROIs.
  - "Cell mask" for intensities is the marker-union mask intersected with ROI:
      cell_mask_all = (MBP ∪ PDGFRa ∪ pPKM2) ∩ ROI

Background:
  - Per-scene, per-channel GLOBAL background, computed from pixels far from
    any ROI and marker mask (robust median).

IMPORTANT:
  - Ilastik class-1 masks are cleaned via nuclei-seeded reconstruction.
  - All downstream area/intensity/morphology calculations use cleaned masks.

Extra measurement:
  - pPKM2 intensity in the UNION cell mask (cell_mask_all):
      ppkm2_mean_in, ppkm2_mean_bg, ppkm2_int_ratio
  - pPKM2 intensity in the nucleus assigned to each cell:
      ppkm2_nuc_mean_in, ppkm2_nuc_mean_bg, ppkm2_nuc_int_ratio

REFRACTOR FEATURES
-----------------
1) Analyse every toggle:
   - Only process every Nth scene in the LIF (SCENE_STRIDE + SCENE_OFFSET).

2) Writes a CSV every 10 images:
   - Incrementally appends buffered rows to a per-LIF *_partial.csv every
     FLUSH_EVERY_SCENES processed scenes.

3) Diagnostics if it crashes:
   - QC/run.log for progress
   - QC/pipeline_failures.log for failures + full tracebacks
   - faulthandler enabled -> QC/faulthandler.log
   - per-LIF checkpoint JSON to resume from last completed scene

4) MORPHOLOGY ONLY: three versions
   - morph_all_* computed from (MBP ∪ PDGFRa ∪ pPKM2) ∩ ROI
   - morph_mbppdg_* computed from (MBP ∪ PDGFRa) ∩ ROI
   - morph_ppkm2_* computed from (pPKM2) ∩ ROI
   (Intensities remain computed from the ALL-union mask for consistency.)

5) Tie heavy non-mask outputs (full_raw / cellpose overlay)
   to the per-scene QC frequency (QC_SCENE_EVERY) to reduce IO.

MINIMAL CHANGES (per request)
-----------------------------
A) Add per-LIF starting scene override toggle (START_SCENE_BY_LIF).
B) Wipe the existing tmp folder EVERY 5 *processed* scenes (TMP_WIPE_EVERY_SCENES),
   without changing tmp architecture (still OUTPUT_DIR/_tmp).

Requirements:
  pip install readlif tifffile numpy scikit-image cellpose pandas scipy matplotlib
"""

import os
import pathlib
import subprocess
from typing import Dict, List, Tuple
import gc
import re
import sys
import json
import time
import traceback
import logging
import faulthandler

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
    reconstruction,
    skeletonize,
    convex_hull_image,
    closing,
    remove_small_holes,
)
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


# ==========================
# CONFIG
# ==========================

OUTPUT_DIR = r"C:\Users\JackM\5C_signalling\pkm2_python_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FULLMASK_DIR = os.path.join(OUTPUT_DIR, "full_masks")
QC_DIR       = os.path.join(OUTPUT_DIR, "QC")
CSV_DIR      = os.path.join(OUTPUT_DIR, "csvs")
for _d in [FULLMASK_DIR, QC_DIR, CSV_DIR]:
    os.makedirs(_d, exist_ok=True)

RUN_LOG_PATH   = os.path.join(QC_DIR, "run.log")
FAIL_LOG_PATH  = os.path.join(QC_DIR, "pipeline_failures.log")
FAULT_LOG_PATH = os.path.join(QC_DIR, "faulthandler.log")

ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.0.post1\ilastik.exe"

NUC_MODEL_PATH  = r"C:\Users\JackM\5C_signalling\cellpose nuclei\models\nuc1"
CELL_MODEL_PATH = r"C:\Users\JackM\40xMBPCNP\cellpose\models\CP_20250902_074451"

LIF_CONFIG: Dict[str, Dict[str, str]] = {
    r"C:\Users\JackM\5C_signalling\N1_0921_PKM2.lif": {
        "mbp_ilp":    r"C:\Users\JackM\5C_signalling\ilastik\N1_755\N1_755.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\5C_signalling\ilastik\N1_PDGFRa\N1_PDGFRa.ilp",
        "o4_ilp":     r"C:\Users\JackM\5C_signalling\ilastik\N1_O4\N1_O4.ilp",
        "ppkm2_ilp":  r"C:\Users\JackM\5C_signalling\ilastik\N1_PKM2\N1_PKM2.ilp",  # TODO: point to pPKM2 ilastik project
    },
    r"C:\Users\JackM\5C_signalling\N4_0929_PKM2.lif": {
        "mbp_ilp":    r"C:\Users\JackM\5C_signalling\ilastik\N4_755\N4_755.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\5C_signalling\ilastik\N4_PDGFRa\N4_PDGFRa.ilp",
        "o4_ilp":     r"C:\Users\JackM\5C_signalling\ilastik\N4_O4\N4_O4.ilp",
        "ppkm2_ilp":  r"C:\Users\JackM\5C_signalling\ilastik\N4_PKM2\N4_PKM2.ilp",  # TODO: point to pPKM2 ilastik project
    },
    r"E:\N6_1013_PKM2.lif": {
        "mbp_ilp":    r"C:\Users\JackM\5C_signalling\ilastik\N6_755\N6_755.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\5C_signalling\ilastik\N6_PDGFRa\N6_PDGFRa.ilp",
        "o4_ilp":     r"C:\Users\JackM\5C_signalling\ilastik\N6_O4\N6_O4.ilp",
        "ppkm2_ilp":  r"C:\Users\JackM\5C_signalling\ilastik\N6_PKM2\N6_PKM2.ilp",  # TODO: point to pPKM2 ilastik project
    },
    r"E:\N7_20251020_PKM2.lif": {
        "mbp_ilp":    r"C:\Users\JackM\5C_signalling\ilastik\N7_755\N7_755.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\5C_signalling\ilastik\N7_PDGFRa\N7_PDGFRa.ilp",
        "o4_ilp":     r"C:\Users\JackM\5C_signalling\ilastik\N7_O4\N7_O4.ilp",
        "ppkm2_ilp":  r"C:\Users\JackM\5C_signalling\ilastik\N7_PKM2\N7_PKM2.ilp",  # TODO: point to pPKM2 ilastik project
    }, 

    r"E:\N9_20251512_PKM2.lif": {
        "mbp_ilp":    r"C:\Users\JackM\5C_signalling\ilastik\N9_755\N9_755.ilp",
        "pdgfra_ilp": r"C:\Users\JackM\5C_signalling\ilastik\N9_PDGFRa\N9_PDGFRa.ilp",
        "o4_ilp":     r"C:\Users\JackM\5C_signalling\ilastik\N9_O4\N9_O4.ilp",
        "ppkm2_ilp":  r"C:\Users\JackM\5C_signalling\ilastik\N9_PKM2\N9_PKM2.ilp",  # TODO: point to pPKM2 ilastik project
    },
}

# --- MINIMAL CHANGE A: per-LIF starting scene override (0-based scene_idx) ---
# If a LIF is not listed here, it defaults to None and uses checkpoint logic.
START_SCENE_BY_LIF: Dict[str, int] = {
    # r"C:\Users\JackM\5C_signalling\ppkm2\N1_0921_pPKM2.lif": 195,
    # r"C:\Users\JackM\5C_signalling\ppkm2\N4_0929_pPKM2.lif": 0,
}
USE_START_SCENE_OVERRIDE = True

# --- MINIMAL CHANGE B: wipe tmp every N processed scenes ---
TMP_WIPE_EVERY_SCENES = 5  # set 0 to disable

CHAN_DAPI   = 0
CHAN_PDGFRa = 1
CHAN_O4     = 2
CHAN_pPKM2  = 3
CHAN_MBP    = 4

# Cellpose params
NUC_DIAMETER  = 70.0
CELL_DIAMETER = 180.0
NUC_MIN_OVERLAP = 0.8  # nucleus->cell assignment requires >= 80% overlap with best-matching cell

# Filtering thresholds
MIN_CELL_AREA_PX = 2000  # threshold applied to ALL-union mask area (cell_mask_all)

# pPKM2 background (strict mode helps avoid condition-signal leakage into "background")
PPKM2_BG_MODE = "strict"  # "legacy" or "strict"
PPKM2_BG_CELL_MARGIN_PX = 20
PPKM2_BG_MARKER_MARGIN_PX = 12
PPKM2_BG_INTENSITY_PERCENTILE = 90.0
PPKM2_BG_INTENSITY_MARGIN_PX = 2
PPKM2_BG_PERCENTILE = 20.0

# Per-cell QC plots (lightweight)
PLOT_QC_CELLS = True
QC_PLOT_EVERY = 10  # every Nth cell_id (modulo) gets a QC panel

# --- Analyse every toggle (per-scene) ---
SCENE_STRIDE = 2
SCENE_OFFSET = 0

# --- Incremental CSV flushing ---
FLUSH_EVERY_SCENES = 10
APPEND_PARTIAL_CSV = True
WRITE_FINAL_CSV    = True

# --- Checkpoint/resume ---
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Heavy per-scene outputs ---
QC_SCENE_EVERY = 10  # save overlay every N processed scenes
SAVE_FULL_RAW_SCENE = False  # keep OFF: do not write full raw TIFF scene stacks
SAVE_MASKS_EVERY_SCENE = True  # write mask files for every processed scene


# ==========================
# LOGGING + CRASH DIAGNOSTICS
# ==========================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ppkm2_pipeline")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(RUN_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        fault_f = open(FAULT_LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(file=fault_f, all_threads=True)
    except Exception:
        pass

    return logger


LOGGER = setup_logging()


# ==========================
# SAFETY CHECK
# ==========================

def _extract_N_prefix(path_str: str) -> str:
    s = pathlib.Path(path_str).stem
    m = re.search(r"(N\d+)", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def safety_check_or_die():
    problems = []

    if not os.path.exists(ILASTIK_EXE):
        problems.append(f"ILASTIK_EXE missing: {ILASTIK_EXE}")
    if not os.path.exists(NUC_MODEL_PATH):
        problems.append(f"NUC_MODEL_PATH missing: {NUC_MODEL_PATH}")
    if not os.path.exists(CELL_MODEL_PATH):
        problems.append(f"CELL_MODEL_PATH missing: {CELL_MODEL_PATH}")
    if not LIF_CONFIG:
        problems.append("LIF_CONFIG is empty: add your LIF + ilastik .ilp paths.")

    for lif_path, cfg in LIF_CONFIG.items():
        if not os.path.exists(lif_path):
            problems.append(f"LIF missing: {lif_path}")

        for key in ("mbp_ilp", "pdgfra_ilp", "o4_ilp", "ppkm2_ilp"):
            if key not in cfg:
                problems.append(f"Missing key '{key}' in config for LIF: {lif_path}")

        for k, ilp_path in cfg.items():
            if not os.path.exists(ilp_path):
                problems.append(f"ILP missing ({k}): {ilp_path}")

        lif_N = _extract_N_prefix(lif_path)
        if lif_N:
            for k, ilp_path in cfg.items():
                ilp_parent = pathlib.Path(ilp_path).parent.name
                ilp_stem   = pathlib.Path(ilp_path).stem
                combo = f"{ilp_parent}_{ilp_stem}"
                ilp_N = _extract_N_prefix(combo)
                if ilp_N and ilp_N != lif_N:
                    problems.append(
                        f"N-prefix mismatch for {lif_path} ({lif_N}) vs {k} ilp ({ilp_N}): {ilp_path}"
                    )

    if SCENE_STRIDE < 1:
        problems.append(f"SCENE_STRIDE must be >= 1 (got {SCENE_STRIDE})")
    if not (0 <= SCENE_OFFSET < max(SCENE_STRIDE, 1)):
        problems.append(f"SCENE_OFFSET must be in [0, SCENE_STRIDE-1] (got {SCENE_OFFSET})")
    if FLUSH_EVERY_SCENES < 1:
        problems.append(f"FLUSH_EVERY_SCENES must be >= 1 (got {FLUSH_EVERY_SCENES})")
    if QC_SCENE_EVERY < 1:
        problems.append(f"QC_SCENE_EVERY must be >= 1 (got {QC_SCENE_EVERY})")

    if problems:
        LOGGER.error("\n================ SAFETY CHECK FAILED ================\n")
        for p in problems:
            LOGGER.error(f" - {p}")
        LOGGER.error("\nFix the above before running.\n")
        raise RuntimeError("Safety check failed.")
    else:
        LOGGER.info("[OK] Safety check passed: paths look consistent.")


# ==========================
# CHECKPOINTS
# ==========================

def checkpoint_path_for_lif(lif_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(CHECKPOINT_DIR) / f"{lif_path.stem}_checkpoint.json"


def load_checkpoint(lif_path: pathlib.Path) -> Dict:
    cp_path = checkpoint_path_for_lif(lif_path)
    if not cp_path.exists():
        return {"last_completed_scene_idx": -1, "updated_at": None}
    try:
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_completed_scene_idx": -1, "updated_at": None}


def save_checkpoint(lif_path: pathlib.Path, last_completed_scene_idx: int):
    cp_path = checkpoint_path_for_lif(lif_path)
    payload = {
        "lif": str(lif_path),
        "last_completed_scene_idx": int(last_completed_scene_idx),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        LOGGER.warning(f"[checkpoint] failed to write checkpoint {cp_path}: {e}")


# ==========================
# FAILURE LOGGING
# ==========================

def log_failure(lif_path: pathlib.Path, scene_idx: int, base_name: str, exc: BaseException):
    tb = traceback.format_exc()
    msg = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"{lif_path.name}\tScene {scene_idx}\t{base_name}\t{repr(exc)}\n"
        f"{tb}\n"
        f"{'-'*80}\n"
    )
    try:
        with open(FAIL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass


# ==========================
# HELPERS
# ==========================

def _wipe_tmp_dir_contents(tmp_root: pathlib.Path):
    """
    MINIMAL CHANGE B:
    Wipe tmp_root contents (keep tmp_root path/architecture). Called every N processed scenes.
    """
    try:
        if not tmp_root.exists():
            tmp_root.mkdir(parents=True, exist_ok=True)
            return
        for p in tmp_root.glob("*"):
            try:
                if p.is_dir():
                    # rarely used here, but safe
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
            except Exception:
                pass
        LOGGER.info(f"[tmp] wiped tmp contents: {tmp_root}")
    except Exception as e:
        LOGGER.warning(f"[tmp] failed to wipe tmp contents: {e}")


def run_ilastik_on_array(
    img: np.ndarray,
    ilp_path: str,
    tmp_root: pathlib.Path,
    tag: str,
) -> np.ndarray:
    tmp_root.mkdir(parents=True, exist_ok=True)

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

    LOGGER.info(f"[ilastik] {tag}: running Ilastik...")
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

    try:
        tmp_in.unlink(missing_ok=True)
    except Exception:
        pass

    return seg.astype(np.uint8)


def clean_mask_with_nuclei_seed(mask: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
    """
    Match the nuclei-seeded reconstruction approach used in full_pipeline_batch_v1.py.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if nuclei_labels is None:
        return mask_bool.copy()

    seed = (np.asarray(nuclei_labels) > 0) & mask_bool
    if not np.any(seed):
        return mask_bool.copy()

    recon = reconstruction(seed.astype(np.uint8), mask_bool.astype(np.uint8), method="dilation")
    return recon > 0


def load_scene_as_cyx_and_name(lif: LifFile, scene_idx: int) -> Tuple[np.ndarray, str]:
    lif_img = lif.get_image(scene_idx)
    raw_name = str(lif_img.name)

    chans = []
    for ch_img in lif_img.get_iter_c(z=0, t=0, m=0):
        chans.append(np.array(ch_img))

    data = np.stack(chans, axis=0).astype(np.float64)
    return data, raw_name


def clean_series_name(lif_path: pathlib.Path, raw_series_name: str, scene_idx: int) -> str:
    parts = raw_series_name.split(":", 1)
    series_desc = parts[1].strip() if len(parts) > 1 else raw_series_name.strip()

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
    forbidden = all_cells_mask | all_marker_union
    forbidden = binary_dilation(forbidden, disk(margin_px))

    bg_region = ~forbidden
    bg_vals = channel_raw[bg_region]

    if bg_vals.size == 0:
        return np.nan

    return float(np.percentile(bg_vals, bg_percentile))


def compute_ppkm2_background_strict(
    ppkm2_raw: np.ndarray,
    all_cells_mask: np.ndarray,
    mbp_raw_mask: np.ndarray,
    pdgfra_raw_mask: np.ndarray,
    o4_raw_mask: np.ndarray,
    ppkm2_raw_mask: np.ndarray,
    cell_margin_px: int = 20,
    marker_margin_px: int = 12,
    bright_percentile: float = 90.0,
    bright_margin_px: int = 2,
    bg_percentile: float = 20.0,
) -> float:
    """
    More conservative pPKM2 background:
      1) Exclude dilated cell ROIs.
      2) Exclude dilated raw marker masks (all channels).
      3) Exclude top-intensity pPKM2 pixels (diffuse bright haze) + small dilation.
      4) Estimate background from a lower percentile of what's left.
    """
    h, w = ppkm2_raw.shape
    forbidden = np.zeros((h, w), dtype=bool)

    cell_margin_px = max(0, int(cell_margin_px))
    marker_margin_px = max(0, int(marker_margin_px))
    bright_margin_px = max(0, int(bright_margin_px))
    bright_percentile = float(np.clip(bright_percentile, 0.0, 100.0))
    bg_percentile = float(np.clip(bg_percentile, 0.0, 100.0))

    cells = np.asarray(all_cells_mask, dtype=bool)
    if cell_margin_px > 0:
        cells = binary_dilation(cells, disk(cell_margin_px))
    forbidden |= cells

    marker_union_raw = (
        np.asarray(mbp_raw_mask, dtype=bool)
        | np.asarray(pdgfra_raw_mask, dtype=bool)
        | np.asarray(o4_raw_mask, dtype=bool)
        | np.asarray(ppkm2_raw_mask, dtype=bool)
    )
    if marker_margin_px > 0:
        marker_union_raw = binary_dilation(marker_union_raw, disk(marker_margin_px))
    forbidden |= marker_union_raw

    finite_vals = ppkm2_raw[np.isfinite(ppkm2_raw)]
    if finite_vals.size == 0:
        return np.nan

    bright_thr = float(np.percentile(finite_vals, bright_percentile))
    bright = ppkm2_raw >= bright_thr
    if bright_margin_px > 0:
        bright = binary_dilation(bright, disk(bright_margin_px))
    forbidden |= bright

    bg_region = ~forbidden
    bg_vals = ppkm2_raw[bg_region]
    if bg_vals.size == 0:
        return np.nan

    return float(np.percentile(bg_vals, bg_percentile))


def mfi_with_global_background(
    cell_mask: np.ndarray,
    channel_mask: np.ndarray,
    channel_raw: np.ndarray,
    global_bg: float,
):
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


def assign_nuclei_to_cells(
    masks_nuc: np.ndarray,
    masks_cell: np.ndarray,
    min_overlap: float = 0.8,
) -> Dict[int, List[int]]:
    """
    Assign each nucleus to the single cell with the largest overlap,
    only if overlap fraction >= min_overlap.
    Returns mapping: cell_id -> list of nucleus labels.
    """
    cell_to_nucs: Dict[int, List[int]] = {}
    for prop in regionprops(masks_nuc):
        if prop.area <= 0:
            continue
        coords = prop.coords
        cell_ids = masks_cell[coords[:, 0], coords[:, 1]]
        if cell_ids.size == 0:
            continue
        vals, counts = np.unique(cell_ids, return_counts=True)
        nonzero = vals > 0
        if not np.any(nonzero):
            continue
        vals = vals[nonzero]
        counts = counts[nonzero]
        idx = int(np.argmax(counts))
        best_cell = int(vals[idx])
        overlap = float(counts[idx]) / float(prop.area)
        if overlap >= min_overlap:
            cell_to_nucs.setdefault(best_cell, []).append(int(prop.label))
    return cell_to_nucs


def sholl_from_cell_mask(
    cell_mask: np.ndarray,
    r_min: int = 5,
    r_step: int = 5,
    band_width: int = 5,
):
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
    sholl_mask_pad = np.pad(sholl_mask, pad_width=pad, mode="constant", constant_values=0)
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
        CriticalValue = float(np.mean(radii[intersections > 0])) if np.any(intersections > 0) else np.nan
    else:
        AUC = 0.0
        Imax = 0.0
        Rmax = np.nan
        CriticalValue = np.nan

    return int(max_r), AUC, Imax, Rmax, CriticalValue, radii, intersections


def skeleton_morphology(cell_mask: np.ndarray):
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
    o4_raw: np.ndarray,
    mbp_raw: np.ndarray,
    cell_labels: np.ndarray,
    out_dir: str,
):
    Y, X = dapi_raw.shape
    rgb = np.zeros((Y, X, 3), dtype=np.float32)

    rgb[..., 0] = pdgfra_raw
    rgb[..., 1] = mbp_raw + o4_raw
    rgb[..., 2] = dapi_raw

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

    LOGGER.info(f"[QC] wrote cellpose overlay: {out_path}")


# ==========================
# INCREMENTAL CSV WRITING
# ==========================

def append_records_to_csv(csv_path: pathlib.Path, records: List[Dict]):
    if not records:
        return
    df = pd.DataFrame(records)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, index=False, mode="a", header=write_header)


# ==========================
# MAIN PROCESSING
# ==========================

def scene_should_process(scene_idx: int) -> bool:
    return (scene_idx % SCENE_STRIDE) == SCENE_OFFSET


def process_lif(
    lif_path: str,
    cfg: Dict[str, str],
    nuc_model,
    cell_model,
):
    lif_path = pathlib.Path(lif_path)
    LOGGER.info(f"\n=== Processing LIF: {lif_path} ===")

    lif = LifFile(str(lif_path))
    num_scenes = lif.num_images

    tmp_root = pathlib.Path(OUTPUT_DIR) / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    lif_stem = lif_path.stem
    partial_csv_path = pathlib.Path(CSV_DIR) / f"{lif_stem}_per_cell_stats_partial.csv"
    final_csv_path   = pathlib.Path(CSV_DIR) / f"{lif_stem}_per_cell_stats.csv"

    # resume / start-scene override (MINIMAL CHANGE A)
    start_scene = 196
    if USE_START_SCENE_OVERRIDE:
        override = START_SCENE_BY_LIF.get(str(lif_path), None)
        if override is not None:
            start_scene = max(0, int(override))
            LOGGER.info(f"[start] override found -> starting at scene {start_scene} (0-based) for {lif_path.name}")
        elif RESUME_FROM_CHECKPOINT:
            cp = load_checkpoint(lif_path)
            last_done = int(cp.get("last_completed_scene_idx", -1))
            start_scene = max(0, last_done + 1)
            if start_scene > 0:
                LOGGER.info(f"[resume] checkpoint found: last_completed_scene_idx={last_done} -> starting at scene {start_scene}")
    else:
        if RESUME_FROM_CHECKPOINT:
            cp = load_checkpoint(lif_path)
            last_done = int(cp.get("last_completed_scene_idx", -1))
            start_scene = max(0, last_done + 1)
            if start_scene > 0:
                LOGGER.info(f"[resume] checkpoint found: last_completed_scene_idx={last_done} -> starting at scene {start_scene}")

    buffered_records: List[Dict] = []
    processed_scene_counter = 0
    any_cells = False

    for scene_idx in range(start_scene, num_scenes):
        if not scene_should_process(scene_idx):
            continue

        processed_scene_counter += 1

        # MINIMAL CHANGE B: wipe tmp every N processed scenes (before doing work for this scene)
        if TMP_WIPE_EVERY_SCENES and TMP_WIPE_EVERY_SCENES > 0:
            if processed_scene_counter > 1 and ((processed_scene_counter - 1) % TMP_WIPE_EVERY_SCENES == 0):
                _wipe_tmp_dir_contents(tmp_root)

        save_scene_qc = (QC_SCENE_EVERY > 0) and (processed_scene_counter % QC_SCENE_EVERY == 0)
        save_scene_masks = SAVE_MASKS_EVERY_SCENE or save_scene_qc

        t0 = time.time()
        base_name = f"{lif_stem}_S{scene_idx+1}"

        try:
            data, raw_series_name = load_scene_as_cyx_and_name(lif, scene_idx)
            base_name = clean_series_name(lif_path, raw_series_name, scene_idx)

            LOGGER.info(
                f"Scene {scene_idx}/{num_scenes-1} | processed#{processed_scene_counter} | "
                f"saveQC={save_scene_qc} | saveMasks={save_scene_masks} | {base_name} | CYX={data.shape}"
            )

            if SAVE_FULL_RAW_SCENE and save_scene_qc:
                full_out = pathlib.Path(FULLMASK_DIR) / f"{base_name}_full_raw.tif"
                tiff.imwrite(
                    full_out,
                    data.astype(np.float32),
                    imagej=True,
                    metadata={"axes": "CYX"},
                )

            # 1) nuclei (DAPI)
            dapi_raw = data[CHAN_DAPI]
            nuc_res = nuc_model.eval(
                dapi_raw,
                diameter=NUC_DIAMETER,
                channels=[0, 0],
                do_3D=False,
            )
            masks_nuc = np.asarray(nuc_res[0])

            if save_scene_masks:
                nuc_labels_out = np.clip(masks_nuc, 0, 65535).astype(np.uint16)
                tiff.imwrite(
                    pathlib.Path(QC_DIR) / f"{base_name}_DAPI_nuc_masks.tif",
                    nuc_labels_out,
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_nuc_cellpose_labels.tif",
                    nuc_labels_out,
                    imagej=True,
                )

            # 2) Ilastik masks (RAW class==1)
            pdgfra_raw = data[CHAN_PDGFRa]
            o4_raw     = data[CHAN_O4]
            ppkm2_raw  = data[CHAN_pPKM2]
            mbp_raw    = data[CHAN_MBP]

            mbp_seg_raw = run_ilastik_on_array(mbp_raw, cfg["mbp_ilp"], tmp_root, f"{base_name}_MBP")
            mbp_mask_raw = (mbp_seg_raw == 1)
            mbp_mask = clean_mask_with_nuclei_seed(mbp_mask_raw, masks_nuc)

            pdgfra_seg_raw = run_ilastik_on_array(pdgfra_raw, cfg["pdgfra_ilp"], tmp_root, f"{base_name}_PDGF")
            pdgfra_mask_raw = (pdgfra_seg_raw == 1)
            pdgfra_mask = clean_mask_with_nuclei_seed(pdgfra_mask_raw, masks_nuc)

            o4_seg_raw = run_ilastik_on_array(o4_raw, cfg["o4_ilp"], tmp_root, f"{base_name}_O4")
            o4_mask_raw = (o4_seg_raw == 1)
            o4_mask = clean_mask_with_nuclei_seed(o4_mask_raw, masks_nuc)

            ppkm2_seg_raw = run_ilastik_on_array(ppkm2_raw, cfg["ppkm2_ilp"], tmp_root, f"{base_name}_pPKM2")
            ppkm2_mask_raw = (ppkm2_seg_raw == 1)
            ppkm2_mask = clean_mask_with_nuclei_seed(ppkm2_mask_raw, masks_nuc)

            if save_scene_masks:
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_MBP_raw.tif",
                    (mbp_mask_raw.astype(np.uint8) * 255),
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_PDGFRa_raw.tif",
                    (pdgfra_mask_raw.astype(np.uint8) * 255),
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_O4_raw.tif",
                    (o4_mask_raw.astype(np.uint8) * 255),
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_pPKM2_raw.tif",
                    (ppkm2_mask_raw.astype(np.uint8) * 255),
                    imagej=True,
                )

                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_MBP_clean.tif",
                    (mbp_mask.astype(np.uint8) * 255),
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_PDGFRa_clean.tif",
                    (pdgfra_mask.astype(np.uint8) * 255),
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_O4_clean.tif",
                    (o4_mask.astype(np.uint8) * 255),
                    imagej=True,
                )
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_mask_pPKM2_clean.tif",
                    (ppkm2_mask.astype(np.uint8) * 255),
                    imagej=True,
                )

            # 3) Cellpose ROIs on composite
            comp_sum = pdgfra_raw + o4_raw + mbp_raw
            comp_img = np.stack([dapi_raw, comp_sum], axis=-1)

            cell_res = cell_model.eval(
                comp_img,
                diameter=CELL_DIAMETER,
                channels=[2, 1],
                do_3D=False,
            )
            masks_cell = cell_res[0]

            if save_scene_masks:
                labels_out = np.clip(masks_cell, 0, 65535).astype(np.uint16)
                tiff.imwrite(
                    pathlib.Path(FULLMASK_DIR) / f"{base_name}_cellpose_labels.tif",
                    labels_out,
                    imagej=True,
                )

            if save_scene_qc:
                save_full_cellpose_overlay(
                    base_name=base_name,
                    dapi_raw=dapi_raw,
                    pdgfra_raw=pdgfra_raw,
                    o4_raw=o4_raw,
                    mbp_raw=mbp_raw,
                    cell_labels=masks_cell,
                    out_dir=QC_DIR,
                )

            # 4) Global backgrounds
            mbp_b    = mbp_mask.astype(bool)
            pdgfra_b = pdgfra_mask.astype(bool)
            o4_b     = o4_mask.astype(bool)
            ppkm2_b  = ppkm2_mask.astype(bool)
            mbp_raw_b    = mbp_mask_raw.astype(bool)
            pdgfra_raw_b = pdgfra_mask_raw.astype(bool)
            o4_raw_b     = o4_mask_raw.astype(bool)
            ppkm2_raw_b  = ppkm2_mask_raw.astype(bool)

            marker_union_all    = mbp_b | pdgfra_b | ppkm2_b
            marker_union_mbppdg = mbp_b | pdgfra_b
            marker_union_ppkm2  = ppkm2_b
            marker_union_bg     = marker_union_all | o4_b

            all_cells_mask = masks_cell > 0

            mbp_bg_global    = compute_global_background(mbp_raw,    all_cells_mask, marker_union_bg)
            pdgfra_bg_global = compute_global_background(pdgfra_raw, all_cells_mask, marker_union_bg)
            o4_bg_global     = compute_global_background(o4_raw,     all_cells_mask, marker_union_bg)
            if str(PPKM2_BG_MODE).strip().lower() == "strict":
                ppkm2_bg_global = compute_ppkm2_background_strict(
                    ppkm2_raw=ppkm2_raw,
                    all_cells_mask=all_cells_mask,
                    mbp_raw_mask=mbp_raw_b,
                    pdgfra_raw_mask=pdgfra_raw_b,
                    o4_raw_mask=o4_raw_b,
                    ppkm2_raw_mask=ppkm2_raw_b,
                    cell_margin_px=PPKM2_BG_CELL_MARGIN_PX,
                    marker_margin_px=PPKM2_BG_MARKER_MARGIN_PX,
                    bright_percentile=PPKM2_BG_INTENSITY_PERCENTILE,
                    bright_margin_px=PPKM2_BG_INTENSITY_MARGIN_PX,
                    bg_percentile=PPKM2_BG_PERCENTILE,
                )
            else:
                ppkm2_bg_global = compute_global_background(ppkm2_raw, all_cells_mask, marker_union_bg)

            # 5) Per-cell stats
            labels_unique = np.unique(masks_cell)
            labels_unique = labels_unique[labels_unique > 0]
            cell_to_nucs = assign_nuclei_to_cells(
                masks_nuc=masks_nuc,
                masks_cell=masks_cell,
                min_overlap=NUC_MIN_OVERLAP,
            )

            for cid in labels_unique:
                cellpose_roi = (masks_cell == cid)
                cellpose_area = int(cellpose_roi.sum())
                if cellpose_area == 0:
                    continue

                # intensity mask (ALL union)
                cell_mask_all = marker_union_all & cellpose_roi
                cell_area_all = int(cell_mask_all.sum())
                if cell_area_all < MIN_CELL_AREA_PX:
                    continue

                # morphology-only second mask (MBP+PDGFRa)
                cell_mask_mbppdg = marker_union_mbppdg & cellpose_roi
                cell_area_mbppdg = int(cell_mask_mbppdg.sum())

                # morphology-only third mask (pPKM2 only)
                cell_mask_ppkm2 = marker_union_ppkm2 & cellpose_roi
                cell_area_ppkm2 = int(cell_mask_ppkm2.sum())

                # nuclei per ROI
                nuc_ids = cell_to_nucs.get(int(cid), [])
                num_nuc = int(len(nuc_ids))
                nuc_in_cell = np.isin(masks_nuc, nuc_ids)

                # marker areas within ROI (cleaned channel masks)
                mbp_area    = int((mbp_b & cellpose_roi).sum())
                pdgfra_area = int((pdgfra_b & cellpose_roi).sum())
                o4_area     = int((o4_b & cellpose_roi).sum())
                ppkm2_area  = int((ppkm2_b & cellpose_roi).sum())

                # intensities (use ALL union cell mask)
                mbp_mean_in, mbp_mean_bg, mbp_ratio = mfi_with_global_background(
                    cell_mask_all, mbp_b, mbp_raw, mbp_bg_global
                )
                pdg_mean_in, pdg_mean_bg, pdg_ratio = mfi_with_global_background(
                    cell_mask_all, pdgfra_b, pdgfra_raw, pdgfra_bg_global
                )
                o4_mean_in, o4_mean_bg, o4_ratio = mfi_with_global_background(
                    cell_mask_all, o4_b, o4_raw, o4_bg_global
                )
                ppkm2_mean_in, ppkm2_mean_bg, ppkm2_ratio = mfi_with_global_background(
                    cell_mask_all, cell_mask_all, ppkm2_raw, ppkm2_bg_global
                )
                ppkm2_nuc_mean_in, ppkm2_nuc_mean_bg, ppkm2_nuc_ratio = mfi_with_global_background(
                    nuc_in_cell, nuc_in_cell, ppkm2_raw, ppkm2_bg_global
                )

                # morphology A (ALL union)
                num_proc_all, num_b3_all, num_b4_all, avg_len_all, convexity_all = skeleton_morphology(cell_mask_all)
                max_r_all, AUC_all, Imax_all, Rmax_all, crit_all, sholl_r_all, sholl_i_all = sholl_from_cell_mask(
                    cell_mask_all, r_min=5, r_step=5, band_width=5
                )

                # morphology B (MBP+PDGFRa union)
                num_proc_mbppdg, num_b3_mbppdg, num_b4_mbppdg, avg_len_mbppdg, convexity_mbppdg = skeleton_morphology(cell_mask_mbppdg)
                max_r_mbppdg, AUC_mbppdg, Imax_mbppdg, Rmax_mbppdg, crit_mbppdg, sholl_r_mbppdg, sholl_i_mbppdg = sholl_from_cell_mask(
                    cell_mask_mbppdg, r_min=5, r_step=5, band_width=5
                )

                # morphology C (pPKM2-only mask)
                num_proc_ppkm2, num_b3_ppkm2, num_b4_ppkm2, avg_len_ppkm2, convexity_ppkm2 = skeleton_morphology(cell_mask_ppkm2)
                max_r_ppkm2, AUC_ppkm2, Imax_ppkm2, Rmax_ppkm2, crit_ppkm2, sholl_r_ppkm2, sholl_i_ppkm2 = sholl_from_cell_mask(
                    cell_mask_ppkm2, r_min=5, r_step=5, band_width=5
                )

                buffered_records.append({
                    "lif_path":    str(lif_path),
                    "lif_name":    lif_path.stem,
                    "scene_idx":   int(scene_idx),
                    "series_name": raw_series_name,
                    "base_name":   base_name,
                    "cell_id":     int(cid),

                    # areas
                    "cellpose_area_px": int(cellpose_area),
                    "cell_area_px":     int(cell_area_all),
                    "cell_area_mbppdg_px": int(cell_area_mbppdg),
                    "cell_area_ppkm2_px": int(cell_area_ppkm2),
                    "num_nuclei":       int(num_nuc),
                    "mbp_area_px":      int(mbp_area),
                    "pdgfra_area_px":   int(pdgfra_area),
                    "o4_area_px":       int(o4_area),
                    "ppkm2_area_px":    int(ppkm2_area),

                    # MORPHOLOGY (ALL union)
                    "morph_all_num_proc":           int(num_proc_all),
                    "morph_all_num_branch3_px":     int(num_b3_all),
                    "morph_all_num_branch4_px":     int(num_b4_all),
                    "morph_all_avg_segment_len_px": float(avg_len_all),
                    "morph_all_convexity_ratio":    float(convexity_all),

                    # SHOLL (ALL union)
                    "morph_all_max_r_px":      int(max_r_all),
                    "morph_all_AUC":           float(AUC_all),
                    "morph_all_Imax":          float(Imax_all),
                    "morph_all_Rmax_px":       float(Rmax_all) if not np.isnan(Rmax_all) else np.nan,
                    "morph_all_CriticalValue": float(crit_all) if not np.isnan(crit_all) else np.nan,

                    # MORPHOLOGY (MBP+PDGFRa union)
                    "morph_mbppdg_num_proc":           int(num_proc_mbppdg),
                    "morph_mbppdg_num_branch3_px":     int(num_b3_mbppdg),
                    "morph_mbppdg_num_branch4_px":     int(num_b4_mbppdg),
                    "morph_mbppdg_avg_segment_len_px": float(avg_len_mbppdg),
                    "morph_mbppdg_convexity_ratio":    float(convexity_mbppdg),

                    # SHOLL (MBP+PDGFRa union)
                    "morph_mbppdg_max_r_px":      int(max_r_mbppdg),
                    "morph_mbppdg_AUC":           float(AUC_mbppdg),
                    "morph_mbppdg_Imax":          float(Imax_mbppdg),
                    "morph_mbppdg_Rmax_px":       float(Rmax_mbppdg) if not np.isnan(Rmax_mbppdg) else np.nan,
                    "morph_mbppdg_CriticalValue": float(crit_mbppdg) if not np.isnan(crit_mbppdg) else np.nan,

                    # MORPHOLOGY (pPKM2-only mask)
                    "morph_ppkm2_num_proc":           int(num_proc_ppkm2),
                    "morph_ppkm2_num_branch3_px":     int(num_b3_ppkm2),
                    "morph_ppkm2_num_branch4_px":     int(num_b4_ppkm2),
                    "morph_ppkm2_avg_segment_len_px": float(avg_len_ppkm2),
                    "morph_ppkm2_convexity_ratio":    float(convexity_ppkm2),

                    # SHOLL (pPKM2-only mask)
                    "morph_ppkm2_max_r_px":      int(max_r_ppkm2),
                    "morph_ppkm2_AUC":           float(AUC_ppkm2),
                    "morph_ppkm2_Imax":          float(Imax_ppkm2),
                    "morph_ppkm2_Rmax_px":       float(Rmax_ppkm2) if not np.isnan(Rmax_ppkm2) else np.nan,
                    "morph_ppkm2_CriticalValue": float(crit_ppkm2) if not np.isnan(crit_ppkm2) else np.nan,

                    # MBP intensity
                    "mbp_mean_in":   mbp_mean_in,
                    "mbp_mean_bg":   mbp_mean_bg,
                    "mbp_int_ratio": mbp_ratio,

                    # PDGFRa intensity
                    "pdg_mean_in":   pdg_mean_in,
                    "pdg_mean_bg":   pdg_mean_bg,
                    "pdg_int_ratio": pdg_ratio,

                    # O4 intensity
                    "o4_mean_in":    o4_mean_in,
                    "o4_mean_bg":    o4_mean_bg,
                    "o4_int_ratio":  o4_ratio,

                    # pPKM2 intensity (in ALL union cell mask)
                    "ppkm2_mean_in":   ppkm2_mean_in,
                    "ppkm2_mean_bg":   ppkm2_mean_bg,
                    "ppkm2_int_ratio": ppkm2_ratio,
                    "ppkm2_bg_mode":   str(PPKM2_BG_MODE),

                    # pPKM2 intensity (nucleus assigned to this cell)
                    "ppkm2_nuc_mean_in":   ppkm2_nuc_mean_in,
                    "ppkm2_nuc_mean_bg":   ppkm2_nuc_mean_bg,
                    "ppkm2_nuc_int_ratio": ppkm2_nuc_ratio,
                })

                any_cells = True

                # per-cell QC plot (lightweight; independent of save_scene_qc)
                if PLOT_QC_CELLS and QC_PLOT_EVERY > 0 and (cid % QC_PLOT_EVERY == 0):
                    cell_composite = np.stack([pdgfra_raw, mbp_raw + o4_raw, dapi_raw], axis=-1)
                    cell_composite_display = cell_composite.copy()
                    cell_composite_display[~cellpose_roi] = 0

                    skel_all = skeletonize(cell_mask_all)
                    hull_all = convex_hull_image(cell_mask_all) if np.any(cell_mask_all) else cell_mask_all

                    skel_mbppdg = skeletonize(cell_mask_mbppdg)
                    hull_mbppdg = convex_hull_image(cell_mask_mbppdg) if np.any(cell_mask_mbppdg) else cell_mask_mbppdg

                    fig, axes = plt.subplots(1, 7, figsize=(26, 4))

                    if np.any(cell_composite_display > 0):
                        vmax = np.percentile(cell_composite_display[cell_composite_display > 0], 99)
                    else:
                        vmax = 1.0
                    axes[0].imshow(np.clip(cell_composite_display, 0, vmax).astype(np.float32) / float(vmax))
                    axes[0].set_title("Composite (ROI)")
                    axes[0].axis("off")

                    axes[1].imshow(cell_mask_all, cmap="gray")
                    axes[1].set_title("Mask ALL (MBP|PDGFRa|pPKM2)")
                    axes[1].axis("off")

                    axes[2].imshow(cell_mask_mbppdg, cmap="gray")
                    axes[2].set_title("Mask MBP|PDGFRa")
                    axes[2].axis("off")

                    axes[3].imshow(hull_all, cmap="gray")
                    if np.any(skel_all):
                        axes[3].contour(skel_all, colors="r", linewidths=0.7)
                    axes[3].set_title("Skel on Hull (ALL)")
                    axes[3].axis("off")

                    axes[4].imshow(hull_mbppdg, cmap="gray")
                    if np.any(skel_mbppdg):
                        axes[4].contour(skel_mbppdg, colors="r", linewidths=0.7)
                    axes[4].set_title("Skel on Hull (MBP|PDGFRa)")
                    axes[4].axis("off")

                    ppkm2_roi = ppkm2_raw.copy()
                    ppkm2_roi[~cellpose_roi] = 0
                    if np.any(ppkm2_roi > 0):
                        vmax2 = np.percentile(ppkm2_roi[ppkm2_roi > 0], 99)
                    else:
                        vmax2 = 1.0
                    axes[5].imshow(np.clip(ppkm2_roi, 0, vmax2) / float(vmax2), cmap="gray")
                    axes[5].set_title("pPKM2 (ROI)")
                    axes[5].axis("off")

                    # Sholl curves (ALL vs MBP|PDGFRa vs pPKM2)
                    if sholl_r_all.size > 0:
                        axes[6].plot(sholl_r_all, sholl_i_all, marker="o", label="ALL")
                    if sholl_r_mbppdg.size > 0:
                        axes[6].plot(sholl_r_mbppdg, sholl_i_mbppdg, marker="o", label="MBP|PDGFRa")
                    if sholl_r_ppkm2.size > 0:
                        axes[6].plot(sholl_r_ppkm2, sholl_i_ppkm2, marker="o", label="pPKM2-only")
                    axes[6].invert_xaxis()
                    axes[6].set_xlabel("Radius (px)")
                    axes[6].set_ylabel("Intersections")
                    axes[6].set_title("Sholl")
                    axes[6].grid(alpha=0.3)
                    axes[6].legend()

                    plt.tight_layout()
                    qc_path = pathlib.Path(QC_DIR) / f"{base_name}_cell{cid}_QC.png"
                    plt.savefig(qc_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)

            # checkpoint after successful scene
            save_checkpoint(lif_path, scene_idx)

        except Exception as e:
            LOGGER.error(f"[ERROR] scene failed: {lif_path.name} | scene {scene_idx} | {base_name} | {e}")
            log_failure(lif_path, scene_idx, base_name, e)

        finally:
            # per-scene cleanup
            for var in [
                "data", "dapi_raw", "masks_nuc",
                "mbp_raw", "pdgfra_raw", "o4_raw", "ppkm2_raw",
                "mbp_mask_raw", "pdgfra_mask_raw", "o4_mask_raw", "ppkm2_mask_raw",
                "mbp_mask", "pdgfra_mask", "o4_mask", "ppkm2_mask",
                "masks_cell", "mbp_seg_raw", "pdgfra_seg_raw", "o4_seg_raw", "ppkm2_seg_raw",
            ]:
                if var in locals():
                    try:
                        del locals()[var]
                    except Exception:
                        pass
            gc.collect()

        # periodic flush to disk
        if APPEND_PARTIAL_CSV and (processed_scene_counter % FLUSH_EVERY_SCENES == 0):
            try:
                append_records_to_csv(partial_csv_path, buffered_records)
                LOGGER.info(f"[flush] appended {len(buffered_records)} rows -> {partial_csv_path.name}")
                buffered_records = []
            except Exception as e:
                LOGGER.error(f"[flush] failed to append partial CSV: {e}")
                log_failure(lif_path, scene_idx, base_name, e)

        LOGGER.info(f"[timing] scene {scene_idx} done in {time.time() - t0:.1f}s")

    # final flush
    if APPEND_PARTIAL_CSV and buffered_records:
        append_records_to_csv(partial_csv_path, buffered_records)
        LOGGER.info(f"[flush] final append {len(buffered_records)} rows -> {partial_csv_path.name}")
        buffered_records = []

    try:
        lif.close()
    except Exception:
        pass
    del lif
    gc.collect()

    # final consolidated CSV
    if WRITE_FINAL_CSV and partial_csv_path.exists():
        try:
            df = pd.read_csv(partial_csv_path)
            df.to_csv(final_csv_path, index=False)
            LOGGER.info(f"[stats] wrote FINAL CSV: {final_csv_path}")
        except Exception as e:
            LOGGER.error(f"[stats] failed to write FINAL CSV for {lif_stem}: {e}")
            log_failure(lif_path, -1, lif_stem, e)

    if not any_cells:
        LOGGER.warning(f"[stats] no cells found for {lif_stem}")

    return any_cells


def main():
    safety_check_or_die()

    LOGGER.info("[cellpose] loading models...")
    nuc_model  = models.CellposeModel(pretrained_model=NUC_MODEL_PATH,  gpu=True)
    cell_model = models.CellposeModel(pretrained_model=CELL_MODEL_PATH, gpu=True)

    any_cells_any_lif = False
    for lif_path, cfg in LIF_CONFIG.items():
        ok = process_lif(lif_path, cfg, nuc_model, cell_model)
        any_cells_any_lif = any_cells_any_lif or ok

    try:
        del nuc_model, cell_model
    except Exception:
        pass
    gc.collect()

    if not any_cells_any_lif:
        LOGGER.warning("[stats] no cells found in any LIF; no final CSVs produced (partial may still exist).")


if __name__ == "__main__":
    main()
