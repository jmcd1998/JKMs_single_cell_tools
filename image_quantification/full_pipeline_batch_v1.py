"""
Batch calcium + stain pipeline (multi-run script).

What it does:
1) Scan DATA_ROOT and pair staining + calcium + landmarks + ilastik models.
2) Write VERIFY_CSV first (before heavy compute).
3) Run per-bundle pipeline (cellpose, warping, ilastik, metrics).
"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from cellpose import models

from skimage.transform import estimate_transform, warp
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries
from skimage.morphology import reconstruction, binary_dilation, disk


# =========================
# PATHS + BATCH CONFIG
# =========================
DATA_ROOT = Path(r"C:\Users\JackM\Calcium\multiplex")
ILASTIK_ROOT = Path(r"C:\Users\JackM\calcium\ilastik_models")
OUTPUT_ROOT = Path(r"C:\Users\JackM\calcium\out_registration_batch")
VERIFY_CSV = OUTPUT_ROOT / "verify_inputs.csv"

STRICT_VERIFY = False
DUPLICATE_POLICY = "error"  # "error" | "newest" | "first"

STAINING_MODEL_PATH = Path(r"C:\Users\JackM\calcium\cyto_cellpose\models\calcium_post_fix")
NUCLEUS_MODEL_PATH = Path(r"C:\Users\JackM\.cellpose\models\DAPI Nuclie")
CALCIUM_MAX_PROJ_MODEL_PATH = Path(r"C:\Users\JackM\calcium\training\models\calcium6")


# =========================
# CHANNEL CONFIG
# =========================
STAIN_CH_DAPI   = 0
STAIN_CH_O4     = 1
STAIN_CH_PDGFRa = 2
STAIN_CH_MBP    = 3
STAIN_CH_BF     = 4

CALCIUM_CH_FLUO8 = 0
CALCIUM_CH_BF    = 1
CALCIUM_BF_IMG_PATH = None  # set to a BF tif if separate

# If BigWarp landmarks were made on a rotated stain image, set k=1 for 90 deg left (CCW).
STAIN_ROTATE_K = 1

# Landmarks units (BigWarp)
LANDMARKS_IN_UM = True
STAIN_UM_PER_PX   = (0.325000, 0.325000)   # stain/moving image
CALCIUM_UM_PER_PX = (0.454326, 0.454326)   # calcium/fixed image

# Stain cropping (for cellpose/ilastik; uses landmarks to crop roughly to calcium FOV)
CROP_STAIN_FOR_CELLPOSE = True
CROP_STAIN_FOR_ILASTIK = True
CROP_PAD_PX = 100
CROP_MIN_SIZE = 256


# =========================
# ILASTIK CONFIG
# =========================
ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.0.post1\ilastik.exe"

CH_MAP = {
    "O4":     STAIN_CH_O4,
    "PDGFRa": STAIN_CH_PDGFRa,
    "MBP":    STAIN_CH_MBP,
    "BF":     STAIN_CH_BF,
}


# =========================
# QC OUTPUT
# =========================
QC_SAVE = True
QC_SAVE_EVERY_N = 1
QC_SAVE_INDIVIDUAL = True
SHOW_PLOTS = False

# Calcium trace plot
TRACE_MAX_ROIS = None  # set an int to cap number of ROI traces
TRACE_ALPHA = 0.15
TRACE_MEAN_COLOR = "red"
TRACE_DPI = 300

# Calcium windowed metrics config (gap detection on raw values)
CALCIUM_GAP_MIN_LEN = 3
CALCIUM_GAP_ZERO_THRESH = 10.0  # values 0–10 treated as "zero"
CALCIUM_N_WINDOWS = 3
CALCIUM_POSITIVE_ONLY = True
CALCIUM_RESPONSE_SD_MULT = 5.0

# =========================
# PARSING CONFIG
# =========================
TREATMENT_ALIASES = {
    "HAMI3379": ["HAMI3379", "HAMI"],
    "MDL29951": ["MDL29951", "MDL"],
    "pranlukast": ["pranlukast", "Pranlukast"],
}

EXCLUDE_TOKENS = {"veh", "vehicle", "aborted", "series"}

BIO_RE = re.compile(r"(N\d+)", re.IGNORECASE)
TECH_RE = re.compile(r"_(\d+)\.(tif|tiff|csv)$", re.IGNORECASE)
DATE_RE_8 = re.compile(r"(20\d{6})")
DATE_RE_6 = re.compile(r"\b(\d{6})\b")

# =========================
# HELPERS
# =========================
def ensure_cyx(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Stain image must be 3D, got shape {img.shape}")
    if img.shape[-1] <= 10 and img.shape[0] > 10:
        img = np.moveaxis(img, -1, 0)
    if img.shape[0] > 10:
        raise ValueError(f"Could not interpret stain image channels. Got {img.shape}")
    return img


def get_calcium_tyx(calcium: np.ndarray, channel: int = 0) -> np.ndarray:
    if calcium.ndim == 3:
        return calcium
    return calcium[:, channel, :, :]


def max_project_t(img_tyx: np.ndarray) -> np.ndarray:
    return np.max(img_tyx, axis=0)


def save_labels_imagej(labels: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, labels.astype(np.uint16, copy=False), imagej=True)


def save_labels_png_glasbey(labels: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.colors as mcolors
    try:
        import colorcet as cc
        colors = [mcolors.to_rgb(c) for c in cc.glasbey]
    except Exception:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i)[:3] for i in range(cmap.N)]

    palette = (np.array(colors) * 255).astype(np.uint8)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    mask = labels > 0
    if mask.any():
        color_idx = ((labels[mask] - 1) % len(palette)).astype(np.int64)
        rgb[mask] = palette[color_idx]
    plt.imsave(out_path, rgb)


def make_stain_composite(stain_cyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dapi   = stain_cyx[STAIN_CH_DAPI].astype(np.float32, copy=False)
    o4     = stain_cyx[STAIN_CH_O4].astype(np.float32, copy=False)
    pdgfra = stain_cyx[STAIN_CH_PDGFRa].astype(np.float32, copy=False)
    mbp    = stain_cyx[STAIN_CH_MBP].astype(np.float32, copy=False)
    bf     = stain_cyx[STAIN_CH_BF].astype(np.float32, copy=False)
    comp_sum = o4 + pdgfra + mbp
    comp = np.stack([dapi, comp_sum], axis=-1)
    return comp, dapi, o4, pdgfra, mbp, bf

def run_cellpose_on_composite(comp_yx2: np.ndarray, model_path: Path, diameter: float | None = None, use_gpu: bool = True) -> np.ndarray:
    if comp_yx2.ndim != 3 or comp_yx2.shape[-1] != 2:
        raise ValueError(f"Composite must be (Y,X,2), got {comp_yx2.shape}")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    x = comp_yx2.astype(np.float32, copy=False)
    masks, _, _ = model.eval(
        x,
        channels=[2, 1],
        channel_axis=-1,
        diameter=diameter,
        do_3D=False,
    )
    return masks.astype(np.uint16, copy=False)


def run_cellpose_on_nucleus_channel(nuc_img: np.ndarray, model_path: Path, diameter: float | None = None, use_gpu: bool = True) -> np.ndarray:
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    x = nuc_img.astype(np.float32, copy=False)
    masks, _, _ = model.eval(x, diameter=diameter, do_3D=False)
    return masks.astype(np.uint16, copy=False)


def run_cellpose_on_calcium_max_project(calcium_max: np.ndarray, model_path: Path, diameter: float | None = None, use_gpu: bool = True) -> np.ndarray:
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    x = calcium_max.astype(np.float32, copy=False)
    masks, _, _ = model.eval(x, diameter=diameter, do_3D=False)
    return masks.astype(np.uint16, copy=False)


def read_bigwarp_landmarks_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, header=None)
    active = df.iloc[:, 1].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df = df[active]
    mx = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(np.float64)
    my = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(np.float64)
    fx = pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy(np.float64)
    fy = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy(np.float64)
    moving_xy = np.column_stack([mx, my])
    fixed_xy  = np.column_stack([fx, fy])
    good = np.isfinite(moving_xy).all(axis=1) & np.isfinite(fixed_xy).all(axis=1)
    moving_xy = moving_xy[good]
    fixed_xy  = fixed_xy[good]
    if moving_xy.shape[0] < 3:
        raise ValueError(f"Need at least 3 valid landmarks, got {moving_xy.shape[0]}")
    return moving_xy, fixed_xy


def build_landmark_tform(landmarks_csv: Path | None):
    if landmarks_csv is None or not Path(landmarks_csv).exists():
        return None
    moving_xy, fixed_xy = read_bigwarp_landmarks_csv(landmarks_csv)
    if LANDMARKS_IN_UM:
        mx_um_per_px, my_um_per_px = STAIN_UM_PER_PX
        fx_um_per_px, fy_um_per_px = CALCIUM_UM_PER_PX
        moving_xy = moving_xy.copy()
        fixed_xy = fixed_xy.copy()
        moving_xy[:, 0] = moving_xy[:, 0] / mx_um_per_px
        moving_xy[:, 1] = moving_xy[:, 1] / my_um_per_px
        fixed_xy[:, 0] = fixed_xy[:, 0] / fx_um_per_px
        fixed_xy[:, 1] = fixed_xy[:, 1] / fy_um_per_px
    return estimate_transform("affine", src=moving_xy, dst=fixed_xy)


def compute_stain_crop_from_tform(
    tform,
    fixed_shape_yx: Tuple[int, int],
    moving_shape_yx: Tuple[int, int],
    pad_px: int = 100,
    min_size: int = 256,
) -> Tuple[int, int, int, int] | None:
    fy, fx = fixed_shape_yx
    corners = np.array(
        [
            [0, 0],
            [fx - 1, 0],
            [0, fy - 1],
            [fx - 1, fy - 1],
        ],
        dtype=np.float64,
    )
    moving = tform.inverse(corners)
    xs = moving[:, 0]
    ys = moving[:, 1]
    x0 = int(np.floor(xs.min())) - pad_px
    x1 = int(np.ceil(xs.max())) + pad_px
    y0 = int(np.floor(ys.min())) - pad_px
    y1 = int(np.ceil(ys.max())) + pad_px

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(moving_shape_yx[0], y1)
    x1 = min(moving_shape_yx[1], x1)

    if (x1 - x0) < min_size or (y1 - y0) < min_size:
        return None
    return (y0, y1, x0, x1)


def embed_labels_into_full(
    labels_crop: np.ndarray,
    full_shape_yx: Tuple[int, int],
    crop_bounds: Tuple[int, int, int, int],
) -> np.ndarray:
    y0, y1, x0, x1 = crop_bounds
    full = np.zeros(full_shape_yx, dtype=labels_crop.dtype)
    full[y0:y1, x0:x1] = labels_crop
    return full

def match_labels_by_overlap(labels_a: np.ndarray, labels_b: np.ndarray, min_frac: float = 0.7) -> List[Tuple[int, int, int, float, int, int]]:
    a = labels_a.ravel()
    b = labels_b.ravel()
    a_max = int(a.max())
    b_max = int(b.max())
    if a_max == 0 or b_max == 0:
        return []
    hist = np.zeros((a_max + 1, b_max + 1), dtype=np.int64)
    np.add.at(hist, (a, b), 1)
    area_a = hist.sum(axis=1)
    area_b = hist.sum(axis=0)
    candidates = []
    for a_id in range(1, a_max + 1):
        if area_a[a_id] == 0:
            continue
        overlaps = hist[a_id, 1:]
        if overlaps.size == 0:
            continue
        b_id = int(overlaps.argmax() + 1)
        overlap = int(overlaps[b_id - 1])
        frac = overlap / area_a[a_id] if area_a[a_id] else 0.0
        if overlap > 0 and frac >= min_frac:
            candidates.append((a_id, b_id, overlap, float(frac), int(area_a[a_id]), int(area_b[b_id])))
    candidates.sort(key=lambda x: x[2], reverse=True)
    used_a, used_b, matches = set(), set(), []
    for a_id, b_id, overlap, frac, area_a_i, area_b_i in candidates:
        if a_id in used_a or b_id in used_b:
            continue
        used_a.add(a_id)
        used_b.add(b_id)
        matches.append((a_id, b_id, overlap, frac, area_a_i, area_b_i))
    return matches


def measure_timeseries(stain_TYX: np.ndarray, labels_yx: np.ndarray, baseline_frames: int = 6) -> pd.DataFrame:
    T, Y, X = stain_TYX.shape
    lab = labels_yx
    ids = np.unique(lab)
    ids = ids[ids > 0]
    areas = {i: int((lab == i).sum()) for i in ids}
    pix = {i: np.where(lab == i) for i in ids}
    rows = []
    for i in ids:
        y_idx, x_idx = pix[i]
        F = np.empty(T, dtype=np.float32)
        for t in range(T):
            F[t] = stain_TYX[t, y_idx, x_idx].mean()
        if baseline_frames >= 1 and T >= baseline_frames:
            F0 = float(np.median(F[:baseline_frames]))
        else:
            F0 = float(F[0]) if T > 0 else 0.0
        # dF/F0 normalization (baseline-corrected)
        dff = (F - F0) / (F0 + 1e-8)
        for t in range(T):
            rows.append((t, int(i), float(F[t]), areas[i], float(dff[t])))
    return pd.DataFrame(rows, columns=["t", "cell_id", "mean", "area", "dff"])


def _runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return list(zip(edges[0::2], edges[1::2]))  # (start, end)


def _gap_mask_from_low(y: np.ndarray, min_len: int, low_thresh: float) -> np.ndarray:
    low_mask = y <= low_thresh
    runs = _runs_from_mask(low_mask)
    gap = np.zeros_like(low_mask, dtype=bool)
    for a, b in runs:
        if (b - a) >= min_len:
            gap[a:b] = True
    return gap


def _define_windows_from_mask(gap_mask: np.ndarray, n_windows: int) -> Tuple[Tuple[int, int] | None, List[Tuple[int, int]]]:
    non_gap = ~gap_mask
    segments = _runs_from_mask(non_gap)
    if len(segments) < (1 + n_windows):
        return None, []
    baseline = segments[0]
    windows = segments[1:1 + n_windows]
    return baseline, windows


def compute_calcium_metrics(calcium_tyx: np.ndarray, labels_yx: np.ndarray, baseline_frames: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    T, Y, X = calcium_tyx.shape
    lab = labels_yx
    ids = np.unique(lab)
    ids = ids[ids > 0]
    areas = {i: int((lab == i).sum()) for i in ids}
    pix = {i: np.where(lab == i) for i in ids}

    rows_long: List[Tuple[int, int, float, int, float]] = []
    metrics_rows: List[Dict[str, float]] = []
    t = np.arange(T, dtype=float)

    for cid in ids:
        y_idx, x_idx = pix[cid]
        F = np.empty(T, dtype=np.float32)
        for ti in range(T):
            F[ti] = calcium_tyx[ti, y_idx, x_idx].mean()

        gap_mask = _gap_mask_from_low(F, CALCIUM_GAP_MIN_LEN, CALCIUM_GAP_ZERO_THRESH)
        baseline, windows = _define_windows_from_mask(gap_mask, CALCIUM_N_WINDOWS)
        if baseline is None:
            if baseline_frames >= 1 and T >= baseline_frames:
                baseline = (0, baseline_frames)
            else:
                baseline = (0, T)
            windows = []

        base_mask = np.zeros(T, dtype=bool)
        base_mask[baseline[0]:baseline[1]] = True
        base_mask &= ~gap_mask
        if base_mask.any():
            F0 = float(np.nanmedian(F[base_mask]))
        else:
            F0 = np.nan

        if not np.isfinite(F0) or F0 == 0:
            dff = np.full(T, np.nan, dtype=np.float32)
        else:
            # dF/F0 normalization (baseline-corrected)
            dff = (F - F0) / F0

        for ti in range(T):
            rows_long.append((ti, int(cid), float(F[ti]), areas[cid], float(dff[ti]) if np.isfinite(dff[ti]) else np.nan))

        base_vals = dff[base_mask] if base_mask.any() else np.array([])
        base_mean = float(np.nanmean(base_vals)) if base_vals.size else np.nan
        base_sd = float(np.nanstd(base_vals, ddof=1)) if base_vals.size else np.nan
        thr = base_mean + CALCIUM_RESPONSE_SD_MULT * base_sd if np.isfinite(base_mean) and np.isfinite(base_sd) else np.nan

        def _window_metrics(window: Tuple[int, int] | None) -> Tuple[float, float, float, float]:
            if window is None:
                return (np.nan, np.nan, np.nan, np.nan)
            a, b = window
            m = np.zeros(T, dtype=bool)
            m[a:b] = True
            m &= ~gap_mask
            if np.sum(m) < 2:
                return (np.nan, np.nan, np.nan, np.nan)
            t_win = t[m]
            y_win_raw = dff[m]
            if CALCIUM_POSITIVE_ONLY:
                y_win = np.where(y_win_raw > 0, y_win_raw, 0)
            else:
                y_win = y_win_raw
            auc = float(np.trapz(y_win, t_win))
            active_time = float(t_win[-1] - t_win[0])
            auc_norm = auc / active_time if active_time > 0 else np.nan
            if np.isfinite(thr) and active_time > 0:
                resp_frac = float(np.mean(y_win_raw > thr))
            else:
                resp_frac = np.nan
            return (auc, auc_norm, active_time, resp_frac)

        wlist = [None] * CALCIUM_N_WINDOWS
        for idx, w in enumerate(windows[:CALCIUM_N_WINDOWS]):
            wlist[idx] = w

        auc_w1, auc_norm_w1, active_w1, rt_w1 = _window_metrics(wlist[0])
        auc_w2, auc_norm_w2, active_w2, rt_w2 = _window_metrics(wlist[1])
        auc_w3, auc_norm_w3, active_w3, rt_w3 = _window_metrics(wlist[2])

        if np.isfinite(auc_w2) and np.isfinite(auc_w3):
            auc_w2_w3 = auc_w2 + auc_w3
        else:
            auc_w2_w3 = np.nan

        if np.isfinite(active_w2) and np.isfinite(active_w3) and (active_w2 + active_w3) > 0 and np.isfinite(auc_w2_w3):
            auc_norm_w2_w3 = auc_w2_w3 / (active_w2 + active_w3)
        else:
            auc_norm_w2_w3 = np.nan

        metrics_rows.append({
            "calcium_label": int(cid),
            "calcium AUC": auc_w2_w3,
            "calcium response time": rt_w2,
            "calcium_auc_w1": auc_w1,
            "calcium_auc_w2": auc_w2,
            "calcium_auc_w3": auc_w3,
            "calcium_auc_norm_w1": auc_norm_w1,
            "calcium_auc_norm_w2": auc_norm_w2,
            "calcium_auc_norm_w3": auc_norm_w3,
            "calcium_active_time_w1": active_w1,
            "calcium_active_time_w2": active_w2,
            "calcium_active_time_w3": active_w3,
            "calcium_response_time_w1": rt_w1,
            "calcium_response_time_w2": rt_w2,
            "calcium_response_time_w3": rt_w3,
            "calcium_auc_w2_w3": auc_w2_w3,
            "calcium_auc_norm_w2_w3": auc_norm_w2_w3,
            "calcium_AUC_norm": auc_norm_w2_w3,
            "calcium_baseline_F0": F0,
            "calcium_baseline_mean_dff": base_mean,
            "calcium_baseline_sd_dff": base_sd,
        })

    df_long = pd.DataFrame(rows_long, columns=["t", "cell_id", "mean", "area", "dff"])
    df_metrics = pd.DataFrame(metrics_rows)
    return df_long, df_metrics


def plot_calcium_traces(df_long: pd.DataFrame, out_path: Path, max_rois: int | None = None) -> None:
    if df_long.empty:
        return
    pivot = df_long.pivot(index="t", columns="cell_id", values="dff").sort_index()
    if max_rois is not None and pivot.shape[1] > max_rois:
        pivot = pivot.iloc[:, :max_rois]

    t = pivot.index.to_numpy()
    plt.figure(figsize=(10, 5))
    for col in pivot.columns:
        plt.plot(t, pivot[col].to_numpy(), color="black", alpha=TRACE_ALPHA, linewidth=0.8)
    mean_trace = pivot.mean(axis=1).to_numpy()
    plt.plot(t, mean_trace, color=TRACE_MEAN_COLOR, linewidth=2.5, label="Mean")
    plt.xlabel("Frame")
    plt.ylabel("dF/F0")
    plt.title("Calcium ROI traces (thin) + mean (thick)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=TRACE_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def run_ilastik_on_array(img: np.ndarray, ilp_path: str, tmp_root: Path, tag: str) -> np.ndarray:
    tmp_root.mkdir(parents=True, exist_ok=True)
    for p in tmp_root.glob(f"{tag}_*.tif*"):
        try:
            p.unlink()
        except Exception:
            pass
    tmp_in = tmp_root / f"{tag}_in.tiff"
    tmp_out = tmp_root / f"{tag}_seg.tiff"
    tifffile.imwrite(tmp_in, img.astype(np.float64))
    cmd = [
        ILASTIK_EXE,
        "--headless",
        f"--project={ilp_path}",
        "--export_source=Simple Segmentation",
        "--output_format=tiff",
        f"--output_filename_format={tmp_out}",
        str(tmp_in),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ilastik failed for tag '{tag}'.\nSTDERR:\n{result.stderr}")
    if not tmp_out.exists():
        raise FileNotFoundError(f"Expected Ilastik output '{tmp_out}' not found.")
    seg = tifffile.imread(tmp_out)
    try:
        tmp_in.unlink(missing_ok=True)
    except Exception:
        pass
    if seg.ndim > 2:
        seg = np.squeeze(seg)
    return seg.astype(np.uint8)


def remove_components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    lab = label(mask)
    touching = np.unique(lab[seed])
    touching = touching[touching > 0]
    if touching.size == 0:
        return mask.copy()
    out = mask.copy()
    out[np.isin(lab, touching)] = False
    return out


# =========================
# PAIRING HELPERS
# =========================
def parse_date_token(name: str) -> Tuple[str | None, int | None]:
    m = DATE_RE_8.search(name)
    if m:
        token = m.group(1)
        return token, int(token)
    m = DATE_RE_6.search(name)
    if m:
        token = m.group(1)
        return token, int("20" + token)
    return None, None


def infer_bio(name: str) -> str | None:
    m = BIO_RE.search(name)
    return m.group(1).upper() if m else None


def infer_tech(name: str) -> int:
    m = TECH_RE.search(name)
    return int(m.group(1)) if m else 1


def infer_modality(name: str) -> str | None:
    n = name.lower()
    if "staining" in n:
        return "staining"
    if "calcium" in n:
        return "calcium"
    return None


def infer_treatment(name: str) -> str | None:
    n = name.lower()
    if any(tok in n for tok in EXCLUDE_TOKENS):
        return None
    for canon, aliases in TREATMENT_ALIASES.items():
        for a in aliases:
            if a.lower() in n:
                return canon
    return None


def make_file_info(path: Path) -> Dict[str, object]:
    token, num = parse_date_token(path.name)
    return {"path": path, "date_token": token, "date_num": num}


def select_file(files: List[Dict[str, object]], label: str) -> Tuple[Dict[str, object] | None, str]:
    if len(files) == 0:
        return None, f"missing_{label}"
    if len(files) == 1:
        return files[0], ""
    if DUPLICATE_POLICY == "error":
        return None, f"duplicate_{label}"
    if DUPLICATE_POLICY == "first":
        return files[0], f"duplicate_{label}_first"
    if DUPLICATE_POLICY == "newest":
        def _score(info: Dict[str, object]) -> int:
            val = info.get("date_num")
            return int(val) if val is not None else -1
        chosen = max(files, key=_score)
        return chosen, f"duplicate_{label}_newest"
    return files[0], f"duplicate_{label}_first"


def ilp_paths_for_bio(bio: str) -> Dict[str, Path]:
    b = bio.lower()
    return {
        "O4": ILASTIK_ROOT / f"{b}_o4" / f"{b}_o4.ilp",
        "PDGFRa": ILASTIK_ROOT / f"{b}_pdgfra" / f"{b}_pdgfra.ilp",
        "MBP": ILASTIK_ROOT / f"{b}_mbp" / f"{b}_mbp.ilp",
        "BF": ILASTIK_ROOT / f"{b}_bf" / f"{b}_bf.ilp",
    }


def build_bundle_map(root: Path) -> Dict[Tuple[str, str, int], Dict[str, List[Dict[str, object]]]]:
    bundles: Dict[Tuple[str, str, int], Dict[str, List[Dict[str, object]]]] = defaultdict(
        lambda: {"staining": [], "calcium": [], "landmarks": []}
    )
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".tif", ".tiff", ".csv"}:
            continue
        name = p.name
        bio = infer_bio(name)
        treatment = infer_treatment(name)
        if bio is None or treatment is None:
            continue
        tech = infer_tech(name)
        key = (bio, treatment, tech)
        info = make_file_info(p)

        if p.suffix.lower() == ".csv":
            bundles[key]["landmarks"].append(info)
            continue

        modality = infer_modality(name)
        if modality == "staining":
            bundles[key]["staining"].append(info)
        elif modality == "calcium":
            bundles[key]["calcium"].append(info)
    return bundles

def verify_and_prepare(bundle_map: Dict[Tuple[str, str, int], Dict[str, List[Dict[str, object]]]]) -> List[Dict[str, object]]:
    rows = []
    prepared = []

    for key in sorted(bundle_map.keys()):
        bio, treatment, tech = key
        parts = bundle_map[key]

        stain_info, stain_note = select_file(parts["staining"], "staining")
        cal_info, cal_note = select_file(parts["calcium"], "calcium")
        csv_info, csv_note = select_file(parts["landmarks"], "landmarks")

        ilp_paths = ilp_paths_for_bio(bio)
        ilp_exists = {k: p.exists() for k, p in ilp_paths.items()}

        notes = [n for n in [stain_note, cal_note, csv_note] if n]

        stain_path = stain_info["path"] if stain_info else None
        cal_path = cal_info["path"] if cal_info else None
        csv_path = csv_info["path"] if csv_info else None

        complete = True
        if stain_path is None or cal_path is None or csv_path is None:
            complete = False
        if not all(ilp_exists.values()):
            complete = False

        bundle_id = f"{bio}_{treatment}_T{tech}"

        rows.append({
            "bundle_id": bundle_id,
            "bio": bio,
            "treatment": treatment,
            "tech": tech,
            "staining_img": str(stain_path) if stain_path else "",
            "calcium_img": str(cal_path) if cal_path else "",
            "landmarks_csv": str(csv_path) if csv_path else "",
            "staining_candidates": "|".join(str(p["path"]) for p in parts["staining"]),
            "calcium_candidates": "|".join(str(p["path"]) for p in parts["calcium"]),
            "landmarks_candidates": "|".join(str(p["path"]) for p in parts["landmarks"]),
            "ilastik_used": "|".join([
                f"O4={ilp_paths['O4']}",
                f"PDGFRa={ilp_paths['PDGFRa']}",
                f"MBP={ilp_paths['MBP']}",
                f"BF={ilp_paths['BF']}",
            ]),
            "ilp_o4": str(ilp_paths["O4"]),
            "ilp_pdgfra": str(ilp_paths["PDGFRa"]),
            "ilp_mbp": str(ilp_paths["MBP"]),
            "ilp_bf": str(ilp_paths["BF"]),
            "staining_exists": bool(stain_path and Path(stain_path).exists()),
            "calcium_exists": bool(cal_path and Path(cal_path).exists()),
            "landmarks_exists": bool(csv_path and Path(csv_path).exists()),
            "ilp_o4_exists": ilp_exists["O4"],
            "ilp_pdgfra_exists": ilp_exists["PDGFRa"],
            "ilp_mbp_exists": ilp_exists["MBP"],
            "ilp_bf_exists": ilp_exists["BF"],
            "complete": complete,
            "notes": ";".join(notes),
        })

        if complete:
            prepared.append({
                "bundle_id": bundle_id,
                "bio": bio,
                "treatment": treatment,
                "tech": tech,
                "staining_img": Path(stain_path),
                "calcium_img": Path(cal_path),
                "landmarks_csv": Path(csv_path),
                "ilastik_paths": ilp_paths,
            })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(VERIFY_CSV, index=False)
    print(f"Wrote verification CSV: {VERIFY_CSV}")

    if STRICT_VERIFY:
        if df.empty:
            raise RuntimeError("No candidate bundles found.")
        bad = df[~df["complete"]]
        if not bad.empty:
            raise RuntimeError(f"Incomplete or ambiguous inputs. Fix and re-run. See {VERIFY_CSV}")

    if not prepared:
        raise RuntimeError("No complete bundles to process.")

    return prepared

# =========================
# PIPELINE PER BUNDLE
# =========================
def process_bundle(bundle: Dict[str, object], qc_index: int = 0) -> None:
    calcium_img_path: Path = bundle["calcium_img"]
    stain_img_path: Path = bundle["staining_img"]
    landmarks_csv: Path = bundle["landmarks_csv"]
    ilp_paths: Dict[str, Path] = bundle["ilastik_paths"]

    output_dir = OUTPUT_ROOT / bundle["bundle_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ilastik_tmp = output_dir / "ilastik_tmp"

    # -------------------------
    # Load calcium + stain
    # -------------------------
    calcium = tifffile.imread(calcium_img_path)
    calcium_tyx = get_calcium_tyx(calcium, channel=CALCIUM_CH_FLUO8)
    calcium_max = max_project_t(calcium_tyx)

    if CALCIUM_BF_IMG_PATH is not None:
        calcium_bf = tifffile.imread(CALCIUM_BF_IMG_PATH)
        if calcium_bf.ndim == 3:
            calcium_bf = max_project_t(calcium_bf)
    elif calcium.ndim == 4 and CALCIUM_CH_BF is not None:
        calcium_bf = max_project_t(calcium[:, CALCIUM_CH_BF, :, :])
    else:
        calcium_bf = calcium_max

    stain = tifffile.imread(stain_img_path)
    stain_cyx = ensure_cyx(stain)
    if STAIN_ROTATE_K:
        stain_cyx = np.rot90(stain_cyx, k=STAIN_ROTATE_K, axes=(1, 2))

    # Build landmark transform early (used for optional crop + later warp)
    tform = build_landmark_tform(landmarks_csv)

    # Optionally crop stain image (for cellpose/ilastik) based on calcium FOV
    crop_bounds = None
    stain_cyx_cellpose = stain_cyx
    stain_cyx_ilastik = stain_cyx
    if (CROP_STAIN_FOR_CELLPOSE or CROP_STAIN_FOR_ILASTIK) and tform is not None:
        crop_bounds = compute_stain_crop_from_tform(
            tform,
            fixed_shape_yx=calcium_bf.shape,
            moving_shape_yx=stain_cyx.shape[1:],
            pad_px=CROP_PAD_PX,
            min_size=CROP_MIN_SIZE,
        )
        if crop_bounds is not None:
            y0, y1, x0, x1 = crop_bounds
            if CROP_STAIN_FOR_CELLPOSE:
                stain_cyx_cellpose = stain_cyx[:, y0:y1, x0:x1]
            if CROP_STAIN_FOR_ILASTIK:
                stain_cyx_ilastik = stain_cyx[:, y0:y1, x0:x1]
            print(f"[crop] stain crop y:{y0}-{y1} x:{x0}-{x1}")

    # Full-size BF for QC/warp overlays
    bf_full = stain_cyx[STAIN_CH_BF].astype(np.float32, copy=False)

    comp, dapi, *_ = make_stain_composite(stain_cyx_cellpose)

    # -------------------------
    # Cellpose on stain + nuclei
    # -------------------------
    labels_stain_crop = run_cellpose_on_composite(comp, STAINING_MODEL_PATH)
    labels_nuc_crop = run_cellpose_on_nucleus_channel(dapi, NUCLEUS_MODEL_PATH)

    labels_stain = labels_stain_crop
    labels_nuc = labels_nuc_crop
    if crop_bounds is not None and CROP_STAIN_FOR_CELLPOSE:
        labels_stain = embed_labels_into_full(labels_stain_crop, stain_cyx.shape[1:], crop_bounds)
        labels_nuc = embed_labels_into_full(labels_nuc_crop, stain_cyx.shape[1:], crop_bounds)

    labels_nuc_for_ilastik = labels_nuc
    if crop_bounds is not None and CROP_STAIN_FOR_ILASTIK:
        y0, y1, x0, x1 = crop_bounds
        labels_nuc_for_ilastik = labels_nuc[y0:y1, x0:x1]

    save_labels_imagej(labels_stain, output_dir / "labels_stain_imagej.tif")
    save_labels_png_glasbey(labels_stain, output_dir / "labels_stain_glasbey.png")
    save_labels_imagej(labels_nuc, output_dir / "labels_nuc_imagej.tif")
    save_labels_png_glasbey(labels_nuc, output_dir / "labels_nuc_glasbey.png")

    # -------------------------
    # Cellpose on calcium max projection
    # -------------------------
    labels_calcium_max = run_cellpose_on_calcium_max_project(calcium_max, CALCIUM_MAX_PROJ_MODEL_PATH)
    save_labels_imagej(labels_calcium_max, output_dir / "labels_calcium_max_imagej.tif")
    save_labels_png_glasbey(labels_calcium_max, output_dir / "labels_calcium_max_glasbey.png")

    # -------------------------
    # Calcium metrics
    # -------------------------
    df_long, calcium_metrics_df = compute_calcium_metrics(calcium_tyx, labels_calcium_max, baseline_frames=6)
    df_long.to_csv(output_dir / "calcium_timeseries_long.csv", index=False)
    calcium_metrics_df.to_csv(output_dir / "calcium_metrics.csv", index=False)
    plot_calcium_traces(df_long, output_dir / f"calcium_traces_{qc_index:03d}.png", max_rois=TRACE_MAX_ROIS)

    # -------------------------
    # Warping + matching
    # -------------------------
    labels_calcium = None
    if tform is not None:
        labels_calcium = warp(
            labels_stain,
            inverse_map=tform.inverse,
            output_shape=calcium_bf.shape,
            order=0,
            preserve_range=True,
            mode="constant",
            cval=0,
        ).astype(np.uint16)

    # label matching (>=70% overlap, one-to-one)
    df_matches = None
    matches = []
    if labels_calcium is not None:
        matches = match_labels_by_overlap(labels_calcium, labels_calcium_max, min_frac=0.70)
        df_matches = pd.DataFrame(
            matches,
            columns=["stain_label", "calcium_label", "overlap_px", "frac_stain", "stain_area", "calcium_area"],
        )
        df_matches = df_matches.merge(calcium_metrics_df, how="left", on="calcium_label")
        df_matches.to_csv(output_dir / "label_matches_70pct.csv", index=False)

    # -------------------------
    # Ilastik masks + cleaning
    # -------------------------
    ilastik_masks: Dict[str, np.ndarray] = {}
    ilastik_clean: Dict[str, np.ndarray] = {}

    for name, ilp in ilp_paths.items():
        ch = CH_MAP.get(name)
        if ilp is None or ch is None:
            continue
        img = stain_cyx_ilastik[ch]
        tag = f"stain_{name.lower()}"
        seg = run_ilastik_on_array(img, str(ilp), ilastik_tmp, tag)
        # Ilastik simple segmentation: 1 = positive, 2 = negative
        mask = seg < 2

        # initial clean (optional nuclei seed-based reconstruction)
        if labels_nuc_for_ilastik is not None:
            seed = (labels_nuc_for_ilastik > 0) & mask
            recon = reconstruction(seed.astype(np.uint8), mask.astype(np.uint8), method="dilation")
            clean = recon > 0
        else:
            clean = mask.copy()

        if crop_bounds is not None and CROP_STAIN_FOR_ILASTIK:
            mask = embed_labels_into_full(mask.astype(np.uint8), stain_cyx.shape[1:], crop_bounds).astype(bool)
            clean = embed_labels_into_full(clean.astype(np.uint8), stain_cyx.shape[1:], crop_bounds).astype(bool)

        ilastik_masks[name] = mask
        ilastik_clean[name] = clean

        tifffile.imwrite(output_dir / f"ilastik_{name.lower()}_mask.tif", mask.astype(np.uint8))
        tifffile.imwrite(output_dir / f"ilastik_{name.lower()}_clean.tif", clean.astype(np.uint8))

    # reverse reconstruction: remove objects touching BF clean
    bf_clean = ilastik_clean.get("BF", None)
    if bf_clean is None:
        raise RuntimeError("bf_clean not found from Ilastik.")
    seed_bf = bf_clean > 0
    for name, mask in list(ilastik_clean.items()):
        if name == "BF":
            continue
        cleaned = remove_components_touching_seed(mask.astype(bool), seed_bf)
        ilastik_clean[name] = cleaned
        tifffile.imwrite(output_dir / f"ilastik_{name.lower()}_clean.tif", cleaned.astype(np.uint8))

    # -------------------------
    # Stain metrics
    # -------------------------
    combined_clean = None
    for name in ["O4", "PDGFRa", "MBP"]:
        m = ilastik_clean.get(name)
        if m is None:
            continue
        combined_clean = m if combined_clean is None else (combined_clean | m)

    orig_union = None
    for name in ["O4", "PDGFRa", "MBP"]:
        m = ilastik_masks.get(name)
        if m is None:
            continue
        orig_union = m if orig_union is None else (orig_union | m)
    if orig_union is None:
        raise RuntimeError("No ilastik masks available for background computation.")
    if combined_clean is None:
        raise RuntimeError("No ilastik clean masks available for stain area computation.")

    # Background = dilation of NOT-in-mask region (10 px)
    bg_mask = binary_dilation(~orig_union.astype(bool), disk(10))
    bg_mfi: Dict[str, float] = {}
    for name in ["O4", "PDGFRa", "MBP"]:
        ch = CH_MAP.get(name)
        if ch is None:
            continue
        bg_mfi[name] = float(np.mean(stain_cyx[ch][bg_mask]))

    records = []
    cell_ids = np.unique(labels_stain)
    cell_ids = cell_ids[cell_ids > 0]
    for cid in cell_ids:
        cell_mask = labels_stain == cid
        clean_cell_mask = combined_clean & cell_mask
        cell_area = int(clean_cell_mask.sum())
        rec = {"stain_label": int(cid), "cell_area": cell_area}

        for name in ["O4", "PDGFRa", "MBP"]:
            ch = CH_MAP.get(name)
            if ch is None:
                continue
            m = ilastik_clean.get(name)
            if m is None:
                continue
            pos_area = int((m & cell_mask).sum())
            prop = pos_area / cell_area if cell_area > 0 else 0.0
            mfi = float(stain_cyx[ch][clean_cell_mask].mean()) if clean_cell_mask.any() else np.nan
            bg = bg_mfi.get(name, np.nan)
            mfi_norm = (mfi / bg) if (bg is not None and bg > 0) else np.nan
            rec[f"{name}_prop"] = prop
            rec[f"{name}_mfi_norm"] = mfi_norm
        records.append(rec)

    df_stain = pd.DataFrame(records)
    df_stain.to_csv(output_dir / "stain_metrics.csv", index=False)

    # merge all
    if df_matches is None:
        raise RuntimeError("label_matches_70pct.csv not generated (missing warp?).")
    df_merged = df_matches.merge(df_stain, how="left", on="stain_label")
    df_merged.to_csv(output_dir / "matched_cells_with_stain_metrics.csv", index=False)

    # -------------------------
    # QC plots
    # -------------------------
    bf_stain = bf_full
    bf_stain_warp = None
    labels_nuc_warp = None
    if tform is not None:
        bf_stain_warp = warp(bf_stain, inverse_map=tform.inverse, output_shape=calcium_bf.shape, order=1, preserve_range=True, mode="constant", cval=0)
        labels_nuc_warp = warp(labels_nuc, inverse_map=tform.inverse, output_shape=calcium_bf.shape, order=0, preserve_range=True, mode="constant", cval=0).astype(np.uint16)

    matches_list = [(a, b) for a, b, *_ in matches] if matches else []

    def plot_nuclei_qc(ax):
        ax.imshow(find_boundaries(labels_nuc), cmap="gray")
        ax.set_title("Nuclei boundaries")
        ax.axis("off")

    def plot_clean_qc(ax):
        ax.imshow(combined_clean if combined_clean is not None else np.zeros_like(labels_stain), cmap="gray")
        ax.set_title("Combined clean mask")
        ax.axis("off")

    def plot_grid_qc(ax):
        ax.imshow(calcium_bf, cmap="gray")
        if bf_stain_warp is not None:
            ax.imshow(bf_stain_warp, cmap="magma", alpha=0.35)
        ax.set_title("Grid registration (stain warp on calcium BF)")
        ax.axis("off")

    def plot_pair_qc(ax):
        ax.imshow(calcium_bf, cmap="gray")
        if labels_calcium is not None:
            ax.contour(find_boundaries(labels_calcium), colors="r", linewidths=0.5)
        ax.contour(find_boundaries(labels_calcium_max), colors="c", linewidths=0.5)
        if matches_list and labels_calcium is not None:
            props_stain = {p.label: p.centroid for p in regionprops(labels_calcium)}
            props_cal = {p.label: p.centroid for p in regionprops(labels_calcium_max)}
            for midx, (a_id, b_id) in enumerate(matches_list, start=1):
                if a_id in props_stain:
                    y, x = props_stain[a_id]
                    ax.text(x, y, str(midx), color="r", fontsize=8, ha="center", va="center")
                if b_id in props_cal:
                    y, x = props_cal[b_id]
                    ax.text(x, y, str(midx), color="c", fontsize=8, ha="center", va="center")
        ax.set_title("Pairing QC (red=stain warp, cyan=calcium)")
        ax.axis("off")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plot_nuclei_qc(axes[0, 0])
    plot_grid_qc(axes[0, 1])
    plot_pair_qc(axes[1, 0])
    plot_clean_qc(axes[1, 1])
    plt.tight_layout()

    if QC_SAVE and (qc_index % QC_SAVE_EVERY_N == 0):
        fig.savefig(output_dir / f"qc_big_{qc_index:03d}.png", dpi=300, bbox_inches="tight")
    if QC_SAVE_INDIVIDUAL and (qc_index % QC_SAVE_EVERY_N == 0):
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        plot_nuclei_qc(ax1)
        fig1.savefig(output_dir / f"qc_nuclei_{qc_index:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        plot_grid_qc(ax2)
        fig2.savefig(output_dir / f"qc_grid_{qc_index:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        plot_pair_qc(ax3)
        fig3.savefig(output_dir / f"qc_pair_{qc_index:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(6, 6))
        plot_clean_qc(ax4)
        fig4.savefig(output_dir / f"qc_clean_{qc_index:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig4)

        # individual stain clean masks (O4 / PDGFRa / MBP)
        for name in ["O4", "PDGFRa", "MBP"]:
            m = ilastik_clean.get(name)
            if m is None:
                continue
            figm, axm = plt.subplots(figsize=(6, 6))
            axm.imshow(m, cmap="gray")
            axm.set_title(f"{name} clean mask")
            axm.axis("off")
            figm.savefig(output_dir / f"qc_{name.lower()}_clean_{qc_index:03d}.png", dpi=300, bbox_inches="tight")
            plt.close(figm)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# =========================
# MAIN
# =========================
def main() -> None:
    bundle_map = build_bundle_map(DATA_ROOT)
    bundles = verify_and_prepare(bundle_map)

    total = len(bundles)
    for idx, bundle in enumerate(bundles, start=1):
        print(f"\n=== Processing {bundle['bundle_id']} ({idx}/{total}) ===")
        print(f"stain: {bundle['staining_img']}")
        print(f"calcium: {bundle['calcium_img']}")
        print(f"landmarks: {bundle['landmarks_csv']}")
        process_bundle(bundle, qc_index=idx - 1)


if __name__ == "__main__":
    main()




