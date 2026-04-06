#!/usr/bin/env python3
"""
Batch-export ilastik segmentations for ensheathment crops.

What it does
------------
1. Scans the root cropped-image folder for TIFFs exported from the Fiji tracing step.
2. Matches each image to the correct biological replicate folder (`n1`..`n4`) using the
   date-code mapping supplied by Jack:
      2403 -> n1
      2703 -> n2
      0404 -> n3
      1605 -> n4
3. Routes each image to the matching ilastik project (`.ilp`) for:
      - MBP / C2: full-cell and soma crops
      - C3: full-cell crops only
4. Saves ilastik simple-segmentation TIFFs under `ensheathment/ilastik_outputs/...`.
5. Writes a manifest CSV and preview PNGs.
6. Optionally pops up class-colour preview figures so the user can visually confirm
   which integer label corresponds to which class.

Notes
-----
- This script is intended as the first "get ilastik outputs" step before downstream
  ensheathment quantification.
- It prefers the canonical TIFF when duplicate Windows copies like " (1)" exist.
- Preview figures are saved even if interactive display is disabled.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


DEFAULT_ILASTIK_EXE = Path(r"C:\Program Files\ilastik-1.4.0.post1\ilastik.exe")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "original_data" / "ensheathment" / "traced_exports"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "ilastik_outputs"

DATE_TO_BIO_REP = {
    "2403": "n1",
    "2703": "n2",
    "0404": "n3",
    "1605": "n4",
}

MARKER_CONFIG = {
    "mbp": {
        "channel": "c2",
        "project_root": "c2_ilastik",
        "region_kinds": {"cell", "soma"},
        "preview_title": "MBP / C2 ilastik classes",
    },
    "c3": {
        "channel": "c3",
        "project_root": "c3_ilastik",
        "region_kinds": {"cell"},
        "preview_title": "C3 ilastik classes",
    },
}

FILE_RE = re.compile(
    r"^(?P<prefix>[A-Z])"
    r"(?P<date_code>\d{4})_"
    r"(?P<scene>\d+)_"
    r"cell(?P<cell>\d+)_"
    r"(?P<region>cell|soma)_"
    r"(?P<channel>c[123])$"
)


@dataclass(frozen=True)
class CropRecord:
    source_path: Path
    prefix: str
    date_code: str
    scene: int
    cell_index: int
    region: str
    channel: str
    bio_rep: str
    normalized_stem: str


@dataclass(frozen=True)
class IlastikJob:
    marker: str
    source: CropRecord
    ilp_path: Path
    output_path: Path


def ilastik_output_path(output_dir: Path, stem: str) -> Path:
    # Ilastik normalizes TIFF output names to `.tiff` even if `.tif` is requested.
    return output_dir / f"{stem}.tiff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ilastik-exe", type=Path, default=DEFAULT_ILASTIK_EXE)
    parser.add_argument(
        "--export-source",
        default="Simple Segmentation",
        help="Ilastik export source. Defaults to 'Simple Segmentation'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run ilastik even if the output TIFF already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the job manifest and preview plan; do not run ilastik.",
    )
    parser.add_argument(
        "--no-show-previews",
        action="store_true",
        help="Skip preview generation entirely and just write segmentation outputs.",
    )
    return parser.parse_args()


def canonical_name(path: Path) -> str:
    return re.sub(r" \(\d+\)$", "", path.stem)


def collect_root_tiffs(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.glob("*.tif")
        if path.is_file()
    )


def pick_canonical_files(paths: Iterable[Path]) -> list[Path]:
    chosen: dict[str, Path] = {}
    for path in paths:
        key = canonical_name(path)
        current = chosen.get(key)
        if current is None:
            chosen[key] = path
            continue

        current_is_copy = current.stem != key
        new_is_copy = path.stem != key
        if current_is_copy and not new_is_copy:
            chosen[key] = path
        elif not current_is_copy and new_is_copy:
            continue
        else:
            print(
                f"[warn] duplicate TIFFs for '{key}': keeping '{current.name}', skipping '{path.name}'",
                file=sys.stderr,
            )
    return sorted(chosen.values())


def parse_crop_record(path: Path) -> CropRecord | None:
    match = FILE_RE.match(canonical_name(path))
    if not match:
        return None

    date_code = match.group("date_code")
    bio_rep = DATE_TO_BIO_REP.get(date_code)
    if bio_rep is None:
        raise ValueError(f"No biological replicate mapping for date code {date_code} ({path.name})")

    return CropRecord(
        source_path=path,
        prefix=match.group("prefix"),
        date_code=date_code,
        scene=int(match.group("scene")),
        cell_index=int(match.group("cell")),
        region=match.group("region"),
        channel=match.group("channel"),
        bio_rep=bio_rep,
        normalized_stem=canonical_name(path),
    )


def collect_records(input_root: Path) -> list[CropRecord]:
    records: list[CropRecord] = []
    skipped: list[str] = []

    for path in pick_canonical_files(collect_root_tiffs(input_root)):
        record = parse_crop_record(path)
        if record is None:
            skipped.append(path.name)
            continue
        records.append(record)

    if skipped:
        print(f"[info] skipped {len(skipped)} non-crop TIFFs in root folder", file=sys.stderr)

    return sorted(
        records,
        key=lambda r: (r.bio_rep, r.prefix, r.date_code, r.scene, r.cell_index, r.region, r.channel),
    )


def build_jobs(records: list[CropRecord], input_root: Path, output_root: Path) -> list[IlastikJob]:
    jobs: list[IlastikJob] = []

    for record in records:
        for marker, cfg in MARKER_CONFIG.items():
            if record.channel != cfg["channel"]:
                continue
            if record.region not in cfg["region_kinds"]:
                continue

            ilp_path = input_root / cfg["project_root"] / record.bio_rep / f"{record.bio_rep}.ilp"
            output_dir = output_root / marker / record.bio_rep
            output_path = ilastik_output_path(output_dir, f"{record.normalized_stem}__{marker}_seg")
            jobs.append(
                IlastikJob(
                    marker=marker,
                    source=record,
                    ilp_path=ilp_path,
                    output_path=output_path,
                )
            )

    return jobs


def validate_jobs(jobs: list[IlastikJob], ilastik_exe: Path) -> None:
    problems: list[str] = []

    if not ilastik_exe.exists():
        problems.append(f"Ilastik executable not found: {ilastik_exe}")

    if not jobs:
        problems.append("No ilastik jobs were discovered.")

    for job in jobs:
        if not job.source.source_path.exists():
            problems.append(f"Missing input TIFF: {job.source.source_path}")
        if not job.ilp_path.exists():
            problems.append(f"Missing ilastik project: {job.ilp_path}")

    if problems:
        joined = "\n".join(f"- {problem}" for problem in problems)
        raise FileNotFoundError(f"Validation failed:\n{joined}")


def write_manifest(jobs: list[IlastikJob], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "job_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "marker",
                "bio_rep",
                "prefix",
                "date_code",
                "scene",
                "cell_index",
                "region",
                "channel",
                "source_path",
                "ilp_path",
                "output_path",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "marker": job.marker,
                    "bio_rep": job.source.bio_rep,
                    "prefix": job.source.prefix,
                    "date_code": job.source.date_code,
                    "scene": job.source.scene,
                    "cell_index": job.source.cell_index,
                    "region": job.source.region,
                    "channel": job.source.channel,
                    "source_path": str(job.source.source_path),
                    "ilp_path": str(job.ilp_path),
                    "output_path": str(job.output_path),
                }
            )
    return manifest_path


def run_ilastik(job: IlastikJob, ilastik_exe: Path, export_source: str, overwrite: bool) -> Path:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if job.output_path.exists() and not overwrite:
        print(f"[skip] existing output: {job.output_path.name}")
        return job.output_path

    cmd = [
        str(ilastik_exe),
        "--headless",
        f"--project={job.ilp_path}",
        f"--export_source={export_source}",
        "--output_format=tiff",
        f"--output_filename_format={job.output_path}",
        str(job.source.source_path),
    ]

    print(f"[run] {job.marker} | {job.source.bio_rep} | {job.source.source_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ilastik failed for {job.source.source_path.name}\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
    if not job.output_path.exists():
        raise FileNotFoundError(f"Ilastik finished but did not write {job.output_path}")
    return job.output_path


def class_colormap(max_label: int) -> tuple[ListedColormap, BoundaryNorm]:
    palette = [
        "#000000",
        "#ff3366",
        "#00c8ff",
        "#ffd400",
        "#44dd88",
        "#8a5cff",
        "#ff8c1a",
        "#ffffff",
    ]
    colors = [palette[i % len(palette)] for i in range(max_label + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_label + 1.5, 1), cmap.N)
    return cmap, norm


def pick_preview_jobs(jobs: list[IlastikJob]) -> dict[str, list[IlastikJob]]:
    grouped: dict[str, dict[str, IlastikJob]] = {"mbp": {}, "c3": {}}

    for job in jobs:
        if job.marker not in grouped:
            continue
        existing = grouped[job.marker].get(job.source.bio_rep)
        if existing is None:
            grouped[job.marker][job.source.bio_rep] = job
            continue

        # Prefer full-cell previews over soma previews for MBP.
        if existing.source.region == "soma" and job.source.region == "cell":
            grouped[job.marker][job.source.bio_rep] = job

    ordered: dict[str, list[IlastikJob]] = {}
    rep_order = ["n1", "n2", "n3", "n4"]
    for marker, by_rep in grouped.items():
        ordered[marker] = [by_rep[rep] for rep in rep_order if rep in by_rep]
    return ordered


def save_and_show_preview(
    marker: str,
    jobs: list[IlastikJob],
    output_root: Path,
    show_preview: bool,
) -> Path | None:
    if not jobs:
        return None

    arrays: list[np.ndarray] = []
    labels_union: set[int] = set()
    for job in jobs:
        arr = np.asarray(tifffile.imread(job.output_path)).squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D ilastik output for {job.output_path}, found shape {arr.shape}")
        arrays.append(arr)
        labels_union.update(int(v) for v in np.unique(arr))

    max_label = max(labels_union) if labels_union else 0
    cmap, norm = class_colormap(max_label)

    n = len(jobs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, job, arr in zip(axes[0], jobs, arrays):
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        labels_here = ", ".join(str(int(v)) for v in np.unique(arr))
        ax.set_title(
            f"{job.source.bio_rep} | {job.source.source_path.name}\nlabels: {labels_here}",
            fontsize=9,
        )
        ax.axis("off")

    legend_labels = sorted(labels_union)
    handles = [
        Patch(facecolor=cmap(norm(label)), edgecolor="none", label=f"class {label}")
        for label in legend_labels
    ]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)))
    fig.suptitle(MARKER_CONFIG[marker]["preview_title"], fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{marker}_class_preview.png"
    fig.savefig(preview_path, dpi=160, bbox_inches="tight")
    print(f"[preview] saved {preview_path}")

    if show_preview:
        plt.show()
    else:
        plt.close(fig)

    return preview_path


def main() -> None:
    args = parse_args()

    records = collect_records(args.input_root)
    jobs = build_jobs(records, args.input_root, args.output_root)
    validate_jobs(jobs, args.ilastik_exe)
    manifest_path = write_manifest(jobs, args.output_root)
    print(f"[manifest] wrote {manifest_path}")
    print(f"[jobs] total={len(jobs)}")

    if args.dry_run:
        preview_jobs = pick_preview_jobs(jobs)
        for marker, items in preview_jobs.items():
            print(f"[dry-run] {marker} preview reps: {[job.source.bio_rep for job in items]}")
        return

    for job in jobs:
        run_ilastik(
            job=job,
            ilastik_exe=args.ilastik_exe,
            export_source=args.export_source,
            overwrite=args.overwrite,
        )

    if args.no_show_previews:
        print("[preview] skipped by --no-show-previews")
        return

    preview_jobs = pick_preview_jobs(jobs)
    for marker, items in preview_jobs.items():
        try:
            save_and_show_preview(
                marker=marker,
                jobs=items,
                output_root=args.output_root,
                show_preview=True,
            )
        except Exception as exc:
            print(f"[preview] failed for {marker}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
