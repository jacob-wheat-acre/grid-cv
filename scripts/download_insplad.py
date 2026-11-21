"""
download_insplad.py – Download and reformat InsPLAD for grid-cv training.

InsPLAD (Inspection of Power Lines and Assets Dataset)
  Paper : https://arxiv.org/abs/2311.01979
  Source: https://universe.roboflow.com/insplad

Two sub-datasets are downloaded:
  insplad-det   -> Stage 1 detector training  (asset localisation)
  insplad-fault -> Stage 2 classifier training (defect classification)

Both are remapped to grid-cv class names via config.INSPLAD_DET_MAP and
config.INSPLAD_FAULT_MAP before writing the final dataset.yaml files.

Usage
-----
  # First time: get a free API key at https://app.roboflow.com/ -> Settings
  python scripts/download_insplad.py --api-key YOUR_KEY --output datasets/

  # Skip a sub-dataset you already have
  python scripts/download_insplad.py --api-key YOUR_KEY --output datasets/ --skip-fault
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(pkg: str, install: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Required package '{pkg}' not found. Install with:\n  pip install {install}")
        sys.exit(1)


def _remap_yolo_labels(
    label_dir: Path,
    old_names: list[str],
    new_names: list[str],
    name_map: dict[str, str],
) -> int:
    """
    Rewrite every .txt label file in *label_dir*, remapping class indices
    from *old_names* -> *new_names* via *name_map*.

    Returns the number of files modified.
    """
    # Build old_index -> new_index lookup
    index_map: dict[int, int] = {}
    for old_idx, old_name in enumerate(old_names):
        new_name = name_map.get(old_name, old_name)
        if new_name in new_names:
            index_map[old_idx] = new_names.index(new_name)
        # else: label is dropped (class not in our taxonomy)

    modified = 0
    for lbl_file in label_dir.rglob("*.txt"):
        lines = lbl_file.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            old_idx = int(parts[0])
            if old_idx in index_map:
                parts[0] = str(index_map[old_idx])
                new_lines.append(" ".join(parts))
            # else: drop this annotation (unmapped class)
        lbl_file.write_text("\n".join(new_lines))
        modified += 1
    return modified


def _write_dataset_yaml(
    root: Path,
    names: list[str],
    filename: str = "dataset.yaml",
) -> Path:
    content = (
        f"path: {root.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    out = root / filename
    out.write_text(content)
    return out


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    fmt: str = "yolov8",
) -> Path:
    from roboflow import Roboflow  # type: ignore
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt, location=str(output_dir), overwrite=False)
    return Path(dataset.location)


# ---------------------------------------------------------------------------
# Stage 1 – detector dataset
# ---------------------------------------------------------------------------

def download_detector_dataset(api_key: str, output_root: Path) -> Path:
    """Download insplad-det and remap to grid-cv component classes."""
    from grid_cv.config import COMPONENT_CLASSES, INSPLAD_DET_MAP

    out = output_root / "insplad_det_raw"
    print("Downloading InsPLAD-det (asset detection)...")
    dataset_path = _download_roboflow(
        api_key=api_key,
        workspace="insplad",
        project="insplad-det",
        version=1,
        output_dir=out,
    )

    # Read the original class names from the downloaded dataset.yaml
    import yaml  # type: ignore
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        yaml_path = dataset_path / "dataset.yaml"
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)
    original_names: list[str] = meta.get("names", [])
    print(f"  Original classes ({len(original_names)}): {original_names}")

    # Remap labels
    for split in ("train", "valid", "test"):
        lbl_dir = dataset_path / split / "labels"
        if lbl_dir.exists():
            n = _remap_yolo_labels(lbl_dir, original_names, COMPONENT_CLASSES, INSPLAD_DET_MAP)
            print(f"  Remapped {n} label files in {split}/")

    # Write final dataset.yaml
    final_dir = output_root / "detector"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(dataset_path, final_dir)
    yaml_out = _write_dataset_yaml(final_dir, COMPONENT_CLASSES)
    print(f"  Detector dataset ready -> {final_dir}")
    print(f"  dataset.yaml           -> {yaml_out}")
    return final_dir


# ---------------------------------------------------------------------------
# Stage 2 – classifier dataset
# ---------------------------------------------------------------------------

def download_fault_dataset(api_key: str, output_root: Path) -> Path:
    """
    Download insplad-fault and convert to an ImageFolder layout for the
    YOLOv8 classifier (one subdirectory per class).

    InsPLAD-fault is a detection dataset — we convert each detected crop
    into a classification example by:
      1. Reading each label file
      2. Cropping the image at each bbox
      3. Saving the crop into  <class_name>/<image_stem>_<idx>.jpg
    """
    import cv2
    import numpy as np
    from grid_cv.config import DEFECT_CLASSES, INSPLAD_FAULT_MAP

    out = output_root / "insplad_fault_raw"
    print("\nDownloading InsPLAD-fault (defect detection)...")
    dataset_path = _download_roboflow(
        api_key=api_key,
        workspace="insplad",
        project="insplad-fault",
        version=1,
        output_dir=out,
    )

    import yaml  # type: ignore
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        yaml_path = dataset_path / "dataset.yaml"
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)
    original_names: list[str] = meta.get("names", [])
    print(f"  Original fault classes ({len(original_names)}): {original_names}")

    # Build output ImageFolder structure
    clf_root = output_root / "classifier"
    clf_root.mkdir(parents=True, exist_ok=True)

    crop_count = 0
    padding = 0.15  # match BBOX_PADDING_RATIO from config

    for split in ("train", "valid", "test"):
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        if not img_dir.exists():
            continue

        split_out = clf_root / ("val" if split == "valid" else split)
        split_out.mkdir(exist_ok=True)

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            for idx, line in enumerate(lbl_path.read_text().strip().splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                if cls_idx >= len(original_names):
                    continue
                orig_name = original_names[cls_idx]
                new_name = INSPLAD_FAULT_MAP.get(orig_name)
                if new_name is None or new_name not in DEFECT_CLASSES:
                    continue

                # YOLO format: cx cy w h (normalised)
                cx, cy, bw, bh = (float(x) for x in parts[1:])
                x1 = max(0, int((cx - bw / 2 - bw * padding) * w))
                y1 = max(0, int((cy - bh / 2 - bh * padding) * h))
                x2 = min(w, int((cx + bw / 2 + bw * padding) * w))
                y2 = min(h, int((cy + bh / 2 + bh * padding) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                class_dir = split_out / new_name
                class_dir.mkdir(exist_ok=True)
                out_name = f"{img_path.stem}_{idx:03d}.jpg"
                cv2.imwrite(str(class_dir / out_name), crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                crop_count += 1

    print(f"  Extracted {crop_count} classifier crops -> {clf_root}")

    # Print class distribution
    for split_dir in sorted(clf_root.iterdir()):
        if not split_dir.is_dir():
            continue
        counts = {d.name: len(list(d.glob("*.jpg")))
                  for d in split_dir.iterdir() if d.is_dir()}
        total = sum(counts.values())
        print(f"  {split_dir.name}: {total} crops across {len(counts)} classes")

    return clf_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download InsPLAD and reformat for grid-cv training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Get a free Roboflow API key at:\n"
            "  https://app.roboflow.com/ -> Settings -> Roboflow API\n\n"
            "Example:\n"
            "  python scripts/download_insplad.py \\\n"
            "      --api-key abc123 --output datasets/"
        ),
    )
    p.add_argument("--api-key", required=True, help="Roboflow API key")
    p.add_argument("--output", type=Path, default=Path("datasets"),
                   help="Root output directory (default: datasets/)")
    p.add_argument("--skip-det",   action="store_true",
                   help="Skip InsPLAD-det (detector dataset)")
    p.add_argument("--skip-fault", action="store_true",
                   help="Skip InsPLAD-fault (classifier dataset)")
    return p.parse_args()


if __name__ == "__main__":
    _require("roboflow", "roboflow")
    _require("yaml",     "pyyaml")
    _require("cv2",      "opencv-python")

    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.skip_det:
        download_detector_dataset(args.api_key, args.output)

    if not args.skip_fault:
        download_fault_dataset(args.api_key, args.output)

    print("\nAll done. Next steps:")
    print("  python scripts/train.py detector "
          "--data datasets/detector/dataset.yaml")
    print("  python scripts/train.py classifier "
          "--data datasets/classifier/")
