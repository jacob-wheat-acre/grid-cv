"""
prepare_dataset.py – Scaffold and augment a YOLO-format dataset for
electric distribution asset detection.

Designed to work with:
  - InsPLAD (https://github.com/andreluizbvs/InsPLAD)
  - TTPLA   (https://github.com/r09921135/TTPLA)

Usage
-----
    python scripts/prepare_dataset.py \\
        --source ./raw_data/insplad \\
        --output ./datasets/grid_cv \\
        --split 0.8 0.1 0.1

Output layout (YOLO ultralytics format):
    datasets/grid_cv/
      images/{train,val,test}/
      labels/{train,val,test}/
      dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Augmentation pipeline – aerial-inspection-tuned
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(p: float = 0.5) -> A.Compose:
    """
    Returns an Albumentations pipeline suited for drone aerial inspection:
      - Random 90° rotations (drone banks, nadir vs oblique angles)
      - Brightness/contrast shift (time-of-day, cloud shadows)
      - Glare simulation (MotionBlur + RandomBrightnessContrast)
      - Motion blur (wind shake, fast flyby)
      - Hue-saturation for thermal-false-colour drift
    """
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4,
            ),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _read_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return list of (class_id, cx, cy, w, h) from a YOLO label file."""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
    return rows


def _write_yolo_label(label_path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    label_path.write_text(
        "\n".join(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows)
    )


def _augment_and_save(
    img_path: Path,
    lbl_path: Path,
    out_img: Path,
    out_lbl: Path,
    aug: A.Compose,
) -> bool:
    """Apply augmentation to one image+label pair. Returns True on success."""
    image = cv2.imread(str(img_path))
    if image is None:
        return False
    rows = _read_yolo_label(lbl_path)
    bboxes = [(r[1], r[2], r[3], r[4]) for r in rows]
    class_labels = [r[0] for r in rows]

    try:
        result = aug(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception:
        return False

    cv2.imwrite(str(out_img), result["image"])
    new_rows = [
        (cls, *bb)
        for cls, bb in zip(result["class_labels"], result["bboxes"])
    ]
    _write_yolo_label(out_lbl, new_rows)  # type: ignore[arg-type]
    return True


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """
    Collect (image_path, label_path) pairs from a YOLO-format source directory.
    Looks for images in source_dir/images/** and labels in source_dir/labels/**.
    """
    img_root = source_dir / "images"
    lbl_root = source_dir / "labels"
    if not img_root.exists():
        # Flat layout: images in root, labels in root
        img_root = source_dir
        lbl_root = source_dir

    pairs = []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for img in sorted(img_root.rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        # Find matching label
        rel = img.relative_to(img_root)
        lbl = (lbl_root / rel).with_suffix(".txt")
        if not lbl.exists():
            # Try sibling directory
            lbl = img.with_suffix(".txt")
        pairs.append((img, lbl))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train: float,
    val: float,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def prepare(
    source: Path,
    output: Path,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    augment_factor: int = 2,
    class_names: Optional[list[str]] = None,
    seed: int = 42,
) -> None:
    if class_names is None:
        from grid_cv.config import COMPONENT_CLASSES
        class_names = COMPONENT_CLASSES

    print(f"Collecting image/label pairs from {source} …")
    pairs = collect_pairs(source)
    if not pairs:
        print("No pairs found. Check your source directory layout.")
        sys.exit(1)
    print(f"  Found {len(pairs)} pairs.")

    splits = split_pairs(pairs, *split[:2], seed=seed)
    aug = build_augmentation_pipeline()

    for split_name, split_pairs in splits.items():
        img_dir = output / "images" / split_name
        lbl_dir = output / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(split_pairs, desc=f"  {split_name}"):
            # Copy original
            shutil.copy2(img_path, img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dir / lbl_path.with_suffix(".txt").name)

            # Augmented copies for training split only
            if split_name == "train":
                for i in range(augment_factor):
                    aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                    _augment_and_save(
                        img_path, lbl_path,
                        img_dir / aug_name,
                        lbl_dir / (img_path.stem + f"_aug{i}.txt"),
                        aug,
                    )

    # Write dataset.yaml
    yaml_content = (
        f"path: {output.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    (output / "dataset.yaml").write_text(yaml_content)
    print(f"\nDataset written to {output}")
    print(f"  dataset.yaml → {output / 'dataset.yaml'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare YOLO dataset for grid-cv.")
    p.add_argument("--source", type=Path, required=True, help="Raw YOLO dataset root")
    p.add_argument("--output", type=Path, default=Path("datasets/grid_cv"))
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--augment-factor", type=int, default=2,
                   help="Number of augmented copies per training image")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        source=args.source,
        output=args.output,
        split=tuple(args.split),  # type: ignore[arg-type]
        augment_factor=args.augment_factor,
        seed=args.seed,
    )
