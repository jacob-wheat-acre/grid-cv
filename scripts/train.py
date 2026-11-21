"""
train.py – YOLOv8/YOLO11 training & validation harness for grid-cv.

Usage
-----
# Train detector (Stage 1)
python scripts/train.py detector \\
    --data datasets/grid_cv/dataset.yaml \\
    --epochs 100 --imgsz 640 --batch 16

# Train classifier (Stage 2 — defect classification)
python scripts/train.py classifier \\
    --data datasets/defects \\
    --epochs 50 --imgsz 224 --batch 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def _best_device() -> str:
    """
    Return the best available compute device string for Ultralytics:
      'mps'  – Apple Silicon (M1/M2/M3/M4) via Metal Performance Shaders
      '0'    – first CUDA GPU
      'cpu'  – fallback
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Shared training configuration
# ---------------------------------------------------------------------------

DETECTOR_BASE_MODEL   = "yolov8m.pt"      # medium: good balance for aerial
CLASSIFIER_BASE_MODEL = "yolov8m-cls.pt"

DETECTOR_HYPERPARAMS = {
    "lr0":        0.01,
    "lrf":        0.01,
    "momentum":   0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box":        7.5,
    "cls":        0.5,
    "dfl":        1.5,
    "degrees":    10.0,   # rotation augmentation
    "translate":  0.1,
    "scale":      0.5,
    "flipud":     0.3,    # vertical flip common for nadir imagery
    "fliplr":     0.5,
    "mosaic":     1.0,
    "mixup":      0.1,
    "copy_paste": 0.1,
}


# ---------------------------------------------------------------------------
# Detector training
# ---------------------------------------------------------------------------

def train_detector(
    data_yaml: Path,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    output_dir: Path = Path("runs/detect"),
    resume: bool = False,
    workers: int = 8,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    resolved_device = device or _best_device()
    print(f"Device: {resolved_device}")
    model = YOLO(DETECTOR_BASE_MODEL)
    print(f"Training detector on {data_yaml}")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(output_dir),
        name="grid_cv_detector",
        exist_ok=True,
        resume=resume,
        workers=workers,
        **DETECTOR_HYPERPARAMS,
    )

    # Validate on val split
    metrics = model.val()
    print("\n=== Detector Validation Metrics ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.p.mean():.4f}")
    print(f"  Recall:   {metrics.box.r.mean():.4f}")

    # Export best weights to models/
    best_weights = Path(output_dir) / "grid_cv_detector" / "weights" / "best.pt"
    if best_weights.exists():
        from grid_cv.config import WEIGHTS_DETECTOR
        WEIGHTS_DETECTOR.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_weights, WEIGHTS_DETECTOR)
        print(f"\nBest weights saved to {WEIGHTS_DETECTOR}")


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_classifier(
    data_dir: Path,
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 32,
    device: str | None = None,
    output_dir: Path = Path("runs/classify"),
    resume: bool = False,
) -> None:
    """
    data_dir should follow ImageFolder layout:
        data_dir/
          healthy/     *.jpg ...
          broken_shed/ *.jpg ...
          ...
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    resolved_device = device or _best_device()
    print(f"Device: {resolved_device}")
    model = YOLO(CLASSIFIER_BASE_MODEL)
    print(f"Training classifier on {data_dir}")
    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(output_dir),
        name="grid_cv_classifier",
        exist_ok=True,
        resume=resume,
    )

    metrics = model.val()
    print("\n=== Classifier Validation Metrics ===")
    print(f"  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")

    best_weights = Path(output_dir) / "grid_cv_classifier" / "weights" / "best.pt"
    if best_weights.exists():
        from grid_cv.config import WEIGHTS_CLASSIFIER
        WEIGHTS_CLASSIFIER.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_weights, WEIGHTS_CLASSIFIER)
        print(f"\nBest weights saved to {WEIGHTS_CLASSIFIER}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train grid-cv detection / classification models.")
    sub = p.add_subparsers(dest="stage", required=True)

    # --- detector ---
    det = sub.add_parser("detector", help="Train Stage-1 YOLOv8 detector")
    det.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml")
    det.add_argument("--epochs", type=int, default=100)
    det.add_argument("--imgsz",  type=int, default=640)
    det.add_argument("--batch",  type=int, default=16)
    det.add_argument("--device", default="0")
    det.add_argument("--output", type=Path, default=Path("runs/detect"))
    det.add_argument("--resume", action="store_true")
    det.add_argument("--workers", type=int, default=8)

    # --- classifier ---
    cls = sub.add_parser("classifier", help="Train Stage-2 defect classifier")
    cls.add_argument("--data", type=Path, required=True,
                     help="ImageFolder root directory with class subdirs")
    cls.add_argument("--epochs", type=int, default=50)
    cls.add_argument("--imgsz",  type=int, default=224)
    cls.add_argument("--batch",  type=int, default=32)
    cls.add_argument("--device", default="0")
    cls.add_argument("--output", type=Path, default=Path("runs/classify"))
    cls.add_argument("--resume", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.stage == "detector":
        train_detector(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_dir=args.output,
            resume=args.resume,
            workers=args.workers,
        )
    else:
        train_classifier(
            data_dir=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_dir=args.output,
            resume=args.resume,
        )
