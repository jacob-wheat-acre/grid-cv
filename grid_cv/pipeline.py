"""
pipeline.py – Two-stage inference pipeline.

Stage 1  Component Localization
        SAHI wrapping YOLOv8/YOLO11 at 640×640 with 20% overlap.
        Gracefully falls back to a mock detector when real weights are absent.

Stage 2  Defect Classification
        Crops each detection (+ 15% padding) and runs a secondary classifier.
        Falls back to a mock classifier when real weights are absent.

Output is a list of ``Detection`` Pydantic objects ready for geo.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from grid_cv.config import (
    BBOX_PADDING_RATIO,
    CLASSIFIER_CONF_THRESHOLD,
    COMPONENT_CLASSES,
    DEFECT_CLASSES,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    OVERLAP_RATIO,
    SEVERITY_COLORS,
    SEVERITY_MAP,
    SLICE_HEIGHT,
    SLICE_WIDTH,
    WEIGHTS_CLASSIFIER,
    WEIGHTS_DETECTOR,
)
from grid_cv.telemetry import Telemetry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height


class Detection(BaseModel):
    """Single asset detection with defect classification."""

    image_path: str
    component_class: str
    component_conf: float = Field(ge=0.0, le=1.0)
    defect_class: str
    defect_conf: float = Field(ge=0.0, le=1.0)
    severity: str
    bbox: BBox
    telemetry: Optional[Telemetry] = None


class InspectionResult(BaseModel):
    """Full inspection result for one image."""

    image_path: str
    image_width: int
    image_height: int
    detections: list[Detection]
    inference_ms: float
    telemetry: Optional[Telemetry] = None

    @property
    def worst_severity(self) -> str:
        from grid_cv.config import SEVERITY_ORDER
        if not self.detections:
            return "NONE"
        return max(
            (d.severity for d in self.detections),
            key=lambda s: SEVERITY_ORDER.get(s, 0),
        )


# ---------------------------------------------------------------------------
# Mock detectors (used when no real weights exist)
# ---------------------------------------------------------------------------

class _MockDetector:
    """Returns synthetic detections spread across the image."""

    def predict(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        rng = np.random.default_rng(seed=42)
        n = int(rng.integers(2, 6))
        results = []
        for _ in range(n):
            x1 = float(rng.integers(0, w - 100))
            y1 = float(rng.integers(0, h - 100))
            x2 = float(x1 + rng.integers(50, 200))
            y2 = float(y1 + rng.integers(50, 200))
            x2, y2 = min(x2, w), min(y2, h)
            cls_idx = int(rng.integers(0, len(COMPONENT_CLASSES)))
            conf = float(rng.uniform(0.45, 0.95))
            results.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_name": COMPONENT_CLASSES[cls_idx],
                "conf": conf,
            })
        return results


class _MockClassifier:
    """Returns a deterministic defect label for a given crop."""

    def predict(self, crop: np.ndarray) -> tuple[str, float]:
        rng = np.random.default_rng(seed=int(crop.mean() * 1000) % (2**32))
        idx = int(rng.integers(0, len(DEFECT_CLASSES)))
        conf = float(rng.uniform(0.50, 0.92))
        return DEFECT_CLASSES[idx], conf


# ---------------------------------------------------------------------------
# Real detector/classifier wrappers
# ---------------------------------------------------------------------------

def _load_sahi_detector(weights: Path):
    """Return a SAHI AutoDetectionModel or raise ImportError."""
    from sahi import AutoDetectionModel  # type: ignore
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=DETECTOR_CONF_THRESHOLD,
        device="cpu",
    )


def _run_sahi(model, image_path: str) -> list[dict]:
    """Run SAHI sliced prediction and return normalised result dicts."""
    from sahi.predict import get_sliced_prediction  # type: ignore
    result = get_sliced_prediction(
        image=image_path,
        detection_model=model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMM",
        postprocess_match_threshold=DETECTOR_IOU_THRESHOLD,
        verbose=0,
    )
    out = []
    for obj in result.object_prediction_list:
        b = obj.bbox
        out.append({
            "x1": b.minx, "y1": b.miny, "x2": b.maxx, "y2": b.maxy,
            "class_name": obj.category.name,
            "conf": obj.score.value,
        })
    return out


def _load_classifier(weights: Path):
    """Load a torchvision/ultralytics classifier checkpoint."""
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO(str(weights))
    except Exception as exc:
        log.warning("Could not load classifier weights %s: %s", weights, exc)
        return None


def _run_classifier(model, crop: np.ndarray) -> tuple[str, float]:
    """Run a YOLO classify model on a BGR crop, return (class_name, conf)."""
    results = model(crop, verbose=False)
    top = results[0].probs
    idx = int(top.top1)
    conf = float(top.top1conf)
    name = results[0].names.get(idx, DEFECT_CLASSES[0])
    return name, conf


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def _padded_crop(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Return a padded crop from *image* with BBOX_PADDING_RATIO extra margin."""
    h, w = image.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    px = bw * BBOX_PADDING_RATIO
    py = bh * BBOX_PADDING_RATIO
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    return image[cy1:cy2, cx1:cx2]


# ---------------------------------------------------------------------------
# Annotation renderer
# ---------------------------------------------------------------------------

def annotate_image(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on *image* (BGR). Returns a copy."""
    out = image.copy()
    for det in detections:
        color = SEVERITY_COLORS.get(det.severity, (200, 200, 200))
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.component_class} | {det.defect_class} ({det.severity})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class InspectionPipeline:
    """
    Two-stage drone imagery inspection pipeline.

    Parameters
    ----------
    detector_weights:
        Path to YOLOv8/YOLO11 detection weights.  Uses mock if file absent.
    classifier_weights:
        Path to YOLOv8-classify weights.  Uses mock if file absent.
    use_sahi:
        If False, skip SAHI slicing (useful for small test images).
    """

    def __init__(
        self,
        detector_weights: Optional[Path] = None,
        classifier_weights: Optional[Path] = None,
        use_sahi: bool = True,
    ) -> None:
        det_w = detector_weights or WEIGHTS_DETECTOR
        cls_w = classifier_weights or WEIGHTS_CLASSIFIER

        # --- Stage 1 ---
        self._use_sahi = use_sahi
        if det_w.exists():
            try:
                self._detector = _load_sahi_detector(det_w)
                self._detector_mode = "sahi"
                log.info("Loaded SAHI detector from %s", det_w)
            except Exception as exc:
                log.warning("SAHI load failed (%s); falling back to mock detector.", exc)
                self._detector = _MockDetector()
                self._detector_mode = "mock"
        else:
            log.info("No detector weights at %s — using mock detector.", det_w)
            self._detector = _MockDetector()
            self._detector_mode = "mock"

        # --- Stage 2 ---
        if cls_w.exists():
            model = _load_classifier(cls_w)
            if model is not None:
                self._classifier = model
                self._classifier_mode = "real"
                log.info("Loaded classifier from %s", cls_w)
            else:
                self._classifier = _MockClassifier()
                self._classifier_mode = "mock"
        else:
            log.info("No classifier weights at %s — using mock classifier.", cls_w)
            self._classifier = _MockClassifier()
            self._classifier_mode = "mock"

    # ------------------------------------------------------------------
    def inspect(self, image_path: Path | str, telemetry: Optional[Telemetry] = None) -> InspectionResult:
        """Run full two-stage pipeline on a single image."""
        image_path = Path(image_path)
        t0 = time.perf_counter()

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = image.shape[:2]

        # Stage 1 – localization
        if self._detector_mode == "sahi" and self._use_sahi:
            raw_detections = _run_sahi(self._detector, str(image_path))
        else:
            raw_detections = self._detector.predict(image)

        detections: list[Detection] = []
        for raw in raw_detections:
            if raw["conf"] < DETECTOR_CONF_THRESHOLD:
                continue

            # Stage 2 – classification
            crop = _padded_crop(image, raw["x1"], raw["y1"], raw["x2"], raw["y2"])
            if crop.size == 0:
                continue

            if self._classifier_mode == "real":
                defect_class, defect_conf = _run_classifier(self._classifier, crop)
            else:
                defect_class, defect_conf = self._classifier.predict(crop)

            if defect_conf < CLASSIFIER_CONF_THRESHOLD:
                defect_class = DEFECT_CLASSES[0]  # fall back to "healthy"

            severity = SEVERITY_MAP.get(defect_class, "LOW")
            detections.append(Detection(
                image_path=str(image_path),
                component_class=raw["class_name"],
                component_conf=raw["conf"],
                defect_class=defect_class,
                defect_conf=defect_conf,
                severity=severity,
                bbox=BBox(x1=raw["x1"], y1=raw["y1"], x2=raw["x2"], y2=raw["y2"]),
                telemetry=telemetry,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return InspectionResult(
            image_path=str(image_path),
            image_width=w,
            image_height=h,
            detections=detections,
            inference_ms=elapsed_ms,
            telemetry=telemetry,
        )
