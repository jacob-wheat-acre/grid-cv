"""Central configuration: thresholds, slice sizes, severity mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Final[Path] = Path(__file__).parent.parent
MODEL_DIR: Final[Path] = ROOT_DIR / "models"
WEIGHTS_DETECTOR: Final[Path] = MODEL_DIR / "detector.pt"
WEIGHTS_CLASSIFIER: Final[Path] = MODEL_DIR / "classifier.pt"

# ---------------------------------------------------------------------------
# SAHI slice parameters
# ---------------------------------------------------------------------------
SLICE_HEIGHT: Final[int] = 640
SLICE_WIDTH: Final[int] = 640
OVERLAP_RATIO: Final[float] = 0.20   # 20% overlap in both dimensions

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
DETECTOR_CONF_THRESHOLD: Final[float] = 0.30
DETECTOR_IOU_THRESHOLD: Final[float] = 0.45
CLASSIFIER_CONF_THRESHOLD: Final[float] = 0.40
BBOX_PADDING_RATIO: Final[float] = 0.15  # 15% padding around detected crop

# ---------------------------------------------------------------------------
# Component classes (Stage 1 — localization)
# ---------------------------------------------------------------------------
COMPONENT_CLASSES: Final[list[str]] = [
    "insulator",
    "splice",
    "arrester",
    "crossarm",
    "transformer",
    "pole",
]

# ---------------------------------------------------------------------------
# Defect / health classes (Stage 2 — classification)
# ---------------------------------------------------------------------------
DEFECT_CLASSES: Final[list[str]] = [
    "healthy",
    "broken_shed",
    "flashover_burn",
    "corrosion",
    "oil_leak",
    "vegetation_encroachment",
]

# ---------------------------------------------------------------------------
# Severity mapping
# defect_class -> severity tier
# ---------------------------------------------------------------------------
SEVERITY_MAP: Final[dict[str, str]] = {
    "healthy":                 "NONE",
    "broken_shed":             "MEDIUM",
    "flashover_burn":          "CRITICAL",
    "corrosion":               "LOW",
    "oil_leak":                "CRITICAL",
    "vegetation_encroachment": "MEDIUM",
}

SEVERITY_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "NONE":     (0,   200,   0),   # green  (BGR)
    "LOW":      (0,   165, 255),   # orange
    "MEDIUM":   (0,   140, 255),   # amber
    "CRITICAL": (0,     0, 220),   # red
}

SEVERITY_ORDER: Final[dict[str, int]] = {
    "NONE": 0, "LOW": 1, "MEDIUM": 2, "CRITICAL": 3
}

# ---------------------------------------------------------------------------
# API & CLI defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Final[Path] = ROOT_DIR / "results"
API_HOST: Final[str] = "0.0.0.0"
API_PORT: Final[int] = 8000
MAX_UPLOAD_MB: Final[int] = 50
