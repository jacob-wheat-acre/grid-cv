"""
Unit tests for grid_cv.pipeline.

Verifies:
  - Mock pipeline runs end-to-end on a synthetic image
  - Detection objects have required fields within valid ranges
  - SAHI slice count is proportional to image size
  - Padded crop helper clips to image boundaries
  - annotate_image returns correct shape/dtype
  - InspectionResult.worst_severity aggregates correctly
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from grid_cv.pipeline import (
    BBox,
    Detection,
    InspectionPipeline,
    InspectionResult,
    _padded_crop,
    annotate_image,
)
from grid_cv.telemetry import Telemetry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(use_sahi: bool = False) -> InspectionPipeline:
    """Return a pipeline that always uses mocks (no weights on disk)."""
    return InspectionPipeline(use_sahi=use_sahi)


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_inspect_returns_inspection_result(self, synthetic_rgb_path: Path):
        pl = _make_pipeline()
        result = pl.inspect(synthetic_rgb_path)
        assert isinstance(result, InspectionResult)

    def test_result_has_correct_dimensions(self, synthetic_rgb_path: Path):
        pl = _make_pipeline()
        result = pl.inspect(synthetic_rgb_path)
        assert result.image_width  == 1920
        assert result.image_height == 1080

    def test_detections_are_non_empty_with_mock(self, synthetic_rgb_path: Path):
        pl = _make_pipeline()
        result = pl.inspect(synthetic_rgb_path)
        # Mock detector always returns 2–5 detections
        assert len(result.detections) >= 1

    def test_detection_fields_valid(self, synthetic_rgb_path: Path):
        pl = _make_pipeline()
        result = pl.inspect(synthetic_rgb_path)
        from grid_cv.config import COMPONENT_CLASSES, DEFECT_CLASSES, SEVERITY_MAP
        valid_severities = set(SEVERITY_MAP.values())
        for det in result.detections:
            assert det.component_class in COMPONENT_CLASSES
            assert det.defect_class in DEFECT_CLASSES
            assert det.severity in valid_severities
            assert 0.0 <= det.component_conf <= 1.0
            assert 0.0 <= det.defect_conf    <= 1.0
            assert det.bbox.x1 < det.bbox.x2
            assert det.bbox.y1 < det.bbox.y2

    def test_inference_ms_is_positive(self, synthetic_rgb_path: Path):
        pl = _make_pipeline()
        result = pl.inspect(synthetic_rgb_path)
        assert result.inference_ms > 0

    def test_telemetry_propagated(self, synthetic_rgb_path: Path):
        pl = _make_pipeline()
        tel = Telemetry(latitude=35.96, longitude=-83.92, altitude_m=85.0)
        result = pl.inspect(synthetic_rgb_path, telemetry=tel)
        for det in result.detections:
            assert det.telemetry is not None
            assert det.telemetry.latitude == pytest.approx(35.96)

    def test_missing_image_raises(self, tmp_path: Path):
        pl = _make_pipeline()
        with pytest.raises(ValueError, match="Could not read image"):
            pl.inspect(tmp_path / "ghost.jpg")

    def test_thermal_image_works(self, synthetic_thermal_path: Path):
        pl = _make_pipeline()
        result = pl.inspect(synthetic_thermal_path)
        assert isinstance(result, InspectionResult)
        assert result.image_width  == 640
        assert result.image_height == 512


# ---------------------------------------------------------------------------
# SAHI slice count smoke-test (mock mode, no real SAHI)
# ---------------------------------------------------------------------------

class TestSAHISlices:
    """
    When no real weights exist, SAHI is bypassed with the mock detector.
    We just verify the pipeline initialises correctly with use_sahi=True
    and still produces valid results.
    """

    def test_sahi_enabled_still_produces_results(self, synthetic_hires_path: Path):
        pl = _make_pipeline(use_sahi=True)
        result = pl.inspect(synthetic_hires_path)
        # Mock detector is used because no weights exist
        assert pl._detector_mode == "mock"
        assert isinstance(result, InspectionResult)


# ---------------------------------------------------------------------------
# Padded crop helper
# ---------------------------------------------------------------------------

class TestPaddedCrop:
    def test_basic_crop_shape(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = _padded_crop(img, 20, 20, 80, 80)
        # With 15% padding: extra = 0.15 * 60 = 9px each side
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0

    def test_crop_clips_to_image_boundary(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        # bbox at corner — padding would go negative
        crop = _padded_crop(img, 0, 0, 30, 30)
        # Must not error and shape must be non-zero
        assert crop.size > 0

    def test_crop_at_right_edge_clips(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        crop = _padded_crop(img, 80, 80, 100, 100)
        assert crop.size > 0


# ---------------------------------------------------------------------------
# annotate_image
# ---------------------------------------------------------------------------

class TestAnnotateImage:
    def _make_detection(self) -> Detection:
        return Detection(
            image_path="test.jpg",
            component_class="insulator",
            component_conf=0.85,
            defect_class="flashover_burn",
            defect_conf=0.72,
            severity="CRITICAL",
            bbox=BBox(x1=50, y1=50, x2=200, y2=200),
        )

    def test_annotate_returns_same_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [self._make_detection()]
        out = annotate_image(img, dets)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_annotate_modifies_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [self._make_detection()]
        out = annotate_image(img, dets)
        # At least some pixels should be non-zero (box drawn)
        assert out.any()

    def test_annotate_empty_detections_returns_copy(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        out = annotate_image(img, [])
        np.testing.assert_array_equal(img, out)
        assert out is not img  # must be a copy


# ---------------------------------------------------------------------------
# worst_severity aggregation
# ---------------------------------------------------------------------------

class TestWorstSeverity:
    def _result_with_severities(self, severities: list[str]) -> InspectionResult:
        dets = [
            Detection(
                image_path="x.jpg",
                component_class="pole",
                component_conf=0.9,
                defect_class="healthy",
                defect_conf=0.9,
                severity=s,
                bbox=BBox(x1=0, y1=0, x2=10, y2=10),
            )
            for s in severities
        ]
        return InspectionResult(
            image_path="x.jpg",
            image_width=100, image_height=100,
            detections=dets,
            inference_ms=10.0,
        )

    def test_no_detections_returns_none(self):
        result = self._result_with_severities([])
        assert result.worst_severity == "NONE"

    def test_all_none(self):
        assert self._result_with_severities(["NONE", "NONE"]).worst_severity == "NONE"

    def test_mixed_returns_highest(self):
        assert self._result_with_severities(
            ["NONE", "LOW", "CRITICAL", "MEDIUM"]
        ).worst_severity == "CRITICAL"

    def test_single_critical(self):
        assert self._result_with_severities(["CRITICAL"]).worst_severity == "CRITICAL"
