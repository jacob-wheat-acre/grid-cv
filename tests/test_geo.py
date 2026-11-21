"""
Unit tests for grid_cv.geo.

Verifies:
  - results_to_geojson produces a valid FeatureCollection schema
  - Each Feature has correct properties from Detection
  - GPS-less detections produce null geometry (valid GeoJSON)
  - GPS-bearing detections produce Point geometry with [lon, lat]
  - save_geojson writes parseable JSON to disk
  - save_csv writes expected columns
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from grid_cv.geo import results_to_geojson, save_csv, save_geojson
from grid_cv.pipeline import BBox, Detection, InspectionResult
from grid_cv.telemetry import Telemetry


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_result(
    n_detections: int = 2,
    lat: float | None = 35.96,
    lon: float | None = -83.92,
    alt: float | None = 85.0,
) -> InspectionResult:
    tel = Telemetry(latitude=lat, longitude=lon, altitude_m=alt) if lat is not None else Telemetry()
    dets = [
        Detection(
            image_path="test_image.jpg",
            component_class="insulator",
            component_conf=0.85,
            defect_class="flashover_burn",
            defect_conf=0.78,
            severity="CRITICAL",
            bbox=BBox(x1=100.0, y1=50.0, x2=250.0, y2=180.0),
            telemetry=tel,
        )
        for _ in range(n_detections)
    ]
    return InspectionResult(
        image_path="test_image.jpg",
        image_width=1920,
        image_height=1080,
        detections=dets,
        inference_ms=42.0,
        telemetry=tel,
    )


# ---------------------------------------------------------------------------
# GeoJSON structure tests
# ---------------------------------------------------------------------------

class TestGeoJSON:
    def test_feature_collection_type(self):
        gj = results_to_geojson([_make_result()])
        assert gj["type"] == "FeatureCollection"
        assert "features" in gj

    def test_feature_count_matches_detections(self):
        result = _make_result(n_detections=3)
        gj = results_to_geojson([result])
        assert len(gj["features"]) == 3

    def test_multiple_results_accumulate_features(self):
        gj = results_to_geojson([_make_result(2), _make_result(3)])
        assert len(gj["features"]) == 5

    def test_feature_has_correct_type(self):
        gj = results_to_geojson([_make_result(1)])
        assert gj["features"][0]["type"] == "Feature"

    def test_point_geometry_for_gps_detection(self):
        gj = results_to_geojson([_make_result(1, lat=35.96, lon=-83.92)])
        geom = gj["features"][0]["geometry"]
        assert geom is not None
        assert geom["type"] == "Point"
        coords = geom["coordinates"]
        # GeoJSON: [lon, lat, alt]
        assert coords[0] == pytest.approx(-83.92)
        assert coords[1] == pytest.approx(35.96)

    def test_null_geometry_for_no_gps(self):
        gj = results_to_geojson([_make_result(1, lat=None, lon=None)])
        geom = gj["features"][0]["geometry"]
        assert geom is None

    def test_properties_contain_required_fields(self):
        gj = results_to_geojson([_make_result(1)])
        props = gj["features"][0]["properties"]
        for field in ("component_class", "defect_class", "severity",
                      "component_conf", "defect_conf",
                      "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            assert field in props, f"Missing property: {field}"

    def test_properties_severity_value(self):
        gj = results_to_geojson([_make_result(1)])
        assert gj["features"][0]["properties"]["severity"] == "CRITICAL"

    def test_empty_results_gives_empty_features(self):
        gj = results_to_geojson([])
        assert gj["features"] == []

    def test_result_with_no_detections(self):
        result = _make_result(0)
        gj = results_to_geojson([result])
        assert gj["features"] == []

    def test_valid_json_serialisable(self):
        gj = results_to_geojson([_make_result(2)])
        raw = json.dumps(gj)
        parsed = json.loads(raw)
        assert parsed["type"] == "FeatureCollection"


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------

class TestSaveGeoJSON:
    def test_save_creates_file(self, tmp_path: Path):
        out = tmp_path / "out" / "test.geojson"
        path = save_geojson([_make_result(2)], out)
        assert path.exists()

    def test_saved_file_is_valid_json(self, tmp_path: Path):
        out = tmp_path / "inspections.geojson"
        save_geojson([_make_result(2)], out)
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 2

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "out.geojson"
        save_geojson([_make_result(1)], nested)
        assert nested.exists()


class TestSaveCSV:
    def test_save_creates_file(self, tmp_path: Path):
        out = tmp_path / "summary.csv"
        save_csv([_make_result(2)], out)
        assert out.exists()

    def test_csv_has_header_row(self, tmp_path: Path):
        out = tmp_path / "summary.csv"
        save_csv([_make_result(1)], out)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert "image_path" in reader.fieldnames
            assert "severity" in reader.fieldnames

    def test_csv_row_count_matches_detections(self, tmp_path: Path):
        out = tmp_path / "summary.csv"
        save_csv([_make_result(3)], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_csv_no_detections_writes_one_row(self, tmp_path: Path):
        out = tmp_path / "summary.csv"
        save_csv([_make_result(0)], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["severity"] == "NONE"

    def test_csv_values_match_detection(self, tmp_path: Path):
        out = tmp_path / "summary.csv"
        save_csv([_make_result(1)], out)
        with out.open() as f:
            row = next(csv.DictReader(f))
        assert row["component_class"] == "insulator"
        assert row["defect_class"]    == "flashover_burn"
        assert row["severity"]        == "CRITICAL"
        assert float(row["component_conf"]) == pytest.approx(0.85, abs=1e-3)
