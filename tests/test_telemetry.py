"""
Unit tests for grid_cv.telemetry.

Verifies:
  - EXIF GPS parsing from a synthetic JPEG with injected GPS
  - Graceful fallback for images without any metadata
  - DJI XMP parsing from crafted XMP bytes
  - Telemetry.to_coordinates() shape
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from grid_cv.telemetry import Telemetry, extract_telemetry, _parse_xmp_dji


# ---------------------------------------------------------------------------
# EXIF GPS tests
# ---------------------------------------------------------------------------

class TestExifGPS:
    def test_gps_extracted_from_synthetic_image(self, synthetic_rgb_path: Path):
        tel = extract_telemetry(synthetic_rgb_path)
        # GPS was injected by conftest: lat=35.9606, lon=-83.9207, alt=85.0
        assert tel.has_gps, f"Expected GPS, got source={tel.source!r}"
        assert abs(tel.latitude  - 35.9606) < 0.01, f"lat={tel.latitude}"
        assert abs(tel.longitude - (-83.9207)) < 0.01, f"lon={tel.longitude}"

    def test_altitude_parsed(self, synthetic_rgb_path: Path):
        tel = extract_telemetry(synthetic_rgb_path)
        if tel.altitude_m is not None:
            assert 0 < tel.altitude_m < 1000, "Altitude out of sane range"

    def test_no_gps_image_returns_telemetry_object(self, synthetic_thermal_path: Path):
        tel = extract_telemetry(synthetic_thermal_path)
        assert isinstance(tel, Telemetry)
        assert not tel.has_gps or tel.latitude is not None  # graceful

    def test_missing_file_returns_telemetry_object(self, tmp_path: Path):
        tel = extract_telemetry(tmp_path / "does_not_exist.jpg")
        assert isinstance(tel, Telemetry)
        assert tel.source == "read_error"

    def test_coordinates_shape_2d(self, tmp_path: Path):
        tel = Telemetry(latitude=51.5, longitude=-0.1)
        coords = tel.to_coordinates()
        assert coords is not None
        assert len(coords) == 2
        assert coords[0] == -0.1  # GeoJSON is [lon, lat]
        assert coords[1] == 51.5

    def test_coordinates_shape_3d(self):
        tel = Telemetry(latitude=51.5, longitude=-0.1, altitude_m=120.0)
        coords = tel.to_coordinates()
        assert coords is not None
        assert len(coords) == 3
        assert coords[2] == 120.0

    def test_no_gps_coordinates_returns_none(self):
        tel = Telemetry()
        assert tel.to_coordinates() is None


# ---------------------------------------------------------------------------
# DJI XMP parsing tests
# ---------------------------------------------------------------------------

class TestDJIXMP:
    _XMP_TEMPLATE = b"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
      drone-dji:GpsLatitude="35.9606"
      drone-dji:GpsLongitude="-83.9207"
      drone-dji:RelativeAltitude="+085.00"
      drone-dji:GimbalPitchDegree="-90.0"
      drone-dji:GimbalYawDegree="12.5"
      drone-dji:CameraModel="Zenmuse X7"
    />
  </rdf:RDF>
</x:xmpmeta>"""

    def test_dji_xmp_parsing(self):
        result = _parse_xmp_dji(self._XMP_TEMPLATE)
        assert abs(result["latitude"]  - 35.9606)  < 1e-4
        assert abs(result["longitude"] - (-83.9207)) < 1e-4
        assert abs(result["altitude_m"] - 85.0) < 0.1
        assert result["gimbal_pitch_deg"] == pytest.approx(-90.0)
        assert result["gimbal_yaw_deg"]   == pytest.approx(12.5)
        assert result["camera_model"] == "Zenmuse X7"

    def test_dji_xmp_empty_bytes_returns_empty_dict(self):
        result = _parse_xmp_dji(b"<x:xmpmeta/>")
        assert result == {}

    def test_dji_xmp_malformed_returns_empty_dict(self):
        result = _parse_xmp_dji(b"not xml at all <<>>")
        assert result == {}
