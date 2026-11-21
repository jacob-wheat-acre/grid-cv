"""
telemetry.py – Extract GPS and gimbal telemetry from drone images.

Supports:
  • EXIF GPS IFD  (lat, lon, altitude)
  • DJI XMP block (GpsLatitude, GpsLongitude, RelativeAltitude,
                   GimbalPitchDegree, GimbalYawDegree, CameraModel)

Returns a ``Telemetry`` Pydantic model (all fields optional so the pipeline
never hard-fails on images without metadata).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:
    import piexif
    _PIEXIF_AVAILABLE = True
except ImportError:
    _PIEXIF_AVAILABLE = False

try:
    from lxml import etree as ET
    _LXML_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[assignment]
    _LXML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Telemetry(BaseModel):
    """Parsed flight & camera telemetry from a single drone image."""

    latitude: Optional[float] = Field(None, description="WGS-84 decimal degrees")
    longitude: Optional[float] = Field(None, description="WGS-84 decimal degrees")
    altitude_m: Optional[float] = Field(None, description="Relative altitude above takeoff (m)")
    gimbal_pitch_deg: Optional[float] = Field(None, description="Gimbal pitch (°), nadir = -90")
    gimbal_yaw_deg: Optional[float] = Field(None, description="Gimbal yaw (°), North = 0")
    camera_model: Optional[str] = None
    source: str = Field("none", description="Where telemetry was parsed from")

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_coordinates(self) -> Optional[list[float]]:
        """GeoJSON-compatible [lon, lat] or [lon, lat, alt] tuple."""
        if not self.has_gps:
            return None
        if self.altitude_m is not None:
            return [self.longitude, self.latitude, self.altitude_m]
        return [self.longitude, self.latitude]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dms_to_decimal(dms_rational: tuple, ref: bytes) -> float:
    """Convert EXIF DMS rational triples to signed decimal degrees."""
    def _r(val: tuple[int, int]) -> float:
        return val[0] / val[1] if val[1] else 0.0

    deg, mn, sec = (_r(dms_rational[i]) for i in range(3))
    decimal = deg + mn / 60.0 + sec / 3600.0
    if ref in (b"S", b"W"):
        decimal = -decimal
    return decimal


def _parse_exif_gps(exif_dict: dict) -> dict:
    """Return partial telemetry dict from piexif GPS IFD."""
    gps = exif_dict.get("GPS", {})
    result: dict = {}

    lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)

    if lat_dms and lat_ref and lon_dms and lon_ref:
        result["latitude"] = _dms_to_decimal(lat_dms, lat_ref)
        result["longitude"] = _dms_to_decimal(lon_dms, lon_ref)

    alt_r = gps.get(piexif.GPSIFD.GPSAltitude)
    alt_ref = gps.get(piexif.GPSIFD.GPSAltitudeRef, 0)
    if alt_r and alt_r[1]:
        alt = alt_r[0] / alt_r[1]
        result["altitude_m"] = -alt if alt_ref == 1 else alt

    return result


# DJI XMP namespace prefixes
_DJI_NS = {
    "drone-dji": "http://www.dji.com/drone-dji/1.0/",
    "rdf":       "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "x":         "adobe:ns:meta/",
}

_XMP_RE = re.compile(rb"<x:xmpmeta.+?</x:xmpmeta>", re.DOTALL)


def _extract_xmp_bytes(image_bytes: bytes) -> Optional[bytes]:
    """Return raw XMP packet bytes from JPEG/TIFF binary, or None."""
    m = _XMP_RE.search(image_bytes)
    return m.group(0) if m else None


def _parse_xmp_dji(xmp_bytes: bytes) -> dict:
    """Parse DJI-specific XMP attributes into a partial telemetry dict."""
    result: dict = {}
    try:
        root = ET.fromstring(xmp_bytes)
    except Exception:
        return result

    # Find all rdf:Description nodes regardless of nesting
    ns_rdf = _DJI_NS["rdf"]
    ns_dji = _DJI_NS["drone-dji"]

    desc_nodes = root.iter(f"{{{ns_rdf}}}Description")
    for node in desc_nodes:
        def _attr(name: str) -> Optional[str]:
            val = node.get(f"{{{ns_dji}}}{name}")
            return val

        lat = _attr("GpsLatitude")
        lon = _attr("GpsLongitude")
        if lat and lon:
            try:
                result["latitude"] = float(lat)
                result["longitude"] = float(lon)
            except ValueError:
                pass

        alt = _attr("RelativeAltitude")
        if alt:
            try:
                result["altitude_m"] = float(alt)
            except ValueError:
                pass

        pitch = _attr("GimbalPitchDegree")
        if pitch:
            try:
                result["gimbal_pitch_deg"] = float(pitch)
            except ValueError:
                pass

        yaw = _attr("GimbalYawDegree")
        if yaw:
            try:
                result["gimbal_yaw_deg"] = float(yaw)
            except ValueError:
                pass

        model = _attr("CameraModel")
        if model:
            result["camera_model"] = model

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_telemetry(image_path: Path | str) -> Telemetry:
    """
    Extract telemetry from *image_path*.

    Parsing order:
      1. DJI XMP block (most precise for DJI drones)
      2. EXIF GPS IFD (fallback / non-DJI cameras)

    Returns a ``Telemetry`` with ``source`` indicating which path succeeded.
    """
    image_path = Path(image_path)
    merged: dict = {}
    source_parts: list[str] = []

    # --- Read raw bytes once (needed for XMP regex scan) ---
    try:
        raw = image_path.read_bytes()
    except (OSError, PermissionError):
        return Telemetry(source="read_error")

    # --- 1. DJI XMP ---
    xmp_bytes = _extract_xmp_bytes(raw)
    if xmp_bytes:
        xmp_data = _parse_xmp_dji(xmp_bytes)
        if xmp_data:
            merged.update(xmp_data)
            source_parts.append("xmp_dji")

    # --- 2. EXIF GPS (piexif) ---
    if _PIEXIF_AVAILABLE:
        try:
            exif_dict = piexif.load(raw)
            gps_data = _parse_exif_gps(exif_dict)
            # Only overwrite GPS fields that XMP didn't provide
            for k, v in gps_data.items():
                merged.setdefault(k, v)
            if gps_data:
                source_parts.append("exif_gps")
            # Camera model from EXIF 0th IFD
            zeroth = exif_dict.get("0th", {})
            model_bytes = zeroth.get(piexif.ImageIFD.Model)
            if model_bytes and "camera_model" not in merged:
                merged["camera_model"] = model_bytes.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            pass  # piexif fails on non-JPEG or corrupt EXIF – that's fine

    source = ",".join(source_parts) if source_parts else "none"
    return Telemetry(**merged, source=source)
