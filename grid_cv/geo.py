"""
geo.py – GeoJSON and CSV export utilities.

Converts InspectionResult objects into:
  - A valid GeoJSON FeatureCollection (ready for QGIS / ArcGIS)
  - A flat CSV summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from grid_cv.pipeline import Detection, InspectionResult


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def _detection_to_feature(det: Detection, image_path: str) -> dict[str, Any]:
    """Convert a single Detection to a GeoJSON Feature."""
    tel = det.telemetry
    if tel and tel.has_gps:
        coords = tel.to_coordinates()
        geometry: dict[str, Any] = {
            "type": "Point",
            "coordinates": coords,
        }
    else:
        geometry = None  # type: ignore[assignment]

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "image_path":      image_path,
            "component_class": det.component_class,
            "component_conf":  round(det.component_conf, 4),
            "defect_class":    det.defect_class,
            "defect_conf":     round(det.defect_conf, 4),
            "severity":        det.severity,
            "bbox_x1":         det.bbox.x1,
            "bbox_y1":         det.bbox.y1,
            "bbox_x2":         det.bbox.x2,
            "bbox_y2":         det.bbox.y2,
            "latitude":        tel.latitude  if tel else None,
            "longitude":       tel.longitude if tel else None,
            "altitude_m":      tel.altitude_m if tel else None,
            "gimbal_pitch":    tel.gimbal_pitch_deg if tel else None,
            "gimbal_yaw":      tel.gimbal_yaw_deg   if tel else None,
            "camera_model":    tel.camera_model      if tel else None,
        },
    }


def results_to_geojson(results: list[InspectionResult]) -> dict[str, Any]:
    """Convert a list of InspectionResult objects to a GeoJSON FeatureCollection."""
    features = []
    for result in results:
        for det in result.detections:
            features.append(_detection_to_feature(det, result.image_path))

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def save_geojson(results: list[InspectionResult], output_path: Path) -> Path:
    """Serialise results to *output_path* as GeoJSON and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geojson = results_to_geojson(results)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    return output_path


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "image_path",
    "image_width",
    "image_height",
    "worst_severity",
    "inference_ms",
    "component_class",
    "component_conf",
    "defect_class",
    "defect_conf",
    "severity",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "latitude",
    "longitude",
    "altitude_m",
    "gimbal_pitch_deg",
    "gimbal_yaw_deg",
    "camera_model",
]


def save_csv(results: list[InspectionResult], output_path: Path) -> Path:
    """Write a flat CSV summary of all detections."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            tel = result.telemetry
            common = {
                "image_path":    result.image_path,
                "image_width":   result.image_width,
                "image_height":  result.image_height,
                "worst_severity": result.worst_severity,
                "inference_ms":  round(result.inference_ms, 1),
                "latitude":      tel.latitude        if tel else "",
                "longitude":     tel.longitude       if tel else "",
                "altitude_m":    tel.altitude_m      if tel else "",
                "gimbal_pitch_deg": tel.gimbal_pitch_deg if tel else "",
                "gimbal_yaw_deg":   tel.gimbal_yaw_deg   if tel else "",
                "camera_model":     tel.camera_model      if tel else "",
            }
            if not result.detections:
                writer.writerow({**common,
                                  "component_class": "", "component_conf": "",
                                  "defect_class": "", "defect_conf": "",
                                  "severity": "NONE",
                                  "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": ""})
            for det in result.detections:
                writer.writerow({**common,
                                  "component_class": det.component_class,
                                  "component_conf":  round(det.component_conf, 4),
                                  "defect_class":    det.defect_class,
                                  "defect_conf":     round(det.defect_conf, 4),
                                  "severity":        det.severity,
                                  "bbox_x1":         det.bbox.x1,
                                  "bbox_y1":         det.bbox.y1,
                                  "bbox_x2":         det.bbox.x2,
                                  "bbox_y2":         det.bbox.y2})
    return output_path
