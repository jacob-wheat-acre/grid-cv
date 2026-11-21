"""
api.py – FastAPI service for drone image inspection.

Endpoints
---------
POST /inspect
    Upload one or more images; returns JSON inspection report.

GET /health
    Returns service status and model mode (mock/real).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from grid_cv.config import MAX_UPLOAD_MB
from grid_cv.geo import results_to_geojson
from grid_cv.pipeline import InspectionPipeline, InspectionResult
from grid_cv.telemetry import extract_telemetry

log = logging.getLogger(__name__)

app = FastAPI(
    title="grid-cv",
    version="0.1.0",
    description="Electric distribution asset failure detection from drone imagery",
)

# Lazy-initialise the pipeline on first request so the import never blocks startup
_pipeline: InspectionPipeline | None = None


def _get_pipeline() -> InspectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InspectionPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _inspection_to_dict(result: InspectionResult) -> dict[str, Any]:
    return result.model_dump()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    pl = _get_pipeline()
    return {
        "status": "ok",
        "detector_mode": pl._detector_mode,
        "classifier_mode": pl._classifier_mode,
    }


@app.post("/inspect")
async def inspect(
    files: list[UploadFile] = File(..., description="One or more drone image files (JPEG/PNG/TIFF)"),
) -> JSONResponse:
    """
    Inspect uploaded drone images for electric distribution asset failures.

    Returns a JSON object with:
    - ``results``: list of per-image InspectionResult objects
    - ``geojson``: GeoJSON FeatureCollection of all detections
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    pl = _get_pipeline()
    results: list[InspectionResult] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            content = await upload.read()
            if len(content) > MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {upload.filename!r} exceeds {MAX_UPLOAD_MB} MB limit.",
                )
            # Only allow image content types
            ct = (upload.content_type or "").lower()
            if not any(ct.startswith(p) for p in ("image/", "application/octet-stream")):
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported content type: {upload.content_type}",
                )

            tmp_path = Path(tmpdir) / (upload.filename or "image.jpg")
            tmp_path.write_bytes(content)

            telemetry = extract_telemetry(tmp_path)
            try:
                result = pl.inspect(tmp_path, telemetry=telemetry)
            except Exception as exc:
                log.exception("Inspection failed for %s", upload.filename)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            results.append(result)

    geojson = results_to_geojson(results)
    return JSONResponse({
        "results": [r.model_dump() for r in results],
        "geojson": geojson,
    })
