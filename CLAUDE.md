# grid-cv — Claude Code Guide

Electric distribution asset failure detection from drone imagery.
Two-stage pipeline: SAHI + YOLOv8 localization → defect classifier.

---

## Project Layout

```
grid_cv/        – importable package (config, telemetry, pipeline, geo, api, cli)
scripts/        – training & dataset prep (not part of the package)
tests/          – pytest suite (67 tests, all must pass)
models/         – place detector.pt / classifier.pt here (gitignored)
sample_images/  – test drone images (gitignored)
results/        – CLI output (gitignored)
```

## Development Commands

```bash
# Install (editable + dev deps)
python -m pip install -e ".[dev]"

# Run all tests
pytest

# Single-image inspection (mock weights)
python -m grid_cv inspect sample_images/ --output results/

# Start API server
python -m grid_cv serve --reload

# Download training data (free, no keys; add --limit 10 for a smoke test)
python -m pip install -e ".[data]"
python scripts/download_insplad.py      # Hugging Face, CC BY-NC 3.0
python scripts/download_mpid.py         # Google Drive, insulator boxes only

# Assemble a license-filtered build (add --commercial-only for production)
python scripts/build_training_set.py --name poc

# Train Stage 1 detector
python scripts/train.py detector --data datasets/build/poc/detector.yaml --epochs 100

# Train Stage 2 classifier
python scripts/train.py classifier --data datasets/build/poc/classifier --epochs 50
```

## Architecture Decisions

### Mock-first fallback
Both the detector and classifier fall back to seeded NumPy mocks when
`models/detector.pt` / `models/classifier.pt` don't exist.  The full
pipeline (including GeoJSON and CSV export) runs without any trained
weights — this is intentional so development and testing never block on a
training run.

### SAHI slicing
Slices are `640×640` with `20%` overlap (`config.py`).  SAHI is bypassed
automatically in mock mode — the mock detector receives the full image and
returns synthetic bounding boxes.  Set `use_sahi=False` on the pipeline
when running on thumbnails or test images to avoid unnecessary overhead.

### Two-stage inference
1. **Stage 1** – YOLOv8 localized via SAHI → component bboxes
   (`insulator`, `splice`, `arrester`, `crossarm`, `transformer`, `pole`)
2. **Stage 2** – Crop + 15% padding → defect classifier
   (`healthy`, `broken_shed`, `flashover_burn`, `corrosion`,
   `oil_leak`, `vegetation_encroachment`)

### Severity tiers
`NONE` → `LOW` → `MEDIUM` → `CRITICAL` (defined in `config.SEVERITY_MAP`).
GeoJSON features always carry a `severity` property; `worst_severity` is
computed on `InspectionResult` from all detections in that image.

### GeoJSON null geometry
Detections without GPS coordinates emit `"geometry": null`.
This is valid per RFC 7946 and loads in QGIS/ArcGIS.

### Training-data licensing
Every downloaded image gets a row in `datasets/<source>/provenance.csv`
(source, license, `commercial_ok`).  Sources and their license status live
in `scripts/data_sources.py`.  InsPLAD and MPID are **not** cleared for
commercial use and are proof-of-concept only.  A production model must be
trained from base weights on a `build_training_set.py --commercial-only`
build, never fine-tuned from a POC checkpoint.  Ultralytics YOLOv8 is
AGPL-3.0, which is free to use but needs legal review before any closed-source deployment.

## Key Files & What They Own

| File | Responsibility |
|---|---|
| `grid_cv/config.py` | **Single source of truth** for all numeric thresholds, class lists, severity mappings. Change here, nowhere else. |
| `grid_cv/telemetry.py` | EXIF GPS parsing (piexif) + DJI XMP block parsing (lxml). Returns `Telemetry` Pydantic model. |
| `grid_cv/pipeline.py` | `InspectionPipeline` class, `Detection` / `InspectionResult` models, `annotate_image()`. |
| `grid_cv/geo.py` | `results_to_geojson()`, `save_geojson()`, `save_csv()`. No inference logic here. |
| `grid_cv/api.py` | FastAPI app. Thin wrapper — delegates entirely to `pipeline.py` and `geo.py`. |
| `grid_cv/cli.py` | Typer CLI. `inspect` and `serve` subcommands. |

## Testing

Tests live in `tests/` and use **synthetic images only** — no real drone
data required.  Fixtures in `conftest.py` create:
- `synthetic_rgb_path` — 1920×1080 JPEG with injected EXIF GPS
- `synthetic_thermal_path` — 640×512 pseudo-thermal PNG
- `synthetic_hires_path` — 4096×3072 JPEG for SAHI/slicing tests

**All 67 tests must pass before any PR.**

```bash
pytest -v          # verbose
pytest -x          # stop on first failure
pytest tests/test_pipeline.py  # one module
```

## piexif Note

`piexif.insert(exif_bytes, jpeg_bytes, output)` requires the 3-argument
form in v1.1.3; use `io.BytesIO()` as `output` for in-memory injection
(see `tests/conftest.py::inject_exif_gps`).

## Adding New Defect Classes

1. Add the class name to `DEFECT_CLASSES` in `config.py`
2. Add a severity mapping in `SEVERITY_MAP`
3. Retrain Stage 2 classifier with updated labels
4. Update the mock classifier's `DEFECT_CLASSES` reference (auto, same list)

## Adding New Component Classes

1. Add to `COMPONENT_CLASSES` in `config.py`
2. Retrain Stage 1 detector
3. Update `dataset.yaml` `names:` list to match

## Deployment

```bash
# Docker (single image, CPU)
docker build -t grid-cv .
docker run -p 8000:8000 -v ./models:/app/models grid-cv

# API health check
curl http://localhost:8000/health

# Inspect via API
curl -X POST http://localhost:8000/inspect \
  -F "files=@drone_photo.jpg"
```
