# grid-cv

**Electric distribution asset failure detection from drone imagery.**

A production-structured Python computer vision pipeline that processes high-resolution RGB and radiometric thermal drone photos to detect and classify failures on power-line equipment (insulators, splices, arresters, transformers, poles, crossarms).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Input: 4K/8K drone image (RGB or thermal JPEG/TIFF/PNG)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  EXIF/XMP Telemetry     │  lat/lon/alt/gimbal
              │  (telemetry.py)         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Stage 1 – Localization │  SAHI + YOLOv8
              │  640×640, 20% overlap   │  insulators, poles…
              └────────────┬────────────┘
                           │  bboxes
              ┌────────────▼────────────┐
              │  Stage 2 – Classify     │  YOLOv8-cls
              │  +15% padded crop       │  healthy/burn/corrosion…
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Geo Export (geo.py)    │  GeoJSON + CSV
              └─────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run on a folder of drone images (uses mock weights out of the box)
python -m grid_cv inspect ./sample_images --output ./results

# Start API server
python -m grid_cv serve
# Then POST images to http://localhost:8000/inspect

# Run tests
pytest
```

## Project Structure

```
grid_cv/
  config.py       – thresholds, slice sizes, severity mappings
  telemetry.py    – EXIF/XMP GPS & gimbal parser
  pipeline.py     – SAHI two-stage detector + OpenCV annotation
  geo.py          – GeoJSON & CSV exporters
  api.py          – FastAPI endpoints
  cli.py          – Typer CLI
scripts/
  prepare_dataset.py  – YOLO dataset prep with aerial augmentations
  train.py            – YOLOv8 training harness
tests/
  conftest.py     – synthetic image fixtures
  test_telemetry.py
  test_pipeline.py
  test_geo.py
models/           – place detector.pt and classifier.pt here
```

## Outputs

| File | Description |
|---|---|
| `results/inspections.geojson` | GeoJSON FeatureCollection — load directly in QGIS/ArcGIS |
| `results/summary.csv` | Flat table of all detections |
| `results/annotated/*.jpg` | Images with drawn bounding boxes and severity labels |

## Severity Tiers

| Tier | Defects |
|---|---|
| `NONE` | healthy |
| `LOW` | corrosion |
| `MEDIUM` | broken_shed, vegetation_encroachment |
| `CRITICAL` | flashover_burn, oil_leak |

## Training Custom Weights

```bash
# 1. Prepare dataset (InsPLAD / TTPLA format)
python scripts/prepare_dataset.py --source ./raw_insplad --output ./datasets/grid_cv

# 2. Train detector (Stage 1)
python scripts/train.py detector --data ./datasets/grid_cv/dataset.yaml --epochs 100

# 3. Train classifier (Stage 2)
python scripts/train.py classifier --data ./datasets/defects --epochs 50
```

Weights are automatically copied to `models/detector.pt` and `models/classifier.pt`.

## API

```
POST /inspect
  multipart/form-data, field: files[]
  Returns: { results: [...], geojson: {...} }

GET /health
  Returns: { status: "ok", detector_mode: "mock|real", classifier_mode: "mock|real" }
```
