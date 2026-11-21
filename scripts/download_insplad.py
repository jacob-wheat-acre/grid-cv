"""
download_insplad.py – Download InsPLAD from Hugging Face and convert it for grid-cv.

InsPLAD (Inspection of Power Line Assets Dataset)
  Paper  : https://arxiv.org/abs/2311.01619
  Source : https://huggingface.co/datasets/Voxel51/InsPLAD  (no account or key needed)
  License: CC BY-NC 3.0, non-commercial only.  Every converted image is
           recorded in datasets/insplad/provenance.csv so it can be excluded
           later with `build_training_set.py --commercial-only`.

Output
------
  datasets/insplad/detector/images/{train,val}/*.jpg   YOLO detection (Stage 1)
  datasets/insplad/detector/labels/{train,val}/*.txt
  datasets/insplad/classifier/{train,val}/<defect>/    ImageFolder (Stage 2)
  datasets/insplad/provenance.csv
  datasets/_raw/insplad_hf/                            downloaded originals

Converted images are hard links to the downloaded originals, so they take
no extra disk space.  Downloads resume: re-running skips files you already have.

Usage
-----
  python scripts/download_insplad.py                    # everything (~5-6 GB)
  python scripts/download_insplad.py --limit 50         # quick smoke test
  python scripts/download_insplad.py --skip-fault       # detector data only

Set HF_TOKEN (free Hugging Face account) if you hit rate limits (HTTP 429).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_sources import (  # noqa: E402
    PROVENANCE_FILENAME,
    link_or_copy,
    provenance_row,
    read_provenance,
    safe_name,
    source,
    write_provenance,
)

HF_REPO = "Voxel51/InsPLAD"
# Pinned so the dataset can't change underneath a training run.
HF_REVISION = "a68d69d8afaebe32afd2707c88e6d71114986f44"

_FILEPATH_RE = re.compile(r"^data/data_\d+/[^/\\]+\.(jpg|jpeg|png)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure conversion helpers (unit-tested)
# ---------------------------------------------------------------------------

def fo_bbox_to_yolo(bbox: list[float]) -> tuple[float, float, float, float] | None:
    """FiftyOne [x, y, w, h] (top-left, relative) -> YOLO (cx, cy, w, h), clamped."""
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def detections_to_yolo_lines(
    detections: list[dict],
    det_map: dict[str, str],
    component_classes: list[str],
) -> list[str]:
    lines = []
    for d in detections:
        target = det_map.get(d.get("label", ""))
        if target is None:
            continue
        box = fo_bbox_to_yolo(d["bounding_box"])
        if box is None:
            continue
        idx = component_classes.index(target)
        lines.append(f"{idx} " + " ".join(f"{v:.6f}" for v in box))
    return lines


def sample_split(sample: dict) -> str:
    tags = sample.get("tags", [])
    return "val" if "val" in tags else "train"


def _limit_per_group(samples: list[dict], n: int) -> list[dict]:
    """Keep up to *n* samples per (split, fault label) so smoke runs cover every class."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for s in samples:
        key = (sample_split(s), (s.get("fault") or {}).get("label", ""))
        if seen.get(key, 0) < n:
            seen[key] = seen.get(key, 0) + 1
            out.append(s)
    return out


def output_name(filepath: str) -> str:
    """data/data_7/foo bar.jpg -> data_7_foo_bar.jpg (unique and space-free)."""
    _, folder, name = filepath.split("/")
    return safe_name(f"{folder}_{name}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _hf_download(filename: str, raw_dir: Path, retries: int = 6) -> Path:
    from huggingface_hub import hf_hub_download

    local = raw_dir / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    delay = 2.0
    for attempt in range(retries):
        try:
            return Path(hf_hub_download(
                HF_REPO, filename, repo_type="dataset",
                revision=HF_REVISION, local_dir=raw_dir,
            ))
        except Exception as exc:  # network errors, 429s
            if attempt == retries - 1:
                raise
            msg = str(exc)
            wait = delay * (4 if "429" in msg else 1)
            time.sleep(wait)
            delay *= 2
    raise RuntimeError("unreachable")


def load_samples(raw_dir: Path) -> list[dict]:
    path = _hf_download("samples.json", raw_dir)
    with path.open() as f:
        samples = json.load(f)["samples"]
    good = [s for s in samples if _FILEPATH_RE.match(s.get("filepath", ""))]
    if len(good) != len(samples):
        print(f"  Skipped {len(samples) - len(good)} samples with unexpected paths")
    return good


def download_images(filepaths: list[str], raw_dir: Path, workers: int) -> list[str]:
    """Download images concurrently. Returns the filepaths that failed."""
    from tqdm import tqdm

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_hf_download, fp, raw_dir): fp for fp in filepaths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  images"):
            try:
                fut.result()
            except Exception as exc:
                failed.append(futures[fut])
                tqdm.write(f"  FAILED {futures[fut]}: {exc}")
    return failed


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

def convert_detector(
    samples: list[dict], raw_dir: Path, datasets_root: Path
) -> list[dict[str, str]]:
    from grid_cv.config import COMPONENT_CLASSES, INSPLAD_DET_MAP

    src = source("insplad")
    out = datasets_root / "insplad" / "detector"
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for s in samples:
        raw = raw_dir / s["filepath"]
        if not raw.exists():
            continue
        split = sample_split(s)
        name = output_name(s["filepath"])
        img_dst = out / "images" / split / name
        lbl_dst = out / "labels" / split / (Path(name).stem + ".txt")
        link_or_copy(raw, img_dst)
        dets = (s.get("ground_truth") or {}).get("detections", [])
        lines = detections_to_yolo_lines(dets, INSPLAD_DET_MAP, COMPONENT_CLASSES)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(provenance_row(
            datasets_root=datasets_root, image_path=img_dst, src=src,
            task="detector", split=split, origin=s["filepath"],
        ))
    return rows


def convert_classifier(
    samples: list[dict], raw_dir: Path, datasets_root: Path
) -> list[dict[str, str]]:
    from grid_cv.config import DEFECT_CLASSES, INSPLAD_FAULT_MAP

    src = source("insplad")
    out = datasets_root / "insplad" / "classifier"
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for s in samples:
        raw = raw_dir / s["filepath"]
        if not raw.exists():
            continue
        label = INSPLAD_FAULT_MAP.get((s.get("fault") or {}).get("label", ""))
        if label is None or label not in DEFECT_CLASSES:
            continue
        split = sample_split(s)
        dst = out / split / label / output_name(s["filepath"])
        link_or_copy(raw, dst)
        rows.append(provenance_row(
            datasets_root=datasets_root, image_path=dst, src=src,
            task="classifier", split=split, label=label,
            origin=f"{s['filepath']} (asset={s.get('asset')})",
        ))
    return rows


def _summarise(rows: list[dict[str, str]], task: str) -> None:
    from collections import Counter
    c = Counter((r["split"], r["label"] or "-") for r in rows if r["task"] == task)
    if not c:
        return
    print(f"  {task}:")
    for (split, label), n in sorted(c.items()):
        print(f"    {split:5s} {label:28s} {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download InsPLAD (Hugging Face) and convert for grid-cv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output", type=Path, default=Path("datasets"),
                   help="Datasets root (default: datasets/)")
    p.add_argument("--skip-det", action="store_true", help="Skip detector data")
    p.add_argument("--skip-fault", action="store_true", help="Skip classifier data")
    p.add_argument("--limit", type=int, default=None,
                   help="Only take N samples per split/label (smoke test)")
    p.add_argument("--workers", type=int, default=8, help="Parallel downloads")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root: Path = args.output.resolve()
    raw_dir = root / "_raw" / "insplad_hf"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching InsPLAD index from {HF_REPO}@{HF_REVISION[:8]} ...")
    samples = load_samples(raw_dir)

    wanted: dict[str, list[dict]] = {}
    if not args.skip_det:
        wanted["detection"] = [s for s in samples if s["task"] == "detection"]
    if not args.skip_fault:
        wanted["fault_classification"] = [
            s for s in samples if s["task"] == "fault_classification"
        ]
    if args.limit:
        wanted = {k: _limit_per_group(v, args.limit) for k, v in wanted.items()}

    files = sorted({s["filepath"] for v in wanted.values() for s in v})
    print(f"Downloading {len(files)} images to {raw_dir} ...")
    failed = download_images(files, raw_dir, args.workers)

    rows: list[dict[str, str]] = []
    if "detection" in wanted:
        rows += convert_detector(wanted["detection"], raw_dir, root)
    if "fault_classification" in wanted:
        rows += convert_classifier(wanted["fault_classification"], raw_dir, root)

    # Keep provenance for any task we skipped this run.
    regenerated = {"detection": "detector", "fault_classification": "classifier"}
    redone = {regenerated[k] for k in wanted}
    existing = root / "insplad" / PROVENANCE_FILENAME
    if existing.exists():
        rows += [r for r in read_provenance(existing) if r["task"] not in redone]

    prov = write_provenance(root / "insplad", rows)
    print(f"\nWrote {len(rows)} provenance rows -> {prov}")
    _summarise(rows, "detector")
    _summarise(rows, "classifier")

    if failed:
        print(f"\n{len(failed)} downloads failed; re-run to retry them.")
        sys.exit(1)

    print("\nNext: python scripts/build_training_set.py --name poc")


if __name__ == "__main__":
    main()
