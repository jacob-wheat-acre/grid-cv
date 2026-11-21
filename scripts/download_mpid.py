"""
download_mpid.py – Download MPID (insulator boxes) from Google Drive for grid-cv.

MPID – Merged Public Insulator Dataset
  Repo   : https://github.com/phd-benel/MPID
  Data   : public Google Drive folder (no account or key needed)
  Content: ~5,000 UAV images of glass / porcelain / composite insulators,
           one box class per folder.  All three map to `insulator`.
  License: uploader says CC BY 4.0, but MPID merges third-party datasets
           (including CPLID, which has no license).  Recorded as NOT cleared
           for commercial use in datasets/mpid/provenance.csv.

MPID ships only a train split, so 10% is held out for val.  Images are
grouped by their pre-Roboflow name first, so augmented copies of one photo
never land on both sides of the split.

Output
------
  datasets/mpid/detector/images/{train,val}/*.jpg
  datasets/mpid/detector/labels/{train,val}/*.txt
  datasets/mpid/provenance.csv
  datasets/_raw/mpid/                               downloaded originals

Usage
-----
  python scripts/download_mpid.py               # everything (~10k files)
  python scripts/download_mpid.py --limit 30    # quick smoke test

Google Drive throttles bulk downloads.  If files fail, wait a while and
re-run: finished files are skipped.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_sources import (  # noqa: E402
    link_or_copy,
    provenance_row,
    safe_name,
    source,
    stable_val_split,
    write_provenance,
)

DRIVE_FOLDER_ID = "1HfByI7MaTWUkrbozej27tye3aAKKJQMy"
VAL_FRACTION = 0.10

# folder -> the single class name in that folder's data.yaml
SUBSETS: dict[str, str] = {
    "composite_MPID": "iso_poly",
    "glass_MPID":     "1",
    "porcelain_MPID": "iso-su",
}

_PATH_RE = re.compile(
    r"^(?P<subset>composite_MPID|glass_MPID|porcelain_MPID)/train/"
    r"(?P<kind>images|labels)/(?P<name>[^/\\]+)$"
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def group_key(filename: str) -> str:
    """'110_dl-1-_jpg.rf.<hash>.jpg' -> '110_dl-1-_jpg' (Roboflow augment group)."""
    stem = Path(filename).stem
    return stem.split(".rf.")[0]


def convert_label_line(line: str, target_idx: int) -> str | None:
    """
    Convert one YOLO label line to a bbox line with class *target_idx*.

    Accepts plain boxes (cls cx cy w h) and polygons (cls x1 y1 x2 y2 ...),
    which some of MPID's Roboflow sources export.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        cx, cy, w, h = vals
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (f"{target_idx} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
            f"{x2 - x1:.6f} {y2 - y1:.6f}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def list_drive_files() -> dict[str, str]:
    """Return {relative_path: drive_file_id} for MPID image/label files."""
    import gdown

    entries = gdown.download_folder(
        id=DRIVE_FOLDER_ID, skip_download=True, quiet=True, output="unused",
    )
    files = {}
    for e in entries:
        if _PATH_RE.match(e.path):
            files[e.path] = e.id
    return files


def _drive_download(file_id: str, dst: Path, retries: int = 5) -> None:
    import gdown

    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    delay = 5.0
    for attempt in range(retries):
        try:
            out = gdown.download(id=file_id, output=str(dst), quiet=True)
            if out and dst.exists() and dst.stat().st_size > 0:
                return
            raise RuntimeError("empty download (likely Drive quota)")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def download_all(files: dict[str, str], raw_dir: Path, workers: int) -> list[str]:
    from tqdm import tqdm

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_drive_download, fid, raw_dir / rel): rel
            for rel, fid in files.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  files"):
            try:
                fut.result()
            except Exception as exc:
                failed.append(futures[fut])
                tqdm.write(f"  FAILED {futures[fut]}: {exc}")
    return failed


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

def convert(raw_dir: Path, datasets_root: Path) -> list[dict[str, str]]:
    from grid_cv.config import COMPONENT_CLASSES, MPID_DET_MAP

    src = source("mpid")
    out = datasets_root / "mpid" / "detector"
    if out.exists():
        shutil.rmtree(out)

    rows = []
    for subset, class_name in SUBSETS.items():
        target_idx = COMPONENT_CLASSES.index(MPID_DET_MAP[class_name])
        img_dir = raw_dir / subset / "train" / "images"
        lbl_dir = raw_dir / subset / "train" / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue  # unlabelled image: skip rather than train as background
            lines = [
                c for c in (convert_label_line(l, target_idx)
                            for l in lbl.read_text().splitlines())
                if c is not None
            ]
            split = stable_val_split(f"{subset}/{group_key(img.name)}", VAL_FRACTION)
            name = safe_name(f"{subset}_{img.name}")
            img_dst = out / "images" / split / name
            lbl_dst = out / "labels" / split / (Path(name).stem + ".txt")
            link_or_copy(img, img_dst)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append(provenance_row(
                datasets_root=datasets_root, image_path=img_dst, src=src,
                task="detector", split=split, origin=f"{subset}/train/images/{img.name}",
            ))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download MPID (Google Drive) and convert for grid-cv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output", type=Path, default=Path("datasets"))
    p.add_argument("--limit", type=int, default=None,
                   help="Only download the first N images per subset (smoke test)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel downloads (keep low; Drive throttles)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root: Path = args.output.resolve()
    raw_dir = root / "_raw" / "mpid"

    print("Listing MPID Google Drive folder (takes a minute) ...")
    files = list_drive_files()

    if args.limit:
        keep: dict[str, str] = {}
        for subset in SUBSETS:
            imgs = sorted(p for p in files
                          if p.startswith(f"{subset}/train/images/"))[: args.limit]
            for p in imgs:
                keep[p] = files[p]
                lbl = f"{subset}/train/labels/{Path(p).stem}.txt"
                if lbl in files:
                    keep[lbl] = files[lbl]
        files = keep

    print(f"Downloading {len(files)} files to {raw_dir} ...")
    failed = download_all(files, raw_dir, args.workers)

    rows = convert(raw_dir, root)
    prov = write_provenance(root / "mpid", rows)
    n_val = sum(r["split"] == "val" for r in rows)
    print(f"\nConverted {len(rows)} images ({len(rows) - n_val} train / {n_val} val)")
    print(f"Provenance -> {prov}")

    if failed:
        print(f"\n{len(failed)} downloads failed (Drive throttling?); re-run later to retry.")
        sys.exit(1)

    print("\nNext: python scripts/build_training_set.py --name poc")


if __name__ == "__main__":
    main()
