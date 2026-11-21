"""
build_training_set.py – Assemble detector/classifier training sets from every
downloaded source, filtered by license.

Reads every datasets/<source>/provenance.csv and writes:

  datasets/build/<name>/detector.yaml        -> scripts/train.py detector --data
  datasets/build/<name>/detector_{train,val}.txt
  datasets/build/<name>/classifier/{train,val}/<class>/   -> train.py classifier --data
  datasets/build/<name>/provenance.csv       exactly what went into this build

Usage
-----
  # Proof of concept: everything you have, including non-commercial data
  python scripts/build_training_set.py --name poc

  # Production: only sources registered commercial_ok=True
  python scripts/build_training_set.py --name prod --commercial-only

A production model must be trained from the base pretrained weights on a
--commercial-only build.  Never fine-tune it from a POC checkpoint: those
weights were learned from the non-commercial data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_sources import (  # noqa: E402
    PROVENANCE_FILENAME,
    link_or_copy,
    read_provenance,
    source,
    write_provenance,
)


def collect_rows(
    datasets_root: Path,
    sources: list[str] | None,
    commercial_only: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for prov in sorted(datasets_root.glob(f"*/{PROVENANCE_FILENAME}")):
        if prov.parent.name in {"build", "_raw"}:
            continue
        for r in read_provenance(prov):
            if sources and r["source"] not in sources:
                continue
            # The registry wins over the CSV, so a source re-classified in
            # data_sources.py takes effect without re-downloading.
            if commercial_only and not source(r["source"]).commercial_ok:
                continue
            if not (datasets_root / r["path"]).exists():
                continue
            rows.append(r)
    return rows


def build(
    datasets_root: Path,
    name: str,
    sources: list[str] | None = None,
    commercial_only: bool = False,
) -> Path:
    from grid_cv.config import COMPONENT_CLASSES, DEFECT_CLASSES

    out = datasets_root / "build" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = collect_rows(datasets_root, sources, commercial_only)
    if not rows:
        raise SystemExit(
            "No images matched. Run a download script first"
            + (" or register a commercial_ok source." if commercial_only else ".")
        )

    # ---- detector: list files + yaml (labels resolve via /images/ -> /labels/)
    det = [r for r in rows if r["task"] == "detector"]
    if det:
        for split in ("train", "val"):
            paths = [str((datasets_root / r["path"]).resolve())
                     for r in det if r["split"] == split]
            (out / f"detector_{split}.txt").write_text("\n".join(paths) + "\n")
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COMPONENT_CLASSES))
        (out / "detector.yaml").write_text(
            f"# Built by scripts/build_training_set.py "
            f"(commercial_only={commercial_only})\n"
            f"path: {out.resolve()}\n"
            f"train: detector_train.txt\n"
            f"val: detector_val.txt\n"
            f"names:\n{names}\n"
        )

    # ---- classifier: ImageFolder of hard links.  Ultralytics assigns class
    # indices from folder names per split, so keep only classes present in
    # both train and val.
    clf = [r for r in rows if r["task"] == "classifier"]
    dropped: set[str] = set()
    if clf:
        by_split: dict[str, set[str]] = defaultdict(set)
        for r in clf:
            by_split[r["split"]].add(r["label"])
        usable = by_split["train"] & by_split["val"]
        dropped = (by_split["train"] | by_split["val"]) - usable
        unknown = usable - set(DEFECT_CLASSES)
        if unknown:
            raise SystemExit(f"Classifier labels not in DEFECT_CLASSES: {sorted(unknown)}")
        kept = []
        for r in clf:
            if r["label"] not in usable:
                continue
            src_path = datasets_root / r["path"]
            dst = out / "classifier" / r["split"] / r["label"] / f"{r['source']}_{src_path.name}"
            link_or_copy(src_path, dst)
            kept.append(r)
        rows = det + kept

    write_provenance(out, rows)

    # ---- summary
    print(f"Build '{name}' -> {out}  (commercial_only={commercial_only})")
    for (task, src_key, lic), n in sorted(
        Counter((r["task"], r["source"], r["license"]) for r in rows).items()
    ):
        print(f"  {task:10s} {src_key:10s} {n:7d}  [{lic}]")
    if dropped:
        print(f"  WARNING: dropped classifier classes missing a train or val split: "
              f"{sorted(dropped)}")
    if det:
        print(f"\n  python scripts/train.py detector --data {out / 'detector.yaml'}")
    if clf:
        print(f"  python scripts/train.py classifier --data {out / 'classifier'}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build license-filtered training sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--datasets", type=Path, default=Path("datasets"))
    p.add_argument("--name", default="poc", help="Build name (default: poc)")
    p.add_argument("--sources", nargs="*", default=None,
                   help="Only these source keys (default: all downloaded)")
    p.add_argument("--commercial-only", action="store_true",
                   help="Exclude every source not cleared for commercial use")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    build(a.datasets.resolve(), a.name, a.sources, a.commercial_only)
