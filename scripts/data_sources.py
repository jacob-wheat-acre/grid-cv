"""
data_sources.py – License registry and provenance helpers for training data.

Every image that a download script writes into ``datasets/`` gets one row
in that source's ``provenance.csv``.  ``build_training_set.py`` reads those
rows to assemble a training set, and ``--commercial-only`` drops every row
whose source is not cleared for commercial use.  That filter is how the
proof-of-concept data gets cut before a production model is trained.

To add your internal imagery, write a ``provenance.csv`` with the same
columns under ``datasets/<source>/`` and register the source below with
``commercial_ok=True``.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

PROVENANCE_FILENAME = "provenance.csv"
PROVENANCE_FIELDS = [
    "path",           # image path relative to the datasets root (POSIX)
    "source",         # key into SOURCES
    "task",           # "detector" or "classifier"
    "split",          # "train" or "val"
    "label",          # classifier class name; blank for detector rows
    "origin",         # where the file came from inside the source
    "license",
    "commercial_ok",  # "true" / "false"
]


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    url: str
    license: str
    commercial_ok: bool
    notes: str


SOURCES: dict[str, Source] = {
    "insplad": Source(
        key="insplad",
        name="InsPLAD (Voxel51 Hugging Face build)",
        url="https://huggingface.co/datasets/Voxel51/InsPLAD",
        license="CC BY-NC 3.0",
        commercial_ok=False,
        notes="Non-commercial only. Proof-of-concept use; exclude from production.",
    ),
    "mpid": Source(
        key="mpid",
        name="MPID – Merged Public Insulator Dataset",
        url="https://github.com/phd-benel/MPID",
        license="CC BY 4.0 (uploader-asserted)",
        commercial_ok=False,
        notes=(
            "The CC BY 4.0 tag is not reliable: MPID merges third-party sets, "
            "including CPLID (no license; State Grid China imagery) and STN PLAD "
            "(terms unclear). Treat as not cleared for commercial use."
        ),
    ),
}


def source(key: str) -> Source:
    try:
        return SOURCES[key]
    except KeyError:
        raise KeyError(
            f"Unknown data source {key!r}. Register it in scripts/data_sources.py."
        ) from None


def write_provenance(root: Path, rows: list[dict[str, str]]) -> Path:
    """Write *rows* to ``<root>/provenance.csv``, replacing any existing file."""
    out = root / PROVENANCE_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out


def read_provenance(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def provenance_row(
    *,
    datasets_root: Path,
    image_path: Path,
    src: Source,
    task: str,
    split: str,
    origin: str,
    label: str = "",
) -> dict[str, str]:
    return {
        "path": image_path.relative_to(datasets_root).as_posix(),
        "source": src.key,
        "task": task,
        "split": split,
        "label": label,
        "origin": origin,
        "license": src.license,
        "commercial_ok": "true" if src.commercial_ok else "false",
    }


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* to *dst* (no extra disk use), copying if that fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stable_val_split(group_key: str, val_fraction: float) -> str:
    """Deterministically assign *group_key* to "train" or "val"."""
    h = int(hashlib.sha1(group_key.encode()).hexdigest()[:8], 16)
    return "val" if (h / 0xFFFFFFFF) < val_fraction else "train"


def safe_name(name: str) -> str:
    """Make a filename safe for Ultralytics list files (no spaces/separators)."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")
