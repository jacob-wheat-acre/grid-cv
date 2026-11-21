"""Tests for the dataset download/build scripts (synthetic data, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_training_set  # noqa: E402
import data_sources  # noqa: E402
import download_insplad  # noqa: E402
import download_mpid  # noqa: E402
from grid_cv.config import (  # noqa: E402
    COMPONENT_CLASSES,
    DEFECT_CLASSES,
    INSPLAD_DET_MAP,
    INSPLAD_FAULT_MAP,
    MPID_DET_MAP,
    SEVERITY_MAP,
)


class TestLabelMaps:
    def test_det_maps_target_known_components(self):
        for m in (INSPLAD_DET_MAP, MPID_DET_MAP):
            assert set(m.values()) <= set(COMPONENT_CLASSES)

    def test_fault_map_targets_known_defects(self):
        assert set(INSPLAD_FAULT_MAP.values()) <= set(DEFECT_CLASSES)

    def test_every_defect_has_severity(self):
        assert set(SEVERITY_MAP) == set(DEFECT_CLASSES)

    def test_healthy_is_index_zero(self):
        assert DEFECT_CLASSES[0] == "healthy"


class TestInsPLADConversion:
    def test_bbox_topleft_to_center(self):
        cx, cy, w, h = download_insplad.fo_bbox_to_yolo([0.1, 0.2, 0.4, 0.2])
        assert (cx, cy, w, h) == pytest.approx((0.3, 0.3, 0.4, 0.2))

    def test_bbox_clamped_to_image(self):
        cx, cy, w, h = download_insplad.fo_bbox_to_yolo([0.9, -0.1, 0.2, 0.3])
        assert (cx, cy, w, h) == pytest.approx((0.95, 0.1, 0.1, 0.2))

    def test_degenerate_bbox_dropped(self):
        assert download_insplad.fo_bbox_to_yolo([1.2, 0.5, 0.1, 0.1]) is None

    def test_unmapped_labels_dropped(self):
        dets = [
            {"label": "glass insulator", "bounding_box": [0.1, 0.1, 0.2, 0.2]},
            {"label": "tower id plate", "bounding_box": [0.5, 0.5, 0.1, 0.1]},
        ]
        lines = download_insplad.detections_to_yolo_lines(
            dets, INSPLAD_DET_MAP, COMPONENT_CLASSES
        )
        assert len(lines) == 1
        assert lines[0].split()[0] == str(COMPONENT_CLASSES.index("insulator"))

    def test_output_name_unique_and_space_free(self):
        name = download_insplad.output_name("data/data_10/Fotos 26-11 x.jpg")
        assert name == "data_10_Fotos_26-11_x.jpg"

    def test_path_validation_rejects_traversal(self):
        assert download_insplad._FILEPATH_RE.match("data/data_3/a.jpg")
        assert not download_insplad._FILEPATH_RE.match("data/data_3/../../x.jpg")
        assert not download_insplad._FILEPATH_RE.match("/etc/passwd")


class TestMPIDConversion:
    def test_box_line_reclassed(self):
        out = download_mpid.convert_label_line("0 0.5 0.5 0.2 0.4", 7)
        assert out == "7 0.500000 0.500000 0.200000 0.400000"

    def test_polygon_line_to_bbox(self):
        out = download_mpid.convert_label_line("0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6", 0)
        assert out == "0 0.200000 0.400000 0.200000 0.400000"

    def test_malformed_line_dropped(self):
        assert download_mpid.convert_label_line("0 0.5 0.5", 0) is None
        assert download_mpid.convert_label_line("0 a b c d", 0) is None

    def test_roboflow_augments_share_group(self):
        a = download_mpid.group_key("110_dl-1-_jpg.rf.abc123.jpg")
        b = download_mpid.group_key("110_dl-1-_jpg.rf.def456.jpg")
        assert a == b == "110_dl-1-_jpg"

    def test_split_is_deterministic(self):
        s = data_sources.stable_val_split("glass_MPID/x", 0.1)
        assert s == data_sources.stable_val_split("glass_MPID/x", 0.1)
        assert s in {"train", "val"}


def _fake_source(root: Path, key: str, task: str, label: str = "") -> None:
    """Write one train + one val image for *key* with a provenance file."""
    src = data_sources.source(key)
    rows = []
    for split in ("train", "val"):
        if task == "detector":
            img = root / key / "detector" / "images" / split / f"{key}_{split}.jpg"
            lbl = root / key / "detector" / "labels" / split / f"{key}_{split}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        else:
            img = root / key / "classifier" / split / label / f"{key}_{split}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8fake")
        rows.append(data_sources.provenance_row(
            datasets_root=root, image_path=img, src=src, task=task,
            split=split, origin="synthetic", label=label,
        ))
    data_sources.write_provenance(root / key, rows)


class TestBuildTrainingSet:
    @pytest.fixture
    def root(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            data_sources.SOURCES, "internal",
            data_sources.Source("internal", "Internal", "-", "Proprietary", True, ""),
        )
        _fake_source(tmp_path, "insplad", "classifier", label="healthy")
        _fake_source(tmp_path, "mpid", "detector")
        _fake_source(tmp_path, "internal", "detector")
        return tmp_path

    def test_poc_build_includes_everything(self, root):
        out = build_training_set.build(root, "poc")
        rows = data_sources.read_provenance(out / data_sources.PROVENANCE_FILENAME)
        assert {r["source"] for r in rows} == {"insplad", "mpid", "internal"}
        assert (out / "classifier" / "train" / "healthy").is_dir()
        assert len((out / "detector_train.txt").read_text().split()) == 2

    def test_commercial_only_excludes_noncommercial(self, root):
        out = build_training_set.build(root, "prod", commercial_only=True)
        rows = data_sources.read_provenance(out / data_sources.PROVENANCE_FILENAME)
        assert {r["source"] for r in rows} == {"internal"}
        assert not (out / "classifier").exists()
        train_list = (out / "detector_train.txt").read_text()
        assert "internal" in train_list and "mpid" not in train_list

    def test_detector_yaml_lists_all_classes(self, root):
        out = build_training_set.build(root, "poc")
        text = (out / "detector.yaml").read_text()
        for name in COMPONENT_CLASSES:
            assert f": {name}" in text

    def test_registered_sources_are_not_commercial(self):
        assert not data_sources.source("insplad").commercial_ok
        assert not data_sources.source("mpid").commercial_ok
