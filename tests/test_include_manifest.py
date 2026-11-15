"""Tests for ``ml.datasets.include_manifest`` with SYNTHETIC metadata and placeholder files.

Nothing here is INCLUDE data: class names, file numbers and bytes are invented to exercise the
code. The output manifest is run through the real canonical reader and validator, the consumers
that previously had no producer for INCLUDE.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.datasets.canonical import read_manifest
from ml.datasets.include_manifest import build, load_records, main
from ml.datasets.validate import validate_rows

CLASSES = {"Animals": ["1. Dog", "2. Cat"], "Colours": ["47. Red", "50. Yellow"]}


def synthetic_records():
    records, number = [], 100
    for category, labels in CLASSES.items():
        for label in labels:
            for i in range(6):
                number += 1 if i % 3 else 40  # jumps start new proxy sessions
                split = "test" if i == 5 else "val" if i == 4 else "train"
                records.append({"parent_label": category, "label": label, "include_50": True, "split": split,
                                "video_path": f"{category}/{label}/MVI_{number}.MOV"})
    # a non-INCLUDE-50 row, and a mislabeled duplicate of an existing video
    records.append({"parent_label": "Animals", "label": "9. Horse", "include_50": False, "split": "train",
                    "video_path": "Animals/9. Horse/MVI_5.MOV"})
    records.append({"parent_label": "Colours", "label": "47. Red", "include_50": True, "split": "train",
                    "video_path": records[0]["video_path"]})
    return records


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(synthetic_records()), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "raw"
    for record in synthetic_records():
        target = root / record["video_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder-not-a-video")
    return root


def run(*args):
    assert main([str(a) for a in args]) == 0


def test_audit_and_selection_counts(metadata):
    rows, report = build(load_records(str(metadata)), include_50_only=True, split_policy="official",
                         data_root=None, dataset_version="t", license_id="CC-BY-4.0",
                         rights_status="review_pending", seed=1)
    assert report["audit_all_rows"]["rows"] == 26
    assert report["audit_all_rows"]["label_path_mismatch_rows"] == 1
    assert report["selection"]["rows_selected"] == 25          # the non-INCLUDE-50 row is not selected
    assert len(rows) == 24 and report["rows_after_label_path_exclusion"] == 24
    assert report["coverage"]["classes"] == 4 and report["schema_issues"] == 0


def test_rights_are_never_approved_implicitly(metadata, tmp_path):
    out = tmp_path / "m.csv"
    run("--metadata", metadata, "--out", out)
    rows = read_manifest(str(out))
    assert {r["rights_status"] for r in rows} == {"review_pending"}
    assert {r["license_id"] for r in rows} == {"CC-BY-4.0"}
    assert {r["split_policy"] for r in rows} == {"official"}


def test_official_manifest_round_trips_through_real_validator(metadata, data_root, tmp_path):
    out, report = tmp_path / "m.csv", tmp_path / "r.json"
    run("--metadata", metadata, "--data-root", data_root, "--rights-status", "approved_research",
        "--out", out, "--report", report)
    summary = json.loads(report.read_text())
    assert summary["media"]["missing_on_disk"] == 0 and summary["media"]["present"] == 24
    result = validate_rows(read_manifest(str(out)), data_root=str(data_root), require_approved_rights=True,
                           require_splits=True, require_group_ids=True)
    errors = [i for i in result["issues"] if i["severity"] == "error"]
    assert result["ok"], errors


def test_unreviewed_rights_fail_the_validator_gate(metadata, data_root, tmp_path):
    out = tmp_path / "m.csv"
    run("--metadata", metadata, "--data-root", data_root, "--out", out)
    result = validate_rows(read_manifest(str(out)), data_root=str(data_root), require_approved_rights=True)
    assert not result["ok"]


def test_missing_media_rows_are_dropped_and_counted(metadata, data_root, tmp_path):
    victim = data_root / synthetic_records()[3]["video_path"]
    victim.unlink()
    out, report = tmp_path / "m.csv", tmp_path / "r.json"
    run("--metadata", metadata, "--data-root", data_root, "--out", out, "--report", report)
    summary = json.loads(report.read_text())
    assert summary["media"]["missing_on_disk"] == 1
    assert len(read_manifest(str(out))) == summary["media"]["present"] == 23


def test_session_proxy_keeps_every_group_in_one_split(metadata, data_root, tmp_path):
    out, report = tmp_path / "m.csv", tmp_path / "r.json"
    run("--metadata", metadata, "--data-root", data_root, "--split-policy", "session-proxy",
        "--rights-status", "approved_research", "--out", out, "--report", report)
    rows = read_manifest(str(out))
    assert {r["split_policy"] for r in rows} == {"session_proxy_grouped"}
    assert json.loads(report.read_text())["coverage"]["groups_in_multiple_splits"] == 0
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    assert all(row["group_id"].startswith("session:") for row in rows)


def test_keep_file_lists_original_case_paths_and_drives_the_extractor(metadata, tmp_path):
    keep = tmp_path / "keep.txt"
    run("--metadata", metadata, "--write-keep", keep)
    lines = keep.read_text().splitlines()
    assert len(lines) == 24 and all(line == line.strip() and "/" in line for line in lines)
    assert any(any(ch.isupper() for ch in line) for line in lines)  # not lowercased: extractor must cope


def test_all_classes_flag_includes_non_include50_rows(metadata):
    rows, _ = build(load_records(str(metadata)), include_50_only=False, split_policy="official", data_root=None,
                    dataset_version="t", license_id="x", rights_status="unknown", seed=1)
    assert any(r["label_text"] == "9. Horse" for r in rows)


def test_refuses_to_write_when_schema_issues_exist(metadata, tmp_path, monkeypatch):
    import ml.datasets.include_manifest as module
    monkeypatch.setattr(module, "validate_row_shape", lambda row: [{"code": "X"}])
    with pytest.raises(SystemExit):
        main(["--metadata", str(metadata), "--out", str(tmp_path / "never.csv")])
    assert not Path(tmp_path / "never.csv").exists()


def test_unsupported_metadata_type_is_rejected(tmp_path):
    bad = tmp_path / "meta.parquet"
    bad.write_bytes(b"x")
    with pytest.raises(SystemExit):
        load_records(str(bad))


def test_unresolvable_conflicting_labels_fail_loudly_instead_of_first_wins():
    records = [{"video_path": "a.mov", "label": "x", "include_50": True, "split": "train"},
               {"video_path": "a.mov", "label": "y", "include_50": True, "split": "train"}]
    with pytest.raises(SystemExit, match="conflicting labels"):
        build(records, include_50_only=True, split_policy="official", data_root=None, dataset_version="t",
              license_id="x", rights_status="unknown", seed=1)


def test_identical_duplicate_rows_are_collapsed_and_counted():
    row = {"video_path": "Animals/1. Dog/MVI_1.MOV", "label": "1. Dog", "parent_label": "Animals",
           "include_50": True, "split": "train"}
    rows, report = build([row, dict(row)], include_50_only=True, split_policy="official", data_root=None,
                         dataset_version="t", license_id="x", rights_status="unknown", seed=1)
    assert len(rows) == 1 and report["duplicate_media_rows_dropped"] == 1


def test_hub_failure_gives_actionable_fallback(monkeypatch):
    import datasets

    def boom(name):
        raise ConnectionError("no route to host")
    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(SystemExit, match="Fallback.*--metadata"):
        load_records(None)


def test_hub_rows_are_loaded_and_splits_normalised(monkeypatch):
    import datasets
    row = {"parent_label": "Animals", "label": "1. Dog", "video_path": "Animals/1. Dog/a.mov", "include_50": True}
    monkeypatch.setattr(datasets, "load_dataset", lambda name: {"train": [row], "validation": [row], "test": [row]})
    records = load_records(None)
    assert [r["split"] for r in records] == ["train", "val", "test"]


def test_unexpected_columns_are_reported_not_guessed(monkeypatch, tmp_path):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda name: {"train": [{"path": "x", "class": "y"}]})
    with pytest.raises(SystemExit, match="missing .*label.*video_path"):
        load_records(None)
    empty = tmp_path / "e.json"
    empty.write_text("[]")
    with pytest.raises(SystemExit, match="no rows"):
        load_records(str(empty))
