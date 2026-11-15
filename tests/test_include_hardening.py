"""Regression tests for two INCLUDE-ingestion defects found while verifying the live source terms.

1. The ai4bharat/INCLUDE card metadata contains rows whose ``label`` disagrees with the folder in
   ``video_path``. The fixtures below are metadata strings copied from the public dataset-card
   preview (no media): a video listed under both its true label and a wrong one.
2. ``safe_extract_zip`` matched archive members lowercased but compared them to a keep list that
   was not lowercased, so a keep list in the metadata's own capitalisation extracted nothing.
"""
from __future__ import annotations

import zipfile

import pytest

from isl.download_include import required_free_bytes, safe_extract_zip
from ml.datasets.adapters import (adapt_include_records, audit_include_records,
                                  include_path_label_consistent)
from ml.datasets.canonical import validate_row_shape

GOOD_UNIVERSITY = {"parent_label": "Places", "label": "26. University",
                   "video_path": "Places/26. University/MVI_3376.MOV", "include_50": False, "split": "train"}
WRONG_BANK = {"parent_label": "Places", "label": "35. Bank",
              "video_path": "Places/26. University/MVI_3376.MOV", "include_50": False, "split": "train"}
GOOD_PAGE = {"parent_label": "Home", "label": "38. Page",
             "video_path": "Home/38. Page/MVI_8809.MP4", "include_50": False, "split": "train"}
WRONG_BIRD = {"parent_label": "Animals", "label": "4. Bird",
              "video_path": "Home/38. Page/MVI_8809.MP4", "include_50": False, "split": "train"}
GOOD_DOG = {"parent_label": "Animals", "label": "1. Dog",
            "video_path": "Animals/1. Dog/MVI_3030.MOV", "include_50": True, "split": "test"}
RECORDS = [GOOD_UNIVERSITY, WRONG_BANK, GOOD_PAGE, WRONG_BIRD, GOOD_DOG]


def test_consistency_check_flags_only_real_disagreements():
    assert include_path_label_consistent(GOOD_UNIVERSITY)
    assert include_path_label_consistent(GOOD_DOG)
    assert not include_path_label_consistent(WRONG_BANK)
    assert not include_path_label_consistent(WRONG_BIRD)


def test_parent_category_mismatch_is_flagged():
    row = dict(GOOD_DOG, parent_label="Jobs")
    assert not include_path_label_consistent(row)


def test_unstructured_paths_cannot_be_judged_and_pass():
    assert include_path_label_consistent({"video_path": "yellow/v1.mp4", "label": "something else"})
    assert include_path_label_consistent({"video_path": "clip.mp4", "label": "x"})


def test_windows_separators_and_case_are_handled():
    row = {"label": "1. DOG", "video_path": "Animals\\1. dog\\MVI_3030.MOV"}
    assert include_path_label_consistent(row)


def test_audit_reports_counts_without_modifying_records():
    before = [dict(r) for r in RECORDS]
    report = audit_include_records(RECORDS)
    assert RECORDS == before
    assert report["rows"] == 5 and report["include_50_rows"] == 1
    assert report["unique_video_paths"] == 3
    assert report["label_path_mismatch_rows"] == 2
    assert report["paths_with_conflicting_labels"] == 2
    assert {e["label"] for e in report["mismatch_examples"]} == {"35. Bank", "4. Bird"}


def test_adapter_excludes_conflicting_rows_by_default(caplog):
    with caplog.at_level("WARNING"):
        rows = adapt_include_records(RECORDS, dataset_version="zenodo-v1")
    assert len(rows) == 3
    assert {r["label_text"] for r in rows} == {"26. University", "38. Page", "1. Dog"}
    assert len({r["sample_id"] for r in rows}) == 3
    assert any("excluded 2 INCLUDE rows" in m for m in caplog.messages)
    assert all(not validate_row_shape(r) for r in rows)


def test_adapter_without_exclusion_keeps_the_collision_visible():
    rows = adapt_include_records(RECORDS, dataset_version="zenodo-v1", exclude_label_path_mismatch=False)
    ids = [r["sample_id"] for r in rows]
    assert len(rows) == 5 and len(set(ids)) == 3  # same video listed twice collides on sample_id


def test_include_50_filter_still_applies_after_exclusion():
    rows = adapt_include_records(RECORDS, dataset_version="zenodo-v1", include_50_only=True)
    assert [r["label_text"] for r in rows] == ["1. Dog"]


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "Adjectives_1of1.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("Adjectives/1. loud/MVI_9536.MOV", b"a" * 64)
        bundle.writestr("Adjectives/2. quiet/MVI_9538.MOV", b"b" * 64)
    return path


def extract(archive, tmp_path, keep):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return safe_extract_zip(archive, out, keep), out


def test_keep_list_in_original_metadata_case_extracts_the_member(archive, tmp_path):
    count, out = extract(archive, tmp_path, {"Adjectives/1. loud/MVI_9536.MOV"})
    assert count == 1
    assert (out / "Adjectives" / "1. loud" / "MVI_9536.MOV").exists()
    assert not (out / "Adjectives" / "2. quiet").exists()


def test_keep_list_tolerates_lowercase_backslashes_and_blank_lines(archive, tmp_path):
    count, _ = extract(archive, tmp_path, ["", "  ", "adjectives\\1. loud\\mvi_9536.mov"])
    assert count == 1


def test_keep_none_extracts_everything_and_unlisted_member_is_skipped(archive, tmp_path):
    assert extract(archive, tmp_path, None)[0] == 2
    other = tmp_path / "second"
    other.mkdir()
    assert safe_extract_zip(archive, other, {"Adjectives/9. nothing/none.mov"}) == 0


def test_path_traversal_in_archive_is_still_rejected(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as bundle:
        bundle.writestr("../outside.mov", b"x")
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(Exception):
        safe_extract_zip(evil, out, None)
    assert not (tmp_path / "outside.mov").exists()


def test_free_space_requirement_depends_on_whether_zips_are_deleted():
    selected = [("a.zip", "", 1_000, ""), ("b.zip", "", 3_000, ""), ("c.zip", "", 2_000, "")]
    assert required_free_bytes(selected, delete_zips=False) == int(6_000 * 1.1)
    assert required_free_bytes(selected, delete_zips=True) == int(3_000 * 1.1)
    assert required_free_bytes([], delete_zips=True) == 0
