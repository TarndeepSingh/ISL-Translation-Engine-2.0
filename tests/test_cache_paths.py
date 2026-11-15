"""resolve_cache_paths: a processed folder must keep working after it is copied to a new location."""
from __future__ import annotations

import shutil

from isl.train import resolve_cache_paths


def make_index(root, names):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        (root / f"{n}.npz").write_bytes(b"x")
        rows.append({"sample_id": n, "cache_file": f"{n}.npz", "cache": str((root / f"{n}.npz").resolve())})
    (root / "index.csv").write_text("x")
    return rows


def test_absolute_paths_from_another_machine_are_repaired_via_cache_file(tmp_path):
    old, new = tmp_path / "old" / "processed", tmp_path / "new" / "processed"
    rows = make_index(old, ["a", "b"])
    shutil.copytree(old, new)
    shutil.rmtree(old.parent)                       # the original location no longer exists
    assert resolve_cache_paths(rows, new / "index.csv") == []
    assert all(r["cache"].startswith(str(new.resolve())) for r in rows)


def test_missing_files_are_reported_not_ignored(tmp_path):
    rows = make_index(tmp_path / "p", ["a", "b", "c"])
    (tmp_path / "p" / "b.npz").unlink()
    assert resolve_cache_paths(rows, tmp_path / "p" / "index.csv") == ["b"]


def test_rows_without_cache_file_fall_back_to_the_recorded_path(tmp_path):
    rows = make_index(tmp_path / "p", ["a"])
    del rows[0]["cache_file"]
    assert resolve_cache_paths(rows, tmp_path / "p" / "index.csv") == []


def test_relative_recorded_path_resolves_against_the_index_folder(tmp_path):
    rows = make_index(tmp_path / "p", ["a"])
    rows[0].pop("cache_file")
    rows[0]["cache"] = "a.npz"
    assert resolve_cache_paths(rows, tmp_path / "p" / "index.csv") == []
    assert rows[0]["cache"] == str((tmp_path / "p" / "a.npz").resolve())
