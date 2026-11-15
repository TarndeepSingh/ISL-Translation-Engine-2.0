"""Resume and failure-handling tests for ``ml.datasets.preprocess``.

Videos are tiny REAL MP4 files (so the decode/hash validation inside preprocessing is genuine);
the landmark extractor is replaced by a counting stand-in that returns synthetic arrays, so no
MediaPipe is needed and nothing here says anything about landmark quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import isl.extract as extract  # noqa: E402
from ml.datasets.canonical import write_manifest  # noqa: E402
from ml.datasets.fingerprint import fingerprint_manifest  # noqa: E402
from ml.datasets.include_manifest import build  # noqa: E402
from ml.datasets.preprocess import main, preprocess_manifest  # noqa: E402

LABELS = ["1. Dog", "2. Cat"]


def write_video(path: Path, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    rng = np.random.default_rng(seed)
    for _ in range(30):
        writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture
def project(tmp_path):
    root, records, n = tmp_path / "raw", [], 0
    for label in LABELS:
        for split in ("train", "val", "test"):
            n += 1
            rel = f"Animals/{label}/MVI_{n}.MOV"
            write_video(root / rel, n)
            records.append({"parent_label": "Animals", "label": label, "video_path": rel,
                            "include_50": True, "split": split})
    rows, _ = build(records, include_50_only=True, split_policy="official", data_root=str(root),
                    dataset_version="t", license_id="CC-BY-4.0", rights_status="approved_research", seed=1)
    manifest = tmp_path / "m.csv"
    write_manifest(fingerprint_manifest(rows, str(root)), str(manifest))
    return {"root": str(root), "manifest": str(manifest), "out": tmp_path / "cache", "n": len(rows)}


@pytest.fixture
def fake_extractor(monkeypatch):
    state = {"calls": [], "fail": set()}

    def video_to_raw(path):
        state["calls"].append(path)
        if any(token in path for token in state["fail"]):
            raise RuntimeError("simulated landmark failure")
        return np.random.default_rng(len(state["calls"])).uniform(0, 1, (30, 75, 3)).astype(np.float32)

    monkeypatch.setattr(extract, "video_to_raw", video_to_raw)
    return state


def run(project, **kwargs):
    return preprocess_manifest(project["manifest"], project["root"], str(project["out"]), "isolated", **kwargs)


def test_first_run_processes_everything(project, fake_extractor):
    summary = run(project)
    assert summary["artifacts"] == project["n"] == len(fake_extractor["calls"])
    assert summary["reused_from_cache"] == 0 and summary["failed"] == 0
    assert (project["out"] / "index.csv").is_file()


def test_second_run_reuses_all_without_calling_the_extractor(project, fake_extractor):
    run(project)
    fake_extractor["calls"].clear()
    summary = run(project)
    assert fake_extractor["calls"] == []
    assert summary["reused_from_cache"] == project["n"] and summary["artifacts"] == project["n"]


def test_interrupted_run_recomputes_only_missing_files(project, fake_extractor):
    run(project)
    victim = sorted(project["out"].glob("*.npz"))[0]
    victim.unlink()
    (project["out"] / "index.csv").unlink()      # an interrupted run never wrote the index
    fake_extractor["calls"].clear()
    summary = run(project)
    assert len(fake_extractor["calls"]) == 1 and summary["reused_from_cache"] == project["n"] - 1
    assert (project["out"] / "index.csv").is_file()


def test_cache_from_a_different_source_hash_is_recomputed(project, fake_extractor):
    run(project)
    victim = sorted(project["out"].glob("*.npz"))[0]
    with np.load(victim, allow_pickle=False) as bundle:
        raw, meta = bundle["raw"], json.loads(str(bundle["metadata_json"]))
    meta["source_sha256"] = "0" * 64
    np.savez_compressed(victim, raw=raw, metadata_json=np.asarray(json.dumps(meta)))
    fake_extractor["calls"].clear()
    run(project)
    assert len(fake_extractor["calls"]) == 1


def test_corrupt_cache_file_is_recomputed_not_trusted(project, fake_extractor):
    run(project)
    sorted(project["out"].glob("*.npz"))[0].write_bytes(b"not an npz")
    fake_extractor["calls"].clear()
    run(project)
    assert len(fake_extractor["calls"]) == 1


def test_no_resume_recomputes_everything(project, fake_extractor):
    run(project)
    fake_extractor["calls"].clear()
    summary = run(project, resume=False)
    assert len(fake_extractor["calls"]) == project["n"] and summary["reused_from_cache"] == 0


def test_default_is_strict_one_bad_video_aborts(project, fake_extractor):
    fake_extractor["fail"].add("MVI_3")
    with pytest.raises(RuntimeError, match="simulated"):
        run(project)
    assert not (project["out"] / "index.csv").exists()


def test_continue_on_error_records_failures_and_excludes_them(project, fake_extractor):
    fake_extractor["fail"].add("MVI_3")
    summary = run(project, continue_on_error=True)
    assert summary["failed"] == 1 and summary["artifacts"] == project["n"] - 1
    report = json.loads((project["out"] / "preprocess_failures.json").read_text())
    assert report["failed"] == 1 and report["attempted"] == project["n"]
    assert "MVI_3" in report["failures"][0]["media_uri"] and "simulated" in report["failures"][0]["error"]
    index_ids = {row["sample_id"] for row in read_manifest_index(project["out"] / "index.csv")}
    assert report["failures"][0]["sample_id"] not in index_ids


def test_rerun_after_fixing_a_failure_computes_only_that_sample(project, fake_extractor):
    fake_extractor["fail"].add("MVI_3")
    run(project, continue_on_error=True)
    fake_extractor["fail"].clear()
    fake_extractor["calls"].clear()
    summary = run(project, continue_on_error=True)
    assert len(fake_extractor["calls"]) == 1 and summary["artifacts"] == project["n"] and summary["failed"] == 0


def test_all_samples_failing_is_an_error_even_with_continue(project, fake_extractor):
    fake_extractor["fail"].add("MVI_")
    with pytest.raises(ValueError, match="no sample could be preprocessed"):
        run(project, continue_on_error=True)


def test_cli_flags_are_accepted(project, fake_extractor):
    code = main(["--manifest", project["manifest"], "--data-root", project["root"], "--task", "isolated",
                 "--out", str(project["out"]), "--continue-on-error", "--no-resume"])
    assert code == 0


def read_manifest_index(path):
    import csv
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
