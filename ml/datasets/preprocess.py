"""Build deterministic, versioned landmark caches from an approved manifest.

Example (single-process; runs only after the manifest has passed media checks):
  python -m ml.datasets.preprocess --manifest data/manifests/isolated.csv \\
    --data-root data/raw --task isolated --out data/processed/landmarks-v1

No augmentation is performed here. Training augmentation happens after split
assignment and retains the source sample ID in the training index.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import ManifestError, read_manifest, resolve_media_path
from .validate import validate_rows


def _artifact_name(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:32] + ".npz"


PREPROCESS_VERSION = "mediapipe-tasks-pose-hands-75x3-v1"


def _cache_is_current(target: Path, row: Mapping[str, Any]) -> bool:
    """True if ``target`` is a readable cache file made from this exact source video and pipeline."""
    if not target.is_file():
        return False
    try:
        import numpy as np
        with np.load(target, allow_pickle=False) as bundle:
            meta = json.loads(str(bundle["metadata_json"]))
            raw = bundle["raw"]
        return (meta.get("source_sha256") == row["sha256"] and meta.get("preprocess_version") == PREPROCESS_VERSION
                and meta.get("sample_id") == row["sample_id"] and raw.ndim == 3 and len(raw) >= 5)
    except Exception:  # noqa: BLE001 - any unreadable/foreign file is simply recomputed
        return False


def _extract_to_cache(row, data_root, target: Path, destination: Path, task, video_to_raw, n_raw) -> None:
    import numpy as np
    source = resolve_media_path(data_root, row["media_uri"])
    raw = video_to_raw(source)
    if raw.ndim != 3 or raw.shape[1:] != (n_raw, 3) or len(raw) < 5:
        raise ValueError(f"landmark extraction failed quality/shape check for {row['sample_id']}")
    if np.isinf(raw).any():
        raise ValueError(f"landmark output contains infinity for {row['sample_id']}")
    metadata = {
        "schema_version": "landmark-cache-v1",
        "sample_id": row["sample_id"], "dataset_id": row["dataset_id"],
        "dataset_version": row["dataset_version"], "task": task,
        "split": row["split"], "group_id": row["group_id"],
        "source_video_id": row["source_video_id"],
        "preprocess_version": PREPROCESS_VERSION,
        "source_sha256": row["sha256"], "frame_count": int(len(raw)),
    }
    # Atomic replacement prevents partially written cache files after interruption.
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as temp:
        tmp_path = temp.name
    try:
        np.savez_compressed(tmp_path, raw=raw.astype(np.float32, copy=False),
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def preprocess_manifest(
    manifest: str, data_root: str, out_dir: str, task: str, *,
    require_approved_rights: bool = True, verify_hashes: bool = True,
    resume: bool = True, continue_on_error: bool = False,
) -> dict[str, Any]:
    """Extract raw 75x3 pose/hand landmarks; return an auditable summary.

    ``resume``: reuse an existing cache file only if its stored source SHA-256 and preprocess version
    match the manifest row (so an interrupted run continues where it stopped, stale files are redone).
    ``continue_on_error``: record a video whose extraction fails in ``preprocess_failures.json`` and
    carry on instead of aborting; failed samples are excluded from the index. Default stays strict.
    """
    if task not in {"isolated", "continuous"}:
        raise ValueError("task must be isolated or continuous")
    rows = [row for row in read_manifest(manifest) if row.get("task") == task]
    if not rows:
        raise ValueError(f"manifest has no rows for task={task}")
    report = validate_rows(
        rows, data_root=data_root, verify_hashes=verify_hashes, decode_media=True,
        require_approved_rights=require_approved_rights, require_splits=True, require_group_ids=True,
    )
    if not report["ok"]:
        codes = sorted({issue["code"] for issue in report["issues"] if issue["severity"] == "error"})
        raise ValueError(f"manifest/media validation failed: {', '.join(codes)}")

    # Lazy imports keep manifest tooling usable without the computer-vision stack.
    import numpy as np
    from isl.config import N_RAW
    from isl.extract import video_to_raw

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    failures: list[dict[str, str]] = []
    reused = 0
    for row in rows:
        name = _artifact_name(row["sample_id"])
        target = destination / name
        if resume and _cache_is_current(target, row):
            reused += 1
        else:
            try:
                _extract_to_cache(row, data_root, target, destination, task, video_to_raw, N_RAW)
            except Exception as exc:  # noqa: BLE001 - recorded, or re-raised below
                if not continue_on_error:
                    raise
                failures.append({"sample_id": row["sample_id"], "media_uri": row["media_uri"],
                                 "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
        artifacts.append({
            "cache": str(target.resolve()), "cache_file": name,
            "sample_id": row["sample_id"], "dataset_id": row["dataset_id"],
            "dataset_version": row["dataset_version"], "task": task,
            "group_id": row["group_id"], "source_video_id": row["source_video_id"],
            "split": row["split"],
            "label": row.get("label_id") or row.get("label_text", ""),
            "reference_texts_json": row.get("reference_texts_json", "[]"),
            "sha256": row["sha256"], "license_id": row["license_id"],
            "rights_status": row["rights_status"],
            "preprocess_version": PREPROCESS_VERSION,
        })
    if failures:
        (destination / "preprocess_failures.json").write_text(
            json.dumps({"failed": len(failures), "attempted": len(rows), "failures": failures},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not artifacts:
        raise ValueError(f"no sample could be preprocessed ({len(failures)} failures)")

    index_path = destination / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(artifacts[0]))
        writer.writeheader()
        writer.writerows(artifacts)
    (destination / "index.json").write_text(
        json.dumps({"schema_version": "landmark-index-v1", "task": task,
                    "artifact_count": len(artifacts), "index_csv": index_path.name},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"task": task, "rows": len(rows), "artifacts": len(artifacts), "reused_from_cache": reused,
            "failed": len(failures), "output": str(destination.resolve()), "index": str(index_path.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task", choices=("isolated", "continuous"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-unreviewed-rights", action="store_true",
                         help="For private synthetic fixtures only; never for training data")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="record videos that fail extraction in preprocess_failures.json and carry on")
    parser.add_argument("--no-resume", action="store_true",
                        help="recompute every cache file instead of reusing matching ones")
    args = parser.parse_args(argv)
    try:
        result = preprocess_manifest(
            args.manifest, args.data_root, args.out, args.task,
            require_approved_rights=not args.allow_unreviewed_rights,
            verify_hashes=not args.skip_hash_verification,
            resume=not args.no_resume, continue_on_error=args.continue_on_error,
        )
    except (OSError, ManifestError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"preprocessing blocked: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
