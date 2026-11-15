"""Validate canonical dataset manifests and optional local video media.

This command does not download data, approve licenses, or mutate a manifest.
It produces a JSON report and exits non-zero when required checks fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import (
    ManifestError, RIGHTS_STATES, SPLITS, read_manifest, resolve_media_path, validate_row_shape,
)


@dataclass(frozen=True)
class MediaLimits:
    max_bytes: int = 200 * 1024 * 1024
    min_duration_s: float = 0.5
    max_duration_s: float = 180.0
    max_width: int = 1920
    max_height: int = 1080
    max_frames: int = 10800
    max_decode_seconds: float = 120.0
    allowed_fourcc: tuple[str, ...] = ()


class MediaValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: str, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fourcc_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).rstrip("\x00").strip().lower()


def probe_video(path: str, limits: MediaLimits = MediaLimits()) -> dict[str, Any]:
    """Probe and decode a video under conservative limits using OpenCV.

    Metadata alone is not accepted as proof: the function reads frames until
    EOF or a hard limit. OpenCV is imported lazily so manifest-only checks
    remain usable in the base Python environment.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise MediaValidationError("VIDEO_PROBE_UNAVAILABLE", "opencv-python is required for decoded media validation") from exc

    try:
        size_bytes = os.path.getsize(path)
    except OSError as exc:
        raise MediaValidationError("MEDIA_NOT_FOUND", "media file is missing or unreadable") from exc
    if size_bytes <= 0:
        raise MediaValidationError("MEDIA_EMPTY", "media file is empty")
    if size_bytes > limits.max_bytes:
        raise MediaValidationError("VIDEO_TOO_LARGE", f"media exceeds {limits.max_bytes} byte limit")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise MediaValidationError("VIDEO_DECODE_FAILED", "video stream could not be opened")
    start = time.monotonic()
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = _fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        if width <= 0 or height <= 0 or fps <= 0:
            raise MediaValidationError("VIDEO_METADATA_INVALID", "video has invalid dimensions or frame rate")
        if width > limits.max_width or height > limits.max_height:
            raise MediaValidationError("VIDEO_LIMIT_EXCEEDED", "video dimensions exceed configured limits")
        if declared_frames > limits.max_frames:
            raise MediaValidationError("VIDEO_LIMIT_EXCEEDED", "reported frame count exceeds configured limit")
        estimated_duration = declared_frames / fps if declared_frames else 0.0
        if estimated_duration > limits.max_duration_s:
            raise MediaValidationError("VIDEO_TOO_LONG", "reported duration exceeds configured limit")
        if limits.allowed_fourcc and fourcc not in {x.lower() for x in limits.allowed_fourcc}:
            raise MediaValidationError("VIDEO_CODEC_UNSUPPORTED", f"video codec {fourcc or 'unknown'} is not allowed")

        decoded = 0
        while True:
            if time.monotonic() - start > limits.max_decode_seconds:
                raise MediaValidationError("VIDEO_DECODE_TIMEOUT", "video decoding exceeded its time limit")
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", 0) == 0:
                raise MediaValidationError("VIDEO_DECODE_FAILED", "decoder returned an empty frame")
            h, w = frame.shape[:2]
            if w > limits.max_width or h > limits.max_height:
                raise MediaValidationError("VIDEO_LIMIT_EXCEEDED", "decoded frame dimensions exceed configured limits")
            decoded += 1
            if decoded > limits.max_frames:
                raise MediaValidationError("VIDEO_LIMIT_EXCEEDED", "decoded frame count exceeds configured limit")
        if decoded == 0:
            raise MediaValidationError("VIDEO_DECODE_FAILED", "video contains no decodable frames")
        duration = decoded / fps
        if duration < limits.min_duration_s:
            raise MediaValidationError("VIDEO_TOO_SHORT", "video duration is below configured minimum")
        if duration > limits.max_duration_s:
            raise MediaValidationError("VIDEO_TOO_LONG", "decoded duration exceeds configured limit")
        count_difference = abs(declared_frames - decoded) if declared_frames else 0
        count_tolerance = max(3, int(0.05 * max(declared_frames, 1)))
        return {
            "size_bytes": size_bytes, "width": width, "height": height, "fps": fps,
            "declared_frames": declared_frames, "decoded_frames": decoded, "duration_s": duration,
            "fourcc": fourcc, "frame_count_metadata_mismatch": bool(declared_frames and count_difference > count_tolerance),
        }
    finally:
        cap.release()


def validate_rows(
    rows: Iterable[Mapping[str, Any]], *, data_root: str, verify_hashes: bool = False,
    decode_media: bool = False, require_approved_rights: bool = False,
    require_splits: bool = False, require_group_ids: bool = False,
    limits: MediaLimits = MediaLimits(),
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    row_list = list(rows)
    ids: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    class_split = defaultdict(Counter)
    task_counts = Counter()
    resolved_paths: dict[str, str] = {}

    def add(code: str, sample_id: str, severity: str, message: str) -> None:
        issues.append({"code": code, "sample_id": sample_id, "severity": severity, "message": message})

    for row in row_list:
        sid = str(row.get("sample_id", "") or "")
        ids[sid] += 1
        for issue in validate_row_shape(row):
            issues.append(issue)
        task = str(row.get("task", "") or "")
        task_counts[task] += 1
        split = str(row.get("split", "") or "")
        if require_splits and split not in {"train", "val", "test"}:
            add("SPLIT_UNASSIGNED", sid, "error", "training manifest requires train, val or test")
        if split in {"train", "val", "test"} and task == "isolated":
            class_split[str(row.get("label_id") or row.get("label_text") or "")][split] += 1

        group_id = str(row.get("group_id", "") or "").strip()
        if not group_id:
            group_id = (str(row.get("signer_id", "") or "").strip() or
                        str(row.get("session_id", "") or "").strip() or
                        str(row.get("source_video_id", "") or "").strip())
        if not group_id:
            if require_group_ids:
                add("GROUP_ID_MISSING", sid, "error", "split cannot be proven safe without a source-video/session group")
            else:
                add("GROUP_ID_MISSING", sid, "warning", "no source-video/session group; signer/session independence is unproven")
            group_id = sid
        if split in {"train", "val", "test"}:
            groups[group_id].add(split)

        rights = str(row.get("rights_status", "unknown") or "unknown")
        if require_approved_rights and rights not in {"approved_research", "approved_noncommercial", "approved_commercial"}:
            add("RIGHTS_NOT_APPROVED", sid, "error", "rights status is not approved for ingestion/training")
        elif require_approved_rights and not str(row.get("license_id", "") or "").strip():
            add("LICENSE_ID_MISSING", sid, "error", "approved training rows require a license/source-terms reference")
        elif rights in {"unknown", "review_pending"}:
            add("RIGHTS_NOT_REVIEWED", sid, "warning", "rights/access terms are not verified")

        uri = str(row.get("media_uri", "") or "")
        if uri:
            try:
                path = resolve_media_path(data_root, uri)
                resolved_paths[sid] = path
                if not os.path.isfile(path):
                    add("MEDIA_NOT_FOUND", sid, "error", "media file is not present under data root")
                else:
                    size_bytes = os.path.getsize(path)
                    if size_bytes <= 0:
                        add("MEDIA_EMPTY", sid, "error", "media file is empty")
                    if size_bytes > limits.max_bytes:
                        add("MEDIA_TOO_LARGE", sid, "error", f"media exceeds {limits.max_bytes} byte limit")
                    declared_size = str(row.get("size_bytes", "") or "").strip()
                    if declared_size and declared_size.isdigit() and int(declared_size) != size_bytes:
                        add("MEDIA_SIZE_MISMATCH", sid, "error", "manifest size_bytes does not match file")
                    expected_hash = str(row.get("sha256", "") or "").strip().lower()
                    if verify_hashes and not expected_hash:
                        add("MEDIA_HASH_MISSING", sid, "error", "sha256 is required when hash verification is enabled")
                    if verify_hashes and expected_hash:
                        actual_hash = sha256_file(path)
                        if actual_hash != expected_hash:
                            add("MEDIA_HASH_MISMATCH", sid, "error", "SHA-256 does not match manifest")
                        hashes[actual_hash].add(split)
                    elif expected_hash:
                        hashes[expected_hash].add(split)
                    if decode_media:
                        try:
                            probe = probe_video(path, limits)
                            if probe["frame_count_metadata_mismatch"]:
                                add("FRAME_COUNT_METADATA_MISMATCH", sid, "warning", "container frame count differs from decoded count")
                        except MediaValidationError as exc:
                            add(exc.code, sid, "error", str(exc))
            except ManifestError as exc:
                add("MEDIA_URI_UNSAFE", sid, "error", str(exc))

    for sid, count in ids.items():
        if sid and count > 1:
            add("DUPLICATE_SAMPLE_ID", sid, "error", f"sample_id appears {count} times")
    leaked = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if leaked:
        add("GROUP_SPLIT_LEAKAGE", "", "error", f"{len(leaked)} source groups appear in multiple splits; examples: {leaked[:10]}")
    duplicate_hashes = [digest for digest, splits in hashes.items() if len(splits) > 1]
    if duplicate_hashes:
        add("EXACT_DUPLICATE_SPLIT_LEAKAGE", "", "error", f"{len(duplicate_hashes)} matching SHA-256/size pairs cross splits")

    labels = sorted(class_split)
    missing_coverage = {label: [split for split in ("train", "val", "test") if class_split[label][split] == 0] for label in labels}
    missing_coverage = {label: missing for label, missing in missing_coverage.items() if missing}
    if missing_coverage:
        add("CLASS_SPLIT_COVERAGE", "", "warning", f"some labels lack one or more splits: {list(missing_coverage.items())[:10]}")

    errors = sum(item["severity"] == "error" for item in issues)
    return {
        "schema_version": "dataset-manifest-v1",
        "data_root": os.path.abspath(data_root),
        "rows": len(row_list),
        "tasks": dict(task_counts),
        "rights_status_counts": dict(Counter(str(r.get("rights_status", "unknown") or "unknown") for r in row_list)),
        "split_counts": dict(Counter(str(r.get("split", "") or "") for r in row_list)),
        "isolated_labels": len(class_split),
        "isolated_class_split_coverage_warnings": missing_coverage,
        "error_count": errors,
        "warning_count": len(issues) - errors,
        "ok": errors == 0,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Canonical manifest CSV")
    parser.add_argument("--data-root", required=True, help="Root containing local relative media_uri values")
    parser.add_argument("--out", default="data/reports/manifest_validation.json")
    parser.add_argument("--verify-hashes", action="store_true", help="Recompute SHA-256 for every local media file")
    parser.add_argument("--decode-media", action="store_true", help="Decode every video with bounded OpenCV validation")
    parser.add_argument("--require-approved-rights", action="store_true")
    parser.add_argument("--require-splits", action="store_true")
    parser.add_argument("--require-group-ids", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = read_manifest(args.manifest)
        result = validate_rows(
            rows, data_root=args.data_root, verify_hashes=args.verify_hashes,
            decode_media=args.decode_media, require_approved_rights=args.require_approved_rights,
            require_splits=args.require_splits, require_group_ids=args.require_group_ids,
        )
    except (OSError, ManifestError) as exc:
        result = {"schema_version": "dataset-manifest-v1", "ok": False, "error_count": 1,
                  "warning_count": 0, "issues": [{"code": "MANIFEST_INVALID", "severity": "error", "message": str(exc)}]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("ok", "rows", "error_count", "warning_count", "tasks", "split_counts", "issues")}, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
