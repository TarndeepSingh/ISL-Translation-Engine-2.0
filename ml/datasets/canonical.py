"""Canonical task-aware manifest schema shared by ingestion and validation.

Rows describe metadata only; media stays outside Git. `media_uri` is a path
relative to a data root or a private storage URI, never a user-provided path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


TASKS = {"isolated", "continuous"}
SPLITS = {"train", "val", "test", "unassigned"}
RIGHTS_STATES = {
    "unknown", "review_pending", "approved_research", "approved_noncommercial",
    "approved_commercial", "rejected",
}
MANIFEST_FIELDS = [
    "dataset_id", "dataset_version", "sample_id", "source_sample_id", "source_video_id",
    "task", "media_uri", "sha256", "size_bytes", "license_id", "rights_status",
    "signer_id", "session_id", "group_id", "split_official", "split", "split_policy",
    "label_id", "label_text", "reference_texts_json", "normalized_references_json",
    "media_status", "annotation_status", "exclusion_reason", "preprocess_version",
    "augmentation_parent_id", "metadata_json",
]


class ManifestError(ValueError):
    """Raised when a manifest cannot be parsed or a path is unsafe."""


def stable_sample_id(dataset_id: str, dataset_version: str, source_sample_id: str) -> str:
    """Return a stable opaque ID tied to a source release and source item."""
    parts = [str(dataset_id).strip(), str(dataset_version).strip(), str(source_sample_id).strip()]
    if any(not value for value in parts):
        raise ManifestError("dataset_id, dataset_version and source_sample_id are required")
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{parts[0]}:{digest[:24]}"


def safe_relative_uri(media_uri: str) -> str:
    """Validate and normalize a local relative media path."""
    value = str(media_uri or "").strip().replace("\\", "/")
    if not value or "://" in value or "\x00" in value:
        raise ManifestError("media_uri must be a non-empty relative local path")
    path = PurePosixPath(value)
    # PurePosixPath does not recognize Windows drive paths as absolute.
    if (not path.parts or path.is_absolute() or (path.parts and ":" in path.parts[0]) or
            any(part in ("", ".", "..") for part in path.parts) or
            any(ord(ch) < 32 for ch in value)):
        raise ManifestError(f"unsafe media_uri: {media_uri!r}")
    return path.as_posix()


def resolve_media_path(data_root: str, media_uri: str) -> str:
    """Resolve a safe media URI under data_root and reject traversal/symlink escape."""
    rel = safe_relative_uri(media_uri)
    root = os.path.realpath(data_root)
    path = os.path.realpath(os.path.join(root, *PurePosixPath(rel).parts))
    try:
        common = os.path.commonpath([root, path])
    except ValueError as exc:
        raise ManifestError(f"media path is outside data root: {media_uri!r}") from exc
    if common != root:
        raise ManifestError(f"media path is outside data root: {media_uri!r}")
    return path


def _json_list(value: Any, field: str) -> str:
    if value is None or value == "":
        value = []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{field} must be a JSON array") from exc
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"{field} must be a JSON array")
    return json.dumps([str(item) for item in value], ensure_ascii=False, separators=(",", ":"))


def _json_object(value: Any, field: str) -> str:
    if value is None or value == "":
        value = {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{field} must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} must be a JSON object")
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_row(
    *, dataset_id: str, dataset_version: str, source_sample_id: str, task: str,
    media_uri: str, source_video_id: str = "", sha256: str = "", size_bytes: int | str = "",
    license_id: str = "", rights_status: str = "unknown", signer_id: str = "",
    session_id: str = "", group_id: str = "", split_official: str = "", split: str = "unassigned",
    split_policy: str = "", label_id: str = "", label_text: str = "", reference_texts: Any = None,
    normalized_references: Any = None, media_status: str = "listed", annotation_status: str = "pending",
    exclusion_reason: str = "", preprocess_version: str = "", augmentation_parent_id: str = "",
    metadata: Any = None,
) -> dict[str, str]:
    """Build a normalized row; this function does not claim media or rights are verified."""
    task = str(task).strip().lower()
    if task not in TASKS:
        raise ManifestError(f"task must be one of {sorted(TASKS)}")
    split = str(split or "unassigned").strip().lower()
    if split not in SPLITS:
        raise ManifestError(f"split must be one of {sorted(SPLITS)}")
    rights_status = str(rights_status or "unknown").strip()
    if rights_status not in RIGHTS_STATES:
        raise ManifestError(f"rights_status must be one of {sorted(RIGHTS_STATES)}")
    source_sample_id = str(source_sample_id).strip()
    sample_id = stable_sample_id(dataset_id, dataset_version, source_sample_id)
    rel = safe_relative_uri(media_uri)
    try:
        size = str(int(size_bytes)) if str(size_bytes).strip() else ""
    except (ValueError, TypeError) as exc:
        raise ManifestError("size_bytes must be a non-negative integer") from exc
    if size and int(size) < 0:
        raise ManifestError("size_bytes must be a non-negative integer")
    digest = str(sha256 or "").strip().lower()
    if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
        raise ManifestError("sha256 must be 64 lowercase hexadecimal characters")
    return {
        "dataset_id": str(dataset_id).strip(),
        "dataset_version": str(dataset_version).strip(),
        "sample_id": sample_id,
        "source_sample_id": source_sample_id,
        "source_video_id": str(source_video_id or "").strip(),
        "task": task,
        "media_uri": rel,
        "sha256": digest,
        "size_bytes": size,
        "license_id": str(license_id or "").strip(),
        "rights_status": rights_status,
        "signer_id": str(signer_id or "").strip(),
        "session_id": str(session_id or "").strip(),
        "group_id": str(group_id or (f"signer:{signer_id}" if signer_id else
                                      f"session:{session_id}" if session_id else
                                      f"video:{source_video_id}" if source_video_id else "")).strip(),
        "split_official": str(split_official or "").strip(),
        "split": split,
        "split_policy": str(split_policy or "").strip(),
        "label_id": str(label_id or "").strip(),
        "label_text": str(label_text or "").strip(),
        "reference_texts_json": _json_list(reference_texts, "reference_texts_json"),
        "normalized_references_json": _json_list(normalized_references, "normalized_references_json"),
        "media_status": str(media_status or "listed").strip(),
        "annotation_status": str(annotation_status or "pending").strip(),
        "exclusion_reason": str(exclusion_reason or "").strip(),
        "preprocess_version": str(preprocess_version or "").strip(),
        "augmentation_parent_id": str(augmentation_parent_id or "").strip(),
        "metadata_json": _json_object(metadata, "metadata_json"),
    }


def validate_row_shape(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return schema issues without touching filesystem media."""
    issues: list[dict[str, str]] = []
    sid = str(row.get("sample_id", "") or "")

    def issue(code: str, message: str) -> None:
        issues.append({"code": code, "sample_id": sid, "severity": "error", "message": message})

    for name in ("dataset_id", "dataset_version", "sample_id", "source_sample_id", "task", "media_uri"):
        if not str(row.get(name, "") or "").strip():
            issue("REQUIRED_FIELD_MISSING", f"{name} is required")
    task = str(row.get("task", "") or "").strip().lower()
    if task and task not in TASKS:
        issue("TASK_INVALID", f"task must be one of {sorted(TASKS)}")
    if all(str(row.get(name, "") or "").strip() for name in ("dataset_id", "dataset_version", "source_sample_id", "sample_id")):
        expected_id = stable_sample_id(row["dataset_id"], row["dataset_version"], row["source_sample_id"])
        if str(row["sample_id"]).strip() != expected_id:
            issue("SAMPLE_ID_MISMATCH", "sample_id does not match dataset/version/source identity")
    split = str(row.get("split", "") or "").strip().lower()
    if split and split not in SPLITS:
        issue("SPLIT_INVALID", f"split must be one of {sorted(SPLITS)}")
    rights = str(row.get("rights_status", "unknown") or "unknown").strip()
    if rights not in RIGHTS_STATES:
        issue("RIGHTS_STATUS_INVALID", f"rights_status must be one of {sorted(RIGHTS_STATES)}")
    digest = str(row.get("sha256", "") or "").strip().lower()
    if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
        issue("SHA256_INVALID", "sha256 must be 64 hexadecimal characters")
    size = str(row.get("size_bytes", "") or "").strip()
    if size:
        try:
            if int(size) < 0:
                raise ValueError
        except ValueError:
            issue("SIZE_BYTES_INVALID", "size_bytes must be a non-negative integer")
    for field in ("reference_texts_json", "normalized_references_json"):
        value = row.get(field, "[]") or "[]"
        try:
            decoded = json.loads(value) if isinstance(value, str) else value
            if not isinstance(decoded, list):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            issue("REFERENCE_JSON_INVALID", f"{field} must contain a JSON array")
    try:
        metadata = row.get("metadata_json", "{}") or "{}"
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        if not isinstance(metadata, Mapping):
            raise ValueError
    except (json.JSONDecodeError, TypeError, ValueError):
        issue("METADATA_JSON_INVALID", "metadata_json must contain a JSON object")
    if task == "isolated" and not (str(row.get("label_id", "") or "").strip() or str(row.get("label_text", "") or "").strip()):
        issue("ISOLATED_LABEL_MISSING", "isolated samples require label_id or label_text")
    if task == "continuous":
        if not str(row.get("source_video_id", "") or "").strip():
            issue("SOURCE_VIDEO_ID_MISSING", "continuous samples require their parent source_video_id")
        try:
            refs = json.loads(str(row.get("reference_texts_json", "[]") or "[]"))
            if not isinstance(refs, list) or not any(str(x).strip() for x in refs):
                issue("REFERENCE_TEXT_MISSING", "continuous samples require at least one non-empty reference")
        except json.JSONDecodeError:
            issue("REFERENCE_JSON_INVALID", "reference_texts_json must contain a JSON array")
    uri = str(row.get("media_uri", "") or "").strip()
    if uri:
        try:
            safe_relative_uri(uri)
        except ManifestError as exc:
            issue("MEDIA_URI_UNSAFE", str(exc))
    return issues


def write_manifest(rows: Iterable[Mapping[str, Any]], path: str) -> None:
    """Write canonical rows as UTF-8 CSV with a stable field order."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def read_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ManifestError("manifest has no header")
        missing = [field for field in MANIFEST_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ManifestError(f"manifest is missing fields: {', '.join(missing)}")
        return [dict(row) for row in reader]
