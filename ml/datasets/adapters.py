"""Explicit source-to-canonical adapters; field mappings are never guessed silently."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .canonical import ManifestError, make_row, safe_relative_uri
from ..continuous.text import normalize_for_model


log = logging.getLogger(__name__)
_NUMBER_PREFIX = re.compile(r"^\d+\.\s*")


def _clean_label(value: Any) -> str:
    """'50. Yellow' -> 'yellow' (same normalisation as the legacy isl.hf_manifest)."""
    return _NUMBER_PREFIX.sub("", str(value or "")).strip().casefold()


def include_path_label_consistent(record: Mapping[str, Any]) -> bool:
    """False only when a Category/Label/file path positively disagrees with the row's label.

    The ai4bharat/INCLUDE metadata contains rows whose ``label`` names a different class than the
    folder in ``video_path`` (seen in the dataset card preview, e.g. label '35. Bank' pointing at
    'Places/26. University/...'). Paths without that structure cannot be judged and pass.
    """
    parts = str(record.get("video_path", "")).replace("\\", "/").strip("/").split("/")
    if len(parts) < 3:
        return True
    if _clean_label(parts[-2]) != _clean_label(record.get("label")):
        return False
    parent = str(record.get("parent_label") or "").strip()
    return not parent or parts[-3].strip().casefold() == parent.casefold()


def audit_include_records(records: Iterable[Mapping[str, Any]], examples: int = 10) -> dict[str, Any]:
    """Count label/path conflicts in INCLUDE card rows without modifying or dropping anything."""
    rows = include_50 = 0
    mismatched: list[dict[str, str]] = []
    labels_by_path: dict[str, set[str]] = defaultdict(set)
    for record in records:
        rows += 1
        include_50 += int(_bool(record.get("include_50", False)))
        path = str(record.get("video_path", "")).replace("\\", "/")
        labels_by_path[path].add(str(record.get("label", "")))
        if not include_path_label_consistent(record):
            mismatched.append({"label": str(record.get("label", "")), "video_path": path})
    return {
        "rows": rows,
        "include_50_rows": include_50,
        "unique_video_paths": len(labels_by_path),
        "label_path_mismatch_rows": len(mismatched),
        "paths_with_conflicting_labels": sum(1 for labels in labels_by_path.values() if len(labels) > 1),
        "mismatch_examples": mismatched[:examples],
    }


def _split(value: Any) -> str:
    name = str(value or "").strip().lower()
    return {"validation": "val", "valid": "val", "dev": "val"}.get(name, name)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def adapt_include_records(
    records: Iterable[Mapping[str, Any]], *, dataset_version: str,
    license_id: str = "", rights_status: str = "unknown",
    include_50_only: bool = False, use_official_splits: bool = True,
    exclude_label_path_mismatch: bool = True,
) -> list[dict[str, str]]:
    """Adapt INCLUDE card-style rows with `video_path`, `label`, `split` fields.

    The adapter does not assert that the card's license covers the source video
    archive or a derived model. Rights remain unknown unless explicitly reviewed.

    Rows whose label disagrees with their ``video_path`` folder are excluded by default
    (``exclude_label_path_mismatch``) and counted in a logged warning; run
    ``audit_include_records`` to record the exact numbers. Passing False keeps them, in which
    case a video that appears under two labels collides on sample_id and fails validation.
    """
    rows = []
    excluded = 0
    for record in records:
        include_50 = _bool(record.get("include_50", False))
        if include_50_only and not include_50:
            continue
        video_path = record.get("video_path")
        label = str(record.get("label", "")).strip()
        if not video_path or not label:
            raise ManifestError("INCLUDE row requires video_path and label")
        if exclude_label_path_mismatch and not include_path_label_consistent(record):
            excluded += 1
            continue
        media_uri = safe_relative_uri(str(video_path))
        source_id = str(record.get("id") or media_uri)
        source_video_id = str(record.get("video_id") or media_uri)
        official = _split(record.get("split"))
        split = official if use_official_splits and official in {"train", "val", "test"} else "unassigned"
        rows.append(make_row(
            dataset_id="include", dataset_version=dataset_version,
            source_sample_id=source_id, source_video_id=source_video_id, task="isolated",
            media_uri=media_uri, license_id=license_id, rights_status=rights_status,
            signer_id=str(record.get("signer_id") or ""),
            session_id=str(record.get("session_id") or ""),
            group_id=str(record.get("group_id") or ""), split_official=official,
            split=split, split_policy="official" if split != "unassigned" else "pending",
            label_id=str(record.get("label_id") or label), label_text=label,
            media_status="listed", annotation_status="source_metadata",
            metadata={"include_50": include_50},
        ))
    if excluded:
        log.warning("excluded %d INCLUDE rows whose label disagrees with their video_path folder", excluded)
    return rows


def adapt_continuous_records(
    records: Iterable[Mapping[str, Any]], *, dataset_id: str, dataset_version: str,
    media_field: str, reference_field: str, video_id_field: str,
    sample_id_field: str, split_field: str | None = None,
    license_id: str = "", rights_status: str = "unknown",
) -> list[dict[str, str]]:
    """Adapt paired video/text rows with explicit source-specific field names.

    Multiple sentence segments may refer to one source recording. The video ID
    is retained as both source identity and default split group so no rows from
    one recording cross train/validation/test.
    """
    rows = []
    for record in records:
        try:
            media_uri = safe_relative_uri(str(record[media_field]))
            video_id = str(record[video_id_field]).strip()
            source_id = str(record[sample_id_field]).strip()
            reference = record[reference_field]
        except KeyError as exc:
            raise ManifestError(f"continuous source row is missing configured field {exc.args[0]!r}") from exc
        refs = reference if isinstance(reference, (list, tuple)) else [reference]
        refs = [str(value).strip() for value in refs if str(value).strip()]
        if not video_id or not source_id or not refs:
            raise ManifestError("continuous rows require sample ID, parent video ID and non-empty references")
        official = _split(record.get(split_field)) if split_field else ""
        split = official if official in {"train", "val", "test"} else "unassigned"
        rows.append(make_row(
            dataset_id=dataset_id, dataset_version=dataset_version,
            source_sample_id=source_id, source_video_id=video_id, task="continuous",
            media_uri=media_uri, license_id=license_id, rights_status=rights_status,
            signer_id=str(record.get("signer_id") or ""), session_id=str(record.get("session_id") or ""),
            group_id=f"video:{video_id}", split_official=official, split=split,
            split_policy="official" if split != "unassigned" else "pending",
            reference_texts=refs, normalized_references=[normalize_for_model(value) for value in refs],
            media_status="listed", annotation_status="paired_reference",
            metadata={"source_fields": {"media": media_field, "video_id": video_id_field,
                                        "reference": reference_field, "sample_id": sample_id_field}},
        ))
    return rows
