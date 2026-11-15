"""Bounded upload staging and actual video validation for inference requests."""
from __future__ import annotations

import hashlib
import asyncio
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ml.datasets.validate import MediaLimits, MediaValidationError, probe_video


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
DEFAULT_UPLOAD_TTL_SECONDS = 60 * 60
_UPLOAD_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("ISL_MAX_CONCURRENT_UPLOADS", "4"))))
_DECODE_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("ISL_MAX_CONCURRENT_DECODES", "2"))))
_INFERENCE_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("ISL_MAX_CONCURRENT_INFERENCES", "2"))))


@dataclass
class UploadProblem(Exception):
    code: str
    message: str
    status_code: int = 400

    def __str__(self) -> str:
        return self.message


def upload_root(configured: str | Path | None = None) -> Path:
    if configured is not None:
        return Path(configured)
    configured = os.getenv("ISL_UPLOADS")
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "isl-engine" / "uploads"


def cleanup_expired_uploads(root: Path | None = None, ttl_seconds: int | None = None) -> int:
    """Remove only this service's generated upload files after their TTL."""
    root = root or upload_root()
    ttl_seconds = ttl_seconds if ttl_seconds is not None else int(
        os.getenv("ISL_UPLOAD_TTL_SECONDS", DEFAULT_UPLOAD_TTL_SECONDS)
    )
    if ttl_seconds < 60:
        raise ValueError("upload TTL must be at least 60 seconds")
    cutoff = time.time() - ttl_seconds
    removed = 0
    try:
        entries = list(root.iterdir())
    except FileNotFoundError:
        return 0
    for item in entries:
        if not item.is_file() or not item.name.startswith("upload_"):
            continue
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
        except FileNotFoundError:
            pass
    return removed


async def stage_upload(file: UploadFile, root: str | Path | None = None) -> dict[str, Any]:
    """Stream an UploadFile to a generated temp name, enforce bytes, decode it.

    Client filename is used only for its whitelisted suffix and display name.
    The raw file is never written under a static directory or kept after a
    synchronous request/job unless the caller explicitly holds the temp path.
    """
    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise UploadProblem("UNSUPPORTED_MEDIA_TYPE", "Upload a supported video file.", 415)
    max_bytes = int(os.getenv("ISL_MAX_UPLOAD_BYTES", DEFAULT_MAX_BYTES))
    max_duration = float(os.getenv("ISL_MAX_VIDEO_SECONDS", "180"))
    max_frames = int(os.getenv("ISL_MAX_VIDEO_FRAMES", "10800"))
    if max_bytes < 1024 or max_duration <= 0 or max_frames < 1:
        raise RuntimeError("upload limits are invalid")
    root = upload_root(root)
    if not _UPLOAD_SLOTS.acquire(blocking=False):
        raise UploadProblem("UPLOAD_CAPACITY_REACHED", "The upload service is busy; retry shortly.", 503)
    hasher = hashlib.sha256()
    size = 0
    path: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        cleanup_expired_uploads(root)
        with tempfile.NamedTemporaryFile(prefix="upload_", suffix=suffix, dir=root, delete=False) as stream:
            path = Path(stream.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UploadProblem("UPLOAD_TOO_LARGE", "Video exceeds the configured upload limit.", 413)
                hasher.update(chunk)
                stream.write(chunk)
        if size < 1024:
            raise UploadProblem("VIDEO_EMPTY_OR_TOO_SMALL", "Video is empty or too small to decode.", 400)
        limits = MediaLimits(max_bytes=max_bytes, max_duration_s=max_duration, max_frames=max_frames)
        if not _DECODE_SLOTS.acquire(blocking=False):
            raise UploadProblem("DECODE_CAPACITY_REACHED", "The video validation service is busy; retry shortly.", 503)
        try:
            media = await asyncio.to_thread(probe_video, str(path), limits)
        except MediaValidationError as exc:
            status = 503 if exc.code == "VIDEO_PROBE_UNAVAILABLE" else 422
            raise UploadProblem(exc.code, str(exc), status) from exc
        finally:
            _DECODE_SLOTS.release()
        return {
            "path": path,
            "sha256": hasher.hexdigest(),
            "size_bytes": size,
            "filename": original_name[:255],
            "media": media,
        }
    except BaseException:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        _UPLOAD_SLOTS.release()


def discard_upload(path: str | Path | None, root: str | Path | None = None) -> None:
    if path is None:
        return
    candidate = Path(path)
    # Only delete a generated file directly within our configured upload root.
    try:
        if candidate.parent.resolve() == upload_root(root).resolve() and candidate.name.startswith("upload_"):
            candidate.unlink(missing_ok=True)
    except OSError:
        pass


async def run_bounded_inference(function, *args):
    """Run CPU/GPU inference off the event loop with a small fixed concurrency."""
    if not _INFERENCE_SLOTS.acquire(blocking=False):
        raise UploadProblem("INFERENCE_CAPACITY_REACHED", "The inference service is busy; retry shortly.", 503)
    try:
        return await asyncio.to_thread(function, *args)
    finally:
        _INFERENCE_SLOTS.release()
