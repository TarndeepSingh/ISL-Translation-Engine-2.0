"""End-to-end upload/decode/cleanup tests for the /api/v1 inference routes.

What is REAL here: the FastAPI router, ``stage_upload`` streaming/byte limits, the OpenCV
decode probe (``probe_video``), duration/frame/resolution limits, capacity semaphores and
temp-file cleanup. Videos are real MP4 files written by OpenCV (random noise frames).

What is a STUB here: the model. ``StubPipeline`` returns a fixed placeholder so the request
path after validation can be exercised. These tests therefore prove the *serving* contract
(validation, limits, safe errors, cleanup). They say nothing about recognition quality, and
a ``TEST-STUB`` result must never be treated as a model prediction.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402
from backend import media  # noqa: E402


class _StubPredictor:
    model_version = "TEST-STUB"


class StubPipeline:
    predictor = _StubPredictor()
    model_version = "TEST-STUB"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.seen_paths: list[str] = []

    def predict_video(self, path):
        self.seen_paths.append(path)
        if self.fail:
            raise RuntimeError("simulated inference crash with /secret/internal/path")
        return {"label": "STUB", "conf": 0.5, "margin": 0.1, "top": [("STUB", 0.5)],
                "status": "uncertain", "ms": 1.0, "duration_s": 2.0, "quality": {}}

    def translate_video(self, path):
        self.seen_paths.append(path)
        if self.fail:
            raise RuntimeError("simulated translation crash")
        return {"text": "STUB TEXT", "ms": 1.0}


def write_video(path, frames=50, fps=25, size=(64, 64)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    monkeypatch.setenv("ISL_UPLOADS", str(root))
    return root


@pytest.fixture
def stub(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(main.app.state, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(main.app.state, "translation_pipeline", pipeline, raising=False)
    return pipeline


@pytest.fixture
def client(uploads, stub):
    return TestClient(main.app)


@pytest.fixture
def good_video(tmp_path):
    return write_video(tmp_path / "good.mp4")


def leftovers(root):
    return sorted(p.name for p in root.glob("*")) if root.exists() else []


def post(client, route, path, filename=None, content_type="video/mp4"):
    with open(path, "rb") as stream:
        return client.post(route, files={"file": (filename or path.name, stream, content_type)})


def error_code(response):
    return response.json()["detail"]["code"]


def test_valid_video_reaches_model_and_is_cleaned_up(client, good_video, uploads, stub):
    response = post(client, "/api/v1/recognize", good_video)
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "isolated" and body["model_version"] == "TEST-STUB"
    assert len(body["input_sha256"]) == 64
    assert len(stub.seen_paths) == 1
    assert leftovers(uploads) == []


def test_continuous_route_validates_and_cleans_up(client, good_video, uploads):
    response = post(client, "/api/v1/translate", good_video)
    assert response.status_code == 200
    assert response.json()["task"] == "continuous"
    assert leftovers(uploads) == []


@pytest.mark.parametrize("route", ["/api/v1/recognize", "/api/v1/translate"])
def test_missing_file_is_422_when_model_present(client, route):
    response = client.post(route)
    assert response.status_code == 422 and error_code(response) == "VIDEO_REQUIRED"


def test_corrupt_video_bytes_rejected_by_real_decoder(client, tmp_path, uploads, stub):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(np.random.default_rng(1).bytes(4096))
    response = post(client, "/api/v1/recognize", junk)
    assert response.status_code == 422 and error_code(response) == "VIDEO_DECODE_FAILED"
    assert stub.seen_paths == [] and leftovers(uploads) == []


def test_tiny_file_rejected_before_decode(client, tmp_path, uploads, stub):
    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"x" * 10)
    response = post(client, "/api/v1/recognize", tiny)
    assert response.status_code == 400 and error_code(response) == "VIDEO_EMPTY_OR_TOO_SMALL"
    assert stub.seen_paths == [] and leftovers(uploads) == []


@pytest.mark.parametrize("name", ["evil.exe", "notes.txt", "noext", "video.mp4.php"])
def test_non_video_extensions_rejected(client, good_video, uploads, stub, name):
    response = post(client, "/api/v1/recognize", good_video, filename=name)
    assert response.status_code == 415 and error_code(response) == "UNSUPPORTED_MEDIA_TYPE"
    assert stub.seen_paths == [] and leftovers(uploads) == []


def test_client_filename_never_controls_storage_path(client, good_video, uploads, stub):
    response = post(client, "/api/v1/recognize", good_video, filename="../../etc/passwd.mp4")
    assert response.status_code == 200
    staged = stub.seen_paths[0]
    assert staged.startswith(str(uploads)) and "passwd" not in staged and "etc" not in staged
    assert leftovers(uploads) == []


def test_byte_cap_enforced_while_streaming(client, good_video, uploads, stub, monkeypatch):
    monkeypatch.setenv("ISL_MAX_UPLOAD_BYTES", "2048")
    response = post(client, "/api/v1/recognize", good_video)
    assert response.status_code == 413 and error_code(response) == "UPLOAD_TOO_LARGE"
    assert stub.seen_paths == [] and leftovers(uploads) == []


def test_duration_cap(client, good_video, uploads, stub, monkeypatch):
    monkeypatch.setenv("ISL_MAX_VIDEO_SECONDS", "1")
    response = post(client, "/api/v1/recognize", good_video)
    assert response.status_code == 422 and error_code(response) == "VIDEO_TOO_LONG"
    assert stub.seen_paths == [] and leftovers(uploads) == []


def test_frame_cap(client, good_video, uploads, stub, monkeypatch):
    monkeypatch.setenv("ISL_MAX_VIDEO_FRAMES", "10")
    response = post(client, "/api/v1/recognize", good_video)
    assert response.status_code == 422 and error_code(response) == "VIDEO_LIMIT_EXCEEDED"
    assert stub.seen_paths == [] and leftovers(uploads) == []


def test_too_short_video(client, tmp_path, uploads, stub):
    short = write_video(tmp_path / "short.mp4", frames=5)
    response = post(client, "/api/v1/recognize", short)
    assert response.status_code == 422 and error_code(response) == "VIDEO_TOO_SHORT"
    assert leftovers(uploads) == []


def test_resolution_cap(client, tmp_path, uploads, stub):
    big = write_video(tmp_path / "big.mp4", frames=30, size=(2560, 1440))
    response = post(client, "/api/v1/recognize", big)
    assert response.status_code == 422 and error_code(response) == "VIDEO_LIMIT_EXCEEDED"
    assert leftovers(uploads) == []


@pytest.mark.parametrize("route", ["/api/v1/recognize", "/api/v1/translate"])
def test_inference_crash_is_safe_and_still_cleans_up(client, good_video, uploads, stub, route):
    stub.fail = True
    response = post(client, route, good_video)
    assert response.status_code == 422 and error_code(response) == "INFERENCE_FAILED"
    assert "secret" not in response.text and "Traceback" not in response.text
    assert len(stub.seen_paths) == 1 and leftovers(uploads) == []


@pytest.mark.parametrize("slot_name,code", [
    ("_UPLOAD_SLOTS", "UPLOAD_CAPACITY_REACHED"),
    ("_DECODE_SLOTS", "DECODE_CAPACITY_REACHED"),
    ("_INFERENCE_SLOTS", "INFERENCE_CAPACITY_REACHED"),
])
def test_capacity_exhaustion_returns_503_and_cleans_up(client, good_video, uploads, monkeypatch,
                                                       slot_name, code):
    full = threading.BoundedSemaphore(1)
    full.acquire()
    monkeypatch.setattr(media, slot_name, full)
    response = post(client, "/api/v1/recognize", good_video)
    assert response.status_code == 503 and error_code(response) == code
    assert leftovers(uploads) == []


def test_slots_are_released_after_failures(client, good_video, tmp_path, uploads):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(np.random.default_rng(2).bytes(4096))
    for _ in range(8):  # more failures than any default slot count
        assert post(client, "/api/v1/recognize", junk).status_code == 422
    assert post(client, "/api/v1/recognize", good_video).status_code == 200


def test_expired_uploads_are_swept_but_foreign_files_are_not(uploads):
    uploads.mkdir(parents=True)
    old, fresh, foreign = uploads / "upload_old.mp4", uploads / "upload_new.mp4", uploads / "keep.txt"
    for item in (old, fresh, foreign):
        item.write_bytes(b"x")
    import os, time
    past = time.time() - 7200
    os.utime(old, (past, past))
    os.utime(foreign, (past, past))
    assert media.cleanup_expired_uploads(uploads, ttl_seconds=3600) == 1
    assert not old.exists() and fresh.exists() and foreign.exists()


def test_responses_carry_request_id(client):
    assert client.get("/api/v1/health").headers.get("x-request-id")


def test_readiness_without_models_reports_not_ready(monkeypatch):
    monkeypatch.setattr(main.app.state, "pipeline", None, raising=False)
    monkeypatch.setattr(main.app.state, "translation_pipeline", None, raising=False)
    body = TestClient(main.app).get("/api/v1/readiness").json()
    assert body["status"] == "not_ready"
    assert body["isolated_recognition"]["reason_code"] == "MODEL_NOT_READY"
    assert body["continuous_translation"]["reason_code"] == "MODEL_NOT_READY"
