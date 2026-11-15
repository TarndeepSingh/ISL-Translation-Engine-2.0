# API and Component Design

**Status:** Architecture baseline plus implementation update, 10 November 2026. A subset of the target contracts is implemented under `/api/v1`; this document marks routes that remain planned.

## 1. Component boundaries

| Component | Responsibility | Existing source to reuse | Target changes |
|---|---|---|---|
| React web client | Task choice, file workflows, progress, results and readiness | web/src, web/package.json | Typed v1 client and distinct task pages implemented; browser review pending |
| API routers and schemas | Versioned HTTP contracts and stable task errors | backend/main.py, backend/api_ml.py, backend/api_v1.py, backend/schemas.py | `/api/v1` health/readiness/recognize/translate implemented; old `/api` remains compatibility; v1 jobs/models route not yet built |
| Video validation | Safe upload, actual media probing, bounded decode and limits | backend/media.py, ml/datasets/validate.py | Streaming upload, actual decode, byte/duration/frame caps, concurrency bounds and TTL cleanup implemented |
| Task router | Select isolated or continuous adapter from explicit route | backend/api_v1.py | Separate routes dispatch only to their matching model adapter; missing models return MODEL_NOT_READY |
| Isolated adapter | Whole-sign inference and abstention result | isl/features.py, landmarks.py, spotter.py, model.py, backend/engine.py, ml/inference/pipeline.py | ONNX/label contract validation, artifact hash, validation calibration sidecar and explicit raw-score status implemented; no accepted package exists |
| Continuous adapter | Continuous visual sequence to generated English | ml/continuous/* | Pose/hand Transformer, training-only vocabulary, grouped-split trainer, evaluation and offline inference scaffold implemented; data/model gate remains closed |
| Dataset layer | Source-specific ingestion, provenance, checksums, split/leakage and reports | ml/datasets/canonical.py, adapters.py, fingerprint.py, validate.py, near_duplicates.py, preprocess.py | Implemented generically; no real source release has been ingested |
| Training/evaluation | Reproducible task experiments and reports | isl/train.py, ml/continuous/train.py, ml/experiments/*, ml/evaluation/* | Group leak gates, experiment metadata, validation calibration, multiple-reference metrics and model-card output are coded; no data-backed run |
| Model registry/loader | Immutable artifact selection, integrity, task compatibility, readiness | backend/engine.py | Add manifests/checksums and per-task lifecycle; do not load arbitrary checkpoints |
| Job repository/worker | Queue metadata, result, error, TTL and bounded execution | SQLite logic in backend/api_ml.py | Legacy job API is bounded and expires metadata; v1 cancellation/restart recovery not implemented |
| Artifact/media stores | Keep datasets/model packages and temporary user video separate | data/, models/ placeholders | Local dev adapter and future private object-store adapter; TTL for user clips |
| Operations | Structured logs, timings, capacity, health, retention and rollback | Docker, Compose, GitHub Actions | Model-specific readiness, privacy-safe logs, resource limits and release evidence |

## 2. REST API v1

### Endpoint inventory

| Method and path | Purpose | Success | Relevant errors |
|---|---|---|---|
| GET /api/v1/health | Process liveness only | 200 with service/version | 503 only when process itself cannot respond |
| GET /api/v1/readiness | Dependency and per-task model readiness | 200 with task states; overall status is degraded if a required task is absent | 503 when configured deployment cannot serve required tasks |
| GET /api/v1/models | Current isolated and continuous package IDs, task scope and limitations | 200 | Planned; use `/readiness` and legacy `/api/model` for current status |
| POST /api/v1/recognize | Validate and classify a short one-sign video | 200 RecognitionResult | 400/413/415/422/503/504 |
| POST /api/v1/translate | Validate and translate a continuous video | 200 TranslationResult | 400/413/415/422/503/504 |
| POST /api/v1/jobs | Submit a long inference job; explicit task field is required | 202 JobAccepted | Planned; current compatibility route is `POST /api/jobs` and only runs isolated inference |
| GET /api/v1/jobs/{job_id} | Poll queued/running/done/failed/cancelled/expired job | 200 JobStatus | Planned; current compatibility route is `GET /api/jobs/{id}` |
| GET /api/v1/jobs/{job_id}/result | Fetch typed final result | 200 task-specific result | Planned |
| DELETE /api/v1/jobs/{job_id} | Cancel queued work and request cleanup | 202/204 | Planned |
| GET /api/v1/evaluation/summary | Owner/developer evaluation report summary | 200 | 404 not run, 403 if private route is protected |

The synchronous recognize/translate routes enforce a measured maximum duration. Longer requests use POST /jobs; no web request can enqueue unbounded work. Keep current /api routes as a temporary compatibility layer only while migrating the existing UI. Remove that layer after the React app is switched and route usage is checked.

### Request and response contracts

Recognition request is multipart with a single video file and optional request options limited to supported vocabulary/version selection. The server ignores the client filename for storage. Client MIME type is informational only.

~~~json
{
  "task": "isolated",
  "status": "complete",
  "label_id": "include:yellow",
  "text": "yellow",
  "candidates": [
    {"label_id": "include:yellow", "text": "yellow", "score": 0.82}
  ],
  "confidence": null,
  "model": {"id": "isl-isolated", "version": "unavailable-until-trained"},
  "preprocess_version": "isolated-landmarks-v1",
  "processing_ms": 0,
  "warnings": ["model package unavailable"],
  "input_quality": {"usable": false, "reasons": ["no trained model loaded"]}
}
~~~

This example describes a schema shape only. The real endpoint must not return a successful result when no model is loaded; it returns MODEL_NOT_READY. An actual successful response must use a real package ID/version and real measured timing. Confidence remains null unless fitted and evaluated on held-out validation data. Candidate scores are not called probabilities unless calibrated.

~~~json
{
  "task": "continuous",
  "status": "complete",
  "generated_text": "",
  "model": {"id": "isl-continuous", "version": "unavailable-until-trained"},
  "preprocess_version": "continuous-pose-v1",
  "processing_ms": 0,
  "warnings": []
}
~~~

No translation confidence is returned by default. A translation result may include generated segments/timestamps only if the model was trained/evaluated to produce them. References are evaluation-only and never sent back as predictions.

### Job status contract

~~~json
{
  "job_id": "server-generated-opaque-id",
  "task": "continuous",
  "status": "queued",
  "created_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC",
  "result_url": null,
  "error": null,
  "expires_at": "ISO-8601 UTC"
}
~~~

Persist only opaque IDs, task, status, bounded filename display metadata, request digest, timestamps, model version, result JSON and safe error code. Do not persist raw media after processing unless the user explicitly chooses a documented retention feature. A local single-process worker polls durable rows; a future multi-instance worker can replace the implementation behind the same interface.

### Error envelope

~~~json
{
  "error": {
    "code": "VIDEO_DECODE_FAILED",
    "message": "The uploaded file could not be decoded as a supported video.",
    "request_id": "server-generated-id"
  }
}
~~~

Stable codes: UNSUPPORTED_MEDIA, VIDEO_TOO_LARGE, VIDEO_TOO_LONG, VIDEO_DECODE_FAILED, VIDEO_LIMIT_EXCEEDED, TASK_INPUT_MISMATCH, MODEL_NOT_READY, INFERENCE_FAILED, INFERENCE_TIMEOUT, JOB_NOT_FOUND, JOB_NOT_READY and JOB_EXPIRED. Public messages do not contain stack traces, secrets or local paths.

## 3. Live WebSocket protocols

### Live isolated recognition (gated)

The browser first establishes WSS and sends a protocol-version/task handshake. The server returns per-task readiness, model/preprocessing versions and negotiated limits. Client frames carry monotonically increasing sequence number, capture timestamp and canonical pose/hand values with explicit missingness/confidence. Server messages distinguish session state, segment-start/active, interim quality state, finalized sign, uncertain result and error. Frames have a byte/array bound; server rejects malformed lengths, stale sequence numbers and unsupported versions. Backpressure keeps the newest frame and drops older unprocessed frames. Server normalizes/segments/infer; client displays state without inventing labels.

The legacy `/ws` route is disabled by default. An explicit local-only development flag still requires a loaded model and loopback origin and applies a 64 KiB message cap. It is not a released protocol: versioned sequencing, full payload validation, authentication, Python/browser parity and live stability evidence are not complete. Golden clips must yield comparable landmarks/features before live inference is enabled.

### Live continuous translation (future)

Use a separate WSS session namespace. Frames or encoded chunks contain sequence/timestamp and model version. Server events use explicit partial/final/revised transcript types and stable hypothesis IDs. Implement auth, origin allowlist, reconnect, cancellation, idle timeout, backpressure and chunk/memory limits. The model must itself support causal or bounded chunk inference. Otherwise disable this protocol and direct users to record-then-translate. Use WebRTC only if the product later transports two-way audio/video calls.

## 4. Video and inference service interfaces

The code should expose narrow internal contracts independent of FastAPI:

~~~text
MediaValidator.validate(upload_stream, task, limits) -> ValidatedMedia
VideoDecoder.iter_frames(validated_media, sampling_config) -> Frame(timestamp_ms, rgb)
Preprocessor.version -> string
Preprocessor.transform(validated_media) -> ModelInput + InputQuality
ModelRegistry.load(task, version=None) -> ReadyModel | NotReady
IsolatedModel.predict(model_input) -> RecognitionResult
ContinuousModel.translate(model_input, generation_config) -> TranslationResult
JobRepository.create(task, digest, metadata) -> Job
JobRepository.claim_next() -> Job | None
JobRepository.finish(job_id, result) / fail(job_id, safe_code)
MediaStore.stage(upload_stream) -> PrivateTemporaryMedia
MediaStore.delete(media_id)
~~~

All model adapters own preprocessing compatibility checks and convert tensors to typed DTOs. API routers do not manipulate tensors, class-index order, tokenizer internals or output formatting. Inference sessions/model weights load once per worker, not once per request. Readiness states are per task so an unavailable continuous model does not hide a healthy isolated model.

## 5. Proposed repository structure and migration mapping

~~~text
.
├── backend/
│   ├── api/v1/                  # FastAPI route modules
│   ├── core/                    # settings, errors, request IDs
│   ├── schemas/                 # Pydantic request/result/job DTOs
│   ├── media/                   # validation, decode, temporary storage
│   ├── services/                # task routing and bounded jobs
│   ├── registry/                # artifact manifests, integrity and loaders
│   └── storage/                 # local/SQLite adapters; future object/Postgres adapters
├── isl/                         # existing isolated feature, model and training modules
├── translation/
│   ├── datasets/                # continuous source adapters and caption normalization
│   ├── models/                  # visual/temporal/text decoder candidates
│   ├── training/                # configs and trainer
│   └── inference/               # continuous preprocessing and generation adapter
├── ml/                          # existing cross-task manifests, audit, metrics and reports
├── web/                         # existing React/Vite/TypeScript app
├── frontend/                    # legacy static live UI during migration only
├── data/
│   ├── raw/                     # ignored, externally acquired media
│   ├── manifests/               # versioned metadata, no media
│   ├── processed/               # ignored caches
│   ├── reports/                 # local generated reports
│   └── fixtures/                # only permitted, small test media
├── artifacts/                  # ignored immutable packages by task/version
├── runs/                        # ignored training outputs
├── docs/                        # audits, progress, requirements and model/data cards
├── notebooks/                  # thin Colab orchestration, no copied source logic
├── tests/                       # unit/data/API/integration/e2e layers
├── pyproject.toml               # pinned Python package/runtime groups
├── uv.lock                      # reproducible local environment lock
└── SYSTEM_ARCHITECTURE.md etc.  # top-level review artifacts
~~~

This is a target layout, not a reason to move all current files at once. The first implementation should add a v1 API, typed contracts and new translation package while keeping imports working. Move modules only when a change provides a tested migration path. Keep the historical isolated package stable until model-package compatibility has been determined.

## 6. Frontend workflows

| View | User steps | Visible state and evidence |
|---|---|---|
| Landing/task choice | Choose Isolated Recognition or Continuous Translation | Explain isolated vocabulary limits vs sentence generation |
| Upload | Choose/drop video, see preview, remove/reset, submit | Formats and limits; client checks are convenience only; server status/progress |
| Isolated result | Read label/candidates if supported | Confidence only if calibrated, model/preprocess version, latency and vocabulary warning |
| Continuous result | Read generated English, copy text, retry/reset | No confidence %, model version, latency and semantic-error warning |
| Test library/evaluation | Run permitted manifest samples and inspect report | Reference only in evaluation mode; failed rows and excluded counts visible |
| Model/system | Inspect availability, package and supported task | No fake/demo output in inference UI; explicit not-ready state |
| Live isolated | Grant permission, start/stop/reset, observe segmentation | Camera state, processing, uncertain result, dropped frames, recoverable errors |
| Live continuous | Not shown as supported until phase gate passes | If experimental, explicit provisional/final/revised transcript state |

Accessibility: semantic HTML, keyboard operation, visible focus, status announcements, readable contrast, responsive layout and reduced motion. Use environment-based API base URL. Keep `web/src/api.ts` as the single typed network layer.

## 7. Security and deployment assumptions

- Accept multipart video using a spooled file handle; never read the full configured maximum into RAM in one call. Reject if reverse proxy or application byte limit is exceeded.
- Generate random temporary path; store outside static/executable roots; no client path concatenation. Validate actual media and decoded frames. Cap concurrent decode/inference work.
- Retain uploads only for the processing/job TTL. Cleanup on exceptions, cancellation, timeout and restart. Persist only the minimum result metadata.
- Restrict CORS to configured origins; bind local development to loopback. Public WSS requires authentication, origin checks, per-connection limits and TLS.
- Use environment configuration and `.env.example` placeholders only. Ignore `.env`, keys, dataset media, model packages, caches and reports.
- Container is a single deployable API/web asset initially; model files mount read-only or arrive from a versioned private artifact store. Health is not readiness. Rollback is switching the immutable model version.

## 8. Verification strategy by boundary

The test design includes pure preprocessing/manifest unit checks, data-integrity fixtures, model-package/schema checks, mock-model API tests, video validation tests, React workflow/UI tests, and a separately labelled real-model integration run. Add no metric claim from mocks. All 44 Python tests, compileall and the React production build passed. The suite was run outside the sandbox because its Windows asyncio socket-pair setup blocks under sandbox restrictions. Real-model integration still requires permitted media and model artifacts. No model metric is claimed.
