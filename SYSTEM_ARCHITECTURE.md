# ISL Translation Engine 2.0 — System Architecture

**Status:** Architecture baseline, 10 November 2026  
**Evidence labels:** *Present* means found in this workspace or restored source archive; *reported* means stated by a supplied document or public source; *proposed* means a design choice not yet built or measured.  
**Product boundary:** Video to English for two separate tasks: one bounded isolated sign, or a continuous signed phrase/sentence.

## Architecture goals

1. Give isolated recognition and continuous translation separate data, preprocessing, model, evaluation and result contracts.
2. Reuse the validated dataset and inference utilities already in the source archive, while keeping unverified weights, scores and data claims out of the shipped path.
3. Make uploaded video the first end-to-end workflow. Add camera recognition only after upload inference is stable; treat streaming translation as a research gate.
4. Make every prediction traceable to an input digest, task, model package and preprocessing version. Do not retain user video by default.
5. Fit local Windows development and a single Google Colab GPU workflow before introducing distributed infrastructure.

## System context and component interactions

~~~mermaid
flowchart LR
  U[User] --> UI[React and TypeScript web app]
  UI -->|HTTPS multipart video| API[FastAPI modular application]
  API --> V[Media validation and bounded decode]
  V --> TMP[Ephemeral local or object storage]
  API --> ROUTER[Task router]
  ROUTER --> ISO[Isolated recognition adapter]
  ROUTER --> CONT[Continuous translation adapter]
  ISO --> PREI[Versioned isolated preprocessing]
  CONT --> PREC[Versioned continuous preprocessing]
  PREI --> MR[Model package registry]
  PREC --> MR
  MR --> ISOM[Isolated model]
  MR --> CONTM[Sequence-to-text model]
  API --> JOBS[Job metadata repository]
  JOBS --> DB[(SQLite local; PostgreSQL when multi-instance)]
  ISOM --> RESP[Typed result and safe error mapping]
  CONTM --> RESP
  RESP --> API --> UI
  TMP --> CLEAN[TTL cleanup]
  TRAIN[Offline training on Colab or local CPU] --> DS[Dataset adapters and immutable manifests]
  DS --> QA[Media, annotation, rights and leakage validation]
  QA --> CACHE[Versioned processed cache]
  CACHE --> TRAIN
  TRAIN --> EVAL[Held-out evaluation and model card]
  EVAL --> MR
~~~

The first deployment is one Python application process, one bounded inference worker, a local artifact directory and SQLite job metadata. API routes do validation and orchestration only; they do not contain model code. Dataset preparation and training are offline commands, not request-time operations.

## Isolated sign recognition

**Input contract:** one clip intended to contain one lexical sign. The result is a known vocabulary label or an explicit unusable/uncertain status. It is never presented as arbitrary sentence translation.

**Initial path:** validated video → deterministic frame sampling/decode → the existing MediaPipe Tasks landmark extractor → scale/translation-normalized 170-value frame features → whole-clip temporal resampling → candidate classifier → validation-calibrated abstention policy → typed recognition result. The existing 170-value pipeline and segmenter are useful baselines, not approved production behavior: they omit facial landmarks and per-point visibility, and their confidence thresholds have not been calibrated on held-out signers.

**Model ladder:** majority/frequency sanity baseline; existing Conv1D–BiLSTM–attention model if its checkpoint, labels and preprocessing can be verified; compact temporal CNN/GRU candidate; RGB or RGB-plus-landmark comparison only when the same split and compute budget are available. The selected model is the one that generalizes and abstains safely, not the largest model or highest leaky score. Keep the training framework in PyTorch. Export a compact approved isolated model to ONNX only after numerical parity and measured benefit; ONNX Runtime supports cross-platform inference and multiple execution providers ([ONNX Runtime](https://onnxruntime.ai/docs/)).

**Unknown and idle behavior:** use input-quality/activity checks plus a rejection policy calibrated on validation data. Raw softmax is a score, not a calibrated probability. Idle examples, if used, must come from source-matched rest segments rather than a different room or signer domain. No robust open-vocabulary detector is claimed until it has its own evaluation.

## Continuous ISL-to-English translation

**Input contract:** a complete clip containing continuous signing and one or more preserved English references for training/evaluation. Output is generated English text, with model version and processing time. It has no label-classifier confidence field.

**Model boundary:** video/pose visual encoder → temporal encoder → English subword decoder. Sentence-level supervision trains direct visual-to-text generation. The first low-compute experiment should be a pose-to-text baseline using a verified compatible pose representation; compare it with RGB features on the same source-video-disjoint split. RGB is needed as a serious candidate because keypoints can discard handshape and non-manual/facial information. The existing 170-value isolated-sign features are not automatically a valid continuous representation. Store a source-specific extractor and input contract with each model.

Use token-level sequence-to-sequence loss and autoregressive decoding when only sentence references are present. CTC is reserved for a separately annotated monotonic gloss/label task; applying it directly to English sentence text would assume frame/token alignments that the supplied sources do not provide. Do not append isolated class predictions and call the result translation. Do not add RAG, a chatbot, agent orchestration or unconstrained language-model rewriting: none supplies missing visual evidence.

The iSign benchmark reports video-to-text and pose-to-text tasks, 118,228 pairs, and cautions about linguistic and dataset limitations. Its current dataset card says research-only/non-commercial, gated access, and recommends grouping rows by their shared video ID. These conditions make it a candidate research corpus, not a production license ([iSign paper](https://arxiv.org/abs/2407.05404), [current iSign card](https://huggingface.co/datasets/Exploration-Lab/iSign)). ISLTranslate reports 31,222 pairs drawn from educational material, but its paper alone does not establish redistribution rights; confirm source terms before using video or derivatives ([ISLTranslate paper](https://aclanthology.org/2023.findings-acl.665/)).

## Dataset and artifact boundary

Source adapters keep INCLUDE, iSign, ISLTranslate and any later corpus separate. Each canonical sample records task, source/version, source sample/video ID, file checksum, original label or reference text, license/access status, official split, grouping key and preprocessing version. Preserve original text and keep metric-normalized text separately. Split by signer when real signer metadata exists; otherwise group by source video/session and label that limitation. Never infer signer identity from filenames.

Raw media, caches, checkpoints and user clips stay outside Git. Manifests, schemas, configs, code and small permitted fixtures are version-controlled. Colab uses its runtime disk for active decoding and feature caches; Drive stores resumable manifests, checkpoints and reports. The present workspace contains no dataset files, trained model, label package or test videos.

## Backend, frontend and user workflow

- **Backend:** FastAPI, Pydantic request/response schemas, explicit task adapters, bounded video validation, model registry, inference services, job repository, structured logging and cleanup service.
- **Frontend:** retain the existing React/Vite/TypeScript application as the product UI. Put network access in one typed API client and keep isolated and continuous forms/results separate. The legacy static webcam frontend is a migration baseline, not a second product to maintain indefinitely.
- **Upload workflow:** choose task → preview and client-side file-size check → server-side byte/container/stream/duration/dimension/frame-count/decode validation → temporary storage → short synchronous inference or accepted job → result with model/preprocessing version, processing time and limitations → cleanup.
- **Job storage:** SQLite for a single-user/local process; only job/result metadata is persisted, under a TTL. Move to PostgreSQL and a durable worker queue only when multi-instance deployment, concurrency or restart guarantees require them.

## Live modes

### Live isolated recognition

Reuse a browser camera preview and MediaPipe Tasks Web where supported. A bounded state machine segments a whole sign; the client sends versioned landmark frames over authenticated WSS rather than camera video. Server-side normalization, segmentation, model inference and prediction stabilization remain explicit. Python and browser extraction must pass golden-video parity before this mode is enabled. Backpressure drops stale frames; sequence numbers, timestamps, max message size, idle timeout, reconnect and stop/reset are part of the protocol. MediaPipe supports image, video and live-stream modes, and live-stream processing may drop input frames when busy ([Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)).

### Live continuous translation

Disabled until a causal/streaming model has shown acceptable offline quality, chunk-boundary behavior, time-to-first-result and hypothesis stability. First support record-then-translate. If streaming later passes the gate, use a separate session protocol for sampled compressed frames or an explicitly approved on-device encoder; do not reuse the isolated-landmark protocol for a model that needs RGB/facial evidence. WebSocket is adequate for sequenced frame/event exchange; WebRTC is unnecessary unless this becomes an interactive audiovisual call.

## Persistence, deployment, security and operations

- **Database:** SQLite in development for expiring jobs; no accounts or permanent history by default. PostgreSQL is the scale-out option for shared job state.
- **File/model storage:** local directories in development; a private object store with lifecycle deletion and scoped access for multi-instance deployment. Model packages are immutable and include weights, labels/tokenizer, architecture, preprocessing, source manifest hash, split ID, metrics, licenses, model card and SHA-256 checksums.
- **Deployment:** one Docker image for the API and static web build at capstone scale. CPU inference is the default. A separate GPU worker is a later option for continuous models that require it. Do not promise Vercel, AWS, GPU or public hosting until configured and exercised with real artifacts.
- **Upload safety:** generated temporary names, content/stream validation, byte/duration/frame/dimension limits, decode timeout/worker bounds, no user path use, cleanup on all outcomes, no raw-video logs, and explicit retention. WSS is disabled outside localhost until authentication, origin checks and message bounds exist.
- **Observability:** structured request ID, task/model/preprocess version, status, stage timings and safe error code; readiness reports each model independently. Record failures, latency, queue depth and resource use. Do not claim drift monitoring or collect raw video without a real method and consent.
- **Responsible use:** describe results as experimental assistive output. Do not represent them as a qualified human interpreter or use them in high-stakes settings.

## Local Windows, VS Code and Colab workflow

Use VS Code with PowerShell and uv. The current shell Python is 3.14.8 and ML dependencies are absent; use a project virtual environment on a supported, pinned Python release (start with Python 3.11, verify MediaPipe/Torch wheels before pinning). Local Windows is for code, API/UI integration, small permitted fixtures and CPU smoke work. The detected GPU is an RTX 2050 with 4 GB; it is not the assumed training target for full corpora. Colab is for GPU training: install from a locked requirements/export file, stage only authorized data to runtime storage, restore/save checkpoints in Drive, record runtime/library/GPU versions, and download the immutable model package. Never replace repository source files from notebook cells.

## What exists, reuse and redesign

| Area | Evidence in this workspace/archive | Decision |
|---|---|---|
| Isolated features/model | isl/features.py, landmarks.py, model.py and spotter.py; source archive commit 13f2ed6 | **Keep with modification.** Candidate baseline only; add confidence/visibility quality, artifact metadata, calibration and parity evidence. |
| Data tooling | INCLUDE downloader/manifest/extractor plus new canonical/adapters/fingerprint/media/near-duplicate/preprocess modules | **Keep with modification.** Generic safeguards exist; exact corpus counts/rights/media remain unverified. |
| Isolated inference | ONNX predictor, label/input contract, validation-temperature sidecar, whole-clip pipeline | **Keep with modification.** No data or artifact exists; calibration and abstention code is implemented but not trained. |
| Backend | FastAPI legacy compatibility routes plus typed `/api/v1` readiness/recognize/translate and bounded media staging | **Redesigned at boundary.** Runtime API/model integration remains unverified. |
| Frontend | React/Vite isolated and continuous workflows; separate legacy webcam page | **Keep React; migrate live flow.** Production TypeScript/Vite build passed; live parity gate remains closed. |
| Continuous translation | `/api/v1/translate` with separate Transformer trainer/inference scaffold | **Built as research baseline.** No authorized data/model artifact, so readiness returns MODEL_NOT_READY. |
| Deployment/security | Multi-stage Docker, non-root runtime, health probe, upload/job TTL and resource limits | **Foundation hardened.** Container build, staging and production controls remain unverified. |
| Dataset/model evidence | No local media, checkpoints, labels, model cards or evaluation reports | **Must be acquired/created.** No real inference or metric can be claimed yet. |

## Validation against supplied materials

The full Markdown specification is the controlling requirements source. Its important boundaries are preserved: different task adapters and metrics, provenance and licensing, split integrity, real model artifacts, upload validation, separate live gates, privacy and no fake outputs. The older SRS describes an isolated 51-class system; it is not evidence that its reported 943 videos, 12 passing tests, or model artifacts are present here. The supplied project archive contains more code than the current checkout originally had, but no trained weights or data. The ZIP’s embedded commit is 13f2ed664e8c880ccbb3b118ab2d319c50a33d6f; its embedded Git directory was not copied into this checkout.

The prior architecture DOCX proposes a Transformer plus CTC, pose-only edge streaming, RAG/agent orchestration, Redis/Postgres, AWS GPU hosting and Vercel. This design changes those proposals for evidence and scope reasons: CTC needs an aligned label sequence and does not solve sentence translation; keypoints can omit handshape/facial information; retrieval cannot replace visual translation; and no current load, access or deployment evidence justifies distributed services. These remain candidates only where a later measured experiment supports them.

## Primary references

1. [INCLUDE dataset card](https://huggingface.co/datasets/ai4bharat/INCLUDE) and [INCLUDE paper](https://doi.org/10.1145/3394171.3413528). The current card describes metadata rows with video paths and the INCLUDE-50 flag, marks CC-BY-4.0, points to Zenodo for the actual videos, and notes Chennai-region coverage. Reconcile current metadata against the actual archive before counting usable samples.
2. [iSign paper](https://arxiv.org/abs/2407.05404) and [iSign dataset card](https://huggingface.co/datasets/Exploration-Lab/iSign).
3. [ISLTranslate paper](https://aclanthology.org/2023.findings-acl.665/).
4. [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) and [MediaPipe Tasks overview](https://developers.google.com/edge/mediapipe/solutions).
5. [ONNX Runtime](https://onnxruntime.ai/docs/).
6. [FastAPI file upload guidance](https://fastapi.tiangolo.com/tutorial/request-files/).

## Implementation evidence update — 10 November 2026

The modular boundary is now represented in code: canonical data adapters/validation/preprocessing under `ml/datasets`, independent isolated and continuous model paths, typed `/api/v1` health/readiness/recognize/translate routes, bounded private upload staging, and distinct React task pages. Isolated calibration is fitted from validation outputs and bound to ONNX/label hashes; uncalibrated inference is labelled as such. Continuous translation has an experimental pose/hand Transformer trainer and offline adapter, but no real corpus or trained artifact is present. The legacy live WebSocket is disabled by default and only allows opt-in loopback use. The Python 3.11 environment, focused tests, frontend build and compose configuration were verified; Docker image/runtime checks remain pending because the Docker Desktop Linux engine is not running.

Dependency-light manifest/text contract tests and the React TypeScript/Vite production build have passed. Full API/model tests, real-media decode, training, Colab, container build and deployment remain unverified. No source counts, model metrics or product acceptance claims have changed.
