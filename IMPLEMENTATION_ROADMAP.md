# Implementation Roadmap — Phases 0–8

**Status:** Architecture and repository validation complete; implementation foundations for Phases 1–5 and security work in Phase 8 are in place. Updated 10 November 2026.  
**Important:** A phase is **implemented** only with the code/artifact and evidence it names. A runnable script without data, model weights or recorded outputs is not a completed ML phase.

## Current baseline

The active workspace started with an empty Git repository and no tracked files. I restored the 75 non-Git source/project files from the supplied 294 KB `isl-engine.zip` into the workspace to make the prior implementation reviewable and reusable. I excluded its nested `.git` directory. The archive itself points at commit `13f2ed664e8c880ccbb3b118ab2d319c50a33d6f`, message “Word-first recovery: CV runner, vocab selection, idle from dataset rest frames.” The checkout still has no commit. No dataset media, trained checkpoint, label artifact, test video, experiment report or production config exists here. The host has Python 3.14.8, uv, an NVIDIA RTX 2050 with 4 GB VRAM and about 32 GB free on C:. A separate project Python 3.11.9 environment is now synchronized. Colab account/runtime access is not visible in this workspace.

The provided `isl-debug-update.zip` is not applied: its backend removes the evaluation/job API and its minimal requirements remove MediaPipe/OpenCV; its changes do not supersede the complete source archive.

## Current phase state

| Phase | Implementation status | Acceptance status |
|---|---|---|
| 0 — audit | Static audit and five architecture documents complete | Passed for repository/document inspection; no data/model assumptions accepted |
| 1 — data/provenance | Canonical manifest, source adapters, fingerprinting, media/group validation, visual candidate scan and preprocessing path implemented; `ml.datasets.include_manifest` now produces the canonical INCLUDE manifest (previously no producer existed) with a label/path audit | Blocked on exact source release/rights and local media; no corpus run |
| 2 — isolated recognition | Existing classifier retained; split-safe trainer, opt-in mirror augmentation, validation temperature/abstention calibration and hash-bound ONNX sidecar added | No authorized data/model artifact; no real evaluation |
| 3 — continuous translation | Separate pose/hand Transformer, train-only word vocabulary, grouped trainer, multi-reference metrics and offline adapter added | Research scaffold only; no authorized paired data or trained model |
| 4 — backend | Typed `/api/v1` readiness/recognize/translate and bounded upload/decode/inference implemented; legacy API retained | Serving contract verified with real synthetic MP4s and a stub model (`tests/test_v1_media_e2e.py`: corrupt/short/long/oversize/high-resolution/non-video rejection, capacity 503s, crash safety, temp cleanup; mutation-checked). Real-model inference still unverified; both models currently not ready |
| 5 — frontend | Separate React workflows/readiness/result semantics wired to v1 client | `npm run build` passed; browser/API integration pending |
| 6 — live isolated | Legacy websocket disabled by default and loopback-only under opt-in | Gated on model, full protocol, browser/Python parity and measured stability |
| 7 — live continuous | Not implemented as streaming | Deferred until real causal/chunk model passes quality and latency gates |
| 8 — security/testing/deployment | Upload/job TTL, queue/concurrency caps, safe errors/paths, ignore rules, Docker/CI foundation implemented | Python environment, focused tests, compile and compose config pass; complete API suite, container and deployment/real-media checks pending |

## Architecture-to-spec validation and resolved contradictions

| Requirement or supplied proposal | Evidence checked | Decision / resolution |
|---|---|---|
| Separate isolated recognition and continuous translation | Full spec, SRS and code: current `/api/translate` is deliberately 501; isolated model is a 51-label candidate | Separate adapters, data, preprocessing, model contracts, metrics and UI. Never concatenate isolated labels as translation. |
| Preserve valid components | Full ZIP source and tests inspected; 75 files restored, no weights/data | Keep isolated feature/model/spotter, dataset tooling, evaluation utilities, FastAPI/React foundation, Docker/CI with bounded modifications. |
| The prior model/metrics are complete | SRS/report claims 943 videos, 12 tests passing, 0.88M parameters, live performance; current workspace has no checkpoint, local data or run output | Treat every count/score as reported only. Current INCLUDE card lists 5,250 metadata rows and paths to Zenodo videos; source counts are unreconciled. |
| A stronger architecture means Transformer + CTC | Architecture DOCX proposes both; continuous sources provide sentence-level references, no verified token-to-frame or gloss alignment | Use sequence-to-sequence visual encoder/English decoder; CTC only for an aligned gloss subtask. |
| Pose-only supports continuous translation | Existing model has 170 pose/hand values; iSign provides pose task, but paper highlights face/non-manual markers and modality differences | Pose-to-text is a low-compute baseline. Compare with RGB; no assertion pose is sufficient. |
| iSign is unrestricted training/production data | Current HF card requires gated access and says research-only/non-commercial | Candidate research corpus only after acceptance and terms review; no production rights inferred. |
| ISLTranslate can be downloaded and used from its paper | ACL paper documents 31,222 pairs and educational source; it does not alone establish current redistribution/model rights | Verify release and rights before acquisition; adapter/experiment remains gated. |
| RAG/LangGraph, Redis/Postgres, AWS/Vercel, DVC/MLflow are required | No deployment, load, knowledge source or multi-user requirement exists in workspace | Exclude from initial architecture. Keep interfaces and add infrastructure only for measured needs. |
| Live continuous mode can update a full-clip result periodically | Spec correctly distinguishes streaming from clip inference; no trained continuous model exists | Defer streaming until a causal/chunk model passes latency, quality and stability gates. |
| Dataset/reference is verified because the card exists | HF INCLUDE card has metadata and paths; source archive `data/` has only `.gitkeep` | Keep data status as reported/listed/downloaded/decoded/rights-reviewed separately; no local data accepted. |
| Master prompt is available | The earlier DOCX name from the conversation is not in Downloads; the complete 63.9 KB Markdown specification is present | Use the Markdown spec as the controlling requirements document; DOCX prompt is not treated as independently inspected. |

## Phase sequence and exit gates

### Phase 0 — Audit (done: static evidence)

**Deliverables:** repository, notebook, dataset, architecture-gap, environment and phase-decision audits; progress file; requirement traceability.  
**Exit gate:** source/archive inventory, stale claims, artifacts, commands and environment constraints are recorded. Notebook execution and runtime claims remain unverified because the environment has no dependencies/datasets and no compatible Colab session.  
**Next:** data provenance and reproducible manifest/validation implementation; do not wait for downloaded corpora to implement the generic checks.

### Phase 1 — Dataset acquisition, validation and preprocessing (implementation in place; data access gate open)

1. Implement one canonical manifest schema and source-specific INCLUDE/iSign/ISLTranslate adapters.
2. Implement license/access register, checksum verification, media/annotation validation, exact/near duplicate reports and split-overlap checks.
3. Generate group splits only from known source/signer/session IDs; label proxy grouping and benchmark splits honestly.
4. Add versioned preprocessing config and deterministic small-fixture pipeline; preserve the prior 170-feature pipeline as v1.
5. Acquire data only after permitted access and terms review; stage to Colab scratch and record real counts.

**Exit gate:** one command validates a permitted manifest, another preprocesses a permitted fixture; outputs are deterministic; license and grouping state are traceable. Actual corpus counts remain incomplete until media is present.

### Phase 2 — Isolated recognition and release (training/inference hardening implemented; model evidence missing)

1. Harden task contract, model loader, class map, whole-clip sampler, input-quality checks and calibrated abstention.
2. Reproduce sanity baseline and prior candidate on a verified split; compare a compact temporal model only if it answers a clear hypothesis.
3. Train in Colab after data access; save best/last, config, labels, split/data hashes and model card.
4. Evaluate per-class/macro metrics, calibration/risk-coverage, latency/size; verify ONNX parity before promotion.

**Exit gate:** permitted real held-out video passes stable inference, output maps to verified label ID, reproducible evaluation and provenance exist. Current state: no weights/data; not passed.

### Phase 3 — Continuous translation (separate research baseline implemented; data-rights gate open)

1. Preserve continuous video IDs/references and group all segments from a source recording.
2. Build dataset loader and text normalization that preserve originals and multiple references.
3. Implement a small pose-to-text pipeline, then a controlled RGB comparison if data/compute/terms allow.
4. Train and evaluate generated outputs with BLEU/chrF/ROUGE-L/WER/CER plus semantic error review; report per-source and failure counts.

**Exit gate:** real held-out continuous video generates text with exact split and metric provenance, model card and example/error review. If data/compute or licenses remain unavailable, keep translation disabled and document the exact blocker; the pipeline can still be runnable.

### Phase 4 — Backend integration (v1 inference routes implemented; integration evidence pending)

1. Add `/api/v1` typed schemas, health vs readiness, model metadata and task adapters.
2. Replace extension-only upload checks and full-memory copy with bounded/spooled uploads, actual video stream/decode validation and generated temp paths.
3. Separate short synchronous routes from bounded async jobs; add expiring job metadata, cleanup and safe errors.
4. Load model packages once, verify task/preprocess/label/tokenizer/checksum and fail readiness per missing task.

**Exit gate:** an actual permitted model/video passes validation → decode → preprocessing → inference → typed response; corrupt media, missing model, timeout and cleanup are exercised. No model means API reports MODEL_NOT_READY.

### Phase 5 — Frontend and evaluation workflows (distinct task workflows implemented; API integration pending)

1. Connect task selection and upload/preview to `/api/v1` using typed API client.
2. Add empty/loading/progress/result/error/not-ready states and separate task outputs.
3. Keep developer test library and batch evaluation references separate from normal prediction; preserve error rows and excluded counts.
4. Add accessibility and responsive review; do not show fabricated sample predictions in the live path.

**Exit gate:** both workflows and evaluation views operate against real API contracts; a real trained model is needed for success-state inference. Until then UI clearly shows model unavailability.

### Phase 6 — Live isolated recognition (deliberately gated; legacy route disabled by default)

1. Define bounded versioned WSS schema, authentication/origin policy, sequence numbers and backpressure.
2. Verify browser/Python landmark parity on golden authorized videos; segment whole signs; calibrate stabilization on validation videos.
3. Implement permission, camera active, stop/reset, processing, uncertain, network failure and reconnect states.
4. Measure capture-to-display latency, dropped frames, sustained CPU/RAM and supported browsers.

**Exit gate:** model artifact, parity, state handling, permissions and live latency/stability evidence pass. No trained model means keep it disabled.

### Phase 7 — Live continuous translation (deferred pending Phase 3 evidence)

1. Determine whether selected model is causal or supports bounded chunk inference.
2. Evaluate chunk size/overlap against complete-clip output, sign-boundary errors, revisions, pauses, packet drops and long-context bounds.
3. Implement separate WSS protocol with partial/final/revised states, auth, backpressure and cancellation only if quality is acceptable.

**Exit gate:** real causal/chunk model has measured time-to-first/final output, semantic quality, stability and resource use. Otherwise upload mode remains supported and Phase 7 is explicitly deferred.

### Phase 8 — Security, testing, performance and deployment (security foundations implemented; acceptance pending)

1. Threat-model uploads, temp storage, WSS, model files, secrets, logs and retention.
2. Add server-side limits, stream validation, safe errors, cleanup, environment example and ignore rules.
3. Build unit/data/model/API/UI/deployment test layers; separate mocks from real-model integration.
4. Pin packages and runtime; measure decode, preprocess, model, total, startup, memory and concurrency on named hardware.
5. Document single-container setup, staging, model promotion/rollback and monitoring; no public deployment claim without a real staging run.

**Exit gate:** clean setup, readiness, permitted real inference, invalid/oversized rejection, cleanup, deployment smoke, model/license traceability and rollback evidence. Current state: not passed.

## Development workflow

1. Use the uv-managed Python 3.11–3.12 project range; system Python 3.14.8 remains outside the supported project range. The local 3.11 wheel matrix is verified; smoke-test the lock and MediaPipe/Torch imports on the selected Colab 3.12 runtime.
2. Keep Windows/VS Code changes in repository modules and small permitted fixtures. Do not install or download large datasets in this task's workspace without checking terms, space and source.
3. Keep Colab notebooks thin: mount Drive, install the locked environment, stage authorized inputs to `/content`, call module CLIs, checkpoint to Drive and print hashes/counts. No notebook-local code that overwrites package files.
4. Log commands, test/verification outputs, data/model hashes and blockers in docs/PROJECT_PROGRESS.md after each meaningful change.

## Next concrete actions

1. ~~Rerun the complete suite outside the Windows sandbox.~~ Done on Linux (Python 3.11 and 3.12): 103 pass, including backend API tests that hung under the Windows sandbox. Repeat against a real model package once one exists; repeat the 3.12 check on the chosen Colab runtime with a GPU.
2. Build/review the Docker image when Docker Desktop's Linux engine is available and add runtime checks for corrupt/oversized videos, capacity responses and temp cleanup.
3. Reconcile a permitted dataset release and available disk/runtime; do not download the 56.8 GB INCLUDE archive into the current 32 GB free local space.
4. If data rights and Colab access are provided, run grouped isolated and continuous baselines, preserve full provenance, inspect errors and promote only measured artifacts.
5. Keep live isolated disabled and live continuous deferred until their independent quality/parity/latency gates pass.
