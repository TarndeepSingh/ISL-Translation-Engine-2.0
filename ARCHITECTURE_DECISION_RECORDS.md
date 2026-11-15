# Architecture Decision Records

**Status:** Proposed baseline for ISL Translation Engine 2.0  
**Date:** 10 November 2026  
**Decision rule:** evidence and reproducibility take priority over novelty or infrastructure breadth. A model or service is not approved merely because its code exists.

## ADR-001 — Use a modular monolith for the first supported release

**Context:** The restored repository has one FastAPI app, a React web client, a local SQLite job store, and no running deployment or observed request load.

**Options:** (A) split API, isolated inference, translation inference, queue, and media services now; (B) keep clear Python service boundaries in one deployable app, then split a measured bottleneck later.

**Decision:** B. Use one FastAPI application with API routers, schemas, validation, preprocessing, registry, task adapters and storage behind interfaces. Long work uses a bounded local worker and job record. Add PostgreSQL/object storage/durable queue only when multiple replicas, durability or measured concurrency require them.

**Consequences:** Lower setup and deployment cost; one process is a scaling ceiling and local background jobs need restart recovery. Define task and storage interfaces now so later separation is bounded.

## ADR-002 — Treat isolated recognition and continuous translation as separate ML products

**Context:** A closed-set classifier returns one label from a fixed vocabulary. Sentence translation requires temporal and linguistic context and generates a variable-length English sequence.

**Options:** (A) concatenate isolated predictions; (B) separate data, preprocessors, model packages, interfaces and evaluations while sharing only API, storage and UI shell.

**Decision:** B. The isolated adapter returns a label/gloss, optional top-k and validation-supported confidence; the continuous adapter returns generated text and no fabricated confidence. The UI offers explicit modes.

**Consequences:** More adapters and task-specific reports, but prevents word classification from being misrepresented as translation and permits independent release gates.

## ADR-003 — Preserve source identity and split before transformation

**Context:** Current reports disagree on INCLUDE counts and splits; iSign contains multiple ordered rows from one source video; datasets differ in signers, labels and terms.

**Options:** (A) merge by label and random-split rows; (B) source-specific adapters, immutable canonical IDs, exact and near-duplicate review, video/signer/session grouping, then train-only augmentation.

**Decision:** B. Use official splits where verified. For iSign group all translation rows by video ID. For isolated data use signer grouping only when true signer IDs exist; otherwise use the strongest available source-video/session group and state the limitation. Never manufacture signer IDs from filename patterns.

**Consequences:** Some classes/sources may be absent from a strict split and scores may fall. Results are more defensible. Keep official benchmark scores separate from any signer-generalization score.

## ADR-004 — Keep the 170-feature pipeline as a baseline, not as a product assumption

**Context:** The archive contains MediaPipe Tasks extraction and scale/translation-normalized pose-and-hand features, with a whole-segment model. It does not include a compatible trained checkpoint. Current features omit face and landmark visibility.

**Options:** (A) adopt the existing 170-value contract as final; (B) preserve it as version 1 baseline, add explicit quality/confidence handling and compare RGB or richer features under matching splits.

**Decision:** B. Retain the source implementation behind a preprocessing version and isolate future feature versions. Use MediaPipe Tasks for repeatable pose/hand baselines; compare against RGB or selected face/non-manual features because visual appearance and non-manual markers can carry meaning. Do not silently change features for an old checkpoint.

**Consequences:** A new model package is required for changed preprocessing. Browser/Python parity must be measured before live use. Face tracking raises privacy and compute costs and must be justified by evaluation.

## ADR-005 — Select isolated models by grouped validation, risk and coverage

**Context:** Prior reports describe an 51-class Conv1D–BiLSTM–attention model and a single-signer live result, but this workspace has no checkpoint, label JSON, evaluation output or data. Existing code has raw softmax output.

**Options:** (A) claim/ship the reported model; (B) use the old architecture only if exact artifact/preprocessing/label compatibility is proven, and compare it to simple and compact baselines on non-leaking validation.

**Decision:** B. Report top-1/top-5, macro and per-class precision/recall/F1, calibration, selective risk/coverage, latency and model size. Fit calibration and rejection thresholds on validation only. Hold the final test split until selection is finished.

**Consequences:** Model may be unavailable or limited to a subset of classes until enough data exist. A rejected/uncertain result is preferable to claiming arbitrary OOV recognition.

## ADR-006 — Use direct sequence-to-text for sentence supervision; do not apply CTC to English without alignments

**Context:** The continuous datasets described in supplied material provide video/pose and sentence references, not verified frame-aligned English tokens or full gloss sequences. ISL grammar, co-reference and non-manual markers make word concatenation inadequate.

**Options:** (A) Transformer encoder-decoder plus CTC over English; (B) video/pose encoder, temporal encoder and autoregressive English decoder trained with sequence-to-sequence loss; CTC only as a separate gloss/recognition auxiliary task when aligned labels exist.

**Decision:** B. Start with a modest pose-to-text baseline that matches a verified research task, then compare RGB using identical source-video-grouped splits. Keep a modality interface so fusion can be tested. Use a decoder/tokenizer saved with the artifact. Do not use a general LLM to rewrite output without visual grounding.

**Consequences:** Requires enough paired data and autoregressive evaluation; pose may lose visual detail and RGB is more expensive. Research baselines may perform poorly, so no translation mode is marked ready based on fluency alone.

## ADR-007 — Use PyTorch for training; export only when deployment evidence supports it

**Context:** The existing classifier is PyTorch and already has an ONNX export script. ONNX Runtime can run exported models across hardware, but conversion does not guarantee numerical parity or speed.

**Options:** (A) move the whole system to TensorFlow/TFLite; (B) retain PyTorch for research/training, run the continuous candidate in PyTorch first, and export the compact isolated model to ONNX only after parity and performance checks.

**Decision:** B. Keep ONNX Runtime CPU as the first portable isolated inference target. Do not require ONNX for an unsupported continuous graph until the chosen architecture exports cleanly and produces equivalent outputs.

**Consequences:** Two runtime formats may exist. Each export needs its own checksums, version and parity record; optimize only after measurements.

## ADR-008 — Use a bounded upload/job pipeline and ephemeral media storage

**Context:** Existing upload code reads up to 200 MB into memory, checks extension/size rather than actual stream properties, stores uploads persistently, and has no expiry cleanup. FastAPI UploadFile uses a spooled temporary file suitable for larger uploads.

**Options:** (A) keep the current endpoint; (B) stream/spool to generated private temp paths, validate decoded media and configured duration/frame/dimension limits, run bounded inference, persist only expiring job metadata/results and clean media on every terminal path.

**Decision:** B. Keep short inference synchronous if measured under request timeout; return a job ID for longer work. Do not introduce Redis/Celery until a single-process bounded worker is insufficient.

**Consequences:** Jobs and video cleanup must survive error and restart paths. The first supported deployment is a single instance. API error codes remain stable and never expose stack traces or internal paths.

## ADR-009 — Keep the current React/TypeScript UI; retire duplicate product surfaces after migration

**Context:** The ZIP includes a Vite/React UI for upload, evaluation, model and test-library pages plus a separate static webcam app.

**Options:** (A) rewrite in Flutter/mobile-first or maintain both clients; (B) extend React with two explicit workflows and migrate only the live isolated page once its model and parity gate passes.

**Decision:** B. Use typed API contracts and one API client. Retain the static page as a legacy reference until the React workflow is complete, then remove it only after route and functionality checks.

**Consequences:** The existing web stack is reusable now. Native mobile remains possible through the API if a demonstrated user need appears.

## ADR-010 — Defer genuine continuous streaming

**Context:** Offline translation models can use future frames; that is not streaming. No trained continuous model or temporal-chunk evaluation exists here.

**Options:** (A) emit periodic text from an offline model and call it live; (B) ship record-then-translate first, and require a causal/chunked model with bounded context, revisions and stability evaluation before enabling streaming.

**Decision:** B. Later live translation has a separate WSS session protocol, sequence numbers, backpressure, partial/final markers, revisions, cancellation, authentication and resource limits.

**Consequences:** Streaming can be explicitly deferred without blocking upload translation. No real-time claim is made from periodic UI refreshes.

## ADR-011 — Use local filesystem plus SQLite first; provide scale-out interfaces

**Context:** Current goal is a reproducible local capstone, not a multi-user service. Current files show SQLite as a development job store and no deployment infrastructure or load data.

**Options:** (A) add PostgreSQL, Redis, object storage, MLflow, DVC and an API gateway immediately; (B) local paths/SQLite behind interfaces and immutable artifacts, with documented production adapters.

**Decision:** B. Keep source code and small manifests in Git; keep raw datasets and model artifacts out. Use immutable model directories with manifest hashes and SHA-256. Adopt PostgreSQL/object storage/durable queue/model registry service only when operational needs justify them.

**Consequences:** Simple local setup and transparent provenance, but single-host limits and manual model promotion until scale requirements change.

## ADR-012 — Do not add retrieval, agents, or external language rewriting

**Context:** Supplied architecture documents propose RAG/LangGraph/agents. The project has no approved knowledge base that could determine visual sign meaning, and retrieval cannot replace a learned visual-to-language mapping.

**Decision:** Exclude from the translation path. Add a curated glossary lookup only if it is an explicit, user-visible aid after a grounded model output; never use retrieval to mask missing sign recognition.

**Consequences:** Fewer moving parts and lower hallucination risk. Future explainability or glossary support remains an optional post-processing component with provenance.

## ADR-013 — Gate dataset use on source-specific rights and release verification

**Context:** The INCLUDE card currently marks CC-BY-4.0 metadata and directs video download to Zenodo. The current iSign card requires acceptance of gated research-only/non-commercial terms. ISLTranslate’s paper describes sources but does not by itself prove the terms for copying or distributing media/model derivatives.

**Decision:** Record metadata state separately from actual media rights. Do not download or train on a corpus until its current release, access conditions, license, consent/usage restrictions and derivative model terms are recorded. Do not scrape ISH-News or public sources based only on availability.

**Consequences:** Continuous training may wait for terms or account acceptance. Architecture and adapters can be implemented while keeping the model unavailable.

## ADR-014 — Make training environment explicit and keep Colab source immutable

**Context:** Current local Python is 3.14.8, ML packages are absent, and the visible GPU is an RTX 2050 with 4 GB. The project needs MediaPipe and PyTorch wheels; Colab is the intended GPU environment.

**Decision:** Use uv and a pinned Python environment (begin with 3.11 and verify exact wheel compatibility). Keep notebooks as orchestration clients of repository modules. Stage data on Colab runtime disk, checkpoint to Drive, and never overwrite source modules using notebook cell magic.

**Consequences:** One extra environment setup step on this PC; reproducible scripts remain usable from local, CI and Colab. Colab runtime details and actual GPU availability must be logged per run.

## ADR-015 — Treat all reported results as unverified until reproduced

**Context:** SRS and reports mention accuracy, sample counts, tests and latency. This checkout originally contained only an empty Git repository; the source archive has no model weights, training logs or dataset media.

**Decision:** Use evidence labels in audit and progress docs. No accuracy, current data count, passing test count, completed training or deployment claim is made without a local artifact or recorded command output tied to a split/model/data version.

**Consequences:** The handover may list real blockers instead of a working trained model. This is necessary for a measurable and maintainable result.

## ADR-016 — Calibrate isolated scores on validation and default to abstention

**Context:** The previous predictor exposed raw softmax as confidence and used a fixed threshold without recorded calibration. A confidence percentage in the UI would overstate evidence.

**Decision:** Fit scalar temperature and a minimum-precision operating point on validation logits only. Bind the sidecar to the exact checkpoint, label map and ONNX file. Show raw scores as uncalibrated and do not emit a confident status without a matching validation sidecar.

**Consequences:** If validation cannot support the target precision and minimum accepted count, the chosen threshold abstains on all examples. No calibration artifact or operating point has been trained in this workspace.

## ADR-017 — Do not mirror signs unless label preservation is measured

**Context:** The prior isolated trainer mirrored 30% of training clips by default. ISL signs can encode meaning through handedness and direction.

**Decision:** Keep mirroring disabled by default. Enable it only after a task/vocabulary-specific label-preservation experiment and held-out evaluation. Keep the cache unaugmented and perform any accepted augmentation after splitting.

**Consequences:** Some left-handed augmentation benefit may be lost until validated; training records the explicit mirror option.

## ADR-018 — Bound upload, decode and inference concurrency

**Context:** A per-file byte limit does not protect a single-process API from many simultaneous decodes or inference calls.

**Decision:** Stream uploads to generated private temp paths, cap concurrent upload/decode/inference slots and pending jobs, validate the actual stream and TTL-clean raw media. Return stable retryable capacity errors instead of queueing unbounded work.

**Consequences:** The first service is single-process and bounded. Persistent multi-worker queues and object storage remain a scale-out decision backed by observed load.
