# Architecture Gap Analysis

**Audit date:** 10 November 2026  
**Comparison:** complete Markdown engineering specification + current restored source/archive + supplied SRS, reports, blueprint, architecture and dataset-roadmap DOCX.

This table records the repository state before the implementation work below. It is a baseline audit, not the current completion status; see [Project Progress](PROJECT_PROGRESS.md) and the phase gates in the root [Implementation Roadmap](../IMPLEMENTATION_ROADMAP.md).

| Area | What exists | Gap or contradiction | Target action |
|---|---|---|---|
| Product tasks | Isolated classifier and a separate `/api/translate` 501 stub | No actual continuous model; isolated classifier cannot be sentence translator | Keep task adapters, manifest schemas, model packages, metrics and results distinct |
| Isolated feature/model | 75×3 raw landmarks, 170-value normalized features, segmenter, Conv1D/BiLSTM/attention model/trainer/export script | No checkpoint/labels; raw confidence uncalibrated; feature masks incomplete; model not evaluated here | Preserve as v1 candidate; prove artifact compatibility and grouped evaluation before release |
| Continuous modeling | No trainer/model/adapter; only a deliberate 501 endpoint | Some supplied architectures propose CTC on English and pose-only live translation | Build visual-to-text sequence model; CTC only with real aligned gloss task; compare pose and RGB |
| Dataset ingestion | INCLUDE downloader, HF manifest, extraction; unified manifest/CV tools | No media; no rights register; mismatched source counts; missing checksum verification; some custom rows are split independently; no verified signer metadata | Canonical task-specific manifest, actual media validation, SHA-256, rights, group split and audit reports |
| Notebook consistency | Starter, action-detection and two different Colab pipeline revisions | Different frame/features (30×1,662, 30×258, or 32×170); row-level random split; unpinned installs; notebook source overwrites; none provides a clean run record | One thin notebook that calls repository CLIs and never writes source code |
| Inference/API | FastAPI, ONNX Runtime loader, REST upload/jobs, WebSocket | Old `/api` not `/api/v1`; no model package verification/readiness; free-form WS; extension-only upload; large in-memory read; persistent upload storage; no cleanup/retention | Typed v1 endpoints, actual bounded media validation, per-model readiness, safe temp store and TTL cleanup |
| Frontend | React/Vite developer screens plus duplicate static live page | No verified end-to-end API success path; duplicate UI state; no actual model | Make React the primary client; task-specific workflows and honest model-unavailable state |
| Database/files | SQLite job metadata and local paths; model/data placeholders | No TTL, durable job worker or object store; raw upload retention; model artifacts not present | SQLite/local files at dev scale behind interfaces; private object storage/Postgres later only if needed |
| Testing and CI | 6 test modules / 31 declared test functions; GitHub workflow | Not run in this task; README claims 12 pass; workflow installs partial dependencies; Docker omits `ml/` and `web/dist` | Run and record tests when implementation verification begins; repair CI/container and distinguish mock vs real-model integration |
| Deployment/security | Dockerfile, compose, .gitignore | Secret/data/model exclusions incomplete; no real deployment, resource bounds, retention, WSS controls, model rollback evidence | Single-container local/staging first; add controls, manifests, checksums and rollback |
| MLOps | Experiment/CV report utilities | No run IDs/manifest hash/model card/registry state or artifacts | Keep immutable packages and small JSON registry, add tools only if run volume justifies them |
| Monitoring | health route and timing fields in word pipeline | health can return `ok` with no model; no structured request IDs/counters/latency measurements | Separate health/readiness, log safe request/model/task/status timings and resource failures |

## Claims in supplied documents that remain unverified or are rejected

- The SRS says the isolated model has 51 classes, about 0.88 million parameters, 943 videos, browser/FastAPI/ONNX integration and 12 tests passing. Source code exists, but there is no checkpoint, dataset or test output in the workspace. Keep these as report claims only.
- The previous-vs-new report itself states it had not seen the implementation archive or run code. It correctly marks multiple issues as hypotheses. The later archive can confirm code paths, but cannot confirm metrics/training.
- The technical blueprint suggests RAG/agents and various production services; the supplied spec explicitly says not to add a chatbot/RAG merely for appearance. No approved textual knowledge source can replace video-to-language training, so these are excluded from the inference path.
- The architecture DOCX proposes Transformer+CTC, edge pose-only continuous translation, RAG/LangGraph, Postgres/Redis, AWS GPU and Vercel. CTC is not the sentence decoder when only sentence references exist. Pose-only is a candidate, not a sufficiency claim. No observed scale/deployment needs justify those services or hosts.
- The dataset roadmap contains quantitative optimization and accuracy/latency claims, but there are no corresponding model assets or measurements in this workspace; do not repeat them as verified results.

## Reuse versus redesign

**Reuse:** pure feature normalization/resampling/hand assignment; existing segmenter; dataset fingerprint/session experiment utilities; word metrics and test-library report generation; FastAPI and React scaffolds; Docker/CI as starting points.

**Redesign:** dataset schema and rights/provenance flow; split policy and confidence interpretation; artifact registry and readiness; upload validation/retention; REST/WSS contracts; frontend source of truth; training notebook/reproducibility.

**Build:** continuous data adapter, model/trainer/evaluator, continuous inference service and generation UI; real model package; integrated media/quality validation; model cards/registry records; real integration/performance evidence.

## Implementation after the baseline audit

The source gaps have since been addressed at code/scaffold level: canonical data/provenance tooling; isolated and continuous train/inference paths; API v1 readiness and bounded media routes; separate React workflows; security/retention controls; and deployment/CI configuration. Python 3.11 environment verification, compileall, the 44-test Python suite, frontend production build, lock check and Compose configuration all pass. The remaining release blockers are real authorized datasets, trained artifacts, real-video/model integration, Colab access and a running Docker engine; no performance or accuracy claim is made without those artifacts.

## Architecture mapping

Full component and phase rationale is in the five root architecture documents: `SYSTEM_ARCHITECTURE.md`, `ARCHITECTURE_DECISION_RECORDS.md`, `DATA_FLOW_AND_MODEL_PIPELINES.md`, `API_AND_COMPONENT_DESIGN.md` and `IMPLEMENTATION_ROADMAP.md`. The roadmap is the current status source; older `docs/IMPLEMENTATION_PLAN.md` is the archive's earlier plan and not completion evidence.
