# Requirements Traceability

**Updated:** 10 November 2026. “Implemented” describes code or docs present. “Verified” requires an executed check. Model/data acceptance remains separate from scaffold verification.

| ID | Requirement | Implementation and evidence | Status |
|---|---|---|---|
| SYS-01 | Isolated and continuous tasks have separate models/data/results | `isl/` isolated classifier; `ml/continuous/` seq2seq baseline; `/api/v1/recognize` and `/api/v1/translate`; separate UI pages | Implemented; no trained artifact |
| SYS-02 | Real supported video produces task-appropriate output | Upload/decode/inference code exists; no authorized dataset, local video fixture or model artifact | Blocked on data/model |
| DATA-01 | Source, release, rights, checksums and provenance are recorded | Canonical manifest, INCLUDE/continuous adapters, fingerprint tool and dataset register | Implemented; source terms/media not locally accepted |
| DATA-02 | Media and annotations are validated with retained error reports | SHA-256/size checks, bounded full decode, task-specific labels/references, rights and split checks | Implemented; no real corpus run |
| DATA-03 | Exact, near-duplicate and split leakage are checked | Exact hash/split checks; visual dHash near-duplicate candidate report requires manual review | Implemented; no real corpus run |
| DATA-04 | Augmentation follows splitting and retains lineage | Cache stage performs no augmentation; isolated mirroring is opt-in; augmentations remain runtime-only | Partial; lineage/label-preservation experiment still needed |
| ISO-01 | Isolated clip and vocabulary contract are explicit | Whole-clip 170D candidate, stable label output, quality checks, model package hash | Implemented; weights/data absent |
| ISO-02 | Isolated metrics include macro/per-class and calibration | Existing per-class/ECE evaluation plus validation-only temperature and precision-target threshold fitting | Implemented; calibration artifact is not trained/generated |
| ISO-03 | Artifact includes weights, labels, preprocessing and provenance | Checkpoint metadata, ONNX export, label map and checksum-bound calibration sidecar | Partial; no approved immutable release/model card |
| CON-01 | Continuous signing video maps to generated English | Separate visual Transformer, training-only vocabulary, grouped-split trainer and inference adapter | Implemented as research baseline; training blocked on rights-reviewed data |
| CON-02 | Translation evaluation has exact split/version provenance | Multi-reference BLEU/chrF, WER/CER, ROUGE-L, sample hypotheses and experiment metadata | Implemented; no evaluation run |
| API-01 | Versioned health/readiness/recognize/translate contracts | Typed `/api/v1` routes and independent per-model readiness | Implemented; mock/API contract tests pass |
| API-02 | Uploads are bounded and actual video streams are validated | Streaming byte limit, extension allowlist, full decode/duration/frame limits, bounded worker slots, generated temp files | Implemented; upload/job route tests pass with a mocked decoder; real media decode pending |
| API-03 | Model absence returns a stable error; no fake output | `MODEL_NOT_READY` on both task routes; model load failures are logged and readiness degrades | Implemented; unavailable-model route tests pass |
| API-04 | Jobs and media have bounded retention | Upload TTL cleanup, request cleanup, job metadata TTL and capped pending queue | Implemented; cancellation and durable multi-worker queue are not present |
| UI-01 | Task workflows are separate, typed and accessible | React/TypeScript isolated and continuous pages use v1 client; production build succeeds | Verified: `npm run build` |
| UI-02 | Users see version, score semantics and limitations | Isolated results distinguish raw from validation-calibrated score and show model version; continuous UI shows readiness gate | Implemented; browser review pending |
| LIVE-01 | Live isolated capture is bounded and measured | Legacy `/ws` is disabled by default and loopback-only when explicitly enabled | Gated; model, browser/Python parity and stability checks absent |
| LIVE-02 | Live translation is causal and has partial/final states | No streaming route/model is exposed; uploaded clip mode is separate | Deferred until a real causal model and quality/latency evidence exist |
| SEC-01 | Raw media, logs, path traversal and retention are controlled | Temporary generated filenames, cleanup, safe test-library path resolution, request IDs, ignore rules, size/decode/concurrency caps | Implemented; deployment threat review pending |
| SEC-02 | Dataset and derivative-model rights are reviewed | Rights states enforced by preprocessing/training and documented per source | Partial; no actual source media accepted |
| OPS-01 | Local, VS Code and Colab setup is reproducible | Python 3.11 project config, refreshed `uv.lock`, protected INCLUDE downloader, Docker multi-stage build | Partial; Python 3.11 sync and compose config pass; Docker engine and Colab unavailable |
| OPS-02 | CI/API/model/UI/deployment checks distinguish mocks from real data | Contract tests, route tests, existing model tests, Python CI and TypeScript build | Partial; all 44 Python tests, compileall and frontend build pass; real-data/model/container checks remain |
| OPS-03 | Monitoring, rollback and immutable releases are traceable | Request IDs, per-model version/hash, experiment records and container readiness probe | Partial; no production monitoring/release registry/staging |

## Evidence interpretation

- On Python 3.11.9, all 44 tests passed in 109.04 seconds when run outside the sandbox. `python -m compileall -q backend isl ml tests` passed. Within the sandbox, a minimal FastAPI `TestClient` request reproduced a wait inside Windows asyncio `socketpair` initialization; this is why the full suite was run with the approved local-socket environment.
- The frontend TypeScript/Vite production build passed after installing the locked npm packages.
- `docker compose config --quiet` passed; image build could not start because the Docker Desktop Linux engine is not running.
- API route tests pass with fixtures/mocked media decode. No model training, real-media decoding, dataset acquisition, continuous metric, real-model API inference, container build or deployment has been run.
- Synthetic test bytes are metadata fixtures only and are never counted as ISL videos.
