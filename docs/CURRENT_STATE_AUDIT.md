# Current State Audit

**Audit date:** 10 November 2026  
**Workspace:** `C:\Users\TARNDEEP SINGH\OneDrive\Documents\ChatGPT\ISL`  
**Audit type:** static inventory and source review; no tests, notebooks, training or inference were run during this audit.

## Repository and provenance

- The workspace initially contained only `.git`; `git status` reported `No commits yet on master`, with no source files or user changes to preserve.
- The user supplied `C:\Users\TARNDEEP SINGH\Downloads\isl-engine.zip` (294,521 bytes). It contains 75 source/project files, including an embedded Git history pointing to commit `13f2ed664e8c880ccbb3b118ab2d319c50a33d6f` (`Word-first recovery: CV runner, vocab selection, idle from dataset rest frames`). I restored its project files into this empty workspace and excluded the nested `.git/`; the original ZIP remains untouched.
- The working tree now contains those restored files as untracked source. No current workspace commit/hash exists.
- `isl-debug-update.zip` is a partial backend/frontend package, not a complete replacement: it removes the ML API router/job/evaluation endpoints from backend/main.py and removes MediaPipe/OpenCV from its requirements. It was inspected and not overlaid on the fuller archive.
- The earlier named `ISL_Translation_Engine_2_0_Codex_Master_Prompt_All_Phases.docx` is not present in Downloads under the provided name. The complete Markdown specification is present and is the controlling requirement source.

## Inventory

Restored project inventory: 75 files total, including 41 Python files and one repository Colab notebook. There are six test modules with 31 statically visible `test_*` functions. The existence/count of tests is not evidence they pass; none were executed here.

| Path | Observed responsibility | Audit classification |
|---|---|---|
| `isl/` | Isolated landmark feature extraction, normalization, segmentation, model, training, export, download, manifests and live simulation | Useful prior implementation; candidate code, not validated runtime |
| `ml/datasets/` | Manifest assembly, session/group splitting, fingerprints, idle-cache and vocabulary experiments | Keep with audit and rights/schema hardening |
| `ml/experiments/` | Cross-validation and vocabulary support analysis | Useful experiment code; no real corpus run evidence in workspace |
| `ml/evaluation/` | Word metrics, per-class report, batch inference and report files | Reuse with schema and provenance updates |
| `ml/audit/` | Read-only tree/environment/dataset inventory | Useful baseline; current audit documents supersede the empty generated audit |
| `ml/test_library/` | Copies held-out samples and builds manifests | Keep with data rights/cleanup boundaries |
| `backend/` | FastAPI health, raw landmark WebSocket, model loading, upload/jobs, test library, reports, subtitles and 501 translation route | Reusable service skeleton; API/security redesign needed |
| `web/` | React/Vite/TypeScript upload, result, test library, evaluation and model screens | Keep as primary UI; API migration needed |
| `frontend/` | Separate static live webcam client | Legacy reference; migrate only after live gates |
| `tests/` | Synthetic-data/preprocessing, manifest, metric, API and test-library tests | Existing verification assets; not run in this audit |
| `notebooks/` | Colab pipeline in the project archive | Thin-ish orchestration is useful, but two differing notebook revisions exist and need consolidation |
| `data/`, `models/` | `.gitkeep` placeholders only | No media, cache, weights or labels |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | Container, local service and CI | Foundation exists; Docker/CI omit code/dependencies required by ML API and must be repaired |
| `README.md`, `docs/` | Setup, dataset inventory, implementation plan, known issues | Several claims are stale or unverified; replace with evidence-labelled docs |

## Model/data/artifact facts

- No INCLUDE/ISLTranslate/iSign/CISLR media is inside the workspace.
- No label mapping JSON, trained checkpoint, ONNX/TFLite/H5 file, tokenizer, model card, training log or real evaluation report is inside the workspace.
- `models/` contains only `.gitkeep`; `data/` contains only `.gitkeep`.
- `backend.engine.load_predictor` returns no predictor if `isl.onnx` and its label file are missing. `backend.api_ml` returns 503 for prediction when no pipeline is loaded; `/api/translate` returns 501. These are honest stubs, not real translation.
- Existing README and SRS mention 12 passing tests, but the current source tree has 31 statically declared test functions and no test output in the current checkout. Treat both counts/statuses as unverified reports until run.
- Source dependencies are mostly unbounded lower limits; the backend requirements are a partial subset. No root `pyproject.toml`/uv lock exists.

## Application entry points and known integration gaps

- API entry point is `backend.main:app`; it loads the isolated model at import time and serves legacy `/health` and `/ws` plus `/api/*` router routes.
- The legacy WebSocket accepts free-form JSON landmark messages and has no protocol version, auth, origin policy or message bounds.
- Upload code checks extension and byte length but reads the full request limit into memory, does not validate actual video streams/duration/codecs, stores uploads permanently under `data/uploads`, and has no retention cleanup.
- Job state is SQLite and inference is globally constrained through a single thread pool. This is acceptable only as a local prototype, not a multi-instance job system.
- Model confidence is raw softmax; no calibration artifact or validation policy is recorded.
- The current continuous `/api/translate` path intentionally returns 501.
- Docker copies `backend`, `isl`, `frontend` and `models` but omits `ml/`, `web/dist` and some root dependencies used by the API. The CI workflow installs only backend requirements and may omit test imports/dependencies.
- `README.md` advertises 12 passed and lists commands that have not been run in this workspace. The old docs' completion tables are not current status evidence.

## Relevant supplied files reviewed

- Full Markdown specification: `C:\Users\TARNDEEP SINGH\Downloads\ISL_Translation_Engine_2_0_Complete_50_Page_Codex_Specification.md`.
- SRS/project report, previous-vs-new report, technical blueprint, architecture DOCX and dataset roadmap DOCX were opened and text inspected. Architecture proposals are treated as hypotheses, not requirements.
- Both the full `isl-engine.zip` and partial `isl-debug-update.zip` inventories/code were inspected.
- Notebook review is in `NOTEBOOK_AUDIT.md`.

## Evidence limits

This is a static baseline. No package installation, API launch, test execution, dataset download, video decode, notebook restart, GPU training or deployment was performed. The README, SRS and report claims that say a result passed are not reproduced by this audit. The model and data integration gates therefore remain open.
