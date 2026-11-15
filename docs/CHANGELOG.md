# Changelog

## 2026-11-10 — Python 3.12 lock, INCLUDE ingestion hardening, Colab notebook

- Regenerated `uv.lock` for Python `>=3.11,<3.13` (it was `==3.11.*`, which made CI's `--locked` sync fail); verified `uv sync --locked` and the full suite on 3.11 and 3.12 (CPU-only Linux).
- Added `tests/test_v1_media_e2e.py`: real synthetic MP4s through the real `/api/v1` upload/decode/limit/cleanup path with a labeled stub model.
- Fixed `isl.download_include`: case-insensitive `--keep`, zero-match warning, delete-zips-aware free-space check.
- Added INCLUDE label/path audit and exclusion to `adapt_include_records`, and `ml.datasets.include_manifest` (canonical manifest + keep-list + audit report; never approves rights implicitly).
- Added `notebooks/colab_include50_isolated.ipynb` (statically checked; not run on Colab).
- Re-checked INCLUDE and iSign terms and file sizes against primary pages; see `docs/DATASET_INVENTORY.md`.
- Read AI4Bharat/INCLUDE (4 of ~14 files; see `docs/REFERENCE_REPO_ANALYSIS.md`); added opt-in `--aug-rotation-deg` / `--aug-time-warp`; rewrote the Colab notebook with step explanations, Drive persistence and a same-seed ablation.
- `ml.datasets.preprocess` is now resumable (cache reused only when stored source SHA-256 + pipeline version match) and has `--continue-on-error` (failures listed in `preprocess_failures.json`) and `--no-resume`; previously one unreadable video aborted the whole single-process run and a rerun redid everything. Default behaviour stays strict. 11 tests, mutation-checked.
- Colab notebook: recommends the CPU runtime (training measured at ~1.75 s/epoch on one CPU thread with synthetic data; MediaPipe is CPU-only and single-process), no longer requires Colab's own Python to be 3.11/3.12 (uv installs 3.12).
- Fixed notebook Step 1: it called `nvidia-smi` unguarded and crashed on the recommended CPU runtime (found on the user's first Colab run: Python 3.13.18, 2 CPU cores). Now checks `shutil.which` first.
- Fixed Colab crash: `MPLBACKEND=module://matplotlib_inline...` made `import matplotlib` (and therefore MediaPipe) fail in the project venv. `isl/__init__.py` now swaps any `module://` backend for `Agg`; notebook Step 1 also sets it. 6 fresh-interpreter tests, mutation-checked.
- Hub metadata loader now explains failures (with a manual fallback) and checks columns; notebook gained Step 7b (lists real Zenodo archives, picks the smallest one containing INCLUDE-50 videos for the smoke test) instead of assuming an archive name.
- Recorded the first real INCLUDE metadata audit (5,250 rows; INCLUDE-50 = 958 rows = 943 unique videos + 14 mislabeled + 1 exact duplicate).
- Downloader: fixed crash in the archive-total line that blocked every invocation (it summed checksum strings); made `--delete-zips` runs resumable; extraction now lands at the keep-list path regardless of archive prefix/case; `main()` is now a testable function with 9 end-to-end tests against a fake Zenodo record.
- `isl/flow.py` + rewritten notebook: idempotent, resumable "new day" workflow (see PROJECT_PROGRESS).
- Downloader: zero-match diagnostics, unambiguous label/file fallback, `--stop-after`/`--max-archives`, `--inspect all` totals and clear "ranges unsupported" exit (code 3).
- Reviewed and kept another contributor's `--remote`/`--inspect`/`RangeFile` and `resolve_cache_paths` changes (see PROJECT_PROGRESS).
- 175 tests pass. Still no dataset, trained model, Colab run, Docker build, GitHub push or deployment.

## 2026-11-10 — Architecture and evidence baseline

- Restored prior project source from `isl-engine.zip` into the empty workspace, excluding the archive's nested `.git` metadata.
- Added the five requested architecture documents and Phase 0 audits/progress/traceability.
- Recorded that no dataset, model package or test video is present; no tests, notebooks, training or inference were run in the static audit.
- Identified source count disagreements, two differing Colab pipeline revisions, security/API gaps and architecture proposals that need evidence before adoption.

## 2026-11-10 — Architecture and Phases 1–8 implementation foundation

- Added the canonical task-aware manifest, source adapters, rights/split/media validation, fingerprinting, near-duplicate candidate reports and provenance-preserving landmark preprocessing.
- Hardened isolated training, validation-only temperature/abstention calibration, opt-in mirroring and ONNX parity checks; added a separate continuous pose/hand-to-English Transformer training/evaluation/inference baseline.
- Added bounded upload staging, actual video validation, per-task v1 readiness/inference contracts, resource limits, cleanup and separate React workflows. Legacy live recognition stays gated; live continuous translation stays deferred until a causal model is measured.
- Added Python 3.11 lock/setup, optional experiment dependencies, CI/container configuration and updated architecture, traceability, audit and progress documentation.
- Verification: `uv lock --check --python 3.11`, Python compileall, 44 Python tests, `npm run build`, and `docker compose config --quiet` passed. No real dataset, model, training run, real-video inference or Docker image run was available.
- Added `COLAB_END_TO_END_GUIDE.md` with a non-commercial research/demo dataset recommendation, rights-aware Drive policy, Colab setup, data-validation, training, model promotion, integration, GitHub and deployment steps. Expanded declared Python support to 3.11–3.12 and added a Python 3.12 CI matrix for Colab compatibility; the 3.12 runtime smoke check is pending.
