# Project Progress

**Last updated:** 10 November 2026  
**Current phase:** Phases 1–5 implementation foundation complete; Phase 8 hardening in progress  
**Overall status:** Architecture and implementation scaffold are in place. No authorized local data, trained model or Colab output is present, so real-world model acceptance remains blocked.

## Phase status

| Phase | Status | Evidence / blocker |
|---|---|---|
| 0 — audit | Complete | Source archive, specs, notebooks, current repository, environment and constraints audited; architecture reconciled with the Markdown spec. |
| 1 — data/provenance | Code foundation implemented; corpus gate open | Canonical task-aware manifest, adapters, fingerprinting, validation, duplicate candidates, preprocessing and protected downloader; no media accepted because release/rights/counts remain unresolved. |
| 2 — isolated recognition | Training/export hardening implemented; model acceptance blocked | Group-leakage gates, validation calibration, opt-in mirroring, metadata and ONNX parity check; no approved training data or weights. |
| 3 — continuous translation | Research baseline implemented; model acceptance blocked | Independent Transformer, training-only vocabulary, grouped trainer, multi-reference evaluation and offline inference; no rights-reviewed paired media. |
| 4 — backend integration | v1 routes and bounded media path implemented | Readiness and task routes, upload/decode, resource caps, cleanup and mock API/job tests pass in the full suite; real-model/video integration is pending artifacts. |
| 5 — frontend/evaluation | Separate workflows implemented | React v1 client and readiness/result semantics; `npm run build` passes; browser/model-success integration unavailable without model. |
| 6 — live isolated | Deliberately gated | Legacy WebSocket disabled by default and local-only on explicit opt-in; needs trained artifact, browser/Python parity and stability evidence. |
| 7 — live continuous | Deferred | No streaming route/model; requires a causal/chunk model and separate quality/latency gate. |
| 8 — security/testing/deployment | Foundations implemented; acceptance partial | Bounded uploads/jobs, safe errors, retention, Docker/CI config; compileall and all 175 Python tests pass on Python 3.11 (3.12 passed the earlier 94) (Linux; serving path verified with real synthetic video and a stub model, not a trained model); compose config passes; Docker daemon unavailable. |

## Work recorded this session

- Restored 75 source/project files from the user-provided `isl-engine.zip` into the empty checkout, excluding its nested `.git` folder; retained the original archive.
- Inspected source docs, models, dataset utilities, API, frontend, tests, Docker/CI, two distinct Colab pipeline notebook revisions, the starter INCLUDE notebook, and the action-detection notebook.
- Read the supplied Markdown specification and inspected the supplied SRS, previous-vs-new report, technical blueprint, architecture document and dataset roadmap.
- Researched current primary dataset cards/papers and official MediaPipe, ONNX Runtime and FastAPI documentation. Dataset access and terms remain separate from published counts.
- Created the five requested root architecture documents plus Phase 0 audit/progress documents.

## Commands actually executed

- `git status --short --branch`, `git ls-files`, static `rg`, PowerShell listings and source reads.
- Python standard-library scripts for ZIP/notebook/docx text inventory, file counts, hashes and static test-function count.
- `python --version`: 3.14.8; `uv` command located; `nvidia-smi`: RTX 2050, 4096 MiB; C: free space about 32.1 GB.
- Initial imported-package inventory: NumPy/Torch/ONNX Runtime/MediaPipe/OpenCV/FastAPI/Uvicorn/pytest/datasets/sacrebleu absent; project packages were later installed in the uv environment below.
- `uv sync --python 3.11 --group dev --extra experiments` completed with Python 3.11.9. Runtime imports passed for NumPy 2.4.6, PyTorch 2.15.0+cpu, OpenCV 4.14.0, MediaPipe 0.10.14 and FastAPI 0.162.2.
- `python -m compileall -q backend isl ml tests` passed.
- Focused tests passed: 7 dataset contract tests and 6 continuous/calibration/model-shape tests. The complete 44-test Python suite passed outside the sandbox in 109.04 seconds. FastAPI `TestClient` was separately confirmed to hang during Windows `asyncio` socket-pair creation inside the sandbox, including with a minimal FastAPI app.
- `npm run build` passed (TypeScript check and Vite production bundle).
- `docker compose config --quiet` passed. Docker image build/runtime was unavailable because Docker Desktop's Linux engine pipe was not running.
- `uv lock` regenerated the lockfile and `uv lock --check --python 3.11` passed. The default environment uses MediaPipe's single OpenCV distribution; scikit-learn is in the optional `experiments` extra.
- Python support was expanded to 3.11–3.12 and CI now covers both; local tests remain verified on 3.11. Python 3.12 Colab runtime/import/test smoke check is still outstanding.
- Added a dataset decision and full Colab/Drive workflow guide. The user selected a non-commercial research/demo path; INCLUDE-50 and gated iSign are candidates subject to their recorded data-access/storage gates.
- 10 Nov 2026 (Linux sandbox): merged the user's newer files (Colab guide, Python 3.12 metadata, CI matrix) over the ZIP; regenerated `uv.lock` for `>=3.11,<3.13` (only change: Emscripten-only `httpx2-jsfetch`); `uv sync --locked` + full suite pass on Python 3.11 and 3.12 (CPU only, no GPU); `npm ci` + `npm run build` pass.
- Same date: fetched the primary INCLUDE (HF card, Zenodo) and iSign (HF page + file tree) pages; results are in `docs/DATASET_INVENTORY.md`. iSign access-agreement text was not readable without login.
- Defects found and fixed by tests: (1) `isl.download_include --keep` was case-sensitive, so a keep list in the metadata's capitalisation extracted 0 files yet marked the part done (and with `--delete-zips` deleted it); (2) the canonical INCLUDE adapter had no label/path consistency check; (3) no CLI produced a canonical INCLUDE manifest, so `fingerprint`/`validate`/`preprocess` had no input; (4) the free-space check required space for all archives even with `--delete-zips`. Added `ml/datasets/include_manifest.py`, `audit_include_records`, `required_free_bytes` and 25 tests (mutation-checked).
- Added `notebooks/colab_include50_isolated.ipynb`: syntax and every CLI flag statically verified against `--help`; **not executed on Colab**.
- 11 Nov 2026 (user's Colab): Step 6 metadata audit ran against the real Hub table; measured numbers and the 958 vs 943 reconciliation are in `docs/DATASET_INVENTORY.md`. Still no video downloaded, no training.
- 11 Nov 2026 (user's Colab): Step 7b listed 46 Zenodo archives, 56.8 GB total; smoke-test archive picked: `Electronics_2of2.zip`. Step 8 and Step 9 then crashed in `isl.download_include` with `TypeError: unsupported operand type(s) for +: 'int' and 'str'`: the original total line used `sum(s for *_, s in files)`, which takes the last tuple element (the checksum string) instead of the size, so **no download could ever start** (it also broke `--list`). It was in the original source and untested because `main` was not a function. Fixed, plus two related defects found while reading: (a) with `--delete-zips` a rerun re-downloaded already-finished archives (resume looked for the deleted zip); (b) extraction wrote each file at the archive's internal path, so an extra top-level folder or different capitalisation would have left Step 11 finding 0 videos. Files now land exactly at the keep-list/manifest path; the `.done` marker includes a keep-list digest. 9 tests, each fix mutation-checked.
- 11 Nov 2026, user's Colab Steps 7b-8 (screenshots): Step 8 downloaded `Electronics_2of2.zip` (0.82 GB), then printed `extracted 0 files` + "no archive member matched the --keep list" and `ALL FINISHED`. The archive had been deleted and marked done, so its contents could not be inspected. Cause NOT yet established: either that archive holds no INCLUDE-50 videos (my smoke-test picker chose it by category name, which cannot tell), or its internal folder layout differs from the metadata paths. Mitigations built: zero-match archives now print their member count and first names; extraction falls back to an unambiguous `label/file` match; `--inspect all` reads every archive's index with HTTP Range requests (no video download) and reports how many wanted videos match by 3 parts / 2 parts / file name; the flow runs it as a pre-flight and refuses to download if fewer than 95% of the wanted videos are found; smoke test is bounded (`--stop-after`, `--max-archives`).
- **Unexplained workspace changes (11 Nov, ~13:02 UTC):** while I was working, files in the project directory changed that I had not edited: new `isl/remote_zip.py`, `tests/test_remote_zip.py`, `tests/test_cache_paths.py`; modified `isl/download_include.py` (`--remote`, `--inspect`), `isl/train.py` (`resolve_cache_paths`), `tests/test_download_main.py` (one message string); plus an empty `.git` directory (removed by me). I do not know who made them. I reviewed all of it (careful, tested, no unsafe behaviour found: Range requests only to the Zenodo file URL, non-206 responses refused, length checks, zip CRC verification) and kept it; the full suite passes. The Range-based `--remote` mode has NOT been run against real Zenodo.
- Added `isl/flow.py` and a re-runnable 5-cell notebook: state on Drive, restore-instead-of-recompute, videos fetched only when landmarks are missing, download batches and landmark extraction synced to Drive as they progress, rights gate, pre-flight, failure stops with a clear message, results table with seed spread and the 95% test-set interval. 21 tests, 7 behaviours mutation-checked.
- No notebook execution, training, real-media decoding, dataset download, real-model API inference, Docker build or deployment has been run. API tests use fixtures/mocks and are not evidence of a trained model working.

## Verified counts and metrics

- Workspace source restored: 75 files from the source ZIP; 175 pytest cases passing: 44 original + 21 workflow-orchestration (`isl/flow.py`, simulated Drive/VMs + contract tests against the real downloader output) + 6 more downloader tests + the other contributor's 15 (remote zip, cache paths) + 9 downloader end-to-end (fake Zenodo record/archives) + 3 Hub-loader failure-handling tests + 25 serving-contract (`tests/test_v1_media_e2e.py`, synthetic video + stub model) + 13 INCLUDE downloader/adapter hardening + 12 INCLUDE manifest CLI (synthetic metadata) + 9 augmentation (synthetic landmarks) + 11 preprocessing resume/failure handling (tiny real MP4s, stand-in extractor) + 6 headless-matplotlib regression tests (fresh-interpreter).
- **First real Colab run (user, 11 Nov 2026):** CPU runtime, Colab Python 3.13.18 (uv supplied project Python 3.12), 2 cores, 115.6 GB free. `uv sync --locked` succeeded; the post-install check and 1 of 114 tests failed with `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value` - Colab exports MPLBACKEND for its notebook backend, which is absent from the project environment, and MediaPipe imports matplotlib. Reproduced locally and fixed in `isl/__init__.py`; after setting `MPLBACKEND=Agg` the user's re-run showed the stack importing (python 3.12.3, torch 2.15.0+cu132, cuda False, opencv 4.14.0, mediapipe 0.10.14) and 114/114 tests passing on Colab; would also have broken landmark extraction (Step 13).
- Local ISL videos/checkpoints/labels/model packages: 0.
- No training run or measured model metrics in this session.
- External dataset counts remain reported/listed only; exact available media are not known.

## Next concrete work

0. User actions that unblock data: (a) read the INCLUDE terms yourself and set `RIGHTS_REVIEWED = True` in the notebook; (b) run notebook cells 1-6 on a Colab GPU runtime and paste back the runtime check and `include_audit.json`; (c) for iSign, log in, read the access agreement and send its exact text.
1. Review the React workflow in a browser and run a clean Docker build/runtime check when Docker Desktop is available.
2. Reconcile a permitted dataset release and terms before acquiring media; the current INCLUDE archive estimate exceeds local free space.
3. If permitted media and Colab are available, run isolated and continuous baselines, record group-aware evaluation, inspect failures and promote only measured artifacts.

## Blockers requiring external state

- INCLUDE media source archive is not local; reported archive size is far larger than free C: space. Current exact release/rights/count needs reconciliation.
- iSign access requires acceptance of current gated research/non-commercial terms; no authenticated account action was performed.
- ISLTranslate media/model derivative terms must be confirmed.
- No Colab runtime, GPU training result or permitted real video fixture is attached.
