# Historical implementation-plan snapshot (from restored archive)

> **Status warning, 10 November 2026:** This file is preserved as evidence from the supplied source archive. Its phase/status tables and commands predate the current audit and are not verified for this checkout. The current controlling plan is the top-level [IMPLEMENTATION_ROADMAP.md](../IMPLEMENTATION_ROADMAP.md); current phase evidence is [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md). No dataset/model/test result is accepted solely because this historical file says it was built or tested.

---

## Archived plan content

## Scope decision, with evidence
| Module | Decision | Evidence |
|---|---|---|
| A. Isolated words | **Ship, as the reliable core.** Vocabulary = whatever the trained model covers. | INCLUDE: 263 words, 4,287 videos, 16.3 videos/word, **7 signers** (iSign paper, Table 1). Real, supervised labels exist. |
| B. Continuous sentences | **Research pipeline, labelled experimental. Do not present as working translation.** | ISLTranslate (31,222 pairs): baseline BLEU-4 6.09, ROUGE-L 12.91. iSign (118,228 pairs): best published BLEU-4 8.58 (SpaMo-OF, WSLP 2025). Outputs at this level are mostly wrong English. |
| C. Automated testing | **Ship.** Held-out dataset videos, metrics, reports, UI. | Needs no signing by the user. Caveat below. |

Isolated-word labels cannot supervise sentence translation. CISLR has 4,765 words but only 1.5 videos/word (71 signers): useful for signer diversity, not for a 263-way classifier by itself.

**Caveat on Module C.** Official INCLUDE test videos come from the same few signers as training. A high score there does NOT show the system works for a new person. Use `--split-mode session` for model selection and report both.

## Status
| Phase | Item | Status |
|---|---|---|
| 0 | Audit tool `ml/audit/inspect_project.py` (read-only; writes CURRENT_STATE_AUDIT.md, DATASET_INVENTORY.md) | built + tested; **must be run on your PC** |
| 1 | Unified manifest, session inference, group split, leakage checker (`ml/datasets`) | built + tested |
| 1 | Augmentation experiments, frame extraction, signer metadata | NOT built |
| 2 | Word recognition: legacy BiLSTM (`isl/`), evaluation (`ml/evaluation`), test library, API, UI | built + tested (no trained model in this repo) |
| 2 | Baselines B (Transformer) and C (RGB encoder) | NOT built |
| 3 | Sentence translation (D/E/F) | NOT built; blocked on verified data access + GPU |
| 4 | Upload -> job -> result -> SRT/VTT/TXT (word results only) | built + tested |
| 5 | React+TS UI: Test library, Upload, Evaluation, Model | built (4 of 12 screens) |
| 6 | Live streaming | legacy page at `/live/` (works with the legacy model) |
| 7 | Postgres, Redis, RAG, agents, CI additions, monitoring | NOT built (SQLite job store used in dev) |

## Run order
1. `python -m ml.audit.inspect_project --project . --datasets <INCLUDE folder>` -> read `docs/`.
2. `python -m ml.datasets.build_manifest --root <INCLUDE folder> --keep data/keep.txt --official-from-hf --split-mode official --out data/manifest_official.csv` (read the LEAKAGE REPORT), then again with `--split-mode session --out data/unified_manifest.csv`.
3. Train on the `train` rows (Colab), validate on `val`, and touch `test` once.
4. `python -m ml.evaluation.word_eval --manifest data/unified_manifest.csv --split test --model models/isl.onnx --out reports/word_eval`
5. `python -m ml.test_library.build --manifest data/unified_manifest.csv --out test_videos --per-class 2`
6. `cd web && npm install && npm run build`, then `uvicorn backend.main:app` -> http://localhost:8000

## Word-first recovery plan (v2.1) — uses ONLY data you already have
Why: no ISL word dataset offers many videos per word AND many signers (INCLUDE: ~16/word, 7 signers; CISLR: ~1.5/word, 71 signers). So we (1) measure honestly, (2) keep only words that are learnable, (3) get negatives from the dataset itself.

| Step | Command | Where | Output |
|---|---|---|---|
| 1 | `python -m ml.datasets.build_manifest --root /content/drive/MyDrive/INCLUDE50 --keep data/keep.txt --official-from-hf --split-mode session --out data/unified_manifest.csv` | Colab | manifest + LEAKAGE REPORT. Paths match the old cache, so no re-extraction. |
| 2 | `python -m ml.experiments.cv_word --manifest data/unified_manifest.csv --cache /content/drive/MyDrive/isl_cache --folds 5 --out reports/cv` | Colab or PC (CPU, minutes) | random-vs-session CV, leakage gap, `reliable_words.json` |
| 3 | `python -m ml.datasets.apply_vocab --manifest data/unified_manifest.csv --vocab reports/cv/reliable_words.json --out data/vocab_manifest.csv` | any | unreliable words -> `unknown_sign` |
| 4 | rebuild `index.csv` for the new labels, then `python -m ml.datasets.make_idle_cache --index cache/index.csv --out-index cache/index_idle.csv` | Colab | idle examples from the dataset's own rest frames (same signers/camera) |
| 5 | `python -m isl.train --index cache/index_idle.csv --out runs/v2` -> `isl.export_onnx` | Colab GPU | model |
| 6 | `python -m ml.evaluation.word_eval ...` and `ml.test_library.build` | PC | reports + test library |

Decision rule for success (stated before training): report SESSION-grouped CV mean +/- std. A word is "supported" only if grouped recall >= 0.7 over >= 6 test videos.
