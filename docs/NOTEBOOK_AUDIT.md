# Notebook Audit

**Audit date:** 10 November 2026  
**Method:** inspect notebook JSON, cell sources, execution counts and stored outputs. No notebook was run or restarted in this task.

## ISL_INCLUDE_Project_Setup.ipynb

- Downloads contains two byte-identical copies (`ISL_INCLUDE_Project_Setup.ipynb` and `ISL_INCLUDE_Project_Setup (1).ipynb`), each 45 cells and SHA-256 `567334197b311558be35bd03c58a357b629691fa0ca9035a9ee6924039bb4800`.
- Purpose: early Colab workflow to unpack the full INCLUDE archives in Drive, extract MediaPipe Holistic features and train an isolated classifier.
- Paths: hard-coded `/content/drive/MyDrive/ISL_Project/...` and one literal `/content/drive/.../Adjectives/23. high/MVI_9654.MOV` path.
- Feature/model contract: legacy Holistic output (33 pose × 4, 468 face × 3, both hands × 21 × 3 = 1,662 values/frame); clips are fixed to 30 frames by taking the first 30 or zero-padding. The model is a 2-layer BiLSTM with 128 units/layer and 40 outputs in the shown training block.
- Data order: unpacks source archives to Drive, later chooses 40 classes randomly, saves processed `.npy`, then uses `train_test_split(... test_size=0.2, stratify=y)`. This random video split does not keep signers/sessions together. It computes mean/std from train but uses the test set as `validation_data` for early stopping, then reports test accuracy on that same split.
- Reproducibility: package installs occur in multiple cells and are not pinned consistently. The notebook has saved outputs and execution counts, but these do not prove a clean-kernel top-to-bottom run. It writes an H5 model and relies on notebook-global variables for prediction.
- Decision: historical exploration only. Do not use its weights/accuracy or 1,662-D preprocessing as an approved v2 baseline. Reuse only small general patterns after modularization.

## Action_Detection_Refined_Tasks_Compatible.ipynb

- Downloads file has 65 cells (1,278,833 bytes; SHA-256 `29fe165e9fcfa5dda34eca4d05944c661b9cce41a2a0c6a782d97b51086539cd`). It contains executed-cell counts and stored output blocks from an earlier session; it was not run in this task.
- Purpose: webcam action/gesture tutorial with example actions `hello`, `thanks`, `iloveyou`; collects around 30 sequences per action, each 30 frames.
- This is not a verified INCLUDE or continuous ISL-to-English dataset. Its split is an unseeded 5% row-level `train_test_split`, not signer/session grouped; the 2,000-epoch fit has no validation data or early stopping.
- The notebook is internally inconsistent: an early live loop calls a model on a 30-frame buffer with a legacy Holistic workflow, while a later Tasks extractor is introduced. The feature examples include 258 values and 1,662-value Holistic outputs, while the displayed LSTM input is `(30, 258)`. Later data-collection cells still call `mp_holistic.Holistic` although the notebook describes Tasks APIs. Runtime state includes a missing/undefined `extract_keypoints` in some flow paths.
- Decision: do not reuse its model, weights, labels or reported metrics. Camera capture/drawing ideas may be consulted as UI patterns only; use the maintained `isl/` pipeline instead.

## Colab pipeline notebooks: two different revisions

### Project-archive revision

- `notebooks/colab_pipeline.ipynb` restored from `isl-engine.zip` has 15 cells, no execution counts and no outputs. It is not byte-identical to the Downloads notebook: SHA-256 `e8da77a845f8df35c93b320f626c13e69ab286ab393aff2f2a935e9cd6e09832`.
- Presents a resumable v2 staged pipeline. It mounts Drive, unzips `isl-engine.zip`, installs unpinned packages, writes a copy of `isl/download_include.py` from notebook text, downloads INCLUDE-50 into Drive, expects custom idle clips, extracts landmarks/cache to Drive, trains and exports ONNX.
- Code overwrites a repository module in the Colab working copy. It stages large downloads/features in Drive, has no execution evidence and depends on external Drive files and dataset access.

### Downloads revision

- `C:\Users\TARNDEEP SINGH\Downloads\colab_pipeline.ipynb` has 21 cells, no execution counts and no training/model outputs; SHA-256 `4eb1563f09fd33fcd98e9b27a2208dde3e254ce6b5a7261eeee459427f9c8c59`.
- It unzips the project into `/content`, installs unpinned `mediapipe`, `datasets`, `onnx`, `onnxruntime`, builds INCLUDE-50 keep paths from HF metadata, deletes non-matching video files, and cell-writes `isl/download_include.py` over the repository version.
- It downloads/extracts to Colab local storage, copies extracted files/cache to Drive, builds the manifest from the local data, trains, exports ONNX and downloads two model files. A cache restore cell rewrites cache paths by basename only.
- As written, data/source state can diverge from the ZIP; checksum verification is absent in the notebook, dataset terms are not recorded, training/test use depends on unverified manifest split behavior, and no ONNX parity step is shown.

### Missing notebook mentioned in reports/spec

The previous-vs-new report references `colab_pipeline_v2.ipynb`; no file with that name is present in Downloads or the workspace. The two pipeline revisions above are not equivalent. Do not call either executed or resumable in practice without a clean Colab run.

## Notebook migration decision

Keep a single thin `notebooks/colab_pipeline.ipynb` that calls package CLIs, uses pinned environment files and never writes source files. Stage active data/cache in `/content`, save progress/checkpoints/reports to Drive, validate checksums, and log actual runtime/GPU/version/count information. The notebook should stop on rights/data-integrity failures instead of printing expected counts. Model training and results stay task-specific.
