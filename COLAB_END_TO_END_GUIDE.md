# ISL Translation Engine 2.0 — Colab and Google Drive Guide

This is the working path from dataset choice to training, local integration, GitHub, and deployment. It keeps isolated recognition and continuous translation as separate experiments and product features.

## 1. Decide the data-use path before downloading

| Dataset | Best fit | Recommendation and gate |
|---|---|---|
| INCLUDE-50 | Isolated sign classification | Best first benchmark for the isolated pipeline. The INCLUDE card describes isolated recognition and links metadata to the actual Zenodo videos. Reconcile the current metadata, official splits, and exact video archive before downloading; the repository audit found count differences and estimated the full archive at about 56.8 GB. Confirm the video release terms and your storage quota. |
| iSign v1.1 | Continuous sign video/pose to English; also has word-level tasks | Best research-only continuous baseline candidate because it provides sign-to-text tasks plus video/pose and translation data. It is gated and marked CC BY-NC-SA 4.0 (public page: free for research, not commercial); the access-agreement text, which the earlier version of this guide paraphrased as a re-upload ban, is only visible after login and has not been read here, so treat it as unverified. Its card says to group rows by parent `video_id`. Size: 228 GB total (videos 57.9 GB, poses 170.2 GB) in concatenable parts, which cannot be staged whole on Colab; plan a subset. Do not copy iSign media or pose files into Google Drive or GitHub. Use it only after accepting the terms and confirming that your intended processing location is permitted. |
| ISLTranslate | Continuous sign video to English | Useful alternative: its paper reports 31,222 sentence/phrase pairs. The current repository describes the source-video and derivative-model terms as unresolved, so get explicit permission/terms before training or putting videos, poses, or checkpoints in Drive. |
| ISH-NEWS (2026 paper) | News-domain continuous translation research | Not the default product dataset. The paper says raw video cannot be redistributed, the sample is from a few trained broadcast signers, and the article is CC BY-NC-ND. Obtain direct video/model permission before training or deploying. |

**My recommendation for a non-commercial research prototype:** start isolated recognition with an audited INCLUDE-50 subset; start continuous translation with iSign only under its accepted research terms, keeping restricted data out of Drive/GitHub. Keep those models, labels, vocabularies, splits, and results separate.

**For a public or commercial product:** first obtain written rights for the source video and derivative model, or collect/commission consented recordings with an explicit training and deployment release. Public availability alone is not permission. Do not publish restricted checkpoints just because the application code is public.

## 2. Colab runtime and project setup

Colab runtimes are temporary, and available hardware/usage limits vary. The current default runtime is Python 3.13; the Colab runtime-version page currently lists the 2025.10 runtime with Python 3.12.12. The project was tested on Python 3.11 and now declares Python 3.11–3.12 support; verify the smoke checks below before a long run. If the 2025.10 runtime is not offered in your account, stop and choose a runtime compatible with the project instead of installing packages blindly.

1. Open Colab and create a new notebook named `ISL_Translation_Engine_2_0.ipynb`.
2. Choose **Runtime → Change runtime type**. Select a GPU accelerator. If available, select runtime version **2025.10** (Python 3.12.12). Restart the runtime if Colab prompts you.
3. Upload the project ZIP to a private Drive folder or clone the repository after it has been published. The ZIP is source-only; it contains the architecture docs, scripts, tests, handoff notes, and supplied Markdown specification, but no model or dataset.
4. Mount Drive and create private working folders:

```python
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive/ISL_Translation_Engine")
DRIVE.mkdir(parents=True, exist_ok=True)
(DRIVE / "manifests").mkdir(exist_ok=True)
(DRIVE / "runs").mkdir(exist_ok=True)
(DRIVE / "checkpoints").mkdir(exist_ok=True)
```

Only put dataset files in Drive if that dataset's current terms allow this storage. Keep Drive sharing private. Never put secrets, HF tokens, or restricted dataset media in the GitHub repository.

5. Extract code to Colab's temporary disk (`/content`), not the mounted Drive. If your ZIP is at `MyDrive/ISL_Translation_Engine/ISL_Translation_Engine_2.0_for_Claude.zip`:

```python
import shutil, zipfile
archive = DRIVE / "ISL_Translation_Engine_2.0_for_Claude.zip"
shutil.copy2(archive, "/content/project.zip")
with zipfile.ZipFile("/content/project.zip") as zf:
    zf.extractall("/content")
WORK = Path("/content/ISL")
```

6. Install the locked environment and check the runtime. If Python is not 3.11 or 3.12, stop here.

```python
%cd /content/ISL
!pip -q install uv
!uv sync --python 3.12 --group dev --extra experiments --locked
!python --version
!nvidia-smi
```

Then run the checks:

```python
!uv run python -c "import torch, cv2, mediapipe, fastapi; print(torch.__version__, torch.cuda.is_available(), cv2.__version__, mediapipe.__version__, fastapi.__version__)"
!uv run python -m pytest -q tests
```

If `torch.cuda.is_available()` is false under a GPU runtime, stop and fix the PyTorch/CUDA environment before training. Do not replace Torch or MediaPipe with arbitrary latest versions mid-run; record the selected versions in the run report.

## 3. Store data safely and avoid slow training I/O

Use Google Drive for permitted source archives, manifests, experiment reports, and checkpoints. Use `/content` for active extraction, decoded clips, landmark caches, and training. Drive-mounted files can be slower and Colab may disconnect; Colab's own guidance recommends staging large archives locally and using Drive for durable storage.

Suggested layout:

```text
MyDrive/ISL_Translation_Engine/
  source_archives/          # only if the license permits Drive storage
  manifests/                # canonical CSV plus validation reports
  runs/                     # configs, metrics, error examples, model cards
  checkpoints/              # best/last checkpoints and exports if licensing permits

/content/isl-data/          # temporary extracted media and feature caches
/content/ISL/               # repository checkout
```

For iSign, do not use `source_archives/` for the restricted dataset. The iSign access card says not to re-upload the data elsewhere. If you are unsure whether Colab processing or a Drive checkpoint is allowed, ask the dataset maintainers before proceeding.

## 4. Ingest and validate a permitted dataset

Do a small authorized subset first. Do not download the complete INCLUDE archive until exact disk/storage size, source counts, and use terms are checked. The current project includes an INCLUDE downloader with checksum, free-space, and safe-extraction checks; it requires an explicit rights-review acknowledgment and should only be used after you have actually reviewed the release terms.

For INCLUDE, `python -m ml.datasets.include_manifest` builds the canonical manifest, the `--keep` list and an audit report (see `notebooks/colab_include50_isolated.ipynb`, cells 6–10); it never approves rights unless you pass `--rights-status`. The older `ml.datasets.build_manifest` writes a different (legacy) schema that `fingerprint`/`validate`/`preprocess` do not accept. For each task-specific source, a canonical manifest has dataset/version, sample ID, parent/source-video ID, split, label or English references, relative media path, SHA-256, license/rights state, and group ID. Preserve original text and labels. Never make random row-level splits for continuous segments: all segments from one source video must stay in one split. Preserve official splits where supplied and record any new group split policy.

After the manifest is ready, run:

```bash
uv run python -m ml.datasets.fingerprint \
  --manifest /content/isl-data/manifests/source.csv \
  --data-root /content/isl-data/raw \
  --out /content/isl-data/manifests/source-hashed.csv

uv run python -m ml.datasets.validate \
  --manifest /content/isl-data/manifests/source-hashed.csv \
  --data-root /content/isl-data/raw \
  --verify-hashes --decode-media --require-approved-rights \
  --require-splits --require-group-ids \
  --out /content/isl-data/reports/validation.json
```

Read the entire validation report. Fix failures or record deliberate exclusions; do not train when license, parent-video grouping, media integrity, or split checks fail. Run the near-duplicate candidate report and review candidates manually; never drop them automatically.

## 5. Preprocess and train isolated recognition

1. Run the versioned MediaPipe preprocessing pipeline on the authorized isolated subset:

```bash
uv run python -m ml.datasets.preprocess \
  --manifest /content/isl-data/manifests/isolated-hashed.csv \
  --data-root /content/isl-data/raw --task isolated \
  --out /content/isl-data/processed/isolated-v1
```

2. Confirm that `index.csv` exists and has non-empty train, validation, and test splits with no group or exact-hash leakage.
3. Train and save checkpoints in Drive so they survive Colab disconnects:

```bash
uv run python -m isl.train \
  --index /content/isl-data/processed/isolated-v1/index.csv \
  --out /content/drive/MyDrive/ISL_Translation_Engine/runs/isolated/include50-run-001 \
  --epochs 150 --bs 32 --seed 13
```

4. Inspect per-class metrics, validation calibration/coverage, errors, and training metadata. Do not treat validation results as test results. Export only after review:

```bash
uv run python -m isl.export_onnx \
  --ckpt /content/drive/MyDrive/ISL_Translation_Engine/runs/isolated/include50-run-001/best.pt \
  --out /content/drive/MyDrive/ISL_Translation_Engine/checkpoints/isl.onnx
```

The export checks PyTorch/ONNX parity and binds calibration to the checkpoint, labels, and ONNX file. Keep the original dataset split and its provenance with the model card.

## 6. Preprocess and train continuous translation

Use a separate, rights-approved paired video-to-English manifest. Do not reuse isolated labels or concatenate classifier outputs into sentences. Convert the selected source's video/pose records into the repository's canonical continuous schema; the source-specific iSign adapter and pose conversion may need to be completed after the exact release files are inspected.

```bash
uv run python -m ml.datasets.preprocess \
  --manifest /content/isl-data/manifests/continuous-hashed.csv \
  --data-root /content/isl-data/raw --task continuous \
  --out /content/isl-data/processed/continuous-v1

uv run python -m ml.continuous.train \
  --index /content/isl-data/processed/continuous-v1/index.csv \
  --out /content/drive/MyDrive/ISL_Translation_Engine/runs/continuous/isign-run-001 \
  --epochs 40 --batch-size 4 --seed 13
```

The current baseline is pose/hand-only. It may miss facial expression and other non-manual grammar; it is a research baseline, not a finished translator. Review BLEU/chrF, WER/CER, ROUGE-L, generated samples, source-video split integrity, and human ISL review before considering it for users. If it fails to learn on a small authorized subset, diagnose the input/label alignment rather than scaling training blindly.

## 7. Save, resume, and record every run

For each run, save the source manifest and SHA-256, validation report, preprocessing config, split counts/groups, Git commit, runtime/Python/Torch/CUDA versions, GPU name, seed, command, checkpoint, metrics, example errors, and model card under the run folder. Use a unique run name; never overwrite a prior accepted model. Keep active media/cache in `/content`; save a compressed cache archive to Drive only when permitted and useful. Copy a checkpoint after each epoch or use the current trainer's `last.pt`/`best.pt` output path on Drive so a disconnect does not erase the only copy.

## 8. Integrate locally, publish, then deploy

1. Download only rights-permitted model artifacts and reports back to the local repository's ignored `models/` or `runs/` directories. Keep all datasets, raw videos, access tokens, `.env`, and private checkpoints out of GitHub.
2. Start the FastAPI backend and React frontend locally. Confirm `/api/v1/readiness` reports each model independently and that the correct model package is loaded. Test a small authorized video end-to-end, then test corrupt, unsupported, and oversized inputs.
3. Build the container and verify health/readiness with real model files before choosing a host. Continuous inference may need GPU service; benchmark CPU first on representative clips. Keep live translation disabled until chunk quality and latency have been measured.
4. Publish source code and documentation to GitHub only after checking the full Git diff, secrets, licenses, and repository visibility. Store data/model artifacts outside Git unless the exact terms explicitly allow redistribution.
5. Deploy only after deciding whether the service is educational/non-commercial or commercial, confirming model/dataset rights for that purpose, and measuring hosting costs. Set environment variables through the host's secret manager, never in source code.

## Current gates

- No dataset media or model artifacts are in this repository yet.
- The 103-test Python suite passes on Python 3.11 and 3.12 (locked, CPU-only Linux) and the frontend build passes. The Colab GPU runtime check has not been run; the notebook's cells 1–5 do it.
- The full INCLUDE media release needs exact count/rights/storage reconciliation. iSign requires acceptance of its gated non-commercial terms and should not be re-uploaded. ISLTranslate's release/model rights still need confirmation.
- No real-model training, real-video inference, Docker image run, GitHub publication, or deployment has been completed.

## Primary references checked

- [INCLUDE dataset card](https://huggingface.co/datasets/ai4bharat/INCLUDE) and [INCLUDE Zenodo record](https://zenodo.org/records/4010759)
- [iSign dataset card and access terms](https://huggingface.co/datasets/Exploration-Lab/iSign) and [iSign project page](https://exploration-lab.github.io/iSign/)
- [ISLTranslate ACL paper](https://aclanthology.org/2023.findings-acl.665/) and [official dataset repository](https://github.com/exploration-lab/isltranslate)
- [ISH-NEWS 2026 paper and data availability statement](https://www.nature.com/articles/s41598-026-60893-0)
- [Colab runtime-version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) and [Colab FAQ](https://research.google.com/colaboratory/faq.html)
