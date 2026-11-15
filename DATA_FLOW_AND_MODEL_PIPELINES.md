# Data Flow and Model Pipelines

**Status:** Architecture baseline plus implementation update, 10 November 2026. No dataset or trained model is present. Pipeline code scaffolds exist; commands and metric values remain targets until executed and logged.

## 1. Dataset acquisition and provenance flow

~~~mermaid
flowchart TD
  SRC[Official dataset or consented collection] --> TERMS[Access, license, consent and task review]
  TERMS -->|approved for intended use| ADAPTER[Source-specific adapter]
  TERMS -->|unclear or incompatible| HOLD[Register only; do not ingest]
  ADAPTER --> RAW[Immutable raw media plus source metadata]
  RAW --> HASH[SHA-256 and stable sample identity]
  HASH --> MEDIA[Container, stream, duration and bounded decode validation]
  HASH --> ANN[Label/reference and pairing validation]
  MEDIA --> DUP[Exact and near-duplicate review]
  ANN --> DUP
  DUP --> GROUP[Signer/source-video/session group assignment]
  GROUP --> SPLIT[Official or documented grouped train/validation/test split]
  SPLIT --> MAN[Versioned canonical manifest]
  MAN --> PRE[Task-specific preprocessing and quality reports]
  PRE --> CACHE[Content-addressed feature or frame cache]
  CACHE --> TRAIN[Task-specific training]
  TRAIN --> EVAL[Validation/model selection and one final test]
  EVAL --> PKG[Model package plus model card and checksums]
~~~

### Acquisition register

Maintain one row per dataset release with: dataset ID/name, official source URL, paper, exact release/commit/version, download/authentication steps, archive size, local location, task, media and annotation fields, signer/source-video IDs, official splits, declared license, additional restrictions, source checksum, current access state and actual local counts. Count states independently: **reported** in a paper, **listed** by a dataset card, **downloaded**, **decoded**, **annotation-valid**, **rights-reviewed**, and **accepted for a model**.

Current source findings:

| Source | Task / possible use | Present evidence | Gate before use |
|---|---|---|---|
| AI4Bharat INCLUDE | Isolated signs; current 170-feature path baseline | The Hugging Face card is metadata with video paths and an INCLUDE-50 flag, marks CC-BY-4.0, and points to Zenodo for video files. The current card lists 5,250 metadata rows across train/val/test (3,816/425/1,009); older supplied reports cite 4,287/4,292 full videos and 958 INCLUDE-50. These counts are unreconciled. | Read current Zenodo record terms, reconcile row IDs/archive members and subset count, verify checksums, labels and split files, then decode a sample. Do not equate metadata rows with usable videos. |
| iSign v1.1 | Continuous video-to-text and pose-to-text research | Paper reports 118,228 pairs and separate video/pose tasks. Current card states research-only/non-commercial gated access; translation rows use ordered IDs containing source video ID and recommend keeping every row for a video in one split. | Account holder accepts current terms; download only permitted parts; audit video/pose alignment, official split, duplicates and source provenance. No commercial/production use implied. |
| ISLTranslate | Continuous video-to-English research | Paper reports 31,222 pairs and an educational domain; paper's baseline is pose-to-text. | Confirm current author distribution, original source terms, redistribution and derived-model rights, pair alignment, source-video grouping and annotation quality. The paper is not a substitute for the release license. |
| CISLR | Isolated vocabulary breadth or representation research | iSign paper reports 4,765 glosses, 7,050 videos and 71 signers; few examples per gloss make it unsuitable as a standalone high-coverage classifier. | Read current source license/access conditions; preserve its label ontology. Do not merge labels by string or assume cross-corpus equivalence. |
| FDMSE-ISL / ISH-News / other public video | Future research only | Supplied documents list them, but no accepted release, license or local data exists here. | Do not scrape, train or distribute until official terms, consent, labels and feasibility are established. |

References: [INCLUDE card](https://huggingface.co/datasets/ai4bharat/INCLUDE), [iSign paper](https://arxiv.org/abs/2407.05404), [iSign card](https://huggingface.co/datasets/Exploration-Lab/iSign), [ISLTranslate paper](https://aclanthology.org/2023.findings-acl.665/).

### Canonical manifest

The manifest is metadata, not a copy of media. Store at least:

| Field | Meaning |
|---|---|
| sample_id | Stable internal identifier, independent of local path |
| dataset_id / dataset_version | Source and exact release |
| source_sample_id / source_video_id | Original item ID and parent recording ID |
| task | isolated or continuous; never infer from directory name alone |
| media_uri / sha256 / size_bytes | Relative or private URI, exact content digest, bytes |
| label_id / label_text | Isolated label, preserved as source value |
| reference_texts | One or more unchanged translation references |
| normalized_references | Metric-only text under versioned normalization |
| signer_id / session_id / group_id | Original metadata only; group provenance and whether observed or inferred |
| official_split / project_split / split_policy | Original and experiment assignment, separately retained |
| license_id / rights_status | Specific terms and review state |
| media_status / annotation_status / exclusions | Validation state and auditable failure reason |
| preprocess_version / feature_uri | Cache contract and optional derived representation location |
| parent_sample_id / augmentation_recipe | Derivative lineage, if augmented |

Use CSV/Parquet for large manifests and small JSON schemas/configs in Git. Do not store absolute workstation paths as the only reference. Hash raw files with SHA-256. Use perceptual frame hashes as **review candidates**; do not automatically delete near-duplicates. Check repeated source IDs and identical hashes across splits before training.

## 2. Media validation, split and preprocessing

### Validation sequence

1. Check file existence, byte limit and safe generated storage name. Do not trust extension, MIME type or supplied path.
2. Probe actual video stream/container and validate allowed codecs, duration, dimensions, frame rate and frame count against configuration.
3. Decode within resource/time limits. Reject unreadable, empty, truncated or overlong input with a stable error code and a reason stored in the audit report.
4. Validate annotation: isolated label belongs to a source-specific label map; continuous item has a non-empty paired reference and valid temporal range if provided. Keep multiple references.
5. Compute SHA-256, repeated-ID checks, exact duplicates and near-duplicate review candidates.
6. Assign groups before any augmentation. A source video with several sentence rows stays in one split. A signer-level split requires real signer metadata. Session inference from filename sequences is only a proxy and must be marked as such.
7. Emit deterministic manifests and reports: counts, class balance, signer/session/group coverage, duration/resolution/FPS, corrupt/missing items, text lengths, near-duplicate candidates, split overlaps, exclusions and rights state.

### Task-specific preprocessing contract

Every model package captures the complete settings: decoder/library version, accepted codec/container, sampling timestamps or sampling rate, resize/crop, color order, normalization, max frames/duration, variable-length mask/padding, landmark model/version, landmark order/dimensions, visibility/confidence policy, missing-point representation, coordinate frame, hand-side convention, temporal alignment, tokenizer/text normalization, and feature schema version.

Do not pad missing landmarks with an ordinary valid position. Preserve missingness and confidence explicitly. For existing cache/model compatibility, keep the historical 170-feature schema under a named v1 version; do not retrofit new channels into it. A new schema gets new cache keys and new weights.

Augmentation occurs after split assignment and only within train. Record source sample, operation, parameters, seed and code version. Bounded temporal crop/speed, small scale/translation and realistic compression/brightness may be tested. Horizontal mirror is disabled by default because it can change handedness, direction and sign meaning; any use requires a task-specific preservation argument and held-out evaluation.

## 3. Isolated recognition pipeline

~~~mermaid
flowchart LR
  V[One-sign video] --> VAL[Media validation]
  VAL --> DEC[Deterministic frame sampler]
  DEC --> LM[MediaPipe Tasks or validated RGB extractor]
  LM --> FEAT[Versioned pose and hand features plus quality mask]
  FEAT --> SEG[Whole-sign boundary or validated clip bounds]
  SEG --> FIX[Temporal resample plus mask]
  FIX --> CLS[Approved isolated model]
  CLS --> CAL[Validation-fitted calibration and reject policy]
  CAL --> RES[Label or uncertain/unusable result]
  RES --> EVAL[Top-k, macro/per-class, calibration, risk-coverage, latency]
~~~

### Training and selection

- Establish class counts and label map from the manifest. Keep official benchmark evaluation separate from grouped generalization experiments.
- Train a majority/frequency sanity baseline first. Run the previous model only if checkpoint dimensions, label ordering, preprocessing and split can be independently reconstructed.
- Compare one compact temporal baseline with the existing Conv1D–BiLSTM–attention candidate. Add RGB/fusion only for a specific handshape/non-manual hypothesis and only under the same split and comparable compute.
- Save best validation and last checkpoints; seed Python/NumPy/PyTorch; record config, manifest hash, split, preprocessing version, environment, GPU and elapsed time. Never select by final test.
- Report top-1/top-5, per-class and macro P/R/F1, confusion matrix, latency, model size, calibration error, selective risk/coverage and failure/exclusion count. Include signer/session breakdown when true metadata supports it.
- Choose a reject policy from validation data. Any confidence shown in the UI is calibrated on validation and named as confidence only if evidence supports that interpretation. Otherwise show a score or omit it.

### Inference artifact

Store a versioned package under an immutable path, e.g. **artifacts/isolated/<version>/**, containing model weights, architecture/config, labels, preprocessing JSON, model-card metadata, source/data/split hashes, evaluation report, runtime requirements and checksums. Training may keep optimizer checkpoints in a private run directory. Do not place untrusted pickle weights in the public model registry. Export to ONNX only after output and metric parity checks.

## 4. Continuous video-to-English pipeline

~~~mermaid
flowchart LR
  CV[Continuous signing video] --> VAL[Media and rights validation]
  VAL --> GROUP[Preserve source-video group and split]
  GROUP --> FR[RGB frame stream]
  GROUP --> POSE[Verified pose stream, if available]
  FR --> ENC[RGB visual encoder]
  POSE --> PENC[Pose encoder]
  ENC --> TEMP[Temporal encoder with masks]
  PENC --> TEMP
  TEMP --> DEC[English subword autoregressive decoder]
  DEC --> TXT[Generated English sequence]
  TXT --> MET[BLEU, chrF, ROUGE-L, WER/CER and semantic review]
~~~

Start with two controlled candidates rather than one assumed winner:

1. **Pose-to-text baseline:** use the corpus's supplied pose only after its skeleton schema, normalization and licensing are verified. This keeps compute modest and aligns with published ISLTranslate/iSign tasks, but may omit important visual details.
2. **RGB-to-text baseline:** deterministic sampled frames through a licensed pretrained visual encoder, temporal transformer and English decoder. This preserves appearance/handshape/facial evidence but costs more GPU/VRAM and can overfit if data/domain are limited.

Fusion is a later experiment only after each modality has an independent baseline. Do not mix data sources unless labels, source-video overlap, caption style, split policy and licenses are reconciled. Keep the original reference, normalized reference, tokenizer/library versions and decoding settings in every evaluation record. Use autoregressive generation for evaluation; teacher-forced loss alone is not translation quality. Report corpus-level and sentence-level metrics, examples, failures and data counts. BLEU is wording-sensitive; automatic scores require qualified semantic review where available. Keep names, negation, person, number, time, omissions and hallucinations in error analysis.

Do not assume gloss labels exist. A separate gloss-recognition head or CTC loss is valid only if a corpus provides genuine frame/gloss alignment and the experiment has its own task definition. No automatic language-model post-editing is allowed in the baseline because it can turn weak visual evidence into fluent false content.

## 5. Model governance and lifecycle

Each run records experiment ID/time, Git commit, source manifest and hash, split policy and hash, preprocessing version, architecture/config, label map/tokenizer, seed, library versions, hardware, command, training duration, validation metrics, final test metrics only when the model is frozen, artifact path and checksums. Keep best and last checkpoints. A model moves from experimental → candidate → approved only after leakage checks, reproducible evaluation, model card, inference compatibility, real integration checks and license review. Promote by changing a registry pointer/config, not overwriting a prior model; rollback to the previous immutable package.

Use a small JSON/Markdown registry first. MLflow/DVC are not mandatory for this single-user repo; add only if multiple runs, collaborators or remote artifacts make the simpler record insufficient.

## 6. Storage and Colab data lifecycle

~~~mermaid
flowchart TB
  GIT[Git: source, schemas, configs, small permitted fixtures]
  EXT[Authorized raw data outside Git]
  WIN[Windows local: code, API, UI, small validation fixtures]
  DRIVE[Google Drive: manifests, resumable checkpoints, reports]
  COLAB[Colab runtime disk: staged data and hot feature cache]
  ART[Immutable model package plus evaluation report]
  EXT --> COLAB
  DRIVE --> COLAB
  COLAB --> DRIVE
  COLAB --> ART
  GIT --> WIN
  GIT --> COLAB
  ART --> WIN
~~~

Do not copy 56.8 GB archives into the repository or repeatedly process them from Drive. Prefer resumable, checksum-verified downloads; stage active files on Colab runtime storage; persist compact manifests, progress markers and checkpoints to Drive. The prior Colab notebook extracts entire archives to Drive and uses unpinned installs/source-overwrite cells; replace that workflow. The task's local GPU has 4 GB VRAM and no ML packages are installed, so a full training run is not assumed locally.

## 7. Inference data flow and failure handling

For uploads, stream/spool bytes to a generated private temp file; enforce byte cap before model work; inspect actual video stream; enforce duration/dimensions/frames/codec; decode under bounded workers; produce a validation report; pass only a safe path and selected task to the adapter; return task-typed result; delete media on success, failure, cancellation and timeout. Persist only expiring job/result metadata unless the user explicitly opts into retention. Never log image/video bytes, raw transcripts or unneeded identifiers.

Stable failures include unsupported media, too large/long, missing video stream, decode failure, input/task mismatch, poor tracking/insufficient frames, model not ready, inference timeout and inference failure. A batch evaluation records a failed row and continues; interactive inference returns a useful error and a request ID. Model absence is `MODEL_NOT_READY`, never a placeholder prediction.

## 8. Implemented pipeline entry points and current gate

The canonical data path is now `ml.datasets.adapters` → `ml.datasets.fingerprint` → `ml.datasets.validate` → `ml.datasets.near_duplicates` (manual candidate review) → `ml.datasets.preprocess`. The INCLUDE downloader requires an explicit rights-review acknowledgment, checks Zenodo checksums and free space, and extracts selected ZIP members with traversal, symlink and size protections. It was not run because no authorized media was present and the reported full archive exceeds the available local disk.

The isolated trainer consumes the split-aware `index.csv` emitted by preprocessing, rejects group/hash leakage, requires train/validation/test coverage, leaves mirror augmentation off by default, and can fit a scalar temperature plus a validation precision/coverage operating point. ONNX export binds calibration to checkpoint, label-map and ONNX hashes. No such model artifact has been generated here.

The continuous baseline is in `ml/continuous/`: it builds a vocabulary from training references only, uses its own visual-sequence Transformer and English decoder, checks grouped splits, and writes multi-reference BLEU/chrF, WER/CER, ROUGE-L and hypotheses. It uses the existing 170D pose/hand features, so its limitation around face/non-manual cues is recorded. It has not been trained; readiness remains disabled.

Current commands (run after source terms are approved and local paths are ready):

```powershell
python -m ml.datasets.fingerprint --manifest data/manifests/source.csv --data-root data/raw --out data/manifests/source-hashed.csv
python -m ml.datasets.validate --manifest data/manifests/source-hashed.csv --data-root data/raw --verify-hashes --decode-media --require-approved-rights --require-splits --require-group-ids
python -m ml.datasets.near_duplicates --manifest data/manifests/source-hashed.csv --data-root data/raw --out data/reports/near_duplicates.json
python -m ml.datasets.preprocess --manifest data/manifests/source-hashed.csv --data-root data/raw --task isolated --out data/processed/isolated-v1
python -m isl.train --index data/processed/isolated-v1/index.csv --out runs/isolated/exp1
python -m isl.export_onnx --ckpt runs/isolated/exp1/best.pt --out models/isl.onnx
python -m ml.datasets.preprocess --manifest data/manifests/continuous.csv --data-root data/raw --task continuous --out data/processed/continuous-v1
python -m ml.continuous.train --index data/processed/continuous-v1/index.csv --out runs/continuous/exp1
```

These are documented commands, not claims of completed dataset or model work. Source, index, run reports and final packages must be reviewed before promotion.
