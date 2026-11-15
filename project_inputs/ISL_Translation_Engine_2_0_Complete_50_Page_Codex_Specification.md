# ISL Translation Engine 2.0 --- Complete Codex Engineering Specification

**Document type:** Master product, research, ML engineering, software
engineering, QA and deployment specification\
**Scope:** Phases 0--8, end to end\
**Primary product:** Indian Sign Language (ISL) video → English text\
**Tasks:** Isolated sign/word recognition and continuous signing-video
translation\
**Intended execution environment:** Existing project repository, VS
Code/Codex, local Windows development and Google Colab/GPU where
required\
**Status:** Engineering plan and execution contract; not a claim that
any dataset, model or deployment has already been verified.

> **Core instruction:** Codex must inspect the actual project and
> attachments, preserve valid work, implement the system incrementally,
> test every meaningful change, document evidence, and continue through
> all phases. It must not stop after Phase 0 unless a genuine blocker
> requires user input.

------------------------------------------------------------------------

## Document control

  -----------------------------------------------------------------------
  Field                               Specification
  ----------------------------------- -----------------------------------
  Project                             ISL Translation Engine 2.0

  Product input                       Video of a person signing in Indian
                                      Sign Language

  Product output                      English text

  Capability A                        Isolated sign/word recognition

  Capability B                        Continuous sentence/phrase
                                      translation

  First delivery priority             Reliable uploaded-video workflow

  Later capabilities                  Live isolated recognition, then
                                      live continuous translation if
                                      technically justified

  Primary engineering rule            Evidence before claims

  Model rule                          No mock predictions, invented
                                      metrics or unsupported accuracy
                                      claims

  Data rule                           Provenance, licence, split
                                      integrity and reproducibility are
                                      mandatory
  -----------------------------------------------------------------------

## How to use this specification

1.  Place this Markdown file in the Codex task or project workspace.
2.  Attach the existing repository as a local workspace folder or clone.
    A repository URL alone may not expose all files or local artifacts.
3.  Attach the previous implementation notebook,
    `Action_Detection_Refined_Tasks_Compatible.ipynb`, SRS/project
    report, previous-vs-new report, architecture document, technical
    blueprint, model weights and label mappings where available.
4.  Provide dataset references and only those local dataset files that
    may legally be shared.
5.  Ask Codex to execute the **Master Execution Prompt** in Part 20.
6.  Require Codex to maintain `docs/PROJECT_PROGRESS.md`, update it
    after each work session, and continue automatically through the
    phases.

This document is intentionally detailed. It is not a demand to implement
every optional technology. Codex must choose the simplest technically
sound design that satisfies the requirements and must justify deviations
from this specification.

------------------------------------------------------------------------

# 1. Product vision, problem definition and measurable outcomes

### 1.1 Problem statement

The system is intended to help a user submit a video containing Indian
Sign Language and receive an English text interpretation. The product
must distinguish two fundamentally different recognition problems:

-   **Isolated sign recognition:** a bounded clip contains one sign or
    one labelled lexical unit. The model predicts one class from a known
    vocabulary.
-   **Continuous translation:** a clip contains a sequence of signs
    forming a phrase or sentence. The model generates an English
    sequence using temporal context.

These tasks must not be treated as interchangeable. An isolated
classifier can identify a known label but does not, by itself,
understand sentence structure, grammatical relations, coarticulation,
non-manual markers or context. Concatenating isolated predictions does
not create a reliable sentence translator.

### 1.2 Intended user journey

1.  The user opens the application.
2.  The user chooses **Isolated Recognition** or **Continuous
    Translation**.
3.  The user uploads a supported video.
4.  The system validates the file and previews it.
5.  The backend decodes and preprocesses the video using the same
    versioned preprocessing contract used during model training.
6.  The appropriate model performs inference.
7.  The application displays the result, model version, processing
    status and relevant limitations.
8.  Optionally, the user adds the video to a test library with a
    verified reference label or sentence. This is an evaluation
    workflow, not ordinary inference.

### 1.3 Product success measures

Do not define success only as a high accuracy number. Success has
multiple dimensions:

-   Functional: a real supported video completes the upload-to-inference
    path.
-   Scientific: metrics are computed on an appropriately held-out test
    split.
-   Generalisation: signer-independent evaluation is performed when
    signer metadata permits.
-   Linguistic: translation output is semantically grounded, not merely
    fluent.
-   Engineering: setup, tests, model loading and deployment are
    reproducible.
-   Usability: the user understands what the model supports and what it
    cannot guarantee.
-   Governance: dataset licences, model licences, privacy and video
    retention are documented.

### 1.4 Explicit non-goals

Unless separately approved, do not implement text-to-sign generation,
speech-to-text, other sign languages, arbitrary open-vocabulary ISL
understanding, automatic sign-language tutoring, emotion inference,
identity recognition or surveillance functionality. Do not add a chatbot
or RAG layer merely to make the project appear more advanced.

------------------------------------------------------------------------

# 2. Scope boundaries, definitions and system assumptions

### 2.1 Terminology

  -----------------------------------------------------------------------
  Term                                Meaning in this project
  ----------------------------------- -----------------------------------
  Signer                              Person performing signs in a
                                      recording

  Isolated clip                       A video segment intended to contain
                                      one labelled sign

  Continuous clip                     A sequence of signs expressing a
                                      phrase or sentence

  Gloss                               A written label or annotation
                                      representing a sign/lexical unit

  Reference translation               English sentence supplied by
                                      dataset annotators

  Landmark                            A tracked keypoint, such as a hand,
                                      pose or face coordinate

  Signer-independent split            Train/test split that prevents the
                                      same signer appearing across both

  OOV                                 Out of vocabulary: a sign not
                                      represented in the model's
                                      supported label set

  Model artifact                      Saved weights plus architecture,
                                      label map, preprocessing and
                                      version metadata
  -----------------------------------------------------------------------

### 2.2 Scope freeze

The initial release is ISL-only and English-output-only. The product
accepts video. The two ML tasks have separate model adapters, data
loaders, evaluation scripts and result types. A shared UI and backend
are acceptable, but shared infrastructure must not erase task-specific
behavior.

### 2.3 Assumptions that must be verified

-   The existing repository may contain useful code, but its current
    state is unknown until inspected.
-   The earlier project may contain an isolated model with a reported
    262-class or 263-label configuration. The authoritative class count
    must come from actual label files and checkpoint dimensions.
-   The supplied action-detection notebook may contain reusable temporal
    modelling or preprocessing patterns, but it is not automatically a
    valid ISL translation implementation.
-   Dataset counts cited in papers do not establish that the
    corresponding files are currently accessible or licensed for this
    use.
-   Hardware and training capacity must be inspected. Do not assume
    local GPU availability.
-   Google Drive can be used for storage, but active training data
    should be staged on the Colab runtime disk where possible to reduce
    repeated Drive I/O.

### 2.4 Requirement traceability

Every product requirement should have: 1. a unique ID; 2. a description;
3. a phase that implements it; 4. one or more tests; 5. an acceptance
condition; 6. a status and evidence link.

Create `docs/REQUIREMENTS_TRACEABILITY.md`. A requirement without a test
or a reason why it cannot be tested is not considered verified.

------------------------------------------------------------------------

# 3. Existing-project preservation and migration policy

### 3.1 Preserve before changing

Before edits: - inspect `git status`; - identify uncommitted user
work; - create a branch or ask the user to commit if the working tree is
unsafe; - record the current commit hash; - inventory large artifacts
and local-only files; - ensure `.gitignore` excludes datasets, temporary
videos, secrets and large generated outputs as appropriate.

Never reset, clean, force-checkout, delete checkpoints, overwrite
notebooks or remove data to make the repository look tidy.

### 3.2 Classify existing components

For every major component, assign one status:

-   **Keep:** correct, tested and compatible.
-   **Keep with modification:** useful but requires a bounded change.
-   **Replace:** incorrect or fundamentally incompatible; replacement
    plan and migration evidence required.
-   **Retire:** obsolete, unused and safe to remove only after
    confirming references.
-   **Unknown:** insufficient evidence; do not delete.

### 3.3 Compatibility strategy

Do not make a broad rewrite before establishing a working baseline.
Introduce adapters around old code where feasible. For example, wrap an
existing TensorFlow model behind a stable inference interface rather
than immediately porting it to PyTorch. If the old model cannot load or
its preprocessing is irrecoverable, document the evidence and then
replace it.

### 3.4 Migration records

For every significant change, write: - old behavior; - new behavior; -
reason; - affected files; - data/checkpoint compatibility; - test
evidence; - rollback method.

Keep this in `docs/DECISION_LOG.md` and `CHANGELOG.md`.

------------------------------------------------------------------------

# 4. Phase 0 --- Full repository and implementation audit

## Purpose

Phase 0 establishes the factual baseline. It must not become a stopping
point. Codex must complete the audit, record its findings, and proceed
into Phase 1 unless a genuine blocker prevents safe work.

### 4.1 Repository inventory

Inspect: - root files and folders; - frontend and backend entry
points; - Python modules and package configuration; - notebooks and
scripts; - data folders and manifests; - model weights and label
encoders; - test files and CI configuration; - README and setup
instructions; - environment variables and deployment files; - generated
artifacts and temporary files.

Produce a concise tree in `CURRENT_STATE_AUDIT.md`. For each important
module, describe its responsibility and whether it is imported or used
by the application.

### 4.2 Notebook audit

Open each notebook and inspect the actual cell sequence. Record: -
notebook purpose; - dataset source and exact path assumptions; - label
mapping; - preprocessing and tensor shape; - split creation order; -
augmentation order; - architecture and output dimensions; - loss and
optimizer; - training epochs and stopping rule; - metrics and whether
they are train, validation or test; - checkpoint save/load behavior; -
inference implementation; - execution status and missing dependencies.

A notebook with outputs is not proof that all cells can run from a clean
kernel. Where feasible, restart and run top-to-bottom on a small
permitted sample. Otherwise mark it **not executed**.

### 4.3 Previous isolated model audit

Verify: - number of classes from label map; - output layer dimension; -
whether label indices map consistently to class names; - number of
original samples; - number of augmented samples; - whether original and
augmented versions are split across train/test; - whether signer/session
leakage is possible; - whether class distribution is severely
imbalanced; - whether saved weights match the current architecture; -
whether inference uses identical frame sampling and normalization to
training.

Do not report the old accuracy as reliable until the split and
evaluation methodology are understood.

### 4.4 Action-detection notebook audit

For `Action_Detection_Refined_Tasks_Compatible.ipynb`, determine: -
whether its dataset is action recognition, gesture recognition or
sign-language data; - how labels are defined; - how temporal sequences
are sampled; - whether the model is frame-based, landmark-based or
multimodal; - whether it uses a compatible number of frames and feature
dimensions; - whether its code can be extracted into reusable modules; -
what assumptions are incompatible with ISL.

The correct output may be that only selected utilities are reusable. Do
not force its model into the ISL pipeline.

### 4.5 Environment audit

Record: - OS and shell; - Python version; - package manager; - CPU, RAM
and available GPU; - CUDA/cuDNN versions if applicable; - free disk
space; - local video decoding support; - Colab availability and expected
runtime; - storage layout and upload/download constraints.

### 4.6 Required Phase 0 artifacts

-   `docs/CURRENT_STATE_AUDIT.md`
-   `docs/NOTEBOOK_AUDIT.md`
-   `docs/DATASET_INVENTORY.md`
-   `docs/ARCHITECTURE_GAP_ANALYSIS.md`
-   `docs/ENVIRONMENT_AUDIT.md`
-   `docs/PHASE_0_DECISIONS.md`
-   `docs/PROJECT_PROGRESS.md`

### 4.7 Phase 0 exit gate

Phase 0 is complete when the important files are inventoried, existing
run commands and failures are known, the old model is understood,
dataset access is classified, environment constraints are recorded, and
a prioritized backlog exists. Then continue automatically to Phase 1.

------------------------------------------------------------------------

# 5. Phase 1 --- Dataset acquisition and provenance

## Purpose

Create a reproducible dataset layer that can support both isolated
recognition and continuous translation without mixing incompatible
annotations or violating data terms.

### 5.1 Dataset acquisition register

Create `data/README.md` or `docs/DATASET_INVENTORY.md` with one entry
per source: - official source URL; - paper or project citation; -
version/release date; - download method; - authentication requirement; -
expected archive size; - local storage location; - task type; -
annotation fields; - signer metadata; - official split information; -
licence and permitted use; - actual local count; - checksum; - known
restrictions.

Distinguish **reported in literature**, **available at source**,
**downloaded**, **decoded successfully**, and **usable after
validation**. These are different states.

### 5.2 Candidate sources

Investigate the following as candidates, not guaranteed inputs:

-   AI4Bharat INCLUDE: the project's earlier implementation references a
    262-class/263-label setup. Verify the exact release, subset, label
    list and media count.
-   iSign: a continuous ISL-English benchmark described in research.
    Verify exact corpus files, task partitions, caption alignment and
    access terms.
-   ISLTranslate: a continuous ISL-English sentence/phrase dataset
    described in research. Verify exact downloadable data and split
    protocol.
-   CISLR: a word-level ISL recognition resource described as having a
    large vocabulary. Verify current dataset access, class structure and
    licence.
-   FDMSE-ISL: reported figures must be checked against an authoritative
    source. Do not assume availability.
-   ISH-News or other raw video sources: do not scrape or train on these
    without checking consent, licence, terms and annotation feasibility.

### 5.3 No blind dataset merging

Different sources may differ in: - sign vocabulary and gloss
conventions; - signer population and regional variation; - camera angle
and background; - recording frame rate and resolution; - sentence
segmentation; - caption style and normalization; - signer metadata; -
train/test split definitions; - rights and licence.

Create source-specific adapters. Preserve the original labels and
references. Build a mapping table only where equivalence is justified.
Unknown equivalence must remain unmapped.

### 5.4 Data layout

Suggested layout:

``` text
data/
  raw/
    include/
    cislr/
    isign/
    isltranslate/
  interim/
  processed/
    isolated/
    continuous/
    landmarks/
  manifests/
    isolated_train.parquet
    isolated_validation.parquet
    isolated_test.parquet
    continuous_train.parquet
    continuous_validation.parquet
    continuous_test.parquet
  reports/
  fixtures/
```

Raw and large processed files must normally be excluded from Git. Keep
manifests, scripts, schemas and small permitted fixtures under version
control.

### 5.5 Provenance

Every sample needs a stable internal `sample_id` and original
`source_sample_id`. Keep dataset name, version, file checksum, source
path, licence identifier, task type and preprocessing version. A model
prediction should be traceable to the model version and input file; a
training sample should be traceable to its dataset source.

### 5.6 Phase 1 exit gate

No dataset is accepted merely because it downloaded. It must pass file
validation, annotation validation, licence review and split-integrity
checks. A small end-to-end preprocessing smoke set must run
successfully.

------------------------------------------------------------------------

# 6. Phase 1 --- Data validation, split design and preprocessing

### 6.1 Media validation

Implement validators for: - file exists and is readable; - supported
container and codec; - video stream present; - duration above a sensible
minimum and below configured maximum; - valid width/height; - nonzero
frame rate; - frames can be decoded; - no unexpected empty or corrupted
sections; - frame count agrees reasonably with duration and FPS.

Do not assume file extension proves media type. Use actual media
probing/decoding.

### 6.2 Annotation validation

For isolated data: - label is present; - label ID maps to one known
class; - label text is nonempty; - clip segmentation is consistent with
the dataset specification.

For continuous data: - reference sentence is present and nonempty; -
video/caption pairing is valid; - start/end times, if supplied, are
valid; - multiple references are preserved rather than silently
collapsed; - text normalization is recorded as a separate field.

### 6.3 Duplicate and leakage detection

Use: - SHA-256 for exact file duplicates; - metadata comparisons for
repeated source IDs; - perceptual hashes or sampled-frame similarity for
near-duplicates; - signer/session grouping when available; - source
recording and augmentation lineage checks.

Near-duplicate detection is a review aid. Do not automatically delete
similar clips if they may represent legitimate repeated examples.
Produce a report with candidate pairs and decision status.

### 6.4 Split design

Preferred isolated split: - group by signer; - optionally group by
recording session; - preserve class coverage where possible; - document
classes absent from a split.

Preferred continuous split: - use official benchmark splits if they are
reliable and compatible; - otherwise group by signer/source recording; -
do not randomly split adjacent segments from the same original video
across train and test.

If signer metadata is missing, use the strongest available grouping
(source video, session, recording batch) and state that signer
independence cannot be guaranteed.

### 6.5 Preprocessing contract

Define one configuration for: - frame sampling strategy; - target frame
count or variable-length policy; - resize and crop; - RGB/BGR
convention; - pixel normalization; - padding and attention masks; -
landmark extraction version; - coordinate normalization; -
missing-landmark handling; - temporal alignment.

Save this config with every model. The inference service must load the
exact compatible preprocessing version.

### 6.6 Landmark extraction

If using MediaPipe or another pose system: - record model/version; -
record landmark order and dimensions; - retain confidence/visibility; -
represent missing detections explicitly; - do not silently replace
missing points with arbitrary coordinates; - assess hand and face
tracking failures; - verify left/right hand conventions; - ensure
training and inference use identical extraction settings.

Landmarks reduce input dimensionality but can discard appearance cues,
handshape detail or non-manual information. Treat pose-only as an
experimental modality, not a guaranteed improvement.

### 6.7 Augmentation protocol

Apply augmentation only after split assignment and only to training
data. Candidate transformations include bounded temporal jitter, slight
speed changes, modest spatial translation/scale, realistic
brightness/contrast and compression changes. Each transformation needs a
semantic-preservation rationale.

Do not horizontally flip by default. A flip can change handedness,
spatial direction or meaning. Only permit it if the dataset/task and
qualified review establish that it preserves the target label.

Track each derivative with source sample, recipe, parameter values,
random seed and pipeline version. Do not call augmented examples
independent signers.

### 6.8 Data reports

Generate: - class frequency table; - signer and session counts; -
duration histogram; - resolution/FPS distribution; - caption length
distribution; - missing media/annotation report; - duplicate
candidates; - split overlap report; - licence/access summary.

### 6.9 Exit gate

A developer must be able to run one command to validate manifests and
another to preprocess a small fixture set. Both must produce
deterministic, inspectable outputs.

------------------------------------------------------------------------

# 7. Phase 2 --- Isolated recognition: task definition and baselines

## Purpose

Deliver a measurable isolated-sign recognition model with a clearly
defined vocabulary and input contract.

### 7.1 Task contract

Specify: - what constitutes one clip; - supported clip duration; -
supported class list; - expected camera orientation and visible body
region; - input frame policy; - accepted resolution/codec; - model
output format; - unknown/OOV behavior; - confidence interpretation; -
preprocessing version.

Do not imply the model recognizes arbitrary signs outside its label set.

### 7.2 Baseline ladder

Build models in increasing complexity: 1. **Sanity baseline:**
majority-class or simple frequency baseline. 2. **Existing model
baseline:** run the previous implementation if valid. 3. **Lightweight
temporal baseline:** a compact temporal CNN, GRU or LSTM on valid
features. 4. **Stronger candidate:** BiLSTM, temporal convolution,
transformer encoder or graph-based temporal model where justified. 5.
**Multimodal candidate:** RGB plus landmarks if data and compute support
a fair comparison.

Do not train five architectures without a hypothesis. Each experiment
must answer a question.

### 7.3 Reuse decision

If the previous 262-class model has: - valid labels; - correct output
dimension; - consistent preprocessing; - no known train/test leakage; -
reproducible checkpoint loading; - acceptable baseline evaluation;

then wrap and benchmark it. If not, record the specific defects and
rebuild the minimum necessary pipeline.

### 7.4 Training design

Use config-driven training. Record: - dataset manifest version; - split
version; - architecture; - input dimensions; - batch size; - learning
rate; - optimizer; - loss; - class weighting; - augmentation; - seed; -
epochs; - early stopping; - hardware; - package versions.

Save best validation checkpoint and last checkpoint. Do not select a
model based on test results.

### 7.5 Imbalance handling

Inspect class distribution before selecting class weights or sampling.
If weighting is used, document formula and cap extreme weights if
justified. Report macro metrics. A model can achieve high overall
accuracy by predicting frequent classes while failing rare classes.

### 7.6 Evaluation

Required: - top-1 accuracy; - top-5 accuracy where class count makes it
useful; - macro precision, recall and F1; - per-class
precision/recall/F1; - confusion matrix; - calibration analysis; -
latency and model size.

Add repeated-seed results or confidence intervals when feasible. Record
the exact test split and evaluation command.

### 7.7 Error analysis

Inspect: - confusable labels; - signs with low sample counts; -
signer-specific errors; - poor lighting/background; - occluded hands; -
motion blur; - out-of-frame hands; - long/short clips; - left/right hand
variation.

Keep representative error examples with IDs and metadata. Do not expose
private videos in public reports.

### 7.8 Confidence and abstention

Raw softmax values are not automatically calibrated probabilities.
Calibrate using validation data and report calibration method. Define
abstention behavior from validation risk/coverage analysis. If no
reliable OOV detector exists, state that limitation and avoid claiming
robust unknown-sign detection.

### 7.9 Export

The inference artifact must include: - weights; - architecture
identifier/config; - class-to-index map; - preprocessing
config/version; - framework version; - training manifest hash; - split
identifier; - evaluation metrics; - model version; - intended-use and
limitations metadata.

### 7.10 Exit gate

A real isolated video must produce a valid output through a stable
inference function. The evaluation report must be reproducible and based
on held-out data. If the new model does not beat the old baseline,
report that honestly rather than hiding the comparison.

------------------------------------------------------------------------

# 8. Phase 2 --- Isolated recognition: inference, optimization and release criteria

### 8.1 Inference interface

Implement a stable function conceptually equivalent to:

``` python
result = isolated_model.predict(video_path, config=runtime_config)
```

Return a typed result with: - predicted label ID; - predicted
gloss/word; - top-k candidates if useful; - calibrated confidence if
available; - model version; - preprocessing version; - warnings; -
elapsed time.

Do not return a raw tensor to the API layer. Do not embed label mapping
logic in the frontend.

### 8.2 Runtime optimization

Measure before optimizing. Candidate optimizations: - decode only
required frames; - batch model inputs; - avoid repeated model loading; -
use inference mode/no-gradient context; - mixed precision if numerically
safe; - model quantization if supported and tested; - ONNX/TensorFlow
Lite export only if it yields measurable deployment benefit.

For every optimized artifact, compare output consistency and metrics
with the original checkpoint.

### 8.3 Unit tests

Test: - correct label mapping; - empty/corrupt video; -
too-short/too-long clip; - frame sampler output shape; - missing
landmarks; - checkpoint mismatch; - unknown label ID; - deterministic
inference where expected; - model readiness and error handling.

### 8.4 Model card

Document: - intended task; - supported vocabulary; - training sources; -
signer/split policy; - evaluation metrics; - known biases; - known
failure cases; - hardware/runtime; - licence; - limitations and
prohibited interpretations.

### 8.5 Release gate

Do not promote a model simply because it has the highest accuracy.
Consider macro-F1, signer generalisation, latency, size, calibration and
failure severity. The approved artifact must have a unique version and
reproducible evaluation evidence.

------------------------------------------------------------------------

# 9. Phase 3 --- Continuous translation: dataset and linguistic problem

## Purpose

Build a separate continuous visual translation model. It must map a
temporal visual sequence to English text using sentence-level
supervision.

### 9.1 Why this is not isolated classification

Continuous signing includes transitions between signs, coarticulation,
timing, spatial relationships, facial expressions and context. Sentence
translation may involve grammatical restructuring rather than a
one-to-one word mapping. A list of gloss labels is not equivalent to a
natural English sentence.

### 9.2 Dataset requirements

A suitable training item generally needs: - a video or valid video
segment; - one or more English reference translations; - reliable
pairing between video and reference; - split assignment; - source
identity; - signer/session metadata if available; - licence and
provenance.

Gloss annotations are useful if provided, but must not be fabricated. If
only sentence-level captions are available, build a direct video-to-text
pipeline and describe the absence of gloss supervision.

### 9.3 Caption normalization

Keep original reference text unchanged. Create a separate normalized
field for metric computation. Document: - Unicode normalization; -
punctuation handling; - casing; - whitespace; - contractions; -
tokenization; - number normalization; - whether multiple references are
retained.

Do not normalize away meaningful content. Do not use a different
normalization policy for model output and reference without documenting
it.

### 9.4 Data mixtures

Before combining iSign, ISLTranslate or any other corpus: - compare
sample IDs and hashes; - compare source videos and signer overlap; -
compare caption conventions; - compare split protocols; - compare
licence terms; - compare annotation quality; - identify duplicated or
near-duplicated sentences.

Train on a mixture only if it is defensible. Keep source IDs and
sampling weights so the mixture can be audited.

------------------------------------------------------------------------

# 10. Phase 3 --- Continuous translation: model design and experiments

### 10.1 Baseline first

Create a minimal baseline that proves the data loader, visual input,
text target and decoding pipeline work. A small baseline is better than
an unrepeatable large training run.

### 10.2 Candidate architecture

A general sequence-to-sequence design may contain: 1. video decoder and
frame sampler; 2. visual feature encoder; 3. temporal/spatiotemporal
encoder; 4. optional landmark stream; 5. projection into decoder
dimension; 6. text tokenizer; 7. autoregressive text decoder; 8.
constrained output post-processing.

The exact architecture must be chosen after dataset and compute
inspection. Do not assume a particular pretrained model is licensed,
available or appropriate.

### 10.3 RGB, pose and fusion experiments

-   RGB can preserve hand appearance, clothing/background context and
    visual details.
-   Pose/landmarks can provide compact motion structure but may miss
    handshape or facial information.
-   Fusion may improve or degrade performance depending on data quality
    and alignment.

Compare modalities with identical splits and comparable training
budgets. Record the cost of landmark extraction and inference.

### 10.4 Temporal handling

Specify: - frame sampling rate; - maximum sequence length; -
variable-length handling; - padding and attention masks; - truncation
policy; - chunking policy if any; - temporal augmentation; - handling of
very long videos.

Never silently discard the final portion of a sentence. If a maximum
duration is enforced, reject or warn explicitly.

### 10.5 Decoder and training

Use a text tokenizer suitable for English output. Save tokenizer files
and special token IDs with the checkpoint. Record: - teacher-forcing
policy; - loss; - optimizer; - schedule; - gradient clipping; - mixed
precision; - batch construction; - checkpoint selection; - decoding
settings.

Training loss is not translation quality. Evaluate with autoregressive
generation.

### 10.6 Decoding

Compare greedy decoding and beam search only if useful. Record beam
width, length penalty, maximum output length and stopping token. Avoid
post-processing that inserts unsupported words or silently changes
meaning.

### 10.7 Evaluation metrics

Report: - BLEU with exact library/version/tokenization; - chrF or
chrF++; - ROUGE-L; - WER/CER where meaningful; - sentence-level
examples; - human semantic review if qualified reviewers are available.

Metrics are not interchangeable. BLEU is sensitive to tokenization and
exact wording. WER can penalize valid paraphrases. Explain the limits of
each metric.

### 10.8 Semantic error analysis

Review: - omitted signs or concepts; - hallucinated content; - incorrect
negation; - incorrect person/number/time; - repeated phrases; - sentence
boundary errors; - grammatical fluency without visual grounding; -
ambiguity where multiple translations are plausible.

A language decoder can produce fluent but incorrect English. Fluency
must not be used as evidence of visual understanding.

### 10.9 Exit gate

A real continuous video must produce a generated English sequence.
Held-out evaluation must be reported with the exact data split, metric
versions and sample outputs. If compute or data are insufficient,
deliver a runnable experiment pipeline and state that full training
remains incomplete.

------------------------------------------------------------------------

# 11. Phase 4 --- Backend architecture and API contracts

## Purpose

Integrate the real models behind a stable API. The backend is
responsible for input validation, preprocessing, model routing,
structured output and safe file handling.

### 11.1 Application boundaries

Separate: - API routers; - request/response schemas; - core settings; -
video validation; - preprocessing services; - model registry/loader; -
isolated inference adapter; - continuous translation adapter; -
temporary storage; - logging and error handling.

Do not put model architecture code directly inside route handlers.

### 11.2 API routes

Minimum routes: - `GET /api/v1/health`: process responds. -
`GET /api/v1/readiness`: required model artifacts and dependencies are
ready. - `POST /api/v1/recognize`: isolated video recognition. -
`POST /api/v1/translate`: continuous video translation.

Optional: - job creation/status/result routes for long-running
inference; - model metadata route; - version route.

### 11.3 Upload validation

Validate actual media, not just extension. Enforce: - maximum bytes; -
maximum duration; - allowed containers/codecs; - video stream exists; -
decode succeeds; - dimensions and frame count are within supported
limits; - temporary path is generated safely; - filename is sanitized; -
file is removed according to retention policy.

Do not trust user-provided paths or filenames. Prevent path traversal.

### 11.4 Response design

Isolated response should contain task, label ID, gloss, top-k candidates
if supported, calibrated confidence if supported, model version,
warnings and processing time.

Translation response should contain task, generated English text, model
version, warnings and processing time. Do not invent a confidence score
for a generative translation model if no meaningful calibration method
exists.

### 11.5 Long-running inference

If inference exceeds request timeout: - create a job identifier; -
return accepted status; - provide status/result endpoint; - support
cancellation if safe; - expire completed jobs; - clean temporary
files; - avoid duplicate job execution where practical.

Do not add a distributed queue until concurrency or reliability requires
it.

### 11.6 Error model

Return stable error codes such as: - `UNSUPPORTED_MEDIA`; -
`VIDEO_TOO_LARGE`; - `VIDEO_TOO_LONG`; - `VIDEO_DECODE_FAILED`; -
`MODEL_NOT_READY`; - `INFERENCE_FAILED`; - `TASK_INPUT_MISMATCH`.

Messages should be useful but must not expose stack traces, secrets or
internal file paths.

### 11.7 Backend tests

Use test fixtures and mocked model adapters for route logic, then add
integration tests using a real small model artifact and permitted video
fixture. Keep mock-based tests clearly separate from real-model tests.

------------------------------------------------------------------------

# 12. Phase 4 --- Backend integration, performance and observability

### 12.1 Model lifecycle

Load model artifacts once during application startup or through a
controlled lazy loader. Read the artifact manifest and verify: -
architecture compatibility; - framework version compatibility; - label
map presence; - preprocessing version; - tokenizer presence for
translation; - checksum or integrity metadata.

If a model is unavailable, readiness must fail clearly. The API must not
substitute sample output.

### 12.2 Resource controls

Measure CPU, RAM, GPU memory and processing time. Configure upload
limits based on observed capacity. Limit concurrent jobs if they can
exhaust memory. Avoid unbounded in-memory video buffering.

### 12.3 Logging

Use structured logs with request ID, task type, model version, status,
processing duration and safe error code. Do not log raw video, sensitive
user content or unnecessary identifying information. Keep debug traces
out of public responses.

### 12.4 Privacy and retention

Default to temporary processing. Define: - whether uploaded videos are
retained; - where they are stored; - who can access them; - when they
are deleted; - how users request deletion if accounts/history exist; -
whether any videos are used for training.

Never silently add user uploads to training data.

### 12.5 API documentation

Publish request formats, supported codecs, size/duration limits,
response schemas, errors, example requests and model limitations in
OpenAPI and `docs/API_REFERENCE.md`.

### 12.6 Integration gate

A real video must pass through upload validation, decode, preprocessing,
model inference and response serialization. The output must be traceable
to a model version. Test corrupt files, missing model files, timeouts
and cleanup.

------------------------------------------------------------------------

# 13. Phase 5 --- Frontend product experience

## Purpose

Provide a clear browser interface that makes the two tasks
understandable and allows a user to test real videos without running
notebooks.

### 13.1 Information architecture

Recommended screens: - landing/dashboard; - isolated recognition; -
continuous translation; - test library/evaluation (developer or owner
mode); - model/system information; - optional history only if retention
is deliberately implemented.

The two inference modes must be visibly separate.

### 13.2 Upload experience

For each mode: 1. show supported formats and limits; 2. accept drag/drop
and file picker; 3. validate file size client-side as a convenience, but
repeat validation server-side; 4. preview the video; 5. allow
removal/reset; 6. submit and show progress; 7. display result or a
useful error.

Do not imply that client-side checks replace backend validation.

### 13.3 Isolated result design

Display: - predicted gloss/word; - top-k candidates if useful; -
calibrated confidence only if available; - model version; - warning that
only supported classes are recognized; - processing time.

Do not present a closed-set output as a guaranteed translation of any
arbitrary sign.

### 13.4 Continuous result design

Display: - generated English text; - model version; - processing time; -
warning that output can omit or misinterpret signs; - optional copy
action; - optional comparison with reference only in evaluation mode.

Do not show a confidence percentage unless the model has a defensible
calibrated measure.

### 13.5 Accessibility and usability

Implement keyboard navigation, semantic labels, visible focus, contrast,
responsive layout and clear status announcements. Avoid excessive
animations. Provide useful empty, loading, error and success states.

### 13.6 Frontend architecture

Use typed API response interfaces. Keep network code in an API client
layer. Keep task-specific result components separate. Use environment
variables for API base URL. Do not hard-code deployment URLs throughout
the UI.

### 13.7 Demo mode

If sample output is required for a presentation, label it clearly as
**sample/demo data** and keep it separate from the real inference path.
Never use fake predictions to make an incomplete model look functional.

------------------------------------------------------------------------

# 14. Phase 5 --- Test-video library and batch evaluation

### 14.1 Purpose

The owner should be able to test multiple videos with known ground truth
without editing model code or manually executing notebook cells.

### 14.2 Manifest format

A test manifest should include: - `test_id`; - `media_path`; -
`task_type`; - `ground_truth_label` for isolated samples; -
`reference_text` for continuous samples; - `dataset_id`; -
`source_sample_id`; - `signer_id` if available; - `split`; -
`licence_id`; - `notes`.

### 14.3 Separation from training

Evaluation-only files must not accidentally enter training. The
evaluation runner must verify split metadata and report suspicious
overlap. The application must not automatically learn from a user's test
upload.

### 14.4 Batch workflow

1.  Select a manifest or test folder.
2.  Validate each file.
3.  run the correct task adapter;
4.  save one row per sample;
5.  record failures without stopping the entire batch;
6.  calculate aggregate metrics for successful, valid samples;
7.  generate per-class and per-sample reports;
8.  export CSV/JSON and a readable Markdown/HTML report.

### 14.5 Isolated evaluation output

Include expected label, predicted label, correctness, top-k, confidence
if calibrated, latency, model version and error status.

### 14.6 Translation evaluation output

Include reference text, generated text, normalized forms, metric values,
latency, model version and error status. Preserve original text
separately from normalized text.

### 14.7 Report interpretation

Always report sample count and excluded/failed count. Do not silently
drop invalid samples. Explain whether metrics are corpus-level or
sentence-level and identify tokenizer/library versions.

### 14.8 Exit gate

Adding a valid test manifest row should be enough to include a video in
the next evaluation run. The runner must produce reproducible per-sample
predictions and summary reports.

------------------------------------------------------------------------

# 15. Phase 6 --- Live isolated-sign recognition

## Purpose

Add webcam recognition for one isolated sign at a time, only after
upload inference is stable.

### 15.1 Capture design

The browser requests camera permission and displays a preview. The user
must be able to start, stop and reset capture. Provide a clear
indication when frames are being processed.

### 15.2 Temporal buffer

Define: - capture frame rate; - rolling buffer duration; - buffer
length; - frame sampling policy; - start/end capture behavior; - minimum
valid frames; - maximum clip duration; - behavior when hands leave the
frame.

Do not send every camera frame by default. Use measured capture
frequency and model needs.

### 15.3 Prediction stabilization

Frame-by-frame predictions may flicker. Consider: - temporal voting; -
confidence smoothing; - minimum consecutive agreement; - cooldown after
finalizing a sign; - explicit user confirmation.

Choose parameters using validation videos, not arbitrary values. Avoid
excessive smoothing that hides real transitions.

### 15.4 Interaction states

Implement: - camera not requested; - permission pending; - camera
active; - recording/buffering; - processing; - prediction ready; -
low-confidence/unknown; - camera error; - network error; - stopped.

### 15.5 Latency and stability

Measure capture-to-display latency, dropped frames, CPU/GPU load, memory
and behavior over extended sessions. Test supported browsers and
devices. Report the hardware used.

### 15.6 Exit gate

Live isolated recognition is ready only when permissions, buffer logic,
prediction stabilization, failure handling and latency are tested. If it
is unstable, keep upload recognition as the supported workflow.

------------------------------------------------------------------------

# 16. Phase 7 --- Live continuous translation

## Purpose

Investigate streaming sentence translation. This phase is optional in
the sense that it may be deferred if the model, data or compute cannot
support genuine streaming.

### 16.1 Streaming versus clip inference

A model that receives a complete video and returns one sentence is not a
streaming model. A real streaming design must accept sequential chunks
or a live temporal stream and manage partial hypotheses.

### 16.2 Key technical questions

-   Does the model support causal/online inference?
-   How much past context is required?
-   How are sign boundaries or utterance endings detected?
-   Can a partial hypothesis be revised?
-   How is output finalized?
-   What happens when a signer pauses?
-   How does the system handle network jitter and dropped frames?
-   How is long-range context retained without unbounded memory?

### 16.3 Chunking and segmentation

Test chunk sizes and overlap. Compare chunk-based results with
complete-clip translation. Measure whether chunk boundaries split signs
or remove context. If segmentation is not reliable, use explicit
record-then-translate.

### 16.4 Partial and final hypotheses

Mark partial text as provisional. The UI must distinguish: - incoming
video; - processing; - partial translation; - finalized translation; -
failure/retry.

Do not silently replace previously displayed words without indicating
that the hypothesis changed.

### 16.5 Transport

A WebSocket may be suitable, but use it only if continuous frame/event
transport is needed. Implement authentication if exposed, message-size
limits, backpressure, cancellation, reconnect and idle timeout.

### 16.6 Evaluation

Measure: - end-to-end latency; - time to first partial output; - final
output latency; - partial hypothesis stability; - translation metrics; -
quality difference vs upload mode; - dropped frame rate; - resource
consumption.

### 16.7 Deferral decision

If streaming quality or latency is inadequate, document why and keep a
record-then-translate workflow. This is a valid engineering decision. Do
not claim real-time translation merely because the UI updates
periodically.

------------------------------------------------------------------------

# 17. Phase 8 --- Security, privacy and responsible data handling

### 17.1 Threat model

Consider: - malicious uploads; - oversized files and
decompression/resource exhaustion; - path traversal; - arbitrary file
execution; - exposed debug endpoints; - leaked API keys; - unauthorized
access to uploaded videos; - accidental retention of sensitive
footage; - data leakage through logs; - misuse of model outputs as
authoritative interpretation.

### 17.2 Upload security

-   Generate server-side temporary filenames.
-   Store uploads outside executable/static directories.
-   Validate actual media streams.
-   Set size, duration, frame and processing limits.
-   Reject malformed content safely.
-   Clean up on success, failure, cancellation and timeout.
-   Never trust client MIME type or filename.

### 17.3 Secrets and configuration

Use environment variables or secret management. Provide `.env.example`
with dummy values only. Ensure `.env`, keys, tokens and credentials are
ignored by Git. Rotate any secret accidentally committed.

### 17.4 Privacy

State whether videos are transient or retained. Provide deletion
behavior if stored. Do not add uploads to training without explicit
informed permission and a documented consent process. Minimize logging
and access.

### 17.5 Dataset and model licences

Record licence for each source and pretrained model. Check whether
redistribution, commercial use, derivative model distribution and public
demos are allowed. Do not assume research availability equals
unrestricted use.

### 17.6 Responsible output

The interface should describe the system as an assistive experimental
tool unless evidence supports a stronger claim. It must not be presented
as a replacement for a qualified human interpreter in high-stakes
settings.

------------------------------------------------------------------------

# 18. Phase 8 --- Testing, performance and deployment

### 18.1 Test layers

-   Unit tests: preprocessing, schemas, label mapping, video validation,
    decoding.
-   Data tests: manifest validity, missing media, duplicate checks,
    split leakage.
-   Model tests: checkpoint load, tensor contract, output shape, smoke
    inference.
-   Evaluation tests: known held-out sample outputs and metric
    computation.
-   API tests: upload, validation, inference adapter, errors and
    cleanup.
-   UI tests: mode selection, upload, progress, result, reset and error
    states.
-   Deployment tests: clean setup, readiness, limits, model loading and
    smoke inference.

### 18.2 Performance tests

Use representative clip lengths, resolutions and file sizes. Measure: -
decode time; - preprocessing time; - model inference time; - total
request time; - peak RAM/VRAM; - throughput; - concurrent request
behavior; - model startup time.

Set limits from actual measurements. Do not invent latency targets
without considering hardware.

### 18.3 Dependency and build quality

Pin tested dependency versions. Add formatter, linter and type checker
where appropriate. Run dependency vulnerability checks. Avoid adding
packages for trivial tasks.

### 18.4 Deployment topology

Separate frontend hosting from Python inference hosting if required. A
static frontend platform does not automatically host a long-running
Python ML backend or GPU model. Select deployment based on model size,
compute, file upload limits, persistent storage, cold-start behavior and
budget.

### 18.5 Staging before production

Deploy a staging environment with actual model artifacts. Run: -
health/readiness; - valid isolated upload; - valid continuous upload; -
invalid file rejection; - oversized file rejection; - cleanup check; -
basic latency test.

Only then prepare public release.

### 18.6 Release package

Include version, changelog, environment configuration, model artifact
identifiers, data/model licence notices, known issues, monitoring
instructions and rollback procedure.

------------------------------------------------------------------------

# 19. MLOps, experiment governance and model lifecycle

### 19.1 Experiment record

Each experiment must save: - experiment ID and timestamp; - Git
commit; - data manifest hash/version; - split version; - preprocessing
version; - model architecture/config; - tokenizer and label-map
version; - random seed; - environment and library versions; -
hardware; - training duration; - validation metrics; - test metrics only
for final evaluation; - checkpoint path and checksum.

### 19.2 Model registry

Maintain a simple registry, even if initially a Markdown/JSON file. For
each model: - task; - version; - status (experimental, candidate,
approved, retired); - artifact path; - data version; - preprocessing
version; - evaluation report; - known limitations.

### 19.3 Promotion policy

A candidate model becomes approved only after: - reproducible
evaluation; - no unresolved data leakage; - acceptable
per-class/semantic performance; - compatible inference contract; -
successful integration tests; - model card completed.

### 19.4 Rollback

Keep the previous approved model and configuration. A new model
deployment must be reversible without retraining. Document how to switch
model versions.

### 19.5 Monitoring

Monitor failures, latency, resource usage and model version. Do not
claim model drift detection unless a real monitoring method and
reference data are implemented. Do not collect raw videos unnecessarily.

------------------------------------------------------------------------

# 20. Detailed Codex execution protocol and master prompt

## 20.1 Operating role

You are the lead machine-learning engineer, computer-vision engineer,
data engineer, backend engineer, frontend engineer, test engineer and
technical writer responsible for completing ISL Translation Engine 2.0
in the provided workspace.

## 20.2 Mission

Build an evidence-based Indian Sign Language video-to-English
application with two separate capabilities: 1. isolated ISL sign/word
recognition; 2. continuous ISL video-to-English translation.

Uploaded video is the first dependable workflow. Live isolated
recognition and live continuous translation are later capabilities and
must be implemented only when technically supported and tested.

## 20.3 Mandatory execution behavior

1.  Read this entire specification and inspect all attachments.
2.  Start with Phase 0, but do not stop there.
3.  Execute Phases 0--8 in order.
4.  After each phase meets its exit criteria, proceed automatically to
    the next phase.
5.  Do not ask the user to repeatedly approve ordinary implementation
    steps.
6.  Ask only for genuine blockers, permissions, irreversible actions or
    decisions that cannot be safely inferred.
7.  Preserve useful code and data; do not delete user work.
8.  Never invent dataset access, counts, labels, signer IDs, metrics,
    model outputs, training completion or deployment success.
9.  Keep isolated and continuous tasks separate.
10. Do not blindly merge datasets or use data without checking
    licence/terms.
11. Split before augmentation; no horizontal flips by default.
12. No fake predictions in the actual application.
13. If GPU training is unavailable, create runnable training assets, run
    feasible smoke tests and identify the exact manual Colab/GPU step.
14. Update documentation and progress after every meaningful change.

## 20.4 Required phase routine

At the start of each phase: - inspect the current repository and
previous phase outputs; - write a concise implementation plan; -
identify files to create/change; - identify risks and tests; - implement
the smallest coherent increment.

At the end of each phase: - run tests and record actual outputs; - fix
regressions; - update documentation; - update
`docs/PROJECT_PROGRESS.md`; - mark exit criteria as passed, failed or
blocked; - if passed, begin the next phase without waiting for a new
user prompt.

## 20.5 Progress file

Maintain `docs/PROJECT_PROGRESS.md` with: - current phase; - task
status: not started / in progress / done / blocked; - files created and
modified; - commands actually run; - tests actually executed and
pass/fail counts; - dataset counts actually verified; - training runs
actually completed; - metrics with dataset and split provenance; -
blockers and user actions; - next concrete action.

## 20.6 Required final handover

Provide: - exact environment setup commands; - exact commands to run
frontend and backend; - exact commands to validate data and
preprocess; - exact commands/notebooks to train each model; - exact
commands to evaluate each model; - exact commands to run tests; - exact
deployment steps; - actual model metrics with split provenance; - model
and data limitations; - instructions for adding a test video and
dataset; - list of incomplete or blocked items.

**Begin Phase 0 now. Continue through Phases 1--8 automatically. Do not
stop after the audit.**

------------------------------------------------------------------------

# Appendix A --- Required attachment checklist

Attach the following in the same Codex workspace:

  -------------------------------------------------------------------------------------------------------------------------------
  Item                                                What to provide                                     Priority
  --------------------------------------------------- --------------------------------------------------- -----------------------
  Main ISL project repository                         Full local clone/folder with frontend, backend,     Required
                                                      source, scripts, configs, requirements, tests and   
                                                      shareable artifacts                                 

  Previous 262-class implementation                   Notebook, class map, preprocessing files,           Required if available
                                                      referenced data and weights if available            

  `Action_Detection_Refined_Tasks_Compatible.ipynb`   Exact notebook and files it references              Required

  Previous SRS/project report                         Latest DOCX/PDF; identify duplicates and which is   Required
                                                      authoritative                                       

  Previous-vs-new approach report                     Full report and supporting diagrams                 Recommended

  ISL Engine 2.0 architecture document                Architecture and technical blueprint                Required

  INCLUDE references                                  https://github.com/AI4Bharat/INCLUDE and            Required reference
                                                      https://huggingface.co/datasets/ai4bharat/INCLUDE   

  Other dataset references                            iSign, ISLTranslate, CISLR and any verified         Recommended
                                                      FDMSE-ISL materials                                 

  Test videos                                         Small permitted isolated and continuous sample set  Strongly recommended
                                                      with ground truth                                   

  Environment details                                 OS, Python, GPU/RAM/disk, Colab availability and    Required
                                                      deployment target                                   
  -------------------------------------------------------------------------------------------------------------------------------

A GitHub link is not a local repository. Codex needs the workspace or a
clone. A dataset landing page is not proof that the data are downloaded.
Do not attach secrets, credentials or restricted data.

------------------------------------------------------------------------

# Appendix B --- Research references and verification policy

Starting points for verification:

-   AI4Bharat INCLUDE GitHub: https://github.com/AI4Bharat/INCLUDE
-   INCLUDE Hugging Face:
    https://huggingface.co/datasets/ai4bharat/INCLUDE
-   iSign paper: https://arxiv.org/html/2407.05404v1
-   iSign project: https://exploration-lab.github.io/iSign/
-   ISLTranslate paper: https://aclanthology.org/2023.findings-acl.665/
-   CISLR paper: https://aclanthology.org/2022.emnlp-main.707/
-   CISLR dataset: https://huggingface.co/datasets/Exploration-Lab/CISLR
-   RKMV ISL project: https://cs.rkmvu.ac.in/\~isl/index.html

These references are starting points, not a guarantee of current
availability, complete data access or usage rights. Codex must inspect
primary sources and current dataset terms. Any literature-reported
figures (including iSign, ISLTranslate, CISLR or FDMSE-ISL counts) must
be labelled as reported figures until verified against the actual
release. Do not add corpus counts together without checking overlap and
compatibility.

------------------------------------------------------------------------

# Appendix C --- Acceptance checklist

### Product

-   [ ] Isolated and continuous workflows are distinct.
-   [ ] Uploaded videos are validated and previewed.
-   [ ] Real model inference is used.
-   [ ] Unsupported input returns a clear error.
-   [ ] Vocabulary and limitations are visible.
-   [ ] Live modes are enabled only after their own acceptance gates.

### Data and ML

-   [ ] Dataset sources, licences and versions are documented.
-   [ ] Canonical manifests exist.
-   [ ] Duplicate and leakage checks are recorded.
-   [ ] Split-before-augmentation is enforced.
-   [ ] Isolated model metrics include macro/per-class results.
-   [ ] Translation metrics and semantic examples are reported.
-   [ ] Model cards and artifact metadata exist.

### Engineering

-   [ ] Fresh setup instructions work.
-   [ ] Unit, integration and end-to-end tests exist.
-   [ ] API schemas and errors are documented.
-   [ ] Upload security and retention are implemented.
-   [ ] Deployment uses actual model artifacts.
-   [ ] Monitoring, release notes and rollback are documented.

### Handover

-   [ ] Exact run/test/train/evaluate/deploy commands supplied.
-   [ ] Actual metrics include dataset/split/model provenance.
-   [ ] Known limitations and blocked items listed.
-   [ ] Instructions for adding data and test videos supplied.
-   [ ] Project progress and decision logs are current.

------------------------------------------------------------------------

# Appendix D --- Engineering decision templates

### Decision record

``` text
Decision ID:
Date:
Context/problem:
Options considered:
Evidence gathered:
Chosen option:
Reason:
Trade-offs:
Risks:
Affected files/components:
Tests required:
Rollback:
Status:
```

### Experiment record

``` text
Experiment ID:
Task: isolated / continuous
Git commit:
Dataset manifest/version:
Split:
Preprocessing version:
Model architecture:
Hyperparameters:
Random seed:
Hardware/environment:
Training command:
Run status:
Validation metrics:
Test metrics (only final evaluation):
Checkpoint:
Known limitations:
Next experiment:
```

### Phase completion record

``` text
Phase:
Objective:
Tasks completed:
Files created/modified:
Commands executed:
Tests executed:
Results:
Exit criteria:
Unresolved issues:
User action required:
Next phase:
```

### Blocker report

``` text
Blocker:
Why it prevents progress:
Evidence:
What has already been attempted:
Options:
Recommended option:
Exact user input/permission needed:
Work that can continue in parallel:
```

------------------------------------------------------------------------

## Final operating principle

The strongest version of this capstone is not the one with the most
frameworks, the largest claimed dataset, or the longest list of model
names. It is the one that can take a real held-out ISL video, process it
through a reproducible pipeline, return a grounded result, measure its
errors honestly, and explain exactly what it can and cannot do.

Codex must optimize for that outcome. It must not optimize for
impressive-looking architecture diagrams, fake demo results, unsupported
accuracy claims or unnecessary infrastructure.
