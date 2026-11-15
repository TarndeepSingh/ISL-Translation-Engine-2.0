# Reference repository analysis: AI4Bharat/INCLUDE

**Read 10 Nov 2026.** Files actually read: `README.md`, `generate_keypoints.py`, `configs.py`, `augment.py`. **Not read:**
`runner.py`, `dataset.py`, `evaluate.py`, `train_nn.py`, `train_xgb.py`, `utils.py`, `models/`, `pretrained_links.json`, `train_test_paths/`
(the `models/` directory listing is blocked by GitHub's robots rules). Anything about those files below is unverified.

## What the repo is
MIT-licensed training code for the INCLUDE isolated-sign dataset (Sridhar et al., ACM MM 2020). It is a research code release, last
pushed about five years before this audit, not a product. It has no serving layer, no calibration, no provenance tracking and no tests that we saw.

## What it does (verified from the files above)
- **Keypoints:** MediaPipe Hands (21 points x2) + Pose (25 upper-body points, x/y only), saved per video as JSON; 67 points x 2 = **134 inputs per frame**.
  Uses the legacy `Pose(upper_body_only=True)` argument, which modern MediaPipe removed, so the code needs an old MediaPipe.
- **Hand assignment:** a hand is swapped to left/right by distance to the pose wrists (same idea as our `assign_hands`).
- **Models in `configs.py`:** 5-layer bidirectional LSTM (hidden 256), XGBoost (depth 2), BERT-style Transformer (small: 2 layers/256; large: 4 layers/512), MobileNetV2 CNN on frames (timm).
- **Augmentations:** +/-7 degree rotation, Gaussian keypoint jitter, random keypoint cutout (3 points/frame), frame dropping, frame interpolation.
- **Splits:** `train_test_paths/include{,50}_{train,val,test}.txt` (names from the code; contents not inspected).

## Reported accuracy (published claims, not reproduced here)
| Source | INCLUDE-50 | INCLUDE (263) | Note |
|---|---|---|---|
| Sridhar et al. 2020 (Zenodo/ACM abstract) | 94.5% | 85.6% | best model: pretrained feature extractor + encoder, decoder-only trained |
| Yadav & Divakar 2023 (MediaPipe Holistic + LSTM) | 94.8% | 87.4% | secondary summary, paper not read |
| Conference submission (docx on a conference site) | 96.42% | n/a | not peer-reviewed as far as seen; treat as unverified |
| IEEE DataPort derivative (60x225 tensors) | n/a | 1D-CNN 94.55% / BiLSTM 91.74% on its own random 70/15/15 split | different split; not comparable |

**Reading the numbers:** >90% on INCLUDE-50 with the *official video-level split* is well supported by the literature. >90% on all 263 classes is
**not** (published results are 85-87%). These scores measure the same school, camera and signers that appear in training; they do not predict webcam
accuracy for a new person. Report official-split and session-proxy numbers separately and never present one as the other.

## Comparison with our code
| Aspect | AI4Bharat | Ours |
|---|---|---|
| Normalisation | raw x/y (normalisation in unread `dataset.py`) | shoulder-centred/scaled pose, wrist-and-palm-relative hands, presence flags (170 features) |
| Model | LSTM / Transformer / XGBoost / CNN | Conv1D -> BiLSTM -> attention (~0.88M parameters) |
| Augmentation | rotation, jitter, cutout, drop/up-sample frames | trim jitter, hand-tracking dropout, feature noise, opt-in mirror; **now** opt-in rotation + time-warp |
| Split safety, rights, provenance, calibration, serving, tests | none seen | present |

## What we adopted, what we did not
- **Adopted (opt-in, default off):** `--aug-rotation-deg` and `--aug-time-warp` in `isl.train` (`isl/features.py`: `rotate_raw`, `time_warp_raw`; 9 tests).
- **Not adopted:** per-keypoint cutout (our features treat a hand as all-or-nothing, so zeroing 3 points would create an artifact; whole-hand dropout already exists).
- **Not yet evaluated:** whether either option helps. The Colab notebook runs a same-seed baseline vs augmentation pair; repeat with 3 seeds before concluding anything.

## Improvement plan, in order (each step needs measured evidence before the next)
1. Reproduce a baseline on the official split with our pipeline (notebook steps 11-14). Record top-1/top-3, per-class errors.
2. Run the augmentation ablation (step 15), 3 seeds each. Keep what helps beyond seed noise.
3. Compare against the AI4Bharat pretrained models and splits once `pretrained_links.json` and `train_test_paths/` are read (needs a licence/terms check on the weights).
4. Only if (1)-(2) fall short of target: try a small Transformer encoder or an ensemble of seeds; then an RGB/CNN feature branch, which is what the paper's best model used.
5. Evaluate on the session-proxy split and on a few consented webcam recordings to estimate real-world accuracy.

## Licence flag
Zenodo and the Hugging Face card state CC-BY-4.0 (checked 10 Nov 2026). An IEEE DataPort derivative describes INCLUDE as CC BY-NC-SA 4.0. Re-read the
licence line on the Zenodo page before relying on either; for a private non-commercial demo both are workable, but share-alike would affect publishing derived data.
