# Dataset Inventory and Access Register

**Audit date:** 10 November 2026  
**Local media/checkpoints:** none. The repository has only `.gitkeep` placeholders in `data/` and `models/`. All counts below are source-reported or source-listed, not local usable counts.

## Status vocabulary

- **Reported:** count/description appears in paper or supplied project report.
- **Listed:** current card/repository lists metadata or download references.
- **Downloaded:** media files are present locally and checksummed.
- **Decoded:** media passes stream and frame validation.
- **Rights reviewed:** current terms are recorded for this use.
- **Accepted:** annotation, rights and split checks pass for a model experiment.

No corpus below is downloaded, decoded or accepted in this workspace.

## Register

| Dataset | Task | Source facts | Terms/access state | Local state / next gate |
|---|---|---|---|---|
| AI4Bharat INCLUDE | Isolated ISL signs | INCLUDE card has label/path/include_50 metadata and points to Zenodo for actual media. Current card lists 3,816 train + 425 validation + 1,009 test metadata rows (5,250 total) and marks CC-BY-4.0. Older paper/project docs report 4,287/4,292 videos and 958 INCLUDE-50 items. | Card's license is recorded as CC-BY-4.0. Verify Zenodo record/files and exact media rights before model/derivative use. Card notes videos are from Chennai and ISL varies regionally. | No files. Reconcile metadata count, archive members, include_50 rows and official splits; download only after source/checksum/rights review. |
| iSign v1.1 | Continuous video/pose to English; additional NLP tasks | Paper reports 118,228 video-sentence/phrase pairs and SignVideo2Text/SignPose2Text tasks. | Current HF card says gated, research use only/non-commercial; it recommends grouping by parent video ID so all sequence rows remain together. | No account acceptance or files. Candidate research corpus after user-authorized account terms are accepted and exact version/split/rights are logged. |
| ISLTranslate | Continuous ISL-English sentence/phrase pairs | Paper reports 31,222 pairs; educational/textbook domain; published train/validation/test split is 24,978/3,122/3,122. Paper describes pose-to-text baseline and notes annotation/noise constraints. | The paper describes source material and research use but does not by itself establish current download, redistribution or derived-model license terms. | No files. Confirm official release, rights, source video groups, references and access before ingestion. |
| CISLR | Isolated word/gloss recognition | iSign paper reports 4,765 word classes, 7,050 videos and 71 signers; average about 1.5 examples/class. | Current release/access/terms not checked during this audit. | No files. Possible representation research; not sufficient alone for a dense classifier; no label merge with INCLUDE. |
| FDMSE-ISL | Candidate isolated corpus | Listed in supplied specification only; no authoritative current release was inspected. | Unknown | Do not ingest until source, contents, consent and rights are verified. |
| ISH-News / public web videos | Candidate continuous video | Listed in supplied docs; research papers mention public-source video. | Public availability is not consent/license. No scraping authorized or performed. | Excluded until an official permitted release or direct rights are verified. |
| ISH-NEWS 2026 paper dataset | Continuous news-domain video to English | Paper reports sentence annotations and timestamps over broadcast sources, with a small set of professional interpreters. | Paper states raw videos cannot be redistributed and the article uses CC BY-NC-ND; this does not establish rights to train/deploy on the source videos. | Not selected without direct permission from the rights holder. |

## Source terms re-checked 10 Nov 2026 (primary pages fetched)

| Dataset | What the primary source shows | Drive storage | Colab processing |
|---|---|---|---|
| INCLUDE | HF card and Zenodo record 4010759 both state CC-BY-4.0. Zenodo: open access, 56.8 GB in per-category zip parts with MD5s, 4,292 videos, INCLUDE-50 = 766 train + 192 test = 958, train/test split only. HF card: metadata only (5,250 rows, train 3.82k / val 425 / test 1.01k), videos on Zenodo; asks users to respect the privacy and dignity of the people filmed. | Permitted by the license (attribution required). Recommended anyway: keep raw videos on `/content`, put only manifests/reports/runs/checkpoints in Drive. | Permitted by the license. Attribute AI4Bharat/IIT Madras; CC licenses do not waive privacy/personality rights of the people filmed. Not legal advice. |
| iSign v1.1 | HF page: gated, license cc-by-nc-sa-4.0, "free for research use but not for commercial purposes". File listing: videos 32.2 + 25.7 GB, poses 3 x 48.3 + 25.3 GB = 170.2 GB, total 228 GB, as concatenable `part_*` files. The access-agreement text is only shown after login and was NOT read. | **Do not.** Project rule, and the agreement is unread. | **Unverified.** Non-commercial research processing is consistent with the public license, but confirm the agreement text (you must log in) before use. NC-SA share-alike may reach derived data; whether it reaches model weights is unsettled, so keep checkpoints private until confirmed. |
| ISLTranslate | Not re-checked this session. | Unconfirmed | Unconfirmed |

Findings that change the plan:

- **INCLUDE counts disagree across sources** (Zenodo 4,292 videos vs HF 5,250 metadata rows, plus a validation split Zenodo does not have). The HF preview shows rows whose `label` disagrees with the `video_path` folder, e.g. one video listed under two labels. `python -m ml.datasets.include_manifest` audits and excludes these; run it and record the real counts here.
- **iSign cannot be staged whole.** 228 GB exceeds Colab local disk and must not go to Drive. A subset strategy (random-access reads of selected members from the multi-part archive, after access terms are confirmed) is required; the pose files use the `pose-format` layout, not this repository's 170-value MediaPipe features, so a conversion step is also needed. No iSign adapter or conversion has been run.

## Current research/demo recommendation

Use INCLUDE-50 first for isolated recognition after reconciling the current metadata, official split, video archive, and exact license. For a continuous research baseline, iSign is the strongest candidate only after accepting its gated research/non-commercial terms. Its current dataset card prohibits re-uploading; keep iSign media and pose files out of Google Drive and GitHub. The user's stated target is a non-commercial research/demo deployment. If those access terms do not permit Colab processing or saving the resulting model, use consented recordings or obtain written permission instead. ISLTranslate remains an alternative only after its video and derivative-model terms are confirmed.

## Provenance and split rules

Every row in a future manifest must carry stable internal sample ID, source sample ID, source video ID, dataset/version, task, original label/reference, SHA-256, path/URI, license/rights state, source split, project split and split policy. Preserve original translation text and store normalized text separately. Group all segments from a continuous source video together. Use signer grouping only when the source supplies real signer identity; do not turn filename-based session heuristics into signer IDs. Keep each dataset's labels and references separate until equivalence and rights are reviewed.

## Primary source links

- [INCLUDE Hugging Face dataset card](https://huggingface.co/datasets/ai4bharat/INCLUDE) — metadata/video-path explanation, license marker, INCLUDE-50 field and regional coverage note.
- [INCLUDE ACM paper](https://doi.org/10.1145/3394171.3413528) and [Zenodo record 4010759](https://zenodo.org/records/4010759) — paper/archive provenance; archive must be reconciled to current metadata before counting usable media.
- [iSign ACL paper](https://arxiv.org/abs/2407.05404) and [current iSign dataset card](https://huggingface.co/datasets/Exploration-Lab/iSign) — task descriptions, access and video-ID grouping.
- [ISLTranslate ACL paper](https://aclanthology.org/2023.findings-acl.665/) — reported pairs/splits and dataset methodology; not a substitute for current release terms.
- [CISLR ACL paper](https://aclanthology.org/2022.emnlp-main.707/) — reported isolated-recognition corpus.

## Known count disagreement

The supplied SRS/report say 943 matched INCLUDE-50 videos versus 958 in older benchmark figures. The current HF card lists 5,250 metadata records across train/validation/test, while pointing to actual media archives on Zenodo. These may reflect different release versions, records, labels or matching logic. None is a local count. The next data task is an exact diff of current metadata IDs, include_50 flags, Zenodo members, source split IDs and decoded files; until that audit is run, do not claim a supported training set size.

## Measured INCLUDE metadata audit (user's Colab run, 11 Nov 2026)

Source: `python -m ml.datasets.include_manifest` reading `ai4bharat/INCLUDE` from the Hugging Face Hub (metadata only; no video downloaded).

| Quantity | Measured |
|---|---|
| Metadata rows (all classes) | 5,250 |
| Unique `video_path` values | 4,257 |
| Rows whose label disagrees with the path folder | 49 (47 paths with conflicting labels) |
| INCLUDE-50 rows (`include_50` true) | 958 |
| INCLUDE-50 unique video paths | 943 |
| INCLUDE-50 mislabeled duplicate rows (excluded) | 14 |
| INCLUDE-50 exact duplicate rows (collapsed) | 1 |
| INCLUDE-50 rows after cleaning | 943, 50 classes, 14-21 videos per class |
| Official-split rows after cleaning (HF splits) | train 675, val 77, test 191 |
| Keep-list entries for the downloader | 943 |

**Reconciliation of the earlier disagreement.** 958 - 14 - 1 = 943: the "958" figure counts metadata *rows*, and "943 matched videos" (the earlier project report) is the number of *unique, label-consistent* videos. Train+val (675+77 = 752) and test (191) are consistent with Zenodo's 766/192 official split minus the same 15 rows (766-14 = 752, 192-1 = 191). That last step is an inference from the totals, not a per-row check. Zenodo's 4,292 total vs 4,257 unique paths here (35 apart) is not explained and no media has been counted yet.

**Consequences.** The test set is 191 videos (about 3.8 per class), so one video is about 0.5 percentage points and a measured 90% has a 95% binomial interval of roughly +/-4.3 points. The validation set is 77 videos (about 1.5 per class), which is small for early stopping and temperature calibration. The official split is video-level; group ids are per video, so `groups_in_multiple_splits = 0` says nothing about shared signers or sessions.
