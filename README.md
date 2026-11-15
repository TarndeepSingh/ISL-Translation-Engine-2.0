---
title: ISL Translation Engine 2.0
emoji: 🤟
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: cc-by-4.0
---

# ISL Translation Engine 2.0 — Real-Time Indian Sign Language Recognition

An end-to-end **Indian Sign Language (ISL) Isolated Word Recognition System** built on the **AI4Bharat INCLUDE-50** benchmark (IIT Madras). The system captures live webcam video (or uploaded video clips), extracts anatomical hand and upper-body skeletal landmarks via **MediaPipe**, processes temporal sequences through a calibrated **Bidirectional LSTM (Bi-LSTM)** exported to **ONNX**, and delivers Top-3 word predictions with confidence scoring in a modern **React + FastAPI** web application.

---

## 1. Verified INCLUDE-50 Experimental Results

Models were trained and evaluated across three random seeds (`13, 14, 15`) on the **943 usable videos** of the **AI4Bharat INCLUDE-50** dataset (`balanced` split policy, 150 epochs):

| Experiment Variant | Augmentation Strategy | Top-1 Accuracy (3-Seed Mean) | Top-3 Accuracy (3-Seed Mean) |
| :--- | :--- | :---: | :---: |
| **Baseline** | None (Standard normalized sequences) | **96.3%** | **99.7%** |
| **`aug7-warp0.3` (Deployed)** | 7x spatial-temporal augmentation + 0.3 temporal warping | **97.6%** | **99.8%** |

*Deployed Production Artifact:* `include50-balanced-aug7-warp0.3-seed13` (`models/isl.onnx`, `models/isl.labels.json`, `models/isl.calibration.json`).

### Important Scope & Accuracy Limitations
* **Isolated Word Recognition Only (50 Classes):** This model classifies isolated single-sign video windows across the 50-word INCLUDE-50 vocabulary. It does **not** perform unrestricted continuous sentence translation.
* **Benchmark vs. Unconstrained Webcam Performance:** The **97.6% Top-1 accuracy** is measured on held-out INCLUDE-50 test clips recorded under consistent studio framing. Live webcam accuracy on unseen users depends on camera framing (upper body and both hands visible), lighting, signing speed, and background motion.

---

## 2. Dataset Acknowledgements & Licensing

* **Dataset:** **INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition** by **AI4Bharat & IIT Madras** (Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, Mitesh Khapra, *ACM Multimedia 2020*, DOI: `10.1145/3394171.3413528`), recorded in collaboration with **St. Louis School for the Deaf, Adyar, Chennai**.
* **Official Source & License:** [Zenodo Record 4010759](https://zenodo.org/records/4010759) — licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
* **Privacy Notice:** All webcam frames submitted for live inference are processed ephemerally in memory (RAM) to extract skeletal coordinates and are immediately discarded. No user video or images are stored on disk.