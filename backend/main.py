"""
FastAPI Application Entrypoint — ISL Translation Engine 2.0
Uses the exact isl.features (assign_hands, normalize, resample -> (1, 32, 170))
and isl.landmarks.Landmarker pipeline from training, and prevents the 0.05
temperature floor from saturating Softmax confidence to 100%.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from isl.config import FEATURE_DIM, LH_SLICE, N_RAW, POSE_SLICE, RH_SLICE, T
from isl.features import assign_hands, normalize, resample

logger = logging.getLogger("isl.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("ISL_MODEL_DIR", str(PROJECT_ROOT / "models")))
ONNX_PATH = MODEL_DIR / "isl.onnx"
LABELS_PATH = MODEL_DIR / "isl.labels.json"
CALIB_PATH = MODEL_DIR / "isl.calibration.json"
EVAL_PATH = MODEL_DIR / "isl.eval.json"  # written by: python -m ml.evaluation.class_report
WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"
LEGACY_FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _load_labels(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict) and "labels" in raw and isinstance(raw["labels"], list):
        return [str(x) for x in raw["labels"]]
    return []


def _load_calibration(path: Path) -> Dict[str, Any]:
    default = {"temperature": 1.0, "confidence_threshold": 0.45, "top_k": 3}
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {**default, **data}
    except Exception as exc:
        logger.warning("Failed to parse calibration JSON (%s): %s", path, exc)
    return default


class UnifiedONNXPredictor:
    def __init__(self) -> None:
        self.session = None
        self.input_name: str = "x"
        self.labels: List[str] = []
        self.calibration: Dict[str, Any] = {}
        self.landmarker: Any = None
        self.last_error: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        self.labels = _load_labels(LABELS_PATH)
        self.calibration = _load_calibration(CALIB_PATH)

        if not ONNX_PATH.exists():
            self.session = None
            self.last_error = f"Missing {ONNX_PATH}. Copy isl.onnx, isl.labels.json, and isl.calibration.json into {MODEL_DIR}."
            logger.warning(self.last_error)
            return

        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(ONNX_PATH),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.last_error = None
            logger.info(
                "Loaded ONNX model %s | input=%s shape=(1, %d, %d) | labels=%d",
                ONNX_PATH.name,
                self.input_name,
                T,
                FEATURE_DIM,
                len(self.labels),
            )
        except Exception as exc:
            self.session = None
            self.last_error = f"Failed to initialize ONNX Runtime: {exc}"
            logger.error(self.last_error)

    @property
    def is_ready(self) -> bool:
        if not self.session or not self.labels:
            self.reload()
        return self.session is not None and len(self.labels) > 0

    def _get_server_landmarker(self):
        if self.landmarker is None:
            from isl.landmarks import Landmarker
            self.landmarker = Landmarker()
        return self.landmarker

    def _raw_from_browser_landmarks(self, landmarks_seq: List[Dict[str, Any]]) -> np.ndarray:
        """
        Builds exact (L, 75, 3) raw array with np.nan for missing landmarks
        and assigns hands using isl.features.assign_hands(pose, hands) —
        100% identical to isl/landmarks.py!
        """
        raw_list: List[np.ndarray] = []

        for snap in landmarks_seq:
            raw_frame = np.full((N_RAW, 3), np.nan, dtype=np.float32)

            pose_arr: Optional[np.ndarray] = None
            raw_pose = snap.get("pose")
            if raw_pose and len(raw_pose) >= 33:
                pose_arr = np.array([[p[0], p[1], p[2]] for p in raw_pose[:33]], dtype=np.float32)
                if np.isfinite(pose_arr).all():
                    raw_frame[POSE_SLICE] = pose_arr
                else:
                    pose_arr = None

            # Use raw detected hands list if provided, or left_hand/right_hand fallback
            hands_list: List[np.ndarray] = []
            raw_hands = snap.get("hands")
            if isinstance(raw_hands, list) and len(raw_hands) > 0:
                for h in raw_hands[:2]:
                    if h and len(h) == 21:
                        h_arr = np.array([[pt[0], pt[1], pt[2]] for pt in h], dtype=np.float32)
                        if np.isfinite(h_arr).all() and np.any(np.abs(h_arr) > 1e-5):
                            hands_list.append(h_arr)
            else:
                for key in ("left_hand", "right_hand"):
                    h = snap.get(key)
                    if h and len(h) == 21:
                        h_arr = np.array([[pt[0], pt[1], pt[2]] for pt in h], dtype=np.float32)
                        if np.isfinite(h_arr).all() and np.any(np.abs(h_arr) > 1e-5):
                            hands_list.append(h_arr)

            left, right = assign_hands(pose_arr, hands_list)
            if left is not None:
                raw_frame[LH_SLICE] = left
            if right is not None:
                raw_frame[RH_SLICE] = right

            raw_list.append(raw_frame)

        return np.stack(raw_list, axis=0)

    def _raw_from_bgr_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Extracts (L, 75, 3) using isl.landmarks.Landmarker on un-mirrored RGB frames.
        """
        lmk = self._get_server_landmarker()
        raw_list: List[np.ndarray] = []
        base_ts = int(time.monotonic() * 1000)

        for idx, bgr in enumerate(frames_bgr):
            h, w = bgr.shape[:2]
            if w > 640:
                scale = 640.0 / w
                bgr = cv2.resize(bgr, (640, int(round(h * scale))))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raw_frame = lmk.process(rgb, base_ts + idx * 33)
            raw_list.append(raw_frame)

        return np.stack(raw_list, axis=0)

    @staticmethod
    def _trim_idle_edges(raw: np.ndarray) -> np.ndarray:
        """
        Trims leading/trailing frames where neither hand is visible so the 32
        resampled timesteps focus on the active sign gesture.
        """
        lh_ok = np.isfinite(raw[:, LH_SLICE]).all(axis=(1, 2))
        rh_ok = np.isfinite(raw[:, RH_SLICE]).all(axis=(1, 2))
        active = lh_ok | rh_ok
        active_indices = np.where(active)[0]
        if len(active_indices) >= 4:
            start = max(0, int(active_indices[0]) - 1)
            end = min(len(raw), int(active_indices[-1]) + 2)
            return raw[start:end]
        return raw

    def predict(
        self,
        frames_bgr: List[np.ndarray],
        landmarks_seq: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.is_ready or self.session is None:
            raise HTTPException(
                status_code=503,
                detail=self.last_error or f"Model files missing in {MODEL_DIR}.",
            )

        t0 = time.perf_counter()

        # 1. Build raw (L, 75, 3) using browser landmarks or server Landmarker
        raw: Optional[np.ndarray] = None
        if landmarks_seq and len(landmarks_seq) >= 4:
            raw = self._raw_from_browser_landmarks(landmarks_seq)
            # Check if browser landmarks actually captured any hands; if not and frames exist, try server Landmarker
            lh_ok = np.isfinite(raw[:, LH_SLICE]).all(axis=(1, 2))
            rh_ok = np.isfinite(raw[:, RH_SLICE]).all(axis=(1, 2))
            if not np.any(lh_ok | rh_ok) and len(frames_bgr) >= 4:
                try:
                    raw = self._raw_from_bgr_frames(frames_bgr)
                except Exception as exc:
                    logger.warning("Server Landmarker fallback note: %s", exc)

        if raw is None:
            if len(frames_bgr) < 4:
                raise HTTPException(status_code=400, detail="Minimum 4 video frames required.")
            raw = self._raw_from_bgr_frames(frames_bgr)

        # 2. Check if hands were actually visible in the clip
        lh_present = np.isfinite(raw[:, LH_SLICE]).all(axis=(1, 2))
        rh_present = np.isfinite(raw[:, RH_SLICE]).all(axis=(1, 2))
        hand_frames = int(np.sum(lh_present | rh_present))
        total_frames = len(raw)

        if hand_frames < 2:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "status": "no_hands",
                "predicted_word": "No Hands Detected — Raise Hands in Frame",
                "raw_top1_word": "None",
                "confidence": 0.0,
                "threshold": 0.45,
                "is_confident": False,
                "motion_energy": 0.0,
                "top_predictions": [
                    {"rank": 1, "class_id": -1, "label": "Raise hands to sign", "raw_label": "", "confidence": 0.0},
                    {"rank": 2, "class_id": -1, "label": "—", "raw_label": "", "confidence": 0.0},
                    {"rank": 3, "class_id": -1, "label": "—", "raw_label": "", "confidence": 0.0},
                ],
                "frames_processed": total_frames,
                "latency_ms": latency_ms,
                "model_variant": "include50-balanced-aug7-warp0.3",
                "temperature": 1.0,
            }

        # 3. Trim idle start/end frames and run exact training normalize() + resample()
        feats = normalize(raw)  # (L, 170) - full trajectory matching training
        x = resample(feats, T)[None].astype(np.float32)  # (1, 32, 170)

        motion_energy = float(np.mean(np.abs(np.diff(x[0, :, :162], axis=0))))

        # 4. Run ONNX Runtime
        logits = self.session.run(None, {self.input_name: x})[0][0]
        logits = np.asarray(logits, dtype=np.float64)

        # IMPORTANT FIX: isl.calibration.json has temperature=0.05 which multiplies logits by 20x
        # and forces 100.0% confidence on everything. We clamp temperature to >= 1.0 for realistic probabilities!
        raw_temp = float(self.calibration.get("temperature", 1.0) or 1.0)
        effective_temp = max(raw_temp, 1.0)

        scaled = logits / effective_temp
        scaled -= np.max(scaled)
        exp_scores = np.exp(scaled)
        probs = exp_scores / np.sum(exp_scores)

        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            label = self.labels[idx] if idx < len(self.labels) else f"Class {idx}"
            clean_label = (
                label.split(".", 1)[-1].strip()
                if "." in label and label.split(".", 1)[0].strip().isdigit()
                else label
            )
            top_predictions.append(
                {
                    "rank": rank,
                    "class_id": int(idx),
                    "label": clean_label,
                    "raw_label": label,
                    "confidence": round(float(probs[idx]), 4),
                }
            )

        # Use a realistic confidence threshold (0.45) instead of the 0.999999999999651 artifact from T=0.05
        threshold = 0.45
        best = top_predictions[0]
        is_confident = best["confidence"] >= threshold and hand_frames >= 3

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "status": "ok" if is_confident else "low_confidence",
            "predicted_word": best["label"] if is_confident else f"{best['label']} (Low Confidence)",
            "raw_top1_word": best["label"],
            "confidence": best["confidence"],
            "threshold": threshold,
            "is_confident": is_confident,
            "motion_energy": round(motion_energy, 5),
            "top_predictions": top_predictions,
            "frames_processed": total_frames,
            "latency_ms": latency_ms,
            "model_variant": "include50-balanced-aug7-warp0.3",
            "temperature": round(effective_temp, 2),
        }


predictor = UnifiedONNXPredictor()

app = FastAPI(
    title="ISL Translation Engine 2.0 API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameBatchRequest(BaseModel):
    frames: List[str] = Field(default_factory=list)
    landmarks: Optional[List[Dict[str, Any]]] = Field(default=None)


@app.get("/health")
@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    ready = predictor.is_ready
    return {
        "status": "healthy" if ready else "degraded",
        "model_loaded": ready,
        "vocabulary_size": len(predictor.labels),
        "onnx_path_exists": ONNX_PATH.exists(),
        "labels_path_exists": LABELS_PATH.exists(),
        "calibration_path_exists": CALIB_PATH.exists(),
        "error": predictor.last_error,
    }


@app.get("/api/v1/meta")
@app.get("/api/ml/model-info")
def get_model_metadata() -> Dict[str, Any]:
    _ = predictor.is_ready
    clean_vocab = []
    for i, lbl in enumerate(predictor.labels):
        clean_lbl = lbl.split(".", 1)[-1].strip() if "." in lbl and lbl.split(".", 1)[0].strip().isdigit() else lbl
        clean_vocab.append({"id": i, "word": clean_lbl, "raw": lbl})

    return {
        "project": "ISL Translation Engine 2.0",
        "dataset": "AI4Bharat INCLUDE-50 (IIT Madras)",
        "dataset_license": "CC BY 4.0 (Zenodo Record 4010759)",
        "model_Loaded": predictor.is_ready,
        "architecture": "MediaPipe Landmarks + Bi-LSTM (PyTorch -> ONNX)",
        "experiment_run": "include50-balanced-aug7-warp0.3-seed13",
        "benchmarks": {
            "split_policy": "balanced (943 usable INCLUDE-50 videos, 3 seeds: 13, 14, 15)",
            "variants": [
                {"name": "Baseline Bi-LSTM", "top1_mean": 96.3, "top3_mean": 99.7},
                {"name": "Augmented (aug7-warp0.3)", "top1_mean": 97.6, "top3_mean": 99.8},
            ],
            "note": "Reported 97.6% Top-1 mean is evaluated on held-out INCLUDE-50 studio clips.",
        },
        "input_shape": [1, T, FEATURE_DIM],
        "calibration": predictor.calibration,
        "vocabulary_count": len(clean_vocab),
        "vocabulary": clean_vocab,
    }


@app.get("/api/v1/evaluation")
def get_evaluation() -> Dict[str, Any]:
    """Held-out test results: per-class accuracy, overall/average accuracy, confusion matrix.

    Served from models/isl.eval.json (created offline by ml.evaluation.class_report). Nothing is computed
    here, so the endpoint is cheap and the numbers cannot drift from what was actually evaluated.
    """
    if not EVAL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation report yet. Run `python -m ml.evaluation.class_report` and copy the "
                   f"result to {EVAL_PATH}.",
        )
    try:
        report = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Could not read %s: %s", EVAL_PATH, exc)
        raise HTTPException(status_code=500, detail="Evaluation report is unreadable.")

    # Guard against showing numbers that belong to a different model: the class list in the report
    # must be exactly the class list the loaded ONNX model uses (same names, same order).
    _ = predictor.is_ready
    report_labels = [item.get("raw") for item in report.get("labels", [])]
    report["labels_match_deployed_model"] = (report_labels == predictor.labels) if predictor.labels else None
    return report


@app.post("/api/v1/predict-frames")
async def predict_webcam_frames(payload: FrameBatchRequest) -> Dict[str, Any]:
    decoded_frames: List[np.ndarray] = []
    for b64_str in (payload.frames or [])[:64]:
        raw_b64 = b64_str.split(",", 1)[-1] if "," in b64_str else b64_str
        try:
            img_bytes = base64.b64decode(raw_b64)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                decoded_frames.append(bgr)
        except Exception:
            continue

    if len(decoded_frames) < 4 and not payload.landmarks:
        raise HTTPException(status_code=400, detail="Could not decode enough valid video frames.")

    return predictor.predict(decoded_frames, payload.landmarks)


@app.post("/api/v1/predict-video")
async def predict_uploaded_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        frames: List[np.ndarray] = []
        while cap.isOpened() and len(frames) < 150:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 4:
            raise HTTPException(status_code=400, detail="Uploaded video could not be decoded.")

        return predictor.predict(frames, None)
    finally:
        tmp_path.unlink(missing_ok=True)


if (WEB_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST_DIR / "assets")), name="assets")


@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(request: Request, full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "API endpoint not found"}, status_code=404)

    if WEB_DIST_DIR.exists():
        candidate = WEB_DIST_DIR / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        index_html = WEB_DIST_DIR / "index.html"
        if index_html.exists():
            return FileResponse(str(index_html))

    if (LEGACY_FRONTEND_DIR / "index.html").exists():
        return FileResponse(str(LEGACY_FRONTEND_DIR / "index.html"))

    return HTMLResponse("<h2>ISL Translation Engine 2.0 Backend is Running</h2>")