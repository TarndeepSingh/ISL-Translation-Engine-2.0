"""Word-recognition pipeline: video file -> landmarks -> features -> model. NO fallback predictions:
without a trained model the caller gets an error, never a made-up answer."""
import time
import numpy as np
from isl.config import LH_SLICE, RH_SLICE
from isl.features import normalize

IDLE = "idle"


def default_extractor(path):
    from isl.extract import video_to_raw           # needs opencv + mediapipe (same Tasks models as training)
    return video_to_raw(path)


def _duration(path, frames):
    try:
        import cv2
        c = cv2.VideoCapture(path); fps = c.get(cv2.CAP_PROP_FPS) or 30.0; c.release()
        return frames / fps
    except Exception:
        return frames / 30.0


class WordPipeline:
    def __init__(self, predictor, extractor=None, tau=None, margin=0.0):
        if predictor is None:
            raise RuntimeError("no trained model loaded")
        self.predictor, self.extract = predictor, extractor or default_extractor
        self.tau = tau if tau is not None else getattr(predictor, "confidence_threshold", None)
        self.margin = margin

    def predict_video(self, path):
        t0 = time.perf_counter()
        raw = self.extract(path)
        if raw.ndim != 3 or raw.shape[1:] != (75, 3) or len(raw) < 5:
            raise ValueError("video has fewer than 5 decodable frames")
        if np.isinf(raw).any():
            raise ValueError("video landmark extraction produced invalid values")
        label, conf, marg, top = self.predictor.predict(normalize(raw), k=5)
        q = {"frames": int(len(raw)),
             "shoulders_visible": float(np.isfinite(raw[:, [11, 12]]).all(axis=(1, 2)).mean()),
             "left_hand_visible": float(np.isfinite(raw[:, LH_SLICE]).all(axis=(1, 2)).mean()),
             "right_hand_visible": float(np.isfinite(raw[:, RH_SLICE]).all(axis=(1, 2)).mean())}
        if q["shoulders_visible"] < 0.8 or max(q["left_hand_visible"], q["right_hand_visible"]) < 0.2:
            status = "poor_input"
        elif label == IDLE:
            status = "no_sign"
        elif getattr(self.predictor, "score_semantics", "raw_softmax_uncalibrated") != "temperature_scaled_validation_calibrated":
            status = "uncalibrated"
        elif self.tau is None or conf < self.tau or marg < self.margin:
            status = "uncertain"
        else:
            status = "confident"
        return {"label": label, "conf": round(conf, 4), "score_semantics": getattr(self.predictor, "score_semantics", "raw_softmax_uncalibrated"),
                "margin": round(marg, 4), "top": [[l, p] for l, p in top],
                "status": status, "quality": q, "duration_s": round(_duration(path, len(raw)), 3),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "model_version": getattr(self.predictor, "model_version", "legacy-unversioned")}
