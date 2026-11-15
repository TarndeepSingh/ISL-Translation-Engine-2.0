"""ONNX predictor. Imports ONNX Runtime lazily so readiness can be per-model."""
import hashlib, json, logging, os
import numpy as np
from isl.config import T, FEATURE_DIM
from isl.features import resample


class Predictor:
    def __init__(self, path):
        import onnxruntime as ort
        labels_path = path.replace(".onnx", ".labels.json")
        with open(labels_path, encoding="utf-8") as stream:
            self.labels = json.load(stream)
        if (not isinstance(self.labels, list) or not self.labels or
                any(not isinstance(label, str) or not label.strip() for label in self.labels) or
                len(set(self.labels)) != len(self.labels)):
            raise ValueError("model labels must be a non-empty list of unique strings")
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inputs = self.sess.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "x":
            raise ValueError("ONNX input contract must contain one input named x")
        shape = inputs[0].shape
        if len(shape) != 3 or shape[-2:] != [T, FEATURE_DIM]:
            raise ValueError(f"ONNX input shape must be (batch,{T},{FEATURE_DIM})")
        outputs = self.sess.get_outputs()
        if len(outputs) != 1:
            raise ValueError("ONNX output contract must contain one logits output")
        digest = hashlib.sha256()
        onnx_digest = hashlib.sha256()
        for artifact in (path, labels_path):
            with open(artifact, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    if artifact == path:
                        onnx_digest.update(chunk)
        labels_digest = hashlib.sha256(json.dumps(self.labels, ensure_ascii=False,
                                                   separators=(",", ":")).encode("utf-8")).hexdigest()
        calibration_path = path.replace(".onnx", ".calibration.json")
        self.temperature = 1.0
        self.confidence_threshold = None
        self.score_semantics = "raw_softmax_uncalibrated"
        if os.path.exists(calibration_path):
            with open(calibration_path, encoding="utf-8") as stream:
                calibration = json.load(stream)
            if (calibration.get("schema_version") != "temperature-calibration-v1" or
                    calibration.get("onnx_sha256") != onnx_digest.hexdigest() or
                    calibration.get("labels_sha256") != labels_digest):
                raise ValueError("calibration sidecar does not match this ONNX model and label map")
            self.temperature = float(calibration.get("temperature", 0))
            if not np.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError("calibration temperature must be positive and finite")
            self.confidence_threshold = float(calibration.get("confidence_threshold", 1.000001))
            if not np.isfinite(self.confidence_threshold) or self.confidence_threshold < 0:
                raise ValueError("calibration confidence threshold must be finite and non-negative")
            self.score_semantics = "temperature_scaled_validation_calibrated"
            with open(calibration_path, "rb") as stream:
                digest.update(stream.read())
        self.model_version = f"isl-onnx-{digest.hexdigest()[:12]}"

    def predict(self, clip, k=3):
        """clip (L,170) -> (label, confidence, margin, top3)."""
        x = resample(clip, T)[None].astype(np.float32)
        if x.shape != (1, T, FEATURE_DIM) or not np.isfinite(x).all():
            raise ValueError("preprocessed feature tensor violates the model input contract")
        logits = self.sess.run(None, {"x": x})[0]
        if logits.shape != (1, len(self.labels)) or not np.isfinite(logits).all():
            raise ValueError("ONNX output violates the label/logit contract")
        lg = logits[0]
        scaled_logits = lg / self.temperature
        p = np.exp(scaled_logits - scaled_logits.max()); p /= p.sum()
        o = np.argsort(-p)
        top3 = [(self.labels[i], round(float(p[i]), 3)) for i in o[:k]]
        margin = float(p[o[0]] - p[o[1]]) if len(o) > 1 else float(p[o[0]])
        return self.labels[o[0]], float(p[o[0]]), margin, top3


def load_predictor(path):
    if os.path.exists(path) and os.path.exists(path.replace(".onnx", ".labels.json")):
        try:
            return Predictor(path)
        except Exception:
            logging.getLogger(__name__).exception("isolated model package could not be loaded")
            return None
    logging.getLogger(__name__).info("isolated model not found; readiness is not ready")
    return None
