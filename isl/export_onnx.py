"""python -m isl.export_onnx --ckpt runs/exp1/best.pt --out models/isl.onnx"""
import argparse, hashlib, json, os, numpy as np, torch
from .config import T, FEATURE_DIM
from .model import ISLNet

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="models/isl.onnx")
ap.add_argument("--calibration", default="", help="Validation calibration sidecar; defaults beside --ckpt")
ap.add_argument("--max-parity-error", type=float, default=1e-4,
                help="Maximum allowed absolute logit difference between PyTorch and ONNX")
a = ap.parse_args()
if a.max_parity_error <= 0:
    ap.error("--max-parity-error must be positive")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
calibration_target = a.out.replace(".onnx", ".calibration.json")
if os.path.exists(calibration_target):
    os.remove(calibration_target)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = ISLNet(n_classes=len(ck["labels"])); m.load_state_dict(ck["model"]); m.eval()
d = torch.randn(1, T, FEATURE_DIM)
kw = dict(input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17)
try:
    torch.onnx.export(m, d, a.out, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(m, d, a.out, **kw)
labels_path = a.out.replace(".onnx", ".labels.json")
with open(labels_path, "w", encoding="utf-8") as stream:
    json.dump(ck["labels"], stream, ensure_ascii=False, indent=2)
import onnxruntime as ort
s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
with torch.no_grad():
    expected = m(d).detach().numpy()
actual = s.run(None, {"x": d.numpy()})[0]
diff = float(np.abs(actual - expected).max())
print(f"exported {a.out} | max |torch-onnx| = {diff:.2e}")
if not np.isfinite(diff) or diff > a.max_parity_error:
    raise SystemExit(f"ONNX parity error {diff:.3e} exceeds tolerance {a.max_parity_error:.3e}")

calibration_source = a.calibration or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "calibration.json")
if os.path.isfile(calibration_source):
    with open(calibration_source, encoding="utf-8") as stream:
        calibration = json.load(stream)
    digest = hashlib.sha256()
    with open(a.ckpt, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    ckpt_hash = digest.hexdigest()
    labels_hash = hashlib.sha256(json.dumps(ck["labels"], ensure_ascii=False,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
    if calibration.get("checkpoint_sha256") != ckpt_hash or calibration.get("labels_sha256") != labels_hash:
        raise SystemExit("calibration sidecar does not match the selected checkpoint/labels")
    digest = hashlib.sha256()
    with open(a.out, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    calibration["onnx_sha256"] = digest.hexdigest()
    with open(calibration_target, "w", encoding="utf-8") as stream:
        json.dump(calibration, stream, ensure_ascii=False, indent=2)
