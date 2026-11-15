"""Module 3: local webcam demo (OpenCV window). python -m isl.live --model models/isl.onnx"""
import argparse, json, time
from collections import deque
import cv2, numpy as np, onnxruntime as ort
from .config import IDLE_LABEL
from .features import normalize, resample
from .landmarks import Landmarker
from .spotter import SignSegmenter

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/isl.onnx"); ap.add_argument("--conf", type=float, default=0.70)
ap.add_argument("--margin", type=float, default=0.20); ap.add_argument("--cam", type=int, default=0)
a = ap.parse_args()
labels = json.load(open(a.model.replace(".onnx", ".labels.json")))
sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
lm, seg, words, last_pose, status, ts = Landmarker(), SignSegmenter(), deque(maxlen=6), None, "", 0
cap = cv2.VideoCapture(a.cam)
while True:
    ok, fr = cap.read()
    if not ok: break
    ts = max(ts + 1, int(time.time() * 1000))
    raw = lm.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), ts)[None]      # model sees UN-mirrored frame
    if not np.isfinite(raw[0, [11, 12]]).all() and last_pose is not None:
        raw[0, :33] = last_pose
    else:
        last_pose = raw[0, :33].copy()
    clip = seg.update(normalize(raw)[0])
    if clip is not None:
        lg = sess.run(None, {"x": resample(clip)[None]})[0][0]
        p = np.exp(lg - lg.max()); p /= p.sum(); o = np.argsort(-p)
        top, conf, marg = labels[o[0]], p[o[0]], p[o[0]] - p[o[1]]
        ok_ = top != IDLE_LABEL and conf >= a.conf and marg >= a.margin
        status = f"{top} {conf:.2f}" + ("" if ok_ else " (rejected)")
        if ok_: words.append(top)
    disp = cv2.flip(fr, 1)                                              # mirror for display only
    cv2.putText(disp, ("SIGNING" if seg.active else "idle") + f" v={seg.speed:.3f}", (10, 25), 0, .6, (0, 255, 255), 2)
    cv2.putText(disp, "last: " + status, (10, 55), 0, .6, (200, 200, 200), 1)
    cv2.putText(disp, " ".join(words), (10, disp.shape[0] - 20), 0, 1.0, (0, 255, 0), 2)
    cv2.imshow("ISL (Q to quit)", disp)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
