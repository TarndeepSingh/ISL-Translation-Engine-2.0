"""Module 1: video -> raw landmark cache (75x3 per frame, NaN where missing).
python -m isl.extract --manifest data/manifest.csv --out cache --workers 4"""
import argparse, csv, hashlib, os
from multiprocessing import Pool
import cv2, numpy as np
from .config import N_RAW


def video_to_raw(path, max_w=640):
    from .landmarks import Landmarker
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lm, out, ts, i = Landmarker(), [], -1, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > max_w:
            fr = cv2.resize(fr, (max_w, int(h * max_w / w)))
        ts = max(ts + 1, int(i * 1000 / fps)); i += 1
        out.append(lm.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), ts))
    cap.release()
    return np.stack(out) if out else np.zeros((0, N_RAW, 3), np.float32)


def _job(a):
    path, label, split, out = a
    dst = os.path.join(out, hashlib.md5(path.encode()).hexdigest()[:16] + ".npz")
    try:
        if not os.path.exists(dst):
            raw = video_to_raw(path)
            if len(raw) < 5:
                return None
            np.savez_compressed(dst, raw=raw)
        return dst, label, split
    except Exception as e:
        print("FAILED", path, e); return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)   # csv: path,label,split
    ap.add_argument("--out", default="cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from .landmarks import ensure_models
    ensure_models()                                # download .task files once, before workers start
    os.makedirs(a.out, exist_ok=True)
    rows = [(r["path"], r["label"], r["split"], a.out) for r in csv.DictReader(open(a.manifest))]
    res = []
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(_job, rows), 1):
            if r: res.append(r)
            if k % 25 == 0: print(f"{k}/{len(rows)}", flush=True)
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cache", "label", "split"]); w.writerows(res)
    print(f"cached {len(res)}/{len(rows)} videos -> {a.out}/index.csv")
