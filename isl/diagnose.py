"""STEP-0 DIAGNOSTIC. Does the trained model work on videos of KNOWN label, through the same code path as the live app?
  python -m isl.diagnose --model models/isl.onnx --manifest data/manifest.csv --split test --limit 40
  python -m isl.diagnose --model models/isl.onnx --root data/custom_test      (folders named after the word)
For every video it prints two predictions:
  OFFLINE = whole clip -> resample -> model          (how the model was trained/tested)
  LIVE    = frame by frame -> segmenter -> model     (how the web app works)
Reading the result:
  OFFLINE good, LIVE bad   -> the segmenter / live path is the problem (tune isl/spotter.py)
  OFFLINE bad on your OWN videos but good on INCLUDE test videos -> signer/camera domain gap (need your own data)
  OFFLINE bad even on INCLUDE test videos -> bug in extraction/labels/training (not a data-quantity problem)"""
import argparse, csv, os, random, re
from collections import defaultdict
import numpy as np
from .features import normalize
from .spotter import SignSegmenter

clean = lambda n: re.sub(r"^\d+\.\s*", "", n).strip().lower()
EXT = (".mov", ".mp4", ".avi", ".mkv")


def offline_predict(raw, predict):
    return predict(normalize(raw))


def live_predict(raw, predict, seg=None):
    """Replays frames exactly like backend/main.py Session.step. Returns one prediction per detected sign."""
    seg, out, last = seg or SignSegmenter(), [], None
    for i in range(len(raw)):
        r = raw[i:i + 1].copy()
        if not np.isfinite(r[0, [11, 12]]).all() and last is not None:
            r[0, :33] = last
        else:
            last = r[0, :33].copy()
        clip = seg.update(normalize(r)[0])
        if clip is not None:
            out.append(predict(clip))
    return out


def summarize(rows):
    n = len(rows)
    off = sum(r["off"][0] == r["label"] for r in rows)
    found = sum(len(r["live"]) > 0 for r in rows)
    live = sum(any(p[0] == r["label"] for p in r["live"][:1]) for r in rows)
    return {"videos": n, "offline_correct": off, "live_found_sign": found, "live_first_correct": live}


def collect(a):
    items = []
    if a.manifest:
        for r in csv.DictReader(open(a.manifest)):
            if r["split"] == a.split and r["label"] != "idle" and os.path.exists(r["path"]):
                items.append((r["path"], r["label"]))
    else:
        for d, _, fs in os.walk(a.root):
            for f in fs:
                if f.lower().endswith(EXT):
                    items.append((os.path.join(d, f), clean(os.path.basename(d))))
    by = defaultdict(list)
    for it in items: by[it[1]].append(it)
    rng = random.Random(0); out = []
    for v in by.values():
        rng.shuffle(v); out += v[:a.per_class]
    rng.shuffle(out)
    return out[:a.limit]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/isl.onnx"); ap.add_argument("--manifest")
    ap.add_argument("--split", default="test"); ap.add_argument("--root")
    ap.add_argument("--per-class", type=int, default=2); ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    from backend.engine import Predictor
    from .extract import video_to_raw
    P = Predictor(a.model); predict = P.predict
    rows = []
    for path, label in collect(a):
        raw = video_to_raw(path)
        if len(raw) < 5: continue
        r = {"path": path, "label": label, "off": offline_predict(raw, predict), "live": live_predict(raw, predict)}
        rows.append(r)
        lv = ", ".join(f"{p[0]} {p[1]:.2f}" for p in r["live"]) or "NO SIGN DETECTED"
        print(f"{'OK ' if r['off'][0] == label else 'BAD'} true={label:<14} offline={r['off'][0]} {r['off'][1]:.2f} | live: {lv}", flush=True)
    s = summarize(rows)
    print("\n", s)
    print(f"OFFLINE accuracy {s['offline_correct']}/{s['videos']} | LIVE found a sign in {s['live_found_sign']}/{s['videos']} | LIVE first prediction correct {s['live_first_correct']}/{s['videos']}")
