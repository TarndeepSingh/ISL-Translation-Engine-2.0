"""Per-class accuracy of ONE exported model on the train, validation AND test videos.

Why: ``isl.train`` saves predictions only for the TEST split, so the gap between "what the model memorised" and
"what it generalises to" per word cannot be read from the run folders. This script re-scores every video of the
index with the exported ONNX model (the exact file the website serves), using the same preprocessing as training
and validation/test (``resample(normalize(raw))``, no augmentation, no noise).

Needs only numpy + onnxruntime (no torch). Run where the landmark cache lives (Colab / Drive):

  python -m ml.evaluation.split_report --index <DRIVE>/data/processed/include50-balanced/index.csv \
      --onnx models/isl.onnx --labels models/isl.labels.json --out reports/split_eval.json

Important: use the ONNX of the run whose split you pass (the deployed model was trained on the 'balanced' index).
Training-split accuracy is measured WITHOUT augmentation, so it is the accuracy on the clean training videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from isl.features import normalize, resample

SPLITS = ("train", "val", "test")


def resolve_cache(rows: list[dict], index_path: str) -> list[str]:
    """Same lookup order as isl.train.resolve_cache_paths (kept here so torch is not imported)."""
    base = os.path.dirname(os.path.abspath(index_path))
    missing = []
    for r in rows:
        cands = []
        if r.get("cache_file"):
            cands.append(os.path.join(base, r["cache_file"]))
        rec = r.get("cache", "")
        if rec:
            cands.append(rec if os.path.isabs(rec) else os.path.join(base, rec))
        found = next((os.path.abspath(c) for c in cands if os.path.isfile(c)), None)
        if found is None:
            missing.append(r.get("sample_id", "?"))
        else:
            r["cache"] = found
    return missing


def score(rows: list[dict], labels: list[str], session) -> dict:
    l2i = {l: i for i, l in enumerate(labels)}
    inp = session.get_inputs()[0].name
    out: dict = {}
    for split in SPLITS:
        items = [r for r in rows if r["split"] == split]
        n = np.zeros(len(labels), int)
        k = np.zeros(len(labels), int)
        for r in items:
            y = l2i[r["label"]]
            x = resample(normalize(np.load(r["cache"])["raw"]))[None].astype(np.float32)
            p = int(np.argmax(session.run(None, {inp: x})[0][0]))
            n[y] += 1
            k[y] += int(p == y)
        acc = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        out[split] = {
            "n": int(n.sum()), "correct": int(k.sum()),
            "overall_accuracy": float(k.sum() / max(1, n.sum())),
            "average_class_accuracy": float(np.nanmean(acc)) if (n > 0).any() else None,
            "per_class": [{"id": i, "raw": labels[i], "support": int(n[i]), "correct": int(k[i]),
                           "accuracy": None if n[i] == 0 else float(k[i] / n[i])} for i in range(len(labels))],
        }
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="reports/split_eval.json")
    a = ap.parse_args(argv)

    import onnxruntime as ort
    labels = json.loads(Path(a.labels).read_text(encoding="utf-8"))
    with open(a.index, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    bad = [r["label"] for r in rows if r["label"] not in labels]
    if bad:
        raise SystemExit(f"{len(bad)} index rows have labels that are not in {a.labels} (first: {bad[0]!r}) - wrong label file?")
    missing = resolve_cache(rows, a.index)
    if missing:
        raise SystemExit(f"cache file missing for {len(missing)} samples (first: {missing[0]})")
    session = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    splits = score(rows, labels, session)
    report = {"schema_version": "isl-splits-v1", "onnx": Path(a.onnx).name, "index": Path(a.index).name,
              "labels": labels, "splits": splits,
              "note": "Training accuracy is measured on clean (un-augmented) training videos."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in SPLITS:
        d = splits[s]
        print(f"{s:5s}: {d['correct']}/{d['n']} = {100 * d['overall_accuracy']:.2f}%   average class {100 * d['average_class_accuracy']:.2f}%")
    print("wrote", a.out)
    return report


if __name__ == "__main__":
    main()
