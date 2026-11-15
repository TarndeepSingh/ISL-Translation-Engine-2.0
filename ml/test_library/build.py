"""Build test_videos/ from HELD-OUT dataset samples only. Copies files (originals untouched), keeps dataset ids.
python -m ml.test_library.build --manifest data/unified_manifest.csv --out test_videos --per-class 2"""
import argparse, csv, json, os, random, shutil
from collections import defaultdict

COLS = ["id", "file", "label", "dataset", "source_sample_id", "signer_id", "session_id", "split"]


def build_library(rows, out_dir, per_class=2, split="test", seed=0, forbidden_splits=("train", "val")):
    rng, pool, seen, by = random.Random(seed), [r for r in rows if r["split"] == split and r["task"] == "isolated"], set(), defaultdict(list)
    for r in pool:
        key = (r["sha1_head"], r["n_bytes"]) if r["sha1_head"] else r["sample_id"]
        if key in seen: continue                                   # no duplicates
        seen.add(key); by[r["label"]].append(r)
    train_hashes = {(r["sha1_head"], r["n_bytes"]) for r in rows if r["split"] in forbidden_splits and r["sha1_head"]}
    samples = os.path.join(out_dir, "word_recognition", "samples"); os.makedirs(samples, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "word_recognition", "prediction_results"), exist_ok=True)
    for sub in ("sentence_translation/samples", "sentence_translation/prediction_results", "manifests", "reports"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    open(os.path.join(out_dir, "sentence_translation", "README.txt"), "w").write(
        "Empty on purpose: no verified continuous-sentence data and no trained sentence model yet. Nothing is fabricated here.\n")
    out = []
    for label in sorted(by):
        items = by[label]; rng.shuffle(items)
        for r in items[:per_class]:
            if r["sha1_head"] and (r["sha1_head"], r["n_bytes"]) in train_hashes:
                raise ValueError(f"held-out sample {r['sample_id']} is byte-identical to a train/val sample")
            ext = os.path.splitext(r["path"])[1].lower()
            sid = f"{r['dataset']}_{len(out):04d}"
            rel = f"word_recognition/samples/{sid}{ext}"
            shutil.copy2(r["path"], os.path.join(out_dir, rel))
            out.append(dict(id=sid, file=rel, label=r["label"], dataset=r["dataset"], source_sample_id=r["sample_id"],
                            signer_id=r["signer_id"], session_id=r["session_id"], split=r["split"]))
    mdir = os.path.join(out_dir, "manifests")
    with open(os.path.join(mdir, "test_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(out)
    json.dump(out, open(os.path.join(mdir, "test_manifest.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", default="test_videos")
    ap.add_argument("--per-class", type=int, default=2); ap.add_argument("--split", default="test")
    a = ap.parse_args()
    from ml.datasets.manifest import read_manifest
    res = build_library(read_manifest(a.manifest), a.out, a.per_class, a.split)
    print(f"{len(res)} held-out videos from {len({r['label'] for r in res})} classes -> {a.out}")
