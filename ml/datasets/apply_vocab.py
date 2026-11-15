"""Keep only reliable words as classes; every other REAL sign becomes 'unknown_sign' (a real negative class).
python -m ml.datasets.apply_vocab --manifest data/unified_manifest.csv --vocab reports/cv/reliable_words.json --out data/vocab_manifest.csv"""
import argparse, json
from . import manifest as M


def apply_vocab(rows, vocab, unknown="unknown_sign"):
    keep = set(vocab); out = []
    for r in rows:
        r = dict(r); r["text"] = r["text"] or r["label"]           # original label preserved
        if r["label"] not in keep and r["label"] != "idle": r["label"] = unknown
        out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--vocab", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = apply_vocab(M.read_manifest(a.manifest), json.load(open(a.vocab)))
    M.write_manifest(rows, a.out); print(M.summary(rows))
