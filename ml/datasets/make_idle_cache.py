"""Create 'idle' training examples from the dataset's OWN videos (the rest frames before/after each sign).
Same signers, same camera, same lighting as the signs -> the model cannot learn 'my room = idle'.
Each window inherits the split of its source video (no leakage). Writes new .npz files + a new index.
python -m ml.datasets.make_idle_cache --index cache/index.csv --out-index cache/index_idle.csv"""
import argparse, csv, hashlib, os, random
from collections import Counter
import numpy as np
from isl.config import F_WRIST, IDLE_LABEL
from isl.features import normalize


def _median5(a):
    pad = np.pad(a, [(2, 2)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    return np.median(np.stack([pad[i:i + len(a)] for i in range(5)]), axis=0)


def rest_windows(raw, min_len=15, max_len=60, y_down=1.1, v_max=0.02):
    """Frames where BOTH wrists are low (y in shoulder spans, 0 = shoulder line) and nearly still (4-frame average speed).
    Returns [(start, end)] runs of at least min_len frames, each cut to at most max_len."""
    f = normalize(raw); wr = f[:, F_WRIST]
    y = np.stack([wr[:, 1], wr[:, 4]], 1)
    wr = _median5(wr); y = np.stack([wr[:, 1], wr[:, 4]], 1)       # median filter removes tracker spikes
    xy = np.stack([wr[:, 0:2], wr[:, 3:5]], 1)                    # (L,2 hands,2)
    lag = 4                                                       # speed over 4 frames: tracker jitter averages out
    v = np.zeros(len(f))
    if len(f) > lag: v[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=2).max(1) / lag
    ok = (y.min(1) > y_down) & (v < v_max)
    out, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]: j += 1
            if j - i >= min_len: out.append((i, min(j, i + max_len)))
            i = j
        else:
            i += 1
    return out


def make_idle(index_rows, out_dir, per_video=2, cap_per_split=None, seed=0):
    rng, by_split, made = random.Random(seed), {"train": [], "val": [], "test": []}, []
    sizes = Counter(r["label"] for r in index_rows if r["label"] != IDLE_LABEL)
    median = sorted(sizes.values())[len(sizes) // 2] if sizes else 10
    for r in index_rows:
        if r["label"] == IDLE_LABEL or r["split"] not in by_split: continue
        raw = np.load(r["cache"])["raw"]
        for k, (a, b) in enumerate(rest_windows(raw)[:per_video]):
            by_split[r["split"]].append((r["cache"], k, a, b))
    for sp, items in by_split.items():
        rng.shuffle(items)
        cap = cap_per_split or max(3, int(1.5 * median))
        for src, k, a, b in items[:cap]:
            name = "idle_" + hashlib.md5(f"{src}:{k}".encode()).hexdigest()[:12] + ".npz"
            dst = os.path.join(out_dir, name)
            np.savez_compressed(dst, raw=np.load(src)["raw"][a:b])
            made.append({"cache": dst, "label": IDLE_LABEL, "split": sp})
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--out-index", required=True)
    ap.add_argument("--per-video", type=int, default=2); ap.add_argument("--cap-per-split", type=int, default=0)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.index))]
    new = make_idle(rows, os.path.dirname(a.index) or ".", a.per_video, a.cap_per_split or None)
    keep = [r for r in rows if r["label"] != IDLE_LABEL] + new          # recorded-room idle clips are dropped on purpose
    with open(a.out_index, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cache", "label", "split"]); w.writeheader(); w.writerows(keep)
    print(f"idle windows from dataset rest frames: {Counter(r['split'] for r in new)} -> {a.out_index}")
