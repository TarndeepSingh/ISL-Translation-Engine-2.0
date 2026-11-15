"""Build data/manifest.csv using the OFFICIAL INCLUDE-50 list + train/val/test split from the
Hugging Face metadata (ai4bharat/INCLUDE), matched to the videos you downloaded from Zenodo.
Optionally appends your own recordings (folder per label, e.g. data/custom/idle/*.mp4).
python -m isl.hf_manifest --root data/INCLUDE --custom data/custom"""
import argparse, csv, os, random, re

EXT = (".mov", ".mp4", ".avi", ".mkv")
clean = lambda n: re.sub(r"^\d+\.\s*", "", n).strip().lower()      # "50. Yellow" -> "yellow"


def index_videos(root):
    idx = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(EXT):
                p = os.path.join(d, f); parts = p.replace("\\", "/").split("/")
                idx["/".join(parts[-3:]).lower()] = p
    return idx


def build_rows(records, idx):
    """records: dicts with split,label,video_path. Returns rows, n_missing, n_inconsistent."""
    rows, missing, bad = [], 0, 0
    for r in records:
        parts = r["video_path"].replace("\\", "/").split("/")
        if clean(parts[-2]) != clean(r["label"]):          # metadata row whose label disagrees with its path
            bad += 1; continue
        p = idx.get("/".join(parts[-3:]).lower())
        if p is None:
            missing += 1; continue
        rows.append((p, clean(r["label"]), r["split"]))
    return rows, missing, bad


def custom_rows(root, seed=0):
    rng, rows = random.Random(seed), []
    for lab in sorted(os.listdir(root)):
        d = os.path.join(root, lab)
        if not os.path.isdir(d): continue
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith(EXT)); rng.shuffle(fs)
        n_test = max(1, round(.2 * len(fs))) if len(fs) >= 2 else 0
        n_val = 1 if len(fs) >= 4 else 0
        for i, f in enumerate(fs):
            sp = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            rows.append((os.path.join(d, f), lab.lower(), sp))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--custom")
    ap.add_argument("--out", default="data/manifest.csv"); ap.add_argument("--all-classes", action="store_true")
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/INCLUDE")
    name = {"train": "train", "val": "val", "validation": "val", "test": "test"}
    recs = [dict(r, split=name[s]) for s in ds for r in ds[s] if a.all_classes or r["include_50"]]
    rows, missing, bad = build_rows(recs, index_videos(a.root))
    print(f"metadata rows {len(recs)} | matched {len(rows)} | missing on disk {missing} | inconsistent {bad}")
    if a.custom and os.path.isdir(a.custom):
        rows += custom_rows(a.custom)
    labels = sorted({r[1] for r in rows})
    print(f"{len(labels)} classes:", ", ".join(labels))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "split"]); w.writerows(rows)
    for s in ("train", "val", "test"):
        print(s, sum(r[2] == s for r in rows))
    if missing > 0.2 * max(1, len(recs)):
        print("WARNING: many videos missing - download more Zenodo files (see isl.download_include --list)")
