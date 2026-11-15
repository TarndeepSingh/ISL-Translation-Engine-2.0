"""Unified dataset manifest: one row per video, original dataset ids preserved, tasks never mixed.
Also: session inference, group-wise splits (no group in two splits) and a leakage checker."""
import csv, hashlib, json, os, random, re
from collections import defaultdict

FIELDS = ["dataset", "sample_id", "path", "task", "label", "text", "signer_id", "session_id",
          "split_official", "split", "n_bytes", "sha1_head"]
VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm")
clean = lambda n: re.sub(r"^\d+\.\s*", "", n).strip().lower()
_num = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def fingerprint(path, n=1 << 20):
    """sha1 of the first 1 MB + size: cheap duplicate detector."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest(), os.path.getsize(path)


def row(dataset, sample_id, path, task, label="", text="", signer_id="", session_id="", split_official=""):
    return dict(dataset=dataset, sample_id=sample_id, path=path, task=task, label=label, text=text,
                signer_id=signer_id, session_id=session_id, split_official=split_official, split="",
                n_bytes="", sha1_head="")


def from_isolated_tree(root, dataset="include"):
    """root/<category>/<N. word>/<video>  ->  rows (task='isolated'). Label = cleaned folder name."""
    rows = []
    for d, _, fs in os.walk(root):
        for f in sorted(fs):
            if f.lower().endswith(VIDEO_EXT):
                p = os.path.join(d, f)
                rel = os.path.relpath(p, root).replace("\\", "/")
                rows.append(row(dataset, rel, p, "isolated", label=clean(os.path.basename(d))))
    return rows


def add_fingerprints(rows):
    for r in rows:
        if os.path.exists(r["path"]):
            r["sha1_head"], r["n_bytes"] = fingerprint(r["path"])
    return rows


def infer_sessions(rows, gap=30):
    """ASSUMPTION-based proxy for signer/session: camera files are numbered sequentially (MVI_5194.MOV);
    within a category, numbers that jump by more than `gap` start a new session. Verify by eye before trusting."""
    groups = defaultdict(list)
    for r in rows:
        p = r["path"].replace("\\", "/")
        cat = os.path.basename(os.path.dirname(os.path.dirname(p))) or "root"
        m = _num.search(os.path.basename(p))
        groups[(r["dataset"], cat)].append((int(m.group(1)) if m else -1, r))
    for (ds, cat), items in groups.items():
        items.sort(key=lambda t: t[0]); k, prev = 0, None
        for n, r in items:
            if n < 0:
                r["session_id"] = f"{ds}:{cat}:nonum"; continue
            if prev is not None and n - prev > gap:
                k += 1
            prev = n; r["session_id"] = f"{ds}:{cat}:{k}"
    return rows


def make_group_split(rows, fractions=(0.7, 0.15, 0.15), seed=0, key="session_id"):
    """Assign WHOLE groups to train/val/test (greedy by remaining deficit). Never splits a group."""
    names, groups = ("train", "val", "test"), defaultdict(list)
    for r in rows:
        groups[r[key] or r["sample_id"]].append(r)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    target, have = [f * len(rows) for f in fractions], [0, 0, 0]
    for k in keys:
        i = max(range(3), key=lambda j: target[j] - have[j])
        for r in groups[k]:
            r["split"] = names[i]
        have[i] += len(groups[k])
    return rows


def check_leakage(rows, key="session_id"):
    """Returns a dict of problems; empty dict == clean."""
    out, by_g, by_h, by_id = {}, defaultdict(set), defaultdict(set), defaultdict(int)
    for r in rows:
        by_g[r[key] or r["sample_id"]].add(r["split"]); by_id[(r["dataset"], r["sample_id"])] += 1
        if r["sha1_head"]:
            by_h[(r["sha1_head"], r["n_bytes"])].add(r["split"])
    leaked = sorted(g for g, s in by_g.items() if len(s) > 1)
    dup = [h for h, s in by_h.items() if len(s) > 1]
    dup_ids = [i for i, c in by_id.items() if c > 1]
    train_labels = {r["label"] for r in rows if r["split"] == "train"}
    missing = sorted({r["label"] for r in rows if r["split"] in ("val", "test") and r["label"] not in train_labels})
    if leaked: out["groups_in_multiple_splits"] = leaked[:20]
    if dup: out["identical_files_across_splits"] = len(dup)
    if dup_ids: out["duplicate_sample_ids"] = len(dup_ids)
    if missing: out["classes_missing_from_train"] = missing
    return out


def write_manifest(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8"), indent=1)


def read_manifest(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def summary(rows):
    per = defaultdict(int)
    for r in rows: per[r["label"]] += 1
    v = sorted(per.values()) or [0]
    return {"videos": len(rows), "classes": len(per), "min_per_class": v[0], "median_per_class": v[len(v) // 2],
            "max_per_class": v[-1], "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}}


def key3(p):
    """Last three path parts, lower-cased: the join key between local files and dataset metadata."""
    return "/".join(p.replace("\\", "/").split("/")[-3:]).lower()


def filter_keep(rows, keep):
    return [r for r in rows if key3(r["path"]) in keep]
