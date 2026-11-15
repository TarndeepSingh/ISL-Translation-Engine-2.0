"""Module 2: train -> evaluate -> save best checkpoint.
python -m isl.train --index cache/index.csv --epochs 150"""
import argparse, csv, hashlib, json, os, platform, random, subprocess, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .config import *
from .features import mirror_raw, normalize, resample, rotate_raw, time_warp_raw
from .model import ISLNet
from ml.calibration import fit_temperature


def resolve_cache_paths(rows, index_path):
    """Point every row's ``cache`` at a file that exists; return the sample_ids that cannot be found.

    The preprocessing index stores an absolute ``cache`` path from the machine that wrote it, which is
    wrong after the folder is copied (Drive -> a new Colab session). The relative ``cache_file`` next to
    ``index.csv`` is therefore tried first, then the recorded ``cache`` path (resolved against the index
    folder if relative).
    """
    base = os.path.dirname(os.path.abspath(index_path))
    missing = []
    for r in rows:
        candidates = []
        if r.get("cache_file"):
            candidates.append(os.path.join(base, r["cache_file"]))
        recorded = r.get("cache", "")
        if recorded:
            candidates.append(recorded if os.path.isabs(recorded) else os.path.join(base, recorded))
        found = next((os.path.abspath(c) for c in candidates if os.path.isfile(c)), None)
        if found is None:
            missing.append(r["sample_id"])
        else:
            r["cache"] = found
    return missing


def augment_raw(raw, rng, label, allow_mirror=False, rotation_deg=0.0, time_warp=0.0):
    if label == IDLE_LABEL and len(raw) > 15:                 # idle: random crop of long clip
        n = rng.randint(15, min(80, len(raw)))
        s = rng.randint(0, len(raw) - n); raw = raw[s:s + n]
    else:                                                      # signs: jitter start/end
        L = len(raw); a = rng.randint(0, max(0, int(.1 * L))); b = rng.randint(0, max(0, int(.1 * L)))
        raw = raw[a:L - b] if L - a - b >= 8 else raw
    if allow_mirror and rng.random() < 0.3:
        raw = mirror_raw(raw)                                  # left-handed signers
    if time_warp > 0:                                          # opt-in: signer speed variation
        raw = time_warp_raw(raw, rng, time_warp)
    if rotation_deg > 0:                                       # opt-in: camera tilt
        raw = rotate_raw(raw, rng.uniform(-rotation_deg, rotation_deg))
    raw = raw.copy()
    for sl in (LH_SLICE, RH_SLICE):                            # simulate tracking dropouts
        if rng.random() < 0.3:
            n = rng.randint(1, max(1, int(.2 * len(raw)))); s = rng.randint(0, max(0, len(raw) - n))
            raw[s:s + n, sl] = np.nan
    return raw


class DS(Dataset):
    def __init__(self, items, l2i, train, seed=0, allow_mirror=False, rotation_deg=0.0, time_warp=0.0):
        self.raws = [np.load(p)["raw"] for p, _ in items]
        self.y = [l2i[l] for _, l in items]; self.names = [l for _, l in items]
        self.train, self.rng = train, random.Random(seed)
        self.allow_mirror = allow_mirror
        self.rotation_deg, self.time_warp = rotation_deg, time_warp
        self.fixed = None if train else [self._make(i, False) for i in range(len(items))]

    def _make(self, i, aug):
        raw = self.raws[i]
        if aug or self.names[i] == IDLE_LABEL:
            raw = (augment_raw(raw, self.rng, self.names[i], self.allow_mirror, self.rotation_deg, self.time_warp)
                   if aug else raw)
            if not aug and len(raw) > 80:
                s = self.rng.randint(0, len(raw) - 60); raw = raw[s:s + 60]
        x = resample(normalize(raw))
        if aug:
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.from_numpy(x)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self._make(i, True) if self.train else self.fixed[i]
        return x, self.y[i]


def topk(logits, y, k):
    return (logits.topk(k, 1).indices == y[:, None]).any(1).float().sum().item()


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); n = c1 = c3 = 0; P = []; Y = []
    for x, y in dl:
        x, y = x.to(dev), y.to(dev); lg = model(x)
        n += len(y); c1 += topk(lg, y, 1); c3 += topk(lg, y, min(3, lg.shape[1]))
        P += lg.argmax(1).tolist(); Y += y.tolist()
    return c1 / n, c3 / n, np.array(Y), np.array(P)


@torch.no_grad()
def collect_logits(model, dl, dev):
    model.eval(); logits, labels = [], []
    for x, y in dl:
        logits.append(model(x.to(dev)).cpu())
        labels.append(y.cpu())
    if not logits:
        raise ValueError("validation split is empty")
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="cache/index.csv"); ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-mirror-augmentation", action="store_true",
                    help="Enable only after validation shows mirroring preserves every included label")
    ap.add_argument("--aug-rotation-deg", type=float, default=0.0,
                    help="max random camera-tilt rotation in degrees (0=off; AI4Bharat INCLUDE used +/-7)")
    ap.add_argument("--aug-time-warp", type=float, default=0.0,
                    help="non-uniform speed variation strength in [0,0.9] (0=off)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(a.out, exist_ok=True)
    with open(a.index, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"cache", "label", "split", "group_id", "sample_id", "task",
                "rights_status", "license_id", "sha256", "dataset_id", "dataset_version",
                "preprocess_version"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"index must have rows and columns: {', '.join(sorted(required))}")
    if any(r.get("task") != "isolated" for r in rows):
        raise SystemExit("isolated training index contains a non-isolated task")
    allowed_rights = {"approved_research", "approved_noncommercial", "approved_commercial"}
    if any(r.get("rights_status") not in allowed_rights or not r.get("license_id", "").strip() for r in rows):
        raise SystemExit("training requires reviewed rights_status and a non-empty license_id on every row")
    if any(len(r.get("sha256", "")) != 64 or any(c not in "0123456789abcdefABCDEF" for c in r["sha256"])
           for r in rows):
        raise SystemExit("training requires SHA-256 for every source media row")
    if {r["preprocess_version"] for r in rows} != {"mediapipe-tasks-pose-hands-75x3-v1"}:
        raise SystemExit("training index contains an unsupported or mixed landmark preprocessing version")
    if any(not r.get("label", "").strip() for r in rows):
        raise SystemExit("isolated training index contains a blank label")
    ids = [r.get("sample_id", "") for r in rows]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise SystemExit("sample_id values must be present and unique")
    group_splits = {}
    hash_splits = {}
    for r in rows:
        split = r.get("split", "")
        if split not in {"train", "val", "test"}:
            raise SystemExit(f"sample {r['sample_id']} has no accepted train/val/test split")
        group = r.get("group_id", "").strip()
        if not group:
            raise SystemExit(f"sample {r['sample_id']} is missing a split group")
        group_splits.setdefault(group, set()).add(split)
        digest = r.get("sha256", "").strip().lower()
        if digest:
            hash_splits.setdefault(digest, set()).add(split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise SystemExit("split leakage: a group_id occurs across train/val/test")
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise SystemExit("split leakage: an identical SHA-256 occurs across train/val/test")
    missing = resolve_cache_paths(rows, a.index)
    if missing:
        raise SystemExit(f"cache artifact is missing for sample {missing[0]} ({len(missing)} missing in total)")
    labels = sorted({r["label"] for r in rows}); l2i = {l: i for i, l in enumerate(labels)}
    tr = [(r["cache"], r["label"]) for r in rows if r["split"] == "train"]
    te = [(r["cache"], r["label"]) for r in rows if r["split"] == "test"]
    va = [(r["cache"], r["label"]) for r in rows if r["split"] == "val"]
    if not tr or not va or not te:
        raise SystemExit("train, val and test must all be non-empty; provide grouped splits before training")
    for split, items in (("train", tr), ("val", va), ("test", te)):
        absent = sorted(set(labels) - {label for _, label in items})
        if absent:
            raise SystemExit(f"{split} split lacks labels: {absent[:10]}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtr = DS(tr, l2i, True, a.seed, a.allow_mirror_augmentation, a.aug_rotation_deg, a.aug_time_warp)
    dva, dte = DS(va, l2i, False, a.seed), DS(te, l2i, False, a.seed + 1)
    L = lambda d, s: DataLoader(d, a.bs, shuffle=s, drop_last=s and len(d) > a.bs)
    model = ISLNet(n_classes=len(labels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * max(1, len(dtr) // a.bs))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, bad, started = -1., 0, time.time()
    best_epoch = -1
    print(f"{len(labels)} classes | train {len(tr)} val {len(va)} test {len(te)} | {dev}")
    for ep in range(a.epochs):
        model.train()
        for x, y in L(dtr, True):
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            crit(model(x), y).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        v1, v3, _, _ = evaluate(model, L(dva, False), dev)
        if v1 > best:
            best, bad = v1, 0
            best_epoch = ep
            torch.save({"model": model.state_dict(), "labels": labels, "epoch": ep, "seed": a.seed}, f"{a.out}/best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "labels": labels, "epoch": ep, "seed": a.seed}, f"{a.out}/last.pt")
        if ep % 5 == 0: print(f"ep {ep:3d} val top1 {v1:.3f} top3 {v3:.3f} best {best:.3f}")
        if bad >= a.patience: break
    ck = torch.load(f"{a.out}/best.pt", map_location=dev, weights_only=False); model.load_state_dict(ck["model"])
    val_logits, val_y = collect_logits(model, L(dva, False), dev)
    calibration = fit_temperature(val_logits, val_y)
    ck_hash = hashlib.sha256(open(f"{a.out}/best.pt", "rb").read()).hexdigest()
    labels_hash = hashlib.sha256(json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    calibration.update({"schema_version": "temperature-calibration-v1",
                        "checkpoint_sha256": ck_hash, "labels_sha256": labels_hash})
    with open(f"{a.out}/calibration.json", "w", encoding="utf-8") as stream:
        json.dump(calibration, stream, ensure_ascii=False, indent=2)
    t1, t3, Y, P = evaluate(model, L(dte, False), dev)
    print(f"TEST top1 {t1:.3f} top3 {t3:.3f}")
    if IDLE_LABEL in l2i:
        m = Y == l2i[IDLE_LABEL]
        print(f"idle recall {(P[m] == Y[m]).mean():.3f} | signs misread as idle "
              f"{((P == l2i[IDLE_LABEL]) & ~m).sum()}/{(~m).sum()}")
    with open(f"{a.out}/labels.json", "w", encoding="utf-8") as stream:
        json.dump(labels, stream, ensure_ascii=False, indent=2)
    np.save(f"{a.out}/test_pred.npy", np.stack([Y, P]))
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_revision = None
    with open(a.index, "rb") as stream:
        index_sha256 = hashlib.sha256(stream.read()).hexdigest()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    run = {
        "schema_version": "isolated-experiment-v1", "task": "isolated",
        "started_unix": started, "elapsed_seconds": time.time() - started,
        "git_revision": git_revision, "index_path": os.path.abspath(a.index),
        "index_sha256": index_sha256, "row_count": len(rows),
        "split_rows": {name: len(items) for name, items in (("train", tr), ("val", va), ("test", te))},
        "split_groups": {name: len({r["group_id"] for r in rows if r["split"] == name}) for name in ("train", "val", "test")},
        "labels": labels, "seed": a.seed, "epochs_requested": a.epochs,
        "datasets": sorted({f"{r['dataset_id']}@{r['dataset_version']}" for r in rows}),
        "rights_status_counts": {state: sum(r["rights_status"] == state for r in rows)
                                  for state in sorted({r["rights_status"] for r in rows})},
        "license_ids": sorted({r["license_id"] for r in rows}),
        "preprocess_versions": sorted({r["preprocess_version"] for r in rows}),
        "best_epoch": best_epoch, "batch_size": a.bs, "learning_rate": a.lr,
        "mirror_augmentation": bool(a.allow_mirror_augmentation),
        "augmentation": {"rotation_deg": a.aug_rotation_deg, "time_warp": a.aug_time_warp},
        "calibration": calibration,
        "architecture": "ISLNet Conv1D-BiLSTM-MHSA", "input": {"frames": T, "features": FEATURE_DIM},
        "test_top1": t1, "test_top3": t3, "python": platform.python_version(),
        "torch": torch.__version__, "device": dev, "gpu": gpu_name,
    }
    with open(os.path.join(a.out, "experiment.json"), "w", encoding="utf-8") as stream:
        json.dump(run, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "MODEL_CARD.md"), "w", encoding="utf-8") as stream:
        stream.write("# Isolated sign model experiment\n\n")
        stream.write("This artifact is experimental until dataset rights, grouped evaluation, confidence calibration, "
                     "ONNX parity and real-video API integration are independently reviewed.\n\n")
        stream.write(f"- Experiment metadata: `experiment.json`\n- Source index SHA-256: `{index_sha256}`\n")
        stream.write(f"- Best validation epoch: {best_epoch}\n- Test top-1: {t1:.6f}\n- Test top-3: {t3:.6f}\n")
        stream.write(f"- Dataset releases: {', '.join(run['datasets'])}\n- License IDs: {', '.join(run['license_ids'])}\n")
        stream.write(f"- Calibration: `calibration.json`; fitted on validation only.\n")
        stream.write("- This experimental artifact must not be promoted until source terms, signer/session coverage, "
                     "grouped evaluation, calibration, ONNX parity and permitted real-video integration pass review.\n")
