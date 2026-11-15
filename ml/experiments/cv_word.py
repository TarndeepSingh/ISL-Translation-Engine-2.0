"""Honest evaluation on cached landmarks (fast, CPU only; no videos, no GPU needed).
Compares RANDOM k-fold with SESSION-GROUPED k-fold: the gap is the leakage you were fooled by.
Also finds which words are actually learnable (reliable vocabulary).
python -m ml.experiments.cv_word --manifest data/unified_manifest.csv --cache cache --folds 5 --out reports/cv"""
import argparse, hashlib, json, os
import numpy as np
from isl.features import normalize, resample


def cache_path(path, cache_dir):
    return os.path.join(cache_dir, hashlib.md5(path.encode()).hexdigest()[:16] + ".npz")


def load_arrays(rows, cache_dir):
    X, y, g, missing = [], [], [], 0
    for r in rows:
        p = cache_path(r["path"], cache_dir)
        if not os.path.exists(p): missing += 1; continue
        X.append(resample(normalize(np.load(p)["raw"]))); y.append(r["label"]); g.append(r["session_id"] or r["sample_id"])
    return np.stack(X), np.array(y), np.array(g), missing


def make_model(name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "knn": return make_pipeline(StandardScaler(), KNeighborsClassifier(3, metric="cosine", algorithm="brute"))
    if name == "logreg": return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=300))
    raise ValueError(name)


def run_cv(X, y, groups, model_name, folds=5, grouped=True, seed=0):
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    Xf = X.reshape(len(X), -1)
    if grouped: splits = GroupKFold(n_splits=min(folds, len(set(groups)))).split(Xf, y, groups)
    else: splits = StratifiedKFold(min(folds, min(np.unique(y, return_counts=True)[1])), shuffle=True, random_state=seed).split(Xf, y)
    accs, hits, tot = [], {}, {}
    for tr, te in splits:
        m = make_model(model_name).fit(Xf[tr], y[tr]); p = m.predict(Xf[te])
        accs.append(float((p == y[te]).mean()))
        for t, q in zip(y[te], p):
            tot[t] = tot.get(t, 0) + 1; hits[t] = hits.get(t, 0) + (t == q)
    return {"fold_acc": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "recall": {k: hits[k] / tot[k] for k in tot}, "support": tot}


def reliable_words(recall, support, min_recall=0.7, min_support=6):
    return sorted(w for w in recall if recall[w] >= min_recall and support[w] >= min_support and w not in ("idle", "unknown_sign"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--cache", default="cache")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--out", default="reports/cv")
    ap.add_argument("--models", nargs="*", default=["knn", "logreg"]); ap.add_argument("--min-recall", type=float, default=0.7)
    a = ap.parse_args()
    from ml.datasets.manifest import read_manifest
    rows = [r for r in read_manifest(a.manifest) if r["task"] == "isolated"]
    X, y, g, miss = load_arrays(rows, a.cache)
    os.makedirs(a.out, exist_ok=True)
    n_cls = len(set(y)); res = {"videos": len(y), "classes": n_cls, "missing_cache": miss, "chance": 1 / n_cls, "sessions": len(set(g)), "models": {}}
    print(f"{len(y)} videos, {n_cls} classes, {len(set(g))} sessions, chance {100 / n_cls:.1f}%, {miss} without cache")
    for mn in a.models:
        rnd, grp = run_cv(X, y, g, mn, a.folds, grouped=False), run_cv(X, y, g, mn, a.folds, grouped=True)
        res["models"][mn] = {"random_cv": {k: rnd[k] for k in ("mean", "std")}, "grouped_cv": {k: grp[k] for k in ("mean", "std", "recall", "support")},
                             "leakage_gap": rnd["mean"] - grp["mean"], "reliable_words": reliable_words(grp["recall"], grp["support"], a.min_recall)}
        print(f"{mn:7s} random-CV {100 * rnd['mean']:.1f}% +/-{100 * rnd['std']:.1f} | SESSION-grouped CV {100 * grp['mean']:.1f}% +/-{100 * grp['std']:.1f} "
              f"| leakage gap {100 * (rnd['mean'] - grp['mean']):.1f} pts | reliable words: {len(res['models'][mn]['reliable_words'])}")
    json.dump(res, open(os.path.join(a.out, "cv_results.json"), "w"), indent=1)
    best = max(res["models"], key=lambda k: res["models"][k]["grouped_cv"]["mean"])
    json.dump(res["models"][best]["reliable_words"], open(os.path.join(a.out, "reliable_words.json"), "w"), indent=1)
    print("best by grouped CV:", best, "-> reliable_words.json written")
