"""Per-class accuracy, overall / average accuracy and confusion matrix for the isolated word model.

No retraining and no videos needed. ``isl.train`` already saves, for every finished run:
    <run>/test_pred.npy   2 x N int array: row 0 = true class index, row 1 = predicted class index (TEST split only)
    <run>/labels.json     class names in index order
    <run>/experiment.json reported test_top1 / test_top3 (used here as a cross-check)

Run it where those folders live (Google Drive / Colab):

  python -m ml.evaluation.class_report \
      --runs-dir /content/drive/MyDrive/ISL_Translation_Engine/runs/isolated \
      --policy balanced --variant aug7-warp0.3 --seeds 13,14,15 --deployed-seed 13 \
      --out models/isl.eval.json --plots reports/eval_plots

Outputs
  models/isl.eval.json     what the website reads (served by GET /api/v1/evaluation)
  reports/eval_plots/*.png charts for the report / slides

Definitions
  overall accuracy        correct / all test videos            (every video counts equally)
  average class accuracy  mean of the per-class accuracies     (every class counts equally)
  per-class accuracy      correct videos of that class / videos of that class  (= recall)
The two averages differ whenever classes have different numbers of test videos.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

from .metrics import confusion_matrix

SCHEMA = "isl-eval-v1"


# ----------------------------------------------------------------------------------------------- helpers
def clean_word(raw: str) -> str:
    """'61. Father' -> 'Father'. Same rule as backend/main.py so names match the rest of the website."""
    head, dot, tail = raw.partition(".")
    return tail.strip() if dot and head.strip().isdigit() else raw


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n. Honest about tiny per-class samples."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load_run(run_dir: Path) -> dict:
    pred_path, labels_path = run_dir / "test_pred.npy", run_dir / "labels.json"
    for p in (pred_path, labels_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found - was this run finished? (isl.train writes it at the end)")
    arr = np.load(pred_path)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"{pred_path}: expected shape (2, N), got {arr.shape}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    y, p = arr[0].astype(int), arr[1].astype(int)
    n = len(labels)
    if y.min() < 0 or p.min() < 0 or y.max() >= n or p.max() >= n:
        raise ValueError(f"{pred_path}: class index outside 0..{n - 1} - labels.json and predictions do not belong together")
    exp_path = run_dir / "experiment.json"
    exp = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.is_file() else {}
    return {"y": y, "p": p, "labels": labels, "exp": exp, "dir": run_dir}


def summarise(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict:
    cm = confusion_matrix(y, p, n_classes)
    support = cm.sum(1)
    correct = np.diag(cm)
    acc = np.where(support > 0, correct / np.maximum(support, 1), np.nan)
    present = support > 0
    return {
        "cm": cm, "support": support, "correct": correct, "acc": acc,
        "overall": float(correct.sum() / max(1, support.sum())),
        "average": float(np.nanmean(acc[present])) if present.any() else 0.0,
    }


def top_confusions(cm: np.ndarray, words: list[str], k: int = 12) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    support = cm.sum(1)
    pairs = [(int(off[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if off[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], words[t[1]], words[t[2]]))
    return [{"true": words[i], "pred": words[j], "true_id": i, "pred_id": j, "count": c,
             "rate": float(c / support[i])} for c, i, j in pairs[:k]]


# ----------------------------------------------------------------------------------------------- report
def build_report(runs: dict[int, dict], deployed_seed: int, policy: str, variant: str, notes: list[str]) -> dict:
    labels = runs[deployed_seed]["labels"]
    raw = list(labels)
    words = [clean_word(r) for r in raw]
    n_cls = len(raw)
    warnings: list[str] = []

    for seed, r in runs.items():
        if r["labels"] != raw:
            raise ValueError(f"seed {seed} has a different class list than the deployed run - cannot compare them")

    main = runs[deployed_seed]
    s = summarise(main["y"], main["p"], n_cls)

    reported = main["exp"].get("test_top1")
    if reported is not None and abs(reported - s["overall"]) > 1e-6:
        warnings.append(f"overall accuracy from test_pred.npy ({s['overall']:.4f}) differs from experiment.json "
                        f"test_top1 ({reported:.4f}) for the deployed run - check that both files come from the same run.")

    per_seed = {}
    for seed, r in sorted(runs.items()):
        ss = summarise(r["y"], r["p"], n_cls)
        same_test = bool(len(r["y"]) == len(main["y"]) and (r["y"] == main["y"]).all())
        if not same_test:
            warnings.append(f"seed {seed} was scored on a different test list than seed {deployed_seed}; "
                            "per-class seed averages mix different videos.")
        per_seed[seed] = {"s": ss, "same_test": same_test, "exp": r["exp"]}

    per_class = []
    for i in range(n_cls):
        sup, cor = int(s["support"][i]), int(s["correct"][i])
        lo, hi = wilson(cor, sup)
        row = s["cm"][i].copy()
        row[i] = 0
        j = int(row.argmax()) if row.any() else None
        seed_acc = {str(sd): (None if np.isnan(v["s"]["acc"][i]) else float(v["s"]["acc"][i])) for sd, v in per_seed.items()}
        vals = [v for v in seed_acc.values() if v is not None]
        per_class.append({
            "id": i, "raw": raw[i], "word": words[i], "support": sup, "correct": cor,
            "accuracy": None if sup == 0 else cor / sup, "ci_low": lo, "ci_high": hi,
            "most_confused_with": None if j is None else {"word": words[j], "id": j, "count": int(row[j])},
            "seed_accuracy": seed_acc,
            "seed_mean": float(statistics.fmean(vals)) if vals else None,
            "seed_std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        })

    seeds_out = []
    for sd, v in per_seed.items():
        seeds_out.append({"seed": sd, "n_test": int(v["s"]["support"].sum()), "overall_accuracy": v["s"]["overall"],
                          "average_class_accuracy": v["s"]["average"], "reported_top1": v["exp"].get("test_top1"),
                          "reported_top3": v["exp"].get("test_top3"), "same_test_videos": v["same_test"]})
    ov = [x["overall_accuracy"] for x in seeds_out]
    av = [x["average_class_accuracy"] for x in seeds_out]
    seed_summary = {"n_seeds": len(seeds_out),
                    "overall_mean": float(statistics.fmean(ov)), "overall_std": float(statistics.stdev(ov)) if len(ov) > 1 else 0.0,
                    "average_mean": float(statistics.fmean(av)), "average_std": float(statistics.stdev(av)) if len(av) > 1 else 0.0}

    sup = s["support"][s["support"] > 0]
    return {
        "schema_version": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "split": "test", "policy": policy, "variant": variant,
        "deployed": {
            "run": main["dir"].name, "seed": deployed_seed, "n_test": int(s["support"].sum()), "n_classes": n_cls,
            "correct": int(s["correct"].sum()), "overall_accuracy": s["overall"], "average_class_accuracy": s["average"],
            "reported_top1": main["exp"].get("test_top1"), "reported_top3": main["exp"].get("test_top3"),
            "min_videos_per_class": int(sup.min()), "max_videos_per_class": int(sup.max()),
        },
        "labels_sha256": hashlib.sha256(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "labels": [{"id": i, "raw": raw[i], "word": words[i]} for i in range(n_cls)],
        "per_class": per_class,
        "confusion_matrix": s["cm"].tolist(),
        "top_confusions": top_confusions(s["cm"], words),
        "seeds": seeds_out, "seed_summary": seed_summary,
        "warnings": warnings, "notes": notes,
    }


# ----------------------------------------------------------------------------------------------- plots
def make_plots(report: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cm = np.array(report["confusion_matrix"])
    words = [c["word"] for c in report["labels"]]
    pc = [c for c in report["per_class"] if c["support"] > 0]
    dep = report["deployed"]
    paths = []

    # 1) per-class accuracy bars, worst first (top of chart)
    pc_sorted = sorted(pc, key=lambda c: (c["accuracy"], c["word"].lower()))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.27 * len(pc_sorted) + 1.6)))
    ypos = np.arange(len(pc_sorted))
    vals = [c["accuracy"] * 100 for c in pc_sorted]
    colors = ["#16a34a" if v >= 95 else "#f59e0b" if v >= 80 else "#dc2626" for v in vals]
    ax.barh(ypos, vals, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c["word"] for c in pc_sorted], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.axvline(dep["overall_accuracy"] * 100, color="#2563eb", ls="--", lw=1, label=f"overall {dep['overall_accuracy'] * 100:.1f}%")
    ax.axvline(dep["average_class_accuracy"] * 100, color="#7c3aed", ls=":", lw=1.2, label=f"average class {dep['average_class_accuracy'] * 100:.1f}%")
    for y, c, v in zip(ypos, pc_sorted, vals):
        ax.text(min(v, 100) + 1, y, f"{c['correct']}/{c['support']}", va="center", fontsize=6)
    ax.set_xlabel("test accuracy for this word (%)")
    ax.set_title(f"Per-class accuracy - {dep['run']} (n={dep['n_test']} test videos)", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    p = out_dir / "per_class_accuracy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2) confusion matrices: row-normalised (rates) and raw counts
    support = cm.sum(1, keepdims=True)
    for name, mat, fmt, title in (
        ("confusion_matrix_normalized.png", cm / np.maximum(support, 1), None, "Confusion matrix (row-normalised: share of each true word)"),
        ("confusion_matrix_counts.png", cm.astype(float), "d", "Confusion matrix (number of test videos)"),
    ):
        n = len(words)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.26), max(8, n * 0.26)))
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if fmt is None else max(1, mat.max()))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(words, rotation=90, fontsize=6)
        ax.set_yticklabels(words, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    shade = mat[i, j] / (1 if fmt is None else max(1, mat.max()))
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=4.5,
                            color="white" if shade > 0.55 else ("#b91c1c" if i != j else "black"))
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="folder that contains include50-<policy>-<variant>-seed<N>/")
    ap.add_argument("--policy", default="balanced")
    ap.add_argument("--variant", default="aug7-warp0.3")
    ap.add_argument("--seeds", default="13,14,15")
    ap.add_argument("--deployed-seed", type=int, default=13, help="seed of the model that is exported to models/isl.onnx")
    ap.add_argument("--out", default="models/isl.eval.json")
    ap.add_argument("--plots", default="reports/eval_plots", help="folder for PNG charts ('' to skip)")
    a = ap.parse_args(argv)

    runs_dir = Path(a.runs_dir)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    if a.deployed_seed not in seeds:
        seeds.append(a.deployed_seed)
    runs: dict[int, dict] = {}
    for sd in seeds:
        d = runs_dir / f"include50-{a.policy}-{a.variant}-seed{sd}"
        try:
            runs[sd] = load_run(d)
        except FileNotFoundError as e:
            if sd == a.deployed_seed:
                raise SystemExit(f"The deployed run is required: {e}")
            print(f"skipping seed {sd}: {e}")

    notes = [
        "Scores come from the held-out TEST split only (never train or validation videos).",
        "INCLUDE was recorded with a small group of signers, and a test video can come from the same signer and "
        "session as training videos, so these numbers are NOT evidence of accuracy for a new person or a new camera.",
        "Each word has only a handful of test videos, so one wrong video moves that word's accuracy by tens of points. "
        "Read the n and the interval next to every bar before calling a word 'weak' or 'strong'.",
    ]
    report = build_report(runs, a.deployed_seed, a.policy, a.variant, notes)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    d, ss = report["deployed"], report["seed_summary"]
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    print(f"deployed run  : {d['run']}  ({d['n_test']} test videos, {d['n_classes']} classes, "
          f"{d['min_videos_per_class']}-{d['max_videos_per_class']} videos/class)")
    print(f"overall acc   : {100 * d['overall_accuracy']:.2f}%  ({d['correct']}/{d['n_test']})")
    print(f"average class : {100 * d['average_class_accuracy']:.2f}%")
    print(f"{ss['n_seeds']}-seed mean  : overall {100 * ss['overall_mean']:.2f}% +/- {100 * ss['overall_std']:.2f}  |  "
          f"average-class {100 * ss['average_mean']:.2f}% +/- {100 * ss['average_std']:.2f}")
    weak = sorted((c for c in report["per_class"] if c["support"]), key=lambda c: c["accuracy"])[:5]
    print("weakest words :", ", ".join(f"{c['word']} {c['correct']}/{c['support']}" for c in weak))
    for c in report["top_confusions"][:5]:
        print(f"confused      : {c['true']} -> {c['pred']}  x{c['count']}")
    for w in report["warnings"]:
        print("WARNING:", w)
    if a.plots:
        for p in make_plots(report, Path(a.plots)):
            print("plot          :", p)
    return report


if __name__ == "__main__":
    main()
