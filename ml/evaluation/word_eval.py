"""Automated evaluation of the word recogniser on HELD-OUT manifest rows.
python -m ml.evaluation.word_eval --manifest data/unified_manifest.csv --split test --model models/isl.onnx --out reports/word_eval
Writes: predictions.csv, classification_report.json, confusion_matrix.png, error_analysis.html, eval_summary.json
The official test split must not be used for tuning; run it once per finished model."""
import argparse, csv, html, json, os
import numpy as np
from .metrics import confusion_matrix, per_class_report, ece, selective_curve

SPLIT_WARNING = ("Dataset test videos come from the same few signers/sessions as training (INCLUDE has about 7 signers). "
                 "This score is NOT evidence of performance on a new person or a new camera; "
                 "use a session-grouped split or fresh recordings for that.")


def evaluate_rows(rows, predict_fn, labels, out_dir, conf_threshold=0.5, split_note=SPLIT_WARNING):
    os.makedirs(out_dir, exist_ok=True)
    l2i, recs = {l: i for i, l in enumerate(labels)}, []
    for r in rows:
        base = dict(sample_id=r["sample_id"], dataset=r.get("dataset", ""), true=r["label"], path=r["path"])
        try:
            p = predict_fn(r["path"])
        except Exception as e:
            recs.append({**base, "error": f"{type(e).__name__}: {e}"}); continue
        top = [t[0] for t in p["top"]]
        recs.append({**base, "pred": p["label"], "conf": p["conf"], "status": p["status"], "ms": p["ms"],
                     "top": top, "in_vocab": r["label"] in l2i, "correct": p["label"] == r["label"],
                     "top3_hit": r["label"] in top[:3], "top5_hit": r["label"] in top[:5],
                     "uncertain": p["conf"] < conf_threshold})
    ok = [x for x in recs if "error" not in x]
    valid = [x for x in ok if x["in_vocab"]]
    n = len(valid)
    summary = {"note": split_note, "n_requested": len(rows), "n_errors": len(recs) - len(ok),
               "n_out_of_vocabulary": len(ok) - n, "n_scored": n}
    if n:
        y = np.array([l2i[x["true"]] for x in valid]); pr = np.array([l2i[x["pred"]] for x in valid])
        cm = confusion_matrix(y, pr, len(labels)); rep, macro = per_class_report(cm, labels)
        conf, cor = [x["conf"] for x in valid], [x["correct"] for x in valid]
        summary.update(top1=float(np.mean(cor)), top3=float(np.mean([x["top3_hit"] for x in valid])),
                       top5=float(np.mean([x["top5_hit"] for x in valid])), macro_precision=macro["precision"],
                       macro_recall=macro["recall"], macro_f1=macro["f1"], ece=ece(conf, cor),
                       selective=selective_curve(conf, cor), mean_ms=float(np.mean([x["ms"] for x in valid])))
        json.dump(rep, open(os.path.join(out_dir, "classification_report.json"), "w"), indent=1)
        _plot_cm(cm, labels, os.path.join(out_dir, "confusion_matrix.png"))
    oov = [x for x in ok if not x["in_vocab"]]
    if oov:
        summary["out_of_vocab_rejected_rate"] = float(np.mean([x["uncertain"] or x["status"] != "confident" for x in oov]))
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "true", "pred", "conf", "status", "correct", "top5", "ms", "error"])
        for x in recs:
            w.writerow([x["sample_id"], x["true"], x.get("pred", ""), x.get("conf", ""), x.get("status", ""),
                        x.get("correct", ""), "|".join(x.get("top", [])), x.get("ms", ""), x.get("error", "")])
    json.dump(summary, open(os.path.join(out_dir, "eval_summary.json"), "w"), indent=1)
    _html(recs, summary, os.path.join(out_dir, "error_analysis.html"))
    return summary, recs


def _plot_cm(cm, labels, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = [i for i in range(len(labels)) if cm[i].sum() or cm[:, i].sum()]
    m = cm[np.ix_(idx, idx)].astype(float); m = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(5, len(idx) * .28), max(5, len(idx) * .28)))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xticks(range(len(idx))); ax.set_yticks(range(len(idx)))
    ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=6); ax.set_yticklabels([labels[i] for i in idx], fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _html(recs, s, path):
    e = html.escape
    def tbl(items, title):
        if not items: return f"<h2>{e(title)}</h2><p>none</p>"
        rows = "".join(f"<tr><td>{e(x['sample_id'])}</td><td>{e(x['true'])}</td><td>{e(x.get('pred',''))}</td>"
                       f"<td>{x.get('conf','')}</td><td>{e(', '.join(x.get('top', [])[:3]))}</td><td>{e(x.get('error',''))}</td></tr>" for x in items)
        return f"<h2>{e(title)} ({len(items)})</h2><table><tr><th>sample</th><th>true</th><th>pred</th><th>conf</th><th>top-3</th><th>error</th></tr>{rows}</table>"
    wrong = [x for x in recs if "error" not in x and x["in_vocab"] and not x["correct"]]
    low = [x for x in recs if "error" not in x and x["in_vocab"] and x["correct"] and x["uncertain"]]
    oov = [x for x in recs if "error" not in x and not x["in_vocab"]]
    errs = [x for x in recs if "error" in x]
    head = "".join(f"<li><b>{k}</b>: {v:.4f}</li>" if isinstance(v, float) else f"<li><b>{k}</b>: {e(str(v))}</li>"
                   for k, v in s.items() if k != "selective")
    open(path, "w", encoding="utf-8").write(
        "<!doctype html><meta charset=utf-8><title>Error analysis</title><style>body{font:14px system-ui;margin:24px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px}</style><h1>Word recognition: error analysis</h1>"
        f"<ul>{head}</ul>{tbl(wrong, 'Incorrect predictions')}{tbl(low, 'Correct but low confidence')}"
        f"{tbl(oov, 'Out-of-vocabulary / unknown examples')}{tbl(errs, 'Videos that failed to process')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="models/isl.onnx"); ap.add_argument("--out", default="reports/word_eval")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    from ml.datasets.manifest import read_manifest
    from ml.inference.pipeline import WordPipeline
    from backend.engine import Predictor
    P = Predictor(a.model); pipe = WordPipeline(P, tau=a.tau)
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split and r["task"] == "isolated"]
    if a.limit: rows = rows[:a.limit]
    s, _ = evaluate_rows(rows, pipe.predict_video, P.labels, a.out, a.tau)
    print(json.dumps({k: v for k, v in s.items() if k != "selective"}, indent=1))
