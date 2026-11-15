"""Metrics. Classification (top-k, macro P/R/F1, confusion, ECE, selective accuracy) and text (WER, BLEU, chrF, ROUGE-L).
A metric is only reported when its inputs exist; callers must not fabricate references."""
import numpy as np


def topk_hit(true_label, ranked_labels, k):
    return true_label in ranked_labels[:k]


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), np.int64)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    return cm


def per_class_report(cm, labels):
    rep, tp = {}, np.diag(cm)
    for i, l in enumerate(labels):
        sup, pred = int(cm[i].sum()), int(cm[:, i].sum())
        p = tp[i] / pred if pred else 0.0; r = tp[i] / sup if sup else 0.0
        rep[l] = dict(precision=float(p), recall=float(r), f1=float(2 * p * r / (p + r)) if p + r else 0.0, support=sup)
    present = [v for v in rep.values() if v["support"] > 0]
    macro = {k: float(np.mean([v[k] for v in present])) if present else 0.0 for k in ("precision", "recall", "f1")}
    return rep, macro


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0: return 0.0
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any(): e += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def selective_curve(conf, correct, thresholds=(0.0, 0.3, 0.5, 0.7, 0.9)):
    conf, correct, out = np.asarray(conf, float), np.asarray(correct, float), []
    for t in thresholds:
        m = conf >= t
        out.append(dict(threshold=t, coverage=float(m.mean()) if len(m) else 0.0,
                        accuracy=float(correct[m].mean()) if m.any() else None))
    return out


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r: raise ValueError("empty reference: WER undefined")
    return _lev(r, h) / len(r)


def corpus_wer(refs, hyps):
    errs = sum(_lev(r.split(), h.split()) for r, h in zip(refs, hyps)); n = sum(len(r.split()) for r in refs)
    if n == 0: raise ValueError("empty references")
    return errs / n


def bleu(refs, hyps):
    import sacrebleu
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def chrf(refs, hyps):
    import sacrebleu
    return float(sacrebleu.corpus_chrf(hyps, [refs]).score)


def rouge_l(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r or not h: return 0.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if r[i - 1] == h[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    l = dp[-1][-1]
    if l == 0: return 0.0
    p, rc = l / len(h), l / len(r)
    return 2 * p * rc / (p + rc)
