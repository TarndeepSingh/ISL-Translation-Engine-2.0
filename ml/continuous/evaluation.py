"""Corpus metrics for generated English using one or more supplied references."""
from __future__ import annotations

from typing import Sequence


METRIC_POLICY = {
    "bleu": "sacreBLEU corpus BLEU, score on 0..100; references are padded by repeating the final reference when counts vary",
    "chrf": "sacreBLEU corpus chrF, score on 0..100; references are padded by repeating the final reference when counts vary",
    "wer": "corpus token edit distance divided by reference token count; per example uses the reference with the lowest normalized token edit distance",
    "cer": "corpus character edit distance divided by reference character count; per example uses the reference with the lowest normalized character edit distance",
    "rouge_l": "mean per-example F1 of longest-common-subsequence; per example takes the maximum across references",
}


def _distance(left: Sequence[str], right: Sequence[str]) -> int:
    prior = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, prior[j] + 1, prior[j - 1] + (a != b)))
        prior = current
    return prior[-1]


def corpus_metrics(references: list[list[str]], hypotheses: list[str]) -> dict[str, float | dict[str, str]]:
    if len(references) != len(hypotheses) or not hypotheses:
        raise ValueError("each hypothesis must have at least one paired reference")
    if any(not any(str(ref).strip() for ref in group) for group in references):
        raise ValueError("empty reference sets are not scoreable")
    refs = [[str(ref).strip() for ref in group if str(ref).strip()] for group in references]
    hyps = [str(hyp).strip() for hyp in hypotheses]
    token_errors = token_count = char_errors = char_count = 0
    rouge_values = []
    for group, hyp in zip(refs, hyps):
        best_words = min(((_distance(ref.split(), hyp.split()), max(1, len(ref.split()))) for ref in group),
                         key=lambda pair: pair[0] / pair[1])
        token_errors += best_words[0]; token_count += best_words[1]
        best_chars = min(((_distance(list(ref), list(hyp)), max(1, len(ref))) for ref in group),
                         key=lambda pair: pair[0] / pair[1])
        char_errors += best_chars[0]; char_count += best_chars[1]
        rouge_values.append(max(_rouge_l(ref, hyp) for ref in group))
    ref_count = max(map(len, refs))
    ref_streams = [[group[min(i, len(group) - 1)] for group in refs] for i in range(ref_count)]
    import sacrebleu
    return {
        "bleu": float(sacrebleu.corpus_bleu(hyps, ref_streams).score),
        "chrf": float(sacrebleu.corpus_chrf(hyps, ref_streams).score),
        "wer": token_errors / max(1, token_count),
        "cer": char_errors / max(1, char_count),
        "rouge_l": sum(rouge_values) / len(rouge_values),
        "metric_policy": METRIC_POLICY,
    }


def _rouge_l(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref or not hyp:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for token in ref:
        current = [0]
        for j, candidate in enumerate(hyp, 1):
            current.append(previous[j - 1] + 1 if token == candidate else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
