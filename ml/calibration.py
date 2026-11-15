"""Temperature scaling fitted only on the model-selection validation split."""
from __future__ import annotations

import math
from typing import Any


def fit_temperature(logits: Any, labels: Any, *, max_iter: int = 100,
                   minimum_precision: float = 0.95, minimum_accepted: int = 5) -> dict[str, float | int | str]:
    import numpy as np

    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) < 2:
        raise ValueError("validation logits must be (N,C) and labels (N), with N >= 2")
    if not 0 < minimum_precision <= 1 or minimum_accepted < 1:
        raise ValueError("invalid abstention operating-point requirements")
    if not np.isfinite(x).all() or int(y.min()) < 0 or int(y.max()) >= x.shape[1]:
        raise ValueError("validation logits or labels are invalid")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    def nll(log_temperature: float) -> float:
        scaled = x / math.exp(log_temperature)
        row_max = scaled.max(axis=1, keepdims=True)
        log_partition = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1))
        return float(np.mean(log_partition - scaled[np.arange(len(y)), y]))

    before = nll(0.0)
    # Golden-section search is deterministic and avoids a torch optimizer for this
    # one-dimensional validation-only fit.
    left, right = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = nll(c), nll(d)
    for _ in range(max_iter):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = nll(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = nll(d)
    temperature = math.exp((left + right) / 2.0)
    after = nll(math.log(temperature))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature optimizer produced an invalid value")
    scaled = x / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == y
    threshold = 1.000001
    accepted, precision = 0, 0.0
    for candidate in sorted({float(value) for value in confidence}, reverse=True):
        selected = confidence >= candidate
        count = int(selected.sum())
        hit_rate = float(correct[selected].mean())
        if count >= minimum_accepted and hit_rate >= minimum_precision and count > accepted:
            threshold, accepted, precision = candidate, count, hit_rate
    return {"method": "scalar_temperature_scaling", "temperature": temperature,
            "validation_samples": int(len(y)), "nll_before": before, "nll_after": after,
            "fitted_on_split": "val", "minimum_precision_target": minimum_precision,
            "minimum_accepted_target": minimum_accepted, "confidence_threshold": threshold,
            "validation_accepted": accepted, "validation_precision": precision}
