import csv, json, os
import numpy as np
from isl.config import N_RAW
from ml.datasets import manifest as M
from ml.datasets.apply_vocab import apply_vocab
from ml.datasets.make_idle_cache import make_idle, rest_windows
from ml.experiments import cv_word as C


def synth_raw(cls, session, L=60, rng=None):
    """Class-specific wrist trajectory + session-specific camera offset; shoulders fixed; one rest segment at the end."""
    raw = np.full((L, N_RAW, 3), np.nan, np.float32)
    raw[:, 11] = [0.6, 0.4, 0]; raw[:, 12] = [0.4, 0.4, 0]; raw[:, 16] = [0.4, 0.95, 0]
    t = np.linspace(0, 2 * np.pi, L)
    raw[:, 15, 0] = 0.6 + 0.08 * np.sin((cls + 1) * t) + 0.02 * session
    raw[:, 15, 1] = 0.45 + 0.05 * np.cos((cls + 2) * t)
    raw[:, 15, 2] = 0
    raw[-30:, 15] = [0.6, 0.98, 0]                                    # hand lowered and still at the end
    return raw + (rng.normal(0, 0.003, raw.shape).astype(np.float32) if rng is not None else 0)


def make_dataset(tmp, n_cls=4, n_sess=6, per=2):
    rng = np.random.default_rng(0); rows = []; cache = tmp / "cache"; cache.mkdir()
    for c in range(n_cls):
        for s in range(n_sess):
            for k in range(per):
                path = f"/v/{c}_{s}_{k}.mov"
                r = M.row("syn", f"{c}_{s}_{k}", path, "isolated", label=f"w{c}", session_id=f"sess{s}")
                np.savez_compressed(C.cache_path(path, str(cache)), raw=synth_raw(c, s, rng=rng)); rows.append(r)
    return rows, str(cache)


def test_rest_windows_found_only_at_rest():
    w = rest_windows(synth_raw(1, 0))
    assert w and all(b - a >= 15 for a, b in w) and w[0][0] >= 30 - 1


def test_idle_inherits_split_and_drops_recorded_idle(tmp_path):
    rows, cache = make_dataset(tmp_path)
    idx = [{"cache": C.cache_path(r["path"], cache), "label": r["label"], "split": "train" if int(r["sample_id"].split("_")[1]) < 4 else "test"} for r in rows]
    made = make_idle(idx, cache, per_video=1, cap_per_split=5)
    assert made and {m["split"] for m in made} <= {"train", "test"} and all(m["label"] == "idle" for m in made)
    assert sum(m["split"] == "test" for m in made) <= 5 and all(os.path.exists(m["cache"]) for m in made)


def test_cv_runs_and_grouped_not_better_than_random(tmp_path):
    rows, cache = make_dataset(tmp_path)
    X, y, g, miss = C.load_arrays(rows, cache)
    assert X.shape == (len(rows), 32, 170) and miss == 0
    rnd = C.run_cv(X, y, g, "knn", 3, grouped=False); grp = C.run_cv(X, y, g, "knn", 3, grouped=True)
    assert 0 <= grp["mean"] <= 1 and rnd["mean"] >= grp["mean"] - 0.05 and rnd["mean"] > 1 / 4
    assert set(grp["recall"]) == {"w0", "w1", "w2", "w3"}


def test_reliable_words_and_apply_vocab():
    assert C.reliable_words({"a": .9, "b": .5, "c": .95, "idle": 1.0}, {"a": 10, "b": 10, "c": 2, "idle": 9}) == ["a"]
    rows = [M.row("d", "1", "p1", "isolated", label="a"), M.row("d", "2", "p2", "isolated", label="b"), M.row("d", "3", "p3", "isolated", label="idle")]
    out = apply_vocab(rows, ["a"])
    assert [r["label"] for r in out] == ["a", "unknown_sign", "idle"] and out[1]["text"] == "b"
