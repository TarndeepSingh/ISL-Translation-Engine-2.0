import numpy as np
from isl.config import N_RAW
from isl.diagnose import live_predict, offline_predict, summarize


def raw_sign(n_down=15, n_move=30):
    L = n_down + n_move + 20
    raw = np.full((L, N_RAW, 3), np.nan, np.float32)
    raw[:, 11] = [0.6, 0.4, 0]; raw[:, 12] = [0.4, 0.4, 0]; raw[:, 16] = [0.4, 0.9, 0]
    for t in range(L):
        moving = n_down <= t < n_down + n_move
        raw[t, 15] = [0.6 + (0.05 * (-1) ** t if moving else 0), 0.4 if moving else 0.9, 0]
    return raw


stub = lambda clip: ("hello", 0.9, 0.8, [("hello", 0.9)])


def test_live_finds_one_sign_and_offline_runs():
    raw = raw_sign()
    assert len(live_predict(raw, stub)) == 1
    assert offline_predict(raw, stub)[0] == "hello"


def test_summary_counts():
    rows = [{"label": "hello", "off": ("hello", 1, 1, []), "live": [("hello", 1, 1, [])]},
            {"label": "bye", "off": ("hello", 1, 1, []), "live": []}]
    assert summarize(rows) == {"videos": 2, "offline_correct": 1, "live_found_sign": 1, "live_first_correct": 1}
