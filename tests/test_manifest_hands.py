import numpy as np
from isl.features import assign_hands
from isl.hf_manifest import build_rows, index_videos


def _pose():
    p = np.full((33, 3), np.nan); p[15] = [0.6, 0.6, 0]; p[16] = [0.4, 0.6, 0]; return p


def _hand(x, y): h = np.zeros((21, 3)); h[:, 0] = x; h[:, 1] = y; return h


def test_assign_two_hands_any_order():
    a, b = _hand(0.61, 0.6), _hand(0.39, 0.6)
    for hs in ([a, b], [b, a]):
        l, r = assign_hands(_pose(), hs); assert l[0, 0] == 0.61 and r[0, 0] == 0.39


def test_assign_single_hand_and_no_pose_fallback():
    l, r = assign_hands(_pose(), [_hand(0.41, 0.6)]); assert l is None and r is not None
    l, r = assign_hands(None, [_hand(0.8, 0.5)]); assert l is not None and r is None


def test_manifest_matching(tmp_path):
    v = tmp_path / "Colours" / "50. Yellow"; v.mkdir(parents=True); (v / "a.MOV").write_bytes(b"x")
    recs = [dict(split="train", label="50. Yellow", video_path="Colours/50. Yellow/a.MOV"),
            dict(split="test", label="35. Bank", video_path="Colours/50. Yellow/a.MOV"),      # inconsistent
            dict(split="test", label="50. Yellow", video_path="Colours/50. Yellow/zzz.MOV")]  # missing
    rows, missing, bad = build_rows(recs, index_videos(str(tmp_path)))
    assert len(rows) == 1 and rows[0][1:] == ("yellow", "train") and missing == 1 and bad == 1
