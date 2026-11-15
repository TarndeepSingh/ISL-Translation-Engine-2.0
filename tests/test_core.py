import numpy as np
from isl.config import *
from isl.features import normalize, resample, mirror_raw
from isl.spotter import SignSegmenter


def fake_raw(L=40, seed=0):
    r = np.random.RandomState(seed)
    raw = r.rand(L, N_RAW, 3).astype(np.float32) * 0.3 + 0.35
    raw[:, 11] = [0.6, 0.5, 0]; raw[:, 12] = [0.4, 0.5, 0]      # shoulders 0.2 apart
    return raw


def test_shapes():
    f = normalize(fake_raw()); assert f.shape == (40, FEATURE_DIM)
    assert resample(f).shape == (T, FEATURE_DIM)


def test_scale_translation_invariance():
    raw = fake_raw(); a = normalize(raw)
    b = normalize(raw * 1.7 + np.array([0.1, -0.2, 0.0], np.float32))
    assert np.allclose(a, b, atol=1e-3)


def test_missing_hand_is_flagged_not_crashing():
    raw = fake_raw(); raw[:, RH_SLICE] = np.nan
    f = normalize(raw); assert np.isfinite(f).all() and f[:, 169].sum() == 0 and f[:, 168].sum() == 40


def test_mirror_involution():
    raw = fake_raw(); assert np.allclose(mirror_raw(mirror_raw(raw)), raw, atol=1e-6)


def test_segmenter_emits_one_segment():
    seg, out = SignSegmenter(), []
    def frame(y, x):
        f = np.zeros(FEATURE_DIM, np.float32); f[162:165] = [x, y, 0]; f[165:168] = [-0.5, 2.0, 0]; return f
    seq = [frame(2.0, 0.5)] * 20 + [frame(0.3, 0.5 + 0.15 * (-1) ** i) for i in range(30)] + [frame(2.0, 0.5)] * 20
    for f in seq:
        s = seg.update(f)
        if s is not None: out.append(s)
    assert len(out) == 1 and 25 <= len(out[0]) <= 45


def test_idle_never_triggers():
    seg = SignSegmenter(); r = np.random.RandomState(0)
    for _ in range(300):
        f = np.zeros(FEATURE_DIM, np.float32); f[162:168] = [0.5, 2.0, 0, -0.5, 2.0, 0] + r.randn(6) * 0.002
        assert seg.update(f) is None
