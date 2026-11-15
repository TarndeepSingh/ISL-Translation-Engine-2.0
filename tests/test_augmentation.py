"""Tests for the opt-in rotation / time-warp augmentations (synthetic landmark arrays only)."""
from __future__ import annotations

import random

import numpy as np

from isl.config import LH_SLICE
from isl.features import normalize, rotate_raw, time_warp_raw
from isl.train import augment_raw


def synthetic_raw(n=40, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.2, 0.8, (n, 75, 3)).astype(np.float64)
    raw[5, LH_SLICE] = np.nan                       # an undetected hand in one frame
    return raw


def test_rotation_zero_is_identity_and_inverse_round_trips():
    raw = synthetic_raw()
    np.testing.assert_allclose(rotate_raw(raw, 0.0), raw, equal_nan=True)
    np.testing.assert_allclose(rotate_raw(rotate_raw(raw, 7.0), -7.0), raw, atol=1e-12, equal_nan=True)


def test_rotation_keeps_nans_z_and_input_untouched():
    raw = synthetic_raw()
    before = raw.copy()
    out = rotate_raw(raw, 5.0)
    np.testing.assert_array_equal(raw, before)       # no in-place mutation
    assert np.array_equal(np.isnan(out), np.isnan(raw))
    np.testing.assert_array_equal(out[..., 2], raw[..., 2])


def test_rotation_preserves_distances_between_landmarks():
    raw = synthetic_raw()
    out = rotate_raw(raw, 7.0)
    d_before = np.linalg.norm(raw[:, 0, :2] - raw[:, 1, :2], axis=1)
    d_after = np.linalg.norm(out[:, 0, :2] - out[:, 1, :2], axis=1)
    np.testing.assert_allclose(d_before, d_after, atol=1e-12)


def test_rotation_changes_features_only_slightly_for_small_angles():
    raw = synthetic_raw()
    a, b = normalize(raw), normalize(rotate_raw(raw, 3.0))
    assert a.shape == b.shape == (40, 170)
    assert np.isfinite(b).all() and not np.allclose(a, b)


def test_time_warp_keeps_length_endpoints_and_order():
    raw = synthetic_raw(60)
    out = time_warp_raw(raw, random.Random(1), 0.4)
    assert out.shape == raw.shape
    np.testing.assert_array_equal(out[0], raw[0])
    np.testing.assert_array_equal(out[-1], raw[-1])
    frame_ids = [int(np.argmin(np.abs(raw[:, 0, 0] - f[0, 0]))) for f in out[:, None, 0]]
    assert frame_ids == sorted(frame_ids)            # time never runs backwards


def test_time_warp_is_seeded_and_off_by_default():
    raw = synthetic_raw(60)
    a = time_warp_raw(raw, random.Random(3), 0.4)
    b = time_warp_raw(raw, random.Random(3), 0.4)
    np.testing.assert_array_equal(a, b)
    assert time_warp_raw(raw, random.Random(3), 0.0) is raw
    assert time_warp_raw(raw[:3], random.Random(3), 0.5).shape[0] == 3


def test_time_warp_never_blends_nans_into_valid_frames():
    raw = synthetic_raw(60)
    out = time_warp_raw(raw, random.Random(5), 0.6)
    for frame in out:
        assert np.isnan(frame[LH_SLICE]).all() or np.isfinite(frame[LH_SLICE]).all()


def test_defaults_do_not_consume_randomness_so_old_seeded_runs_are_unchanged():
    raw = synthetic_raw(60)
    r1, r2 = random.Random(9), random.Random(9)
    base = augment_raw(raw, r1, "sign")
    same = augment_raw(raw, r2, "sign", False, 0.0, 0.0)
    np.testing.assert_array_equal(base, same)
    assert r1.random() == r2.random()                # generators still in lock-step


def test_enabling_options_changes_output_and_stays_valid():
    raw = synthetic_raw(60)
    out = augment_raw(raw, random.Random(9), "sign", False, 7.0, 0.3)
    assert out.shape[1:] == raw.shape[1:] and out.shape[0] >= 8
    feats = normalize(out)
    assert np.isfinite(feats).all()
