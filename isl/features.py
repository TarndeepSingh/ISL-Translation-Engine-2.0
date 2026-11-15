"""Scale/translation-invariant feature engineering (Module 1 core). Pure numpy."""
import numpy as np
from .config import *

_SWAP = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]


def mirror_raw(raw: np.ndarray) -> np.ndarray:
    """Horizontal flip for left-handed-signer augmentation (swaps hands + pose sides)."""
    r = raw.copy()
    r[..., 0] = 1.0 - r[..., 0]
    out = r.copy()
    for a, b in _SWAP:
        out[:, a], out[:, b] = r[:, b], r[:, a]
    out[:, LH_SLICE], out[:, RH_SLICE] = r[:, RH_SLICE], r[:, LH_SLICE]
    return out


def rotate_raw(raw: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate the image-plane (x, y) of every landmark by ``degrees`` (camera-tilt augmentation).

    Translation is removed later by shoulder-centred normalisation, so rotating about the origin is
    equivalent to rotating about the shoulder midpoint. Coordinates are normalised image units, so
    for non-square frames this is an approximation, adequate for the small angles used (a few degrees).
    NaNs (undetected landmarks) stay NaN; z is untouched.
    """
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    out = raw.copy()
    x, y = raw[..., 0], raw[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def time_warp_raw(raw: np.ndarray, rng, strength: float) -> np.ndarray:
    """Non-uniform speed change: same length, first/last frame kept, frames repeated/skipped smoothly.

    Each inter-frame step is drawn from U(1-strength, 1+strength) and the cumulative positions are
    rescaled to span the clip, so some parts play faster and others slower. Nearest-frame sampling is
    used so undetected-landmark NaNs are never blended into valid values. ``rng`` is random.Random.
    """
    n = len(raw)
    if strength <= 0 or n < 4:
        return raw
    strength = min(float(strength), 0.9)
    steps = np.array([rng.uniform(1 - strength, 1 + strength) for _ in range(n - 1)])
    pos = np.concatenate([[0.0], np.cumsum(steps)])
    idx = np.rint(pos / pos[-1] * (n - 1)).astype(int)
    return raw[idx]


def _fill_anchor(mid, span):
    """Forward/back-fill frames where shoulders were not detected."""
    valid = np.isfinite(span) & (span > 1e-3) & np.isfinite(mid).all(1)
    n = len(mid)
    if not valid.any():
        return np.tile([0.5, 0.5, 0.0], (n, 1)), np.full(n, 0.3)
    idx = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
    idx[idx < 0] = int(np.argmax(valid))
    return mid[idx], span[idx]


def _hand(h):
    L = h.shape[0]
    present = np.isfinite(h).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        palm = np.maximum(np.linalg.norm((h[:, 9] - h[:, 0])[:, :2], axis=1), 1e-3)
        out = (h - h[:, :1]) / palm[:, None, None]
    out = np.where(present[:, None, None], out, 0.0)
    return np.clip(out, -5, 5).reshape(L, 63), present.astype(np.float32)


def normalize(raw: np.ndarray) -> np.ndarray:
    """raw (L,75,3) with NaNs -> features (L,170)."""
    L = raw.shape[0]
    pose = raw[:, POSE_SLICE][:, POSE_IDX]                       # (L,12,3)
    ls, rs = pose[:, 0], pose[:, 1]
    with np.errstate(invalid="ignore"):
        mid = (ls + rs) / 2
        span = np.linalg.norm((ls - rs)[:, :2], axis=1)
    mid, span = _fill_anchor(mid, span)
    pose_n = np.nan_to_num((pose - mid[:, None]) / span[:, None, None])
    pose_n = np.clip(pose_n, -8, 8)
    lh, fl = _hand(raw[:, LH_SLICE])
    rh, fr = _hand(raw[:, RH_SLICE])
    wrist = np.concatenate([pose_n[:, 4], pose_n[:, 5]], axis=1)  # L wrist, R wrist
    feats = np.concatenate(
        [pose_n.reshape(L, 36), lh, rh, wrist, fl[:, None], fr[:, None]], axis=1)
    assert feats.shape[1] == FEATURE_DIM
    return feats.astype(np.float32)


def resample(x: np.ndarray, T: int = T) -> np.ndarray:
    """Uniform linear-interpolation resample (L,D)->(T,D)."""
    L = len(x)
    if L == 1:
        return np.repeat(x, T, 0).astype(np.float32)
    src = np.linspace(0, L - 1, T)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, L - 1)
    f = (src - i0)[:, None]
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)


def assign_hands(pose, hands):
    """Assign detected hands to the PERSON's left/right by proximity to pose wrists (15=L, 16=R).
    pose: (33,3) or None; hands: list of (21,3). Returns (left, right); same logic as frontend/app.js."""
    left = right = None
    if not hands:
        return left, right
    d = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    if pose is None or not np.isfinite(pose[15:17, :2]).all():
        for h in hands[:2]:                      # unmirrored frame: person's left appears on image right
            if h[0][0] > 0.5:
                left = h if left is None else left
            else:
                right = h if right is None else right
        return left, right
    L, R = pose[15], pose[16]
    if len(hands) == 1:
        h = hands[0]
        return (h, None) if d(h[0], L) <= d(h[0], R) else (None, h)
    a, b = hands[0], hands[1]
    return (a, b) if d(a[0], L) + d(b[0], R) <= d(b[0], L) + d(a[0], R) else (b, a)
