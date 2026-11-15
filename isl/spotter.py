"""Kinetic gesture spotter. Segments the live stream into sign candidates so that
inference sees a *whole sign resampled to T frames* -- exactly what training sees."""
from collections import deque
import numpy as np
from .config import F_WRIST


class SignSegmenter:
    def __init__(self, v_thresh=0.02, y_raised=0.9, start_frames=3,
                 end_frames=8, min_len=10, max_len=90, pad=3):
        self.v_thresh, self.y_raised = v_thresh, y_raised
        self.start_frames, self.end_frames = start_frames, end_frames
        self.min_len, self.max_len = min_len, max_len
        self.pre = deque(maxlen=pad + start_frames)
        self.reset()
        self.speed = 0.0

    def reset(self):
        self.active, self.run, self.still, self.frames, self.prev = False, 0, 0, [], None
        self.pre.clear()

    def update(self, feat):
        """Feed one (170,) feature frame. Returns (L,170) segment when a sign ends."""
        wr = feat[F_WRIST]
        cur = np.stack([wr[:3], wr[3:]])                 # y grows downward, shoulders at 0
        self.speed = 0.0 if self.prev is None else float(
            np.linalg.norm(cur[:, :2] - self.prev[:, :2], axis=1).max())
        self.prev = cur
        live = cur[:, 1].min() < self.y_raised and self.speed > self.v_thresh
        if not self.active:
            self.pre.append(feat)
            self.run = self.run + 1 if live else 0
            if self.run >= self.start_frames:
                self.active, self.frames, self.still, self.run = True, list(self.pre), 0, 0
            return None
        self.frames.append(feat)
        self.still = 0 if live else self.still + 1
        if self.still >= self.end_frames or len(self.frames) >= self.max_len:
            keep = len(self.frames) - max(self.still - 2, 0)
            seg = np.stack(self.frames[:keep])
            self.active, self.frames, self.still, self.run = False, [], 0, 0
            self.pre.clear()
            return seg if len(seg) >= self.min_len else None
        return None
