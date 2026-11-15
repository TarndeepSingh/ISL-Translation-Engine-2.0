"""Pose + Hand Landmarker (MediaPipe Tasks) -> raw (75,3). SAME models as frontend/app.js,
so training features match live features. Feed UN-mirrored RGB frames."""
import os, urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from .config import N_RAW, POSE_SLICE, LH_SLICE, RH_SLICE
from .features import assign_hands

BASE = "https://storage.googleapis.com/mediapipe-models"
URLS = {"pose": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "hand": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"}


def model_path(kind, folder="models"):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, os.path.basename(URLS[kind]))
    if not os.path.exists(p):
        print("downloading", p); urllib.request.urlretrieve(URLS[kind], p)
    return p


def ensure_models():
    for k in URLS: model_path(k)


_arr = lambda lm: np.array([[p.x, p.y, p.z] for p in lm], np.float32)


class Landmarker:
    def __init__(self):
        B, RM = mpt.BaseOptions, vision.RunningMode.VIDEO
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=B(model_asset_path=model_path("pose")), running_mode=RM, num_poses=1))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=B(model_asset_path=model_path("hand")), running_mode=RM, num_hands=2))

    def process(self, rgb, ts_ms):
        """rgb uint8 (H,W,3); ts_ms strictly increasing. Returns raw (75,3), NaN where missing."""
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pr, hr = self.pose.detect_for_video(img, ts_ms), self.hand.detect_for_video(img, ts_ms)
        raw = np.full((N_RAW, 3), np.nan, np.float32)
        pose = _arr(pr.pose_landmarks[0]) if pr.pose_landmarks else None
        if pose is not None:
            raw[POSE_SLICE] = pose
        left, right = assign_hands(pose, [_arr(h) for h in hr.hand_landmarks])
        if left is not None: raw[LH_SLICE] = left
        if right is not None: raw[RH_SLICE] = right
        return raw
