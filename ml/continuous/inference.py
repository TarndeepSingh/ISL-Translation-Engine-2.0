"""Trusted local continuous model package loader and offline video adapter."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from isl.extract import video_to_raw
from isl.features import normalize, resample
from .model import VideoToTextTransformer
from .text import BOS, EOS, decode


class ContinuousPipeline:
    def __init__(self, model, vocabulary, model_version, max_frames):
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.model_version = model_version
        self.max_frames = int(max_frames)
        self.device = next(model.parameters()).device

    def translate_video(self, path: str) -> dict[str, object]:
        started = time.perf_counter()
        raw = video_to_raw(path)
        if raw.ndim != 3 or len(raw) < 5:
            raise ValueError("video contains too few decodable frames")
        frames = torch.from_numpy(resample(normalize(raw), self.max_frames)[None]).to(self.device)
        with torch.inference_mode():
            tokens = self.model.generate(frames, self.vocabulary[BOS], self.vocabulary[EOS])[0].cpu().tolist()
        return {"text": decode(tokens, self.vocabulary),
                "ms": round((time.perf_counter() - started) * 1000, 1)}


def load_translator(package_path: str) -> ContinuousPipeline:
    """Load a local, trusted checkpoint exported by ml.continuous.train."""
    if not os.path.isfile(package_path):
        raise FileNotFoundError("continuous model package does not exist")
    checkpoint = torch.load(package_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    vocabulary = {str(token): int(index) for token, index in checkpoint["vocabulary"].items()}
    model = VideoToTextTransformer(**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model_version = str(checkpoint.get("model_version", os.path.basename(os.path.dirname(package_path)) or "unversioned"))
    return ContinuousPipeline(model, vocabulary, model_version, config["max_frames"])
