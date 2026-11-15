import numpy as np
import pytest

from ml.calibration import fit_temperature
from ml.continuous.model import VideoToTextTransformer


def test_validation_temperature_and_precision_threshold_are_finite():
    logits = np.asarray([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0],
                         [0.0, 4.0], [0.0, 3.0], [0.0, 2.0]], dtype=np.float32)
    result = fit_temperature(logits, np.asarray([0, 0, 0, 1, 1, 1]),
                             minimum_precision=0.95, minimum_accepted=5)
    assert result["fitted_on_split"] == "val"
    assert np.isfinite(result["temperature"]) and result["temperature"] > 0
    assert 0 <= result["confidence_threshold"] <= 1.000001
    assert result["validation_accepted"] == 6


def test_calibration_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        fit_temperature([[1.0, 0.0]], [0])


def test_continuous_transformer_output_and_greedy_decode_shapes():
    import torch

    model = VideoToTextTransformer(input_dim=6, vocabulary_size=9, dimension=16,
                                   heads=2, encoder_layers=1, decoder_layers=1,
                                   feedforward=32, dropout=0.0, max_frames=4,
                                   max_tokens=5, pad_id=0)
    frames = torch.zeros((2, 4, 6))
    target_in = torch.tensor([[1, 4], [1, 5]])
    assert model(frames, target_in).shape == (2, 2, 9)
    assert model.generate(frames[:1], bos_id=1, eos_id=2).shape[0] == 1
