"""Small visual-sequence-to-English Transformer baseline (PyTorch)."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, max_length: int):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
        table = torch.zeros(max_length, dimension)
        table[:, 0::2] = torch.sin(position * scale)
        table[:, 1::2] = torch.cos(position * scale[:table[:, 1::2].shape[1]])
        self.register_buffer("table", table.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.table[:, :values.shape[1]])


class VideoToTextTransformer(nn.Module):
    """Encode framewise visual features, decode English tokens autoregressively."""
    def __init__(self, input_dim: int, vocabulary_size: int, *, dimension: int = 128,
                 heads: int = 4, encoder_layers: int = 3, decoder_layers: int = 3,
                 feedforward: int = 512, dropout: float = 0.1,
                 max_frames: int = 256, max_tokens: int = 100, pad_id: int = 0):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by attention heads")
        self.max_tokens, self.pad_id = max_tokens, pad_id
        self.input_projection = nn.Linear(input_dim, dimension)
        self.source_position = PositionEncoding(dimension, dropout, max_frames)
        self.target_embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=pad_id)
        self.target_position = PositionEncoding(dimension, dropout, max_tokens)
        enc = nn.TransformerEncoderLayer(dimension, heads, feedforward, dropout,
                                         batch_first=True, norm_first=True, activation="gelu")
        dec = nn.TransformerDecoderLayer(dimension, heads, feedforward, dropout,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, encoder_layers, norm=nn.LayerNorm(dimension))
        self.decoder = nn.TransformerDecoder(dec, decoder_layers, norm=nn.LayerNorm(dimension))
        self.output = nn.Linear(dimension, vocabulary_size)

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        values = self.source_position(self.input_projection(frames))
        return self.encoder(values)

    def forward(self, frames: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        memory = self.encode_video(frames)
        target = self.target_position(self.target_embedding(target_input))
        length = target_input.shape[1]
        causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=frames.device), diagonal=1)
        padding_mask = target_input.eq(self.pad_id)
        decoded = self.decoder(target, memory, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
        return self.output(decoded)

    @torch.inference_mode()
    def generate(self, frames: torch.Tensor, bos_id: int, eos_id: int) -> torch.Tensor:
        self.eval()
        memory = self.encode_video(frames)
        result = torch.full((frames.shape[0], 1), bos_id, dtype=torch.long, device=frames.device)
        finished = torch.zeros(frames.shape[0], dtype=torch.bool, device=frames.device)
        for _ in range(self.max_tokens - 1):
            target = self.target_position(self.target_embedding(result))
            length = result.shape[1]
            mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=frames.device), diagonal=1)
            decoded = self.decoder(target, memory, tgt_mask=mask, tgt_key_padding_mask=result.eq(self.pad_id))
            next_token = self.output(decoded[:, -1]).argmax(-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            result = torch.cat((result, next_token[:, None]), dim=1)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        return result
