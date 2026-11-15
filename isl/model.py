import torch, torch.nn as nn
from .config import FEATURE_DIM


class ISLNet(nn.Module):
    """Conv1D -> BiLSTM -> MHSA(+residual, LayerNorm) -> mean+max pool -> classes."""
    def __init__(self, in_dim=FEATURE_DIM, n_classes=51, conv=128, hidden=128,
                 heads=4, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, conv, 3, padding=1),
                                  nn.BatchNorm1d(conv), nn.GELU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(conv, hidden, batch_first=True, bidirectional=True)
        d = hidden * 2                                   # 256 -> 4 heads x key_dim 64
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(d * 2, d))
        self.n2 = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d * 2, n_classes))

    def forward(self, x):                                # x: (B,T,F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.n1(h + a)
        h = self.n2(h + self.ff(h))
        return self.head(torch.cat([h.mean(1), h.amax(1)], -1))
