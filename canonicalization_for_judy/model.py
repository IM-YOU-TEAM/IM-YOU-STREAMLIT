"""BioMAT architecture ported from MohsenSharifi1991/BioMAT, commit 0ca003db.

Preserves the unusual time-axis BatchNorm and flattened per-joint heads.
Replaces fastai's auto-initializing Module and uses mathematically equivalent
PyTorch SDPA for attention without materializing all attention matrices.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads, self.dk = heads, d_model // heads
        assert d_model % heads == 0
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = [layer(x).view(b, n, self.heads, self.dk).transpose(1, 2)
                   for layer in (self.W_Q, self.W_K, self.W_V)]
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.W_O(z.transpose(1, 2).contiguous().view(b, n, -1))


class EncoderLayer(nn.Module):
    def __init__(self, seq_len, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = Attention(d_model, heads)
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.BatchNorm1d(seq_len)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        x = self.batchnorm_attn(x + self.dropout_attn(self.self_attn(x)))
        return self.batchnorm_ffn(x + self.dropout_ffn(self.ff(x)))


class Encoder(nn.Module):
    def __init__(self, seq_len, d_model, heads, d_ff, dropout, layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(seq_len, d_model, heads, d_ff, dropout)
                                     for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BioMAT(nn.Module):
    def __init__(self, d_model=256, heads=16, d_ff=128, layers=4,
                 dropout=0.5, seq_len=256, features=18):
        super().__init__()
        self.W_P = nn.Linear(features, d_model)
        self.W_pos = nn.Parameter(torch.zeros(seq_len, d_model))
        self.res_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(seq_len, d_model, heads, d_ff, dropout, layers)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2_1 = nn.Linear(seq_len * d_model, seq_len)
        self.fc2_2 = nn.Linear(seq_len * d_model, seq_len)
        self.fc2_3 = nn.Linear(seq_len * d_model, seq_len)

    def forward(self, x):
        z = self.encoder(self.res_dropout(self.W_P(x) + self.W_pos))
        z = F.relu(self.fc_dropout(z.flatten(1)))
        return torch.stack([self.fc2_1(z), self.fc2_2(z), self.fc2_3(z)], dim=-1)
