"""Encoder and heads. One compact 1-D multi-scale convolutional encoder serves every scenario (vibration windows
(1, 1024), log-Mel (64, 313), C-MAPSS (14, 30)); the novelty and the decision logic sit on top of its embedding.
Heads: softmax classifier (with dropout for Monte Carlo sampling), evidential (Dirichlet) classifier, and a
three-quantile regression head for remaining useful life.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) if (stride != 1 or cin != cout) else nn.Identity()

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))


class Encoder(nn.Module):
    """Wide first kernel (multi-scale view of the raw window) followed by three residual stages and global pooling."""

    def __init__(self, in_ch=1, emb=128, width=32, first_kernel=64, first_stride=8):
        super().__init__()
        k, s = (first_kernel, first_stride) if in_ch == 1 else (7, 1)
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, k, s, k // 2, bias=False), nn.BatchNorm1d(width), nn.ReLU(),
                                  nn.MaxPool1d(2) if in_ch == 1 else nn.Identity())
        self.stages = nn.Sequential(Block(width, width, 1), Block(width, 2 * width, 2), Block(2 * width, 4 * width, 2))
        self.proj = nn.Linear(4 * width, emb)
        self.emb = emb

    def forward(self, x):
        h = self.stages(self.stem(x))
        h = h.mean(-1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, in_ch, n_classes, emb=128, p_drop=0.3, evidential=False):
        super().__init__()
        self.enc = Encoder(in_ch, emb)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(emb, n_classes)
        self.evidential = evidential
        self.n_classes = n_classes
        self.log_T = nn.Parameter(torch.zeros(()), requires_grad=False)   # temperature (set after training)

    def embed(self, x):
        return self.enc(x)

    def forward(self, x, mc=False):
        z = self.enc(x)
        z = F.dropout(z, self.drop.p, training=True) if mc else self.drop(z)
        return self.head(z)

    def probs(self, logits, temperature=False):
        if self.evidential:
            alpha = F.softplus(logits) + 1.0
            return alpha / alpha.sum(1, keepdim=True)
        t = float(torch.exp(self.log_T)) if temperature else 1.0
        return F.softmax(logits / t, 1)

    def uncertainty(self, logits):
        """Evidential total uncertainty K / S (Sensoy et al., 2018); softmax entropy otherwise."""
        if self.evidential:
            alpha = F.softplus(logits) + 1.0
            return self.n_classes / alpha.sum(1)
        p = F.softmax(logits, 1)
        return -(p * torch.log(p + 1e-12)).sum(1)

    def loss(self, logits, y, epoch=0, anneal_epochs=10):
        if not self.evidential:
            return F.cross_entropy(logits, y)
        alpha = F.softplus(logits) + 1.0
        S = alpha.sum(1, keepdim=True)
        yk = F.one_hot(y, self.n_classes).float()
        mse = ((yk - alpha / S) ** 2).sum(1) + (alpha * (S - alpha) / (S * S * (S + 1))).sum(1)
        alpha_t = yk + (1 - yk) * alpha
        kl = _kl_dirichlet(alpha_t)
        lam = min(1.0, epoch / float(anneal_epochs))
        return (mse + lam * kl).mean()


def _kl_dirichlet(alpha):
    K = alpha.shape[1]
    S = alpha.sum(1, keepdim=True)
    t1 = torch.lgamma(S.squeeze(1)) - math.lgamma(K) - torch.lgamma(alpha).sum(1)
    t2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(1)
    return t1 + t2


class Regressor(nn.Module):
    """Quantile regression head: lower, median and upper quantiles of the remaining useful life."""

    def __init__(self, in_ch, emb=128, quantiles=(0.05, 0.5, 0.95)):
        super().__init__()
        self.enc = Encoder(in_ch, emb)
        self.head = nn.Linear(emb, len(quantiles))
        self.q = torch.tensor(quantiles)

    def embed(self, x):
        return self.enc(x)

    def forward(self, x, mc=False):
        return self.head(self.enc(x))

    def loss(self, out, y, epoch=0):
        q = self.q.to(out.device)[None, :]
        e = y[:, None] - out
        return torch.maximum(q * e, (q - 1) * e).mean()


def build_model(meta, evidential=False):
    if meta["task"] == "rul":
        return Regressor(meta["in_ch"])
    return Classifier(meta["in_ch"], meta["n_classes"], evidential=evidential)


def fit_temperature(model, logits, y):
    """Temperature scaling on source validation logits (Guo et al., 2017)."""
    logT = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        l = F.cross_entropy(logits / torch.exp(logT), y)
        l.backward()
        return l

    opt.step(closure)
    with torch.no_grad():
        model.log_T.copy_(logT.detach())
    return float(torch.exp(logT))
