"""Source-free test-time adaptation with a healthy anchor.

Two steps on the target machine, using only unlabelled windows and the healthy commissioning windows:
1. BatchNorm statistics are re-estimated on the unlabelled target windows (norm adaptation).
2. The BatchNorm affine parameters are updated by entropy minimisation on the unlabelled windows (TENT, Wang et al.,
   2021), with an anchor term that keeps the embeddings of the healthy commissioning windows close to the source
   healthy prototype. The anchor prevents the collapse of the healthy class that plain entropy minimisation can cause
   when the target stream is dominated by one fault type.
A copy of the model is adapted, so the source model is kept for the ablations.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F


def _bn_params(model):
    ps = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            ps += [m.weight, m.bias]
    return ps


@torch.no_grad()
def healthy_prototype(model, x_src_healthy, device, bs=256):
    model.eval()
    zs = [model.embed(torch.as_tensor(x_src_healthy[i:i + bs]).to(device)) for i in range(0, len(x_src_healthy), bs)]
    return torch.cat(zs).mean(0)


@torch.no_grad()
def recompute_bn_stats(model, x_unl, device, bs=128):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None            # cumulative average over the passes below
    model.train()
    for i in range(0, len(x_unl), bs):
        model.embed(torch.as_tensor(x_unl[i:i + bs]).to(device))
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = 0.1
    model.eval()


def adapt(model, x_unl, x_comm, proto, device, steps=50, bs=64, lr=1e-3, anchor_weight=1.0, seed=0, task="cls"):
    """Returns an adapted copy of the model. For the regression task only step 1 (BN statistics) is applied."""
    g = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(model)
    recompute_bn_stats(m, x_unl, device)
    if task != "cls" or steps <= 0:
        return m
    params = _bn_params(m)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    m.eval()                                  # BN uses the re-estimated statistics; affine terms are trained
    n = len(x_unl)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g).numpy()
        xb = torch.as_tensor(x_unl[idx]).to(device)
        logits = m(xb)
        p = m.probs(logits)
        ent = -(p * torch.log(p + 1e-12)).sum(1).mean()
        cidx = torch.randint(0, len(x_comm), (min(bs, len(x_comm)),), generator=g).numpy()
        zc = m.embed(torch.as_tensor(x_comm[cidx]).to(device))
        anchor = F.mse_loss(zc, proto[None, :].expand_as(zc))
        loss = ent + anchor_weight * anchor
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


@torch.no_grad()
def embed_all(model, x, device, bs=256):
    model.eval()
    return torch.cat([model.embed(torch.as_tensor(x[i:i + bs]).to(device)).cpu() for i in range(0, len(x), bs)]).numpy()


@torch.no_grad()
def logits_all(model, x, device, bs=256, mc=0):
    """Logits (and, with mc > 0, the mean probability and predictive entropy over mc dropout samples)."""
    model.eval()
    outs = [model(torch.as_tensor(x[i:i + bs]).to(device)).cpu() for i in range(0, len(x), bs)]
    logits = torch.cat(outs)
    if mc <= 0:
        return logits.numpy(), None
    ps = []
    for _ in range(mc):
        ps.append(torch.cat([model.probs(model(torch.as_tensor(x[i:i + bs]).to(device), mc=True)).cpu() for i in range(0, len(x), bs)]))
    pm = torch.stack(ps).mean(0)
    ent = -(pm * torch.log(pm + 1e-12)).sum(1)
    return logits.numpy(), (pm.numpy(), ent.numpy())


class Novelty:
    """Mahalanobis distance to the source healthy embedding cloud (shrunk covariance). Large = far from healthy."""

    def __init__(self, z_healthy, shrink=0.1):
        self.mu = z_healthy.mean(0)
        c = np.cov(z_healthy.T) + 1e-6 * np.eye(z_healthy.shape[1])
        c = (1 - shrink) * c + shrink * np.trace(c) / c.shape[0] * np.eye(c.shape[0])
        self.inv = np.linalg.inv(c)

    def __call__(self, z):
        d = z - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv, d))
