"""Published adaptation methods re-implemented on the same encoder and the same recording-level splits, so that the
comparison isolates the adaptation strategy: DANN (Ganin et al., 2016), Deep CORAL (Sun & Saenko, 2016) and TENT
(Wang et al., 2021; = our test-time adaptation without the healthy anchor, applied post hoc in run_grid).
Both DANN and CORAL use the source labelled windows and the target unlabelled windows (x_unl), exactly the data our
test-time adaptation sees."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Classifier


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DANN(Classifier):
    def __init__(self, in_ch, n_classes, emb=128):
        super().__init__(in_ch, n_classes, emb)
        self.disc = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, 1))

    def domain_logit(self, z, lam):
        return self.disc(GradReverse.apply(z, lam)).squeeze(1)


def coral_loss(zs, zt):
    d = zs.shape[1]
    cs = (zs - zs.mean(0)).T @ (zs - zs.mean(0)) / max(len(zs) - 1, 1)
    ct = (zt - zt.mean(0)).T @ (zt - zt.mean(0)) / max(len(zt) - 1, 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def train_uda(method, d, seed, epochs, dev, bs=64, lr=1e-3, coral_w=1.0):
    """method in {'dann', 'coral'}. Model selection on source validation accuracy (no target labels)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    meta = d["meta"]
    model = (DANN(meta["in_ch"], meta["n_classes"]) if method == "dann" else Classifier(meta["in_ch"], meta["n_classes"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    x, y, xu = d["x_tr"], d["y_tr"], d["x_unl"]
    g = np.random.default_rng(seed)
    best, best_state, curve = -np.inf, None, []
    n_steps = int(np.ceil(len(y) / bs))
    for ep in range(epochs):
        model.train()
        perm = g.permutation(len(y))
        tot = 0.0
        for i in range(n_steps):
            idx = perm[i * bs:(i + 1) * bs]
            tidx = g.integers(0, len(xu), len(idx))
            xb = torch.as_tensor(x[idx]).to(dev)
            yb = torch.as_tensor(y[idx]).to(dev)
            xt = torch.as_tensor(xu[tidx]).to(dev)
            zs, zt = model.embed(xb), model.embed(xt)
            loss = F.cross_entropy(model.head(model.drop(zs)), yb)
            if method == "dann":
                p = (ep * n_steps + i) / float(epochs * n_steps)
                lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                dl = model.domain_logit(torch.cat([zs, zt]), lam)
                dy = torch.cat([torch.zeros(len(zs)), torch.ones(len(zt))]).to(dev)
                loss = loss + F.binary_cross_entropy_with_logits(dl, dy)
            else:
                loss = loss + coral_w * coral_loss(zs, zt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            xv = torch.as_tensor(d["x_va"]).to(dev)
            acc = float((model(xv).argmax(1).cpu().numpy() == d["y_va"]).mean())
        curve.append({"epoch": ep, "train_loss": tot / len(y), "val_metric": acc})
        if acc >= best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, curve
