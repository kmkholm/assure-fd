"""Explanation figure for ASSURE-FD (run after the grid; light GPU work).

For one cross-load scenario and seed 0, the saved source model is adapted exactly as in run_grid.eval_cls (same run-in
split and seed), and integrated gradients (32 steps, zero baseline) of the predicted logit are computed for 100 test
windows per class. Two views are drawn:
  (a) one real window per class, with |IG| smoothed over 32 samples shown as a heat strip under the waveform;
  (b) the share of attribution energy per 375-Hz band (mean and standard deviation over the 100 windows), against the
      share of signal energy per band. Where the two profiles differ, the model attends to bands for reasons other
      than their energy, which is the behaviour expected from a fault-sensitive detector.
Writes paper/figures/fig_xai.pdf|png and results/xai_<scenario>.npz.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASSURE_FULL", "1")
import data as D  # noqa: E402
import tta as T  # noqa: E402
from model import build_model  # noqa: E402

RES = os.environ.get("ASSURE_OUT", os.path.join(HERE, "..", "results"))
FIG = os.path.join(HERE, "..", "paper", "figures")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCEN = sys.argv[1] if len(sys.argv) > 1 else "cwru_L0_L2"
SEED, N_PER, STEPS, N_BANDS = 0, 100, 32, 16
COLS = {"Healthy": "#1b4965", "Inner race": "#c1121f", "Outer race": "#e09f3e", "Ball": "#6a994e"}


def adapted_model():
    d = D.load(SCEN)
    meta = d["meta"]
    m = build_model(meta, False).to(DEV)
    m.load_state_dict(torch.load(os.path.join(RES, "models", f"{SCEN}_s{SEED}_std.pt"), map_location=DEV))
    m.eval()
    n = len(d["x_comm"])
    perm = np.random.default_rng(3000 + SEED).permutation(n)          # identical to run_grid.eval_cls
    comm_anchor = d["x_comm"][perm[: n // 2]]
    proto = T.healthy_prototype(m, d["x_tr"][d["y_tr"] == D.HEALTHY], DEV)
    return d, T.adapt(m, d["x_unl"], comm_anchor, proto, DEV, seed=SEED)


def integrated_gradients(m, x, bs=50):
    out = []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(x[i:i + bs]).to(DEV)
        with torch.no_grad():
            target = m(xb).argmax(1)
        total = torch.zeros_like(xb)
        for a in np.linspace(1.0 / STEPS, 1.0, STEPS):
            xi = (a * xb).requires_grad_(True)
            sel = m(xi).gather(1, target[:, None]).sum()
            g, = torch.autograd.grad(sel, xi)
            total += g
        out.append((xb * total / STEPS).detach().cpu().numpy()[:, 0, :])
    return np.concatenate(out)


def band_share(sig):
    spec = np.abs(np.fft.rfft(sig, axis=1)) ** 2
    edges = np.linspace(0, spec.shape[1], N_BANDS + 1).astype(int)
    b = np.stack([spec[:, edges[i]:edges[i + 1]].sum(1) for i in range(N_BANDS)], 1)
    return b / (b.sum(1, keepdims=True) + 1e-12)


def main():
    d, m = adapted_model()
    meta = d["meta"]
    classes = meta["classes"]
    rng = np.random.default_rng(0)
    idx = {c: rng.permutation(np.where(d["y_te"] == c)[0])[:N_PER] for c in range(meta["n_classes"])}
    ig = {c: integrated_gradients(m, d["x_te"][idx[c]]) for c in idx}
    sig = {c: d["x_te"][idx[c]][:, 0, :] for c in idx}
    np.savez_compressed(os.path.join(RES, f"xai_{SCEN}.npz"), **{f"ig_{c}": ig[c] for c in ig}, **{f"x_{c}": sig[c] for c in sig})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    K = len(classes)
    fig = plt.figure(figsize=(7.2, 1.55 * K + 0.4))
    gs = GridSpec(K, 2, width_ratios=[2.3, 1.0], wspace=0.28, hspace=0.55, figure=fig)
    fs = meta.get("fs", 12000)
    t = np.arange(sig[0].shape[1]) / fs * 1000.0
    f_edges = np.linspace(0, fs / 2, N_BANDS + 1) / 1000.0
    for c in range(K):
        name, col = classes[c], COLS.get(classes[c], "#333333")
        # (a) waveform with smoothed |IG| as a heat strip
        j = int(np.argmax(np.abs(ig[c]).sum(1)))          # the window with the clearest attribution of this class
        ax = fig.add_subplot(gs[c, 0])
        x = sig[c][j]
        a = np.convolve(np.abs(ig[c][j]), np.ones(32) / 32, mode="same")
        a = a / (a.max() + 1e-12)
        lo, hi = x.min(), x.max()
        pad = 0.25 * (hi - lo)
        ax.imshow(a[None, :], aspect="auto", cmap="inferno_r", extent=(t[0], t[-1], lo - pad, lo - 0.05 * pad), vmin=0, vmax=1)
        ax.plot(t, x, color=col, lw=0.6)
        ax.set_ylim(lo - pad, hi + 0.05 * pad)
        ax.set_yticks([])
        ax.set_ylabel(name, color=col, fontsize=9)
        if c == 0:
            ax.set_title("(a) target window and integrated-gradient magnitude", fontsize=9, loc="left")
        if c < K - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (ms)")
        # (b) attribution energy per band vs signal energy per band
        bx = fig.add_subplot(gs[c, 1])
        sa, ss = band_share(ig[c]), band_share(sig[c])
        mid = 0.5 * (f_edges[:-1] + f_edges[1:])
        bx.bar(mid, sa.mean(0), width=np.diff(f_edges) * 0.9, color=col, alpha=0.85, label="attribution")
        bx.errorbar(mid, sa.mean(0), yerr=sa.std(0), fmt="none", ecolor="#333333", lw=0.6, capsize=1.5)
        bx.plot(mid, ss.mean(0), "k--", lw=1.0, label="signal energy")
        bx.set_ylim(0, max(sa.mean(0).max() + sa.std(0).max(), ss.mean(0).max()) * 1.15)
        bx.set_yticks([])
        if c == 0:
            bx.set_title("(b) share per 375-Hz band", fontsize=9, loc="left")
            bx.legend(fontsize=6.5, frameon=False, loc="upper right")
        if c < K - 1:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel("frequency (kHz)")
    sm = plt.cm.ScalarMappable(cmap="inferno_r", norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.13, 0.035, 0.32, 0.012])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("normalised |integrated gradient| (darker = larger)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, "fig_xai.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_xai.png"), dpi=300, bbox_inches="tight")
    # numbers for the text: Jensen-Shannon distance between attribution and energy profiles per class
    from scipy.spatial.distance import jensenshannon
    for c in range(K):
        print(classes[c], "JS(attribution, energy) =", round(float(jensenshannon(band_share(ig[c]).mean(0), band_share(sig[c]).mean(0))), 3),
              "top attribution band (kHz):", round(float(f_edges[np.argmax(band_share(ig[c]).mean(0))]), 2))


if __name__ == "__main__":
    main()
