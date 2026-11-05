"""Grid runner (resumable). For each scenario and seed: train the source model (standard head; evidential head as a
second training), then evaluate every variant post hoc from the same checkpoint:
  full        : TTA + commissioning false-alarm calibration + weighted prediction sets + abstention (inspect)
  no_tta      : as full without test-time adaptation
  no_weight   : as full with unweighted (source-calibrated) prediction sets
  src_calib   : false-alarm threshold calibrated on the source healthy calibration windows (no commissioning data)
  no_abstain  : as full with the prediction set replaced by the arg-max class (no inspect decision)
  no_crc      : plain softmax rule: alarm if arg-max is a fault class (no conformal step at all)
  ts / mcd / evid : alarm thresholds set on source validation data to a nominal false-alarm level, from temperature
                scaled softmax, Monte Carlo dropout entropy and evidential uncertainty (same encoder family)
Alphas, labelled-target sizes k and cost ratios are swept post hoc. One JSON and one NPZ per (scenario, seed).

Usage (server): ASSURE_DATA=/data/FedMamba_ChebKAN python run_grid.py --all [--seeds 5] [--epochs 40]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import baselines as B
import conformal as cp
import data as D
import tta as T
from model import build_model, fit_temperature

OUT = os.environ.get("ASSURE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
ALPHAS = [0.01, 0.02, 0.05, 0.10]
KS = [0, 5, 10, 20, 50]
COSTS = {"c_miss": 100.0, "c_fa": 5.0, "c_insp": 1.0}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ training
def train(d, seed, evidential, epochs, bs=64, lr=1e-3):
    set_seed(seed)
    meta = d["meta"]
    model = build_model(meta, evidential).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    x, y = d["x_tr"], d["y_tr"]
    xv, yv = d["x_va"], d["y_va"]
    ydt = torch.float32 if meta["task"] == "rul" else torch.long
    curve = []
    best, best_state = -np.inf, None
    g = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = g.permutation(len(y))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = torch.as_tensor(x[idx]).to(DEV)
            yb = torch.as_tensor(y[idx], dtype=ydt).to(DEV)
            out = model(xb)
            loss = model.loss(out, yb, ep)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        # validation
        logits, _ = T.logits_all(model, xv, DEV)
        if meta["task"] == "rul":
            med = logits[:, 1]
            score = -float(np.sqrt(np.mean((med - yv) ** 2)))
            vloss = -score
        else:
            p = model.probs(torch.as_tensor(logits)).numpy()
            score = float(np.mean(p.argmax(1) == yv))
            vloss = float(torch.nn.functional.cross_entropy(torch.as_tensor(logits), torch.as_tensor(yv)))
        curve.append({"epoch": ep, "train_loss": tot / len(y), "val_metric": score, "val_loss": vloss})
        if score >= best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    if meta["task"] == "cls" and not evidential:
        logits, _ = T.logits_all(model, xv, DEV)
        fit_temperature(model, torch.as_tensor(logits), torch.as_tensor(yv))
    return model, curve


# ------------------------------------------------------------------ evaluation helpers
def probs_of(model, logits, temperature=False):
    return model.probs(torch.as_tensor(logits), temperature=temperature).numpy()


def eval_cls(model, model_evid, d, seed, extra=None):
    """All classification variants for one trained pair of models; `extra` holds the published baselines (DANN, CORAL)
    trained on the same encoder, evaluated under the softmax rule and under our conformal layer."""
    meta = d["meta"]
    K = meta["n_classes"]
    res = {"variants": {}, "sweeps": {}}
    rng = np.random.default_rng(1000 + seed)
    # The run-in windows are split: the first half anchors the adaptation, the second half calibrates the alarm
    # threshold. Windows used by the anchor are pulled towards the healthy prototype, so their scores would be
    # optimistically low and would break the exchangeability on which the false-alarm bound rests.
    n_comm = len(d["x_comm"])
    perm = np.random.default_rng(3000 + seed).permutation(n_comm)
    comm_anchor = d["x_comm"][perm[: n_comm // 2]]
    d = dict(d)
    d["x_comm"] = d["x_comm"][perm[n_comm // 2:]]
    src_healthy = d["x_tr"][d["y_tr"] == D.HEALTHY]
    proto = T.healthy_prototype(model, src_healthy, DEV)
    adapted = T.adapt(model, d["x_unl"], comm_anchor, proto, DEV, seed=seed)
    bn_only = T.adapt(model, d["x_unl"], comm_anchor, proto, DEV, steps=0, seed=seed)
    tent = T.adapt(model, d["x_unl"], comm_anchor, proto, DEV, seed=seed, anchor_weight=0.0)
    models = {"src": model, "tta": adapted, "bn": bn_only, "tent": tent}
    models.update(extra or {})
    emb, logit, prob = {}, {}, {}
    for name, m in models.items():
        emb[name] = {s: T.embed_all(m, d[f"x_{s}"], DEV) for s in ("cal", "comm", "unl", "lab", "te")}
        emb[name]["tr_h"] = T.embed_all(m, src_healthy, DEV)
        logit[name] = {s: T.logits_all(m, d[f"x_{s}"], DEV)[0] for s in ("cal", "lab", "te", "va", "comm")}
        prob[name] = {s: probs_of(m, logit[name][s]) for s in logit[name]}
    y_te, y_lab, y_cal = d["y_te"], d["y_lab"], d["y_cal"]
    healthy_te = y_te == D.HEALTHY

    def novelty(name):
        nv = T.Novelty(emb[name]["tr_h"])
        return {s: nv(emb[name][s]) for s in emb[name]}

    nov = {name: novelty(name) for name in models}

    # domain-classifier weights (target unlabelled vs source calibration), on adapted embeddings
    def weights(name):
        try:
            from sklearn.linear_model import LogisticRegression
            zs, zt = emb[name]["cal"], emb[name]["unl"]
            clf = LogisticRegression(C=1.0, max_iter=500).fit(np.vstack([zs, zt]), np.r_[np.zeros(len(zs)), np.ones(len(zt))])
            w_cal = cp.likelihood_ratio_weights(clf.predict_proba(zs)[:, 1], len(zs), len(zt))
            w_te = cp.likelihood_ratio_weights(clf.predict_proba(emb[name]["te"])[:, 1], len(zs), len(zt))
            return w_cal, w_te
        except Exception:
            return np.ones(len(emb[name]["cal"])), np.ones(len(emb[name]["te"]))

    w_cal, w_te = weights("tta")

    def hybrid_thresholds(name, alpha, k):
        """Class-conditional thresholds. Healthy class: run-in calibration windows of the target machine (labelled
        healthy by construction, exchangeable with later healthy windows). Fault classes: source calibration windows
        of that class, replaced by k labelled target windows of that class when k is large enough for level alpha."""
        qc = np.empty(K)
        p_comm = prob[name]["comm"]
        h_scores = 1.0 - p_comm[:, D.HEALTHY]
        if k > 0:
            ih = rng.permutation(np.where(y_lab == D.HEALTHY)[0])[:k]
            h_scores = np.r_[h_scores, 1.0 - prob[name]["lab"][ih, D.HEALTHY]]
        qc[D.HEALTHY] = cp.conformal_quantile(h_scores, alpha)
        need = int(np.ceil(1.0 / alpha)) - 1
        for c in range(K):
            if c == D.HEALTHY:
                continue
            if k >= need:
                ic = rng.permutation(np.where(y_lab == c)[0])[:k]
                qc[c] = cp.conformal_quantile(1.0 - prob[name]["lab"][ic, c], alpha)
            else:
                qc[c] = cp.conformal_quantile(1.0 - prob[name]["cal"][y_cal == c, c], alpha)
        return qc

    def run_variant(name, alpha, k, sets_mode="hybrid", calib="comm", abstain=True, crc=True):
        """One decision rule evaluated on the target test set."""
        p_te = prob[name]["te"]
        pred = p_te.argmax(1)
        s_te = nov[name]["te"]
        t_hat = None
        if not crc:
            q = -np.inf          # alarm whenever the arg-max is a fault class
            sets = np.zeros_like(p_te, dtype=bool)
            sets[np.arange(len(pred)), pred] = True
            dec = np.where(pred == D.HEALTHY, "continue", "alarm").astype(object)
        else:
            s_cal = nov[name]["comm"] if calib == "comm" else nov[name]["cal"][y_cal == D.HEALTHY]
            q = cp.far_threshold(s_cal, alpha)
            if sets_mode == "hybrid":
                sets = cp.mondrian_sets(p_te, hybrid_thresholds(name, alpha, k))
            elif sets_mode == "weighted":                # marginal weighted split conformal from source calibration
                qw = cp.weighted_quantile(cp.lac_scores(prob[name]["cal"], y_cal), w_cal, alpha, w_te)
                sets = cp.prediction_sets(p_te, qw)
            else:                                        # marginal unweighted source calibration
                qs = cp.conformal_quantile(cp.lac_scores(prob[name]["cal"], y_cal), alpha)
                sets = cp.prediction_sets(p_te, qs)
            if not abstain:
                sets = np.zeros_like(p_te, dtype=bool)
                sets[np.arange(len(pred)), pred] = True
            dec = cp.decide(s_te, q, sets)
            iff = np.concatenate([rng.permutation(np.where(y_lab == c)[0])[:k] for c in range(K) if c != D.HEALTHY]) if k > 0 else np.array([], int)
            if len(iff) >= int(np.ceil(1.0 / alpha)) - 1:   # conformal risk control of the missed-fault rate,
                t_hat = cp.miss_rate_threshold(1.0 - prob[name]["lab"][iff, D.HEALTHY], alpha)   # only when certifiable
                pf_te = 1.0 - p_te[:, D.HEALTHY]
                dec[(dec == "continue") & (pf_te >= t_hat)] = "inspect"
        cost = cp.policy_cost(dec, y_te, pred, **COSTS)
        out = dict(cost)
        out.update({"accuracy": float(np.mean(pred == y_te)), "coverage": cp.coverage(sets, y_te), "set_size": cp.mean_set_size(sets),
                    "coverage_healthy": cp.coverage(sets[healthy_te], y_te[healthy_te]) if healthy_te.any() else None,
                    "coverage_faulty": cp.coverage(sets[~healthy_te], y_te[~healthy_te]) if (~healthy_te).any() else None,
                    "t_hat": t_hat,
                    "realised_far_alarm_score": cp.realised_far(s_te[healthy_te], q) if np.isfinite(q) else np.nan, "q": float(q) if np.isfinite(q) else None})
        return out, dec, sets, pred

    variants = {"full": dict(name="tta"), "no_tta": dict(name="src"), "bn_only": dict(name="bn"),
                "weighted_sets": dict(name="tta", sets_mode="weighted"), "no_weight": dict(name="tta", sets_mode="source"),
                "src_calib": dict(name="tta", calib="cal"),
                "no_abstain": dict(name="tta", abstain=False), "no_crc": dict(name="tta", crc=False),
                "tent_softmax": dict(name="tent", crc=False), "tent_conformal": dict(name="tent")}
    for b in (extra or {}):
        variants[f"{b}_softmax"] = dict(name=b, crc=False)
        variants[f"{b}_conformal"] = dict(name=b)
    for v, kw in variants.items():
        res["variants"][v] = {}
        for alpha in ALPHAS:
            for k in (KS if v == "full" else [0]):
                r, dec, sets, pred = run_variant(alpha=alpha, k=k, **kw)
                res["variants"][v][f"a{alpha}_k{k}"] = r
                if v == "full" and alpha == 0.05 and k == 0:
                    res["decisions_full"] = dec.tolist()
                    res["pred_full"] = pred.tolist()
                    res["sets_full"] = sets.astype(int).tolist()
                if v == "no_tta" and alpha == 0.05:
                    res["pred_src"] = pred.tolist()

    # uncertainty baselines in the same encoder family: thresholds set on SOURCE validation healthy windows
    yv = d["y_va"]
    base = {}
    lg_va_src = logit["src"]["va"]
    p_ts_va = probs_of(model, lg_va_src, temperature=True)
    p_ts_te = probs_of(model, logit["src"]["te"], temperature=True)
    # each baseline: (fault score on val, fault score on test, uncertainty on val or None, uncertainty on test, pred)
    # decision as a practitioner would use it: inspect if the uncertainty exceeds its (1 - alpha) source-validation
    # quantile, otherwise alarm if the fault score exceeds its (1 - alpha) quantile on healthy source-validation windows
    base["ts"] = (1 - p_ts_va[:, D.HEALTHY], 1 - p_ts_te[:, D.HEALTHY], None, None, p_ts_te.argmax(1))
    _, (pm_va, ent_va) = T.logits_all(model, d["x_va"], DEV, mc=20)
    _, (pm_te, ent_te) = T.logits_all(model, d["x_te"], DEV, mc=20)
    base["mcd"] = (1 - pm_va[:, D.HEALTHY], 1 - pm_te[:, D.HEALTHY], ent_va, ent_te, pm_te.argmax(1))
    if model_evid is not None:
        lg_va_e, _ = T.logits_all(model_evid, d["x_va"], DEV)
        lg_te_e, _ = T.logits_all(model_evid, d["x_te"], DEV)
        pe_va, pe_te = probs_of(model_evid, lg_va_e), probs_of(model_evid, lg_te_e)
        u_va = model_evid.uncertainty(torch.as_tensor(lg_va_e)).numpy()
        u_te = model_evid.uncertainty(torch.as_tensor(lg_te_e)).numpy()
        base["evid"] = (1 - pe_va[:, D.HEALTHY], 1 - pe_te[:, D.HEALTHY], u_va, u_te, pe_te.argmax(1))
    for v, (f_va, f_te, u_va_, u_te_, pred) in base.items():
        res["variants"][v] = {}
        for alpha in ALPHAS:
            thr = np.quantile(f_va[yv == D.HEALTHY], 1 - alpha)      # nominal level on source validation
            dec = np.where(f_te > thr, "alarm", "continue").astype(object)
            if u_va_ is not None:
                dec[u_te_ > np.quantile(u_va_, 1 - alpha)] = "inspect"
            r = cp.policy_cost(dec, y_te, pred, **COSTS)
            r.update({"accuracy": float(np.mean(pred == y_te)), "realised_far_alarm_score": float(np.mean(f_te[healthy_te] > thr))})
            res["variants"][v][f"a{alpha}_k0"] = r

    # sweeps and curves for the figures
    res["sweeps"]["auroc"] = {name: _auroc(nov[name]["te"], ~healthy_te) for name in models}
    for name in models:
        res["sweeps"][f"risk_coverage_{name}"] = [a.tolist() for a in cp.risk_coverage_curve(prob[name]["te"].argmax(1) == y_te, prob[name]["te"].max(1))]
    # commissioning length sweep (full variant, alpha 0.05): realised FAR vs number of commissioning windows
    comm = nov["tta"]["comm"]
    res["sweeps"]["commissioning"] = {}
    for n in (20, 30, 45, 60, len(comm)):
        q = cp.far_threshold(comm[:n], 0.05)
        res["sweeps"]["commissioning"][str(n)] = cp.realised_far(nov["tta"]["te"][healthy_te], q) if np.isfinite(q) else None
    # open-set unknown class (CWRU ball on the source side), if present
    if "x_unknown" in d:
        s_unk = T.Novelty(emb["src"]["tr_h"])(T.embed_all(model, d["x_unknown"], DEV))
        q = cp.far_threshold(nov["src"]["cal"][y_cal == D.HEALTHY], 0.05)
        res["sweeps"]["unknown_alarm_rate"] = float(np.mean(s_unk > q))
    arrays = {"z_src_te": emb["src"]["te"], "z_tta_te": emb["tta"]["te"], "z_tta_comm": emb["tta"]["comm"], "y_te": y_te,
              "nov_src_te": nov["src"]["te"], "nov_tta_te": nov["tta"]["te"], "p_src_te": prob["src"]["te"], "p_tta_te": prob["tta"]["te"],
              "w_te": w_te, "nov_tta_comm": comm}
    return res, arrays, adapted


def _auroc(score, positive):
    s = np.asarray(score)
    pos = np.asarray(positive, bool)
    if pos.all() or (~pos).all():
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_engine_sample(eng, r, rng):
    """r random windows per engine. The test protocol scores one truncated window per engine, so calibration on
    every window of every engine (dominated by capped early-life windows) is not exchangeable with the test points;
    sampling a few random truncation points per engine mimics the test protocol."""
    idx = []
    for u in np.unique(eng):
        w = np.where(eng == u)[0]
        idx.extend(rng.choice(w, min(r, len(w)), replace=False).tolist())
    return np.array(sorted(idx))


def eval_rul(model, d, seed, r=5):
    """RUL intervals on FD003 from a model trained on FD001: source-calibrated and target-calibrated."""
    res = {"variants": {}}
    rng = np.random.default_rng(2000 + seed)
    adapted = T.adapt(model, d["x_unl"], d["x_comm"], None, DEV, steps=0, seed=seed, task="rul")
    sel = {"cal": per_engine_sample(d["eng_cal"], r, rng), "lab": per_engine_sample(d["eng_lab"], r, rng)}
    out = {}
    for name, m in (("src", model), ("bn", adapted)):
        o = {s: T.logits_all(m, d[f"x_{s}"][sel[s]] if s in sel else d[f"x_{s}"], DEV)[0] for s in ("cal", "lab", "te")}
        out[name] = o
        for calib in ("source", "target"):
            xs = "cal" if calib == "source" else "lab"
            ys = d["y_cal"][sel["cal"]] if calib == "source" else d["y_lab"][sel["lab"]]
            for alpha in ALPHAS:
                dlo, dhi = cp.cqr_asymmetric(o[xs][:, 0], o[xs][:, 2], ys, alpha, 0.10)
                res.setdefault("offsets", {})[f"{name}_{calib}_a{alpha}"] = [float(dlo), float(dhi)]
                Lb, Ub = o["te"][:, 0] - dlo, o["te"][:, 2] + dhi
                r = cp.interval_metrics(Lb, Ub, d["y_te"])
                r["rmse_median"] = float(np.sqrt(np.mean((o["te"][:, 1] - d["y_te"]) ** 2)))
                res["variants"][f"{name}_{calib}"] = res["variants"].get(f"{name}_{calib}", {})
                res["variants"][f"{name}_{calib}"][f"a{alpha}_k0"] = r
    arrays = {"te_pred_src": out["src"]["te"], "te_pred_bn": out["bn"]["te"], "y_te": d["y_te"]}
    return res, arrays, adapted


# ------------------------------------------------------------------ XAI: integrated gradients -> frequency bands
def attribution_bands(model, x, fs, n_bands=16, steps=32):
    """Integrated gradients of the arg-max logit with respect to the raw window, then the share of attribution
    energy per frequency band (used for the XAI figure on real signals). Only for 1-channel vibration windows."""
    model.eval()
    xb = torch.as_tensor(x).to(DEV)
    base = torch.zeros_like(xb)
    total = torch.zeros_like(xb)
    for a in np.linspace(0, 1, steps):
        xi = (base + a * (xb - base)).requires_grad_(True)
        logits = model(xi)
        sel = logits.max(1)[0].sum()
        g, = torch.autograd.grad(sel, xi)
        total += g
    ig = ((xb - base) * total / steps).detach().cpu().numpy()[:, 0, :]
    spec = np.abs(np.fft.rfft(ig, axis=1)) ** 2
    edges = np.linspace(0, spec.shape[1], n_bands + 1).astype(int)
    bands = np.stack([spec[:, edges[i]:edges[i + 1]].sum(1) for i in range(n_bands)], 1)
    return ig, bands / (bands.sum(1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ main
def _ckpt(name, seed, tag):
    return os.path.join(OUT, "models", f"{name}_s{seed}_{tag}.pt")


def _load_or_train(name, seed, tag, fit, build, eval_only):
    """Train (and save) a model, or load it in eval-only mode. Curves are kept in a side JSON."""
    path, cpath = _ckpt(name, seed, tag), _ckpt(name, seed, tag) + ".curve.json"
    if eval_only and os.path.exists(path):
        m = build().to(DEV)
        m.load_state_dict(torch.load(path, map_location=DEV))
        m.eval()
        curve = json.load(open(cpath)) if os.path.exists(cpath) else None
        return m, curve
    m, curve = fit()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(m.state_dict(), path)
    json.dump(curve, open(cpath, "w"))
    return m, curve


def run_one(name, seed, epochs, force=False, eval_only=False):
    from model import build_model
    os.makedirs(OUT, exist_ok=True)
    jpath = os.path.join(OUT, f"{name}_s{seed}.json")
    if os.path.exists(jpath) and not (force or eval_only):
        print("skip", jpath)
        return
    # lock so that several workers can share one grid; a lock whose owner process is gone is stale (crashed worker)
    lock = jpath + ".lock"
    if os.path.exists(lock) and not force:
        try:
            import psutil
            owner = int(open(lock).read().strip() or 0)
            alive = psutil.pid_exists(owner) and owner != os.getpid()
        except Exception:
            alive = time.time() - os.path.getmtime(lock) < 4 * 3600
        if alive:
            print("busy", jpath)
            return
    open(lock, "w").write(str(os.getpid()))
    try:
        _run_one(name, seed, epochs, eval_only, jpath)
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def _run_one(name, seed, epochs, eval_only, jpath):
    from model import build_model
    t0 = time.time()
    d = D.load(name)
    meta = d["meta"]
    model, curve = _load_or_train(name, seed, "std", lambda: train(d, seed, False, epochs), lambda: build_model(meta, False), eval_only)
    model_evid, curve_e = (None, None)
    if meta["task"] == "cls":
        model_evid, curve_e = _load_or_train(name, seed, "evid", lambda: train(d, seed + 500, True, epochs), lambda: build_model(meta, True), eval_only)
        extra, curves_b = {}, {}
        for b in ("dann", "coral"):
            builder = (lambda: B.DANN(meta["in_ch"], meta["n_classes"])) if b == "dann" else (lambda: build_model(meta, False))
            extra[b], curves_b[b] = _load_or_train(name, seed, b, lambda b=b: B.train_uda(b, d, seed + 700, epochs, DEV), builder, eval_only)
        res, arrays, adapted = eval_cls(model, model_evid, d, seed, extra)
        res["curve_baselines"] = curves_b
        if meta["in_ch"] == 1:
            idx = np.concatenate([np.where(d["y_te"] == c)[0][:2] for c in range(meta["n_classes"])])
            ig, bands = attribution_bands(adapted, d["x_te"][idx], meta.get("fs", 12000))
            arrays.update({"xai_x": d["x_te"][idx][:, 0, :], "xai_ig": ig, "xai_bands": bands, "xai_y": d["y_te"][idx]})
    else:
        res, arrays, adapted = eval_rul(model, d, seed)
    res.update({"scenario": name, "seed": seed, "meta": meta, "curve": curve, "curve_evid": curve_e, "sizes": D.summary(d),
                "minutes": (time.time() - t0) / 60.0, "epochs": epochs, "alphas": ALPHAS, "ks": KS, "costs": COSTS})
    with open(jpath, "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(os.path.join(OUT, f"{name}_s{seed}.npz"), **arrays)
    print(f"done {name} seed {seed} in {res['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenario", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="reload saved models and redo only the evaluation")
    a = ap.parse_args()
    names = D.SCENARIOS if a.all else a.scenario
    for n in names:
        for s in range(a.seeds):
            run_one(n, s, a.epochs, a.force, a.eval_only)
