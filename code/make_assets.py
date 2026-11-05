"""Tables and figures from results/*.json and results/*.npz. Run after the grid (CPU only, matplotlib).
Writes paper/tables/*.tex (numbers as LaTeX macros too) and paper/figures/*.pdf|png.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAP = os.path.join(HERE, "..", "paper")
FIG = os.path.join(PAP, "figures")
TAB = os.path.join(PAP, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# distinctive palette (not the matplotlib default)
C = {"full": "#1b4965", "no_tta": "#c1121f", "bn_only": "#5fa8d3", "no_weight": "#e09f3e", "src_calib": "#9e2a2b",
     "no_abstain": "#540b0e", "no_crc": "#8d99ae", "weighted_sets": "#b56576", "ts": "#6a994e", "mcd": "#bc4749", "evid": "#386641", "target": "#000000"}
NAMES = {"full": "ASSURE-FD (full)", "no_tta": "w/o test-time adaptation", "bn_only": "BN statistics only", "no_weight": "marginal source sets", "weighted_sets": "shift-weighted marginal sets",
         "src_calib": "source-calibrated alarm", "no_abstain": "w/o abstention", "no_crc": "softmax rule (no conformal)",
         "ts": "temperature scaling", "mcd": "MC dropout", "evid": "evidential",
         "tent_softmax": "TENT, softmax rule", "tent_conformal": "TENT + conformal layer", "dann_softmax": "DANN, softmax rule",
         "dann_conformal": "DANN + conformal layer", "coral_softmax": "Deep CORAL, softmax rule", "coral_conformal": "Deep CORAL + conformal layer"}
for _k in ("tent_softmax", "tent_conformal", "dann_softmax", "dann_conformal", "coral_softmax", "coral_conformal"):
    C.setdefault(_k, "#7f7f7f")
NAMES.update({"continue_all": "continue every window", "inspect_all": "inspect every window"})
C.update({"continue_all": "#cccccc", "inspect_all": "#4d4d4d"})
SCEN = {"cwru_L0_L1": "CWRU 0$\\to$1 hp", "cwru_L0_L2": "CWRU 0$\\to$2 hp", "cwru_L0_L3": "CWRU 0$\\to$3 hp", "cwru_DE_FE": "CWRU DE$\\to$FE",
        "cwru_to_pad": "CWRU$\\to$Paderborn", "pad_to_cwru": "Paderborn$\\to$CWRU", "pad_art_real": "Paderborn art.$\\to$real",
        "mimii_6_0": "MIMII 6$\\to$0 dB", "mimii_6_m6": "MIMII 6$\\to$$-$6 dB", "mimii_id00_id02": "MIMII fan 00/04/06$\\to$02", "cmapss_FD001_FD003": "C-MAPSS FD001$\\to$FD003"}


def load_all():
    R = {}
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        r = json.load(open(f))
        R.setdefault(r["scenario"], []).append(r)
    return R


def agg(rs, variant, key, metric):
    v = [r["variants"][variant][key][metric] for r in rs if variant in r["variants"] and key in r["variants"][variant] and r["variants"][variant][key].get(metric) is not None]
    v = np.array([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


COST0 = {"c_miss": 100.0, "c_fa": 5.0, "c_insp": 1.0}
RATIOS = [(10.0, 5.0), (100.0, 5.0), (1000.0, 5.0), (100.0, 1.0), (100.0, 20.0)]   # (c_miss, c_fa) with c_insp = 1


def _counts(r):
    """Healthy and faulty test windows of a classification run."""
    c = r["sizes"]["y_te_counts"]
    nh, nf = float(c[0]), float(sum(c[1:]))
    return nh, nf, nh + nf


def _parts(r, v, key="a0.05_k0"):
    """Recover the exact decision counts (missed faults, false alarms, inspections incl. wrong-type alarms) from the
    stored rates and the stored cost, so that the cost can be recomputed for any cost ratio."""
    x = r["variants"][v][key]
    nh, nf, n = _counts(r)
    miss, fa = x["miss_rate"] * nf, x["false_alarm_rate"] * nh
    insp = (x["cost_per_1000"] * n / 1000.0 - COST0["c_miss"] * miss - COST0["c_fa"] * fa) / COST0["c_insp"]
    return miss, fa, max(insp, 0.0), n


def cost_at(r, v, c_miss, c_fa, c_insp=1.0, key="a0.05_k0"):
    if v == "full_k20":
        v, key = "full", "a0.05_k20"
    if v == "inspect_all":
        return 1000.0 * c_insp
    if v == "continue_all":
        nh, nf, n = _counts(r)
        return 1000.0 * c_miss * nf / n
    miss, fa, insp, n = _parts(r, v, key)
    return 1000.0 * (c_miss * miss + c_fa * fa + c_insp * insp) / n


def mean_cost(rs, v, c_miss, c_fa, c_insp=1.0):
    return float(np.mean([cost_at(r, v, c_miss, c_fa, c_insp) for r in rs]))


def fmt(m, ci, pct=True, d=1):
    if np.isnan(m):
        return "--"
    s = 100 if pct else 1
    return f"{m * s:.{d}f} $\\pm$ {ci * s:.{d}f}"


def table_main(R, alpha=0.05):
    """Per scenario at a 5% target: ASSURE-FD (FAR with 95% CI, miss, inspection, cost), the softmax rule and DANN
    (FAR, miss, cost), and the two trivial policies (cost only). The lowest cost in each row is set in bold."""
    key = f"a{alpha}_k0"
    rows = []
    for s in SCEN:
        if s not in R or R[s][0]["meta"]["task"] != "cls":
            continue
        rs = R[s]
        m = lambda v, k: agg(rs, v, key, k)[0]
        costs = {v: mean_cost(rs, v, COST0["c_miss"], COST0["c_fa"]) for v in ("full", "full_k20", "no_crc", "dann_softmax", "continue_all", "inspect_all")}
        best = min(costs.values())
        cf = lambda v: (f"\\textbf{{{costs[v]:.0f}}}" if abs(costs[v] - best) < 0.5 else f"{costs[v]:.0f}")
        far = agg(rs, "full", key, "false_alarm_rate")
        row = [SCEN[s], fmt(*far[:2]), f"{100 * m('full', 'miss_rate'):.1f}", f"{100 * m('full', 'inspection_rate'):.1f}", cf("full")]
        for v in ("no_crc", "dann_softmax"):
            row += [f"{100 * m(v, 'false_alarm_rate'):.1f}", f"{100 * m(v, 'miss_rate'):.1f}", cf(v)]
        row += [cf("full_k20"), cf("continue_all"), cf("inspect_all")]
        rows.append(" & ".join(row) + " \\\\")
    hdr = ("\\begin{tabular}{lcccc ccc ccc ccc}\n\\toprule\n"
           " & \\multicolumn{4}{c}{ASSURE-FD, no target labels} & \\multicolumn{3}{c}{Softmax rule} & \\multicolumn{3}{c}{DANN, softmax rule} & ASSURE-FD & Continue & Inspect \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
           "Scenario & FAR (\\%) & Miss (\\%) & Insp. (\\%) & Cost & FAR & Miss & Cost & FAR & Miss & Cost & $k=20$ & all & all \\\\\n\\midrule\n")
    open(os.path.join(TAB, "tab_main.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_cost_sensitivity(R):
    """Mean cost per 1000 decisions over the classification scenarios for several cost ratios (c_insp = 1)."""
    scen = [s for s in SCEN if s in R and R[s][0]["meta"]["task"] == "cls"]
    cols = [("full", "ASSURE-FD"), ("full_k20", "ASSURE-FD, $k=20$"), ("no_crc", "Softmax"), ("dann_softmax", "DANN"), ("tent_softmax", "TENT"), ("mcd", "MC dropout"),
            ("continue_all", "Continue all"), ("inspect_all", "Inspect all")]
    rows = []
    for cm, cfa in RATIOS:
        vals = {v: float(np.mean([mean_cost(R[s], v, cm, cfa) for s in scen])) for v, _ in cols}
        best = min(vals.values())
        cells = [(f"\\textbf{{{vals[v]:.0f}}}" if abs(vals[v] - best) < 0.5 else f"{vals[v]:.0f}") for v, _ in cols]
        rows.append(f"{cm:.0f} : {cfa:.0f} : 1 & " + " & ".join(cells) + " \\\\")
    hdr = ("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n$c_{\\mathrm{miss}} : c_{\\mathrm{fa}} : c_{\\mathrm{insp}}$ & "
           + " & ".join(n for _, n in cols) + " \\\\\n\\midrule\n")
    open(os.path.join(TAB, "tab_cost_sensitivity.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # per-scenario winner at the default ratio, for the text
    win = {}
    for s in scen:
        vals = {v: mean_cost(R[s], v, COST0["c_miss"], COST0["c_fa"]) for v, _ in cols}
        win[s] = min(vals, key=vals.get)
    json.dump(win, open(os.path.join(TAB, "cost_winner_per_scenario.json"), "w"), indent=1)


def table_ablation(R, alpha=0.05):
    key = f"a{alpha}_k0"
    variants = ["full", "no_tta", "bn_only", "weighted_sets", "no_weight", "src_calib", "no_abstain", "no_crc", "ts", "mcd", "evid",
                "tent_softmax", "tent_conformal", "dann_softmax", "dann_conformal", "coral_softmax", "coral_conformal"]
    scen = [s for s in SCEN if s in R and R[s][0]["meta"]["task"] == "cls"]
    rows = []
    for v in variants:
        if not any(v in R[s][0]["variants"] for s in scen):
            continue
        far = np.nanmean([agg(R[s], v, key, "false_alarm_rate")[0] for s in scen])
        miss = np.nanmean([agg(R[s], v, key, "miss_rate")[0] for s in scen])
        insp = np.nanmean([agg(R[s], v, key, "inspection_rate")[0] for s in scen])
        cov = np.nanmean([agg(R[s], v, key, "coverage")[0] for s in scen])
        cost = np.nanmean([agg(R[s], v, key, "cost_per_1000")[0] for s in scen])
        rows.append(f"{NAMES[v]} & {far * 100:.1f} & {miss * 100:.1f} & {insp * 100:.1f} & {cov * 100:.1f} & {cost:.0f} \\\\")
    # trivial policies: continue every window (no alarm, every fault missed) and inspect every window
    cont = np.mean([mean_cost(R[s], "continue_all", COST0["c_miss"], COST0["c_fa"]) for s in scen])
    rows.append("\\midrule")
    rows.append(f"Continue every window & 0.0 & 100.0 & 0.0 & -- & {cont:.0f} \\\\")
    rows.append(f"Inspect every window & 0.0 & 0.0 & 100.0 & -- & {1000 * COST0['c_insp']:.0f} \\\\")
    hdr = "\\begin{tabular}{lccccc}\n\\toprule\nVariant & FAR (\\%) & Miss (\\%) & Inspect (\\%) & Coverage (\\%) & Cost / 1000 \\\\\n\\midrule\n"
    open(os.path.join(TAB, "tab_ablation.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_rul(R):
    s = "cmapss_FD001_FD003"
    if s not in R:
        return
    rows = []
    for v, name in (("src_source", "source model, source calibration"), ("src_target", "source model, target calibration"),
                    ("bn_source", "adapted model, source calibration"), ("bn_target", "adapted model, target calibration")):
        for alpha in (0.02, 0.05, 0.10):
            key = f"a{alpha}_k0"
            cov = agg(R[s], v, key, "coverage"); over = agg(R[s], v, key, "overestimation_rate"); w = agg(R[s], v, key, "mean_width"); rm = agg(R[s], v, key, "rmse_median")
            rows.append(f"{name} & {alpha:.2f} & {fmt(*cov[:2])} & {fmt(*over[:2])} & {w[0]:.1f} & {rm[0]:.1f} \\\\")
    hdr = "\\begin{tabular}{llcccc}\n\\toprule\nSetting & $\\alpha_{\\mathrm{over}}$ & Coverage (\\%) & Overestimation (\\%) & Width (cycles) & RMSE \\\\\n\\midrule\n"
    open(os.path.join(TAB, "tab_rul.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def macros(R):
    """Key numbers as \\newcommand macros for the manuscript."""
    key = "a0.05_k0"
    scen = [s for s in SCEN if s in R and R[s][0]["meta"]["task"] == "cls"]
    m = {}
    m["nScenarios"] = len(R)
    m["nSeeds"] = max(len(v) for v in R.values())
    for v in ("full", "no_tta", "no_crc", "ts", "mcd", "evid"):
        m[f"far{v.replace('_', '')}"] = f"{100 * np.nanmean([agg(R[s], v, key, 'false_alarm_rate')[0] for s in scen]):.1f}"
        m[f"miss{v.replace('_', '')}"] = f"{100 * np.nanmean([agg(R[s], v, key, 'miss_rate')[0] for s in scen]):.1f}"
        m[f"cost{v.replace('_', '')}"] = f"{np.nanmean([agg(R[s], v, key, 'cost_per_1000')[0] for s in scen]):.0f}"
    m["farfullmax"] = f"{100 * np.nanmax([agg(R[s], 'full', key, 'false_alarm_rate')[0] for s in scen]):.1f}"
    m["costKtwenty"] = f"{np.mean([mean_cost(R[s], 'full_k20', COST0['c_miss'], COST0['c_fa']) for s in scen]):.0f}"
    m["costDann"] = f"{np.mean([mean_cost(R[s], 'dann_softmax', COST0['c_miss'], COST0['c_fa']) for s in scen]):.0f}"
    m["covHealthy"] =f"{100 * np.nanmean([agg(R[s], 'full', key, 'coverage_healthy')[0] for s in scen]):.1f}"
    m["covFaultSrc"] = f"{100 * np.nanmean([agg(R[s], 'full', key, 'coverage_faulty')[0] for s in scen]):.1f}"
    m["covFaultKfifty"] = f"{100 * np.nanmean([agg(R[s], 'full', 'a0.05_k50', 'coverage_faulty')[0] for s in scen]):.1f}"
    m["accsrc"] = f"{100 * np.nanmean([agg(R[s], 'no_tta', key, 'accuracy')[0] for s in scen]):.1f}"
    m["acctta"] = f"{100 * np.nanmean([agg(R[s], 'full', key, 'accuracy')[0] for s in scen]):.1f}"
    with open(os.path.join(TAB, "macros.tex"), "w") as f:
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        # per-scenario values, used in the text as \R{scenario}{variant}{metric}
        f.write("\\providecommand{\\R}[3]{\\csname r@#1@#2@#3\\endcsname}\n")
        for s, rs in R.items():
            if rs[0]["meta"]["task"] != "cls":
                continue
            vals = {}
            for v in ("full", "no_tta", "no_crc", "dann_softmax", "coral_softmax", "tent_softmax", "ts", "mcd", "evid",
                      "dann_conformal", "coral_conformal", "tent_conformal", "weighted_sets", "src_calib"):
                for metric, short in (("false_alarm_rate", "far"), ("miss_rate", "miss"), ("inspection_rate", "insp"), ("accuracy", "acc")):
                    vals[(v, short)] = f"{100 * agg(rs, v, 'a0.05_k0', metric)[0]:.1f}"
                vals[(v, "cost")] = f"{mean_cost(rs, v, COST0['c_miss'], COST0['c_fa']):.0f}"
            for k in (10, 20, 50):
                for metric, short in (("false_alarm_rate", "far"), ("miss_rate", "miss"), ("inspection_rate", "insp"), ("coverage_faulty", "covf")):
                    vals[(f"k{k}", short)] = f"{100 * agg(rs, 'full', f'a0.05_k{k}', metric)[0]:.1f}"
                vals[(f"k{k}", "cost")] = f"{float(np.mean([cost_at(r, 'full', COST0['c_miss'], COST0['c_fa'], key=f'a0.05_k{k}') for r in rs])):.0f}"
            vals[("full", "covh")] = f"{100 * agg(rs, 'full', 'a0.05_k0', 'coverage_healthy')[0]:.1f}"
            vals[("full", "covf")] = f"{100 * agg(rs, 'full', 'a0.05_k0', 'coverage_faulty')[0]:.1f}"
            vals[("continue", "cost")] = f"{mean_cost(rs, 'continue_all', COST0['c_miss'], COST0['c_fa']):.0f}"
            vals[("auc", "src")] = f"{np.mean([r['sweeps']['auroc']['src'] for r in rs]):.2f}"
            vals[("auc", "tta")] = f"{np.mean([r['sweeps']['auroc']['tta'] for r in rs]):.2f}"
            for (v, mt), val in vals.items():
                f.write(f"\\expandafter\\def\\csname r@{s}@{v}@{mt}\\endcsname{{{val}}}\n")
        # C-MAPSS
        s = "cmapss_FD001_FD003"
        if s in R:
            for v in ("src_source", "src_target", "bn_source", "bn_target"):
                for metric, short in (("coverage", "cov"), ("overestimation_rate", "over"), ("mean_width", "width"), ("rmse_median", "rmse")):
                    x = agg(R[s], v, "a0.05_k0", metric)[0]
                    val = f"{100 * x:.1f}" if short in ("cov", "over") else f"{x:.1f}"
                    f.write(f"\\expandafter\\def\\csname r@{s}@{v}@{short}\\endcsname{{{val}}}\n")
        # averages over all classification scenarios, for the abstract and the conclusions
        scen = [s for s in SCEN if s in R and R[s][0]["meta"]["task"] == "cls"]
        maxfar = {v: 100 * np.nanmax([agg(R[s], v, "a0.05_k0", "false_alarm_rate")[0] for s in scen]) for v in ("full", "dann_softmax", "coral_softmax", "tent_softmax", "ts", "mcd", "evid")}
        f.write(f"\\newcommand{{\\maxfarDA}}{{{max(maxfar[v] for v in ('dann_softmax', 'coral_softmax', 'tent_softmax')):.1f}}}\n")
        f.write(f"\\newcommand{{\\maxfarUQ}}{{{max(maxfar[v] for v in ('ts', 'mcd', 'evid')):.1f}}}\n")
        f.write(f"\\newcommand{{\\nScenCls}}{{{len(scen)}}}\n")


# ------------------------------------------------------------------ figures
def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    scen = [s for s in SCEN if s in R and R[s][0]["meta"]["task"] == "cls"]

    # Figure: realised FAR vs target alpha, per variant (mean over scenarios), with the identity line
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    alphas = [0.01, 0.02, 0.05, 0.10]
    for v in ("full", "no_tta", "src_calib", "ts", "mcd", "evid"):
        ys = [100 * np.nanmean([agg(R[s], v, f"a{a}_k0", "false_alarm_rate")[0] for s in scen]) for a in alphas]
        ax[0].plot([100 * a for a in alphas], ys, "o-", color=C[v], label=NAMES[v], ms=4)
    ax[0].plot([0, 10], [0, 10], "--", color="k", lw=0.8, label="target")
    ax[0].set_xlabel("target false-alarm rate (%)"); ax[0].set_ylabel("realised false-alarm rate (%)"); ax[0].legend(fontsize=6, frameon=False)
    for v in ("full", "no_tta", "no_weight"):
        ys = [100 * np.nanmean([agg(R[s], v, f"a{a}_k0", "coverage")[0] for s in scen]) for a in alphas]
        ax[1].plot([100 * (1 - a) for a in alphas], ys, "s-", color=C[v], label=NAMES[v], ms=4)
    ax[1].plot([90, 99], [90, 99], "--", color="k", lw=0.8)
    ax[1].set_xlabel("target coverage (%)"); ax[1].set_ylabel("realised coverage (%)"); ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_guarantees.pdf")); fig.savefig(os.path.join(FIG, "fig_guarantees.png"), dpi=300); plt.close(fig)

    # Figure: coverage and set size vs k labelled target windows (full variant)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for s in scen:
        ks = R[s][0]["ks"]
        cov = [100 * agg(R[s], "full", f"a0.05_k{k}", "coverage")[0] for k in ks]
        sz = [agg(R[s], "full", f"a0.05_k{k}", "set_size")[0] for k in ks]
        ax[0].plot(ks, cov, "o-", ms=3, label=SCEN[s].replace("$\\to$", "→").replace("$-$", "−"))
        ax[1].plot(ks, sz, "o-", ms=3)
    ax[0].axhline(95, ls="--", color="k", lw=0.8); ax[0].set_xlabel("labelled target windows per class, k"); ax[0].set_ylabel("coverage (%)"); ax[0].legend(fontsize=5, frameon=False, ncol=2)
    ax[1].set_xlabel("labelled target windows per class, k"); ax[1].set_ylabel("mean set size")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ksweep.pdf")); fig.savefig(os.path.join(FIG, "fig_ksweep.png"), dpi=300); plt.close(fig)

    # Figure: ROC of the novelty score (healthy vs faulty) before/after TTA, one panel per scenario (seed 0)
    n = len(scen)
    fig, axs = plt.subplots(2, int(np.ceil(n / 2)), figsize=(1.8 * np.ceil(n / 2), 3.8))
    for ax_, s in zip(axs.ravel(), scen):
        z = np.load(os.path.join(RES, f"{s}_s0.npz"))
        y = z["y_te"] != 0
        for key, col, lab in (("nov_src_te", C["no_tta"], "source model"), ("nov_tta_te", C["full"], "after TTA")):
            fpr, tpr, auc = _roc(z[key], y)
            ax_.plot(fpr, tpr, color=col, lw=1, label=f"{lab} (AUC {auc:.3f})")
        ax_.plot([0, 1], [0, 1], ":", color="grey", lw=0.6); ax_.set_title(SCEN[s].replace("$\\to$", "→").replace("$-$", "−"), fontsize=7); ax_.legend(fontsize=5, frameon=False, loc="lower right")
        ax_.set_xlabel("FPR", fontsize=7); ax_.set_ylabel("TPR", fontsize=7)
    for ax_ in axs.ravel()[n:]:
        ax_.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_roc.pdf")); fig.savefig(os.path.join(FIG, "fig_roc.png"), dpi=300); plt.close(fig)

    # Figure: confusion matrices (source model vs full) for the cross-machine scenarios
    cm_scen = [s for s in ("cwru_to_pad", "pad_to_cwru", "pad_art_real", "cwru_DE_FE") if s in R]
    fig, axs = plt.subplots(2, len(cm_scen), figsize=(1.9 * len(cm_scen), 3.9))
    for j, s in enumerate(cm_scen):
        r = R[s][0]
        y = np.array(r["sizes"] and json.loads(json.dumps(np.load(os.path.join(RES, f"{s}_s0.npz"))["y_te"].tolist())))
        for i, (pred, lab) in enumerate(((np.array(r["pred_src"]), "source model"), (np.array(r["pred_full"]), "ASSURE-FD"))):
            K = r["meta"]["n_classes"]
            cm = np.zeros((K, K), int)
            for a, b in zip(y, pred):
                cm[a, b] += 1
            cmn = cm / cm.sum(1, keepdims=True).clip(1)
            ax_ = axs[i, j] if len(cm_scen) > 1 else axs[i]
            ax_.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for a in range(K):
                for b in range(K):
                    ax_.text(b, a, f"{cmn[a, b]:.2f}", ha="center", va="center", fontsize=6, color="white" if cmn[a, b] > 0.6 else "black")
            ax_.set_xticks(range(K)); ax_.set_yticks(range(K)); ax_.set_xticklabels([c[:6] for c in r["meta"]["classes"]], fontsize=6, rotation=45); ax_.set_yticklabels([c[:6] for c in r["meta"]["classes"]], fontsize=6)
            ax_.set_title(f"{SCEN[s].replace('$\\to$', '→')}: {lab}", fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_confusion.pdf")); fig.savefig(os.path.join(FIG, "fig_confusion.png"), dpi=300); plt.close(fig)

    # Figure: training curves (seed 0, three scenarios)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax_, s in zip(axs, ["cwru_L0_L2", "pad_art_real", "mimii_6_m6"]):
        if s not in R:
            continue
        c = R[s][0]["curve"]
        ax_.plot([e["train_loss"] for e in c], color=C["full"], label="training loss")
        ax_.plot([e["val_loss"] for e in c], color=C["no_tta"], label="validation loss")
        ax2 = ax_.twinx(); ax2.plot([100 * e["val_metric"] for e in c], color=C["ts"], lw=0.8, label="validation accuracy"); ax2.set_ylabel("accuracy (%)", fontsize=7)
        ax_.set_title(SCEN[s].replace("$\\to$", "→").replace("$-$", "−"), fontsize=8); ax_.set_xlabel("epoch"); ax_.set_ylabel("loss")
        ax_.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_training.pdf")); fig.savefig(os.path.join(FIG, "fig_training.png"), dpi=300); plt.close(fig)

    # Figure: t-SNE of target test embeddings before and after TTA (cross-machine scenario)
    try:
        from sklearn.manifold import TSNE
        s = "cwru_to_pad" if "cwru_to_pad" in R else scen[0]
        z = np.load(os.path.join(RES, f"{s}_s0.npz"))
        classes = R[s][0]["meta"]["classes"]
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
        rng = np.random.default_rng(0)
        y_all = z["y_te"]
        sub = np.concatenate([rng.permutation(np.where(y_all == c)[0])[:1000] for c in range(len(classes))])
        for ax_, key, lab in zip(axs, ("z_src_te", "z_tta_te"), ("source model", "after test-time adaptation")):
            e = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(z[key][sub])
            for c in range(len(classes)):
                m = y_all[sub] == c
                ax_.scatter(e[m, 0], e[m, 1], s=4, alpha=0.7, label=classes[c], color=["#1b4965", "#c1121f", "#e09f3e", "#6a994e"][c])
            ax_.set_title(lab, fontsize=8); ax_.set_xticks([]); ax_.set_yticks([])
        axs[0].legend(fontsize=6, frameon=False, markerscale=2)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tsne.pdf")); fig.savefig(os.path.join(FIG, "fig_tsne.png"), dpi=300); plt.close(fig)
    except Exception as e:
        print("t-SNE skipped:", e)

    # Figure: XAI on real signals (integrated gradients as colour on the raw window + band shares)
    s = "cwru_L0_L2" if "cwru_L0_L2" in R else scen[0]
    z = np.load(os.path.join(RES, f"{s}_s0.npz"))
    if "xai_x" in z:
        classes = R[s][0]["meta"]["classes"]
        n = len(z["xai_y"])
        fig, axs = plt.subplots(n, 2, figsize=(7.2, 1.3 * n), gridspec_kw={"width_ratios": [3, 1]})
        fs = R[s][0]["meta"].get("fs", 12000)
        t = np.arange(z["xai_x"].shape[1]) / fs * 1000
        for i in range(n):
            x, ig = z["xai_x"][i], z["xai_ig"][i]
            a = np.abs(ig) / (np.abs(ig).max() + 1e-12)
            axs[i, 0].scatter(t, x, c=a, cmap="inferno", s=1.5)
            axs[i, 0].set_ylabel(classes[z["xai_y"][i]], fontsize=7); axs[i, 0].set_yticks([])
            b = z["xai_bands"][i]
            edges = np.linspace(0, fs / 2, len(b) + 1) / 1000
            axs[i, 1].bar(edges[:-1], b, width=np.diff(edges), align="edge", color=C["full"])
            axs[i, 1].set_yticks([])
            if i < n - 1:
                axs[i, 0].set_xticks([]); axs[i, 1].set_xticks([])
        axs[-1, 0].set_xlabel("time (ms)"); axs[-1, 1].set_xlabel("frequency band (kHz)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_xai.pdf")); fig.savefig(os.path.join(FIG, "fig_xai.png"), dpi=300); plt.close(fig)

    # Figure: risk-coverage and cost per 1000 decisions by variant
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    s = "pad_art_real" if "pad_art_real" in R else scen[0]
    for key, col, lab in (("risk_coverage_src", C["no_tta"], "source model"), ("risk_coverage_tta", C["full"], "after TTA")):
        cov, risk = R[s][0]["sweeps"][key]
        ax[0].plot(cov, risk, color=col, label=lab)
    ax[0].set_xlabel("coverage (fraction of windows decided automatically)"); ax[0].set_ylabel("selective risk"); ax[0].legend(fontsize=6, frameon=False); ax[0].set_title(SCEN[s].replace("$\\to$", "→"), fontsize=8)
    vs = ["full", "no_tta", "weighted_sets", "no_weight", "src_calib", "no_abstain", "no_crc", "ts", "mcd", "evid", "dann_softmax", "tent_softmax", "continue_all", "inspect_all"]
    costs = [np.mean([mean_cost(R[s_], v, COST0["c_miss"], COST0["c_fa"]) for s_ in scen]) for v in vs]
    ax[1].barh([NAMES[v] for v in vs][::-1], costs[::-1], color=[C[v] for v in vs][::-1])
    ax[1].set_xlabel("cost per 1000 decisions (mean over scenarios)")
    ax[1].tick_params(axis="y", labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cost.pdf")); fig.savefig(os.path.join(FIG, "fig_cost.png"), dpi=300); plt.close(fig)

    # Figure: RUL intervals on FD003 (first 40 test engines, adapted model, target calibration)
    s = "cmapss_FD001_FD003"
    if s in R:
        z = np.load(os.path.join(RES, f"{s}_s0.npz"))
        import conformal as cp
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        o = z["te_pred_bn"]; y = z["y_te"]
        idx = np.argsort(y)[:60]
        ax.fill_between(range(len(idx)), o[idx, 0], o[idx, 2], color=C["full"], alpha=0.25, label="90% interval (uncalibrated quantiles)")
        ax.plot(range(len(idx)), o[idx, 1], color=C["full"], lw=0.8, label="median prediction")
        ax.plot(range(len(idx)), y[idx], "k.", ms=3, label="true RUL")
        ax.set_xlabel("test engine (sorted by true RUL)"); ax.set_ylabel("RUL (cycles)"); ax.legend(fontsize=6, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rul.pdf")); fig.savefig(os.path.join(FIG, "fig_rul.png"), dpi=300); plt.close(fig)


def tsne_fig(R, s_pref=("pad_to_cwru", "cwru_to_pad")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    s = next(x for x in s_pref if x in R)
    z = np.load(os.path.join(RES, f"{s}_s0.npz"))
    classes = R[s][0]["meta"]["classes"]
    rng = np.random.default_rng(0)
    y_all = z["y_te"]
    sub = np.concatenate([rng.permutation(np.where(y_all == c)[0])[:1000] for c in range(len(classes))])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    cols = ["#1b4965", "#c1121f", "#e09f3e", "#6a994e"]
    for ax_, key, lab in zip(axs, ("z_src_te", "z_tta_te"), ("(a) source model", "(b) after test-time adaptation")):
        e = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(z[key][sub])
        for c in range(len(classes)):
            m = y_all[sub] == c
            ax_.scatter(e[m, 0], e[m, 1], s=3, alpha=0.6, color=cols[c], label=classes[c], rasterized=True)
        ax_.set_title(f"{lab}, {_txt(s)}", fontsize=8.5, loc="left"); ax_.set_xticks([]); ax_.set_yticks([])
        for sp in ax_.spines.values():
            sp.set_visible(True); sp.set_color("#999999")
    axs[1].legend(fontsize=7, markerscale=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_tsne.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIG, "fig_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


PAL = ["#1b4965", "#5fa8d3", "#2a9d8f", "#c1121f", "#e09f3e", "#6a994e", "#9e2a2b", "#7b2cbf", "#bc6c25", "#495057"]
SHORT = {"Healthy": "H", "Inner race": "IR", "Outer race": "OR", "Ball": "B", "Normal": "N", "Abnormal": "A"}


def _txt(s):
    return SCEN[s].replace("$\\to$", "→").replace("$-$", "−")


def figures2(R):
    """Publication figures (consistent palette, legends outside the data, panels matching the claims)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False})
    scen = [s for s in SCEN if s in R and R[s][0]["meta"]["task"] == "cls"]
    alphas = [0.01, 0.02, 0.05, 0.10]
    A = [100 * a for a in alphas]

    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def mean_metric(v, k, metric, alpha):
        return 100 * np.nanmean([agg(R[s], v, f"a{alpha}_k{k}", metric)[0] for s in scen])

    # ---- Figure: guarantees. (a) proposed variants, (b) baselines, (c) class coverage
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.3))
    for v, mk in (("full", "o"), ("no_tta", "s"), ("src_calib", "^")):
        ax[0].plot(A, [mean_metric(v, 0, "false_alarm_rate", a) for a in alphas], mk + "-", color=C[v], ms=4, label=NAMES[v])
    ax[0].plot([0, 10], [0, 10], "k--", lw=0.8, label="target")
    ax[0].set_xlim(0, 10.5); ax[0].set_ylim(0, 12)
    ax[0].set_title("(a) proposed system", fontsize=9, loc="left"); ax[0].set_xlabel("target FAR (%)"); ax[0].set_ylabel("realised FAR (%)")
    ax[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    for v, mk in (("ts", "o"), ("mcd", "s"), ("evid", "^"), ("dann_softmax", "D"), ("tent_softmax", "v")):
        ax[1].plot(A, [mean_metric(v, 0, "false_alarm_rate", a) for a in alphas], mk + "-", color=C[v] if v in ("ts", "mcd", "evid") else {"dann_softmax": "#7b2cbf", "tent_softmax": "#495057"}[v], ms=4, label=NAMES[v])
    ax[1].plot([0, 10], [0, 10], "k--", lw=0.8)
    ax[1].set_xlim(0, 10.5); ax[1].set_ylim(0, 100)
    ax[1].set_title("(b) compared methods", fontsize=9, loc="left"); ax[1].set_xlabel("target FAR (%)")
    ax[1].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    cov_t = [100 - a for a in A]
    ax[2].plot(cov_t, [mean_metric("full", 0, "coverage_healthy", a) for a in alphas], "o-", color=C["full"], ms=4, label="healthy class, run-in calibrated")
    ax[2].plot(cov_t, [mean_metric("full", 0, "coverage_faulty", a) for a in alphas], "s--", color="#e09f3e", ms=4, label="fault classes, $k=0$ (source)")
    ok = [a for a in alphas if 50 >= int(np.ceil(1 / a)) - 1]          # levels that k = 50 windows can certify
    ax[2].plot([100 - 100 * a for a in ok], [mean_metric("full", 50, "coverage_faulty", a) for a in ok], "s-", color="#c1121f", ms=4, label="fault classes, $k=50$ (target)")
    ax[2].plot([90, 99], [90, 99], "k--", lw=0.8)
    ax[2].set_title("(c) set coverage", fontsize=9, loc="left"); ax[2].set_xlabel("target coverage (%)"); ax[2].set_ylabel("realised coverage (%)")
    ax[2].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_guarantees")

    # ---- Figure: labelled target windows: fault coverage and inspection rate vs k
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for i, s in enumerate(scen):
        ks = R[s][0]["ks"]
        ax[0].plot(ks, [100 * agg(R[s], "full", f"a0.05_k{k}", "coverage_faulty")[0] for k in ks], "o-", ms=3, lw=1.2, color=PAL[i % len(PAL)], label=_txt(s))
        ax[1].plot(ks, [100 * agg(R[s], "full", f"a0.05_k{k}", "inspection_rate")[0] for k in ks], "o-", ms=3, lw=1.2, color=PAL[i % len(PAL)])
    ax[0].axhline(95, ls="--", color="k", lw=0.8)
    ax[0].set_xlabel("labelled target windows per class, $k$"); ax[0].set_ylabel("fault-class coverage (%)"); ax[0].set_title("(a)", fontsize=9, loc="left")
    ax[1].set_xlabel("labelled target windows per class, $k$"); ax[1].set_ylabel("inspection rate (%)"); ax[1].set_title("(b)", fontsize=9, loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_ksweep")

    # ---- Figure: ROC of the novelty score, AUC in the titles, one legend
    n = len(scen); cols = int(np.ceil(n / 2))
    fig, axs = plt.subplots(2, cols, figsize=(1.6 * cols, 4.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.55})
    for ax_, s in zip(axs.ravel(), scen):
        z = np.load(os.path.join(RES, f"{s}_s0.npz"))
        y = z["y_te"] != 0
        aucs = []
        for key, col in (("nov_src_te", C["no_tta"]), ("nov_tta_te", C["full"])):
            fpr, tpr, auc = _roc(z[key], y)
            ax_.plot(fpr, tpr, color=col, lw=1.1)
            aucs.append(auc)
        ax_.plot([0, 1], [0, 1], ":", color="grey", lw=0.6)
        ax_.set_title(f"{_txt(s)}\nAUC {aucs[0]:.2f} / {aucs[1]:.2f}", fontsize=7)
        ax_.set_aspect("equal")
    for ax_ in axs[-1]:
        ax_.set_xlabel("false-positive rate", fontsize=7)
    for ax_ in axs[:, 0]:
        ax_.set_ylabel("true-positive rate", fontsize=7)
    for ax_ in axs.ravel()[n:]:
        ax_.axis("off")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=C["no_tta"]), Line2D([0], [0], color=C["full"])], ["source model", "after test-time adaptation"],
               loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig_roc")

    # ---- Figure: confusion matrices, source model vs adapted model
    cm_scen = [s for s in ("cwru_DE_FE", "cwru_to_pad", "pad_to_cwru", "pad_art_real") if s in R]
    fig, axs = plt.subplots(2, len(cm_scen), figsize=(1.85 * len(cm_scen), 3.9), squeeze=False)
    for j, s in enumerate(cm_scen):
        r = R[s][0]
        y = np.load(os.path.join(RES, f"{s}_s0.npz"))["y_te"]
        K = r["meta"]["n_classes"]
        lab = [SHORT.get(c, c[:2]) for c in r["meta"]["classes"]]
        for i, (pred, row) in enumerate(((np.array(r["pred_src"]), "source model"), (np.array(r["pred_full"]), "after adaptation"))):
            cm = np.zeros((K, K))
            np.add.at(cm, (y, pred), 1)
            cmn = cm / cm.sum(1, keepdims=True).clip(1)
            ax_ = axs[i, j]
            ax_.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for a in range(K):
                for b in range(K):
                    ax_.text(b, a, f"{cmn[a, b]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cmn[a, b] > 0.55 else "black")
            ax_.set_xticks(range(K)); ax_.set_yticks(range(K)); ax_.set_xticklabels(lab, fontsize=7); ax_.set_yticklabels(lab, fontsize=7)
            if i == 0:
                ax_.set_title(_txt(s), fontsize=7.5)
            if j == 0:
                ax_.set_ylabel(f"{row}\ntrue class", fontsize=7)
            if i == 1:
                ax_.set_xlabel("predicted class", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_confusion")

    # ---- Figure: training curves (seed 0)
    tr_s = [s for s in ("cwru_L0_L2", "pad_to_cwru", "mimii_6_m6") if s in R]
    fig, axs = plt.subplots(1, len(tr_s), figsize=(2.45 * len(tr_s), 2.3), squeeze=False)
    for ax_, s in zip(axs[0], tr_s):
        c = R[s][0]["curve"]
        ep = np.arange(1, len(c) + 1)
        ax_.plot(ep, [e["train_loss"] for e in c], color=C["full"], lw=1.1, label="training loss")
        ax_.plot(ep, [e["val_loss"] for e in c], color="#c1121f", lw=1.1, label="validation loss")
        ax2 = ax_.twinx(); ax2.spines["right"].set_visible(True)
        ax2.plot(ep, [100 * e["val_metric"] for e in c], color="#6a994e", lw=1.0, ls="--", label="validation accuracy")
        ax2.set_ylim(0, 101); ax2.tick_params(labelsize=7)
        ax_.set_title(_txt(s), fontsize=8); ax_.set_xlabel("epoch"); ax_.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("loss")
    ax2.set_ylabel("source validation accuracy (%)", fontsize=7)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    fig.legend(h1 + [Line2D([0], [0], color="#6a994e", ls="--")], l1 + ["validation accuracy"], loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig_training")

    # ---- Figure: selective risk (mean over scenarios) and cost per 1000 decisions (log scale)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for key, col, lab in (("risk_coverage_src", C["no_tta"], "source model"), ("risk_coverage_tta", C["full"], "after test-time adaptation")):
        curves = [np.array(r["sweeps"][key]) for s in scen for r in R[s]]
        cov = curves[0][0]
        risk = np.mean([c[1] for c in curves], 0)
        ax[0].plot(100 * cov, 100 * risk, color=col, lw=1.3, label=lab)
    ax[0].set_xlabel("windows decided automatically (%)"); ax[0].set_ylabel("error among them (%)"); ax[0].set_title("(a) selective risk, mean over scenarios", fontsize=8.5, loc="left")
    ax[0].legend(fontsize=6.5, loc="upper left")
    vs = ["full", "no_tta", "weighted_sets", "no_weight", "src_calib", "no_abstain", "no_crc", "ts", "mcd", "evid", "dann_softmax", "tent_softmax", "continue_all", "inspect_all"]
    costs = [np.mean([mean_cost(R[s_], v, COST0["c_miss"], COST0["c_fa"]) for s_ in scen]) for v in vs]
    yy = np.arange(len(vs))[::-1]
    ax[1].barh(yy, costs, color=[C.get(v, "#7f7f7f") for v in vs], height=0.7)
    ax[1].set_xscale("log")
    ax[1].set_yticks(yy); ax[1].set_yticklabels([NAMES[v] for v in vs], fontsize=6.5)
    for t in ax[1].get_yticklabels():
        if t.get_text() == NAMES["full"]:
            t.set_fontweight("bold")
    for yv, cv in zip(yy, costs):
        ax[1].text(cv * 1.08, yv, f"{cv:.0f}", va="center", fontsize=6)
    ax[1].set_xlabel("cost per 1000 decisions, 100 : 5 : 1 (log scale)"); ax[1].set_title("(b) mean cost over scenarios", fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, "fig_cost")

    # ---- Figure: conformal RUL intervals on FD003 (seed 0), source- vs target-calibrated, overestimates marked
    s = "cmapss_FD001_FD003"
    if s in R and "offsets" in R[s][0]:
        r0 = R[s][0]
        z = np.load(os.path.join(RES, f"{s}_s0.npz"))
        o, y = z["te_pred_src"], z["y_te"]
        idx = np.argsort(y)
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True)
        for ax_, calib, title in ((axs[0], "source", "(a) calibrated on FD001 engines"), (axs[1], "target", "(b) calibrated on FD003 engines")):
            dlo, dhi = r0["offsets"][f"src_{calib}_a0.05"]
            lo, hi = o[idx, 0] - dlo, o[idx, 2] + dhi
            xx = np.arange(len(idx))
            ax_.fill_between(xx, np.clip(lo, 0, None), hi, color=C["full"], alpha=0.22, lw=0, step="mid", label="conformal interval (clipped at 0)")
            ax_.plot(xx, o[idx, 1], "_", color=C["full"], ms=5, mew=1.2, label="median prediction")
            over = y[idx] < lo
            ax_.plot(xx[~over], y[idx][~over], "o", color="k", ms=2.2, label="true RUL")
            ax_.plot(xx[over], y[idx][over], "o", color="#c1121f", ms=3, label="overestimated")
            ax_.set_title(f"{title}: {100 * over.mean():.0f}% over", fontsize=8.5, loc="left"); ax_.set_xlabel("test engine (sorted by true RUL)")
        axs[0].set_ylabel("remaining useful life (cycles)")
        axs[0].set_ylim(0, 160)
        fig.legend(*axs[1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        save(fig, "fig_rul")


def _roc(score, pos):
    s = np.asarray(score); p = np.asarray(pos, bool)
    order = np.argsort(-s)
    tp = np.cumsum(p[order]); fp = np.cumsum(~p[order])
    tpr = np.r_[0, tp / max(p.sum(), 1)]; fpr = np.r_[0, fp / max((~p).sum(), 1)]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


if __name__ == "__main__":
    R = load_all()
    if not R:
        raise SystemExit("no results yet")
    table_main(R); table_ablation(R); table_cost_sensitivity(R); table_rul(R); macros(R)
    tsne_fig(R)                       # t-SNE of target embeddings before and after adaptation
    figures2(R)                       # all other result figures; the XAI figure is drawn by xai.py
    print("assets written for", sorted(R))
