"""Shift scenarios for ASSURE-FD. Every scenario returns one dict with source and target parts.

Source  : x_tr/y_tr (training), x_va/y_va (model selection), x_cal/y_cal (source calibration, used by the
          'source-calibrated' ablation and as the source side of the domain classifier).
Target  : x_comm (healthy commissioning windows: the only target data the plant can always provide),
          x_unl (unlabelled mixed windows for test-time adaptation),
          x_lab/y_lab (a labelled pool from which k windows per class are drawn for the k-sweep),
          x_te/y_te (held-out test windows), plus y_unl kept only for analysis.
Leakage control: splits are made on whole recordings (Paderborn, MIMII clips, C-MAPSS engines) or on contiguous
time segments of a recording with a guard gap (CWRU, which has one recording per class, severity and load), and
windows are cut only after the split. Normalisation statistics come from the source training windows.
Raw-file readers follow the FedMamba-ChebKAN loaders (same file ids and channels); the scenario logic is new.
"""
import glob
import os

import numpy as np
import scipy.io as sio
import scipy.signal

ROOT = os.environ.get("ASSURE_DATA", r"G:/Training_Data/FedMamba_ChebKAN")
MIMII_ROOT = os.environ.get("ASSURE_MIMII", os.path.join(ROOT, "MIMII"))
FULL = os.environ.get("ASSURE_FULL", "0") == "1"   # full data: all windows, all Paderborn bearings and settings
CACHE = os.path.join(ROOT, "cache_assure_full" if FULL else "cache_assure")
SEED = 2026
L = 1024            # window length at 12 kHz
HEALTHY = 0


def _windows(x, length, hop):
    n = 1 + (len(x) - length) // hop if len(x) >= length else 0
    if n <= 0:
        return np.zeros((0, length), np.float32)
    idx = np.arange(length)[None, :] + hop * np.arange(n)[:, None]
    return x[idx].astype(np.float32)


def _take(rng, arr, k):
    if k is None or len(arr) <= k:
        return arr
    return arr[np.sort(rng.choice(len(arr), k, replace=False))]


def _stack(parts):
    xs = [p[0] for p in parts if len(p[0])]
    ys = [p[1] for p in parts if len(p[1])]
    if not xs:
        return np.zeros((0, 1, L), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(xs), np.concatenate(ys)


# ------------------------------------------------------------------ CWRU (12 kHz drive-end experiments)
CWRU_FILES = {  # class -> per load (0..3 hp) -> file ids (7/14/21 mil severities)
    0: [[97], [98], [99], [100]],
    1: [[105, 169, 209], [106, 170, 210], [107, 171, 211], [108, 172, 212]],   # inner race
    2: [[130, 197, 234], [131, 198, 235], [132, 199, 236], [133, 200, 237]],   # outer race @6:00
    3: [[118, 185, 222], [119, 186, 223], [120, 187, 224], [121, 188, 225]],   # ball
}
CWRU_CLASSES = ["Healthy", "Inner race", "Outer race", "Ball"]
CWRU_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def _cwru_signal(fid, sensor):
    folder = "normal_baseline" if fid in (97, 98, 99, 100) else "12k_drive_end"
    m = sio.loadmat(os.path.join(ROOT, "CWRU", folder, f"{fid}.mat"))
    keys = [k for k in m if k.endswith(f"_{sensor}_time")]
    return m[keys[0]].ravel().astype(np.float32) if keys else None


def _segments_source(x):
    """Contiguous 70 / 15 / 15 % segments: train / val / source-calibration (guard gap of one window at each cut)."""
    a, b = int(0.70 * len(x)), int(0.85 * len(x))
    return {"tr": x[:a - L], "va": x[a:b - L], "cal": x[b:]}


def _segments_target(x):
    """Contiguous 45 % pool (commissioning / unlabelled / labelled) and 50 % test with a 5 % guard gap."""
    a, b = int(0.45 * len(x)), int(0.50 * len(x))
    return {"pool": x[:a], "te": x[b:]}


def _cwru_side(loads, sensor, classes, role, rng, cap):
    """role = 'source' or 'target'. Returns dict of (x, y) per split."""
    out = {}
    for cls in classes:
        for load in loads:
            for fid in CWRU_FILES[cls][load]:
                x = _cwru_signal(fid, sensor)
                if x is None:
                    continue
                segs = _segments_source(x) if role == "source" else _segments_target(x)
                for s, seg in segs.items():
                    w = _windows(seg, L, L // 2 if s in ("tr", "pool") else L)
                    out.setdefault(s, []).append((w, np.full(len(w), cls, np.int64)))
    return {s: _stack(v) for s, v in out.items()}


# ------------------------------------------------------------------ Paderborn (64 kHz, resampled to 12 kHz)
PAD_HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]
PAD_REAL_OR = ["KA04", "KA15", "KA16", "KA22", "KA30"]
PAD_REAL_IR = ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
PAD_ART_OR = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"]
PAD_ART_IR = ["KI01", "KI03", "KI05", "KI07", "KI08"]
PAD_SETTINGS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
PAD_CLASSES = ["Healthy", "Inner race", "Outer race"]
SKIPPED = []
MANIFEST = []      # recording-level split records (written by manifest.py)


def _sel(lst, k):
    """All bearings in full-data mode, the first k otherwise."""
    return list(lst) if FULL else list(lst[:k])      # same label order as CWRU classes 0-2


def _paderborn_vib(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m[[k for k in m if not k.startswith("__")][0]]
    for ch in s.Y:
        if ch.Name == "vibration_1":
            x = np.asarray(ch.Data, dtype=np.float32)
            return scipy.signal.resample_poly(x, 3, 16).astype(np.float32)   # 64 kHz -> 12 kHz
    raise KeyError("vibration_1")


def _pad_files(bearing, setting):
    fs = glob.glob(os.path.join(ROOT, "Paderborn", bearing, f"{setting}_{bearing}_*.mat"))
    return sorted(fs, key=lambda p: int(p.rsplit("_", 1)[1][:-4]))


def _pad_side(bearings_by_class, setting, role, rng):
    """Recording-level split of the 20 files of each (bearing, setting). Source: 14 train / 3 val / 3 cal.
    Target: 10 pool / 10 test."""
    out = {}
    settings = PAD_SETTINGS if FULL else [setting]
    for cls, bearings in bearings_by_class.items():
        for b, st in [(b, st) for b in bearings for st in settings]:
            fs = _pad_files(b, st)
            if not fs:
                continue
            fs = [fs[i] for i in rng.permutation(len(fs))]
            if role == "source":
                split = {"tr": fs[:14], "va": fs[14:17], "cal": fs[17:20]}
            else:
                split = {"pool": fs[:10], "te": fs[10:20]}
            MANIFEST.append({"dataset": "Paderborn", "role": role, "bearing": b, "setting": st, "class": PAD_CLASSES[cls],
                             "files": {s: [os.path.basename(f) for f in v] for s, v in split.items()}})
            for s, v in split.items():
                hop = L // 2 if s in ("tr", "pool") else L
                ws = []
                for f in v:
                    try:
                        ws.append(_windows(_paderborn_vib(f), L, hop))
                    except Exception as e:     # a few files of the public release cannot be parsed
                        SKIPPED.append(os.path.basename(f))
                        print("skipped unreadable file", os.path.basename(f), type(e).__name__, flush=True)
                if not ws:
                    continue
                w = np.concatenate(ws)
                out.setdefault(s, []).append((w, np.full(len(w), cls, np.int64)))
    return {s: _stack(v) for s, v in out.items()}


# ------------------------------------------------------------------ MIMII fan (log-Mel 64 x 313)
def _mel_filterbank(sr=16000, n_fft=1024, n_mels=64, fmin=0.0, fmax=8000.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    hz = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def _logmel(wav, fb, n_fft=1024, hop=512):
    _, _, Z = scipy.signal.stft(wav, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary="even", padded=False)
    return 10.0 * np.log10(fb @ (np.abs(Z) ** 2).astype(np.float32) + 1e-10)


def _mimii_feats(mid, snr):
    """Cached log-Mel features of all normal and abnormal clips of one fan at one SNR."""
    feat_dir = os.path.join(ROOT, "cache_assure")      # features do not depend on FULL; shared by both modes
    os.makedirs(feat_dir, exist_ok=True)
    path = os.path.join(feat_dir, f"mimii_{mid}_{snr}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["xn"], z["xa"]
    import soundfile as sf
    fb = _mel_filterbank()
    base = os.path.join(MIMII_ROOT, snr, "fan", mid)
    feats = lambda paths: np.stack([_logmel(sf.read(p)[0][:, 0].astype(np.float32), fb) for p in paths]).astype(np.float32)
    xn = feats(sorted(glob.glob(os.path.join(base, "normal", "*.wav"))))
    xa = feats(sorted(glob.glob(os.path.join(base, "abnormal", "*.wav"))))
    np.savez(path, xn=xn, xa=xa)
    return xn, xa


def _mimii_side(ids, snr, role, rng, n_normal_source=600, n_abn_source=None):
    out = {}
    for mid in ids:
        xn, xa = _mimii_feats(mid, snr)
        pn = rng.permutation(len(xn))
        pa = rng.permutation(len(xa))
        xn, xa = xn[pn], xa[pa]
        nn, na = len(xn), len(xa)
        if role == "source":
            cuts = {"tr": (0, int(0.7 * nn), 0, int(0.7 * na)), "va": (int(0.7 * nn), int(0.85 * nn), int(0.7 * na), int(0.85 * na)),
                    "cal": (int(0.85 * nn), nn, int(0.85 * na), na)}
        else:
            cuts = {"pool": (0, nn // 2, 0, na // 2), "te": (nn // 2, nn, na // 2, na)}
        split = {s: (xn[a:b], xa[c:e]) for s, (a, b, c, e) in cuts.items()}
        base = os.path.join(MIMII_ROOT, snr, "fan", mid)
        names_n = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(base, "normal", "*.wav")))]
        names_a = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))]
        if len(names_n) == nn and len(names_a) == na:          # clip names follow the sorted order of the feature cache
            MANIFEST.append({"dataset": "MIMII", "role": role, "fan": mid, "snr": snr,
                             "files": {s: {"normal": [names_n[i] for i in pn[a:b]], "abnormal": [names_a[i] for i in pa[c:e]]}
                                       for s, (a, b, c, e) in cuts.items()}})
        for s, (n_, a_) in split.items():
            out.setdefault(s, []).append((n_, np.zeros(len(n_), np.int64)))
            out.setdefault(s, []).append((a_, np.ones(len(a_), np.int64)))
    return {s: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v])) for s, v in out.items()}


# ------------------------------------------------------------------ C-MAPSS (RUL)
CMAPSS_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
RUL_CAP = 125
WINDOW = 30


def _cmapss_read(path):
    a = np.loadtxt(path)
    return a[:, 0].astype(int), a[:, [4 + s for s in CMAPSS_SENSORS]].astype(np.float32)


def _cmapss_engine_windows(uid, X, units, mu, sd, last_only=False):
    xs, ys, eng = [], [], []
    for u in units:
        xu = np.clip((X[uid == u] - mu) / sd, -6, 6).astype(np.float32)
        T = len(xu)
        rul = np.minimum(np.arange(T)[::-1], RUL_CAP).astype(np.float32)
        if last_only:
            if T < WINDOW:
                xu = np.vstack([np.repeat(xu[:1], WINDOW - T, 0), xu])
                T = WINDOW
            starts = [T - WINDOW]
        else:
            starts = range(0, T - WINDOW + 1)
        for s0 in starts:
            xs.append(xu[s0:s0 + WINDOW].T)
            ys.append(rul[min(s0 + WINDOW - 1, T - 1)] if not last_only else 0.0)
            eng.append(u)
    return np.stack(xs), np.array(ys, np.float32), np.array(eng)


def cmapss_scenario(src_fd=1, tgt_fd=3, seed=SEED):
    rng = np.random.default_rng(seed)
    base = os.path.join(ROOT, "C-MAPSS", "CMAPSSData")
    uid, X = _cmapss_read(os.path.join(base, f"train_FD00{src_fd}.txt"))
    units = rng.permutation(np.unique(uid))
    n = len(units)
    tr_u, va_u, cal_u = units[: int(0.7 * n)], units[int(0.7 * n): int(0.8 * n)], units[int(0.8 * n):]
    MANIFEST.append({"dataset": "C-MAPSS", "role": "source", "subset": f"FD00{src_fd}",
                     "engines": {"tr": sorted(int(u) for u in tr_u), "va": sorted(int(u) for u in va_u), "cal": sorted(int(u) for u in cal_u)}})
    mu, sd = X[np.isin(uid, tr_u)].mean(0), X[np.isin(uid, tr_u)].std(0) + 1e-6
    d = {}
    d["x_tr"], d["y_tr"], _ = _cmapss_engine_windows(uid, X, tr_u, mu, sd)
    d["x_va"], d["y_va"], _ = _cmapss_engine_windows(uid, X, va_u, mu, sd)
    d["x_cal"], d["y_cal"], d["eng_cal"] = _cmapss_engine_windows(uid, X, cal_u, mu, sd)
    # target: FD003 training engines form the pool (commissioning = early-life windows, unlabelled, labelled);
    # FD003 official test engines with their RUL file form the test set (one prediction per engine)
    tid, TX = _cmapss_read(os.path.join(base, f"train_FD00{tgt_fd}.txt"))
    tunits = rng.permutation(np.unique(tid))
    xp, yp, ep = _cmapss_engine_windows(tid, TX, tunits, mu, sd)
    early = yp >= RUL_CAP                       # windows far from failure: the 'healthy commissioning' analogue
    d["x_comm"] = xp[early][rng.permutation(early.sum())[:800]]
    half = tunits[: len(tunits) // 2]
    MANIFEST.append({"dataset": "C-MAPSS", "role": "target", "subset": f"FD00{tgt_fd}",
                     "engines": {"unlabelled": sorted(int(u) for u in half), "labelled": sorted(int(u) for u in tunits[len(tunits) // 2:]),
                                 "test": "official test engines of the subset, last window of each, with the official RUL file"}})
    d["x_unl"], d["y_unl"] = xp[np.isin(ep, half)], yp[np.isin(ep, half)]
    d["x_lab"], d["y_lab"], d["eng_lab"] = xp[~np.isin(ep, half)], yp[~np.isin(ep, half)], ep[~np.isin(ep, half)]
    qid, QX = _cmapss_read(os.path.join(base, f"test_FD00{tgt_fd}.txt"))
    true_rul = np.loadtxt(os.path.join(base, f"RUL_FD00{tgt_fd}.txt")).astype(np.float32)
    xt, _, et = _cmapss_engine_windows(qid, QX, np.unique(qid), mu, sd, last_only=True)
    d["x_te"], d["y_te"] = xt, np.minimum(true_rul[et - 1], RUL_CAP)
    d["meta"] = {"task": "rul", "in_ch": len(CMAPSS_SENSORS), "n_classes": 0, "src": f"FD00{src_fd}", "tgt": f"FD00{tgt_fd}"}
    return d


# ------------------------------------------------------------------ scenario assembly
def _assemble(src, tgt, meta, rng, caps):
    """Normalise with source training statistics; split the target pool into commissioning (healthy), unlabelled and
    labelled parts (disjoint windows); cap sizes."""
    x_tr, y_tr = src["tr"]
    mu, sd = x_tr.mean(), x_tr.std() + 1e-8
    nz = lambda a: ((a - mu) / sd).astype(np.float32)
    d = {"x_tr": nz(x_tr), "y_tr": y_tr}
    for s, name in (("va", "va"), ("cal", "cal")):
        x, y = src[s]
        sel = _take(rng, np.arange(len(y)), caps.get(name))
        d[f"x_{name}"], d[f"y_{name}"] = nz(x[sel]), y[sel]
    xp, yp = tgt["pool"]
    perm = rng.permutation(len(yp))
    xp, yp = xp[perm], yp[perm]
    healthy = np.where(yp == HEALTHY)[0]
    n_comm = min(caps["comm"], len(healthy) // 2)
    comm_idx = healthy[:n_comm]
    rest = np.setdiff1d(np.arange(len(yp)), comm_idx)
    half = rest[: len(rest) // 2]
    other = rest[len(rest) // 2:]
    d["x_comm"] = nz(xp[comm_idx])
    d["x_unl"], d["y_unl"] = nz(xp[half]), yp[half]
    d["x_lab"], d["y_lab"] = nz(xp[other]), yp[other]
    xt, yt = tgt["te"]
    sel = _take(rng, np.arange(len(yt)), caps.get("te"))
    d["x_te"], d["y_te"] = nz(xt[sel]), yt[sel]
    sel = _take(rng, np.arange(len(d["y_tr"])), caps.get("tr"))
    d["x_tr"], d["y_tr"] = d["x_tr"][sel], d["y_tr"][sel]
    d["meta"] = meta
    return d


CAPS_VIB = {"tr": 3000, "va": 600, "cal": 600, "comm": 150, "te": 1500}
CAPS_MIMII = {"tr": 2400, "va": 500, "cal": 500, "comm": 150, "te": 1200}
if FULL:   # every window of every selected recording; only the run-in set keeps its fixed size
    CAPS_VIB = {"tr": None, "va": None, "cal": None, "comm": 150, "te": None}
    CAPS_MIMII = {"tr": None, "va": None, "cal": None, "comm": 150, "te": None}


def _add_channel(side):
    return {s: (x[:, None, :] if x.ndim == 2 else x, y) for s, (x, y) in side.items()}


def build(name, seed=SEED):
    """Scenario factory. Classes for vibration scenarios: 0 healthy, 1 inner race, 2 outer race, (3 ball, CWRU only)."""
    rng = np.random.default_rng(seed)
    if name.startswith("cwru_L0_L"):
        tl = int(name[-1])
        src = _add_channel(_cwru_side([0], "DE", [0, 1, 2, 3], "source", rng, None))
        tgt = _add_channel(_cwru_side([tl], "DE", [0, 1, 2, 3], "target", rng, None))
        meta = {"task": "cls", "in_ch": 1, "n_classes": 4, "classes": CWRU_CLASSES, "src": "CWRU DE load 0", "tgt": f"CWRU DE load {tl}", "fs": 12000, "rpm": CWRU_RPM[tl]}
        return _assemble(src, tgt, meta, rng, CAPS_VIB)
    if name == "cwru_DE_FE":
        src = _add_channel(_cwru_side([0, 1, 2, 3], "DE", [0, 1, 2, 3], "source", rng, None))
        tgt = _add_channel(_cwru_side([0, 1, 2, 3], "FE", [0, 1, 2, 3], "target", rng, None))
        meta = {"task": "cls", "in_ch": 1, "n_classes": 4, "classes": CWRU_CLASSES, "src": "CWRU drive-end sensor", "tgt": "CWRU fan-end sensor", "fs": 12000, "rpm": 1772}
        return _assemble(src, tgt, meta, rng, CAPS_VIB)
    if name == "cwru_to_pad":
        src = _add_channel(_cwru_side([0, 1, 2, 3], "DE", [0, 1, 2], "source", rng, None))
        tgt = _add_channel(_pad_side({0: _sel(PAD_HEALTHY, 3), 1: _sel(PAD_REAL_IR, 3), 2: _sel(PAD_REAL_OR, 3)}, "N15_M07_F10", "target", rng))
        # the CWRU ball class is kept aside as an open-set 'unknown' for the source-side analysis
        unk = _add_channel(_cwru_side([0, 1, 2, 3], "DE", [3], "source", rng, None))
        meta = {"task": "cls", "in_ch": 1, "n_classes": 3, "classes": PAD_CLASSES, "src": "CWRU (all loads, DE)", "tgt": "Paderborn N15_M07_F10 real damage", "fs": 12000, "rpm": 1500}
        d = _assemble(src, tgt, meta, rng, CAPS_VIB)
        d["x_unknown"] = ((unk["cal"][0] - src["tr"][0].mean()) / (src["tr"][0].std() + 1e-8)).astype(np.float32)
        return d
    if name == "pad_to_cwru":
        src = _add_channel(_pad_side({0: _sel(PAD_HEALTHY, 3), 1: _sel(PAD_REAL_IR, 3), 2: _sel(PAD_REAL_OR, 3)}, "N15_M07_F10", "source", rng))
        tgt = _add_channel(_cwru_side([0, 1, 2, 3], "DE", [0, 1, 2], "target", rng, None))
        meta = {"task": "cls", "in_ch": 1, "n_classes": 3, "classes": PAD_CLASSES, "src": "Paderborn N15_M07_F10 real damage", "tgt": "CWRU (all loads, DE)", "fs": 12000, "rpm": 1772}
        return _assemble(src, tgt, meta, rng, CAPS_VIB)
    if name == "pad_art_real":
        src = _add_channel(_pad_side({0: PAD_HEALTHY[:3], 1: _sel(PAD_ART_IR, 4), 2: _sel(PAD_ART_OR, 4)}, "N15_M07_F10", "source", rng))
        tgt = _add_channel(_pad_side({0: PAD_HEALTHY[3:6], 1: _sel(PAD_REAL_IR, 4), 2: _sel(PAD_REAL_OR, 4)}, "N15_M07_F10", "target", rng))
        meta = {"task": "cls", "in_ch": 1, "n_classes": 3, "classes": PAD_CLASSES, "src": "Paderborn artificial damage", "tgt": "Paderborn real damage (other healthy bearings)", "fs": 12000, "rpm": 1500}
        return _assemble(src, tgt, meta, rng, CAPS_VIB)
    if name in ("mimii_6_0", "mimii_6_m6"):
        tsnr = "0_dB" if name.endswith("_0") else "-6_dB"
        ids = ["id_00", "id_02", "id_04", "id_06"]
        src = _mimii_side(ids, "6_dB", "source", rng)
        tgt = _mimii_side(ids, tsnr, "target", rng)
        meta = {"task": "cls", "in_ch": 64, "n_classes": 2, "classes": ["Normal", "Abnormal"], "src": "MIMII fan +6 dB", "tgt": f"MIMII fan {tsnr}"}
        return _assemble(src, tgt, meta, rng, CAPS_MIMII)
    if name == "mimii_id00_id02":
        src = _mimii_side(["id_00", "id_04", "id_06"], "6_dB", "source", rng)
        tgt = _mimii_side(["id_02"], "6_dB", "target", rng)
        meta = {"task": "cls", "in_ch": 64, "n_classes": 2, "classes": ["Normal", "Abnormal"], "src": "MIMII fans 00/04/06 (+6 dB)", "tgt": "MIMII fan 02 (+6 dB)"}
        return _assemble(src, tgt, meta, rng, CAPS_MIMII)
    if name == "cmapss_FD001_FD003":
        return cmapss_scenario(1, 3, seed)
    raise KeyError(name)


SCENARIOS = ["cwru_L0_L1", "cwru_L0_L2", "cwru_L0_L3", "cwru_DE_FE", "cwru_to_pad", "pad_to_cwru", "pad_art_real",
             "mimii_6_0", "mimii_6_m6", "mimii_id00_id02", "cmapss_FD001_FD003"]


def load(name, seed=SEED):
    """Cached scenario (partition seed is fixed; training seeds vary only the model)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: (z[k].item() if k == "meta" else z[k]) for k in z.files}
    d = build(name, seed)
    np.savez(path, **{k: (np.array(v, dtype=object) if k == "meta" else v) for k, v in d.items()})
    return d


def summary(d):
    s = {k: tuple(v.shape) for k, v in d.items() if isinstance(v, np.ndarray)}
    for k in ("y_tr", "y_va", "y_cal", "y_unl", "y_lab", "y_te"):
        if k in d and d["meta"]["task"] == "cls":
            s[k + "_counts"] = np.bincount(d[k], minlength=d["meta"]["n_classes"]).tolist()
    return s


if __name__ == "__main__":
    import json
    import sys
    names = sys.argv[1:] or SCENARIOS
    for n in names:
        d = load(n)
        print(n, json.dumps(summary(d), default=str))
