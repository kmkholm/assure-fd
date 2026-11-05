"""Write the recording-level partition manifest of every scenario to ../manifests/<scenario>.json.

The scenario builders are re-run with the same seeds, but the signal readers are replaced by stubs, so no vibration or
audio file is read: the random permutations that assign recordings, clips and engines to splits do not depend on the
signals, and the records are therefore identical to those of the experiments. CWRU recordings are split into
contiguous segments of each recording, which are listed with their fractions.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASSURE_FULL", "1")
import data as D  # noqa: E402

OUT = os.path.join(HERE, "..", "manifests")
os.makedirs(OUT, exist_ok=True)

D._paderborn_vib = lambda path: np.zeros(48000, np.float32)


def _mimii_stub(mid, snr):
    base = os.path.join(D.MIMII_ROOT, snr, "fan", mid)
    nn = len(glob.glob(os.path.join(base, "normal", "*.wav")))
    na = len(glob.glob(os.path.join(base, "abnormal", "*.wav")))
    return np.zeros((nn, 1, 4), np.float32), np.zeros((na, 1, 4), np.float32)


D._mimii_feats = _mimii_stub

CWRU_NOTE = {"dataset": "CWRU", "unit": "one recording per class, fault size and load (12 kHz drive-end and normal-baseline files)",
             "files": {str(c): D.CWRU_FILES[c] for c in D.CWRU_FILES},
             "source_segments": "first 70% train (last window dropped), 70-85% validation (last window dropped), 85-100% calibration",
             "target_segments": "first 45% pool (run-in, unlabelled and labelled windows), 45-50% guard gap, 50-100% test",
             "windows": "1024 samples, hop 512 on train/pool segments, hop 1024 on validation, calibration and test segments"}

for name in D.SCENARIOS:
    D.MANIFEST.clear()
    d = D.build(name)
    rec = {"scenario": name, "meta": d["meta"], "sizes": D.summary(d), "recordings": list(D.MANIFEST),
           "pool_assignment": "pool windows are assigned to run-in (150 healthy), unlabelled and labelled sets by "
                              "numpy.random.default_rng(2026) in data._assemble; the run-in set is split 75/75 per seed in run_grid.eval_cls"}
    if name.startswith("cwru") or name in ("cwru_to_pad", "pad_to_cwru"):
        rec["cwru"] = CWRU_NOTE
    if D.SKIPPED:
        rec["skipped_unreadable_files"] = sorted(set(D.SKIPPED))
    json.dump(rec, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=str)
    print(name, "records:", len(D.MANIFEST))
