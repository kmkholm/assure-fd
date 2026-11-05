# ASSURE-FD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23975679.svg)](https://doi.org/10.5281/zenodo.23975679)

Code and partition manifests for the paper

**ASSURE-FD: Certified false-alarm rates for cross-machine bearing fault diagnosis through test-time adaptation and conformal risk control**
(Mohammed Tawfik, Islam S. Fathi; under review).

ASSURE-FD certifies the false-alarm rate of a fault-diagnosis model on a newly commissioned machine from its healthy
run-in data alone. A compact 1-D convolutional encoder is trained on a source rig; on the new machine, source-free
test-time adaptation is anchored on one half of the run-in windows, and the other half calibrates (i) a conformal
alarm threshold with a finite-sample false-alarm bound and (ii) the healthy class of a class-conditional prediction
set. The novelty score and the set give an *alarm*, *inspect* or *continue* decision; with labelled target windows,
conformal risk control also bounds the missed-fault rate.

## Contents

| Path | Content |
|---|---|
| `code/data.py` | the eleven shift scenarios (CWRU, Paderborn, MIMII fan, C-MAPSS), recording-level splits, caches |
| `code/conformal.py` | split, weighted and class-conditional conformal prediction, conformal risk control, conformalised quantile regression, decision rule and cost (pure NumPy; `python conformal.py` runs a self-test) |
| `code/model.py`, `code/tta.py`, `code/baselines.py` | encoder and heads, test-time adaptation with the healthy anchor, DANN and Deep CORAL |
| `code/run_grid.py` | trains and evaluates one (scenario, seed) run; all variants are evaluated post hoc; `--eval-only` re-evaluates saved models |
| `code/worker.py` | runs the grid with several workers that share it through lock files |
| `code/make_assets.py`, `code/xai.py` | tables, LaTeX macros and figures; integrated-gradient explanation |
| `code/manifest.py` | writes the partition manifests |
| `manifests/<scenario>.json` | recording-level assignment of every Paderborn file, MIMII clip and C-MAPSS engine to its split; CWRU segment rules |

Result files, trained models and figures are not included in this repository; they are reproduced by the commands
below.

## Data

All four datasets are public:

* CWRU bearing data — Case Western Reserve University Bearing Data Center (12 kHz drive-end and normal-baseline files).
* Paderborn University bearing data (KAt data center) — vibration channel, 64 kHz resampled to 12 kHz.
  The file `N15_M01_F10_KA08_2.mat` of the public release cannot be parsed and is skipped.
* MIMII — fan recordings at +6, 0 and −6 dB (Zenodo record 3384388).
* NASA C-MAPSS turbofan data (FD001, FD003).

Expected layout under `ASSURE_DATA`:

```
CWRU/normal_baseline/{97..100}.mat   CWRU/12k_drive_end/<id>.mat
Paderborn/<bearing>/<setting>_<bearing>_<n>.mat
C-MAPSS/CMAPSSData/{train,test,RUL}_FD00{1,3}.txt
```

MIMII may live elsewhere (`ASSURE_MIMII=<path to MIMII>` with `{6_dB,0_dB,-6_dB}/fan/id_xx/{normal,abnormal}/*.wav`).

## Reproducing the results

```
pip install -r requirements.txt
cd code
export ASSURE_FULL=1 ASSURE_DATA=/path/to/data ASSURE_MIMII=/path/to/MIMII
python data.py                       # builds the scenario caches and prints the split sizes
python run_grid.py --scenario cwru_L0_L2 --seeds 1 --epochs 3 --force   # smoke test (writes ../results)
python worker.py 1 & python worker.py 2 reverse & python worker.py 3 &  # full grid, 55 runs (~5 h on one 8 GB GPU)
python make_assets.py && python xai.py cwru_L0_L2                       # tables, macros and figures
```

Seeds: the partition seed is 2026; training seeds are 0–4 (evidential head +500, DANN/CORAL +700).

## License

MIT (see `LICENSE`). The datasets keep their own licences.
