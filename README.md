# perclos-reframing-loso

Analysis code for the manuscript

> **Does One Predictive PERCLOS Distribution Serve Every Threshold? A Pre-registered Leave-One-Person-Out Test of
> Train-Once Vigilance Estimation from EOG Features**
> Wiwit Suprihatiningsih, Dedik Romahadi (Universitas Mercu Buana). Submitted to *Pattern Analysis and Applications*.

The study asks whether one predictive distribution of PERCLOS, trained once on EOG features, classifies unseen drivers
as well as a classifier retrained at every cut-point (reframing versus retraining), for three-class vigilance
estimation and for the binary alert-versus-drowsy decision, under leave-one-person-out evaluation on SEED-VIG.

This repository contains **only the computation** (features, models, predictions, statistics). Code that builds the
manuscript, tables or figures is not included. `MANIFEST.sha256` lists the SHA-256 hash of every file; all files are
the versions used for the reported results, except two small portability edits listed at the top of the manifest.

---

## Quick start (about 15 minutes on a normal PC, no GPU needed)

These steps reproduce the **primary result** of the paper: the train-once linear model (M1) against linear discriminant
analysis retrained at every cut-point (M2).

### Step 1. Install Python and the packages

Install Python 3.12 or newer (3.14.0 was used) from https://www.python.org. Then open a terminal (Windows: *Command
Prompt* or *PowerShell*) in this folder and run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

PyTorch is needed only for the neural networks; if `pip` cannot install the listed version on your system, install any
recent version from https://pytorch.org.

### Step 2. Get the SEED-VIG data

SEED-VIG is distributed by its curators on request: https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html. You need the
folder that contains the 23 recording files (`*.mat`) and the two sub-folders `perclos_labels/` and `LABELS/`. Tell the
code where that folder is:

```bash
# Windows (Command Prompt)
set SEEDVIG_RAW_DIR=D:\data\SEED-VIG\SEEDVIG
# Windows (PowerShell)
$env:SEEDVIG_RAW_DIR = "D:\data\SEED-VIG\SEEDVIG"
# macOS / Linux
export SEEDVIG_RAW_DIR=/home/me/data/SEED-VIG/SEEDVIG
```

(Set it again in every new terminal, or set it permanently in your system settings.)

### Step 3. Check the setup

```bash
python check_setup.py
```

It checks Python, the packages, the 23 data files and the GPU, and tells you exactly what is missing. Continue when it
prints `Ready.`

### Step 4. Build the data caches (once)

```bash
cd 00_COMMON
python -I build_cache.py
python -I build_feature_caches.py --kinds litc_trad litc_fore --jobs 4
cd ..
```

`build_cache.py` reads the raw files and checks them; `build_feature_caches.py` computes the causal EOG features (209 from the electrodes
around the eyes, 259 from the forehead electrodes).
Both skip files that already exist, so they can be restarted after an interruption.

### Step 5. Reproduce the primary result

```bash
cd T1_ThresholdConsistentPerclos/code
python -I tests/test_t1.py --fast
python -I run_grid_direct.py litc-trad+kf
python -I run_tc.py litc-trad+kf
python -I stats_t1.py
```

All unit tests must pass except `test_canonical_equals_summarize`, which needs the benchmark files of Step 7 and reports
a missing file until then. Open `T1_ThresholdConsistentPerclos/results/hypotheses.csv`. The rows **H1** and **H2** must read:

| Row | Task | estimate | 95 % CI | Holm p | Decision |
|---|---|---|---|---|---|
| H1 | three-class (49 cells) | −0.0996 | −0.1334 to −0.0654 | 0.001 | inferior |
| H2 | binary pair (47 cells) | −0.0373 | −0.0724 to −0.0051 | 0.349 | inconclusive |

(estimates are differences in balanced accuracy, M1 − M2). Statistics of models that have not been run yet are
skipped (the output ends with `stats done; 2 hypothesis rows; 41 missing`). On a test machine this step reproduced the
published predictions exactly (identical files; distribution parameters within 10⁻¹¹). Timing there: caches 4 min,
M2 0.5 min, M1 1 min, statistics 10 s.

---

## Full reproduction

### Step 6. All caches

```bash
cd 00_COMMON
python -I build_feature_caches.py --kinds pub2orig litc_trad litc_fore --jobs 4
python -I build_pub2_features.py
python -I deep_data.py --build            # raw-signal cache for the networks
cd ..
```

### Step 7. Shared benchmark rows (used only for an identity check)

```bash
cd 00_COMMON
python -I feature_check.py run --clfs lda --sets litc-trad+kf pub2orig-all pub2-type1+kf time
cd ..
```

### Step 8. The registered pipeline

```bash
cd T1_ThresholdConsistentPerclos
mkdir results                                              # skip if the folder already exists
copy config\gpu_plan_used.json results\gpu_plan.json      # Windows
# cp config/gpu_plan_used.json results/gpu_plan.json      # macOS / Linux
cd code
python -I run_all.py
```

Copying `gpu_plan.json` fixes the training setting of the raw-EOG networks to the one used in the paper (every fourth
epoch as training target; see Online Resource 1, D4). Without it the queue decides from the speed of your GPU.

`run_all.py` runs every job in the registered order. All jobs save one file per fold and skip finished folds, so you
can stop and restart at any time. Useful options: `--skip-gpu` (classical models only), `--skip-tier3` (no
ablations). With one consumer GPU the neural networks take several hours; the classical models a few hours on a
4-core CPU. The last three steps of `run_all.py` build reports that are not part of this repository and are skipped.

### Step 9. Secondary analyses (after registration)

```bash
python -I run_alpha_wide.py               # wider ridge grid for M1
python -I run_a23.py a2                   # ablation A2 (Beta head)
python -I run_a23.py a3                   # ablation A3 (auxiliary cross-entropy)
python -I run_a5.py                       # ablation A5 (isotonic recalibration of M5)
python -I run_raw_seeds.py                # seeds 1 and 2 for the raw-EOG networks (GPU)
python -I stats_ext.py                    # families FX1, FX2
python -I mechanism_t1.py                 # descriptive failure mechanism (Table 3 of the paper)
python -I descriptive_tc_vs_m2.py         # post-hoc descriptive comparison
```

### Where the results are

`T1_ThresholdConsistentPerclos/results/`: `hypotheses.csv` (every test), `integrated.csv`, `cell_delta.csv`,
`calibration.csv`, `coherence.csv`, `rank_stability.csv`, `preds/` (per-epoch predictions), `ext/` (secondary analyses).

---

## Notes

* Always run scripts with `python -I` from the folder that contains them; the scripts add the paths they need.
* The CPU models were run with two threads per process (`OMP_NUM_THREADS=2`); the networks with a GPU memory cap of
  1.5 GB per process. Results do not depend on the number of threads except for floating-point rounding.
* "Publication 2", `pub2orig-*` and `pub2-*` refer to the authors' earlier feature pipeline, which was never published;
  the manuscript calls it the P2 pipeline and describes it in full.
* Comments such as "PREREG §4.3" point to sections of the locked analysis plan, which is available from the
  corresponding author on request together with per-person predictions.

## Model identifiers

| Manuscript | Code identifier |
|---|---|
| M1 linear train-once (primary) | `tc-linln_litc-trad+kf` |
| M2 LDA retrained per cell (primary comparator); R4 at 0.35/0.70 | `direct-lda-grid_litc-trad+kf`, `ref-litc-trad+kf_lda` |
| M5 / M6 quantile HGB (train-once) / HGB (direct) | `tc-qhgb_litc-trad+kf` / `direct-hgb_litc-trad+kf` |
| M7 / M8 per-epoch MLP | `tc-mlpln_litc-trad+kf` / `direct-mlp_litc-trad+kf` |
| M9 / M10 feature GRU | `tc-featgru_litc-trad` / `direct-featgru_litc-trad` |
| M11 / M12 sinc hybrid end-to-end | `tc-sinchybrid_trad6` / `direct-sinchybrid_trad6` |
| M13 / M14 ResNet1D-GRU | `tc-resnet1dgru_trad6` / `direct-resnet1dgru_trad6` |
| R1, R2, M15 (P2 pipeline) | `ref-pub2orig-all_lda`, `ref-pub2orig-type2_qsvm`, `pub2nested_pub2orig` |

## Files

```
check_setup.py                     step 3
00_COMMON/                         data loading, features, networks, metrics (shared infrastructure)
T1_ThresholdConsistentPerclos/
  config/gpu_plan_used.json        GPU setting used in the paper (step 8)
  code/                            models, pipeline driver, statistics, secondary analyses, unit tests
```

## Contact and licence

Questions: Dedik Romahadi, dedik.romahadi@mercubuana.ac.id. Code licence: MIT (`LICENSE`). SEED-VIG is subject to the
terms of its curators.
