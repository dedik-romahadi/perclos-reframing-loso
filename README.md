# perclos-reframing-loso

Analysis code for the manuscript

> **Does One Predictive PERCLOS Distribution Serve Every Threshold? A Pre-registered Leave-One-Person-Out Test of
> Train-Once Vigilance Estimation from EOG Features**
> Wiwit Suprihatiningsih, Dedik Romahadi (Universitas Mercu Buana). Submitted to *Pattern Analysis and Applications*.

The study asks whether one predictive distribution of PERCLOS, trained once on EOG features, classifies unseen drivers
as well as a classifier retrained at every cut-point (reframing versus retraining), for three-class vigilance
estimation and for the binary alert-versus-drowsy decision, under person-level leave-one-person-out evaluation on
SEED-VIG.

This repository contains **only the computation**: feature extraction, model fitting, prediction, statistics and the
descriptive analyses. Code that builds the manuscript, tables, figures and reports is not included. Every file is the
exact version used for the reported results; `MANIFEST.sha256` lists their SHA-256 hashes.

## Layout

```
00_COMMON/                         shared infrastructure (loading, features, networks, metrics)
  seedvig_eog.py                   SEED-VIG loader, leave-one-person-out splits, bootstrap CI, sign-flip test, Holm
  build_cache.py                   per-recording EOG cache from the raw .mat files
  eog_features_lit.py              causal and offline literature EOG features (blinks, saccades, fixations, spectra)
  build_feature_caches.py          builds the feature caches (litc-*, lit-*, pub2orig-*, pub2fix-*)
  pub2_features.py, pub2_features_fixed.py, build_pub2_features.py
                                   features of the authors' earlier, unpublished pipeline ("Publication 2", called the
                                   P2 pipeline in the manuscript) and its causal port (pub2-*)
  feature_sets.py                  one loader for every feature set, with modifiers (+kf Kalman, +noblinkall, ...)
  deep_data.py, deep_models.py     raw-EOG input pipeline and the network registry (feat_gru, sinc_hybrid, resnet1d_gru)
  deep_train.py                    shared leave-one-person-out trainer (copied and extended in deep_train_t1.py)
  pub2_loso.py, summarize.py       classifier presets of the P2 pipeline; per-person metrics
T1_ThresholdConsistentPerclos/code/
  t1_common.py                     paths, cut-point grids, balanced Bayes rule, CDF utilities, per-cell metrics
  tc_models.py, fast_lda.py        train-once models (ridge logit-normal, quantile HGB), ordinal logit, LDA variants
  run_tc.py, run_grid_direct.py    M1 (and input variants) / LDA retrained at every cut-point (M2 and variants)
  run_refs.py, run_calib.py, run_hgb.py, run_controls.py
                                   reference models, calibration comparators, HGB models, time-only and permutation controls
  deep_train_t1.py, t1_nets.py, run_neural.py, gpu_queue.py
                                   neural train-once and direct models (M7-M14) and the GPU queue with its timing rule
  run_pub2_nested.py, run_rank.py  nested rebuild of the P2 pipeline (M15); rank-stability models
  ceiling.py, run_a1.py            label-jitter analysis; isotonic recalibration of M1
  stats_t1.py                      all pre-registered statistics
  descriptive_tc_vs_m2.py          post-hoc descriptive comparison of non-linear train-once models with M2
  run_alpha_wide.py, run_a23.py, run_a5.py, run_raw_seeds.py, stats_ext.py
                                   secondary analyses after registration (wider ridge grid, ablations A2/A3/A5, extra seeds)
  mechanism_t1.py                  descriptive analysis of the failure mechanism (critical spread sigma*)
  run_all.py                       driver of the registered pipeline (resumable)
  tests/test_t1.py                 unit tests
```

## Data

SEED-VIG (Zheng and Lu, J Neural Eng 14:026017, 2017) is distributed by its curators on request:
https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html. The code expects the raw files next to this repository:

```
<parent folder>/
  perclos-reframing-loso/          this repository
  DATASET/SADT-SEEDVIG/SEEDVIG/    <recording>.mat, perclos_labels/<recording>.mat, LABELS/<recording>.mat
```

(`00_COMMON/seedvig_eog.py`, `RAW_DIR`). Caches are written to `00_COMMON/cache/`, results to
`T1_ThresholdConsistentPerclos/results/`; both are ignored by git.

## Environment

Python 3.14 with NumPy 2.4, SciPy 1.17, scikit-learn 1.9, PyTorch 2.11 (CUDA 12.8), braindecode 1.5, PyWavelets and
joblib (`requirements.txt`). Run every script with `python -I` from the folder that contains it; the scripts add the
paths they need. Neural networks were trained on one GPU with a memory cap of 1.5 GB per process
(`torch.cuda.set_per_process_memory_fraction`); the CPU models use two threads per process
(`OMP_NUM_THREADS=2`).

## Running

```bash
# 1. caches (once; resumable)
cd 00_COMMON
python -I build_cache.py
python -I build_feature_caches.py --kinds pub2orig litc_trad litc_fore
python -I build_pub2_features.py
python -I deep_data.py --build

# 2. registered pipeline (resumable; every job writes per-fold files and skips finished folds)
cd ../T1_ThresholdConsistentPerclos/code
python -I tests/test_t1.py
python -I run_all.py              # see the docstring for the order of jobs; --skip-gpu, --skip-tier3
python -I stats_t1.py             # statistics only, from saved predictions

# 3. secondary analyses after registration
python -I run_alpha_wide.py
python -I run_a23.py a2 && python -I run_a23.py a3
python -I run_a5.py --order asc   # a second worker may run with --order desc
python -I run_raw_seeds.py        # GPU, resumable
python -I stats_ext.py
python -I mechanism_t1.py
```

`run_all.py` ends by calling the report builders (`make_leaderboard.py`, `results_md.py`, `figures_t1.py`), which are
not part of this repository; the computation is complete before those calls, and `stats_t1.py` writes all test
results (`results/hypotheses.csv`).

## Model identifiers

| Manuscript | Code identifier |
|---|---|
| M1 linear train-once (primary) | `tc-linln_litc-trad+kf` |
| M2 LDA retrained per cell (primary comparator); R4 at 0.35/0.70 | `direct-lda-grid_litc-trad+kf`, `ref-litc-trad+kf_lda` |
| M5 / M6 quantile HGB (TC) / HGB (direct) | `tc-qhgb_litc-trad+kf` / `direct-hgb_litc-trad+kf` |
| M7 / M8 per-epoch MLP | `tc-mlpln_litc-trad+kf` / `direct-mlp_litc-trad+kf` |
| M9 / M10 feature GRU | `tc-featgru_litc-trad` / `direct-featgru_litc-trad` |
| M11 / M12 sinc hybrid end-to-end | `tc-sinchybrid_trad6` / `direct-sinchybrid_trad6` |
| M13 / M14 ResNet1D-GRU | `tc-resnet1dgru_trad6` / `direct-resnet1dgru_trad6` |
| R1, R2, M15 (P2 pipeline) | `ref-pub2orig-all_lda`, `ref-pub2orig-type2_qsvm`, `pub2nested_pub2orig` |

## Integrity

The analysis plan was locked with a SHA-256 hash before any outer-fold prediction; deviations are reported in Online
Resource 1 of the manuscript. The locked plan and per-person predictions are available from the corresponding author
(dedik.romahadi@mercubuana.ac.id) on reasonable request. Some comments in the code refer to sections of that plan
("PREREG §…").

## Licence

MIT (see `LICENSE`). SEED-VIG itself is subject to the licence of its curators.
