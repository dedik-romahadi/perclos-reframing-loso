"""Direct shrinkage-LDA grid (M2 / M17 / M19 / M21 / M23; H7b feature-type LDA grids).

    python -I run_grid_direct.py <featset> [--tasks three,binarypair,single] [--max-minutes 9]

In every outer LOSO fold: median imputer + StandardScaler fitted ONCE on all epochs of the 20 training
persons (label-free); then shrinkage LDA (lsqr, Ledoit-Wolf, uniform priors) retrained at every
three-class cell (49), every VALID binary-pair cell (47; alert < t1 vs drowsy >= t2, middle dropped
from training) and every single threshold (11). Decision = predict (uniform priors = balanced).
Test predictions are stored for all test epochs (binary-pair: dropped rows set to -1 at merge).
Engine: fast_lda.GridLDA (exact moment-based re-implementation of sklearn LDA lsqr/auto; validated against
sklearn on real folds, |proba diff| < 1e-11, identical predictions); --engine sklearn uses sklearn itself.
Resumable: results/folds/direct-lda-grid_<featset>/fold_<person>.npz.
Outputs: results/grid/<task>_direct-lda-grid_<featset>.npz and, when all three tasks were run,
results/preds/{three,binary}_direct-lda-grid_<featset>.npz (canonical cell).
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse
import time

import numpy as np

import t1_common as C
import fast_lda as FL
import tc_models as TM
import seedvig_eog as S


ENGINE = "fast"     # 'fast' = fast_lda (moment-based, = sklearn up to ~1e-11) | 'sklearn'


def run_fold(X, y, groups, person, tasks, valid2):
    tr, te = np.flatnonzero(groups != person), np.flatnonzero(groups == person)
    S.assert_no_person_overlap(groups[tr], groups[te])
    prep = TM.make_prep().fit(X[tr])
    Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
    moments = FL.BinMoments(Xtr, y[tr]) if ENGINE == "fast" else None
    out = {"rows": te}
    for task in tasks:
        cells = C.task_cells(task)
        nc = C.n_classes(task)
        pred = np.full((len(te), len(cells)), -1, np.int8)
        score = np.zeros((len(te), len(cells), nc), np.float32)
        for j, c in enumerate(cells):
            if task == "binarypair" and not valid2[j]:
                continue
            lab = C.cell_labels(y[tr], task, c)
            m = lab >= 0
            k = len(np.unique(lab[m]))
            if ENGINE == "fast":
                assert k == (3 if task == "three" else 2)
                lda = FL.GridLDA(moments, FL.cell_classes(task, c))
            else:
                lda = TM.make_lda(k).fit(Xtr[m], lab[m])
            pred[:, j] = lda.predict(Xte)
            score[:, j, :k] = lda.predict_proba(Xte)
        out[f"pred_{task}"], out[f"score_{task}"] = pred, score
        print(f"  person {person} {task} done", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("featset")
    ap.add_argument("--tasks", default="three,binarypair,single")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--engine", default="fast", choices=["fast", "sklearn"])
    a = ap.parse_args(argv)
    global ENGINE
    ENGINE = a.engine
    tasks = a.tasks.split(",")
    model_id = a.model_id or f"direct-lda-grid_{a.featset}"
    job = model_id + ("" if tasks == ["three", "binarypair", "single"] else "__" + "-".join(tasks))
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    d = C.load_features(a.featset)
    b = C.base()
    X, y, groups = d["X"], b["perclos"], b["groups"]
    valid2 = C.valid_binary_cells()
    for p in b["persons"]:
        if C.fold_done(job, p):
            continue
        if time.time() > deadline - 60:
            print(f"[{job}] time limit; resume later", flush=True)
            return "paused"
        tf = time.time()
        C.save_fold(job, int(p), **run_fold(X, y, groups, int(p), tasks, valid2))
        print(f"[{job}] fold {p} {time.time() - tf:.1f} s", flush=True)
    n = len(y)
    grids = {}
    for task in tasks:
        cells = C.task_cells(task)
        nc = C.n_classes(task)
        pred = np.full((n, len(cells)), -1, np.int8)
        score = np.zeros((n, len(cells), nc), np.float32)
        for p in b["persons"]:
            z = C.load_fold(job, int(p))
            pred[z["rows"]] = z[f"pred_{task}"]
            score[z["rows"]] = z[f"score_{task}"]
        if task == "binarypair":
            for j, c in enumerate(cells):
                pred[C.lab2(y, *c) < 0, j] = -1
        grids[task] = (pred, score)
    C.save_grid(model_id, grids)
    if "three" in tasks:
        C.save_preds("three", model_id, grids["three"][0][:, C.CANON_IDX], grids["three"][1][:, C.CANON_IDX])
    if "binarypair" in tasks:
        C.save_preds("binary", model_id, grids["binarypair"][0][:, C.CANON_IDX],
                     grids["binarypair"][1][:, C.CANON_IDX])
    C.log_compute(job, time.time() - t0)
    print(f"[{job}] complete", flush=True)
    return "complete"


if __name__ == "__main__":
    main()
