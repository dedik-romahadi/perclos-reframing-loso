"""M5 TC-QHGB (19 quantile HistGB, rearranged) and M6 direct balanced HGB on litc-trad+kf.

    python -I run_hgb.py qhgb [--max-minutes 9]
    python -I run_hgb.py hgb  [--max-minutes 9]

Folds: results/folds/tc-qhgb_litc-trad+kf/ (test quantiles) and results/folds/direct-hgb_<task>/.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import tc_models as TM  # noqa: E402
import seedvig_eog as S  # noqa: E402

FEAT = "litc-trad+kf"


def run_qhgb(X, b, deadline):
    job = f"tc-qhgb_{FEAT}"
    y, groups = b["perclos"], b["groups"]
    for p in b["persons"]:
        p = int(p)
        if C.fold_done(job, p):
            continue
        if time.time() > deadline - 240:
            return "paused"
        tf = time.time()
        tr, te = np.flatnonzero(groups != p), np.flatnonzero(groups == p)
        S.assert_no_person_overlap(groups[tr], groups[te])
        m = TM.QHGB().fit(X[tr], y[tr])
        C.save_fold(job, p, rows=te, Q=m.predict_quantiles(X[te]))
        print(f"[{job}] fold {p} {time.time() - tf:.1f} s", flush=True)
    Q = np.zeros((len(y), len(TM.QS)))
    for p in b["persons"]:
        z = C.load_fold(job, int(p))
        Q[z["rows"]] = z["Q"]
    Fcut, G, mean, median, q, pit, K = TM.qhgb_outputs(Q, y)
    C.finalize_tc(job, Fcut, G, mean, median, q, params={"Q": Q}, pit=pit)
    return "complete"


def run_hgb(X, b, deadline):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    groups = b["groups"]
    for task in ("three", "binary"):
        job = f"direct-hgb_{task}"
        keep = np.ones(len(groups), bool) if task == "three" else b["yi"] != 2
        y = (b["yi"] - 1).astype(int) if task == "three" else (b["yi"] == 3).astype(int)
        for p in b["persons"]:
            p = int(p)
            if C.fold_done(job, p):
                continue
            if time.time() > deadline - 180:
                return "paused"
            tf = time.time()
            tr = np.flatnonzero((groups != p) & keep)
            te = np.flatnonzero(groups == p)
            S.assert_no_person_overlap(groups[tr], groups[te])
            imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                               early_stopping=False, class_weight="balanced",
                                               random_state=0).fit(imp.transform(X[tr]), y[tr])
            P = m.predict_proba(imp.transform(X[te]))
            C.save_fold(job, p, rows=te, pred=P.argmax(1), score=P)
            print(f"[{job}] fold {p} {time.time() - tf:.1f} s", flush=True)
        nc = 3 if task == "three" else 2
        pred, score = np.full(len(groups), -1), np.zeros((len(groups), nc))
        for p in b["persons"]:
            z = C.load_fold(job, int(p))
            pred[z["rows"]], score[z["rows"]] = z["pred"], z["score"]
        C.save_preds(task, f"direct-hgb_{FEAT}", pred, score)
    return "complete"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["qhgb", "hgb"])
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    X = C.load_features(FEAT)["X"]
    b = C.base()
    st = run_qhgb(X, b, deadline) if a.what == "qhgb" else run_hgb(X, b, deadline)
    if st == "complete":
        C.log_compute(a.what, time.time() - t0)
    print(f"[{a.what}] {st}", flush=True)
    return st


if __name__ == "__main__":
    main()
