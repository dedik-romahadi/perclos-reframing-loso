"""Calibration comparators M3 (LDA-TS) and M4 (OrdLogit) on litc-trad+kf, canonical tasks.

    python -I run_calib.py [--max-minutes 9]

M3 lda-ts: shrinkage LDA with empirical priors + temperature (inner 5-fold OOF NLL); probabilities only
(decision for reference = argmax of the tempered empirical-prior posterior, not used for claims).
M4 ordlogit: cumulative logit (three-class) / L2 logistic (binary), lambda by inner 5-fold OOF NLL;
natural probabilities; decision = balanced Bayes rule with outer-training prevalences.
Binary: trained and scored on alert/drowsy epochs only (canonical task).
Folds: results/folds/{lda-ts,ordlogit}_<task>/.
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--models", default="lda-ts,ordlogit")
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    d = C.load_features(FEAT)
    b = C.base()
    X, groups = d["X"], b["groups"]
    rows_info = []
    for model in a.models.split(","):
        for task in ("three", "binary"):
            job = f"{model}_{task}"
            keep = np.ones(len(groups), bool) if task == "three" else b["yi"] != 2
            y = (b["yi"] - 1).astype(int) if task == "three" else (b["yi"] == 3).astype(int)
            nc = 3 if task == "three" else 2
            for p in b["persons"]:
                p = int(p)
                if C.fold_done(job, p):
                    continue
                if time.time() > deadline - 120:
                    print("[calib] time limit; resume later", flush=True)
                    return "paused"
                tf = time.time()
                tr = np.flatnonzero((groups != p) & keep)
                te = np.flatnonzero(groups == p)
                S.assert_no_person_overlap(groups[tr], groups[te])
                prior = np.bincount(y[tr], minlength=nc) / len(tr)
                if model == "lda-ts":
                    m = TM.LDATS().fit(X[tr], y[tr], groups[tr])
                    P = m.predict_proba(X[te])
                    pred = P.argmax(1)
                    par = {"T": m.T}
                else:
                    m = TM.OrdLogit(K=nc - 1).fit(X[tr], y[tr], groups[tr])
                    P = m.predict_proba(X[te])
                    pred = C.bayes_balanced(P, prior)
                    par = {"lam": m.lam, "converged": bool(m.res.success)}
                C.save_fold(job, p, rows=te, pred=pred, score=P, prior=prior, **par)
                print(f"[{job}] fold {p} {par} {time.time() - tf:.1f} s", flush=True)
            n = len(groups)
            pred = np.full(n, -1)
            score = np.zeros((n, nc))
            for p in b["persons"]:
                z = C.load_fold(job, int(p))
                pred[z["rows"]], score[z["rows"]] = z["pred"], z["score"]
                rows_info.append({"model": model, "task": task, "person": int(p),
                                  **{k: float(z[k]) for k in ("T", "lam") if k in z}})
            C.save_preds(task, f"{model}_{FEAT}", pred, score)
    C.write_csv(C.RES / "fold_params_calib.csv", rows_info)
    C.log_compute("calib", time.time() - t0)
    print("[calib] complete", flush=True)
    return "complete"


if __name__ == "__main__":
    main()
