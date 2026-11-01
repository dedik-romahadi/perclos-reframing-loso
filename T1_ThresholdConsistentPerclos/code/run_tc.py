"""TC-LinLN (M1 primary; M16 time; M18/M20/M22 robustness inputs) under outer LOSO.

    python -I run_tc.py <featset> [--max-minutes 9]

Per outer fold (20 training persons): LinLN.fit (alpha by inner 5-fold person-grouped CRPS, sigma from the
inner OOF residuals), refit on all training persons, mu for the test person. One logit-normal predictive
distribution per epoch; every class decision is derived from it (t1_common.tc_grid, balanced Bayes rule).
Resumable: results/folds/tc-linln_<featset>/fold_<person>.npz.
Outputs: results/cdf/tc-linln_<featset>.npz, results/grid/*_tc-linln_<featset>.npz,
results/preds/{three,binary}_tc-linln_<featset>.npz, results/fold_params_tc-linln_<featset>.csv.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse
import time

import numpy as np

import t1_common as C
import tc_models as TM
import seedvig_eog as S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("featset")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    model_id = f"tc-linln_{a.featset}"
    job = model_id
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    d = C.load_features(a.featset)
    b = C.base()
    X, y, groups = d["X"], b["perclos"], b["groups"]
    for p in b["persons"]:
        p = int(p)
        if C.fold_done(job, p):
            continue
        if time.time() > deadline - 60:
            print(f"[{job}] time limit; resume later", flush=True)
            return "paused"
        tf = time.time()
        tr, te = np.flatnonzero(groups != p), np.flatnonzero(groups == p)
        S.assert_no_person_overlap(groups[tr], groups[te])
        m = TM.LinLN().fit(X[tr], y[tr], groups[tr])
        C.save_fold(job, p, rows=te, mu=m.predict_mu(X[te]), sigma=m.sigma, alpha=m.alpha,
                    inner_crps=m.inner_crps, inner_sigma=m.inner_sigma)
        print(f"[{job}] fold {p}: alpha {m.alpha:g} sigma {m.sigma:.4f} ({time.time() - tf:.1f} s)", flush=True)
    n = len(y)
    mu, sig = np.zeros(n), np.zeros(n)
    rows = []
    for p in b["persons"]:
        z = C.load_fold(job, int(p))
        mu[z["rows"]] = z["mu"]
        sig[z["rows"]] = float(z["sigma"])
        rows.append({"model_id": model_id, "person": int(p), "alpha": float(z["alpha"]), "sigma": float(z["sigma"]),
                     **{f"inner_crps_a{al:g}": float(v) for al, v in zip(TM.ALPHAS, z["inner_crps"])}})
    C.write_csv(C.RES / f"fold_params_{model_id}.csv", rows)
    Fcut, G, mean, median, q, pit, ls = TM.ln_outputs(mu, sig, y)
    C.finalize_tc(model_id, Fcut, G, mean, median, q, params={"mu": mu, "sigma": sig}, pit=pit, logscore=ls)
    C.log_compute(job, time.time() - t0)
    print(f"[{job}] complete", flush=True)
    return "complete"


if __name__ == "__main__":
    main()
