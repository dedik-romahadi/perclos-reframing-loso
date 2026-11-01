"""A1 ablation / robustness R4 of H1-H2 (PREREG 4.6, 11): isotonic PIT recalibration of M1 (Kuleshov et al. 2018).

    python -I run_a1.py

Per outer fold (20 training persons only): the inner 5-fold person-grouped OOF predictions of TC-LinLN at the
fold's chosen alpha (same inner folds, same imputer/scaler protocol as tc_models.LinLN; the alpha and sigma are
read from the saved M1 fold file) give inner OOF PIT values u_i = Phi((logit(y_i) - mu_oof_i) / sigma). An
isotonic map R (u -> empirical CDF of the inner PIT values; sklearn IsotonicRegression, clipped to [0, 1]) is
fitted on them and applied to the test person's predictive CDF: F'(t) = R(F(t)) at the 14 cut-points and on the
512-point grid. All class decisions are then derived from F' exactly as for M1 (balanced Bayes rule).
The recalibrated CDF is a step-like monotone map, so no log score is stored.
Folds: results/folds/tc-linln-iso_litc-trad+kf/; model id tc-linln-iso_litc-trad+kf.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import time  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import tc_models as TM  # noqa: E402
import seedvig_eog as S  # noqa: E402

FEAT = "litc-trad+kf"
PARENT = f"tc-linln_{FEAT}"
MID = f"tc-linln-iso_{FEAT}"
KNOTS = np.linspace(0.0, 1.0, 1001)


def iso_map(u_oof):
    from sklearn.isotonic import IsotonicRegression
    u = np.sort(np.asarray(u_oof, float))
    ecdf = np.arange(1, len(u) + 1) / len(u)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(u, ecdf)
    return ir.predict(KNOTS)          # tabulated map on a fine knot grid


def apply_map(table, F):
    return np.interp(np.asarray(F, float), KNOTS, table)


def main():
    t0 = time.time()
    d = C.load_features(FEAT)
    b = C.base()
    X, y, groups = d["X"], b["perclos"], b["groups"]
    z = C.zlogit(y)
    for p in b["persons"]:
        p = int(p)
        if C.fold_done(MID, p):
            continue
        tf = time.time()
        par = C.load_fold(PARENT, p)
        alpha, sigma = float(par["alpha"]), float(par["sigma"])
        tr = np.flatnonzero(groups != p)
        te = np.flatnonzero(groups == p)
        S.assert_no_person_overlap(groups[tr], groups[te])
        Xtr, ztr, gtr = X[tr], z[tr], groups[tr]
        oof = np.zeros(len(tr))
        for itr, iva in S.inner_group_folds(gtr, k=5, seed=0):
            S.assert_no_person_overlap(gtr[itr], gtr[iva])
            prep = TM.make_prep().fit(Xtr[itr])
            B, b0 = TM.ridge_path(prep.transform(Xtr[itr]), ztr[itr], (alpha,))
            oof[iva] = prep.transform(Xtr[iva]) @ B[:, 0] + b0[0]
        u = C.ndtr((ztr - oof) / sigma)
        table = iso_map(u)
        C.save_fold(MID, p, rows=te, table=table, alpha=alpha, sigma=sigma,
                    oof_resid_sd=float(np.sqrt(np.mean((ztr - oof) ** 2))))
        print(f"[{MID}] fold {p}: alpha {alpha:g} sigma {sigma:.4f} R(0.5)={apply_map(table, 0.5):.3f} "
              f"({time.time() - tf:.1f} s)", flush=True)
    with np.load(C.CDFD / f"{PARENT}.npz") as zz:
        Fcut = zz["Fcut"].astype(np.float64)
        G = zz["cdf"].astype(np.float64)
        pit = zz["pit"].astype(np.float64)
    Fn, Gn, pitn = Fcut.copy(), G.copy(), pit.copy()
    for p in b["persons"]:
        f = C.load_fold(MID, int(p))
        r = f["rows"]
        Fn[r] = apply_map(f["table"], Fcut[r])
        Gn[r] = apply_map(f["table"], G[r])
        pitn[r] = apply_map(f["table"], pit[r])
    Gn = np.maximum.accumulate(Gn, axis=1)
    mean = C.grid_mean(Gn)
    median = C.grid_quantile(Gn, 0.5)
    q = {lev: (C.grid_quantile(Gn, 0.5 - lev / 200), C.grid_quantile(Gn, 0.5 + lev / 200)) for lev in (50, 80, 90)}
    C.finalize_tc(MID, Fn, Gn, mean, median, q, params={}, pit=pitn, logscore=None)
    C.log_compute(MID, time.time() - t0)
    print(f"[{MID}] complete", flush=True)
    return "complete"


if __name__ == "__main__":
    main()
