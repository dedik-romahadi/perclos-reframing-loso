"""A5 ablation (PREREG 4.6, tier 3; completed late, DEVIATIONS D14): isotonic PIT recalibration of M5 (TC-QHGB).

    python -I run_a5.py [--order asc|desc] [--max-minutes M]

Per outer fold (20 training persons only): 3 person-grouped inner folds (`inner_group_folds(k=3, seed=0)`, PREREG
section 6 table); in each, the 19 quantile HGB regressors of M5 (same settings, tc_models.QHGB) are fitted on the
inner-training persons and give inner OOF quantiles; inner PIT u_i = F_oof_i(clip(y_i)) with the M5 CDF construction
(rearrangement, end knots, piecewise-linear in logit). The isotonic map R (u -> empirical CDF of u; as run_a1.iso_map)
is applied to the saved M5 test CDF: F'(t) = R(F(t)) at the 14 cut-points and on the 512-point grid; decisions as for
every TC model (balanced Bayes). Two workers may run in opposite person order (fold files are written atomically).
Folds: results/folds/tc-qhgb-iso_litc-trad+kf/; model id tc-qhgb-iso_litc-trad+kf.
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
import run_a1 as A1  # noqa: E402
import seedvig_eog as S  # noqa: E402

FEAT = "litc-trad+kf"
PARENT = f"tc-qhgb_{FEAT}"
MID = f"tc-qhgb-iso_{FEAT}"


def fold(X, y, groups, p):
    tr = np.flatnonzero(groups != p)
    te = np.flatnonzero(groups == p)
    S.assert_no_person_overlap(groups[tr], groups[te])
    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    u = np.zeros(len(tr))
    for itr, iva in S.inner_group_folds(gtr, k=3, seed=0):
        S.assert_no_person_overlap(gtr[itr], gtr[iva])
        m = TM.QHGB().fit(Xtr[itr], ytr[itr])
        K = TM.qhgb_knots(m.predict_quantiles(Xtr[iva]))
        yc = np.clip(ytr[iva], C.CLIP_LO, C.CLIP_HI)
        u[iva] = np.array([TM.qhgb_cdf(K[i:i + 1], yc[i:i + 1])[0, 0] for i in range(len(iva))])
    return te, u, A1.iso_map(u)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=["asc", "desc"], default="asc")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    X = C.load_features(FEAT)["X"]
    b = C.base()
    y, groups = b["perclos"], b["groups"]
    persons = [int(p) for p in b["persons"]]
    if a.order == "desc":
        persons = persons[::-1]
    for p in persons:
        if C.fold_done(MID, p):
            continue
        if time.time() > deadline - 600:
            print(f"[{MID}] time limit; resume later", flush=True)
            return "paused"
        tf = time.time()
        te, u, table = fold(X, y, groups, p)
        if C.fold_done(MID, p):           # the other worker finished it meanwhile
            continue
        C.save_fold(MID, p, rows=te, table=table, u_oof=u.astype(np.float32))
        print(f"[{MID}] fold {p}: R(0.5)={A1.apply_map(table, 0.5):.3f} ({time.time() - tf:.1f} s)", flush=True)
    if not all(C.fold_done(MID, p) for p in persons):
        print(f"[{MID}] other worker still running; finalize later", flush=True)
        return "partial"
    finalize(b)
    C.log_compute(MID, time.time() - t0, f"A5 worker {a.order}")
    print(f"[{MID}] complete", flush=True)
    return "complete"


def finalize(b=None):
    b = b or C.base()
    with np.load(C.CDFD / f"{PARENT}.npz") as zz:
        Fcut = zz["Fcut"].astype(np.float64)
        G = zz["cdf"].astype(np.float64)
        pit = zz["pit"].astype(np.float64)
    Fn, Gn, pitn = Fcut.copy(), G.copy(), pit.copy()
    for p in b["persons"]:
        f = C.load_fold(MID, int(p))
        r = f["rows"]
        Fn[r] = A1.apply_map(f["table"], Fcut[r])
        Gn[r] = A1.apply_map(f["table"], G[r])
        pitn[r] = A1.apply_map(f["table"], pit[r])
    Gn = np.maximum.accumulate(Gn, axis=1)          # as run_a1 (R is monotone, so Fn stays monotone)
    mean = C.grid_mean(Gn)
    median = C.grid_quantile(Gn, 0.5)
    q = {lev: (C.grid_quantile(Gn, 0.5 - lev / 200), C.grid_quantile(Gn, 0.5 + lev / 200)) for lev in (50, 80, 90)}
    C.finalize_tc(MID, Fn, Gn, mean, median, q, params={}, pit=pitn, logscore=None)


if __name__ == "__main__":
    main()
