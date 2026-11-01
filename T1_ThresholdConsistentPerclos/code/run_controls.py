"""Label-permutation control (PREREG section 10, H9b; descriptive).

    python -I run_controls.py [--seeds 0,1,2,3,4] [--max-minutes 9]

For permutation seed s and every outer fold (test person p): each TRAINING recording's PERCLOS series (and
the labels derived from it) is circularly shifted by an offset drawn from rng = default_rng([s, p]) in
[100, 785] epochs (one offset per training recording, in recording order). TC-LinLN (M1, incl. its inner
alpha/sigma selection) and the canonical-cell shrinkage LDAs (three-class and binary) are refitted on the
shifted labels and scored on the TRUE labels of the test person.
Folds: results/folds/perm<s>/; merged: results/controls/perm<s>_{three,binary}_{tc-linln,lda}.npz.
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


def shifted_perclos(y, rec, train_rows, seed, person):
    rng = np.random.default_rng([int(seed), int(person)])
    ys = y.copy()
    for r in np.unique(rec[train_rows]):
        idx = np.flatnonzero(rec == r)
        off = int(rng.integers(100, 786))
        ys[idx] = np.roll(y[idx], off)
    return ys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    d = C.load_features(FEAT)
    b = C.base()
    X, y, groups, rec = d["X"], b["perclos"], b["groups"], b["rec"]
    out_dir = C.RES / "controls"
    out_dir.mkdir(exist_ok=True)
    for s in [int(v) for v in a.seeds.split(",")]:
        job = f"perm{s}"
        for p in b["persons"]:
            p = int(p)
            if C.fold_done(job, p):
                continue
            if time.time() > deadline - 120:
                print("[perm] time limit; resume later", flush=True)
                return "paused"
            tr, te = np.flatnonzero(groups != p), np.flatnonzero(groups == p)
            S.assert_no_person_overlap(groups[tr], groups[te])
            ys = shifted_perclos(y, rec, tr, s, p)
            m = TM.LinLN().fit(X[tr], ys[tr], groups[tr])
            mu = m.predict_mu(X[te])
            prep = TM.make_prep().fit(X[tr])
            Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
            l3 = C.lab3(ys[tr], *C.CANON)
            lda3 = TM.make_lda(3).fit(Xtr, l3)
            l2 = C.lab2(ys[tr], *C.CANON)
            k2 = l2 >= 0
            lda2 = TM.make_lda(2).fit(Xtr[k2], l2[k2])
            C.save_fold(job, p, rows=te, mu=mu, sigma=m.sigma, alpha=m.alpha, pred3=lda3.predict(Xte),
                        score3=lda3.predict_proba(Xte), pred2=lda2.predict(Xte), score2=lda2.predict_proba(Xte))
            print(f"[{job}] fold {p} done", flush=True)
        n = len(y)
        mu, sig = np.zeros(n), np.zeros(n)
        p3, s3 = np.zeros(n, int), np.zeros((n, 3))
        p2, s2 = np.zeros(n, int), np.zeros((n, 2))
        for p in b["persons"]:
            z = C.load_fold(job, int(p))
            r = z["rows"]
            mu[r], sig[r] = z["mu"], float(z["sigma"])
            p3[r], s3[r], p2[r], s2[r] = z["pred3"], z["score3"], z["pred2"], z["score2"]
        Fcut = C.ln_Fcut(mu, sig)
        g = C.tc_grid(Fcut, "bayes")
        keep2 = b["yi"] != 2
        for task, model, pred, score in (("three", "tc-linln", g["three"][0][:, C.CANON_IDX], g["three"][1][:, C.CANON_IDX]),
                                         ("binary", "tc-linln", g["binarypair"][0][:, C.CANON_IDX],
                                          g["binarypair"][1][:, C.CANON_IDX]),
                                         ("three", "lda", p3, s3), ("binary", "lda", p2, s2)):
            keep = np.ones(n, bool) if task == "three" else keep2
            yy = (b["yi"] - 1) if task == "three" else (b["yi"][keep2] == 3).astype(int)
            np.savez_compressed(out_dir / f"perm{s}_{task}_{model}.npz", y=yy, pred=np.asarray(pred)[keep],
                                score=np.asarray(score, float)[keep], groups=groups[keep], rec=rec[keep],
                                perclos=y[keep], t=b["t"][keep])
    C.log_compute("perm_controls", time.time() - t0)
    print("[perm] complete", flush=True)
    return "complete"


if __name__ == "__main__":
    main()
