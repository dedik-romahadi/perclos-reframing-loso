"""Reference rows (PREREG section 4.2).

    python -I run_refs.py qsvm [--max-minutes 9]     R2: Publication 2 QSVM preset on pub2orig-type2 (T1 run)
    python -I run_refs.py identity                     identity check of M2's canonical cells vs 00_COMMON R4

R2: pub2_loso.make_clf('qsvm') = StandardScaler -> SVC(poly, degree 2, coef0 1, C 1, gamma 'scale'), no class
weights, plain predict; score = decision function (binary: [-d, d]). Three-class on all epochs, binary on
alert/drowsy epochs (tired dropped), LOSO over persons. Folds: results/folds/ref-qsvm_<task>/.
R1, R3, R4, R5 are read from 00_COMMON/results/feature_check (identical protocol) when the identity check
passes; results/identity_check.json records the outcome.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import seedvig_eog as S  # noqa: E402
import summarize as SM  # noqa: E402

FC = C.COMMON / "results" / "feature_check"
REFS = {  # model_id: (featset, file stem in 00_COMMON feature_check)
    "ref-pub2orig-all_lda": ("pub2orig-all", "pub2orig-all_lda"),
    "ref-pub2-type1+kf_lda": ("pub2-type1+kf", "pub2-type1+kf_lda"),
    "ref-litc-trad+kf_lda": ("litc-trad+kf", "litc-trad+kf_lda"),
    "ref-time_lda": ("time", "time_lda"),
}


def run_qsvm(max_minutes=None):
    import pub2_loso as P2
    t0 = time.time()
    deadline = t0 + 60 * max_minutes if max_minutes else np.inf
    d = C.load_features("pub2orig-type2")
    b = C.base()
    X, groups = d["X"], b["groups"]
    for task in ("three", "binary"):
        job = f"ref-qsvm_{task}"
        keep = np.ones(len(groups), bool) if task == "three" else b["yi"] != 2
        y = (b["yi"] - 1) if task == "three" else (b["yi"] == 3).astype(int)
        for p in b["persons"]:
            p = int(p)
            if C.fold_done(job, p):
                continue
            if time.time() > deadline - 90:
                print("[qsvm] time limit; resume later", flush=True)
                return "paused"
            tf = time.time()
            tr = np.flatnonzero((groups != p) & keep)
            te = np.flatnonzero(groups == p)
            S.assert_no_person_overlap(groups[tr], groups[te])
            m = P2.make_clf("qsvm").fit(X[tr], y[tr])
            dec = m.decision_function(X[te])
            sc = dec if dec.ndim == 2 else np.c_[-dec, dec]
            C.save_fold(job, p, rows=te, pred=m.predict(X[te]), score=sc)
            print(f"[{job}] fold {p} {time.time() - tf:.1f} s", flush=True)
    for task in ("three", "binary"):
        job = f"ref-qsvm_{task}"
        nc = 3 if task == "three" else 2
        pred = np.full(len(groups), -1)
        score = np.zeros((len(groups), nc))
        for p in b["persons"]:
            z = C.load_fold(job, int(p))
            pred[z["rows"]], score[z["rows"]] = z["pred"], z["score"]
        C.save_preds(task, "ref-pub2orig-type2_qsvm", pred, score)
    C.log_compute("ref-qsvm", time.time() - t0)
    return "complete"


def identity():
    """M2 canonical cells vs 00_COMMON feature_check LDA files (>= 99.9 % identical epoch predictions and
    per-person BA equal to 4 decimals)."""
    out = {}
    for task in ("three", "binary"):
        mine = np.load(C.PREDS / f"{task}_direct-lda-grid_litc-trad+kf.npz")
        ref = np.load(FC / f"{task}_litc-trad+kf_lda.npz")
        assert np.array_equal(mine["groups"], ref["groups"]) and np.array_equal(mine["y"], ref["y"])
        same = float(np.mean(mine["pred"] == ref["pred"]))
        pa, pb = SM.per_person(mine, task), SM.per_person(ref, task)
        maxdiff = max(abs(pa[p]["bacc"] - pb[p]["bacc"]) for p in pa)
        ok = same >= 0.999 and all(round(pa[p]["bacc"], 4) == round(pb[p]["bacc"], 4) for p in pa)
        out[task] = {"identical_pred_frac": same, "n_diff": int(np.sum(mine["pred"] != ref["pred"])),
                     "max_abs_person_ba_diff": maxdiff, "pass": bool(ok)}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    (C.RES / "identity_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def refs_t1():
    """Fallback of PREREG 4.2 (identity check failed): R1, R3, R4, R5 re-run in T1's own pipeline = the M2
    protocol at the canonical cells (median imputer + StandardScaler fitted once per fold on ALL epochs of the
    training persons; shrinkage LDA, uniform priors; fast_lda engine). Binary LDA trained on alert/drowsy rows."""
    import fast_lda as FL
    import tc_models as TM
    t0 = time.time()
    b = C.base()
    groups, y = b["groups"], b["perclos"]
    for mid, (fs, _) in REFS.items():
        if (C.PREDS / f"three_{mid}.npz").exists() and (C.PREDS / f"binary_{mid}.npz").exists():
            continue
        X = C.load_features(fs)["X"]
        n = len(y)
        p3, s3 = np.full(n, -1), np.zeros((n, 3))
        p2, s2 = np.full(n, -1), np.zeros((n, 2))
        for p in b["persons"]:
            tr, te = np.flatnonzero(groups != p), np.flatnonzero(groups == p)
            S.assert_no_person_overlap(groups[tr], groups[te])
            prep = TM.make_prep().fit(X[tr])
            Xt, Xe = prep.transform(X[tr]), prep.transform(X[te])
            M = FL.BinMoments(Xt, y[tr])
            l3 = FL.GridLDA(M, FL.cell_classes("three", C.CANON))
            l2 = FL.GridLDA(M, FL.cell_classes("binarypair", C.CANON))
            p3[te], s3[te] = l3.predict(Xe), l3.predict_proba(Xe)
            p2[te], s2[te] = l2.predict(Xe), l2.predict_proba(Xe)
        C.save_preds("three", mid, p3, s3)
        C.save_preds("binary", mid, p2, s2)
        print(f"[refs_t1] {mid} done", flush=True)
    C.log_compute("refs_t1", time.time() - t0)
    return "complete"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["qsvm", "identity", "refs_t1"])
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()
    print(run_qsvm(a.max_minutes) if a.what == "qsvm" else identity() if a.what == "identity" else refs_t1())
