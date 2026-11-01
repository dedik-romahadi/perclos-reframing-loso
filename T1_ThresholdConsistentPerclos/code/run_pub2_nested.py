"""M15: stage-wise nested rebuild of Publication 2's selection procedure under outer LOSO (PREREG 4.7).

    python -I run_pub2_nested.py three [--persons 1,2,3] [--max-minutes 9]

Per outer fold, using only the 20 training persons: inner_group_folds(k=3, seed=0); inner fits on the epochs
with index mod 4 = 0 of each inner-training recording; inner score = mean over the 20 training persons (each
scored in the inner fold where it is validation) of the per-person BA over all its epochs.
  Stage 1: 9 presets (pub2_loso.make_clf, median imputer prepended) x 4 pub2orig types, no selection.
  Stage 2: winner's filter: none | ANOVA-F | Kruskal-Wallis H | chi2 on min-max scaled features (scaler fitted on
           inner-training rows) x k in {ceil(p/4), ceil(p/2)}.
  Stage 3 (qsvm / lsvm winners only): qsvm C in {0.1, 1, 10} x kernel scale s in {0.5, 1, 2} sqrt(p_sel)
           (gamma = 1 / s^2, coef0 = 1); lsvm C in {0.1, 1, 10}; incumbent = preset.
  Ties -> earlier configuration (classifiers lda, nb, tree, knn, lsvm, klr, rf, mlp, qsvm; types type1, type2,
  type3, all; no selection before filters; preset before tuned values).
Final refit of the chosen pipeline on all epochs of the 20 training persons; plain predict, no class weights.
Folds: results/folds/pub2nested_<task>/; selections: results/pub2_nested_selection.csv.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import seedvig_eog as S  # noqa: E402

warnings.filterwarnings("ignore")
CLFS = ["lda", "nb", "tree", "knn", "lsvm", "klr", "rf", "mlp", "qsvm"]
TYPES = ["type1", "type2", "type3", "all"]
TYPE_COLS = {"type1": slice(0, 168), "type2": slice(168, 402), "type3": slice(402, 417), "all": slice(0, 417)}


class Filter:
    """Univariate filter selecting the k best columns (scores computed on the fitting rows)."""

    def __init__(self, kind, k):
        self.kind, self.k = kind, k

    def fit(self, X, y):
        from scipy.stats import kruskal
        from sklearn.feature_selection import chi2, f_classif
        if self.kind == "anova":
            sc = f_classif(X, y)[0]
        elif self.kind == "kw":
            cls = np.unique(y)
            sc = np.array([kruskal(*[X[y == c, j] for c in cls]).statistic if np.ptp(X[:, j]) > 0 else 0.0
                           for j in range(X.shape[1])])
        elif self.kind == "chi2":
            lo, hi = X.min(0), X.max(0)
            Xm = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
            sc = chi2(Xm, y)[0]
        else:
            raise ValueError(self.kind)
        sc = np.nan_to_num(sc, nan=-np.inf)
        self.idx = np.sort(np.argsort(-sc, kind="stable")[: self.k])
        return self

    def transform(self, X):
        return X[:, self.idx]


def make_pipe(clf, svm=None, seed=0):
    import pub2_loso as P2
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    import tc_models as TM
    if svm is None:
        model = P2.make_clf(clf, seed)
    elif clf == "qsvm":
        model = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=2, coef0=1.0, C=svm["C"],
                                                    gamma=svm["gamma"], cache_size=2000))
    else:
        model = make_pipeline(StandardScaler(), LinearSVC(C=svm["C"], dual="auto", max_iter=5000, random_state=seed))
    return TM.MedianImputer(), model


class Pipe:
    def __init__(self, clf, typ, filt=None, svm=None):
        self.clf, self.typ, self.filt, self.svm = clf, typ, filt, svm

    def fit(self, X, y):
        X = X[:, TYPE_COLS[self.typ]]
        self.imp, self.model = make_pipe(self.clf, self.svm)
        Xi = self.imp.fit_transform(X)
        self.sel = Filter(*self.filt).fit(Xi, y) if self.filt else None
        if self.sel is not None:
            Xi = self.sel.transform(Xi)
        self.model.fit(Xi, y)
        return self

    def predict(self, X):
        Xi = self.imp.transform(X[:, TYPE_COLS[self.typ]])
        if self.sel is not None:
            Xi = self.sel.transform(Xi)
        return self.model.predict(Xi)


def inner_score(cfg, X, y, groups, inner, dec):
    """Mean over training persons of per-person BA (each person scored in its validation fold)."""
    pp = {}
    for tr, va in inner:
        trd = tr[dec[tr]]
        m = Pipe(*cfg).fit(X[trd], y[trd])
        pr = m.predict(X[va])
        for p in np.unique(groups[va]):
            mm = groups[va] == p
            pp[int(p)] = C.ba(y[va][mm], pr[mm])
    return float(np.mean(list(pp.values())))


def select(X, y, groups, epoch_idx, log):
    inner = list(S.inner_group_folds(groups, k=3, seed=0))
    for tr, va in inner:
        S.assert_no_person_overlap(groups[tr], groups[va])
    dec = (epoch_idx % 4) == 0
    trace = []
    best, best_s = None, -np.inf
    for clf in CLFS:                                     # stage 1
        for typ in TYPES:
            s = inner_score((clf, typ, None, None), X, y, groups, inner, dec)
            trace.append({"stage": 1, "clf": clf, "type": typ, "score": s})
            if s > best_s + 1e-12:
                best, best_s = (clf, typ, None, None), s
    log(f"   stage 1 winner {best[:2]} {best_s:.4f}")
    clf, typ = best[0], best[1]
    p = TYPE_COLS[typ].stop - TYPE_COLS[typ].start
    for kind in ("anova", "kw", "chi2"):                 # stage 2 (incumbent = no selection)
        for k in (math.ceil(p / 4), math.ceil(p / 2)):
            s = inner_score((clf, typ, (kind, k), None), X, y, groups, inner, dec)
            trace.append({"stage": 2, "clf": clf, "type": typ, "filter": kind, "k": k, "score": s})
            if s > best_s + 1e-12:
                best, best_s = (clf, typ, (kind, k), None), s
    log(f"   stage 2 winner {best[2]} {best_s:.4f}")
    if clf in ("qsvm", "lsvm"):                          # stage 3 (incumbent = preset)
        psel = best[2][1] if best[2] else p
        grid = ([{"C": c, "gamma": 1.0 / (f * math.sqrt(psel)) ** 2, "s_factor": f} for c in (0.1, 1.0, 10.0)
                 for f in (0.5, 1.0, 2.0)] if clf == "qsvm" else [{"C": c} for c in (0.1, 1.0, 10.0)])
        base = best
        for g in grid:
            s = inner_score((clf, typ, base[2], g), X, y, groups, inner, dec)
            trace.append({"stage": 3, "clf": clf, "type": typ, "svm": g, "score": s})
            if s > best_s + 1e-12:
                best, best_s = (clf, typ, base[2], g), s
        log(f"   stage 3 winner {best[3]} {best_s:.4f}")
    return best, best_s, trace


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["three", "binary"])
    ap.add_argument("--persons", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    X = C.load_features("pub2orig-all")["X"]
    b = C.base()
    groups = b["groups"]
    keep = np.ones(len(groups), bool) if a.task == "three" else b["yi"] != 2
    y = (b["yi"] - 1).astype(int) if a.task == "three" else (b["yi"] == 3).astype(int)
    job = f"pub2nested_{a.task}"
    persons = [int(p) for p in (a.persons.split(",") if a.persons else b["persons"])]
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    for p in persons:
        if C.fold_done(job, p):
            continue
        if time.time() > deadline - 300:
            log("[pub2nested] time limit; resume later")
            return "paused"
        tf = time.time()
        tr = np.flatnonzero((groups != p) & keep)
        te = np.flatnonzero(groups == p)
        S.assert_no_person_overlap(groups[tr], groups[te])
        cfg, sc, trace = select(X[tr], y[tr], groups[tr], b["epoch_idx"][tr], log)
        m = Pipe(*cfg).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        C.save_fold(job, p, rows=te, pred=pred, config=json.dumps({"clf": cfg[0], "type": cfg[1], "filter": cfg[2],
                                                                    "svm": cfg[3], "inner_score": sc}),
                    trace=json.dumps(trace))
        log(f"[{job}] fold {p}: {cfg} inner {sc:.4f} ({time.time() - tf:.0f} s)")
    done = [int(p) for p in b["persons"] if C.fold_done(job, int(p))]
    if len(done) == len(b["persons"]):
        nc = 3 if a.task == "three" else 2
        pred = np.full(len(groups), -1)
        score = np.zeros((len(groups), nc))
        rows = []
        for p in b["persons"]:
            z = C.load_fold(job, int(p))
            pred[z["rows"]] = z["pred"]
            score[z["rows"], :] = np.eye(nc)[np.clip(z["pred"], 0, nc - 1)]
            cfg = json.loads(str(z["config"]))
            rows.append({"task": a.task, "person": int(p), "clf": cfg["clf"], "type": cfg["type"],
                         "filter": (cfg["filter"] or ["none", ""])[0], "k": (cfg["filter"] or ["none", ""])[1],
                         "svm": json.dumps(cfg["svm"]) if cfg["svm"] else "preset", "inner_score": cfg["inner_score"]})
        C.write_csv(C.RES / f"pub2_nested_selection{'' if a.task == 'three' else '_binary'}.csv", rows)
        C.save_preds(a.task, "pub2nested_pub2orig", pred, score)
        C.log_compute(job, time.time() - t0)
        log(f"[{job}] complete")
        return "complete"
    return "partial"


if __name__ == "__main__":
    main()
