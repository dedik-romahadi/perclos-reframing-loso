"""Leave-one-PERSON-out rebuild of the Publication 2 pipeline (shared baseline).

    python -I pub2_loso.py <task> <featset> <clf> [--jobs N]
      task    : three (alert/tired/drowsy, 23 recordings) | binary (alert vs drowsy, tired dropped)
      featset : type1 | type2 | type3 | all | time   ('time' = epoch onset only: time-on-task control)
      clf     : knn | rf | lsvm | tree | klr | lda | nb | mlp | qsvm
Output: results/pub2_loso/<task>_<featset>_<clf>.npz with per-epoch predictions/scores, groups,
recording index and labels, so that topics can run paired person-level comparisons.

Classifier settings mirror MATLAB Classification Learner presets used in Publication 2
(medium KNN k=10, medium tree 20 splits, medium NN 25 units, QSVM = polynomial order 2,
box 1, automatic kernel scale), all with z-scoring fitted on the training persons only.
No class re-weighting (as in the original); balanced accuracy / macro-F1 are reported.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
from joblib import Parallel, delayed

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seedvig_eog as S  # noqa: E402

OUT = HERE / "results" / "pub2_loso"


def load_features(featset: str):
    names = S.RECORDINGS
    blocks, t, yi, perclos, groups, rec = [], [], [], [], [], []
    for i, n in enumerate(names):
        z = np.load(S.CACHE_DIR / f"pub2_{n}.npz")
        r = S.load_recording(n)
        if featset == "all":
            blocks.append(np.hstack([z["type1"], z["type2"], z["type3"]]))
        elif featset == "time":
            blocks.append(r["t"][:, None])
        else:
            blocks.append(z[featset])
        t.append(r["t"]); yi.append(r["yi"]); perclos.append(r["perclos"])
        groups.append(np.full(S.N_EPOCH, r["person"])); rec.append(np.full(S.N_EPOCH, i))
    cat = np.concatenate
    return np.vstack(blocks), cat(yi), cat(perclos), cat(groups), cat(rec), cat(t)


def make_clf(name: str, seed: int = 0):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    m = {
        "knn": KNeighborsClassifier(n_neighbors=10),
        "rf": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "lsvm": LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=seed),
        "tree": DecisionTreeClassifier(max_leaf_nodes=21, random_state=seed),
        "klr": make_pipeline(Nystroem(kernel="rbf", n_components=500, random_state=seed),
                             LogisticRegression(max_iter=2000)),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "nb": GaussianNB(),
        "mlp": MLPClassifier(hidden_layer_sizes=(25,), max_iter=1000, early_stopping=False, random_state=seed),
        "qsvm": SVC(kernel="poly", degree=2, coef0=1.0, C=1.0, gamma="scale", cache_size=2000),
    }[name]
    return make_pipeline(StandardScaler(), m)


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:
            pass
    d = model.decision_function(X)
    return d if d.ndim == 2 else np.c_[-d, d]


def run_fold(X, y, groups, test_person, clf):
    tr = np.flatnonzero(groups != test_person)
    te = np.flatnonzero(groups == test_person)
    S.assert_no_person_overlap(groups[tr], groups[te])
    model = make_clf(clf).fit(X[tr], y[tr])
    return te, model.predict(X[te]), _scores(model, X[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["three", "binary"])
    ap.add_argument("featset", choices=["type1", "type2", "type3", "all", "time"])
    ap.add_argument("clf")
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    X, yi, perclos, groups, rec, t = load_features(a.featset)
    if a.task == "three":
        y = S.three_class_labels(yi).astype(int)
        keep = np.ones(len(y), bool)
    else:
        y = S.binary_labels(yi).astype(int)
        keep = y >= 0
    X, y, groups, rec, perclos, t = X[keep], y[keep], groups[keep], rec[keep], perclos[keep], t[keep]
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs)(delayed(run_fold)(X, y, groups, p, a.clf) for p in np.unique(groups))
    pred = np.full(len(y), -1)
    nclass = len(np.unique(y))
    score = np.full((len(y), nclass), np.nan)
    for te, p, s in res:
        pred[te] = p
        score[te, : s.shape[1]] = s
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.task}_{a.featset}_{a.clf}.npz"
    np.savez(out, y=y, pred=pred, score=score, groups=groups, rec=rec, perclos=perclos, t=t)
    print(f"{out.name}: {len(np.unique(groups))} folds in {time.time() - t0:.0f} s; "
          f"pooled bacc {np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]):.4f}")


if __name__ == "__main__":
    main()
