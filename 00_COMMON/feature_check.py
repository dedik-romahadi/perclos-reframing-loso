"""LOSO sanity check of the feature caches (pub2, pub2fix, pub2orig, lit, litc, ...).

    python -I feature_check.py run [--jobs 8] [--clfs lda lsvm rbf] [--tasks three binary] [--sets ...]
    python -I feature_check.py summary

Protocol (identical for every feature set; README rules 1-6):
  * leave-one-PERSON-out, 21 folds; assert_no_person_overlap in every fold.
  * pipeline fitted on the training persons only: median imputation -> z-scoring -> classifier.
  * classifiers fixed a priori (no tuning, so no test information enters model choice):
      lda  : LDA, lsqr + Ledoit-Wolf shrinkage, uniform class priors
      lsvm : linear SVM (plain hinge loss, L2 penalty alpha = 1e-4, i.e. roughly C = 1/(alpha*n)
             ~ 0.5 for the ~19,400 training epochs of the three-class task and ~0.9 for the binary
             task), class_weight = balanced, fitted by averaged SGD (SGDClassifier, 50 epochs,
             seed 0; 50 epochs do not reach the exact optimum). This is NOT LinearSVC's default
             (squared hinge, C = 1); LinearSVC needed 100-1400 s per run on the heavy-tailed Pub2
             ratio features.
      rbf  : RBF SVM (SVC, C = 1, gamma = 'scale', class_weight = balanced)
  * tasks: three (alert/tired/drowsy, 21 persons) and binary (alert vs drowsy, tired dropped;
    persons with >= 20 epochs per class are scored -> 17 persons).
  * metrics per person (summarize.per_person), mean over persons, 95 % percentile bootstrap CI
    (B = 10,000).
  * inference = PRE-SPECIFIED paired contrasts only (CONTRASTS below, fixed in the review step
    of 28 Oct 2026 before the new runs): sign-flip permutation test on per-person BA, Holm
    correction within each family x task x classifier. Every contrast compares like with like
    (same smoothing, same causality, unless the contrast is about that factor). The difference
    to 'pub2orig-all' in the main table is descriptive only.
  * "deployable" = feature_sets.is_deployable (explicit whitelist of causal sets without the
    eye-closure block). The best run is reported separately for deployable and offline sets.
Outputs: results/feature_check/<task>_<featset>_<clf>.npz (per-epoch predictions, same schema as
pub2_loso), summary.csv, contrasts.csv, summary.md.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feature_sets as FSET  # noqa: E402
import seedvig_eog as S  # noqa: E402
import summarize as SM  # noqa: E402

OUT = HERE / "results" / "feature_check"
SETS = ["time",
        "pub2-type1", "pub2-type2", "pub2-type3", "pub2-all",
        "pub2orig-type1", "pub2orig-type2", "pub2orig-type3", "pub2orig-all",
        "pub2fix-type1", "pub2fix-type2", "pub2fix-type3", "pub2fix-all",
        "lit-trad", "lit-fore", "litc-trad", "litc-fore",
        # equal smoothing: causal Kalman on every main set (review issue 2)
        "pub2-type1+kf", "pub2-all+kf", "pub2orig-type1+kf", "pub2orig-all+kf", "pub2fix-type1+kf", "pub2fix-all+kf",
        "lit-trad+kf", "lit-fore+kf", "litc-trad+kf", "litc-fore+kf",
        # sensitivity: blink blocks / explicit eye-closure block (review issue 4)
        "lit-trad+noblinkdur", "lit-trad+noblinkall", "lit-trad+F05",
        "litc-trad+kf+noblinkdur", "litc-trad+kf+noblinkall", "litc-trad+kf+F05",
        # NON-causal smoother, comparison with the SEED-VIG literature only
        "lit-trad+rts", "litc-trad+rts"]
# RBF only on the main sets (expensive); listed in priority order
RBF_SETS = ["time", "pub2-type2", "pub2-all", "pub2orig-type2", "pub2orig-all", "pub2fix-type2", "pub2fix-all",
            "lit-trad", "lit-fore", "litc-trad", "litc-fore", "lit-trad+kf", "lit-fore+kf", "litc-trad+kf",
            "litc-fore+kf", "pub2orig-all+kf", "pub2fix-all+kf"]
REFERENCE = "pub2orig-all"

# Pre-specified paired contrasts: (family, A, B). dBA = BA(A) - BA(B) per person.
CONTRASTS = [
    # corrected vs as-published Pub2, per feature type (review issue 6)
    ("pub2_fix_per_type", "pub2fix-type1", "pub2orig-type1"),
    ("pub2_fix_per_type", "pub2fix-type2", "pub2orig-type2"),
    ("pub2_fix_per_type", "pub2fix-type3", "pub2orig-type3"),
    ("pub2_fix_per_type", "pub2fix-all", "pub2orig-all"),
    # literature vs Pub2 at EQUAL smoothing and equal causality (review issue 2)
    ("featset_equal_smoothing", "lit-trad", "pub2orig-all"),
    ("featset_equal_smoothing", "lit-trad+kf", "pub2orig-all+kf"),
    ("featset_equal_smoothing", "lit-trad+kf", "pub2fix-all+kf"),
    ("featset_equal_smoothing", "lit-trad+kf", "pub2-type1+kf"),
    ("featset_equal_smoothing", "litc-trad+kf", "pub2-type1+kf"),
    # effect of causal Kalman smoothing alone
    ("kalman_smoothing", "pub2orig-all+kf", "pub2orig-all"),
    ("kalman_smoothing", "pub2-type1+kf", "pub2-type1"),
    ("kalman_smoothing", "lit-trad+kf", "lit-trad"),
    ("kalman_smoothing", "litc-trad+kf", "litc-trad"),
    # cost of causality (forward-only filters, offset-assigned fixations)
    ("causal_vs_offline", "litc-trad", "lit-trad"),
    ("causal_vs_offline", "litc-fore", "lit-fore"),
    ("causal_vs_offline", "litc-trad+kf", "lit-trad+kf"),
    # dependence on blink blocks that re-estimate the PERCLOS blink term (review issue 4)
    ("blink_dependence", "lit-trad+noblinkdur", "lit-trad"),
    ("blink_dependence", "lit-trad+noblinkall", "lit-trad"),
    ("blink_dependence", "lit-trad+F05", "lit-trad"),
    ("blink_dependence", "litc-trad+kf+noblinkdur", "litc-trad+kf"),
    ("blink_dependence", "litc-trad+kf+noblinkall", "litc-trad+kf"),
    # montage
    ("montage", "lit-fore", "lit-trad"),
    ("montage", "litc-fore+kf", "litc-trad+kf"),
]


def make_clf(name: str, n_classes: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC

    m = {"lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(n_classes, 1.0 / n_classes)),
         "lsvm": SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, average=True, max_iter=50, tol=None,
                               class_weight="balanced", random_state=0),
         "rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=1000)}[name]
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), m)


def _scores(model, X):
    d = model.decision_function(X)
    return d if d.ndim == 2 else np.c_[-d, d]


def run_fold(X, y, groups, person, clf):
    tr, te = np.flatnonzero(groups != person), np.flatnonzero(groups == person)
    S.assert_no_person_overlap(groups[tr], groups[te])
    model = make_clf(clf, len(np.unique(y))).fit(X[tr], y[tr])
    return te, model.predict(X[te]), _scores(model, X[te])


def run(task: str, featset: str, clf: str, jobs: int, cache: dict) -> None:
    out = OUT / f"{task}_{featset}_{clf}.npz"
    if out.exists():
        return
    if featset not in cache:
        cache.clear()
        cache[featset] = FSET.load(featset)
    d = cache[featset]
    X, yi = d["X"], d["yi"]
    if task == "three":
        y = S.three_class_labels(yi).astype(int)
        keep = np.ones(len(y), bool)
    else:
        y = S.binary_labels(yi).astype(int)
        keep = y >= 0
    X, y = X[keep], y[keep]
    groups, rec, perclos, t = d["groups"][keep], d["rec"][keep], d["perclos"][keep], d["t"][keep]
    t0 = time.time()
    res = Parallel(n_jobs=jobs)(delayed(run_fold)(X, y, groups, p, clf) for p in np.unique(groups))
    pred = np.full(len(y), -1)
    score = np.full((len(y), len(np.unique(y))), np.nan)
    for te, p, s in res:
        pred[te] = p
        score[te, : s.shape[1]] = s
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, y=y, pred=pred, score=score, groups=groups, rec=rec, perclos=perclos, t=t,
             n_features=X.shape[1])
    os.replace(tmp, out)
    print(f"{out.name}: {X.shape[1]} features, {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- summary
def _per_person(path: pathlib.Path, task: str) -> dict:
    return SM.per_person(np.load(path), task)


def _status(featset: str) -> str:
    if featset == "time":
        return "control"
    if FSET.is_deployable(featset):
        return "deployable"
    if "rts" in featset.split("+"):
        return "NON-causal"
    if "F05" in featset.split("+"):
        return "eye-closure"
    return "offline"


def _paired(a_pp: dict, b_pp: dict) -> tuple[np.ndarray, np.ndarray]:
    persons = sorted(set(a_pp) & set(b_pp))
    return (np.array([a_pp[q]["bacc"] for q in persons]), np.array([b_pp[q]["bacc"] for q in persons]))


def contrasts(pp: dict) -> list[dict]:
    rows = []
    for task in ("three", "binary"):
        for clf in ("lda", "lsvm", "rbf"):
            fam_rows: dict[str, list[dict]] = {}
            for fam, A, B in CONTRASTS:
                if (task, A, clf) not in pp or (task, B, clf) not in pp:
                    continue
                a, b = _paired(pp[(task, A, clf)], pp[(task, B, clf)])
                fam_rows.setdefault(fam, []).append(
                    {"family": fam, "task": task, "clf": clf, "A": A, "B": B, "n_persons": len(a),
                     "dBA_pp": 100 * float(np.mean(a - b)), "wins_A": f"{int(np.sum(a > b))}/{len(a)}",
                     "p_raw": S.paired_permutation_p(a, b)})
            for fam, rs in fam_rows.items():
                for r, p_adj in zip(rs, S.holm([r["p_raw"] for r in rs])):
                    r["p_holm"] = p_adj
                    r["family_size"] = len(rs)
                rows += rs
    return rows


def summary() -> None:
    files = sorted(p for p in OUT.glob("*.npz") if not p.name.endswith(".tmp.npz") and p.stem.split("_")[0] in ("three", "binary"))
    rows, pp = [], {}
    for p in files:
        task, featset, clf = p.stem.split("_", 2)
        per = _per_person(p, task)
        pp[(task, featset, clf)] = per
        z = np.load(p)
        r = {"task": task, "featset": featset, "clf": clf, "status": _status(featset), "n_persons": len(per),
             "n_features": int(z["n_features"]) if "n_features" in z.files else -1}
        for k in next(iter(per.values())):
            v = np.array([x[k] for x in per.values()], float)
            m, lo, hi = S.bootstrap_ci(v)
            r[k], r[f"{k}_lo"], r[f"{k}_hi"] = m, lo, hi
        rows.append(r)
    # descriptive difference to the as-published reference (no test: see contrasts)
    for r in rows:
        ref = pp.get((r["task"], REFERENCE, r["clf"]))
        if ref is None or r["featset"] == REFERENCE:
            continue
        a, b = _paired(pp[(r["task"], r["featset"], r["clf"])], ref)
        r["dBA_vs_ref"] = float(np.mean(a - b))
        r["wins_vs_ref"] = f"{int(np.sum(a > b))}/{len(a)}"
    keys = ["task", "featset", "clf", "status", "n_persons", "n_features", "bacc", "bacc_lo", "bacc_hi", "f1_macro",
            "f1_macro_lo", "f1_macro_hi", "kappa", "auroc", "auroc_lo", "auroc_hi", "dBA_vs_ref", "wins_vs_ref"]
    keys += sorted({k for r in rows for k in r} - set(keys))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    con = contrasts(pp)
    ckeys = ["family", "task", "clf", "A", "B", "n_persons", "dBA_pp", "wins_A", "p_raw", "p_holm", "family_size"]
    with open(OUT / "contrasts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ckeys)
        w.writeheader()
        w.writerows(con)

    md = ["# LOSO feature check (person-level, 95 % bootstrap CI over persons)", "",
          "Protocol: leave-one-person-out (21 folds), median imputation + z-scoring fitted on training persons, "
          "fixed classifiers: LDA (Ledoit-Wolf shrinkage, uniform priors); linear SVM (hinge, L2, alpha = 1e-4, "
          "about C = 0.5 three-class / 0.9 binary, balanced class weights) fitted by averaged SGD (50 epochs); "
          "RBF SVM (C = 1, balanced).", "",
          "Status: `deployable` = causal at epoch resolution (look-ahead <= 1 epoch) and without the eye-closure "
          "block (explicit whitelist `feature_sets.is_deployable`); `offline` = zero-phase filtering of the "
          "continuous record (look-ahead up to 7 epochs at 1e-3 and 13 epochs at 1e-6, see `lookahead.md`) or "
          "whole-recording statistics (`pub2orig`); "
          "`NON-causal` = RTS smoother; `eye-closure` = includes F05 (PERCLOS proxy). `+kf` = causal Kalman "
          "smoothing. dBA vs `pub2orig-all` is descriptive; inference uses only the pre-specified contrasts "
          "(second part).", ""]
    order = {s: i for i, s in enumerate(SETS)}
    for task in ("three", "binary"):
        md += [f"## Task: {'three-class (alert / tired / drowsy), 21 persons' if task == 'three' else 'binary (alert vs drowsy), 17 persons'}", ""]
        sec = "macro-F1" if task == "three" else "AUROC"
        md += [f"| feature set | status | n feat | clf | BA % [95% CI] | {sec} | dBA vs pub2orig-all (pp) | wins |",
               "|---|---|---:|---|---|---|---:|---|"]
        sel = sorted([r for r in rows if r["task"] == task], key=lambda r: (order.get(r["featset"], 99), r["clf"]))
        for r in sel:
            ba = f"{100 * r['bacc']:.1f} [{100 * r['bacc_lo']:.1f}; {100 * r['bacc_hi']:.1f}]"
            if task == "three":
                s2 = f"{100 * r['f1_macro']:.1f} [{100 * r['f1_macro_lo']:.1f}; {100 * r['f1_macro_hi']:.1f}]"
            else:
                s2 = f"{r['auroc']:.3f} [{r['auroc_lo']:.3f}; {r['auroc_hi']:.3f}]"
            d = f"{100 * r['dBA_vs_ref']:+.1f}" if "dBA_vs_ref" in r else "ref" if r["featset"] == REFERENCE else ""
            md.append(f"| {r['featset']} | {r['status']} | {r['n_features']} | {r['clf']} | {ba} | {s2} | {d} | {r.get('wins_vs_ref', '')} |")
        for label, cond in (("deployable", lambda r: r["status"] == "deployable"),
                            ("offline (not deployable: zero-phase look-ahead of 1-2 min or whole-recording statistics)",
                             lambda r: r["status"] == "offline")):
            best = max((r for r in rows if r["task"] == task and cond(r)), key=lambda r: r["bacc"], default=None)
            if best:
                md += ["", f"Best {label} run: **{best['featset']} / {best['clf']}**, "
                           f"BA {100 * best['bacc']:.1f} % [{100 * best['bacc_lo']:.1f}; {100 * best['bacc_hi']:.1f}]. "
                           "Selected post hoc over the runs here, so it is optimistic (winner's curse)."]
        md.append("")
    md += ["## Pre-specified paired contrasts (per-person BA, sign-flip permutation, Holm within family x task x clf)", "",
           "| family | task | clf | A | B | dBA A-B (pp) | A better | p raw | p Holm (family size) |",
           "|---|---|---|---|---|---:|---|---:|---:|"]
    for c in con:
        md.append(f"| {c['family']} | {c['task']} | {c['clf']} | {c['A']} | {c['B']} | {c['dBA_pp']:+.1f} | "
                  f"{c['wins_A']} | {c['p_raw']:.3f} | {c['p_holm']:.3f} ({c['family_size']}) |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--clfs", nargs="+", default=["lda", "lsvm", "rbf"])
    ap.add_argument("--tasks", nargs="+", default=["three", "binary"])
    ap.add_argument("--sets", nargs="+", default=None)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary()
        return
    jobs = min(a.jobs, 8)
    cache: dict = {}
    for clf in a.clfs:
        sets = a.sets or (RBF_SETS if clf == "rbf" else SETS)
        for fs in sets:
            for task in a.tasks:
                run(task, fs, clf, jobs, cache)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
