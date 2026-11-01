"""Person-level summaries of LOSO prediction files (shared by all topics).

    python -I summarize.py results/pub2_loso      -> summary.csv + summary.md in that folder

Rules (fixed for all five topics):
  * Metrics are computed per PERSON on that person's pooled test epochs (both recordings of
    persons 4 and 5 together), then averaged over persons; 95 % CI = percentile bootstrap over
    persons (B = 10,000).
  * three-class: macro-F1 over the classes present in the person's true labels, balanced
    accuracy, Cohen's kappa; all 21 persons.
  * binary: balanced accuracy and AUROC; persons with >= 20 epochs of BOTH classes
    (MIN_PER_CLASS) -> 17 persons (6, 11, 13, 19 excluded).
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seedvig_eog as S  # noqa: E402

MIN_PER_CLASS = 20


def per_person(z, task: str) -> dict:
    y, pred, score, groups = z["y"], z["pred"], z["score"], z["groups"]
    rows = {}
    for p in np.unique(groups):
        m = groups == p
        yt, yp = y[m], pred[m]
        if task == "binary":
            if min((yt == 0).sum(), (yt == 1).sum()) < MIN_PER_CLASS:
                continue
            s = score[m, 1] if score.shape[1] > 1 else score[m, 0]
            rows[int(p)] = {"bacc": balanced_accuracy_score(yt, yp),
                            "auroc": roc_auc_score(yt, s) if np.isfinite(s).all() else np.nan}
        else:
            present = np.unique(yt)
            rows[int(p)] = {"f1_macro": f1_score(yt, yp, labels=present, average="macro", zero_division=0),
                            "bacc": balanced_accuracy_score(yt, yp),
                            "kappa": cohen_kappa_score(yt, yp) if len(present) > 1 else np.nan}
    return rows


def summarize_file(path: pathlib.Path) -> dict:
    task = path.stem.split("_")[0]
    z = np.load(path)
    rows = per_person(z, task)
    out = {"run": path.stem, "task": task, "n_persons": len(rows)}
    for k in next(iter(rows.values())):
        v = np.array([r[k] for r in rows.values()], float)
        m, lo, hi = S.bootstrap_ci(v)
        out[k] = m
        out[f"{k}_lo"] = lo
        out[f"{k}_hi"] = hi
    return out


def main(folder: str) -> None:
    d = pathlib.Path(folder)
    res = [summarize_file(p) for p in sorted(d.glob("*.npz"))]
    keys = sorted({k for r in res for k in r}, key=lambda k: (k not in ("run", "task", "n_persons"), k))
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)
    lines = ["| run | n | main metric [95% CI] | second |", "|---|---:|---|---|"]
    for r in res:
        if r["task"] == "binary":
            a = f"BA {100*r['bacc']:.1f} [{100*r['bacc_lo']:.1f}; {100*r['bacc_hi']:.1f}]"
            b = f"AUROC {r['auroc']:.3f}" if np.isfinite(r.get("auroc", np.nan)) else "AUROC n/a"
        else:
            a = f"macro-F1 {100*r['f1_macro']:.1f} [{100*r['f1_macro_lo']:.1f}; {100*r['f1_macro_hi']:.1f}]"
            b = f"BA {100*r['bacc']:.1f}, kappa {r['kappa']:.3f}"
        lines.append(f"| {r['run']} | {r['n_persons']} | {a} | {b} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
