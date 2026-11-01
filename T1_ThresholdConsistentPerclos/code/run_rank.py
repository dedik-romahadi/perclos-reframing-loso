"""H7a classifier-ranking models (PREREG section 4.8): the 9 Publication 2 presets on pub2orig-all, no
selection, trained on the epochs with index mod 4 = 0 of all training recordings, at the 3 x 3 sub-grid
t1 in {0.25, 0.35, 0.45} x t2 in {0.60, 0.70, 0.80} (three-class); all test epochs scored; plain predict.

    python -I run_rank.py [--persons 1,2] [--max-minutes 9]

Folds: results/folds/rank_presets/; merged: results/grid/rank_presets.npz (pred n x 9 cells x 9 presets).
(H7b feature-type ranking uses run_grid_direct.py on pub2orig-type1/2/3/all with --tasks three.)
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import run_pub2_nested as RP  # noqa: E402
import seedvig_eog as S  # noqa: E402

warnings.filterwarnings("ignore")
JOB = "rank_presets"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    X = C.load_features("pub2orig-all")["X"]
    b = C.base()
    groups, y = b["groups"], b["perclos"]
    persons = [int(p) for p in (a.persons.split(",") if a.persons else b["persons"])]
    for p in persons:
        if C.fold_done(JOB, p):
            continue
        if time.time() > deadline - 600:
            print("[rank] time limit; resume later", flush=True)
            return "paused"
        tf = time.time()
        tr = np.flatnonzero((groups != p) & (b["epoch_idx"] % 4 == 0))
        te = np.flatnonzero(groups == p)
        S.assert_no_person_overlap(groups[tr], groups[te])
        pred = np.full((len(te), len(C.SUBGRID), len(RP.CLFS)), -1, np.int8)
        for j, cell in enumerate(C.SUBGRID):
            lab = C.lab3(y[tr], *cell).astype(int)
            for k, clf in enumerate(RP.CLFS):
                pred[:, j, k] = RP.Pipe(clf, "all").fit(X[tr], lab).predict(X[te])
        C.save_fold(JOB, p, rows=te, pred=pred)
        print(f"[{JOB}] fold {p} ({time.time() - tf:.0f} s)", flush=True)
    if all(C.fold_done(JOB, int(p)) for p in b["persons"]):
        pred = np.full((len(y), len(C.SUBGRID), len(RP.CLFS)), -1, np.int8)
        for p in b["persons"]:
            z = C.load_fold(JOB, int(p))
            pred[z["rows"]] = z["pred"]
        np.savez_compressed(C.GRID / "rank_presets.npz", pred=pred, cells=np.array(C.SUBGRID), clfs=np.array(RP.CLFS))
        C.log_compute(JOB, time.time() - t0)
        print(f"[{JOB}] complete", flush=True)
        return "complete"
    return "partial"


if __name__ == "__main__":
    main()
