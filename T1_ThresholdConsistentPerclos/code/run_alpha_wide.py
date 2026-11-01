"""Post-registration extension E1 (DEVIATIONS D13): M1 TC-LinLN with a wider ridge-alpha grid.

    python -I run_alpha_wide.py [--max-minutes 9]

Identical to run_tc.py on `litc-trad+kf` (same inner 5-fold person-grouped CRPS selection, sigma from the inner OOF
residuals at the chosen alpha, refit on all outer-training persons, balanced Bayes rule), except that the alpha grid
is extended upwards: {0.1, 1, ..., 1e4} (PREREG) + {1e5, 1e6, 1e7}. Reason: alpha = 1e4 (upper grid edge) was chosen
in 7/21 outer folds of M1. Secondary, descriptive; not part of any pre-registered family.
Model id: tc-linln-wa_litc-trad+kf; folds in results/folds/tc-linln-wa_litc-trad+kf/.
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
MID = f"tc-linln-wa_{FEAT}"
ALPHAS_WIDE = TM.ALPHAS + (1e5, 1e6, 1e7)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes if a.max_minutes else np.inf
    d = C.load_features(FEAT)
    b = C.base()
    X, y, groups = d["X"], b["perclos"], b["groups"]
    for p in b["persons"]:
        p = int(p)
        if C.fold_done(MID, p):
            continue
        if time.time() > deadline - 60:
            print(f"[{MID}] time limit; resume later", flush=True)
            return "paused"
        tf = time.time()
        tr, te = np.flatnonzero(groups != p), np.flatnonzero(groups == p)
        S.assert_no_person_overlap(groups[tr], groups[te])
        m = TM.LinLN(alphas=ALPHAS_WIDE).fit(X[tr], y[tr], groups[tr])
        C.save_fold(MID, p, rows=te, mu=m.predict_mu(X[te]), sigma=m.sigma, alpha=m.alpha,
                    inner_crps=m.inner_crps, inner_sigma=m.inner_sigma)
        print(f"[{MID}] fold {p}: alpha {m.alpha:g} sigma {m.sigma:.4f} ({time.time() - tf:.1f} s)", flush=True)
    n = len(y)
    mu, sig = np.zeros(n), np.zeros(n)
    rows = []
    for p in b["persons"]:
        z = C.load_fold(MID, int(p))
        mu[z["rows"]] = z["mu"]
        sig[z["rows"]] = float(z["sigma"])
        rows.append({"model_id": MID, "person": int(p), "alpha": float(z["alpha"]), "sigma": float(z["sigma"]),
                     **{f"inner_crps_a{al:g}": float(v) for al, v in zip(ALPHAS_WIDE, z["inner_crps"])}})
    C.write_csv(C.RES / f"fold_params_{MID}.csv", rows)
    Fcut, G, mean, median, q, pit, ls = TM.ln_outputs(mu, sig, y)
    C.finalize_tc(MID, Fcut, G, mean, median, q, params={"mu": mu, "sigma": sig}, pit=pit, logscore=ls)
    C.log_compute(MID, time.time() - t0, "extension E1")
    print(f"[{MID}] complete", flush=True)
    return "complete"


if __name__ == "__main__":
    main()
