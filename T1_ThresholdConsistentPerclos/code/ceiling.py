"""H8 label-jitter analysis (PREREG H8; descriptive except the enrichment decision).

    python -I ceiling.py

Per recording: (A) variogram nugget: gamma(h) = 1/2 mean (y_{t+h} - y_t)^2, h = 1..5, OLS gamma(h) = a + b h,
sigma_A^2 = max(a, 0); (B) 5-epoch centred local-linear smoother (>= 3 points at the edges), sigma_B^2 =
sum residual^2 / (n - tr H). Near = |y - nearest canonical cut (0.35, 0.70)| < 2 sigma_r. Enrichment
E = P(near | error) / P(near) of the canonical three-class errors of M1 (replicate: R4 = M2 canonical), pooled
over epochs, person-cluster bootstrap CI (B = 10,000, seed 0). Ceiling: L = smoothed y (B), 1,000 Monte-Carlo
draws eps ~ N(0, sigma_r^2) per estimator: perfect-latent ceiling = mean per-person BA / macro-F1 / QWK of
class(L) vs class(L + eps); inter-replicate agreement = class(L + eps1) vs class(L + eps2).
Outputs: results/ceiling.csv, results/ceiling_summary.json, results/jitter_sigma.csv.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import json  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402

B = C.base()
Y, G, REC = B["perclos"], B["groups"], B["rec"]
P21 = [int(p) for p in B["persons"]]


def sigma_A(y):
    hs = np.arange(1, 6)
    gam = np.array([0.5 * np.mean((y[h:] - y[:-h]) ** 2) for h in hs])
    b, a = np.polyfit(hs, gam, 1)
    return float(np.sqrt(max(a, 0.0)))


def smoother_B(y):
    n = len(y)
    H = np.zeros((n, n))
    t = np.arange(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if hi - lo < 3:
            lo, hi = (0, 3) if i < 2 else (n - 3, n)
        idx = np.arange(lo, hi)
        Xd = np.c_[np.ones(len(idx)), t[idx] - t[i]]
        w = np.linalg.solve(Xd.T @ Xd, Xd.T)[0]          # local-linear fitted value at t_i
        H[i, idx] = w
    fit = H @ y
    res = y - fit
    s2 = float(np.sum(res ** 2) / (n - np.trace(H)))
    return fit, float(np.sqrt(max(s2, 0.0)))


def enrichment(err, near, groups, b=10000, seed=0):
    def E(mask):
        e, nr = err[mask], near[mask]
        pn = nr.mean()
        return (nr[e].mean() / pn) if (pn > 0 and e.any()) else np.nan
    est = E(np.ones(len(err), bool))
    rng = np.random.default_rng(seed)
    persons = np.unique(groups)
    idx_by_p = {p: np.flatnonzero(groups == p) for p in persons}
    # cluster bootstrap via counts (pooled ratio with person weights)
    e_n = {p: (err[i] & near[i]).sum() for p, i in idx_by_p.items()}
    e_t = {p: err[i].sum() for p, i in idx_by_p.items()}
    n_n = {p: near[i].sum() for p, i in idx_by_p.items()}
    n_t = {p: len(i) for p, i in idx_by_p.items()}
    arr = np.array([[e_n[p], e_t[p], n_n[p], n_t[p]] for p in persons], float)
    draws = rng.integers(0, len(persons), (b, len(persons)))
    cnt = np.stack([np.bincount(d, minlength=len(persons)) for d in draws])
    tot = cnt @ arr
    vals = (tot[:, 0] / tot[:, 1]) / (tot[:, 2] / tot[:, 3])
    return float(est), float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def conf_metrics(conf):
    """conf: (..., 3, 3) true x pred -> BA (classes present), macro-F1 (present), QWK."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    rows = conf.sum(-1)
    cols = conf.sum(-2)
    present = rows > 0
    rec = np.where(present, tp / np.maximum(rows, 1), 0)
    ba = rec.sum(-1) / present.sum(-1)
    f1 = np.where(present, 2 * tp / np.maximum(rows + cols, 1), 0)
    f1m = f1.sum(-1) / present.sum(-1)
    n = conf.sum((-1, -2))
    W = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2 / 4.0
    E = rows[..., :, None] * cols[..., None, :] / np.maximum(n, 1)[..., None, None]
    num = (W * conf).sum((-1, -2))
    den = (W * E).sum((-1, -2))
    qwk = 1 - num / np.where(den > 0, den, np.nan)
    return ba, f1m, qwk


def person_conf(t, p):
    k = np.searchsorted(P21, G) * 9 + t * 3 + p
    return np.bincount(k, minlength=len(P21) * 9).reshape(len(P21), 3, 3)


def main():
    sA, sB, fitB = np.zeros(len(Y)), np.zeros(len(Y)), np.zeros(len(Y))
    rows = []
    for r in np.unique(REC):
        m = REC == r
        a = sigma_A(Y[m])
        fit, b = smoother_B(Y[m])
        sA[m], sB[m], fitB[m] = a, b, fit
        rows.append({"rec": C.S.RECORDINGS[r] if hasattr(C, "S") else int(r), "person": int(G[m][0]), "sigma_A": a,
                     "sigma_B": b, "ratio_A_B": a / b if b > 0 else np.nan})
    C.write_csv(C.RES / "jitter_sigma.csv", rows)
    ratio_med = float(np.nanmedian([r["ratio_A_B"] for r in rows]))
    dist = np.minimum(np.abs(Y - 0.35), np.abs(Y - 0.70))
    truth3 = C.lab3(Y, *C.CANON)
    out = {"median_sigma_A": float(np.median([r["sigma_A"] for r in rows])),
           "median_sigma_B": float(np.median([r["sigma_B"] for r in rows])), "median_ratio_A_B": ratio_med}
    crow = []
    for model, mid in (("M1", "tc-linln_litc-trad+kf"), ("R4", "direct-lda-grid_litc-trad+kf")):
        pred = C.load_grid("three", mid)[0][:, C.CANON_IDX]
        err = pred != truth3
        for est, s in (("A", sA), ("B", sB)):
            near = dist < 2 * s
            e, lo, hi = enrichment(err, near, G)
            crow.append({"quantity": "enrichment E", "model": model, "estimator": est, "value": e, "ci_lo": lo,
                         "ci_hi": hi, "p_near": float(near.mean()), "p_near_given_error": float(near[err].mean())})
            if model == "M1":
                out[f"E_{est}"], out[f"E_{est}_lo"], out[f"E_{est}_hi"] = e, lo, hi
    rng = np.random.default_rng(0)
    L = np.clip(fitB, 0, 1)
    cL = C.lab3(L, *C.CANON)
    for est, s in (("A", sA), ("B", sB)):
        acc = {"ceiling": [], "replicate": []}
        for _ in range(1000):
            t1 = C.lab3(L + rng.normal(0, 1, len(L)) * s, *C.CANON)
            t2 = C.lab3(L + rng.normal(0, 1, len(L)) * s, *C.CANON)
            acc["ceiling"].append(conf_metrics(person_conf(t1, cL)))
            acc["replicate"].append(conf_metrics(person_conf(t1, t2)))
        for kind, vals in acc.items():
            arr = np.array(vals)                        # (1000, 3 metrics, 21 persons)
            mean_pp = np.nanmean(arr, axis=0)           # per person over draws
            for k, name in enumerate(("bacc", "f1_macro", "qwk")):
                v = np.nanmean(mean_pp[k])
                crow.append({"quantity": f"{kind} {name}", "model": "perfect latent" if kind == "ceiling" else
                             "two noisy replicates", "estimator": est, "value": float(v),
                             "ci_lo": float(np.quantile(np.nanmean(arr[:, k], axis=1), 0.025)),
                             "ci_hi": float(np.quantile(np.nanmean(arr[:, k], axis=1), 0.975))})
                if name == "bacc":
                    out[f"{kind}_bacc_{est}"] = float(v)
    C.write_csv(C.RES / "ceiling.csv", crow)
    supported = out["E_A_lo"] > 1.5 and out["E_B_lo"] > 1.5
    agree = 2 / 3 <= ratio_med <= 3 / 2
    out["decision"] = (("supported" if supported else "not supported") +
                       ("; estimators agree (median sigma_A/sigma_B in [2/3, 3/2])" if agree else
                        "; estimators DISAGREE: no claim rests on the ceiling value"))
    out["ceiling_claim_allowed"] = bool(agree)
    out["perclos_window_overlap_check"] = (
        "Zheng & Lu 2017 (J Neural Eng 14:026017; local copy DATASET/SADT-SEEDVIG/SEEDVIG_full.txt) define PERCLOS "
        "as (blink + CLOS) / interval 'over a specified time interval' and state 8-s non-overlapping windows only for "
        "the EOG/EEG features; the PERCLOS window length and overlap are not stated, so overlapping (autocorrelated) "
        "label windows cannot be excluded and the nugget may underestimate the jitter.")
    (C.RES / "ceiling_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
