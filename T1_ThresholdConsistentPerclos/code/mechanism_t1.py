"""Post-registration descriptive analysis E3 (DEVIATIONS D18): why the linear TC model (M1) almost never predicts
the middle ('tired') class. Uses saved prediction / CDF files only (no model is refitted).

    python -I mechanism_t1.py

(1) Analytic feasibility. For a logit-normal predictive distribution LN(mu, sigma) and the balanced Bayes rule with the
    outer-training prevalences pi of the canonical cell (0.35, 0.70), 'tired' is chosen iff
    P(tired)/pi_tired > max(P(alert)/pi_alert, P(drowsy)/pi_drowsy). For each outer fold we find, on a fine mu grid,
    whether any mu makes 'tired' win, and the largest sigma* for which such a mu exists (bisection). M1 uses one
    homoscedastic sigma per fold, so if sigma > sigma* the middle class is unreachable for every epoch of that person.
(2) Empirical, canonical three-class, for M1 and the other TC models: share of epochs predicted 'tired', mean P(tired)
    on truly tired epochs, and (heteroscedastic heads M9/M11/M13 and the 3-seed raw-EOG versions) the share of epochs
    whose own sigma is below the fold's sigma*.
(3) Shape of M1's errors: logit-scale residual SD by quintile of mu (heteroscedasticity) and the observed share of
    tired epochs vs the mean predicted P(tired) by quintile of mu (reliability of the middle band).
Outputs: results/ext/mechanism_folds.csv, mechanism_models.csv, mechanism_shape.csv, MECHANISM.md. Descriptive only.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.special import logit, ndtr  # noqa: E402

import t1_common as C  # noqa: E402

OUT = C.RES / "ext"
OUT.mkdir(exist_ok=True)
T1, T2 = C.CANON
L1, L2 = logit(T1), logit(T2)
MU_GRID = np.linspace(-6, 6, 4801)
TC = {"M1": "tc-linln_litc-trad+kf", "M5": "tc-qhgb_litc-trad+kf", "M7": "tc-mlpln_litc-trad+kf",
      "M9": "tc-featgru_litc-trad", "M11": "tc-sinchybrid_trad6", "M13": "tc-resnet1dgru_trad6",
      "A1": "tc-linln-iso_litc-trad+kf", "M11-ens3": "tc-sinchybrid_trad6-ens3", "M13-ens3": "tc-resnet1dgru_trad6-ens3"}


def tired_wins(mu, sigma, pi):
    Fa = ndtr((L1 - mu) / sigma)
    Fd = ndtr((L2 - mu) / sigma)
    P = np.stack([Fa, Fd - Fa, 1 - Fd], -1) / pi
    return P[..., 1] > np.maximum(P[..., 0], P[..., 2])


def sigma_star(pi):
    lo, hi = 1e-3, 5.0
    if not tired_wins(MU_GRID, lo, pi).any():
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tired_wins(MU_GRID, mid, pi).any() else (lo, mid)
    return lo


def main():
    b = C.base()
    y, g = b["perclos"], b["groups"]
    lab = C.lab3(y, T1, T2)
    persons = [int(p) for p in b["persons"]]
    with np.load(C.CDFD / f"{TC['M1']}.npz") as z:
        mu1, sig1 = z["mu"].astype(float), z["sigma"].astype(float)
    folds, sstar = [], {}
    for p in persons:
        te = g == p
        pi = C.prevalence(lab[~te], 3)
        s_star = sigma_star(pi)
        sstar[p] = s_star
        s = float(sig1[te][0])
        folds.append({"person": p, "pi_alert": pi[0], "pi_tired": pi[1], "pi_drowsy": pi[2], "sigma_M1": s,
                      "sigma_star": s_star, "tired_reachable_M1": bool(s < s_star),
                      "n_pred_tired_M1": int(tired_wins(mu1[te], s, pi).sum())})
    C.write_csv(OUT / "mechanism_folds.csv", folds)

    models = []
    for name, mid in TC.items():
        pth = C.PREDS / f"three_{mid}.npz"
        if not pth.exists():
            continue
        with np.load(pth) as z:
            P, pred, yy, gg = z["score"].astype(float), z["pred"], z["y"], z["groups"]
        row = {"model": name, "model_id": mid, "share_pred_tired": float(np.mean(pred == 1)),
               "mean_P_tired_true_tired": float(P[yy == 1, 1].mean()),
               "mean_P_tired_all": float(P[:, 1].mean()), "share_true_tired": float(np.mean(yy == 1))}
        cdf = C.CDFD / f"{mid}.npz"
        with np.load(cdf) as z:
            if "sigma" in z.files and name != "M1" and not name.startswith("A"):
                sg = z["sigma"].astype(float)
                if sg.ndim == 2:          # several seeds: report the median seed sigma
                    sg = np.median(sg, 1)
                row["median_sigma"] = float(np.median(sg))
                row["share_sigma_below_star"] = float(np.mean(sg < np.array([sstar[int(q)] for q in g])))
        models.append(row)
    C.write_csv(OUT / "mechanism_models.csv", models)

    zt = C.zlogit(y)
    r = zt - mu1
    qs = np.quantile(mu1, [0, .2, .4, .6, .8, 1])
    shape = []
    with np.load(C.PREDS / f"three_{TC['M1']}.npz") as z:
        P1 = z["score"].astype(float)
    for k in range(5):
        m = (mu1 >= qs[k]) & (mu1 <= qs[k + 1]) if k == 4 else (mu1 >= qs[k]) & (mu1 < qs[k + 1])
        shape.append({"mu_quintile": k + 1, "mu_lo": qs[k], "mu_hi": qs[k + 1], "n": int(m.sum()),
                      "resid_sd_logit": float(r[m].std()), "sigma_M1_mean": float(sig1[m].mean()),
                      "obs_share_tired": float(np.mean(lab[m] == 1)), "mean_P_tired_M1": float(P1[m, 1].mean())})
    C.write_csv(OUT / "mechanism_shape.csv", shape)

    reach = sum(f["tired_reachable_M1"] for f in folds)
    L = ["# Why M1 rarely predicts 'tired' (E3, descriptive; DEVIATIONS D18)", "",
         f"Analytic: with the fold's training prevalences, 'tired' can win the balanced Bayes rule only if the "
         f"logit-normal sigma is below sigma* = {min(sstar.values()):.2f}-{max(sstar.values()):.2f} (range over folds). "
         f"M1's homoscedastic sigma was {min(f['sigma_M1'] for f in folds):.2f}-{max(f['sigma_M1'] for f in folds):.2f}; "
         f"'tired' was reachable at all in {reach}/21 folds.", "",
         "| model | share predicted tired | mean P(tired) on true tired | median sigma | share of epochs with sigma < sigma* |",
         "|---|---:|---:|---:|---:|"]
    for m in models:
        L.append(f"| {m['model']} | {100 * m['share_pred_tired']:.2f} % | {m['mean_P_tired_true_tired']:.3f} | "
                 f"{m.get('median_sigma', float('nan')):.2f} | "
                 f"{100 * m['share_sigma_below_star']:.1f} % |" if 'share_sigma_below_star' in m else
                 f"| {m['model']} | {100 * m['share_pred_tired']:.2f} % | {m['mean_P_tired_true_tired']:.3f} | | |")
    L += ["", f"True share of tired epochs: {100 * models[0]['share_true_tired']:.1f} %.", "",
          "| mu quintile (M1) | mu range | residual SD (logit) | M1 sigma | observed share tired | mean P(tired), M1 |",
          "|---:|---|---:|---:|---:|---:|"]
    for s in shape:
        L.append(f"| {s['mu_quintile']} | {s['mu_lo']:.2f} to {s['mu_hi']:.2f} | {s['resid_sd_logit']:.2f} | "
                 f"{s['sigma_M1_mean']:.2f} | {100 * s['obs_share_tired']:.1f} % | {s['mean_P_tired_M1']:.3f} |")
    (OUT / "MECHANISM.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
