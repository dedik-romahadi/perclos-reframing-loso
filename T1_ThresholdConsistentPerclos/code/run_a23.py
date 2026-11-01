"""A2 and A3 ablations of M7 (PREREG 4.6, tier 3; completed late, DEVIATIONS D14).

    python -I run_a23.py a2|a3 [--max-minutes M]

Both use the M7 setting exactly (per-epoch MLP trunk on litc-trad+kf, t1_nets.mlp_trunk, CPU, seeds 0, 1, 2, norm
global, K = 8 (ignored by the per-epoch trunk), target_stride 1, the T1 trainer copy deep_train_t1 with task
'perclos_ln': inner-validation persons, AdamW, cosine schedule, 40 epochs, patience 6, early stopping on the lowest
mean per-person inner CRPS). Only the head/loss differs; deep_train_t1.py itself is NOT edited - the module-level
helpers it calls are replaced inside this process only:
  A2 Beta head:   alpha = 1 + softplus(out0), beta = 1 + softplus(out1); loss = Beta NLL of clip(y, 0.005, 0.995);
                  inner CRPS / test CRPS from the Beta CDF on the trainer's 256-point grid; stored score = [alpha, beta].
  A3 auxiliary CE: logit-normal head and NLL as M7 plus lambda = 0.5 x cross-entropy of the three canonical classes
                  (alert < 0.35 <= tired < 0.70 <= drowsy) with probabilities derived from the same predictive
                  distribution (F(0.35), F(0.70) - F(0.35), 1 - F(0.70)); unweighted mean over the batch, like the NLL.
                  Early stopping and stored score = [mu, sigma] as M7.
Run names: perclos_ln_mlp_trunk_{beta|lnaux}_s<seed> (results/deep). The predictive CDF of the model = equal-weight
mixture of the three seed CDFs (as M7). Model ids: tc-mlpbeta_litc-trad+kf (A2), tc-mlplnaux_litc-trad+kf (A3).
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

SEEDS = (0, 1, 2)
SPEC = {"a2": dict(tag="beta", mid="tc-mlpbeta_litc-trad+kf"),
        "a3": dict(tag="lnaux", mid="tc-mlplnaux_litc-trad+kf")}
LAMBDA_AUX = 0.5


def patch(which: str):
    import torch
    import torch.nn.functional as F
    from scipy.special import betainc
    import deep_train_t1 as DT1
    if which == "a2":
        def ytarget(y):
            return np.clip(np.asarray(y, np.float64), DT1.CLIP_LO, DT1.CLIP_HI)

        def params(out):
            out = np.asarray(out, np.float64)
            return 1.0 + np.logaddexp(0.0, out[:, 0]), 1.0 + np.logaddexp(0.0, out[:, 1])

        def person_crps(a, b, y, groups):
            Fg = betainc(np.asarray(a, float)[:, None], np.asarray(b, float)[:, None], DT1.CRPS_GRID[None, :])
            ind = np.asarray(y, np.float64)[:, None] <= DT1.CRPS_GRID[None, :]
            c = np.mean((Fg - ind) ** 2, axis=1)
            return float(np.mean([c[groups == p].mean() for p in np.unique(groups)]))

        def nll(out, y):
            a = 1.0 + F.softplus(out[:, 0])
            b = 1.0 + F.softplus(out[:, 1])
            logpdf = ((a - 1) * torch.log(y) + (b - 1) * torch.log1p(-y)
                      - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))
            return -logpdf.mean()
        DT1.zlogit_np, DT1.ln_params_np, DT1.person_crps, DT1.ln_nll_torch = ytarget, params, person_crps, nll
    elif which == "a3":
        z1 = float(np.float32(DT1.zlogit_np(0.35)))
        z2 = float(np.float32(DT1.zlogit_np(0.70)))
        base_nll = DT1.ln_nll_torch

        def nll(out, z):
            mu = out[:, 0]
            sig = F.softplus(out[:, 1]) + 1e-3
            F1 = torch.special.ndtr((z1 - mu) / sig)
            F2 = torch.special.ndtr((z2 - mu) / sig)
            P = torch.stack([F1, (F2 - F1).clamp_min(0.0), 1.0 - F2], 1).clamp_min(1e-12)
            cls = (z >= z1).long() + (z >= z2).long()
            ce = -torch.log(P.gather(1, cls[:, None])[:, 0]).mean()
            return base_nll(out, z) + LAMBDA_AUX * ce
        DT1.ln_nll_torch = nll
    else:
        raise ValueError(which)
    return DT1


def run_name(tag, seed):
    return f"perclos_ln_mlp_trunk_{tag}_s{seed}"


def train(which: str, max_minutes: float | None) -> str:
    import t1_nets
    DT1 = patch(which)
    tag = SPEC[which]["tag"]
    t_end = time.time() + 60 * max_minutes if max_minutes else None
    for seed in SEEDS:
        if (C.DEEP / f"{run_name(tag, seed)}.npz").exists():
            continue
        left = None if t_end is None else (t_end - time.time()) / 60
        if left is not None and left < 1.0:
            return "paused"
        r = DT1.run_loso("perclos_ln", t1_nets.mlp_trunk, montage="trad6", norm="global", K=8, seed=seed,
                         out_dir=C.DEEP, max_minutes=left, save_weights=True, tag=f"{tag}_s{seed}",
                         instance_norm=False, target_stride=1, device="cpu")
        print(json.dumps({k: r[k] for k in ("status", "merged", "remaining")}), flush=True)
        if r["status"] != "complete":
            return "paused"
    return "complete"


def beta_outputs(As, Bs, y):
    from scipy.special import betainc
    Fcut = np.mean([betainc(a[:, None], b[:, None], np.array(C.CUTS)[None, :]) for a, b in zip(As, Bs)], 0)
    G = np.mean([betainc(a[:, None], b[:, None], C.GRID512[None, :]) for a, b in zip(As, Bs)], 0)
    mean = C.grid_mean(G)
    median = C.grid_quantile(G, 0.5)
    q = {lev: (C.grid_quantile(G, 0.5 - lev / 200), C.grid_quantile(G, 0.5 + lev / 200)) for lev in (50, 80, 90)}
    yc = np.clip(y, C.CLIP_LO, C.CLIP_HI)
    pit = np.mean([betainc(a, b, yc) for a, b in zip(As, Bs)], 0)
    return Fcut, G, mean, median, q, pit


def finalize(which: str) -> bool:
    import tc_models as TM
    tag, mid = SPEC[which]["tag"], SPEC[which]["mid"]
    paths = [C.DEEP / f"{run_name(tag, s)}.npz" for s in SEEDS]
    if not all(p.exists() for p in paths):
        return False
    b = C.base()
    p1, p2 = [], []
    for p in paths:
        with np.load(p) as z:
            assert np.array_equal(z["groups"], b["groups"]) and np.array_equal(z["rec"], b["rec"])
            p1.append(z["score"][:, 0].astype(np.float64))
            p2.append(z["score"][:, 1].astype(np.float64))
    y = b["perclos"]
    if which == "a2":
        Fcut, G, mean, median, q, pit = beta_outputs(p1, p2, y)
        C.finalize_tc(mid, Fcut, G, mean, median, q, params={"alpha": np.stack(p1, 1), "beta": np.stack(p2, 1)},
                      pit=pit, logscore=None)
    else:
        Fcut, G, mean, median, q, pit, ls = TM.mixture_ln_outputs(p1, p2, y)
        C.finalize_tc(mid, Fcut, G, mean, median, q, params={"mu": np.stack(p1, 1), "sigma": np.stack(p2, 1)},
                      pit=pit, logscore=ls)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=sorted(SPEC))
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    st = "complete" if a.finalize_only else train(a.which, a.max_minutes)
    if st == "complete" and finalize(a.which):
        C.log_compute(f"ablation_{a.which}", time.time() - t0, "tier 3, completed late (D14)")
    print(f"[{a.which}] {st}", flush=True)


if __name__ == "__main__":
    main()
