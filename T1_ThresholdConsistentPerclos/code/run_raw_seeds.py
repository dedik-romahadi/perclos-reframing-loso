"""Post-registration extension E2 (DEVIATIONS D15): extra seeds 1 and 2 for the raw-EOG networks M11-M14.

    python -I run_raw_seeds.py [--max-minutes M]        (one GPU process; resumable, repeat until DONE)
    python -I run_raw_seeds.py finalize

Same setting as the seed-0 runs in results/deep (run_neural.BACKBONE 'sinc' / 'resnet', T1 trainer copy, norm global,
K = 8, instance_norm False, target_stride 4 as decided for seed 0 (D4), GPU memory fraction for <= 1.5 GB); only the
seed differs. Queue order: all six jobs for seed 1, then for seed 2 (sinc_tc, resnet_tc, sinc_three, resnet_three,
sinc_binary, resnet_binary). Run names carry the tag 's<seed>_st4' (seed 0 keeps its original name, tag 'st4').
Finalised model ids (new files only; the seed-0 files used in every pre-registered test are not touched):
  <id>-seed1, <id>-seed2   single-seed versions
  <id>-ens3                3-seed version (TC: equal-weight mixture of the seed CDFs, as M7; direct: mean class
                           probabilities, argmax), built only when all three seeds exist.
Secondary, descriptive; not part of any pre-registered family.
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
import run_neural as RN  # noqa: E402

STRIDE = 4
SEEDS_NEW = (1, 2)
ORDER = ("sinc_tc", "resnet_tc", "sinc_three", "resnet_three", "sinc_binary", "resnet_binary")


def tag(seed):
    return f"st{STRIDE}" if seed == 0 else f"s{seed}_st{STRIDE}"


def merged_path(bb, kind, seed):
    spec = RN.BACKBONE[bb]
    return C.DEEP / f"{RN.run_name(RN.TASK[kind], spec['model'], tag(seed))}.npz"


def train_one(job, seed, left):
    import deep_train_t1 as DT1
    bb, kind = job.rsplit("_", 1)
    spec = RN.BACKBONE[bb]
    if merged_path(bb, kind, seed).exists():
        return "skip"
    hp = dict(montage="trad6", norm="global", K=8, seed=seed, out_dir=C.DEEP, max_minutes=left, save_weights=True,
              tag=tag(seed), instance_norm=False, target_stride=STRIDE)
    r = DT1.run_loso(RN.TASK[kind], spec["model"], **hp)
    print(json.dumps({"job": job, "seed": seed, **{k: r[k] for k in ("status", "merged", "remaining")}}), flush=True)
    return r["status"]


def finalize_job(bb, kind):
    import tc_models as TM
    spec = RN.BACKBONE[bb]
    mid = spec["tc_id"] if kind == "tc" else spec["direct_id"]
    b = C.base()
    loaded = {}
    for seed in (0,) + SEEDS_NEW:
        p = merged_path(bb, kind, seed)
        if p.exists():
            with np.load(p) as z:
                assert np.array_equal(z["groups"], b["groups"][_keep(b, kind)])
                assert np.array_equal(z["rec"], b["rec"][_keep(b, kind)])
                loaded[seed] = z["score"].astype(np.float64)
    made = []
    for seed in SEEDS_NEW:
        if seed in loaded:
            _write(kind, f"{mid}-seed{seed}", [loaded[seed]], b, TM)
            made.append(f"{mid}-seed{seed}")
    if len(loaded) == 3:
        _write(kind, f"{mid}-ens3", [loaded[s] for s in (0, 1, 2)], b, TM)
        made.append(f"{mid}-ens3")
    return made


def _keep(b, kind):
    return b["yi"] != 2 if kind == "binary" else np.ones(len(b["yi"]), bool)


def _write(kind, mid, scores, b, TM):
    if kind == "tc":
        mus = [s[:, 0] for s in scores]
        sig = [s[:, 1] for s in scores]
        y = b["perclos"]
        if len(scores) == 1:
            Fcut, G, mean, median, q, pit, ls = TM.ln_outputs(mus[0], sig[0], y)
        else:
            Fcut, G, mean, median, q, pit, ls = TM.mixture_ln_outputs(mus, sig, y)
        C.finalize_tc(mid, Fcut, G, mean, median, q, params={"mu": np.stack(mus, 1), "sigma": np.stack(sig, 1)},
                      pit=pit, logscore=ls)
        return
    keep = _keep(b, kind)
    sc = np.mean(scores, 0)
    full_s = np.zeros((len(b["yi"]), sc.shape[1]))
    full_s[keep] = sc
    full_p = np.full(len(b["yi"]), -1)
    full_p[keep] = sc.argmax(1)
    C.save_preds(kind, mid, full_p, full_s)


def finalize_all():
    made = []
    for job in ORDER:
        bb, kind = job.rsplit("_", 1)
        made += finalize_job(bb, kind)
    print("finalized:", made, flush=True)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="train", choices=["train", "finalize"])
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    if a.what == "finalize":
        finalize_all()
        return
    t_end = time.time() + 60 * a.max_minutes if a.max_minutes else None
    for seed in SEEDS_NEW:
        for job in ORDER:
            left = None if t_end is None else (t_end - time.time()) / 60
            if left is not None and left < 2:
                print("[raw-seeds] time limit; resume later", flush=True)
                return
            t0 = time.time()
            st = train_one(job, seed, left)
            if st == "skip":
                continue
            if st != "complete":
                print(f"[raw-seeds] {job} seed {seed}: {st}; resume later", flush=True)
                return
            C.log_compute(f"neural_{job}_seed{seed}", time.time() - t0, f"stride {STRIDE}, extension E2")
            bb, kind = job.rsplit("_", 1)
            finalize_job(bb, kind)
    print("RAW_SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()
