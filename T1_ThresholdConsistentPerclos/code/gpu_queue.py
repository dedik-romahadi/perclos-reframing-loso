"""GPU queue of T1 (one GPU process at a time; PREREG sections 15.3-15.4).

    python -I gpu_queue.py [--budget-h 4.5] [--gpu-start ISO]

Order: M11 sinc_tc -> M13 resnet_tc -> M12 sinc_three -> M14 resnet_three -> M12 sinc_binary -> M14 resnet_binary
(-> tier 3: A4 featgrukf_*). The feat_gru jobs (M9/M10) ran before this queue.
Adaptive rule (timing only, never outcomes): after fold 1 of M11 the remaining queue time is projected from
the measured fold time (resnet1d_gru = 67/169 of sinc_hybrid per fold from 00_COMMON SMOKE.md; direct jobs
assumed as long as the TC job; binary = 0.56 x three-class). If the projection exceeds the remaining budget
(budget minus GPU time already used), then in order and only as far as needed: (1) target_stride 4 for
M11-M14 (fold 1 of M11 rerun); (2) drop the binary direct jobs of M12 and M14; (3) drop M13/M14. The decision
is written to results/gpu_plan.json and compute_log.csv.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import t1_common as C  # noqa: E402
import run_neural as RN  # noqa: E402

PLAN = C.RES / "gpu_plan.json"
RATIO_RES = 67 / 169
RATIO_BIN = 0.56


def first_fold_seconds(stride):
    import numpy as np
    d = C.DEEP / f"perclos_ln_sinc_hybrid_st{stride}"
    f = sorted(d.glob("fold_*.npz"))
    if not f:
        return None
    with np.load(f[0]) as z:
        return json.loads(str(z["info"]))["wall_s"]


def decide(t_fold, remaining_h):
    jobs = {"sinc_tc": 1.0, "resnet_tc": RATIO_RES, "sinc_three": 1.0, "resnet_three": RATIO_RES,
            "sinc_binary": RATIO_BIN, "resnet_binary": RATIO_RES * RATIO_BIN}

    def proj(stride_factor, drop):
        tot = sum(v for k, v in jobs.items() if k not in drop) * 21 * t_fold * stride_factor
        return (tot - t_fold * stride_factor) / 3600.0      # fold 1 of M11 already done
    steps = [("none", 2, 1.0, []), ("stride4", 4, 0.5, []),
             ("stride4_drop_binary_direct", 4, 0.5, ["sinc_binary", "resnet_binary"]),
             ("stride4_drop_binary_direct_and_resnet", 4, 0.5,
              ["sinc_binary", "resnet_binary", "resnet_tc", "resnet_three"])]
    for name, stride, fac, drop in steps:
        p = proj(fac, drop)
        if p <= remaining_h:
            return {"step": name, "stride": stride, "drop": drop, "projected_h": round(p, 2)}
    name, stride, fac, drop = steps[-1]
    return {"step": name + "_still_over_budget", "stride": stride, "drop": drop, "projected_h": round(proj(fac, drop), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-h", type=float, default=4.5)
    ap.add_argument("--gpu-start", default=None, help="ISO time the T1 GPU queue started (feat_gru jobs)")
    a = ap.parse_args(argv)
    gpu_start = datetime.datetime.fromisoformat(a.gpu_start) if a.gpu_start else datetime.datetime.now()
    plan = json.loads(PLAN.read_text()) if PLAN.exists() else None
    if plan is None:
        import deep_train_t1 as DT1
        persons = sorted(set(int(p) for p in C.base()["persons"]))
        DT1.run_loso("perclos_ln", "sinc_hybrid", montage="trad6", norm="global", K=8, seed=0, out_dir=C.DEEP,
                     folds=[persons[0]], save_weights=True, tag="st2", instance_norm=False, target_stride=2)
        t_fold = first_fold_seconds(2)
        elapsed_h = (datetime.datetime.now() - gpu_start).total_seconds() / 3600
        remaining = a.budget_h - elapsed_h
        plan = decide(t_fold, remaining)
        plan.update({"fold1_seconds": t_fold, "gpu_elapsed_h": round(elapsed_h, 2), "remaining_h": round(remaining, 2),
                     "decided": datetime.datetime.now().isoformat(timespec="seconds")})
        PLAN.write_text(json.dumps(plan, indent=1))
        C.log_compute("gpu_adaptive_rule", 0, json.dumps(plan))
        print("GPU plan:", plan, flush=True)
        if plan["stride"] == 4:      # fold 1 of M11 is rerun at stride 4; the stride-2 fold is set aside
            src = C.DEEP / "perclos_ln_sinc_hybrid_st2"
            if src.exists():
                shutil.move(str(src), str(C.DEEP / "_discarded_perclos_ln_sinc_hybrid_st2_fold1"))
    s = plan["stride"]
    order = plan.get("order", ["sinc_tc", "resnet_tc", "sinc_three", "resnet_three", "sinc_binary", "resnet_binary"])
    hard_stop = datetime.datetime.fromisoformat(plan["hard_stop"]) if plan.get("hard_stop") else None
    for job in order:
        if job in plan["drop"] and not plan.get("run_dropped_if_time", False):
            print(f"[gpu] {job} dropped by the adaptive rule", flush=True)
            continue
        left = None
        if hard_stop is not None:
            left = (hard_stop - datetime.datetime.now()).total_seconds() / 60
            if left < 2:
                print(f"[gpu] {job} not started: hard stop reached", flush=True)
                continue
        t0 = time.time()
        st = RN.train(job, left, s)
        bb, kind, spec, _ = RN.run_spec(job, s)
        if st == "complete":
            RN.finalize(bb, kind, s)
            C.log_compute(f"neural_{job}", time.time() - t0, f"stride {s}")
        print(f"[gpu] {job}: {st}", flush=True)
    print("GPU_QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
