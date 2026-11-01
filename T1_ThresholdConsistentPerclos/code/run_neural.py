"""Neural models of T1 (M7-M14) through the T1 trainer copy (deep_train_t1.run_loso), then conversion
to T1 outputs (CDF / grid / canonical prediction files).

    python -I run_neural.py <job> [--max-minutes 9] [--stride S]
    python -I run_neural.py finalize            (convert every merged run that exists)

Jobs (PREREG section 4; paired TC and direct versions use identical settings):
  mlp_tc, mlp_three, mlp_binary            M7 / M8: per-epoch MLP trunk on litc-trad+kf, CPU, seeds 0, 1, 2
  featgru_tc, featgru_three, featgru_binary   M9 / M10: registry feat_gru (litc-trad, K = 8), GPU, seed 0
  sinc_tc, sinc_three, sinc_binary         M11 / M12: registry sinc_hybrid, raw trad6 causal 0.1-30 Hz,
                                           norm global, K = 8, litc-trad injected, GPU, target_stride 2
  resnet_tc, resnet_three, resnet_binary   M13 / M14: registry resnet1d_gru, same input, GPU, target_stride 2
  featgrukf_tc, featgrukf_three, featgrukf_binary   A4 ablation (feat_set litc-trad+kf), GPU
All runs: instance_norm False, out_dir results/deep. The run name carries the tag 'st<s>' when s > 1.
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

TASK = {"tc": "perclos_ln", "three": "three", "binary": "binary"}
BACKBONE = {
    "mlp": dict(model="mlp_trunk", seeds=(0, 1, 2), device="cpu", stride=1, cls="hybrid", input="litc-trad+kf",
                tc_id="tc-mlpln_litc-trad+kf", direct_id="direct-mlp_litc-trad+kf"),
    "featgru": dict(model="feat_gru", seeds=(0,), device=None, stride=1, cls="hybrid", input="litc-trad",
                    tc_id="tc-featgru_litc-trad", direct_id="direct-featgru_litc-trad"),
    "featgrukf": dict(model="feat_gru", seeds=(0,), device=None, stride=1, cls="hybrid", input="litc-trad+kf",
                      tc_id="tc-featgru_litc-trad+kf", direct_id="direct-featgru_litc-trad+kf",
                      extra={"feat_set": "litc-trad+kf"}, tag="kf"),
    "sinc": dict(model="sinc_hybrid", seeds=(0,), device=None, stride=2, cls="hybrid_e2e",
                 input="raw trad6 causal 0.1-30 Hz, norm global, K=8, litc-trad injected",
                 tc_id="tc-sinchybrid_trad6", direct_id="direct-sinchybrid_trad6"),
    "resnet": dict(model="resnet1d_gru", seeds=(0,), device=None, stride=2, cls="dl_e2e",
                   input="raw trad6 causal 0.1-30 Hz, norm global, K=8",
                   tc_id="tc-resnet1dgru_trad6", direct_id="direct-resnet1dgru_trad6"),
}


def run_spec(job: str, stride: int | None = None):
    bb, kind = job.rsplit("_", 1)
    spec = dict(BACKBONE[bb])
    s = int(stride or spec["stride"])
    return bb, kind, spec, s


def tag_for(seed: int, s: int, nseeds: int, extra: str | None = None) -> str | None:
    parts = [extra] if extra else []
    if nseeds > 1:
        parts.append(f"s{seed}")
    if s > 1:
        parts.append(f"st{s}")
    return "_".join(parts) or None


def run_name(task, model_name, tag):
    return f"{task}_{model_name}" + (f"_{tag}" if tag else "")


def model_arg(spec):
    if spec["model"] == "mlp_trunk":
        import t1_nets
        return t1_nets.mlp_trunk, "mlp_trunk"
    return spec["model"], spec["model"]


def train(job: str, max_minutes: float | None, stride: int | None = None) -> str:
    import deep_train_t1 as DT1
    bb, kind, spec, s = run_spec(job, stride)
    task = TASK[kind]
    m, mname = model_arg(spec)
    status = "complete"
    t_end = time.time() + 60 * max_minutes if max_minutes else None
    for seed in spec["seeds"]:
        tag = tag_for(seed, s, len(spec["seeds"]), spec.get("tag"))
        merged = C.DEEP / f"{run_name(task, mname, tag)}.npz"
        if merged.exists():
            continue
        left = None if t_end is None else (t_end - time.time()) / 60
        if left is not None and left < 1.0:
            return "paused"
        hp = dict(montage="trad6", norm="global", K=8, seed=seed, out_dir=C.DEEP, max_minutes=left,
                  save_weights=True, tag=tag, instance_norm=False, target_stride=s)
        hp.update(spec.get("extra", {}))
        if spec["device"]:
            hp["device"] = spec["device"]
        r = DT1.run_loso(task, m, **hp)
        print(json.dumps({k: r[k] for k in ("status", "merged", "remaining")}), flush=True)
        if r["status"] != "complete":
            status = "paused"
            break
    return status


def _merged_paths(bb, kind, s):
    spec = BACKBONE[bb]
    _, mname = (spec["model"], "mlp_trunk") if spec["model"] == "mlp_trunk" else (spec["model"], spec["model"])
    return [C.DEEP / f"{run_name(TASK[kind], mname, tag_for(seed, s, len(spec['seeds']), spec.get('tag')))}.npz"
            for seed in spec["seeds"]]


def finalize(bb: str, kind: str, s: int) -> bool:
    import tc_models as TM
    spec = BACKBONE[bb]
    paths = _merged_paths(bb, kind, s)
    if not all(p.exists() for p in paths):
        return False
    b = C.base()
    if kind == "tc":
        mus, sigs = [], []
        for p in paths:
            with np.load(p) as z:
                assert np.array_equal(z["groups"], b["groups"]) and np.array_equal(z["rec"], b["rec"])
                mus.append(z["score"][:, 0].astype(np.float64))
                sigs.append(z["score"][:, 1].astype(np.float64))
        y = b["perclos"]
        if len(paths) == 1:
            Fcut, G, mean, median, q, pit, ls = TM.ln_outputs(mus[0], sigs[0], y)
        else:
            Fcut, G, mean, median, q, pit, ls = TM.mixture_ln_outputs(mus, sigs, y)
        C.finalize_tc(spec["tc_id"], Fcut, G, mean, median, q,
                      params={"mu": np.stack(mus, 1), "sigma": np.stack(sigs, 1)}, pit=pit, logscore=ls)
        return True
    # direct: average class probabilities over seeds; argmax decision (class-balanced CE)
    scores = []
    for p in paths:
        with np.load(p) as z:
            keep = np.ones(len(b["yi"]), bool) if kind == "three" else b["yi"] != 2
            assert np.array_equal(z["groups"], b["groups"][keep]) and np.array_equal(z["rec"], b["rec"][keep])
            assert np.array_equal(z["t"], b["t"][keep])
            scores.append(z["score"].astype(np.float64))
    sc = np.mean(scores, 0)
    full_s = np.zeros((len(b["yi"]), sc.shape[1]))
    full_s[keep] = sc
    full_p = np.full(len(b["yi"]), -1)
    full_p[keep] = sc.argmax(1)
    C.save_preds(kind, spec["direct_id"], full_p, full_s)
    return True


def finalize_all():
    done = []
    plan = C.RES / "gpu_plan.json"
    raw_stride = json.loads(plan.read_text())["stride"] if plan.exists() else None
    for bb, spec in BACKBONE.items():
        for kind in ("tc", "three", "binary"):
            s = raw_stride if (raw_stride and bb in ("sinc", "resnet")) else spec["stride"]
            try:
                if finalize(bb, kind, s):
                    done.append(f"{bb}_{kind}")
            except Exception as e:  # noqa: BLE001
                print(f"finalize {bb}_{kind} failed: {e}", flush=True)
    print("finalized:", done, flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    a = ap.parse_args(argv)
    if a.job == "finalize":
        finalize_all()
        return
    t0 = time.time()
    st = train(a.job, a.max_minutes, a.stride)
    bb, kind, spec, s = run_spec(a.job, a.stride)
    if st == "complete":
        finalize(bb, kind, s)
        C.log_compute(f"neural_{a.job}", time.time() - t0, f"stride {s}")
    print(f"[{a.job}] {st}", flush=True)


if __name__ == "__main__":
    main()
