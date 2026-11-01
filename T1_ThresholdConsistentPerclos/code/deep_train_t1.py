"""T1 COPY of 00_COMMON/deep_train.py (copied 28 Oct 2026, source SHA-256
2e911f95419be7edd599fe40801c27a1fcd51721a51808e8ab3f6258273d4ad4), extended only as prespecified in
T1 PREREG section 4.5:
  (a) task 'perclos_ln': the network's 2 outputs are (mu, s) of a logit-normal predictive distribution of
      PERCLOS, sigma = softplus(s) + 1e-3; loss = logit-normal NLL of z = logit(clip(y, 0.005, 0.995)) on
      ALL epochs (unweighted); early stopping on the LOWEST mean per-person CRPS of the inner-validation
      persons (CRPS on a 256-point grid, raw y); inner-validation persons drawn as for 'three'.
      Fold / merged files store score = [mu, sigma] (pred = 0).
  (b) training option target_stride s (TRAIN_DEFAULTS): only epochs with index mod s = 0 within each
      fitting recording are training targets; context windows, inner-validation and test rows use all
      epochs.
Everything else is the shared protocol, unchanged. Default output folder = T1/results/deep. The SHA-256
recorded in the run JSON covers 00_COMMON/deep_data.py, 00_COMMON/deep_models.py and this file.

Original docstring follows.

Leave-one-PERSON-out trainer for the deep / hybrid models (shared by all five topics).

Library:
    import deep_train as DT
    DT.run_loso("three", "resnet1d_gru", montage="trad6", norm="trailing", K=8, seed=0,
                out_dir=".../T3_GainInvariantEOG/results/deep", folds=None, max_minutes=9,
                input_fn=None, tag=None, instance_norm=True)          # extra kw = hyper-parameters
    DT.causal_smooth_scores(".../three_resnet1d_gru.npz", q_over_r=0.01)  # -> three_resnet1d_gru+kf.npz
CLI (run with python -I; repeat the same call until it reports 'complete'):
    python -I deep_train.py --task three --model eegnet [--montage trad6] [--norm trailing] [--K 8]
        [--seed 0] [--instance-norm] [--out DIR] [--folds 3,7] [--max-minutes 9] [--tag TAG]
        [--hp key=value ...] [--no-weights] [--kf] [--cpu]
    python -I deep_train.py --smooth FILE.npz [--q 0.01]

`model` is a REGISTRY name (deep_models) or a factory callable(n_chans, n_times, n_classes) -> nn.Module.
A custom module may subclass deep_models.DeepModel (attributes seq_len, uses_raw, feat_set, n_feat and
forward_with_emb(x, mask, feats)); a plain module is treated as a single-epoch model forward(x) -> logits
(no embedding saved). input_fn(X, meta, split) -> X' transforms the RAW filtered input (float32
(n, C, 1000), before normalisation) of all epochs of the persons of one split; split is 'train'
(fitting persons), 'val' (the inner early-stopping persons, also training persons) or 'test' (the
held-out person). meta: rows (global row indices), groups, rec, t, yi, perclos, names_rec, ch_names,
montage, fs, test_person. It may change the channel count (e.g. append a mask channel).

Protocol (fixed for the programme):
  * outer LOSO over the 21 persons; seedvig_eog.assert_no_person_overlap for train/val/test.
  * inner validation = n_inner_val (3) training persons, drawn with numpy default_rng([seed, person]);
    for 'binary' only persons with >= 20 alert and >= 20 drowsy epochs. Used ONLY for early stopping
    (max_epochs 40, patience 6, monitor = mean of per-person balanced accuracy over the inner persons).
    The weights of the best inner epoch are kept; the model is NOT retrained on the inner persons
    (they are simply not used for fitting in that fold).
  * every fitted step uses the fitting persons only: 'global' input scale, feature imputation
    (median) and z-scoring (+ clip at +/-8) for injected handcrafted features, class weights.
  * class-balanced cross-entropy (w_c = n / (C n_c) on the fitting targets); AdamW (lr 1e-3, weight
    decay 1e-2); cosine annealing over max_epochs (schedule='plateau' or 'none' available); batch 256
    targets per optimiser step, split into micro-batches (gradient accumulation) where a model needs it
    to stay under the GPU cap (deep_models.REGISTRY[...]['train']); AMP float16 on GPU (sinc filtering,
    descriptors and STFT run in float32); torch.set_num_threads(2); GPU memory capped at 1.5 GB per
    process (torch.cuda.set_per_process_memory_fraction); seeds = (seed, person); cuDNN deterministic
    and torch.use_deterministic_algorithms(warn_only=True). Results on CPU and GPU are not bitwise equal.
  * binary task: tired epochs are dropped as TARGETS (train and test), but stay in the input context of
    sequence models. Test predictions are made for all epochs of the test person; the binary merged file
    also stores them (all_score / all_rec / all_keep) so that causal_smooth_scores smooths in real time.
Outputs (resumable; folds already done are skipped, an interrupted fold resumes from its checkpoint):
  DIR/<task>_<model>[_<tag>]/fold_<person>.npz (+ fold_<person>.pt float16 best weights)
  DIR/<task>_<model>[_<tag>].npz  keys y, pred, score, groups, rec, perclos, t, emb (float16) [+ all_*]
  DIR/<task>_<model>[_<tag>].json settings, per-fold stopping epoch / inner BA / wall time / GPU peak,
                                  SHA-256 of deep_data.py, deep_models.py, deep_train.py.
Changing any setting that affects results under the same name raises an error: use another --tag.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent            # T1/code
COMMON = HERE.parents[1] / "00_COMMON"
for _p in (str(HERE), str(COMMON)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import deep_data as DD  # noqa: E402
import deep_models as DM  # noqa: E402
import eog_features_lit as L  # noqa: E402
import seedvig_eog as S  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

warnings.filterwarnings("once", category=UserWarning)
warnings.filterwarnings("ignore", message="A window was not provided")       # BIOT's STFT, as in the original

DEFAULT_OUT = HERE.parent / "results" / "deep"
MIN_PER_CLASS = 20
TRAIN_DEFAULTS = dict(lr=1e-3, weight_decay=1e-2, batch_size=256, micro_batch=None, max_epochs=40, patience=6,
                      schedule="cosine", n_inner_val=3, amp=True, filt="causal", band=(0.1, 30.0),
                      eval_batch=None, gpu_mem_gb=1.5, device=None, feat_clip=8.0, pause_after=None,
                      target_stride=1)
# keys that do not change results (excluded from the configuration fingerprint); pause_after = stop
# (with a checkpoint) after this many epochs in the current call - used by tests/test_deep.py
_NON_SEMANTIC = ("eval_batch", "gpu_mem_gb", "device", "pause_after")
CODE_FILES = ("deep_data.py", "deep_models.py", "deep_train_t1.py")
CLIP_LO, CLIP_HI = 0.005, 0.995
CRPS_GRID = (np.arange(256) + 0.5) / 256


# ============================================================================== helpers
def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {f: sha256((HERE if f == "deep_train_t1.py" else COMMON) / f) for f in CODE_FILES}


def zlogit_np(y):
    y = np.clip(np.asarray(y, np.float64), CLIP_LO, CLIP_HI)
    return np.log(y) - np.log1p(-y)


def ln_params_np(out):
    """(mu, sigma) from the 2 network outputs (numpy)."""
    out = np.asarray(out, np.float64)
    return out[:, 0], np.logaddexp(0.0, out[:, 1]) + 1e-3


def ln_crps_np(mu, sigma, y):
    from scipy.special import ndtr
    lg = np.log(CRPS_GRID) - np.log1p(-CRPS_GRID)
    F = ndtr((lg[None, :] - mu[:, None]) / sigma[:, None])
    ind = np.asarray(y, np.float64)[:, None] <= CRPS_GRID[None, :]
    return np.mean((F - ind) ** 2, axis=1)


def person_crps(mu, sigma, y, groups) -> float:
    c = ln_crps_np(mu, sigma, y)
    return float(np.mean([c[groups == p].mean() for p in np.unique(groups)]))


def ln_nll_torch(out, z):
    mu = out[:, 0]
    sigma = F.softplus(out[:, 1]) + 1e-3
    return (torch.log(sigma) + 0.5 * ((z - mu) / sigma) ** 2).mean() + 0.5 * math.log(2 * math.pi)


def setup_torch(device=None, gpu_mem_gb: float = 1.5, threads: int = 2) -> torch.device:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_mem_gb * 2 ** 30 / total), dev)
    return dev


def seed_everything(seed: int, person: int) -> None:
    s = int(seed) * 1000 + int(person)
    torch.manual_seed(s)
    np.random.seed(s % (2 ** 32))


def balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    cls = np.unique(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in cls]))


def person_ba(y, pred, groups) -> float:
    return float(np.mean([balanced_acc(y[groups == p], pred[groups == p]) for p in np.unique(groups)]))


def inner_val_persons(groups, y, keep, test_person, seed=0, task="three", n_val=3) -> list[int]:
    """Deterministic inner early-stopping persons (training persons only; see module doc)."""
    cand = []
    for p in np.unique(groups):
        if p == test_person:
            continue
        m = (groups == p) & keep
        if task == "binary" and min((y[m] == 0).sum(), (y[m] == 1).sum()) < MIN_PER_CLASS:
            continue
        cand.append(int(p))
    rng = np.random.default_rng([int(seed), int(test_person)])
    return sorted(int(v) for v in rng.choice(cand, size=n_val, replace=False))


def is_deployable_run(filt: str, norm: str, feat_set: str | None, input_fn) -> bool | None:
    import feature_sets as FS
    ok = DD.is_deployable(filt, norm) and (feat_set is None or FS.is_deployable(feat_set))
    if input_fn is not None and ok:
        return None          # depends on the topic's input_fn: the topic must declare it
    return bool(ok)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if callable(v):
        return getattr(v, "__qualname__", repr(v))
    return v


# ============================================================================== data per split
def _prepare_features(Fall, fit_rows, clip):
    med = np.nanmedian(Fall[fit_rows], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Fi = np.where(np.isnan(Fall), med[None, :], Fall)
    mu = Fi[fit_rows].mean(0)
    sd = Fi[fit_rows].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return np.clip((Fi - mu) / sd, -clip, clip).astype(np.float32)


def _raw_split(Xsrc, rows, rec_rows, norm, input_fn, meta, split, device, store_dtype, global_scale=None):
    """Normalised raw input of one split as a device tensor (n, C', T). Returns (tensor, fitted
    global scale or None)."""
    segs = DD._segments(rec_rows)
    if input_fn is not None:
        Xs = np.asarray(input_fn(Xsrc[rows].astype(np.float32), meta, split), dtype=np.float32)
        if Xs.shape[0] != len(rows):
            raise ValueError("input_fn must keep the number of epochs")
        get = lambda a, b: Xs[a:b]  # noqa: E731
    else:
        get = lambda a, b: Xsrc[rows[a:b]].astype(np.float32)  # noqa: E731
    if norm == "global" and global_scale is None:
        rsd = np.concatenate([DD.epoch_robust_sd(get(a, b)) for a, b in segs])
        global_scale = np.maximum(np.median(rsd, axis=0), DD.SCALE_FLOOR).astype(np.float32)
    first = get(*segs[0])
    out = torch.empty((len(rows),) + first.shape[1:], dtype=store_dtype, device=device)
    np_dt = np.float16 if store_dtype == torch.float16 else np.float32
    for a, b in segs:
        Xr = first if a == segs[0][0] else get(a, b)
        if norm == "global":
            c, s = None, np.broadcast_to(global_scale, (b - a, Xr.shape[1]))
        else:
            c, s = DD.norm_params(Xr, np.zeros(b - a, int), norm)
        out[a:b] = torch.from_numpy(DD.apply_norm(Xr, c, s, np_dt)).to(device)
    return out, global_scale


class _Split:
    pass


def _build_split(D, Xsrc, rows, split, y, keep, Fz, K, uses_raw, norm, input_fn, device, store_dtype, test_person,
                 global_scale=None, yfloat=None):
    sp = _Split()
    sp.rows = rows
    sp.rec = D["rec"][rows]
    sp.groups = D["groups"][rows]
    sp.y_np = y[rows]
    sp.tgt = np.flatnonzero(keep[rows])
    if yfloat is None:
        sp.y = torch.as_tensor(np.where(sp.y_np >= 0, sp.y_np, 0), dtype=torch.long, device=device)
    else:
        sp.y = torch.as_tensor(yfloat[rows], dtype=torch.float32, device=device)
    sp.yraw = D["perclos"][rows]
    idx, mask = DD.seq_index(sp.rec, K)
    sp.idx = torch.as_tensor(idx, device=device)
    sp.mask = torch.as_tensor(mask, device=device)
    sp.F = torch.as_tensor(Fz[rows], device=device) if Fz is not None else None
    sp.X, gs = None, global_scale
    if uses_raw:
        meta = {"rows": rows, "groups": sp.groups, "rec": sp.rec, "t": D["t"][rows], "yi": D["yi"][rows],
                "perclos": D["perclos"][rows], "names_rec": D["names_rec"], "ch_names": D["ch_names"],
                "montage": D["montage"], "fs": D["fs"], "test_person": test_person}
        sp.X, gs = _raw_split(Xsrc, rows, sp.rec, norm, input_fn, meta, split, device, store_dtype, global_scale)
    return sp, gs


def _batch(sp, loc, K, uses_raw):
    t = torch.as_tensor(loc, device=sp.y.device)
    if K > 1:
        idx = sp.idx[t]
        mask = sp.mask[t]
        x = sp.X[idx].float() if uses_raw else None
        f = sp.F[idx] if sp.F is not None else None
    else:
        mask = None
        x = sp.X[t].float() if uses_raw else None
        f = sp.F[t] if sp.F is not None else None
    return x, mask, f, sp.y[t]


def _fwd(model, x, mask, f):
    if hasattr(model, "forward_with_emb"):
        return model.forward_with_emb(x, mask, f)
    return model(x), None


@torch.no_grad()
def _predict(model, sp, loc, K, uses_raw, eval_batch, amp, want_emb=False):
    model.eval()
    dev = sp.y.device
    lo_all, em_all = [], []
    for a in range(0, len(loc), eval_batch):
        x, mask, f, _ = _batch(sp, loc[a:a + eval_batch], K, uses_raw)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            lo, em = _fwd(model, x, mask, f)
        lo_all.append(lo.float().cpu())
        if want_emb and em is not None:
            em_all.append(em.float().cpu())
    logits = torch.cat(lo_all).numpy()
    emb = torch.cat(em_all).numpy() if em_all else None
    return logits, emb


# ============================================================================== model resolution
def _resolve_model(model, model_kw):
    """-> (name, builder(n_chans, n_times, n_classes, n_feat) -> module, spec dict)."""
    if isinstance(model, str):
        spec = DM.model_spec(model, **model_kw)
        name = model

        def builder(n_chans, n_times, n_classes, n_feat, K):
            kw = dict(model_kw)
            if spec["seq"]:
                kw["K"] = K
            if spec["feat_set"]:
                kw.update(n_feat=n_feat, feat_set=spec["feat_set"])
            return DM.build(model, n_chans, n_times, n_classes, **kw)
        return name, builder, spec
    if not callable(model):
        raise TypeError("model must be a registry name or a factory callable")
    name = model_kw.pop("model_name", None) or getattr(model, "__name__", "custom")
    probe = model(6, DD.N_SAMP, 3)
    seq_len = int(getattr(probe, "seq_len", 1))
    spec = {"model_class": getattr(probe, "model_class", "custom"), "seq": seq_len > 1,
            "feat_set": getattr(probe, "feat_set", None), "train": {}, "seq_len": seq_len,
            "uses_raw": bool(getattr(probe, "uses_raw", True))}
    del probe

    def builder(n_chans, n_times, n_classes, n_feat, K):
        return model(n_chans, n_times, n_classes)
    return name, builder, spec


# ============================================================================== main entry
def run_loso(task: str, model, montage: str = "trad6", norm: str = "trailing", K: int = 8, seed: int = 0,
             out_dir=None, folds=None, max_minutes: float | None = None, save_weights: bool = True,
             input_fn=None, tag: str | None = None, verbose: bool = True, **hparams) -> dict:
    """Outer LOSO for one model / task / setting (see the module docstring). Returns a status dict:
    status ('complete' | 'partial' | 'paused'), run_dir, merged (path or None), done, remaining."""
    t_start = time.time()
    deadline = t_start + 60 * max_minutes if max_minutes else math.inf
    hp = dict(hparams)
    model_kw = {k: v for k, v in hp.items() if k not in TRAIN_DEFAULTS}
    name, builder, spec = _resolve_model(model, model_kw)
    train = dict(TRAIN_DEFAULTS)
    train.update(spec.get("train", {}))
    train.update({k: v for k, v in hp.items() if k in TRAIN_DEFAULTS})
    if train["micro_batch"] is None:
        train["micro_batch"] = train["batch_size"]
    train["band"] = tuple(float(b) for b in train["band"])
    K_eff = int(spec.get("seq_len", K if spec["seq"] else 1))
    feat_set = spec["feat_set"]
    uses_raw = bool(spec.get("uses_raw", True))
    if task not in ("three", "binary", "perclos_ln"):
        raise ValueError(task)
    n_classes = 3 if task == "three" else 2
    out_dir = pathlib.Path(out_dir) if out_dir else DEFAULT_OUT
    run = f"{task}_{name}" + (f"_{tag}" if tag else "")
    run_dir = out_dir / run
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"task": task, "model": name, "model_class": spec["model_class"], "montage": montage, "norm": norm,
              "K": K_eff, "seed": int(seed), "tag": tag, "feat_set": feat_set, "model_kw": _jsonable(model_kw),
              "train": {k: _jsonable(v) for k, v in train.items() if k not in _NON_SEMANTIC},
              "input_fn": _jsonable(input_fn) if input_fn is not None else None}
    cfg_json = json.dumps(config, sort_keys=True)
    for f in sorted(run_dir.glob("fold_*.npz")):
        with np.load(f) as zf:
            old = str(zf["config"])
        if old != cfg_json:
            raise RuntimeError(f"{f} was produced with a different configuration:\n{old}\nnow:\n{cfg_json}\n"
                               "use another tag or output folder")

    dev = setup_torch(train["device"], train["gpu_mem_gb"])
    amp = bool(train["amp"]) and dev.type == "cuda"
    store_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    eval_batch = train["eval_batch"] or (max(2 * train["micro_batch"], 128) if K_eff > 1 else 512)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    if uses_raw:
        D = DD.load_raw(montage=montage, filt=train["filt"], band=train["band"], dtype=np.float16)
    else:                                               # feature-only model: metadata only
        D = _meta_only(montage)
    if task == "perclos_ln":
        y, keep = np.zeros(len(D["yi"]), int), np.ones(len(D["yi"]), bool)
        D["zlogit"] = zlogit_np(D["perclos"])
    else:
        y, keep = DD.task_labels(D["yi"], task)
    groups = D["groups"]
    Fall = None
    if feat_set:
        import feature_sets as FS
        Fd = FS.load(feat_set)
        for k in ("groups", "rec", "yi"):
            assert np.array_equal(Fd[k], D[k]), f"feature set {feat_set} rows do not match the raw data ({k})"
        Fall = Fd["X"].astype(np.float32)
    persons = [int(p) for p in np.unique(groups)]
    todo = persons if folds is None else [int(p) for p in folds]
    for p in todo:
        if p not in persons:
            raise ValueError(f"unknown person {p}")

    for p in todo:
        f_out = run_dir / f"fold_{p}.npz"
        if f_out.exists():
            continue
        if time.time() + 30 > deadline:
            log(f"[{run}] time limit reached before fold {p}")
            break
        res = _run_fold(p, run, run_dir, D, y, keep, Fall, task, n_classes, montage, norm, K_eff, seed, train,
                        builder, spec, uses_raw, input_fn, dev, amp, store_dtype, eval_batch, deadline, save_weights,
                        cfg_json, log)
        if res == "paused":
            break

    done = sorted(int(f.stem.split("_")[1]) for f in run_dir.glob("fold_*.npz"))
    remaining = [p for p in persons if p not in done]
    merged = None
    if not remaining:
        merged = _merge(run, run_dir, out_dir, D, y, keep, task, config, spec, feat_set, input_fn, train, montage,
                        norm, filt=train["filt"])
        status = "complete"
    else:
        status = "paused" if any(p in remaining for p in todo) else "partial"
    log(f"[{run}] {status}: {len(done)}/21 folds done ({time.time() - t_start:.0f} s this call)")
    return {"status": status, "run_dir": str(run_dir), "merged": str(merged) if merged else None, "done": done,
            "remaining": remaining}


def _meta_only(montage):
    meta = {k: [] for k in ("yi", "perclos", "groups", "rec", "t")}
    for i, nm in enumerate(S.RECORDINGS):
        z = np.load(S.CACHE_DIR / f"{nm}.npz")
        meta["yi"].append(z["yi"])
        meta["perclos"].append(z["perclos"])
        meta["groups"].append(np.full(S.N_EPOCH, S.person_of(nm)))
        meta["rec"].append(np.full(S.N_EPOCH, i))
        meta["t"].append(np.arange(S.N_EPOCH, dtype=np.float64) * S.EPOCH_S)
    out = {k: np.concatenate(v) for k, v in meta.items()}
    out.update({"X": None, "names_rec": list(S.RECORDINGS), "ch_names": DD.montage_channels(montage),
                "montage": montage, "fs": DD.FS})
    return out


def _run_fold(p, run, run_dir, D, y, keep, Fall, task, n_classes, montage, norm, K, seed, train, builder, spec,
              uses_raw, input_fn, dev, amp, store_dtype, eval_batch, deadline, save_weights, cfg_json, log):
    t0 = time.time()
    groups = D["groups"]
    ln = task == "perclos_ln"
    val_p = inner_val_persons(groups, y, keep, p, seed, "three" if ln else task, train["n_inner_val"])
    test_m = groups == p
    val_m = np.isin(groups, val_p)
    fit_m = ~test_m & ~val_m
    S.assert_no_person_overlap(groups[fit_m], groups[test_m])
    S.assert_no_person_overlap(groups[val_m], groups[test_m])
    S.assert_no_person_overlap(groups[fit_m], groups[val_m])
    rows = {k: np.flatnonzero(m) for k, m in (("train", fit_m), ("val", val_m), ("test", test_m))}
    Fz = _prepare_features(Fall, rows["train"], train["feat_clip"]) if Fall is not None else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    sps, gs = {}, None
    for split in ("train", "val", "test"):
        sps[split], gs = _build_split(D, D["X"], rows[split], split, y, keep, Fz, K, uses_raw, norm, input_fn, dev,
                                      store_dtype, p, gs, yfloat=D["zlogit"] if ln else None)
    fit, val, test = sps["train"], sps["val"], sps["test"]
    stride = int(train.get("target_stride", 1) or 1)
    if stride > 1:
        ep_idx = np.rint(D["t"][fit.rows] / S.EPOCH_S).astype(int)
        fit.tgt = fit.tgt[ep_idx[fit.tgt] % stride == 0]
    n_chans = fit.X.shape[1] if uses_raw else len(D["ch_names"])
    seed_everything(seed, p)
    model = builder(n_chans, DD.N_SAMP, n_classes, Fall.shape[1] if Fall is not None else 0, K).to(dev)
    n_params = int(sum(q.numel() for q in model.parameters() if q.requires_grad))
    opt = torch.optim.AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    if train["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train["max_epochs"])
    elif train["schedule"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    else:
        sched = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ytr = fit.y_np[fit.tgt]
    cnt = np.bincount(ytr, minlength=n_classes).astype(float)
    if ln:
        cnt = np.ones(n_classes)
    cw = torch.tensor(np.where(cnt > 0, len(ytr) / (n_classes * np.maximum(cnt, 1)), 0.0), dtype=torch.float32,
                      device=dev)
    rng = np.random.default_rng([int(seed), int(p), 7])
    st = {"epoch": 0, "best_ba": -np.inf, "best_epoch": 0, "bad": 0, "best_state": None, "history": [],
          "wall_prev": 0.0}
    ckpt = run_dir / f"fold_{p}.ckpt.pt"
    if ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        if c["config"] != cfg_json:
            raise RuntimeError(f"{ckpt} has a different configuration")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        scaler.load_state_dict(c["scaler"])
        if sched is not None and c["sched"] is not None:
            sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]
        torch.set_rng_state(c["torch_rng"])
        if dev.type == "cuda" and c.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(c["cuda_rng"], dev)
        st = c["state"]
        log(f"[{run}] fold {p}: resumed after epoch {st['epoch']}")
    bs, mb = int(train["batch_size"]), int(train["micro_batch"])
    prep_s = time.time() - t0
    log(f"[{run}] fold {p}: inner val {val_p}, fit targets {len(fit.tgt)}, params {n_params}, prep {prep_s:.1f} s")
    ep_times = []
    finished = False
    while st["epoch"] < train["max_epochs"]:
        te = time.time()
        model.train()
        perm = rng.permutation(fit.tgt)
        n_full = len(perm) // bs
        rem = len(perm) - n_full * bs
        n_steps = n_full + (1 if rem >= 32 else 0)
        tot = torch.zeros((), device=dev)
        for s_ in range(n_steps):
            bidx = perm[s_ * bs:(s_ + 1) * bs]
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(bidx), mb):
                mbidx = bidx[j:j + mb]
                if len(mbidx) < 2:
                    continue
                x, mask, f, yb = _batch(fit, mbidx, K, uses_raw)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    logits, _ = _fwd(model, x, mask, f)
                if ln:
                    loss = ln_nll_torch(logits.float(), yb) * (len(mbidx) / len(bidx))
                else:
                    loss = F.cross_entropy(logits.float(), yb, weight=cw) * (len(mbidx) / len(bidx))
                scaler.scale(loss).backward()
                tot += loss.detach() * len(bidx)
            scaler.step(opt)
            scaler.update()
        train_loss = float(tot) / max(1, n_steps * bs)
        lo, _ = _predict(model, val, val.tgt, K, uses_raw, eval_batch, amp)
        if ln:     # monitor = -(mean per-person inner CRPS): higher is better, as for BA
            vmu, vsig = ln_params_np(lo)
            ba = -person_crps(vmu, vsig, val.yraw[val.tgt], val.groups[val.tgt])
        else:
            vpred = lo.argmax(1)
            ba = person_ba(val.y_np[val.tgt], vpred, val.groups[val.tgt])
        if sched is not None:
            sched.step(ba) if train["schedule"] == "plateau" else sched.step()
        st["epoch"] += 1
        dt = time.time() - te
        ep_times.append(dt)
        st["history"].append({"epoch": st["epoch"], "train_loss": round(train_loss, 5), "inner_ba": round(ba, 5),
                              "lr": opt.param_groups[0]["lr"], "sec": round(dt, 2)})
        if ba > st["best_ba"] + 1e-12:
            st.update(best_ba=ba, best_epoch=st["epoch"], bad=0,
                      best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            st["bad"] += 1
        log(f"[{run}] fold {p} ep {st['epoch']:2d} loss {train_loss:.4f} innerBA {ba:.4f} "
            f"best {st['best_ba']:.4f}@{st['best_epoch']} {dt:.1f}s")
        if st["bad"] >= train["patience"]:
            finished = True
            break
        if (time.time() + 1.3 * max(ep_times[-3:]) + 15 > deadline or
                (train["pause_after"] is not None and len(ep_times) >= train["pause_after"])):
            st["wall_prev"] += time.time() - t0
            torch.save({"config": cfg_json, "model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "sched": sched.state_dict() if sched is not None else None,
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None, "state": st},
                       ckpt.with_suffix(".tmp"))
            os.replace(ckpt.with_suffix(".tmp"), ckpt)
            log(f"[{run}] fold {p}: paused after epoch {st['epoch']} (time limit), checkpoint saved")
            return "paused"
    stop_epoch = st["epoch"]
    model.load_state_dict(st["best_state"])
    all_loc = np.arange(len(test.rows))
    logits, emb = _predict(model, test, all_loc, K, uses_raw, eval_batch, amp, want_emb=True)
    if ln:
        tmu, tsig = ln_params_np(logits)
        prob = np.stack([tmu, tsig], 1).astype(np.float32)
    else:
        prob = torch.softmax(torch.from_numpy(logits), 1).numpy().astype(np.float32)
    tl = test.tgt
    info = {"person": p, "inner_val": val_p, "best_epoch": st["best_epoch"], "stop_epoch": stop_epoch,
            "early_stopped": finished, "best_inner_ba": round(float(st["best_ba"]), 5),
            "test_ba": (round(-person_crps(prob[:, 0], prob[:, 1], test.yraw, test.groups), 5) if ln else
                        round(balanced_acc(test.y_np[tl], prob[tl].argmax(1)), 5)),
            "target_stride": stride, "n_params": n_params,
            "n_fit_targets": int(len(fit.tgt)), "n_test_targets": int(len(tl)),
            "wall_s": round(st["wall_prev"] + time.time() - t0, 1), "prep_s": round(prep_s, 1),
            "sec_per_epoch": round(float(np.mean([h["sec"] for h in st["history"]])), 2),
            "peak_gpu_alloc_mb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1) if dev.type == "cuda" else None,
            "peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved(dev) / 2 ** 20, 1) if dev.type == "cuda" else None,
            "device": str(dev), "emb_dim": int(emb.shape[1]) if emb is not None else 0,
            "global_scale": gs.tolist() if gs is not None else None, "history": st["history"]}
    arrs = {"rows": test.rows[tl], "y": (test.yraw[tl] if ln else test.y_np[tl]),
            "pred": (np.zeros(len(tl), int) if ln else prob[tl].argmax(1)), "score": prob[tl],
            "emb": (emb[tl] if emb is not None else np.zeros((len(tl), 0))).astype(np.float16),
            "all_rows": test.rows, "all_score": prob, "info": json.dumps(_jsonable(info)), "config": cfg_json}
    tmp = run_dir / f"fold_{p}.tmp.npz"
    np.savez(tmp, **arrs)
    if save_weights:
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in st["best_state"].items()},
                   run_dir / f"fold_{p}.pt")
    os.replace(tmp, run_dir / f"fold_{p}.npz")
    if ckpt.exists():
        ckpt.unlink()
    log(f"[{run}] fold {p} done: best inner BA {st['best_ba']:.4f} @ {st['best_epoch']}, test BA {info['test_ba']:.4f}, "
        f"{info['wall_s']:.0f} s, peak GPU {info['peak_gpu_alloc_mb']} MB")
    del sps, fit, val, test, model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return "done"


def _merge(run, run_dir, out_dir, D, y, keep, task, config, spec, feat_set, input_fn, train, montage, norm, filt):
    task_rows = np.flatnonzero(keep)
    n = len(task_rows)
    nc = 3 if task == "three" else 2
    if task == "perclos_ln":
        y = D["perclos"]
    pred = np.full(n, -1)
    score = np.full((n, nc), np.nan, np.float32)
    emb = None
    all_score = np.full((len(keep), nc), np.nan, np.float32)
    infos = {}
    for f in sorted(run_dir.glob("fold_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        with np.load(f) as zf:
            z = {k: zf[k] for k in zf.files}
        pos = np.searchsorted(task_rows, z["rows"])
        assert np.array_equal(task_rows[pos], z["rows"])
        assert np.array_equal(y[z["rows"]], z["y"])
        pred[pos], score[pos] = z["pred"], z["score"]
        if emb is None:
            emb = np.zeros((n, z["emb"].shape[1]), np.float16)
        emb[pos] = z["emb"]
        all_score[z["all_rows"]] = z["all_score"]
        info = json.loads(str(z["info"]))
        infos[str(info["person"])] = info
    assert (pred >= 0).all() and np.isfinite(score).all(), "missing test predictions"
    out = {"y": y[task_rows], "pred": pred, "score": score, "groups": D["groups"][task_rows],
           "rec": D["rec"][task_rows], "perclos": D["perclos"][task_rows], "t": D["t"][task_rows], "emb": emb}
    if task == "binary":
        out.update(all_score=all_score, all_rec=D["rec"], all_keep=keep, all_t=D["t"])
    path = out_dir / f"{run}.npz"
    np.savez_compressed(path, **out)
    import braindecode
    walls = [v["wall_s"] for v in infos.values()]
    meta = dict(config)
    meta.update({
        "run": run, "pred_file": path.name, "registry_desc": DM.REGISTRY.get(config["model"], {}).get("desc"),
        "ch_names": D["ch_names"], "filt": filt, "band_hz": list(train["band"]),
        "deployable": is_deployable_run(filt, norm, feat_set, input_fn),
        "transductive": norm == "recording",
        "notes": ("norm='recording' uses the whole unlabeled test recording (offline, transductive)" if norm == "recording"
                  else ""),
        "n_persons_test": len(infos), "total_wall_s": round(float(np.sum(walls)), 1),
        "mean_wall_s_per_fold": round(float(np.mean(walls)), 1),
        "n_params": next(iter(infos.values()))["n_params"],
        "folds": infos, "sha256": code_hashes(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "braindecode": braindecode.__version__, "numpy": np.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "created": datetime.datetime.now().isoformat(timespec="seconds")})
    (out_dir / f"{run}.json").write_text(json.dumps(_jsonable(meta), indent=1), encoding="utf-8")
    return path


# ============================================================================== post-processing
def causal_smooth_scores(npz_path, q_over_r: float = 0.01, out_path=None) -> pathlib.Path:
    """CAUSAL random-walk Kalman filter (eog_features_lit.kalman_filter_rw) of the class
    log-probabilities, per recording in time order, then renormalised (softmax). For binary runs the
    filter runs over ALL epochs of each recording (all_score), tired ones included, and the labelled
    rows are selected afterwards. Writes '<name>+kf.npz' next to the input (deployable post-processing,
    comparable to the '+kf' feature modifier) and returns its path."""
    npz_path = pathlib.Path(npz_path)
    with np.load(npz_path) as zf:
        z = {k: zf[k] for k in zf.files}
    if "all_score" in z:
        Sall, rec_all, keep = z["all_score"].astype(np.float64), z["all_rec"], z["all_keep"].astype(bool)
    else:
        Sall, rec_all, keep = z["score"].astype(np.float64), z["rec"], np.ones(len(z["score"]), bool)
    logp = np.log(np.clip(Sall, 1e-6, 1.0))
    sm = L.kalman_filter_rw(logp, rec_all, q_over_r)
    sm = sm - sm.max(1, keepdims=True)
    p = np.exp(sm)
    p /= p.sum(1, keepdims=True)
    out = dict(z)
    out["score"] = p[keep].astype(np.float32)
    out["pred"] = out["score"].argmax(1)
    if "all_score" in z:
        out["all_score"] = p.astype(np.float32)
    out_path = pathlib.Path(out_path) if out_path else npz_path.with_name(npz_path.stem + "+kf.npz")
    np.savez_compressed(out_path, **out)
    return out_path


# ============================================================================== CLI
def _parse_hp(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=["three", "binary", "perclos_ln"])
    ap.add_argument("--model")
    ap.add_argument("--montage", default="trad6")
    ap.add_argument("--norm", default="trailing", choices=list(DD.NORMS))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instance-norm", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--folds", default=None, help="comma-separated person ids")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hp", action="append", help="extra hyper-parameter key=value (JSON value)")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--kf", action="store_true", help="also write the causal Kalman-smoothed file when complete")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smooth", default=None, help="only smooth this merged npz file")
    ap.add_argument("--q", type=float, default=0.01)
    a = ap.parse_args(argv)
    if a.smooth:
        print(causal_smooth_scores(a.smooth, a.q))
        return
    if not a.task or not a.model:
        ap.error("--task and --model are required")
    hp = _parse_hp(a.hp)
    if a.instance_norm:
        hp["instance_norm"] = True
    if a.cpu:
        hp["device"] = "cpu"
    folds = [int(v) for v in a.folds.split(",")] if a.folds else None
    res = run_loso(a.task, a.model, montage=a.montage, norm=a.norm, K=a.K, seed=a.seed, out_dir=a.out, folds=folds,
                   max_minutes=a.max_minutes, save_weights=not a.no_weights, tag=a.tag, **hp)
    if res["merged"] and a.kf:
        print(causal_smooth_scores(res["merged"], a.q))
    print(json.dumps({k: res[k] for k in ("status", "merged", "remaining")}))


if __name__ == "__main__":
    main()
