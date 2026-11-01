"""Shared helpers of topic T1 (threshold-consistent distributional PERCLOS estimation).

Import first in every T1 script (python -I): it puts 00_COMMON and this folder on sys.path, limits
BLAS threads to 2, and reconfigures stdout to UTF-8.

Contents
  * paths (T1 folder layout of PREREG section 16), topic constants
  * cut-point grids (PREREG section 3.2): 7 x 7 three-class cells, binary-pair cells (alert < t1 vs
    drowsy >= t2, middle dropped), 11 single thresholds, 14 cut-points {0.20, ..., 0.85}
  * label functions per cell, validity of binary-pair cells (labels only)
  * decision rules (PREREG section 5): balanced Bayes rule argmax p_c / prior_c (ties -> lower class)
  * logit-normal CDF utilities, CRPS on the 512-point grid, PIT, interval helpers
  * per-person metrics for any cell with the same logic as 00_COMMON/summarize.per_person
  * writers for prediction files (shared schema) and grid files
"""
from __future__ import annotations

import os
import sys as _sys

_sys.dont_write_bytecode = True     # never write __pycache__ into 00_COMMON
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
T1 = HERE.parent
ROOT = T1.parent
COMMON = ROOT / "00_COMMON"
for _p in (str(HERE), str(COMMON)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

import numpy as np  # noqa: E402
from scipy.special import expit, logit, ndtr  # noqa: E402

import seedvig_eog as S  # noqa: E402

TOPIC = "T1_ThresholdConsistentPerclos"
RES = T1 / "results"
PREDS = RES / "preds"
CDFD = RES / "cdf"
GRID = RES / "grid"
FOLDS = RES / "folds"
DEEP = RES / "deep"
FIG = T1 / "figures"
LOGS = T1 / "logs"
for _d in (RES, PREDS, CDFD, GRID, FOLDS, DEEP, FIG, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- grids
TH1 = tuple(round(0.20 + 0.05 * k, 2) for k in range(7))       # 0.20 .. 0.50
TH2 = tuple(round(0.55 + 0.05 * k, 2) for k in range(7))       # 0.55 .. 0.85
CELLS = [(a, b) for a in TH1 for b in TH2]                      # 49, row-major (t1 slow, t2 fast)
CANON = (0.35, 0.70)
CANON_IDX = CELLS.index(CANON)
CUTS = tuple(round(0.20 + 0.05 * k, 2) for k in range(14))     # 0.20 .. 0.85
SINGLE = tuple(round(0.30 + 0.05 * k, 2) for k in range(11))   # 0.30 .. 0.80
SUBGRID = [(a, b) for a in (0.25, 0.35, 0.45) for b in (0.60, 0.70, 0.80)]   # H7a 3 x 3
MIN_PER_CLASS = 20
MIN_PERSONS_CELL = 10
CLIP_LO, CLIP_HI = 0.005, 0.995
NGRID = 512
GRID512 = (np.arange(NGRID) + 0.5) / NGRID
SIGMA_FLOOR = 1e-3


def cut_idx(th: float) -> int:
    return CUTS.index(round(th, 2))


def lab3(y, t1, t2):
    """Three-class labels at cell (t1, t2): 0 if y < t1, 2 if y >= t2, else 1."""
    y = np.asarray(y)
    return np.where(y < t1, 0, np.where(y >= t2, 2, 1)).astype(np.int8)


def lab2(y, t1, t2):
    """Binary-pair labels: 0 alert (y < t1), 1 drowsy (y >= t2), -1 dropped (t1 <= y < t2)."""
    y = np.asarray(y)
    return np.where(y < t1, 0, np.where(y >= t2, 1, -1)).astype(np.int8)


def labS(y, th):
    """Single threshold: 1 if y >= th else 0."""
    return (np.asarray(y) >= th).astype(np.int8)


def cell_labels(y, task: str, cell):
    if task == "three":
        return lab3(y, *cell)
    if task == "binarypair":
        return lab2(y, *cell)
    if task == "single":
        return labS(y, cell)
    raise ValueError(task)


def task_cells(task: str):
    return {"three": CELLS, "binarypair": CELLS, "single": list(SINGLE)}[task]


def n_classes(task: str) -> int:
    return 3 if task == "three" else 2


# ----------------------------------------------------------------------------- base data
_BASE = None


def base() -> dict:
    """perclos, yi, groups (person), rec, t of all 20,355 epochs in the shared row order."""
    global _BASE
    if _BASE is None:
        parts = {k: [] for k in ("perclos", "yi", "groups", "rec", "t")}
        for i, n in enumerate(S.RECORDINGS):
            with np.load(S.CACHE_DIR / f"{n}.npz") as z:
                parts["perclos"].append(z["perclos"].astype(np.float64))
                parts["yi"].append(z["yi"])
        for i, n in enumerate(S.RECORDINGS):
            parts["groups"].append(np.full(S.N_EPOCH, S.person_of(n)))
            parts["rec"].append(np.full(S.N_EPOCH, i))
            parts["t"].append(np.arange(S.N_EPOCH, dtype=np.float64) * S.EPOCH_S)
        _BASE = {k: np.concatenate(v) for k, v in parts.items()}
        _BASE["epoch_idx"] = np.rint(_BASE["t"] / S.EPOCH_S).astype(int)
        _BASE["persons"] = np.unique(_BASE["groups"])
    return _BASE


def load_features(featset: str) -> dict:
    import feature_sets as FSET
    d = FSET.load(featset)
    b = base()
    for k in ("groups", "rec", "t"):
        assert np.array_equal(d[k], b[k]), k
    assert np.allclose(d["perclos"], b["perclos"])
    return d


def is_deployable_feat(featset: str) -> bool:
    import feature_sets as FSET
    return bool(FSET.is_deployable(featset))


def valid_binary_cells(y=None, groups=None) -> np.ndarray:
    """Boolean (49,) validity of binary-pair cells: >= 10 persons with >= 20 epochs per class."""
    b = base()
    y = b["perclos"] if y is None else y
    groups = b["groups"] if groups is None else groups
    ok = np.zeros(len(CELLS), bool)
    for j, c in enumerate(CELLS):
        lab = lab2(y, *c)
        n = sum(min(((lab == 0) & (groups == p)).sum(), ((lab == 1) & (groups == p)).sum()) >= MIN_PER_CLASS
                for p in np.unique(groups))
        ok[j] = n >= MIN_PERSONS_CELL
    return ok


def canonical_binary_persons() -> list[int]:
    b = base()
    yi = b["yi"]
    out = []
    for p in b["persons"]:
        m = b["groups"] == p
        if min((yi[m] == 1).sum(), (yi[m] == 3).sum()) >= MIN_PER_CLASS:
            out.append(int(p))
    return out


# ----------------------------------------------------------------------------- decision rules
def bayes_balanced(P: np.ndarray, prior) -> np.ndarray:
    """Balanced Bayes rule: argmax_c p_c / prior_c (np.argmax -> ties go to the lower class)."""
    prior = np.maximum(np.asarray(prior, float), 1e-12)
    return np.argmax(np.asarray(P, float) / prior[None, :], axis=1).astype(np.int8)


def prevalence(lab: np.ndarray, ncls: int) -> np.ndarray:
    lab = lab[lab >= 0]
    return np.bincount(lab.astype(int), minlength=ncls) / max(1, len(lab))


# ----------------------------------------------------------------------------- distributions
def zlogit(y):
    return logit(np.clip(np.asarray(y, float), CLIP_LO, CLIP_HI))


def ln_cdf(x, mu, sigma):
    """Logit-normal CDF at PERCLOS values x (broadcast: mu, sigma (n, 1), x (1, m) or (m,))."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        lx = logit(np.clip(x, 1e-12, 1 - 1e-12))
    return ndtr((lx - mu) / sigma)


def ln_Fcut(mu, sigma):
    mu = np.asarray(mu, float)[:, None]
    sigma = np.asarray(sigma, float)[:, None]
    return ln_cdf(np.array(CUTS)[None, :], mu, sigma)


def ln_cdf_grid(mu, sigma, chunk=4096):
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    out = np.empty((len(mu), NGRID), np.float64)
    lg = logit(GRID512)[None, :]
    for a in range(0, len(mu), chunk):
        out[a:a + chunk] = ndtr((lg - mu[a:a + chunk, None]) / sigma[a:a + chunk, None])
    return out


def crps_from_grid(cdf_grid: np.ndarray, y: np.ndarray, chunk=4096) -> np.ndarray:
    """CRPS = int_0^1 (F(z) - 1{y <= z})^2 dz, midpoint rule on the 512-point grid, raw y."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    for a in range(0, len(y), chunk):
        ind = (y[a:a + chunk, None] <= GRID512[None, :]).astype(np.float64)
        out[a:a + chunk] = np.mean((cdf_grid[a:a + chunk].astype(np.float64) - ind) ** 2, axis=1)
    return out


def ln_crps(mu, sigma, y, chunk=4096) -> np.ndarray:
    mu = np.asarray(mu, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), mu.shape)
    y = np.asarray(y, float)
    out = np.empty(len(y))
    lg = logit(GRID512)[None, :]
    for a in range(0, len(y), chunk):
        F = ndtr((lg - mu[a:a + chunk, None]) / sigma[a:a + chunk, None])
        ind = (y[a:a + chunk, None] <= GRID512[None, :])
        out[a:a + chunk] = np.mean((F - ind) ** 2, axis=1)
    return out


def ln_logscore(mu, sigma, y):
    """Negative log density on the logit scale of the clipped target (lower is better)."""
    z = zlogit(y)
    sigma = np.asarray(sigma, float)
    return 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * ((z - mu) / sigma) ** 2


def grid_mean(cdf_grid):
    return np.mean(1.0 - cdf_grid.astype(np.float64), axis=1)


def grid_quantile(cdf_grid, q):
    """Quantile of a CDF given on GRID512 (linear interpolation; first crossing)."""
    G = cdf_grid.astype(np.float64)
    out = np.empty(len(G))
    for i in range(len(G)):
        Fi = np.maximum.accumulate(G[i])
        j = np.searchsorted(Fi, q)
        if j <= 0:
            out[i] = GRID512[0] * (q / max(Fi[0], 1e-12))
        elif j >= NGRID:
            out[i] = GRID512[-1]
        else:
            f0, f1 = Fi[j - 1], Fi[j]
            w = 0.0 if f1 <= f0 else (q - f0) / (f1 - f0)
            out[i] = GRID512[j - 1] + w * (GRID512[j] - GRID512[j - 1])
    return out


# ----------------------------------------------------------------------------- derived class probabilities
def three_from_F(Fcut, cell):
    F1 = Fcut[:, cut_idx(cell[0])]
    F2 = Fcut[:, cut_idx(cell[1])]
    P = np.stack([F1, np.maximum(F2 - F1, 0.0), 1.0 - F2], 1)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-300)


def binpair_from_F(Fcut, cell):
    a = Fcut[:, cut_idx(cell[0])]
    d = 1.0 - Fcut[:, cut_idx(cell[1])]
    den = a + d
    q = np.where(den > 0, d / np.where(den > 0, den, 1.0), 0.5)
    return np.stack([1.0 - q, q], 1)


def single_from_F(Fcut, th):
    p1 = 1.0 - Fcut[:, cut_idx(th)]
    return np.stack([1.0 - p1, p1], 1)


def probs_from_F(Fcut, task, cell):
    return {"three": three_from_F, "binarypair": binpair_from_F, "single": single_from_F}[task](Fcut, cell)


def tc_grid(Fcut: np.ndarray, rule: str = "bayes"):
    """Grid decisions of a train-once model from its CDF at the 14 cut-points.

    rule 'bayes' = balanced Bayes rule with outer-training prevalences (per test person, per cell);
    'argmax' = plain argmax; 'median' = predictive median compared with the cut-points (the median is
    F^{-1}(0.5); class decided from F(t) vs 0.5, which gives nested decisions)."""
    b = base()
    y, groups = b["perclos"], b["groups"]
    out = {}
    for task in ("three", "binarypair", "single"):
        cells = task_cells(task)
        C = n_classes(task)
        pred = np.full((len(y), len(cells)), -1, np.int8)
        score = np.zeros((len(y), len(cells), C), np.float32)
        for j, c in enumerate(cells):
            lab = cell_labels(y, task, c)
            P = probs_from_F(Fcut, task, c)
            score[:, j] = P
            if rule == "argmax":
                pred[:, j] = np.argmax(P, 1)
            elif rule == "median":
                if task == "three":
                    pred[:, j] = np.where(Fcut[:, cut_idx(c[0])] > 0.5, 0,
                                          np.where(Fcut[:, cut_idx(c[1])] <= 0.5, 2, 1))
                elif task == "binarypair":
                    # median below t1 -> alert, at/above t2 -> drowsy, in between -> nearer side by q
                    pred[:, j] = np.where(Fcut[:, cut_idx(c[0])] > 0.5, 0,
                                          np.where(Fcut[:, cut_idx(c[1])] <= 0.5, 1, (P[:, 1] > 0.5).astype(int)))
                else:
                    pred[:, j] = (Fcut[:, cut_idx(c)] <= 0.5).astype(int)
            else:
                for p in np.unique(groups):
                    te = groups == p
                    prior = prevalence(lab[~te], C)
                    pred[te, j] = bayes_balanced(P[te], prior)
            if task == "binarypair":
                pred[lab < 0, j] = -1
        out[task] = (pred, score)
    return out


# ----------------------------------------------------------------------------- metrics
def ba(yt, yp) -> float:
    """Balanced accuracy over the classes present in yt (= sklearn balanced_accuracy_score)."""
    rec = [np.mean(yp[yt == c] == c) for c in np.unique(yt)]
    return float(np.mean(rec))


def f1_macro_present(yt, yp) -> float:
    vals = []
    for c in np.unique(yt):
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        den = 2 * tp + fp + fn
        vals.append(0.0 if den == 0 else 2 * tp / den)
    return float(np.mean(vals))


def qwk(yt, yp, ncls=3) -> float:
    from sklearn.metrics import cohen_kappa_score
    if len(np.unique(yt)) < 2:
        return np.nan
    return float(cohen_kappa_score(yt, yp, labels=list(range(ncls)), weights="quadratic"))


def cell_person_metrics(y_perclos, groups, pred_col, task, cell, want_f1=False, want_qwk=False) -> dict:
    """{person: dict(bacc, f1_macro, qwk, n)} for one grid cell. Binary-pair / single: person scored
    only with >= 20 epochs per class; three-class: every person (classes present)."""
    lab = cell_labels(y_perclos, task, cell)
    rows = {}
    for p in np.unique(groups):
        m = (groups == p) & (lab >= 0)
        yt, yp = lab[m], pred_col[m]
        if task != "three":
            if min((yt == 0).sum(), (yt == 1).sum()) < MIN_PER_CLASS:
                continue
        r = {"bacc": ba(yt, yp), "n": int(m.sum())}
        if want_f1:
            r["f1_macro"] = f1_macro_present(yt, yp)
        if want_qwk and task == "three":
            r["qwk"] = qwk(yt, yp)
        rows[int(p)] = r
    return rows


def integrated_ba(grid_pred, task, persons=None, cells_mask=None, metric="bacc") -> dict:
    """Per-person integrated metric (mean over cells in which the person is scored).
    three: all 49 cells; binarypair: valid cells (cells_mask) in which the person qualifies."""
    b = base()
    cells = task_cells(task)
    acc = {}
    for j, c in enumerate(cells):
        if cells_mask is not None and not cells_mask[j]:
            continue
        rows = cell_person_metrics(b["perclos"], b["groups"], grid_pred[:, j], task, c,
                                   want_f1=metric == "f1_macro", want_qwk=metric == "qwk")
        for p, r in rows.items():
            acc.setdefault(p, []).append(r[metric])
    out = {p: float(np.nanmean(v)) for p, v in acc.items()}
    if persons is not None:
        out = {p: out[p] for p in persons if p in out}
    return out


# ----------------------------------------------------------------------------- statistics wrappers
def ci(values):
    return S.bootstrap_ci(np.asarray(values, float), b=10000, seed=0)


def perm_p(a, b):
    return S.paired_permutation_p(np.asarray(a, float), np.asarray(b, float), n=20000, seed=0)


def paired(dA: dict, dB: dict):
    ps = sorted(set(dA) & set(dB))
    return ps, np.array([dA[p] for p in ps]), np.array([dB[p] for p in ps])


# ----------------------------------------------------------------------------- writers
def save_preds(task: str, model_id: str, pred_all: np.ndarray, score_all: np.ndarray, extra: dict | None = None):
    """Canonical prediction file results/preds/<task>_<model_id>.npz (shared schema).
    pred_all / score_all over all 20,355 rows; binary keeps the alert/drowsy rows only."""
    b = base()
    if task == "three":
        keep = np.ones(len(b["yi"]), bool)
        y = (b["yi"] - 1).astype(int)
    else:
        keep = b["yi"] != 2
        y = (b["yi"][keep] == 3).astype(int)
    out = {"y": y, "pred": np.asarray(pred_all)[keep].astype(int), "score": np.asarray(score_all, np.float64)[keep],
           "groups": b["groups"][keep], "rec": b["rec"][keep], "perclos": b["perclos"][keep], "t": b["t"][keep]}
    if extra:
        out.update({k: (np.asarray(v)[keep] if np.ndim(v) and len(v) == len(keep) else v) for k, v in extra.items()})
    path = PREDS / f"{task}_{model_id}.npz"
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return path


def save_grid(model_id: str, grids: dict):
    """grids: task -> (pred n x cells int8, score n x cells x C)."""
    for task, (pred, score) in grids.items():
        cells = task_cells(task)
        path = GRID / f"{task}_{model_id}.npz"
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, pred=pred.astype(np.int8), score=np.asarray(score, np.float16),
                            cells=np.array(cells, float))
        os.replace(tmp, path)


def load_grid(task: str, model_id: str):
    with np.load(GRID / f"{task}_{model_id}.npz") as z:
        return z["pred"], z["score"]


def save_cdf(model_id: str, Fcut, cdf_grid, mean, median, qlo_hi: dict, params: dict | None = None):
    b = base()
    out = {"Fcut": np.asarray(Fcut, np.float64), "cdf": np.asarray(cdf_grid, np.float16), "mean": mean,
           "median": median, "groups": b["groups"], "rec": b["rec"], "perclos": b["perclos"], "t": b["t"],
           "cuts": np.array(CUTS)}
    for k, (lo, hi) in qlo_hi.items():
        out[f"q{k}_lo"], out[f"q{k}_hi"] = lo, hi
    for k, v in (params or {}).items():
        out[k] = v
    path = CDFD / f"{model_id}.npz"
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return path


def finalize_tc(model_id: str, Fcut, cdf_grid, mean, median, qlo_hi, params=None, pit=None, logscore=None):
    """CDF file + grid decisions (balanced Bayes) + canonical prediction files of a TC model."""
    extra = {}
    if pit is not None:
        extra["pit"] = pit
    if logscore is not None:
        extra["logscore"] = logscore
    save_cdf(model_id, Fcut, cdf_grid, mean, median, qlo_hi, {**(params or {}), **extra})
    grids = tc_grid(Fcut, "bayes")
    save_grid(model_id, grids)
    p3, s3 = grids["three"]
    save_preds("three", model_id, p3[:, CANON_IDX], s3[:, CANON_IDX])
    p2, s2 = grids["binarypair"]
    save_preds("binary", model_id, p2[:, CANON_IDX], s2[:, CANON_IDX])
    return grids


def append_csv(path: pathlib.Path, rows: list[dict], fields: list[str] | None = None):
    import csv
    new = not path.exists()
    fields = fields or list(rows[0].keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str] | None = None):
    import csv
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def log_compute(job: str, seconds: float, note: str = ""):
    import datetime
    append_csv(RES / "compute_log.csv", [{"time": datetime.datetime.now().isoformat(timespec="seconds"),
                                          "job": job, "seconds": round(seconds, 1), "note": note}])


def progress(note: str):
    import datetime
    with open(RES / "PROGRESS.md", "a", encoding="utf-8") as f:
        f.write(f"- {datetime.datetime.now().isoformat(timespec='minutes')}: {note}\n")


def fold_dir(job: str) -> pathlib.Path:
    d = FOLDS / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fold(job: str, person: int, **arrs):
    d = fold_dir(job)
    path = d / f"fold_{person}.npz"
    tmp = d / f"fold_{person}.tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def fold_done(job: str, person: int) -> bool:
    return (FOLDS / job / f"fold_{person}.npz").exists()


def load_fold(job: str, person: int) -> dict:
    with np.load(FOLDS / job / f"fold_{person}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def jdump(obj) -> str:
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
