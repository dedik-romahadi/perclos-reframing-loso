"""Shared data core for the EOG SEED-VIG cross-subject (LOSO) studies.

Every topic folder imports this module (read-only) so that all five studies use the
same epochs, labels, subject grouping, splits and metrics.

Data facts (verified 28 Oct 2026 against the raw files):
  * 23 recordings, 885 epochs x 8 s each. Raw EOG is 7 electrodes at 125 Hz
    (1000 samples per epoch; provider resampled from 1000 Hz).
  * Electrodes 1-4 = traditional EOG around the eyes, 4-7 = forehead EOG (4 shared).
    Provider HEO  = eog_h = ch1 - ch2  (exact),  VEO = eog_v = ch3 - ch4 (exact).
    Forehead approximations (Zheng & Lu 2017): VEOf ~ ch4 - ch7, HEOf ~ ch5 - ch6.
  * PERCLOS is continuous in [0, 1] (eye-tracking glasses). Class code yi:
    1 alert (PERCLOS < 0.35), 2 tired, 3 drowsy (PERCLOS >= 0.70).
  * Subject = provider id prefix of the file name (21 persons, 23 recordings;
    prefixes 4_ and 5_ have two recordings each). LOSO here always leaves one
    PERSON out, so no recording of the test person is ever in the training data.
"""
from __future__ import annotations

import csv
import os
import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# repository version: the raw-data folder can be set with the environment variable SEEDVIG_RAW_DIR
RAW_DIR = pathlib.Path(os.environ.get("SEEDVIG_RAW_DIR", HERE.parents[1] / "DATASET" / "SADT-SEEDVIG" / "SEEDVIG"))
CACHE_DIR = HERE / "cache"

FS = 125                      # EOG sampling rate (Hz)
EPOCH_S = 8                   # epoch length (s)
N_SAMP = FS * EPOCH_S         # 1000 samples per epoch
N_EPOCH = 885                 # epochs per recording
CH_TRAD = [0, 1, 2, 3]        # traditional electrodes 1-4 (0-based)
CH_FOREHEAD = [3, 4, 5, 6]    # forehead electrodes 4-7 (0-based)
ALERT_T, DROWSY_T = 0.35, 0.70

# Recordings ordered by person id, then date (deterministic everywhere).
RECORDINGS = [
    "1_20151124_noon_2", "2_20151106_noon", "3_20151024_noon", "4_20151105_noon",
    "4_20151107_noon", "5_20141108_noon", "5_20151012_night", "6_20151121_noon",
    "7_20151015_night", "8_20151022_noon", "9_20151017_night", "10_20151125_noon",
    "11_20151024_night", "12_20150928_noon", "13_20150929_noon", "14_20151014_night",
    "15_20151126_night", "16_20151128_night", "17_20150925_noon", "18_20150926_noon",
    "19_20151114_noon", "20_20151129_night", "21_20151016_noon",
]
# The 14 recordings with >= 50 alert and >= 50 drowsy epochs (binary-study subset
# used across the lab's EEG lanes; see DATASET/SADT-SEEDVIG/README.md).
BINARY_SUBSET = [
    "12_20150928_noon", "14_20151014_night", "15_20151126_night", "16_20151128_night",
    "18_20150926_noon", "1_20151124_noon_2", "20_20151129_night", "21_20151016_noon",
    "3_20151024_noon", "4_20151105_noon", "4_20151107_noon", "5_20141108_noon",
    "5_20151012_night", "8_20151022_noon",
]


def person_of(name: str) -> int:
    return int(name.split("_")[0])


def session_of(name: str) -> str:
    return "night" if "night" in name else "noon"


# ----------------------------------------------------------------------------- cache
def _cache_path(name: str) -> pathlib.Path:
    return CACHE_DIR / f"{name}.npz"


def build_one(name: str) -> dict:
    """Read one raw recording and write its cache file. Returns a summary row."""
    import scipy.io as sio

    raw = sio.loadmat(RAW_DIR / f"{name}.mat", squeeze_me=True, struct_as_record=False)
    E = raw["EOG"]
    eog = np.asarray(E.eog, dtype=np.float64)
    assert eog.shape == (N_EPOCH * N_SAMP, 7), eog.shape
    assert np.allclose(E.eog_h, eog[:, 0] - eog[:, 1]), "HEO != ch1-ch2"
    assert np.allclose(E.eog_v, eog[:, 2] - eog[:, 3]), "VEO != ch3-ch4"
    ep = eog.reshape(N_EPOCH, N_SAMP, 7).transpose(0, 2, 1).astype(np.float32)
    perclos = np.asarray(sio.loadmat(RAW_DIR / "perclos_labels" / f"{name}.mat",
                                     squeeze_me=True)["perclos"], dtype=np.float64).ravel()
    yi = np.asarray(sio.loadmat(RAW_DIR / "LABELS" / f"{name}.mat",
                                squeeze_me=True)["yi"], dtype=np.int16).ravel()
    assert perclos.shape == (N_EPOCH,) and yi.shape == (N_EPOCH,)
    rule = np.where(perclos < ALERT_T, 1, np.where(perclos >= DROWSY_T, 3, 2))
    assert np.array_equal(rule, yi), "yi does not follow the PERCLOS rule"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_path(name), eog=ep, perclos=perclos, yi=yi)
    return {"name": name, "person": person_of(name), "session": session_of(name),
            "n_alert": int((yi == 1).sum()), "n_tired": int((yi == 2).sum()),
            "n_drowsy": int((yi == 3).sum()), "perclos_mean": float(perclos.mean()),
            "in_binary_subset": int(name in BINARY_SUBSET)}


def load_recording(name: str) -> dict:
    """Epoched data of one recording.

    Returns eog (885, 7, 1000) float32 in microvolts as provided (no filtering),
    perclos (885,), yi (885,), t (885,) epoch onset in seconds, plus ids.
    """
    z = np.load(_cache_path(name))
    return {"name": name, "person": person_of(name), "session": session_of(name),
            "eog": z["eog"], "perclos": z["perclos"], "yi": z["yi"],
            "t": np.arange(N_EPOCH, dtype=np.float64) * EPOCH_S}


def load_all(names: list[str] | None = None, channels: list[int] | None = None) -> dict:
    """Stack recordings. `groups` holds the PERSON id (use it for LOSO);
    `rec` holds the recording index into `names`."""
    names = list(RECORDINGS if names is None else names)
    parts = [load_recording(n) for n in names]
    eog = np.concatenate([p["eog"] if channels is None else p["eog"][:, channels] for p in parts])
    return {
        "names": names,
        "eog": eog,
        "perclos": np.concatenate([p["perclos"] for p in parts]),
        "yi": np.concatenate([p["yi"] for p in parts]),
        "t": np.concatenate([p["t"] for p in parts]),
        "groups": np.concatenate([np.full(N_EPOCH, p["person"]) for p in parts]),
        "rec": np.concatenate([np.full(N_EPOCH, i) for i in range(len(parts))]),
        "session": np.concatenate([np.full(N_EPOCH, p["session"]) for p in parts]),
    }


def derived_signals(eog: np.ndarray) -> dict:
    """Bipolar derivations from the 7-electrode array (..., 7, T)."""
    return {
        "HEO": eog[..., 0, :] - eog[..., 1, :],
        "VEO": eog[..., 2, :] - eog[..., 3, :],
        "VEOf": eog[..., 3, :] - eog[..., 6, :],
        "HEOf": eog[..., 4, :] - eog[..., 5, :],
    }


# ----------------------------------------------------------------------------- labels
def binary_labels(yi: np.ndarray) -> np.ndarray:
    """0 alert, 1 drowsy, -1 tired (drop -1 before binary training/testing)."""
    return np.where(yi == 1, 0, np.where(yi == 3, 1, -1)).astype(np.int8)


def three_class_labels(yi: np.ndarray) -> np.ndarray:
    """0 alert, 1 tired, 2 drowsy."""
    return (yi.astype(np.int8) - 1)


# ----------------------------------------------------------------------------- splits
def loso_splits(groups: np.ndarray):
    """Leave-one-person-out. Yields (person, train_idx, test_idx)."""
    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        yield int(g), train, test


def inner_group_folds(groups: np.ndarray, k: int = 5, seed: int = 0):
    """Person-grouped folds INSIDE a training set (for tuning/early stopping).
    Yields (train_idx, val_idx) relative to the given `groups` array."""
    persons = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(persons)
    for f in range(k):
        val_p = order[f::k]
        val = np.flatnonzero(np.isin(groups, val_p))
        tr = np.flatnonzero(~np.isin(groups, val_p))
        yield tr, val


# ----------------------------------------------------------------------------- metrics
def binary_metrics(y: np.ndarray, score: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    pred = (score >= thr).astype(int)
    out = {"bacc": balanced_accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
           "sens": float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else np.nan,
           "spec": float(np.mean(pred[y == 0] == 0)) if (y == 0).any() else np.nan}
    out["auroc"] = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    return out


def multiclass_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Macro-F1 over the classes PRESENT in y, with 0/0 counted as 0 (audit evaluation-3:
    the original MATLAB scripts dropped NaN per-class values with 'omitnan')."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

    present = np.unique(y)
    return {"acc": accuracy_score(y, pred), "bacc": balanced_accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, labels=present, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y, pred) if len(present) > 1 else np.nan}


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    cor = float(np.corrcoef(y, yhat)[0, 1]) if np.std(yhat) > 0 and np.std(y) > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    my, mh = y.mean(), yhat.mean()
    ccc = float(2 * np.mean((y - my) * (yhat - mh)) / (y.var() + yhat.var() + (my - mh) ** 2))
    return {"cor": cor, "rmse": rmse, "ccc": ccc}


# ----------------------------------------------------------------------------- statistics
def person_mean(values: np.ndarray, persons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average per-recording values within person first (persons with two
    recordings count once). Returns (unique_persons, per_person_values)."""
    up = np.unique(persons)
    return up, np.array([np.nanmean(values[persons == p]) for p in up])


def bootstrap_ci(values: np.ndarray, b: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of the mean over units (persons)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (b, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_p(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired per-person differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((flips * d).mean(1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (n + 1))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def manifest() -> list[dict]:
    with open(CACHE_DIR / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def assert_no_person_overlap(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    """Call inside every LOSO loop: guards against subject leakage."""
    assert not set(np.unique(train_groups)) & set(np.unique(test_groups)), "person leakage"


_NAME_RE = re.compile(r"^\d+_\d{8}_(noon|night)(_\d)?$")
assert all(_NAME_RE.match(n) for n in RECORDINGS) and len(RECORDINGS) == 23
assert len({person_of(n) for n in RECORDINGS}) == 21
