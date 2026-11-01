"""Raw-EOG data layer for the deep-learning / hybrid models (shared by all five topics).

    import deep_data as DD
    D = DD.load_raw(montage="trad6", filt="causal")          # X (20355, 6, 1000) float32 + meta
    y, keep = DD.task_labels(D["yi"], "binary")             # binary: alert 0 / drowsy 1, tired dropped
    Xn = DD.normalise(D["X"], D["rec"], "trailing")         # label-free, causal per-channel scale
    idx, mask = DD.seq_index(D["rec"], K=8)                  # current + previous 7 epochs, same recording

Montages (electrodes 1-7, 0-based ch0..ch6 in the cache; seedvig_eog data facts):
  trad6 = [ch1, ch2, ch3, ch4, ch1-ch2 (HEO), ch4-ch3 (VEO, blinks positive)]
  fore6 = [ch4, ch5, ch6, ch7, ch5-ch6 (HEOf), ch4-ch7 (VEOf, blinks positive)]
  trad2 = [HEO, VEO]  (the last two channels of trad6)
  fore2 = [HEOf, VEOf] (the last two channels of fore6)
  bip4  = [HEO, VEO, HEOf, VEOf] (added option; read from the trad6 and fore6 caches)
Bipolar channels are formed in float64 BEFORE any float16 storage (VEOf of 19_20151114_noon has a
robust SD of ~1.3 uV, which float16 electrode differences would distort by ~3 %).

Filtering (on the CONTINUOUS 2-h recording, before epoching; 4th-order Butterworth band `band`):
  filt="causal"  forward-only sosfilt with steady-state initial conditions
                 (eog_features_lit.causal_filter): y[t] uses x[0..t] only -> deployable.
  filt="zero"    zero-phase sosfiltfilt (eog_features_lit.zero_phase) -> OFFLINE (look-ahead of
                 the 0.1-Hz high-pass impulse response, tens of seconds); never deployable.
  filt="none"    provider data as stored in cache/<rec>.npz (no filtering).
Cache: cache/deep/<filt>_<lo>-<hi>Hz/<montage>_<rec>.npy, float16 (885, 6, 1000), ~10.6 MB per
recording (244 MB per montage and filter). trad2 reads the trad6 cache. load_raw(cache="read")
(the default) reads a cache file when it exists and otherwise filters on the fly WITHOUT writing
(topic agents never write in 00_COMMON); cache="write" also writes missing files (infrastructure
only; `python -I deep_data.py --build` builds trad6 and fore6 for filt="causal", 0.1-30 Hz).

Normalisers (label-free; normalise / norm_params):
  "trailing"  per channel, CAUSAL: scale of epoch e = median of the per-epoch robust SDs
              (1.4826*MAD over the 1000 samples) of the previous 37 epochs of the same recording
              (lag 1, as eog_features_lit.trailing_scale / SCALE_EPOCHS). Start-up rule: epochs
              1..36 use the median over all previous epochs available (1..e); epoch 0 has no
              past and uses its OWN robust SD (known when epoch 0 is complete, i.e. when it is
              classified, so still causal at epoch resolution). No centring (the signals are
              band-passed). Deployable.
  "recording" per recording and channel, whole-recording z-score (mean and SD of all samples
              of the recording). OFFLINE and TRANSDUCTIVE (uses the unlabeled test recording,
              including its future); not deployable. Provided so that T2 can show what
              per-driver normalisation does; must be declared as transductive.
  "global"    per channel, one robust scale = median over all epochs of the FITTING persons
              (train_mask) of the per-epoch robust SD; fitted inside the fold. Deployable.
  "none"      no scaling.
Per-epoch instance normalisation is a MODEL option (deep_models: instance_norm=True), not a
data option.

Sequences: seq_index(rec, K) -> idx (n, K), mask (n, K): column K-1 is the current epoch, column
0 the oldest; slots before the start of the recording repeat the recording's first epoch with
mask 0. Strictly causal, never crosses recordings. Rows must be in time order within each
recording and each recording must be one contiguous block (true for load_raw).

Task labels (task_labels): "three" = yi - 1 for all epochs; "binary" = alert 0 / drowsy 1 with
tired epochs dropped from training AND testing (keep mask), as in pub2_loso.py. The input
CONTEXT of sequence models still contains the tired epochs (they are unlabeled input, available at
deployment); only the target rows are restricted.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import eog_features_lit as L  # noqa: E402
import seedvig_eog as S  # noqa: E402

DEEP_CACHE = S.CACHE_DIR / "deep"
CACHE_VERSION = 1
FS = S.FS
N_SAMP = S.N_SAMP
TRAIL_EPOCHS = L.SCALE_EPOCHS          # 37 previous epochs (~5 min)
SCALE_FLOOR = 1e-3                     # uV; guards flat channels

# montage -> (electrode / bipolar definitions over the 7 electrodes, channel names)
MONTAGES = {
    "trad6": ([("e", 0), ("e", 1), ("e", 2), ("e", 3), ("d", 0, 1), ("d", 3, 2)],
              ["ch1", "ch2", "ch3", "ch4", "HEO", "VEO"]),
    "fore6": ([("e", 3), ("e", 4), ("e", 5), ("e", 6), ("d", 4, 5), ("d", 3, 6)],
              ["ch4", "ch5", "ch6", "ch7", "HEOf", "VEOf"]),
}
# montage -> list of (base montage, channel subset), concatenated along the channel axis
DERIVED = {"trad2": [("trad6", [4, 5])], "fore2": [("fore6", [4, 5])],
           "bip4": [("trad6", [4, 5]), ("fore6", [4, 5])]}
NORMS = ("trailing", "recording", "global", "none")
FILTS = ("causal", "zero", "none")
DEPLOYABLE_NORMS = ("trailing", "global", "none")


def montage_channels(montage: str) -> list[str]:
    if montage in MONTAGES:
        return list(MONTAGES[montage][1])
    return [MONTAGES[b][1][i] for b, sub in DERIVED[montage] for i in sub]


def _cache_file(name: str, montage: str, filt: str, band) -> pathlib.Path:
    lo, hi = band
    return DEEP_CACHE / f"{filt}_{lo:g}-{hi:g}Hz" / f"{montage}_{name}.npy"


def _filter_recording(name: str, montage: str, filt: str, band) -> np.ndarray:
    """(885, 6, 1000) float32 of one recording: montage channels formed in float64 from the raw
    electrodes, filtered on the continuous recording, then epoched."""
    r = S.load_recording(name)
    x = np.asarray(r["eog"], np.float64).transpose(1, 0, 2).reshape(7, -1)     # (7, N) continuous
    defs = MONTAGES[montage][0]
    ch = np.stack([x[d[1]] if d[0] == "e" else x[d[1]] - x[d[2]] for d in defs])
    lo, hi = band
    if filt == "causal":
        ch = L.causal_filter(ch, lo, hi)
    elif filt == "zero":
        ch = L.zero_phase(ch, lo, hi)
    elif filt != "none":
        raise ValueError(f"filt must be one of {FILTS}")
    n_ep = r["eog"].shape[0]
    return ch.reshape(len(defs), n_ep, N_SAMP).transpose(1, 0, 2).astype(np.float32)


def _load_one(name: str, montage: str, filt: str, band, cache: str) -> np.ndarray:
    if filt == "none":
        return _filter_recording(name, montage, filt, band)
    f = _cache_file(name, montage, filt, band)
    if f.exists():
        return np.load(f).astype(np.float32)
    x = _filter_recording(name, montage, filt, band)
    if cache == "write":
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")
        np.save(tmp, x.astype(np.float16))
        os.replace(tmp, f)
        info = f.parent / "CACHE_INFO.json"
        if not info.exists():
            info.write_text(json.dumps({"cache_version": CACHE_VERSION, "filt": filt, "band_hz": list(band),
                                        "order": 4, "dtype": "float16", "shape": "(885, 6, 1000)",
                                        "montages": {k: v[1] for k, v in MONTAGES.items()},
                                        "built_by": "deep_data.py"}, indent=1), encoding="utf-8")
    return x


def load_raw(names: list[str] | None = None, montage: str = "trad6", filt: str = "causal",
             band: tuple = (0.1, 30.0), dtype=np.float32, cache: str = "read") -> dict:
    """Epoched, filtered raw EOG of the given recordings (default: all 23, seedvig_eog order).

    Returns dict: X (n_epochs, C, 1000) `dtype` (float32 default; float16 halves the memory),
    yi, perclos, groups (PERSON id), rec (index into names_rec), t (epoch onset, s), names_rec,
    ch_names, montage, filt, band, fs. Rows are in time order inside each recording.
    cache: "read" (default; never writes), "write" (write missing cache files), "off".
    """
    names = list(S.RECORDINGS if names is None else names)
    if montage in DERIVED:
        parts = DERIVED[montage]
    elif montage in MONTAGES:
        parts = [(montage, None)]
    else:
        raise ValueError(f"unknown montage {montage!r}; use {list(MONTAGES) + list(DERIVED)}")
    band = (float(band[0]), float(band[1]))
    n = len(names) * S.N_EPOCH
    C = len(montage_channels(montage))
    X = np.empty((n, C, N_SAMP), dtype=dtype)
    meta = {k: [] for k in ("yi", "perclos", "groups", "rec", "t")}
    for i, nm in enumerate(names):
        xs = [_load_one(nm, base, filt, band, cache if cache in ("read", "write") else "off")
              for base, _ in parts]
        x = np.concatenate([xx if sub is None else xx[:, sub] for xx, (_, sub) in zip(xs, parts)], axis=1)
        X[i * S.N_EPOCH:(i + 1) * S.N_EPOCH] = x
        r = S.load_recording(nm)
        meta["yi"].append(r["yi"])
        meta["perclos"].append(r["perclos"])
        meta["groups"].append(np.full(S.N_EPOCH, r["person"]))
        meta["rec"].append(np.full(S.N_EPOCH, i))
        meta["t"].append(r["t"])
    out = {k: np.concatenate(v) for k, v in meta.items()}
    out.update({"X": X, "names_rec": names, "ch_names": montage_channels(montage), "montage": montage,
                "filt": filt, "band": band, "fs": FS})
    return out


# ----------------------------------------------------------------------------- labels
def task_labels(yi: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(y, keep). three: y = yi - 1 (0 alert, 1 tired, 2 drowsy), keep all.
    binary: y = 0 alert / 1 drowsy (-1 tired), keep = not tired (tired dropped from train and test)."""
    if task == "three":
        y = S.three_class_labels(yi).astype(np.int64)
        return y, np.ones(len(y), bool)
    if task == "binary":
        y = S.binary_labels(yi).astype(np.int64)
        return y, y >= 0
    raise ValueError("task must be 'three' or 'binary'")


# ----------------------------------------------------------------------------- normalisation
def _segments(rec: np.ndarray) -> list[tuple[int, int]]:
    rec = np.asarray(rec)
    b = np.flatnonzero(np.diff(rec) != 0) + 1
    seg = list(zip(np.r_[0, b].tolist(), np.r_[b, len(rec)].tolist()))
    vals = [rec[a] for a, _ in seg]
    if len(set(vals)) != len(vals):
        raise ValueError("each recording must be one contiguous block of rows (time-ordered)")
    return seg


def epoch_robust_sd(X: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """(n, C) per-epoch, per-channel 1.4826 * MAD over the time axis (chunked, float32)."""
    n, C, _ = X.shape
    out = np.empty((n, C), np.float32)
    for a in range(0, n, chunk):
        x = np.asarray(X[a:a + chunk], np.float32)
        med = np.median(x, axis=-1, keepdims=True)
        out[a:a + chunk] = L.MAD_K * np.median(np.abs(x - med), axis=-1)
    return out


def _trailing_lag1_median(s: np.ndarray, n: int = TRAIL_EPOCHS) -> np.ndarray:
    """(n_ep, C): median of s over epochs e-n..e-1 (all available ones for e < n); epoch 0 -> s[0]."""
    out = np.empty_like(s)
    out[0] = s[0]
    for e in range(1, len(s)):
        out[e] = np.median(s[max(0, e - n):e], axis=0)
    return out


def norm_params(X: np.ndarray, rec: np.ndarray, norm: str, train_mask: np.ndarray | None = None,
                rsd: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(center (n, C) or None, scale (n, C) or None) of a label-free normaliser (see module doc).
    rsd: optional precomputed epoch_robust_sd(X)."""
    if norm == "none":
        return None, None
    n, C, T = X.shape
    if norm == "recording":
        center = np.empty((n, C), np.float32)
        scale = np.empty((n, C), np.float32)
        for a, b in _segments(rec):
            x = np.asarray(X[a:b], np.float64)
            m = x.mean(axis=(0, 2))
            sd = np.sqrt(np.maximum((x ** 2).mean(axis=(0, 2)) - m ** 2, 0.0))
            center[a:b], scale[a:b] = m, np.maximum(sd, SCALE_FLOOR)
        return center, scale
    if rsd is None:
        rsd = epoch_robust_sd(X)
    if norm == "trailing":
        scale = np.empty((n, C), np.float32)
        for a, b in _segments(rec):
            scale[a:b] = _trailing_lag1_median(rsd[a:b])
        return None, np.maximum(scale, SCALE_FLOOR)
    if norm == "global":
        if train_mask is None:
            raise ValueError("norm='global' must be fitted on training persons: pass train_mask")
        s = np.median(rsd[np.asarray(train_mask, bool)], axis=0)
        return None, np.maximum(np.broadcast_to(s, (n, C)).astype(np.float32), SCALE_FLOOR)
    raise ValueError(f"norm must be one of {NORMS}")


def apply_norm(X: np.ndarray, center, scale, out_dtype=np.float32, chunk: int = 2048) -> np.ndarray:
    """(X - center) / scale per epoch and channel, chunked, returned as out_dtype."""
    out = np.empty(X.shape, out_dtype)
    for a in range(0, len(X), chunk):
        x = np.asarray(X[a:a + chunk], np.float32)
        if center is not None:
            x = x - center[a:a + chunk, :, None]
        if scale is not None:
            x = x / scale[a:a + chunk, :, None]
        out[a:a + chunk] = x
    return out


def normalise(X: np.ndarray, rec: np.ndarray, norm: str, train_mask: np.ndarray | None = None,
              out_dtype=np.float32) -> np.ndarray:
    """Label-free normalisation of epoched raw EOG (n, C, T); see the module docstring.
    'global' needs train_mask (rows of the training persons of the current fold)."""
    c, s = norm_params(X, rec, norm, train_mask)
    return apply_norm(X, c, s, out_dtype)


def is_deployable(filt: str, norm: str) -> bool:
    """Data-side deployability: causal filtering and a causal / training-only normaliser."""
    return filt == "causal" and norm in DEPLOYABLE_NORMS


# ----------------------------------------------------------------------------- sequences
def seq_index(rec: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """idx (n, K) int64 and mask (n, K) float32 of the current and previous K-1 epochs of the same
    recording (column K-1 = current). Left padding repeats the recording's first epoch, mask 0."""
    rec = np.asarray(rec)
    n = len(rec)
    start = np.empty(n, np.int64)
    for a, b in _segments(rec):
        start[a:b] = a
    cand = np.arange(n)[:, None] + np.arange(-(K - 1), 1)[None, :]
    valid = cand >= start[:, None]
    idx = np.where(valid, cand, start[:, None]).astype(np.int64)
    return idx, valid.astype(np.float32)


def build_caches(montages=("trad6", "fore6"), filt: str = "causal", band=(0.1, 30.0)) -> None:
    """Write the float16 deep caches (infrastructure agent only)."""
    for m in montages:
        for nm in S.RECORDINGS:
            _load_one(nm, m, filt, band, "write")
        print(f"cache {filt} {band} {m}: done", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--filt", default="causal")
    ap.add_argument("--montages", default="trad6,fore6")
    a = ap.parse_args()
    if a.build:
        build_caches(tuple(a.montages.split(",")), a.filt)
