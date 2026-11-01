"""One loader for every shared feature cache (used by feature_check.py and by the topics).

    import feature_sets as FS
    d = FS.load("lit-trad+kf")      # dict: X, names, yi, perclos, groups, rec, t, names_rec

Feature-set grammar:  <base>[-<part>][+<mod>...]
  pub2-<type>      cache/pub2_<rec>.npz      (current port, D1/D2 deviations; type1|type2|type3|all)
  pub2fix-<type>   cache/pub2fix_<rec>.npz   (corrected extractor, pub2_features_fixed mode='fixed')
  pub2orig-<type>  cache/pub2orig_<rec>.npz  (faithful = as published, bugs included)
  lit-trad, lit-fore   cache/lit_<montage>_<rec>.npz: OFFLINE zero-phase literature features
                   (look-ahead up to LOOKAHEAD_EPOCHS['offline'] = 13 epochs, 1-2 min), default set
                   WITHOUT the explicit eye-closure block F05
  litc-trad, litc-fore cache/litc_<montage>_<rec>.npz: CAUSAL variant (forward-only filters,
                   offset-assigned fixations; look-ahead 1 epoch = 8 s), same columns
  time             epoch onset in the recording (time-on-task control, README rule 6)
Modifiers:
  +F05   keep the F05 eye-closure block (PERCLOS PROXY, near-circular with the label)
  +noblinkdur  drop the blink blocks F01-F04 (duration, closing/closed time, AVR, rate); these
               also re-estimate the blink term of the PERCLOS label (sensitivity analysis)
  +noblinkall  drop F01-F04 and every F12 column derived from blinks (sensitivity analysis)
  +kf    CAUSAL forward Kalman (random-walk LDS, q/r = 0.01) per recording, per feature
  +ma    CAUSAL trailing moving average over 8 epochs (64 s) per recording
  +rts   NON-CAUSAL RTS smoother (uses future epochs; comparison only, never deployable)
  +ctx   append F17 causal trailing-context features (means, SD, slope, delta)
Non-finite values (+/-inf) are returned as NaN; impute with statistics of TRAINING persons only
(e.g. SimpleImputer inside the LOSO pipeline).

Deployability: `is_deployable(featset)` is an explicit WHITELIST. A set is deployable only if
every step is causal at epoch resolution (look-ahead <= 1 epoch) and it has no explicit
eye-closure block: bases pub2-* (per-epoch filtering and thresholds) and litc-*, with
modifiers from CAUSAL_MODS. pub2orig-* (whole-recording detrend/filloutliers), pub2fix-* and
lit-* (zero-phase on the continuous record), +rts and +F05 are never deployable.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eog_features_lit as L  # noqa: E402
import seedvig_eog as S  # noqa: E402

PUB2_TYPES = ("type1", "type2", "type3", "all")
LIT_BASES = {"lit-trad": "lit_trad", "lit-fore": "lit_fore", "litc-trad": "litc_trad", "litc-fore": "litc_fore"}
DEPLOYABLE_BASES = ("pub2-type1", "pub2-type2", "pub2-type3", "pub2-all", "litc-trad", "litc-fore")
CAUSAL_MODS = ("kf", "ma", "ctx", "noblinkdur", "noblinkall")


def is_deployable(featset: str) -> bool:
    """Explicit whitelist of causal feature sets without the eye-closure block (see docstring)."""
    base, *mods = featset.split("+")
    return base in DEPLOYABLE_BASES and all(m in CAUSAL_MODS for m in mods)


def is_offline(featset: str) -> bool:
    """True for sets that look ahead more than one epoch (zero-phase / whole-recording / RTS)."""
    base, *mods = featset.split("+")
    return base.startswith(("pub2orig-", "pub2fix-", "lit-")) or "rts" in mods


def _pub2_block(prefix: str, rec: str, part: str) -> np.ndarray:
    z = np.load(S.CACHE_DIR / f"{prefix}_{rec}.npz")
    if part == "all":
        return np.hstack([z["type1"], z["type2"], z["type3"]])
    return z[part]


def _pub2_names(prefix: str, part: str) -> list[str]:
    z = np.load(S.CACHE_DIR / f"{prefix}_feature_names.npz")
    if part == "all":
        return [f"{k}:{n}" for k in ("type1", "type2", "type3") for n in z[k].tolist()]
    return z[part].tolist()


def load(featset: str, names: list[str] | None = None) -> dict:
    names = list(S.RECORDINGS if names is None else names)
    base, *mods = featset.split("+")
    blocks, meta = [], {k: [] for k in ("yi", "perclos", "groups", "rec", "t")}
    for i, n in enumerate(names):
        r = S.load_recording(n)
        if base == "time":
            blocks.append(r["t"][:, None])
        elif base.startswith(("pub2-", "pub2fix-", "pub2orig-")):
            prefix, part = base.split("-", 1)
            if part not in PUB2_TYPES:
                raise ValueError(featset)
            blocks.append(_pub2_block(prefix, n, part))
        elif base in LIT_BASES:
            blocks.append(np.load(S.CACHE_DIR / f"{LIT_BASES[base]}_{n}.npz")["X"])
        else:
            raise ValueError(f"unknown feature set {featset!r}")
        meta["yi"].append(r["yi"])
        meta["perclos"].append(r["perclos"])
        meta["groups"].append(np.full(S.N_EPOCH, r["person"]))
        meta["rec"].append(np.full(S.N_EPOCH, i))
        meta["t"].append(r["t"])
    X = np.vstack(blocks).astype(np.float64)
    X[~np.isfinite(X)] = np.nan
    if base == "time":
        cols = ["t_onset_s"]
    elif base in LIT_BASES:
        z = np.load(S.CACHE_DIR / f"{LIT_BASES[base]}_feature_names.npz")
        cols, block = z["names"].tolist(), z["block"].tolist()
        keep = np.array([b not in L.DEFAULT_EXCLUDE_BLOCKS or "F05" in mods for b in block])
        if "noblinkdur" in mods or "noblinkall" in mods:
            keep &= np.array([b not in L.BLINK_BLOCKS for b in block])
        if "noblinkall" in mods:
            keep &= np.array(["blink" not in c for c in cols])
        X, cols = X[:, keep], [c for c, k in zip(cols, keep) if k]
    else:
        prefix, part = base.split("-", 1)
        cols = _pub2_names(prefix, part)
    out = {k: np.concatenate(v) for k, v in meta.items()}
    rec = out["rec"]
    for m in mods:
        if m in ("F05", "noblinkdur", "noblinkall"):
            if base not in LIT_BASES:
                raise ValueError(f"modifier {m!r} only applies to literature sets")
            continue
        if m == "kf":
            X = L.kalman_filter_rw(X, rec, 0.01)
        elif m == "ma":
            X = L.causal_moving_average(X, rec, 8)
        elif m == "rts":
            X = L.rts_smoother_rw(X, rec, 0.01)
        elif m == "ctx":
            C, cn = L.trailing_context(X, rec, cols)
            X, cols = np.hstack([X, C]), cols + cn
        else:
            raise ValueError(f"unknown modifier {m!r}")
    out.update({"X": X, "names": cols, "names_rec": names, "featset": featset})
    return out
