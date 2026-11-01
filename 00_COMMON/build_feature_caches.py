"""Build the corrected / faithful Publication 2 caches and the literature caches.

    python -I build_feature_caches.py [--jobs 8] [--kinds pub2fix pub2orig lit_trad lit_fore litc_trad litc_fore]
                                      [--rebuild] [--pub2fix-type3-only] [--max-files N]

Outputs (cache/, one file per recording, skipped when present -> resumable):
  pub2fix_<rec>.npz   type1 (885x168), type2 (885x234), type3 (885x15, NaN = no event), qc (885x2)
  pub2orig_<rec>.npz  faithful (as-published, bugs included) type1/type2/type3; input = raw
                      float64 EOG from the SEED-VIG .mat when readable, else the float32 cache
  lit_trad_<rec>.npz  X (885 x D) literature features, traditional montage (HEO, -VEO), OFFLINE
                      zero-phase variant
  lit_fore_<rec>.npz  X (885 x D) literature features, forehead montage (HEOf, VEOf, ch4-ch7), OFFLINE
  litc_trad_<rec>.npz, litc_fore_<rec>.npz  same columns, CAUSAL variant (forward-only filters)
  <kind>_feature_names.npz   column names (+ 'block' F-codes for lit_*/litc_*)
Each file stores `code_version`. --rebuild recomputes files whose code_version differs from
CODE_VERSION[kind] (one recording at a time, temporary name + rename), so an interrupted
rebuild resumes. --pub2fix-type3-only recomputes only the type-3 block of pub2fix (type 1/2
are unchanged since version 1).
and results/feature_check/cache_summary.csv (per-recording event rates, NaN counts, QC).
Files are written to a temporary name and renamed, so an interrupted run never leaves a
partial cache. pub2_features.py and cache/pub2_*.npz are not touched.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eog_features_lit as L  # noqa: E402
import pub2_features_fixed as F  # noqa: E402
import seedvig_eog as S  # noqa: E402

KINDS = ["pub2fix", "pub2orig", "lit_trad", "lit_fore", "litc_trad", "litc_fore"]
SUMMARY = HERE / "results" / "feature_check" / "cache_summary.csv"
# version 2 (review fixes): pub2fix type-3 fixations segmented on the continuous record; lit F13
# sub-1-Hz bands from a trailing 64-s multitaper and causal fallbacks; litc_* added.
CODE_VERSION = {"pub2fix": 2, "pub2orig": 1, "lit_trad": 2, "lit_fore": 2, "litc_trad": 3, "litc_fore": 3}


def out_path(kind: str, name: str) -> pathlib.Path:
    return S.CACHE_DIR / f"{kind}_{name}.npz"


def is_current(kind: str, name: str) -> bool:
    p = out_path(kind, name)
    if not p.exists():
        return False
    z = np.load(p)
    v = int(z["code_version"]) if "code_version" in z.files else 1
    return v == CODE_VERSION[kind]


def load_raw64(name: str) -> tuple[np.ndarray, str]:
    """Raw EOG as float64 (n_ep, 7, 1000) from the provider .mat (read-only), else the cache."""
    try:
        import scipy.io as sio

        E = sio.loadmat(S.RAW_DIR / f"{name}.mat", variable_names=["EOG"], squeeze_me=True,
                        struct_as_record=False)["EOG"]
        eog = np.asarray(E.eog, np.float64)
        assert eog.shape == (S.N_EPOCH * S.N_SAMP, 7)
        return eog.reshape(S.N_EPOCH, S.N_SAMP, 7).transpose(0, 2, 1), "raw_mat_float64"
    except Exception as exc:  # pragma: no cover - fallback path
        return S.load_recording(name)["eog"].astype(np.float64), f"cache_float32 ({type(exc).__name__})"


def _save(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def one(kind: str, name: str, pub2fix_type3_only: bool = False) -> dict:
    t0 = time.time()
    row = {"kind": kind, "name": name, "code_version": CODE_VERSION[kind]}
    if kind == "pub2orig":
        eog, src = load_raw64(name)
        f = F.extract_all(eog, "faithful")
        row["source"] = src
        row["type3_status"] = ";".join(f["type3_status"].tolist())
        for k in ("type1", "type2", "type3"):
            row[f"nonfinite_{k}"] = int((~np.isfinite(f[k])).sum())
        _save(out_path(kind, name), type1=f["type1"], type2=f["type2"], type3=f["type3"],
              type3_status=f["type3_status"], source=np.array(src), code_version=CODE_VERSION[kind])
    elif kind == "pub2fix":
        eog = S.load_recording(name)["eog"]
        old = out_path(kind, name)
        if pub2fix_type3_only and old.exists():
            z = dict(np.load(old))
            t3, info = F.fixed_type3(eog)
            f = {"type1": z["type1"], "type2": z["type2"], "type3": t3, "qc": z["qc"],
                 "type3_info": np.array([f"{k}={v}" for k, v in info.items()])}
        else:
            f = F.extract_all(eog, "fixed")
        for k in ("type1", "type2", "type3"):
            row[f"nonfinite_{k}"] = int((~np.isfinite(f[k])).sum())
        row["qc_flat"], row["qc_extreme"] = (int(v) for v in f["qc"].sum(0))
        row["type3_info"] = ";".join(f["type3_info"].tolist())
        _save(out_path(kind, name), type1=f["type1"], type2=f["type2"], type3=f["type3"], qc=f["qc"],
              type3_info=f["type3_info"], code_version=CODE_VERSION[kind])
    else:
        montage = kind.split("_")[1]
        eog = S.load_recording(name)["eog"]
        r = L.extract_recording(eog, montage, causal=kind.startswith("litc_"))
        X = r["X"]
        row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["info"].items()})
        row["nonfinite_X"] = int((~np.isfinite(X)).sum())
        row["nonfinite_cols"] = int((~np.isfinite(X)).any(0).sum())
        _save(out_path(kind, name), X=X.astype(np.float64), info=np.array([f"{k}={v}" for k, v in r["info"].items()]),
              code_version=CODE_VERSION[kind])
        _save(S.CACHE_DIR / f"{kind}_feature_names.npz", names=np.array(r["names"]), block=np.array(r["block"]))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _parse_kv(arr) -> dict:
    out = {}
    for item in np.asarray(arr).tolist():
        k, _, v = str(item).partition("=")
        try:
            fv = float(v)
            out[k] = int(fv) if fv.is_integer() and "." not in v else round(fv, 4)
        except ValueError:
            out[k] = v
    return out


def row_from_cache(kind: str, name: str) -> dict:
    """Summary row rebuilt from a cache file (so an interrupted build never loses rows)."""
    z = np.load(out_path(kind, name))
    row = {"kind": kind, "name": name, "code_version": int(z["code_version"]) if "code_version" in z.files else 1}
    if kind in ("pub2fix", "pub2orig"):
        for k in ("type1", "type2", "type3"):
            row[f"nonfinite_{k}"] = int((~np.isfinite(z[k])).sum())
        if kind == "pub2fix":
            row["qc_flat"], row["qc_extreme"] = (int(v) for v in z["qc"].sum(0))
            row["type3_info"] = ";".join(z["type3_info"].tolist())
        else:
            row["source"] = str(z["source"])
            row["type3_status"] = ";".join(z["type3_status"].tolist())
    else:
        row.update(_parse_kv(z["info"]))
        X = z["X"]
        row["nonfinite_X"] = int((~np.isfinite(X)).sum())
        row["nonfinite_cols"] = int((~np.isfinite(X)).any(0).sum())
        if kind.endswith("_fore"):
            v = L.montage_signals(S.load_recording(name)["eog"], "fore")["V"]
            row["p99_absVEOf_0.5-20Hz_uV"] = round(float(np.percentile(np.abs(L.zero_phase(v, 0.5, 20.0)), 99)), 2)
    return row


def resummarize(kinds) -> None:
    rows = [row_from_cache(k, n) for k in kinds for n in S.RECORDINGS if out_path(k, n).exists()]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("kind", "name"), k))
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["kind"], S.RECORDINGS.index(r["name"]))))
    print(f"{SUMMARY}: {len(rows)} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--rebuild", action="store_true", help="recompute files whose code_version is stale")
    ap.add_argument("--pub2fix-type3-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="stop after this many files (chunked runs)")
    ap.add_argument("--resummarize", action="store_true", help="only rewrite cache_summary.csv from the caches")
    a = ap.parse_args()
    if a.resummarize:
        resummarize(KINDS)
        return
    a.jobs = min(a.jobs, 8)
    done = is_current if a.rebuild else (lambda k, n: out_path(k, n).exists())
    todo = [(k, n) for k in a.kinds for n in S.RECORDINGS if not done(k, n)]
    if a.max_files:
        todo = todo[:a.max_files]
    print(f"{len(todo)} cache files to build with {a.jobs} processes", flush=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one, k, n, a.pub2fix_type3_only): (k, n) for k, n in todo}
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            print(r, flush=True)
    names = {"type1": np.array(F.feature_names("type1")), "type2": np.array(F.feature_names("type2")),
             "type3": np.array(F.feature_names("type3"))}
    for k in ("pub2fix", "pub2orig"):
        if not (S.CACHE_DIR / f"{k}_feature_names.npz").exists():
            _save(S.CACHE_DIR / f"{k}_feature_names.npz", **names)
    resummarize(KINDS)
    missing = [(k, n) for k in a.kinds for n in S.RECORDINGS if not done(k, n)]
    print("missing:", missing if missing else "none")
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
