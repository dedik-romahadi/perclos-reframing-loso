"""Extract Publication 2 features for all 23 recordings -> cache/pub2_<recording>.npz.

Run from this folder:  python -I build_pub2_features.py
"""
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pub2_features as P  # noqa: E402
import seedvig_eog as S  # noqa: E402


def one(name: str) -> str:
    rec = S.load_recording(name)
    f = P.extract_all(rec["eog"])
    for k, v in f.items():
        assert np.isfinite(v).all(), (name, k, np.argwhere(~np.isfinite(v))[:5])
    np.savez(S.CACHE_DIR / f"pub2_{name}.npz", **f)
    return f"{name}: " + ", ".join(f"{k} {v.shape}" for k, v in f.items())


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as ex:
        for line in ex.map(one, S.RECORDINGS):
            print(line, flush=True)
    np.savez(S.CACHE_DIR / "pub2_feature_names.npz",
             type1=np.array(P.feature_names("type1")), type2=np.array(P.feature_names("type2")),
             type3=np.array(P.feature_names("type3")))
    print("done")
