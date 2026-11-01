"""Build 00_COMMON/cache/<recording>.npz from the raw SEED-VIG files (read-only).

Run from this folder:  python -I build_cache.py
Checks done per recording (assertions in seedvig_eog.build_one):
  raw shape 885000 x 7, HEO == ch1-ch2, VEO == ch3-ch4, yi == PERCLOS rule.
Writes cache/manifest.csv and cache/SHA256SUMS.
"""
import csv
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import seedvig_eog as S  # noqa: E402


def main() -> None:
    S.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(S.build_one, S.RECORDINGS))
    with open(S.CACHE_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(S.CACHE_DIR / "SHA256SUMS", "w") as f:
        for n in S.RECORDINGS:
            h = hashlib.sha256((S.CACHE_DIR / f"{n}.npz").read_bytes()).hexdigest()
            f.write(f"{h}  {n}.npz\n")
    for r in rows:
        print(r)
    print("persons:", len({r["person"] for r in rows}), "recordings:", len(rows))


if __name__ == "__main__":
    main()
