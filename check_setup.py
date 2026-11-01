"""Check that this computer is ready to run the analysis.

    python check_setup.py

Checks the Python version, the required packages, the SEED-VIG data folder (23 recordings with their PERCLOS and label
files) and whether a CUDA GPU is available. Prints what is missing and how to fix it; changes nothing.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RAW = HERE.parent / "DATASET" / "SADT-SEEDVIG" / "SEEDVIG"
PACKAGES = {"numpy": "2.4.1", "scipy": "1.17.0", "sklearn": "1.9.0", "torch": "2.11.0", "braindecode": "1.5.2",
            "pywt": "1.8.0", "joblib": "1.5.3"}
problems = []


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg, fix):
    print(f"  [!!]   {msg}")
    problems.append(fix)


print("1. Python")
if sys.version_info[:2] >= (3, 12):
    ok(f"Python {sys.version.split()[0]} (results were produced with 3.14.0)")
else:
    bad(f"Python {sys.version.split()[0]} is too old", "install Python 3.12 or newer (3.14 was used)")

print("2. Packages")
for mod, used in PACKAGES.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        ok(f"{mod} {v}" + ("" if v.split("+")[0] == used else f" (results used {used})"))
    except Exception:  # noqa: BLE001
        bad(f"{mod} is not installed", "pip install -r requirements.txt")

print("3. SEED-VIG data")
raw = pathlib.Path(os.environ.get("SEEDVIG_RAW_DIR", DEFAULT_RAW))
print(f"       folder: {raw}  (set SEEDVIG_RAW_DIR to use another folder)")
recs = sorted(p for p in raw.glob("*.mat")) if raw.is_dir() else []
if len(recs) == 23:
    ok("23 recording files (*.mat)")
else:
    bad(f"found {len(recs)} recording files, expected 23",
        "download SEED-VIG (https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html) and point SEEDVIG_RAW_DIR to the "
        "folder that holds the 23 .mat files and the sub-folders perclos_labels/ and LABELS/")
for sub in ("perclos_labels", "LABELS"):
    n = len(list((raw / sub).glob("*.mat"))) if (raw / sub).is_dir() else 0
    if n == 23:
        ok(f"{sub}/ with 23 files")
    else:
        bad(f"{sub}/ has {n} files, expected 23", f"copy the SEED-VIG folder {sub}/ next to the recordings")

print("4. GPU (needed only for the neural networks M7-M14 and A2-A4)")
try:
    import torch
    if torch.cuda.is_available():
        ok(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  [--]   no CUDA GPU: classical models run fine; the neural networks will be very slow on CPU")
except Exception:  # noqa: BLE001
    pass

print()
if problems:
    print("Not ready. Fix these first:")
    for p in dict.fromkeys(problems):
        print(f"  - {p}")
    sys.exit(1)
print("Ready. Continue with step 3 of the README.")
