"""POST-HOC DESCRIPTIVE table (not pre-registered; no claim, no Holm family): integrated BA of every
non-linear train-once (TC) model and of A1 minus M2 (direct LDA retrained per cell), with the person-bootstrap
95 % CI and the two-sided sign-flip p, both tasks.

    python -I descriptive_tc_vs_m2.py

Written on 29 Oct 2026 in response to audit findings T1-P3 / T1-N5 so that the text can quote CIs for the
"non-linear TC models vs M2" observation instead of point estimates only. Writes results/tc_vs_m2_descriptive.csv.
Note: these contrasts also differ in input (M9/A4 feature sequences, M11/M13 raw EOG) and are therefore not
head-only comparisons.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import stats_t1 as ST  # noqa: E402
import t1_common as C  # noqa: E402

MODELS = [("M5", "tc-qhgb_litc-trad+kf", "classical"), ("M7", "tc-mlpln_litc-trad+kf", "hybrid"),
          ("M9", "tc-featgru_litc-trad", "hybrid"), ("M11", "tc-sinchybrid_trad6", "hybrid_e2e"),
          ("M13", "tc-resnet1dgru_trad6", "dl_e2e"), ("A4", "tc-featgru_litc-trad+kf", "hybrid"),
          ("A1", "tc-linln-iso_litc-trad+kf", "classical")]


def main():
    rows = []
    for task in ("three", "binarypair"):
        b = ST.integrated(task, ST.M2)
        for tag, mid, cls in MODELS:
            if not ST.exists_grid(task, mid):
                continue
            r = ST.two_sided(ST.integrated(task, mid), b)
            rows.append({"id": tag, "model_id": mid, "model_class": cls, "task": f"{task} (integrated)",
                         "contrast": f"{mid} - {ST.M2}", **r, "selection": "post_hoc",
                         "note": "descriptive only; not pre-registered; input differs from M2 except M5/M7/A1"})
            print(tag, task, {k: r[k] for k in ("estimate", "ci_lo", "ci_hi", "p_raw", "k_improved", "n")}, flush=True)
    C.write_csv(C.RES / "tc_vs_m2_descriptive.csv", rows)


if __name__ == "__main__":
    main()
