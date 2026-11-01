"""All T1 statistics from the saved prediction / grid / CDF files (PREREG sections 7-9).

    python -I stats_t1.py

Writes results/: grid_metrics.csv, integrated.csv, hypotheses.csv, calibration.csv,
distributional.csv, regression.csv, coherence.csv, single_threshold.csv, cell_delta.csv, rank_stability.csv,
controls.csv, robustness_primary.csv. (results/person_metrics.csv, the per-person canonical metrics of every
leaderboard row, is written by make_leaderboard.py.) Tests that involve a model without predictions are removed from their
family (Holm over the tests run) and listed in results/missing_models.txt.
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import seedvig_eog as S  # noqa: E402
import summarize as SM  # noqa: E402

warnings.filterwarnings("ignore")
DELTA = 0.02
FC = C.COMMON / "results" / "feature_check"
B = C.base()
Y = B["perclos"]
G = B["groups"]
VALID2 = C.valid_binary_cells()
BIN17 = C.canonical_binary_persons()
P21 = [int(p) for p in B["persons"]]
MISSING: list[str] = []

# model ids
M1, M2 = "tc-linln_litc-trad+kf", "direct-lda-grid_litc-trad+kf"
TC_MODELS = {"M1": M1, "M5": "tc-qhgb_litc-trad+kf", "M7": "tc-mlpln_litc-trad+kf", "M9": "tc-featgru_litc-trad",
             "M11": "tc-sinchybrid_trad6", "M13": "tc-resnet1dgru_trad6", "M16": "tc-linln_time",
             "M18": "tc-linln_litc-trad+kf+noblinkall", "M20": "tc-linln_pub2-type1+kf", "M22": "tc-linln_litc-fore+kf",
             "A4": "tc-featgru_litc-trad+kf", "A1": "tc-linln-iso_litc-trad+kf"}
DIRECT_GRIDS = {"M2": M2, "M17": "direct-lda-grid_time", "M19": "direct-lda-grid_litc-trad+kf+noblinkall",
                "M21": "direct-lda-grid_pub2-type1+kf", "M23": "direct-lda-grid_litc-fore+kf"}


def exists_grid(task, mid):
    return (C.GRID / f"{task}_{mid}.npz").exists()


def exists_pred(task, mid):
    return pred_path(task, mid).exists()


def pred_path(task, mid):
    """T1 prediction files only (the reference rows R1, R3, R4, R5 were re-run in T1's pipeline, DEVIATIONS D1)."""
    return C.PREDS / f"{task}_{mid}.npz"


def canon_pp(task, mid) -> dict:
    with np.load(pred_path(task, mid)) as z:
        return SM.per_person({k: z[k] for k in ("y", "pred", "score", "groups")}, task)


def metric(pp: dict, key="bacc") -> dict:
    return {p: v[key] for p, v in pp.items()}


# ============================================================================== grid metrics
_GRID_CACHE: dict = {}


def grid_pp(task, mid, want_f1=False, want_qwk=False):
    """{cell_index: {person: {bacc, f1_macro, qwk, n}}} for a grid file."""
    key = (task, mid, want_f1, want_qwk)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    pred, _ = C.load_grid(task, mid)
    out = {}
    for j, c in enumerate(C.task_cells(task)):
        if task == "binarypair" and not VALID2[j]:
            continue
        if (pred[:, j] < 0).all():
            continue
        out[j] = C.cell_person_metrics(Y, G, pred[:, j], task, c, want_f1=want_f1, want_qwk=want_qwk)
    _GRID_CACHE[key] = out
    return out


def integrated(task, mid, key="bacc", cells=None, persons=None):
    gp = grid_pp(task, mid, want_f1=key == "f1_macro", want_qwk=key == "qwk")
    acc = {}
    for j, rows in gp.items():
        if cells is not None and j not in cells:
            continue
        for p, r in rows.items():
            acc.setdefault(p, []).append(r[key])
    out = {p: float(np.nanmean(v)) for p, v in acc.items()}
    if persons is None:
        persons = P21 if task == "three" else BIN17
    return {p: out[p] for p in persons if p in out}


def single_integrated(mid):
    """Single-threshold grid: per-threshold person BA; integrated over thresholds for persons scored at >= 6."""
    gp = grid_pp("single", mid)
    acc = {}
    for j, rows in gp.items():
        for p, r in rows.items():
            acc.setdefault(p, []).append(r["bacc"])
    return {p: float(np.mean(v)) for p, v in acc.items() if len(v) >= 6}, gp


# ============================================================================== tests
def ni_test(a: dict, b: dict, delta=DELTA, persons=None):
    ps, x, y = C.paired(a, b)
    if persons is not None:
        keep = [i for i, p in enumerate(ps) if p in persons]
        ps, x, y = [ps[i] for i in keep], x[keep], y[keep]
    d = x - y
    m, lo, hi = C.ci(d)
    p = C.perm_p(d + delta, np.zeros_like(d))
    return {"n": len(d), "estimate": m, "ci_lo": lo, "ci_hi": hi, "p_raw": p, "k_improved": int(np.sum(d > 0)),
            "mean_A": float(x.mean()), "mean_B": float(y.mean()), "delta_star": -lo, "mean_d_plus_delta": m + delta}


def two_sided(a: dict, b: dict):
    ps, x, y = C.paired(a, b)
    d = x - y
    m, lo, hi = C.ci(d)
    return {"n": len(d), "estimate": m, "ci_lo": lo, "ci_hi": hi, "p_raw": C.perm_p(x, y),
            "k_improved": int(np.sum(d > 0)), "mean_A": float(x.mean()), "mean_B": float(y.mean())}


def holm_family(rows):
    ok = [r for r in rows if r.get("p_raw") is not None and np.isfinite(r["p_raw"])]
    adj = S.holm([r["p_raw"] for r in ok]) if ok else []
    for r, a in zip(ok, adj):
        r["p_holm"] = a
    return rows


def ni_decision(r):
    if r["mean_d_plus_delta"] > 0 and r["p_holm"] < 0.05 and r["ci_lo"] > -DELTA:
        return "non-inferior"
    if r["ci_hi"] <= -DELTA:
        return "inferior (CI entirely below -delta)"
    return "inconclusive (not shown non-inferior)"


# ============================================================================== calibration
def rps3(P, y):
    cum = np.cumsum(P, 1)[:, :2]
    o = np.stack([(y <= 0), (y <= 1)], 1).astype(float)
    return np.sum((cum - o) ** 2, 1) / 2


def ece_classwise(P, y, nb=10):
    vals = []
    for c in range(P.shape[1]):
        p = P[:, c]
        o = (y == c).astype(float)
        bins = np.minimum((p * nb).astype(int), nb - 1)
        e = 0.0
        for k in range(nb):
            m = bins == k
            if m.any():
                e += m.mean() * abs(o[m].mean() - p[m].mean())
        vals.append(e)
    return float(np.mean(vals))


def calib_rows(name, task, P, y, groups, persons):
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    out = {}
    for p in persons:
        m = groups == p
        Pp, yp = P[m], y[m]
        r = {"nll": float(-np.mean(np.log(Pp[np.arange(len(yp)), yp]))),
             "brier": float(np.mean(np.sum((Pp - np.eye(P.shape[1])[yp]) ** 2, 1)))}
        if task == "three":
            r["rps"] = float(np.mean(rps3(Pp, yp)))
            r["ece_cw"] = ece_classwise(Pp, yp)
        else:
            r["ece_cw"] = ece_classwise(Pp, yp)
        out[p] = r
    return out


def train_prior(task, person):
    keep = np.ones(len(G), bool) if task == "three" else B["yi"] != 2
    y = (B["yi"] - 1) if task == "three" else (B["yi"] == 3).astype(int)
    m = keep & (G != person)
    nc = 3 if task == "three" else 2
    return np.bincount(y[m].astype(int), minlength=nc) / m.sum()


def reweight_to_prior(task, z):
    """Balanced-trained classifiers: p_c proportional to p~_c * prior_c (outer-training prevalence)."""
    P = np.asarray(z["score"], float).copy()
    for p in np.unique(z["groups"]):
        m = z["groups"] == p
        pr = train_prior(task, int(p))
        Q = P[m] * pr[None, :]
        P[m] = Q / Q.sum(1, keepdims=True)
    return P


# ============================================================================== main analyses
def main():
    rows_h = []          # hypotheses.csv
    out_csv = {}

    # ---------------- grid metrics (long table) + integrated table
    gm_rows, integ_rows = [], []
    grid_models = {**TC_MODELS, **DIRECT_GRIDS}
    for tag, mid in grid_models.items():
        for task in ("three", "binarypair", "single"):
            if not exists_grid(task, mid):
                continue
            gp = grid_pp(task, mid, want_f1=task == "three" and tag in ("M1", "M2"),
                         want_qwk=task == "three" and tag in ("M1", "M2"))
            cells = C.task_cells(task)
            for j, rows in gp.items():
                for p, r in rows.items():
                    gm_rows.append({"model": mid, "task": task, "cell": str(cells[j]), "person": p,
                                    "bacc": r["bacc"], "f1_macro": r.get("f1_macro", ""), "qwk": r.get("qwk", ""),
                                    "n": r["n"]})
        for task in ("three", "binarypair"):
            if exists_grid(task, mid):
                v = integrated(task, mid)
                m, lo, hi = C.ci(list(v.values()))
                integ_rows.append({"id": tag, "model_id": mid, "task": task, "n_persons": len(v), "iba": m,
                                   "iba_lo": lo, "iba_hi": hi})
    for task, mids in (("three", ["direct-lda-grid_pub2orig-type1", "direct-lda-grid_pub2orig-type2",
                                  "direct-lda-grid_pub2orig-type3", "direct-lda-grid_pub2orig-all"]),):
        for mid in mids:
            if exists_grid(task, mid):
                v = integrated(task, mid)
                m, lo, hi = C.ci(list(v.values()))
                integ_rows.append({"id": "H7b", "model_id": mid, "task": task, "n_persons": len(v), "iba": m,
                                   "iba_lo": lo, "iba_hi": hi})
    C.write_csv(C.RES / "grid_metrics.csv", gm_rows)
    C.write_csv(C.RES / "integrated.csv", integ_rows)

    # ---------------- F1 primary (H1, H2)
    f1 = []
    if exists_grid("three", M1) and exists_grid("three", M2):
        r = ni_test(integrated("three", M1), integrated("three", M2))
        f1.append({"hypothesis": "H1", "family": "F1", "task": "three (integrated, 49 cells)",
                   "contrast": f"{M1} - {M2}", **r})
        r = ni_test(integrated("binarypair", M1), integrated("binarypair", M2))
        f1.append({"hypothesis": "H2", "family": "F1", "task": "binary-pair (integrated, 47 valid cells)",
                   "contrast": f"{M1} - {M2}", **r})
        holm_family(f1)
        for r in f1:
            r["decision"] = ni_decision(r)
        rows_h += f1
        # F1s superiority (only NI-established)
        f1s = []
        for r in f1:
            if r["decision"] == "non-inferior":
                task = "three" if r["hypothesis"] == "H1" else "binarypair"
                s = two_sided(integrated(task, M1), integrated(task, M2))
                f1s.append({"hypothesis": r["hypothesis"] + "s", "family": "F1s", "task": r["task"],
                            "contrast": r["contrast"] + " (superiority)", **s})
        holm_family(f1s)
        for r in f1s:
            r["decision"] = ("superior" if (r["p_holm"] < 0.05 and r["estimate"] > 0 and r["ci_lo"] > 0)
                             else "not superior")
        rows_h += f1s
    else:
        MISSING.append("primary M1/M2 grids missing")

    # ---------------- robustness of the primary (descriptive)
    rob = []
    if exists_grid("three", M1) and exists_grid("three", M2):
        from scipy.stats import wilcoxon
        for task in ("three", "binarypair"):
            ps, a, b = C.paired(integrated(task, M1), integrated(task, M2))
            rob.append({"analysis": "R1 Wilcoxon signed-rank (two-sided, on delta + 0.02)", "task": task,
                        "n": len(a), "estimate": float(np.mean(a - b)),
                        "p": float(wilcoxon(a - b + DELTA).pvalue)})
        for key in ("f1_macro", "qwk"):
            r = ni_test(integrated("three", M1, key), integrated("three", M2, key))
            rob.append({"analysis": f"R2 integrated {key} (three)", "task": "three", **r})
        central = [C.CELLS.index((a, b)) for a in (0.30, 0.35, 0.40) for b in (0.65, 0.70, 0.75)]
        for task in ("three", "binarypair"):
            r = ni_test(integrated(task, M1, cells=central), integrated(task, M2, cells=central))
            rob.append({"analysis": "R6 central 3x3 sub-grid (0.30-0.40 x 0.65-0.75)", "task": task, **r})
        # R3 alternative decision rules for M1
        with np.load(C.CDFD / f"{M1}.npz") as z:
            Fcut = z["Fcut"]
        for rule in ("argmax", "median"):
            g = C.tc_grid(Fcut, rule)
            for task in ("three", "binarypair"):
                mid = f"{M1}__{rule}"
                C.save_grid(mid, {task: g[task]})
                r = ni_test(integrated(task, mid), integrated(task, M2))
                rob.append({"analysis": f"R3 M1 decision rule = {rule}", "task": task, **r})
        # R4 isotonic PIT recalibration of M1 (ablation A1, tier 3; code/run_a1.py)
        a1 = TC_MODELS["A1"]
        for task in ("three", "binarypair"):
            if exists_grid(task, a1):
                r = ni_test(integrated(task, a1), integrated(task, M2))
                rob.append({"analysis": "R4 M1 + isotonic PIT recalibration (A1)", "task": task, **r})
            else:
                MISSING.append(f"R4/A1 {task} not run")
    C.write_csv(C.RES / "robustness_primary.csv", rob)

    # ---------------- per-cell delta maps (descriptive)
    cd = []
    if exists_grid("three", M1) and exists_grid("three", M2):
        for task in ("three", "binarypair"):
            ga, gb = grid_pp(task, M1), grid_pp(task, M2)
            for j in sorted(set(ga) & set(gb)):
                pa = {p: r["bacc"] for p, r in ga[j].items()}
                pb = {p: r["bacc"] for p, r in gb[j].items()}
                ps, a, b = C.paired(pa, pb)
                if task == "binarypair":
                    keep = [i for i, p in enumerate(ps) if p in BIN17]
                    a, b = a[keep], b[keep]
                if len(a) == 0:
                    continue
                m, lo, hi = C.ci(a - b)
                cd.append({"task": task, "t1": C.CELLS[j][0], "t2": C.CELLS[j][1], "n": len(a), "ba_M1": a.mean(),
                           "ba_M2": b.mean(), "delta": m, "ci_lo": lo, "ci_hi": hi})
    C.write_csv(C.RES / "cell_delta.csv", cd)

    # ---------------- single-threshold grid (descriptive)
    st = []
    if exists_grid("single", M1) and exists_grid("single", M2):
        ia, gpa = single_integrated(M1)
        ib, gpb = single_integrated(M2)
        for j, th in enumerate(C.SINGLE):
            if j in gpa and j in gpb:
                ps, a, b = C.paired({p: r["bacc"] for p, r in gpa[j].items()},
                                    {p: r["bacc"] for p, r in gpb[j].items()})
                m, lo, hi = C.ci(a - b)
                st.append({"threshold": th, "n": len(a), "ba_M1": a.mean(), "ba_M2": b.mean(), "delta": m,
                           "ci_lo": lo, "ci_hi": hi})
        r = two_sided(ia, ib)
        st.append({"threshold": "integrated (>= 6 of 11)", "n": r["n"], "ba_M1": r["mean_A"], "ba_M2": r["mean_B"],
                   "delta": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_raw": r["p_raw"]})
    C.write_csv(C.RES / "single_threshold.csv", st)

    # ---------------- H3 (F3) and F3x: canonical BA, TC vs direct
    pairs3 = [("H3.1", "three", M1, "ref-litc-trad+kf_lda", "classical"),
              ("H3.2", "binary", M1, "ref-litc-trad+kf_lda", "classical"),
              ("H3.3", "three", "tc-featgru_litc-trad", "direct-featgru_litc-trad", "hybrid"),
              ("H3.4", "binary", "tc-featgru_litc-trad", "direct-featgru_litc-trad", "hybrid"),
              ("H3.5", "three", "tc-sinchybrid_trad6", "direct-sinchybrid_trad6", "hybrid_e2e"),
              ("H3.6", "binary", "tc-sinchybrid_trad6", "direct-sinchybrid_trad6", "hybrid_e2e"),
              ("H3.7", "three", "tc-resnet1dgru_trad6", "direct-resnet1dgru_trad6", "dl_e2e"),
              ("H3.8", "binary", "tc-resnet1dgru_trad6", "direct-resnet1dgru_trad6", "dl_e2e")]
    pairs3x = [("H3.9", "three", "tc-mlpln_litc-trad+kf", "direct-mlp_litc-trad+kf", "hybrid (per-epoch MLP)"),
               ("H3.10", "binary", "tc-mlpln_litc-trad+kf", "direct-mlp_litc-trad+kf", "hybrid (per-epoch MLP)"),
               ("H3.11", "three", "tc-qhgb_litc-trad+kf", "direct-hgb_litc-trad+kf", "classical (HGB)"),
               ("H3.12", "binary", "tc-qhgb_litc-trad+kf", "direct-hgb_litc-trad+kf", "classical (HGB)")]
    for fam, pairs in (("F3", pairs3), ("F3x", pairs3x)):
        fr = []
        for h, task, a, b, cls in pairs:
            if not (exists_pred(task, a) and exists_pred(task, b)):
                MISSING.append(f"{h} ({a} vs {b}, {task}) not run")
                continue
            r = ni_test(metric(canon_pp(task, a)), metric(canon_pp(task, b)))
            fr.append({"hypothesis": h, "family": fam, "task": f"{task} (canonical)", "contrast": f"{a} - {b}",
                       "model_class": cls, **r})
        holm_family(fr)
        for r in fr:
            r["decision"] = ni_decision(r)
        rows_h += fr

    # ---------------- H4 (F4, F4b): integrated BA among class representatives
    reps = {"classical": M1, "hybrid": "tc-featgru_litc-trad", "hybrid_e2e": "tc-sinchybrid_trad6",
            "dl_e2e": "tc-resnet1dgru_trad6"}
    pairs4 = [("H4a", "hybrid", "classical"), ("H4b", "classical", "dl_e2e"), ("H4c", "hybrid_e2e", "dl_e2e"),
              ("H4 (hybrid vs hybrid_e2e)", "hybrid", "hybrid_e2e"), ("H4 (classical vs hybrid_e2e)", "classical",
                                                                       "hybrid_e2e"),
              ("H4 (hybrid vs dl_e2e)", "hybrid", "dl_e2e")]
    best_class = {}
    for fam, task in (("F4", "three"), ("F4b", "binarypair")):
        fr = []
        for h, ca, cb in pairs4:
            a, b = reps[ca], reps[cb]
            if not (exists_grid(task, a) and exists_grid(task, b)):
                MISSING.append(f"{fam} {h} ({a} vs {b}) not run")
                continue
            r = two_sided(integrated(task, a), integrated(task, b))
            fr.append({"hypothesis": h, "family": fam, "task": f"{task} (integrated)", "contrast": f"{a} - {b}",
                       "class_A": ca, "class_B": cb, **r})
        holm_family(fr)
        for r in fr:
            sig = r["p_holm"] < 0.05
            if r["hypothesis"] in ("H4a", "H4b", "H4c"):
                r["decision"] = "supported" if (sig and r["estimate"] > 0) else (
                    "opposite direction (Holm-significant)" if sig else "not supported")
            else:
                r["decision"] = ("A better" if r["estimate"] > 0 else "B better") if sig else "no significant difference"
        rows_h += fr
        wins = {c: 0 for c in reps}
        for r in fr:
            if r["p_holm"] < 0.05:
                wins[r["class_A"] if r["estimate"] > 0 else r["class_B"]] += 1
        best = [c for c, w in wins.items() if w == 3]
        best_class[task] = best[0] if best else "none shown"

    # ---------------- H5 (F5): RPS at the canonical cut-points + calibration table
    cal_rows = []
    calib_models = {"M1": (M1, "tc"), "M3": ("lda-ts_litc-trad+kf", "natural"), "M4": ("ordlogit_litc-trad+kf", "natural"),
                    "R4/M2": ("ref-litc-trad+kf_lda", "balanced"), "M5": ("tc-qhgb_litc-trad+kf", "tc"),
                    "M6": ("direct-hgb_litc-trad+kf", "balanced"), "M7": ("tc-mlpln_litc-trad+kf", "tc"),
                    "M8": ("direct-mlp_litc-trad+kf", "balanced"), "M9": ("tc-featgru_litc-trad", "tc"),
                    "M10": ("direct-featgru_litc-trad", "balanced"), "M11": ("tc-sinchybrid_trad6", "tc"),
                    "M12": ("direct-sinchybrid_trad6", "balanced"), "M13": ("tc-resnet1dgru_trad6", "tc"),
                    "M14": ("direct-resnet1dgru_trad6", "balanced")}
    cal_pp = {}
    for tag, (mid, kind) in calib_models.items():
        for task in ("three", "binary"):
            if not exists_pred(task, mid):
                continue
            with np.load(pred_path(task, mid)) as z:
                zz = {k: z[k] for k in ("y", "score", "groups")}
            P = reweight_to_prior(task, zz) if kind == "balanced" else np.asarray(zz["score"], float)
            if tag == "R4/M2":
                pass
            persons = P21 if task == "three" else BIN17
            pp = calib_rows(mid, task, P, zz["y"].astype(int), zz["groups"], persons)
            cal_pp[(tag, task)] = pp
            row = {"id": tag, "model_id": mid, "task": task, "probabilities": {"tc": "derived from one CDF",
                                                                                 "natural": "natural (training prevalence)",
                                                                                 "balanced": "re-weighted to training prevalence"}[kind],
                   "n_persons": len(pp)}
            for k in next(iter(pp.values())):
                m, lo, hi = C.ci([v[k] for v in pp.values()])
                row[k], row[k + "_lo"], row[k + "_hi"] = m, lo, hi
            cal_rows.append(row)
    C.write_csv(C.RES / "calibration.csv", cal_rows)
    f5 = []
    for h, other in (("H5a", "M3"), ("H5b", "M4")):
        if ("M1", "three") in cal_pp and (other, "three") in cal_pp:
            a = {p: v["rps"] for p, v in cal_pp[("M1", "three")].items()}
            b = {p: v["rps"] for p, v in cal_pp[(other, "three")].items()}
            r = two_sided(a, b)
            r["k_improved"] = int(sum(a[p] < b[p] for p in a))       # lower RPS = improvement
            f5.append({"hypothesis": h, "family": "F5", "task": "three (canonical RPS)",
                       "contrast": f"RPS {M1} - {calib_models[other][0]}", **r})
        else:
            MISSING.append(f"{h} not run")
    holm_family(f5)
    for r in f5:
        r["decision"] = "supported" if (r["p_holm"] < 0.05 and r["estimate"] < 0 and r["ci_hi"] < 0) else "not supported"
    rows_h += f5

    # ---------------- H6 (F6): Publication 2 nested rebuild
    f6 = []
    if exists_pred("three", "pub2nested_pub2orig"):
        nest = metric(canon_pp("three", "pub2nested_pub2orig"))
        for h, other, direction in (("H6a", "ref-litc-trad+kf_lda", "lower"), ("H6c", "ref-pub2orig-type2_qsvm", "higher")):
            if exists_pred("three", other):
                r = two_sided(nest, metric(canon_pp("three", other)))
                f6.append({"hypothesis": h, "family": "F6", "task": "three (canonical)",
                           "contrast": f"pub2nested_pub2orig - {other}", "expected": direction, **r})
        holm_family(f6)
        for r in f6:
            ok = r["p_holm"] < 0.05 and ((r["estimate"] < 0) if r["expected"] == "lower" else (r["estimate"] > 0))
            r["decision"] = "supported" if ok else "not supported"
        rows_h += f6
        r = two_sided(nest, metric(canon_pp("three", "ref-pub2orig-all_lda")))
        r.update({"hypothesis": "H6b", "family": "CI-based", "task": "three (canonical)",
                  "contrast": "pub2nested_pub2orig - ref-pub2orig-all_lda", "p_raw": None,
                  "decision": "non-superiority shown (CI upper < +0.02)" if r["ci_hi"] < DELTA else "not shown"})
        rows_h.append(r)
        import csv
        with open(C.RES / "pub2_nested_selection.csv", encoding="utf-8") as f:
            sel = list(csv.DictReader(f))
        nq = sum(s["clf"] == "qsvm" for s in sel)
        rows_h.append({"hypothesis": "H6d", "family": "descriptive", "task": "three",
                       "contrast": "folds in which QSVM was chosen", "n": len(sel), "estimate": nq,
                       "decision": "supported (<= 10/21)" if nq <= 10 else "not supported (> 10/21)"})
    else:
        MISSING.append("H6 (M15 pub2nested) not run")
    # M15 binary (tier 3): descriptive only (PREREG 8, H6)
    if exists_pred("binary", "pub2nested_pub2orig"):
        nestb = metric(canon_pp("binary", "pub2nested_pub2orig"))
        for other in ("ref-litc-trad+kf_lda", "ref-pub2orig-type2_qsvm", "ref-pub2orig-all_lda"):
            if exists_pred("binary", other):
                r = two_sided(nestb, metric(canon_pp("binary", other)))
                r.update({"hypothesis": "H6 binary", "family": "descriptive", "task": "binary (canonical)",
                          "contrast": f"pub2nested_pub2orig - {other}",
                          "decision": "descriptive (tier 3; no Holm family)"})
                rows_h.append(r)
    else:
        MISSING.append("M15 binary (tier 3) not run")

    # ---------------- H7 rank stability
    rank_rows = rank_stability(rows_h)
    C.write_csv(C.RES / "rank_stability.csv", rank_rows)

    # ---------------- H9 controls (F9) + permutation
    ctrl = []
    lb_models = leaderboard_models()
    f9 = []
    for task in ("three", "binary"):
        if not exists_pred(task, "ref-time_lda"):
            continue
        tm = metric(canon_pp(task, "ref-time_lda"))
        for mid, cls in lb_models:
            # F9 = the PREREG section 13 leaderboard rows only (20 non-control models x 2 tasks = 40 tests);
            # the tier-3 A4 ablation rows are not members (audit finding T1-P2, 29 Oct 2026; DEVIATIONS D12)
            if cls == "control" or mid in A4_IDS or not exists_pred(task, mid):
                continue
            r = two_sided(metric(canon_pp(task, mid)), tm)
            f9.append({"hypothesis": "H9a", "family": "F9", "task": f"{task} (canonical)",
                       "contrast": f"{mid} - ref-time_lda", **r})
    holm_family(f9)
    for r in f9:
        r["decision"] = "beats time-only" if (r["p_holm"] < 0.05 and r["estimate"] > 0) else "not shown"
    rows_h += f9
    for s in range(5):
        for task in ("three", "binary"):
            for model in ("tc-linln", "lda"):
                f = C.RES / "controls" / f"perm{s}_{task}_{model}.npz"
                if f.exists():
                    with np.load(f) as z:
                        pp = SM.per_person({k: z[k] for k in ("y", "pred", "score", "groups")}, task)
                    m, lo, hi = C.ci([v["bacc"] for v in pp.values()])
                    ctrl.append({"control": "label permutation (circular shift)", "seed": s, "task": task,
                                 "model": model, "n": len(pp), "bacc": m, "bacc_lo": lo, "bacc_hi": hi,
                                 "chance": 0.333 if task == "three" else 0.5,
                                 "within_chance_pm_0.05": abs(m - (1 / 3 if task == "three" else 0.5)) <= 0.05})
    for tag, mid in (("M16", "tc-linln_time"), ("M17", "direct-lda-grid_time")):
        for task in ("three", "binarypair"):
            if exists_grid(task, mid):
                v = integrated(task, mid)
                m, lo, hi = C.ci(list(v.values()))
                ctrl.append({"control": f"time-only {tag} integrated", "seed": "", "task": task, "model": mid,
                             "n": len(v), "bacc": m, "bacc_lo": lo, "bacc_hi": hi})
    C.write_csv(C.RES / "controls.csv", ctrl)

    # ---------------- H10 (F10) robustness inputs
    f10 = []
    for h, a, b in (("H10 noblinkall", "tc-linln_litc-trad+kf+noblinkall", "direct-lda-grid_litc-trad+kf+noblinkall"),
                    ("H10 pub2-type1+kf", "tc-linln_pub2-type1+kf", "direct-lda-grid_pub2-type1+kf"),
                    ("H10 litc-fore+kf", "tc-linln_litc-fore+kf", "direct-lda-grid_litc-fore+kf")):
        for task in ("three", "binarypair"):
            if exists_grid(task, a) and exists_grid(task, b):
                r = ni_test(integrated(task, a), integrated(task, b))
                f10.append({"hypothesis": h, "family": "F10", "task": f"{task} (integrated)", "contrast": f"{a} - {b}",
                            **r})
            else:
                MISSING.append(f"{h} {task} not run")
    holm_family(f10)
    for r in f10:
        r["decision"] = ni_decision(r)
    rows_h += f10

    # ---------------- H8 label jitter (from ceiling.py output)
    jf = C.RES / "ceiling_summary.json"
    if jf.exists():
        js = json.loads(jf.read_text())
        rows_h.append({"hypothesis": "H8", "family": "descriptive", "task": "three (canonical)",
                       "contrast": "enrichment of M1 errors within 2 sigma of the cut-points",
                       "estimate": js.get("E_A"), "ci_lo": js.get("E_A_lo"), "ci_hi": js.get("E_A_hi"),
                       "decision": js.get("decision")})

    # ---------------- ablations A1-A5 (tier 3, descriptive; PREREG 4.6)
    abl = []
    for name, a, b in (("A1 isotonic PIT recalibration vs M1", TC_MODELS["A1"], M1),
                       ("A4 DistHybrid on litc-trad+kf vs M9 (litc-trad)", TC_MODELS["A4"], TC_MODELS["M9"])):
        for task in ("three", "binarypair"):
            if exists_grid(task, a) and exists_grid(task, b):
                r = two_sided(integrated(task, a), integrated(task, b))
                abl.append({"ablation": name, "task": f"{task} (integrated)", "contrast": f"{a} - {b}", **r})
    for task in ("three", "binary"):
        a, b = TC_MODELS["A4"], "direct-featgru_litc-trad+kf"
        if exists_pred(task, a) and exists_pred(task, b):
            r = ni_test(metric(canon_pp(task, a)), metric(canon_pp(task, b)))
            abl.append({"ablation": "A4 TC vs direct feat_gru on litc-trad+kf (H3-style NI, descriptive)",
                        "task": f"{task} (canonical)", "contrast": f"{a} - {b}", **r})
    for a in ("A2", "A3", "A5"):
        MISSING.append(f"ablation {a} not run (tier 3; see DEVIATIONS)")
    C.write_csv(C.RES / "ablations.csv", abl)

    # ---------------- distributional + regression + coherence
    distributional()
    coherence()

    fields = ["hypothesis", "family", "task", "contrast", "model_class", "n", "estimate", "ci_lo", "ci_hi", "p_raw",
              "p_holm", "k_improved", "decision", "mean_A", "mean_B", "delta_star", "expected", "class_A", "class_B"]
    C.write_csv(C.RES / "hypotheses.csv", rows_h, fields)
    (C.RES / "missing_models.txt").write_text("\n".join(MISSING) + "\n", encoding="utf-8")
    (C.RES / "best_class.json").write_text(json.dumps(best_class, indent=1), encoding="utf-8")
    print("stats done;", len(rows_h), "hypothesis rows;", len(MISSING), "missing")


def kendall_w(R):
    """R: (m raters, n items) ranks with ties averaged -> Kendall's W with tie correction."""
    from scipy.stats import rankdata  # noqa: F401
    m, n = R.shape
    Rs = R.sum(0)
    Ssum = np.sum((Rs - Rs.mean()) ** 2)
    T = 0.0
    for row in R:
        _, cnt = np.unique(row, return_counts=True)
        T += np.sum(cnt ** 3 - cnt)
    den = m ** 2 * (n ** 3 - n) - m * T
    return float(12 * Ssum / den) if den > 0 else np.nan


def _w_boot(BA, canon_j, b=10000, seed=0):
    """BA: (persons, cells, items) per-person BA. Returns W, CI and bootstrap W values."""
    from scipy.stats import rankdata
    npers = BA.shape[0]
    W0 = kendall_w(rankdata(-BA.mean(0), axis=1))
    rng = np.random.default_rng(seed)
    flat = BA.reshape(npers, -1)
    ws = np.empty(b)
    for i0 in range(0, b, 500):
        idx = rng.integers(0, npers, (min(500, b - i0), npers))
        cnt = np.stack([np.bincount(r, minlength=npers) for r in idx]) / npers
        means = (cnt @ flat).reshape(len(idx), BA.shape[1], BA.shape[2])
        R = rankdata(-means, axis=2)
        for k in range(len(idx)):
            ws[i0 + k] = kendall_w(R[k])
    return W0, float(np.quantile(ws, 0.025)), float(np.quantile(ws, 0.975))


def rank_stability(rows_h):
    out = []
    # H7a: 9 presets x 3x3 sub-grid
    f = C.GRID / "rank_presets.npz"
    sets = []
    if f.exists():
        with np.load(f) as z:
            pred, cells, items = z["pred"], [tuple(c) for c in z["cells"]], list(z["clfs"])
        sets.append(("H7a", "F7a", pred, cells, items))
    else:
        MISSING.append("H7a ranking models not run")
    types = ["pub2orig-type1", "pub2orig-type2", "pub2orig-type3", "pub2orig-all"]
    if all(exists_grid("three", f"direct-lda-grid_{t}") for t in types):
        preds = [C.load_grid("three", f"direct-lda-grid_{t}")[0] for t in types]
        sets.append(("H7b", "F7b", np.stack(preds, 2), list(C.CELLS), types))
    else:
        MISSING.append("H7b feature-type grids not run")
    for h, fam, pred, cells, items in sets:
        BA = np.zeros((len(P21), len(cells), len(items)))
        for j, c in enumerate(cells):
            for k in range(len(items)):
                pp = C.cell_person_metrics(Y, G, pred[:, j, k], "three", c)
                BA[:, j, k] = [pp[p]["bacc"] for p in P21]
        cj = cells.index(C.CANON)
        W, lo, hi = _w_boot(BA, cj)
        cm = BA[:, cj].mean(0)
        order = np.argsort(-cm, kind="stable")
        t1, t2 = order[0], order[1]
        fr = []
        for j, c in enumerate(cells):
            out.append({"analysis": h, "cell": str(tuple(float(x) for x in c)), **{f"ba_{it}": float(BA[:, j, k].mean()) for k, it in enumerate(items)},
                        "top": items[int(np.argmax(BA[:, j].mean(0)))]})
            if j == cj:
                continue
            r = two_sided({p: BA[i, j, t1] for i, p in enumerate(P21)}, {p: BA[i, j, t2] for i, p in enumerate(P21)})
            fr.append({"hypothesis": f"{h} reversal at {tuple(float(x) for x in c)}", "family": fam, "task": "three",
                       "contrast": f"{items[t1]} - {items[t2]} (canonical top-two)", **r})
        holm_family(fr)
        nrev = 0
        for r in fr:
            rev = r["p_holm"] < 0.05 and r["estimate"] < 0
            nrev += rev
            r["decision"] = "Holm-significant reversal" if rev else (
                "Holm-significant, same order" if r["p_holm"] < 0.05 else "no significant difference")
        rows_h += fr
        dec = ("unstable" if (hi < 0.7 or nrev >= 1) else
               "stable" if (lo > 0.9 and nrev == 0) else "moderately stable")
        rows_h.append({"hypothesis": h, "family": fam, "task": "three", "contrast": f"Kendall's W over {len(cells)} cells, "
                       f"{len(items)} items", "n": len(P21), "estimate": W, "ci_lo": lo, "ci_hi": hi,
                       "decision": f"{dec} ({nrev} Holm-significant reversals)"})
        out.append({"analysis": h, "cell": "Kendall W", "W": W, "W_lo": lo, "W_hi": hi, "n_reversals": nrev,
                    "canonical_top_two": f"{items[t1]} > {items[t2]}", "decision": dec})
    return out


A4_IDS = {"tc-featgru_litc-trad+kf", "direct-featgru_litc-trad+kf"}


def leaderboard_models():
    """(model_id, class) of the leaderboard rows (PREREG 13)."""
    return [("ref-pub2orig-all_lda", "classical"), ("ref-pub2orig-type2_qsvm", "classical"),
            ("ref-pub2-type1+kf_lda", "classical"), ("ref-litc-trad+kf_lda", "classical"), ("ref-time_lda", "control"),
            (M1, "classical"), ("ordlogit_litc-trad+kf", "classical"), ("tc-qhgb_litc-trad+kf", "classical"),
            ("direct-hgb_litc-trad+kf", "classical"), ("tc-mlpln_litc-trad+kf", "hybrid"),
            ("direct-mlp_litc-trad+kf", "hybrid"), ("tc-featgru_litc-trad", "hybrid"),
            ("direct-featgru_litc-trad", "hybrid"), ("tc-sinchybrid_trad6", "hybrid_e2e"),
            ("direct-sinchybrid_trad6", "hybrid_e2e"), ("tc-resnet1dgru_trad6", "dl_e2e"),
            ("direct-resnet1dgru_trad6", "dl_e2e"), ("pub2nested_pub2orig", "classical"),
            ("tc-linln_time", "control"), ("tc-linln_litc-trad+kf+noblinkall", "classical"),
            ("tc-linln_pub2-type1+kf", "classical"), ("tc-linln_litc-fore+kf", "classical"),
            ("tc-featgru_litc-trad+kf", "hybrid"), ("direct-featgru_litc-trad+kf", "hybrid")]


def distributional():
    from scipy.stats import spearmanr
    dist, reg = [], []
    for tag, mid in TC_MODELS.items():
        f = C.CDFD / f"{mid}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            G512 = z["cdf"].astype(np.float64)
            mean, median = z["mean"], z["median"]
            pit = z["pit"] if "pit" in z.files else None
            ls = z["logscore"] if "logscore" in z.files else None
            qs = {lev: (z[f"q{lev}_lo"], z[f"q{lev}_hi"]) for lev in (50, 80, 90)}
        crps = C.crps_from_grid(G512, Y)
        var = np.sum((1 - G512) * 2 * C.GRID512[None, :], 1) / C.NGRID - mean ** 2
        sd = np.sqrt(np.maximum(var, 0))
        pp = {}
        for p in P21:
            m = G == p
            r = {"crps": float(crps[m].mean())}
            if ls is not None:
                r["logscore"] = float(ls[m].mean())
            if pit is not None:
                u = np.sort(pit[m])
                n = len(u)
                r["pit_ks"] = float(max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n)))
            for lev, (lo, hi) in qs.items():
                r[f"cov{lev}"] = float(np.mean((Y[m] >= lo[m]) & (Y[m] <= hi[m])))
                r[f"width{lev}"] = float(np.mean(hi[m] - lo[m]))
            r["spread_skill_rho"] = float(spearmanr(sd[m], np.abs(Y[m] - mean[m])).statistic)
            pp[p] = r
            rg = S.regression_metrics(Y[m], mean[m])
            rg["mae"] = float(np.mean(np.abs(Y[m] - mean[m])))
            reg.append({"id": tag, "model_id": mid, "person": p, **rg})
        row = {"id": tag, "model_id": mid, "n_persons": len(pp)}
        for k in next(iter(pp.values())):
            mm, lo, hi = C.ci([v[k] for v in pp.values()])
            row[k], row[k + "_lo"], row[k + "_hi"] = mm, lo, hi
        if pit is not None:
            row["pit_hist"] = json.dumps(np.histogram(pit, bins=10, range=(0, 1))[0].tolist())
        dist.append(row)
    C.write_csv(C.RES / "distributional.csv", dist)
    # regression summary (mean over persons)
    summ = []
    for mid in dict.fromkeys(r["model_id"] for r in reg):
        rr = [r for r in reg if r["model_id"] == mid]
        row = {"model_id": mid, "n_persons": len(rr)}
        for k in ("cor", "rmse", "ccc", "mae"):
            m, lo, hi = C.ci([r[k] for r in rr])
            row[k], row[k + "_lo"], row[k + "_hi"] = m, lo, hi
        summ.append(row)
    C.write_csv(C.RES / "regression.csv", summ)


def coherence():
    rows = []
    ti = {c: j for j, c in enumerate(C.CELLS)}
    adj = []
    for (a, b), j in ti.items():
        for (a2, b2) in ((round(a + 0.05, 2), b), (a, round(b + 0.05, 2))):
            if (a2, b2) in ti:
                adj.append((j, ti[(a2, b2)]))
    for tag, mid in {**TC_MODELS, "M2": M2}.items():
        if not exists_grid("three", mid):
            continue
        pred, _ = C.load_grid("three", mid)
        viol = np.mean([np.mean(pred[:, j2] > pred[:, j1]) for j1, j2 in adj])
        rows.append({"metric": "(a) grid monotonicity violation rate (84 adjacent pairs)", "model": mid, "value": viol,
                     "n_pairs": len(adj)})
    # (b) exceedance-probability crossing of the 11 direct single-threshold LDAs (and of M1 for reference)
    for mid in (M2, M1):
        if exists_grid("single", mid):
            _, sc = C.load_grid("single", mid)
            ex = sc[:, :, 1].astype(np.float64)
            cross = np.mean(ex[:, 1:] > ex[:, :-1] + 1e-9)
            rows.append({"metric": "(b) exceedance-probability crossing rate (11 thresholds)", "model": mid,
                         "value": float(cross), "n_pairs": 10})
    # (c) canonical cross-task contradiction among alert/drowsy epochs
    keep = B["yi"] != 2
    for name, three_mid, bin_mid in (("M1 (one F)", M1, M1), ("R4 three vs R4 binary", M2, M2)):
        if exists_grid("three", three_mid) and exists_grid("binarypair", bin_mid):
            p3 = C.load_grid("three", three_mid)[0][:, C.CANON_IDX]
            p2 = C.load_grid("binarypair", bin_mid)[0][:, C.CANON_IDX]
            m = keep
            contra = ((p3[m] == 0) & (p2[m] == 1)) | ((p3[m] == 2) & (p2[m] == 0))
            rows.append({"metric": "(c) canonical three-vs-binary contradiction rate (alert/drowsy epochs)",
                         "model": name, "value": float(np.mean(contra)), "n_pairs": int(m.sum())})
    C.write_csv(C.RES / "coherence.csv", rows)


if __name__ == "__main__":
    main()
