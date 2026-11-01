"""Analyses of the late ablations and post-registration extensions (DEVIATIONS D13-D16; plan fixed in D16).

    python -I stats_ext.py

Secondary / exploratory only. Families: FX1 = E1 (wide alpha), A2, A3, A5 vs M2 (NI, integrated BA, both tasks) and
vs their parent (two-sided: integrated BA both tasks, canonical three-class RPS, CRPS); FX2 = raw-EOG 3-seed
versions (-ens3): H3.5-H3.8 recomputed (canonical NI), H4 recomputed (integrated, two-sided, both tasks), and the TC
-ens3 models vs M2 (NI). Holm only inside FX1 / FX2; nothing enters F1-F10. Models that are not finished are
skipped and listed. Outputs: results/ext/extensions.csv, results/ext/raw_seed_ba.csv, results/ext/EXTENSIONS.md.
Uses the helpers of stats_t1.py (imported, its main() is not run).
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import t1_common as C  # noqa: E402
import stats_t1 as ST  # noqa: E402

OUT = C.RES / "ext"
OUT.mkdir(exist_ok=True)
M1, M2 = ST.M1, ST.M2
NEW_TC = {"E1": ("tc-linln-wa_litc-trad+kf", M1), "A2": ("tc-mlpbeta_litc-trad+kf", "tc-mlpln_litc-trad+kf"),
          "A3": ("tc-mlplnaux_litc-trad+kf", "tc-mlpln_litc-trad+kf"),
          "A5": ("tc-qhgb-iso_litc-trad+kf", "tc-qhgb_litc-trad+kf")}
RAW = {"M11": "tc-sinchybrid_trad6", "M12": "direct-sinchybrid_trad6", "M13": "tc-resnet1dgru_trad6",
       "M14": "direct-resnet1dgru_trad6"}
SKIPPED: list[str] = []


def have_grid(mid):
    return ST.exists_grid("three", mid) and ST.exists_grid("binarypair", mid)


def rps_pp(mid):
    with np.load(C.PREDS / f"three_{mid}.npz") as z:
        P, y, g = z["score"].astype(float), z["y"].astype(int), z["groups"]
    cr = ST.calib_rows(mid, "three", P, y, g, ST.P21)
    return {p: v["rps"] for p, v in cr.items()}


def crps_pp(mid):
    with np.load(C.CDFD / f"{mid}.npz") as z:
        G = z["cdf"].astype(np.float64)
    c = C.crps_from_grid(G, ST.Y)
    return {p: float(c[ST.G == p].mean()) for p in ST.P21}


def ni_row(fam, label, task, a, b, dA, dB):
    r = ST.ni_test(dA, dB)
    return {"family": fam, "label": label, "task": task, "contrast": f"{a} - {b}", "test": "NI (delta 0.02)", **r}


def ts_row(fam, label, task, a, b, dA, dB, metric="BA"):
    r = ST.two_sided(dA, dB)
    return {"family": fam, "label": label, "task": task, "contrast": f"{a} - {b}", "test": f"two-sided ({metric})", **r}


def fx1():
    rows = []
    for lab, (mid, parent) in NEW_TC.items():
        if not have_grid(mid):
            SKIPPED.append(f"{lab} {mid} not finished")
            continue
        for task in ("three", "binarypair"):
            rows.append(ni_row("FX1", f"{lab} vs M2", task, mid, M2, ST.integrated(task, mid), ST.integrated(task, M2)))
            rows.append(ts_row("FX1", f"{lab} vs parent", task, mid, parent, ST.integrated(task, mid),
                               ST.integrated(task, parent)))
        rows.append(ts_row("FX1", f"{lab} vs parent", "three (canonical RPS)", mid, parent, rps_pp(mid), rps_pp(parent),
                           "RPS, lower is better"))
        rows.append(ts_row("FX1", f"{lab} vs parent", "CRPS", mid, parent, crps_pp(mid), crps_pp(parent),
                           "CRPS, lower is better"))
    return rows


def raw_seed_table():
    out = []
    for lab, mid in RAW.items():
        for task in ("three", "binary"):
            vals = {}
            for s, m in ((0, mid), (1, f"{mid}-seed1"), (2, f"{mid}-seed2"), ("ens3", f"{mid}-ens3")):
                if ST.exists_pred(task, m):
                    vals[s] = float(np.mean(list(ST.metric(ST.canon_pp(task, m)).values())))
            if not vals:
                continue
            single = [vals[s] for s in (0, 1, 2) if s in vals]
            out.append({"model": lab, "model_id": mid, "task": task, "n_seeds": len(single),
                        **{f"ba_seed{s}": vals.get(s) for s in (0, 1, 2)}, "ba_ens3": vals.get("ens3"),
                        "seed_sd": float(np.std(single, ddof=1)) if len(single) > 1 else None,
                        "seed_range": float(max(single) - min(single)) if len(single) > 1 else None})
        tc = mid.startswith("tc-")
        if tc:
            for task in ("three", "binarypair"):
                vals = {}
                for s, m in ((0, mid), (1, f"{mid}-seed1"), (2, f"{mid}-seed2"), ("ens3", f"{mid}-ens3")):
                    if ST.exists_grid(task, m):
                        vals[s] = float(np.mean(list(ST.integrated(task, m).values())))
                single = [vals[s] for s in (0, 1, 2) if s in vals]
                if single:
                    out.append({"model": lab, "model_id": mid, "task": f"{task} (integrated)", "n_seeds": len(single),
                                **{f"ba_seed{s}": vals.get(s) for s in (0, 1, 2)}, "ba_ens3": vals.get("ens3"),
                                "seed_sd": float(np.std(single, ddof=1)) if len(single) > 1 else None,
                                "seed_range": float(max(single) - min(single)) if len(single) > 1 else None})
    return out


def fx2():
    rows = []
    e = {k: f"{v}-ens3" for k, v in RAW.items()}
    pairs3 = [("H3.5 (ens3)", "three", e["M11"], e["M12"]), ("H3.6 (ens3)", "binary", e["M11"], e["M12"]),
              ("H3.7 (ens3)", "three", e["M13"], e["M14"]), ("H3.8 (ens3)", "binary", e["M13"], e["M14"])]
    for lab, task, a, b in pairs3:
        if not (ST.exists_pred(task, a) and ST.exists_pred(task, b)):
            SKIPPED.append(f"{lab} not finished")
            continue
        rows.append(ni_row("FX2", lab, f"{task} (canonical)", a, b, ST.metric(ST.canon_pp(task, a)),
                           ST.metric(ST.canon_pp(task, b))))
    reps = {"classical": M1, "hybrid": "tc-featgru_litc-trad", "hybrid_e2e": e["M11"], "dl_e2e": e["M13"]}
    pairs4 = [("H4a", "hybrid", "classical"), ("H4b", "classical", "dl_e2e"), ("H4c", "hybrid_e2e", "dl_e2e"),
              ("H4 (hybrid vs hybrid_e2e)", "hybrid", "hybrid_e2e"),
              ("H4 (classical vs hybrid_e2e)", "classical", "hybrid_e2e"), ("H4 (hybrid vs dl_e2e)", "hybrid", "dl_e2e")]
    for task in ("three", "binarypair"):
        for lab, ca, cb in pairs4:
            if not {ca, cb} & {"hybrid_e2e", "dl_e2e"}:     # H4a has no 3-seed version: unchanged, not repeated
                continue
            a, b = reps[ca], reps[cb]
            if not (ST.exists_grid(task, a) and ST.exists_grid(task, b)):
                SKIPPED.append(f"{lab} {task} (ens3) not finished")
                continue
            rows.append(ts_row("FX2", f"{lab} (ens3)", task, a, b, ST.integrated(task, a), ST.integrated(task, b)))
    for k in ("M11", "M13"):
        if have_grid(e[k]):
            for task in ("three", "binarypair"):
                rows.append(ni_row("FX2", f"{k}-ens3 vs M2", task, e[k], M2, ST.integrated(task, e[k]),
                                   ST.integrated(task, M2)))
    return rows


def decide(r):
    if r["test"].startswith("NI"):
        return ST.ni_decision(r)
    if r["p_holm"] < 0.05:
        return "Holm-significant difference"
    return "no Holm-significant difference"


def fmt(v, pp=True):
    return "" if v is None else (f"{100 * v:+.2f}" if pp else f"{v:.4f}")


def main():
    rows1, rows2 = fx1(), fx2()
    for fam in (rows1, rows2):
        ST.holm_family(fam)
        for r in fam:
            r["decision"] = decide(r)
    rows = rows1 + rows2
    if rows:
        C.write_csv(OUT / "extensions.csv", rows)
    seeds = raw_seed_table()
    if seeds:
        C.write_csv(OUT / "raw_seed_ba.csv", seeds)
    alpha = []
    wa = C.RES / "fold_params_tc-linln-wa_litc-trad+kf.csv"
    if wa.exists():
        import csv
        with open(wa, encoding="utf-8") as f:
            alpha = [float(r["alpha"]) for r in csv.DictReader(f)]
    L = ["# Extensions and late ablations (secondary; DEVIATIONS D13-D16)", "",
         "Generated by `code/stats_ext.py`. Holm only inside FX1 / FX2; none of these rows changes a pre-registered "
         "decision. Estimates in pp (BA) or score units (RPS, CRPS).", ""]
    if alpha:
        vals, cnt = np.unique(alpha, return_counts=True)
        L += ["E1 chosen alpha per fold: " + ", ".join(f"{v:g} ({c}/21)" for v, c in zip(vals, cnt)), ""]
    L += ["| family | label | task | test | n | estimate | 95 % CI | p raw | p Holm | k improved | decision |",
          "|---|---|---|---|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        pp = "BA" in r["test"] or r["test"].startswith("NI")
        L.append(f"| {r['family']} | {r['label']} | {r['task']} | {r['test']} | {r['n']} | {fmt(r['estimate'], pp)} | "
                 f"[{fmt(r['ci_lo'], pp)}; {fmt(r['ci_hi'], pp)}] | {r['p_raw']:.3f} | {r['p_holm']:.3f} | "
                 f"{r['k_improved']} | {r['decision']} |")
    if seeds:
        L += ["", "## Raw-EOG networks: seed variability (person-mean BA, %)", "",
              "| model | task | seed 0 | seed 1 | seed 2 | 3-seed version | SD over seeds | range |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for s in seeds:
            f = (lambda v: "" if v is None else f"{100 * v:.1f}")
            L.append(f"| {s['model']} | {s['task']} | {f(s['ba_seed0'])} | {f(s['ba_seed1'])} | {f(s['ba_seed2'])} | "
                     f"{f(s['ba_ens3'])} | {f(s['seed_sd'])} | {f(s['seed_range'])} |")
    if SKIPPED:
        L += ["", "Not (yet) available: " + "; ".join(SKIPPED)]
    (OUT / "EXTENSIONS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
