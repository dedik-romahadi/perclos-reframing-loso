"""Resumable driver that re-runs the whole T1 topic (every step skips finished folds / runs).

    python -I code/run_all.py [--skip-gpu] [--skip-tier3]

Order (PREREG 15.3): unit tests -> tier 1 (M2, M1, references, identity check, time controls, R2 QSVM,
permutation controls) -> tier 2 CPU (M3/M4, robustness inputs, M5/M6, M7/M8, M15, H7) -> GPU queue (M9/M10,
then gpu_queue.py: M11, M13, M12, M14 with the adaptive stride rule) -> ceiling (H8) -> statistics ->
leaderboard -> figures. Long jobs can also be run in parallel with code/cpu_queue_*.sh (as done on
28 Oct 2026); this driver runs everything sequentially in one process chain.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def run(*args):
    if not (HERE / args[0]).exists():   # repository version: report builders are not distributed
        print(f">> {args[0]} not in this repository (builds reports, not results); skipped", flush=True)
        return 0
    cmd = [sys.executable, "-I", str(HERE / args[0]), *args[1:]]
    print(">>", " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"!! {args} exited with {r.returncode}", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-tier3", action="store_true")
    a = ap.parse_args()
    if run("tests/test_t1.py", "--fast") != 0:
        sys.exit("unit tests failed")
    # tier 1
    run("run_grid_direct.py", "litc-trad+kf")
    run("run_tc.py", "litc-trad+kf")
    run("run_refs.py", "identity")
    run("run_refs.py", "refs_t1")
    run("run_grid_direct.py", "time")
    run("run_tc.py", "time")
    run("run_refs.py", "qsvm")
    run("run_controls.py")
    # tier 2 CPU
    run("run_calib.py")
    for fs in ("litc-trad+kf+noblinkall", "pub2-type1+kf"):
        run("run_grid_direct.py", fs)
        run("run_tc.py", fs)
    run("run_hgb.py", "qhgb")
    run("run_hgb.py", "hgb")
    for j in ("mlp_tc", "mlp_three", "mlp_binary"):
        run("run_neural.py", j)
    run("run_pub2_nested.py", "three")
    for t in ("type1", "type2", "type3", "all"):
        run("run_grid_direct.py", f"pub2orig-{t}", "--tasks", "three")
    run("run_rank.py")
    if not a.skip_gpu:
        for j in ("featgru_tc", "featgru_three", "featgru_binary"):
            run("run_neural.py", j)
        run("gpu_queue.py", "--gpu-start", "2026-10-28T19:27:00")
    if not a.skip_tier3:
        run("run_grid_direct.py", "litc-fore+kf")
        run("run_tc.py", "litc-fore+kf")
        run("run_pub2_nested.py", "binary")     # M15 binary (descriptive)
        run("run_a1.py")                        # A1 isotonic PIT recalibration of M1 (robustness R4)
        if not a.skip_gpu:
            for j in ("featgrukf_tc", "featgrukf_three", "featgrukf_binary"):
                run("run_neural.py", j)
    run("run_neural.py", "finalize")
    run("ceiling.py")
    run("stats_t1.py")
    run("make_leaderboard.py")
    run("results_md.py")
    run("figures_t1.py")


if __name__ == "__main__":
    main()
