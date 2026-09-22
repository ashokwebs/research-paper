"""
COMPUTATIONAL COUPLING --- The ladder at a matched firing rate.

Matching spectral radius equalises the recurrent gain but not the operating point.
In the main ladder, Erdos-Renyi surrogates settle at roughly twice the firing rate of
the connectomes they are matched to, and firing rate is both correlated with accuracy
and the direct determinant of synaptic-operation count. Any accuracy difference at
matched spectral radius therefore confounds topology with activity level.

Here each network is calibrated individually, by bisection on its spectral radius, to
a common target firing rate, and the ladder is re-run at that matched operating point.
Comparing the two versions separates two distinct questions:

* at equal gain, what operating point does each topology settle into? (main ladder)
* at equal operating point, does topology still change what the network computes?
"""

import argparse
import os
import sys
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.eval.reservoir import ReservoirConfig, calibrate_gain_for_rate, evaluate_network
from src.motifs.fast_census import census
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import census_scale
from src.tasks.shd import load_shd_cached
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")
_CACHE: Dict[str, object] = {}


def _get_data():
    if "train" not in _CACHE:
        _CACHE["train"] = load_shd_cached("train")
        _CACHE["test"] = load_shd_cached("test")
    return _CACHE["train"], _CACHE["test"]


def _job(args) -> Dict:
    key, rung, instance, scale, sweeps, target_rate, n_calib = args
    train, test = _get_data()
    A_real = load_target(key)
    s = build_rung(A_real, rung, instance, scale_counts=scale, n_sweeps=sweeps)

    cfg = ReservoirConfig()
    t0 = time.time()
    rho, achieved = calibrate_gain_for_rate(
        s.A, train.X[:n_calib], cfg, target_rate, node_seed=1000 + instance)
    cfg = ReservoirConfig(spectral_radius=rho)
    res = evaluate_network(s.A, train, test, cfg, node_seed=1000 + instance)
    # Did the bisection actually reach the target?  Recorded per surrogate so that the
    # analysis can exclude cells that were never rate-matched rather than average them
    # in.  Tolerance matches the bisection's own.
    calib_ok = abs(achieved - target_rate) <= 0.10 * target_rate
    return dict(
        dataset=key, rung=rung, instance=instance,
        target_rate=target_rate, calibrated_rho=rho, calib_rate=achieved,
        calib_ok=bool(calib_ok),
        accuracy=res.accuracy, val_accuracy=res.val_accuracy,
        synops_per_sample=res.synops_per_sample, mean_rate=res.mean_rate,
        n_neurons=res.n_neurons, n_edges=res.n_edges,
        seconds=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--target-rate", type=float, default=0.07)
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]

    scales = {}
    for key in keys:
        A = load_target(key)
        scales[key] = census_scale(A, n_samples=10, seed=0)

    jobs = [(k, r, i, scales[k], args.sweeps, args.target_rate, args.calib_samples)
            for k in keys for r in LADDER_RUNGS for i in range(args.instances)]
    print(f"[run] {len(jobs)} rate-matched evaluations (target rate "
          f"{args.target_rate})", flush=True)

    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(row)
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} {el:.0f}s ~{el/n*(len(jobs)-n):.0f}s left",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ladder_rate_matched.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {RESULTS}/ladder_rate_matched.csv")
    print(df.groupby(["dataset", "rung"])[["accuracy", "mean_rate", "calibrated_rho",
                                           "synops_per_sample"]].mean().to_string())


if __name__ == "__main__":
    main()
