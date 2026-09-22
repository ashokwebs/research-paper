"""
COMPUTATIONAL COUPLING --- Dynamical probes across the structural control ladder.

Runs the task-independent measures of :mod:`src.eval.dynamics` (memory capacity,
kernel rank, generalisation rank) on the same ladder used for the task benchmark.
Its purpose is to guard the negative direction of the main result: if motif structure
had a dynamical consequence that the task readout happens to absorb, it should appear
here.  A structural property that moves neither the task metric nor any of these
probes is not doing hidden work.
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

from src.eval.dynamics import probe
from src.eval.reservoir import ReservoirConfig, rescale_spectral_radius, signed_matrix
from src.motifs.fast_census import census
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import census_scale
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")


def _job(args) -> Dict:
    key, rung, instance, scale, sweeps, rho = args
    A_real = load_target(key)
    s = build_rung(A_real, rung, instance, scale_counts=scale, n_sweeps=sweeps)
    W, _ = rescale_spectral_radius(signed_matrix(s.A), rho)
    rng = np.random.default_rng(3000 + instance)
    W_in = rng.normal(0, 1, (s.A.n, 1)) * (rng.random((s.A.n, 1)) < 0.3)
    r = probe(W, W_in, leak=ReservoirConfig().leak, seed=instance)
    return dict(dataset=key, rung=rung, instance=instance, **r.as_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--spectral-radius", type=float, default=1.30)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]

    scales = {}
    for key in keys:
        A = load_target(key)
        scales[key] = census_scale(A, n_samples=10, seed=0)

    jobs = [(k, r, i, scales[k], args.sweeps, args.spectral_radius)
            for k in keys for r in LADDER_RUNGS for i in range(args.instances)]
    print(f"[run] {len(jobs)} dynamics probes", flush=True)

    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(row)
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} {el:.0f}s ~{el/n*(len(jobs)-n):.0f}s left",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "dynamics_ladder.csv"), index=False)
    print(f"\n[done] -> {RESULTS}/dynamics_ladder.csv")
    for metric in ("memory_capacity", "kernel_rank", "capability"):
        print(f"\n--- {metric} ---")
        print(df.groupby(["dataset", "rung"])[metric].mean().unstack(0).to_string())


if __name__ == "__main__":
    main()
