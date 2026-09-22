"""
COMPUTATIONAL COUPLING --- Is the L3 result an artefact of the sampler?

The motif-preserving ensemble is the one rung of the ladder that does not exist in the
literature, and it is produced by an annealed chain rather than by a construction with
a proof behind it.  Two objections follow immediately, and neither is answered by the
validation in ``run_null_validation.py``, which establishes only that the chain hits its
target, randomises, and does not reproduce held-out structure.

1. **Chain length.**  If the chain has not mixed, the surrogates carry residual memory
   of wherever they started, and an L2--L3 null result could simply mean the chain did
   not move far enough for a difference to appear.  We therefore re-run the L3 rung at
   several sweep counts spanning a factor of eight.  If accuracy is flat in chain
   length, incomplete mixing is not what is producing the null result.

2. **Starting point.**  The chain starts at the empirical connectome and walks under a
   census constraint, which guarantees convergence but makes randomisation something to
   be demonstrated.  The opposite construction --- start from a fully rewired graph and
   anneal *towards* the census --- guarantees randomisation and makes convergence the
   thing to demonstrate.  Where both succeed they should give the same answer, and if
   they do not, the ensemble is a property of the algorithm rather than of the
   constraint.  We run both on every network and report where the random start fails to
   converge, which is itself the reason the empirical start is used in the main text.

Neither check can establish uniform sampling; no cheap check can.  What they can do is
rule out the two specific ways the sampler could manufacture the paper's null result,
which is the claim we make for them.

Usage
-----
    python experiments/ladder/run_null_robustness.py [--workers N]
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

from src.eval.reservoir import ReservoirConfig, evaluate_network
from src.motifs.fast_census import census
from src.nulls.swap_chain import census_scale, motif_preserving_null
from src.tasks.shd import load_shd_cached
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")
_CACHE: Dict[str, object] = {}

# Spanning a factor of eight around the 120 sweeps used in the main text.
SWEEP_GRID = (30.0, 60.0, 120.0, 240.0)


def _get_data():
    if "train" not in _CACHE:
        _CACHE["train"] = load_shd_cached("train")
        _CACHE["test"] = load_shd_cached("test")
    return _CACHE["train"], _CACHE["test"]


def _job(args) -> Dict:
    key, init, sweeps, instance, scale, rho = args
    t0 = time.time()
    A_real = load_target(key)
    # Same seed convention as the main ladder's L3 rung, so that an instance index here
    # and there addresses the same surrogate wherever the settings agree.
    seed = 10_000 * 4 + instance
    B, rep = motif_preserving_null(
        A_real, scale_counts=scale, n_sweeps=sweeps, seed=seed, init=init)
    build_s = time.time() - t0

    train, test = _get_data()
    res = evaluate_network(B, train, test, ReservoirConfig(spectral_radius=rho),
                           node_seed=1000 + instance)
    return dict(
        dataset=key, init=init, sweeps=sweeps, instance=instance,
        accuracy=res.accuracy, mean_rate=res.mean_rate,
        synops_per_sample=res.synops_per_sample,
        anneal_converged=float(rep.converged),
        anneal_max_rel_err=max(rep.relative_error().values()),
        anneal_accept_rate=rep.accepted / max(rep.proposed, 1),
        build_seconds=build_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--spectral-radius", type=float, default=1.30)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]

    scales: Dict[str, Dict[str, float]] = {}
    for key in keys:
        A = load_target(key)
        scales[key] = census_scale(A, n_samples=10, seed=0)

    jobs = []
    for key in keys:
        # Arm 1: chain length, at the starting point used in the main text.
        for sw in SWEEP_GRID:
            for i in range(args.instances):
                jobs.append((key, "empirical", sw, i, scales[key], args.spectral_radius))
        # Arm 2: the opposite starting point, at the main text's chain length.
        for i in range(args.instances):
            jobs.append((key, "random", 120.0, i, scales[key], args.spectral_radius))

    print(f"[run] {len(jobs)} sampler-robustness evaluations", flush=True)

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
    out = os.path.join(RESULTS, "null_robustness.csv")
    df.to_csv(out, index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {out}")

    emp = df[df.init == "empirical"]
    print("\n=== Arm 1: accuracy vs chain length (empirical start) ===")
    print(emp.pivot_table(index="dataset", columns="sweeps",
                          values="accuracy").round(4).to_string())
    print("\n=== Arm 2: empirical vs random start at 120 sweeps ===")
    at120 = df[df.sweeps == 120.0]
    print(at120.pivot_table(index="dataset", columns="init",
                            values=["accuracy", "anneal_converged"]).round(4).to_string())


if __name__ == "__main__":
    main()
