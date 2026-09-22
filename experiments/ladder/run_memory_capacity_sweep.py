"""
COMPUTATIONAL COUPLING --- Replicating the connectome memory-capacity advantage, and
locating it on the gain axis.

Suarez et al. (conn2res, Nat Commun 2024) report that empirical connectomes have
significantly greater linear memory capacity than degree-preserving rewirings, and they
report it **at criticality**, at a spectral radius of 1.  That is the single most direct
statement in the literature that connectome topology confers computational function, and
our ladder appears to contradict it: at the gain we used, the unstructured rung leads on
memory capacity in every network we tested.

Both cannot be dismissed, and neither needs to be.  The two results are measured at
different points on the gain axis, and the gain axis is not a nuisance parameter here --
it sets the dynamical regime, and the regime is what a reservoir's memory capacity is a
property of.  This experiment measures the same quantity they measure, on the same kind
of null they use, across a sweep that contains their operating point and ours.

The specific question: **is there a gain at which the sign of the contrast changes, and
does the literature's operating point sit on one side of it and ours on the other?**

If so, the disagreement is not a disagreement about connectomes. It is a disagreement
about where to stand, and every result of this kind in the literature -- including ours
-- inherits a dependence on that choice that has not been reported.

Usage
-----
    python experiments/ladder/run_memory_capacity_sweep.py [--workers N]
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
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import census_scale
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")

# Dense through the critical region, because that is where the contrast is claimed and
# where a sign change, if there is one, has to be located precisely enough to say which
# side each published protocol sits on.
RHO_GRID = (0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30, 1.50, 1.80, 2.20)

# The literature's null preserves density and degree sequence. That is our L2 rung
# (degree-preserving rewiring with valence left on its own neurons); L1 additionally
# permutes valence. Both are reported so the replication is not sensitive to which
# convention the original used.
LIT_NULL = "L2_degree_valence"


def _job(args) -> List[Dict]:
    key, rung, instance, scale, sweeps = args
    A_real = load_target(key)
    s = build_rung(A_real, rung, instance, scale_counts=scale, n_sweeps=sweeps)
    W0 = signed_matrix(s.A)

    # One surrogate swept across the gain axis, so curvature in rho is not confounded
    # with surrogate-to-surrogate variation.
    rng = np.random.default_rng(3000 + instance)
    W_in = rng.normal(0, 1, (s.A.n, 1)) * (rng.random((s.A.n, 1)) < 0.3)

    rows = []
    for rho in RHO_GRID:
        W, _ = rescale_spectral_radius(W0, rho)
        t0 = time.time()
        r = probe(W, W_in, leak=ReservoirConfig().leak, seed=instance)
        rows.append(dict(dataset=key, rung=rung, instance=instance,
                         spectral_radius=rho, seconds=time.time() - t0,
                         **r.as_dict()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]
    scales = {k: census_scale(load_target(k), n_samples=10, seed=0) for k in keys}

    jobs = [(k, r, i, scales[k], args.sweeps)
            for k in keys for r in LADDER_RUNGS for i in range(args.instances)]
    print(f"[run] {len(jobs)} surrogates x {len(RHO_GRID)} gains "
          f"= {len(jobs) * len(RHO_GRID)} probes", flush=True)

    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, batch in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.extend(batch)
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el:.0f}s  ~{el / n * (len(jobs) - n):.0f}s left",
                      flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "memory_capacity_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\n[done] {time.time() - t0:.0f}s -> {out}\n")

    per = df.groupby(["dataset", "rung", "spectral_radius"])["memory_capacity"].mean()
    g = per.unstack(level="rung")

    print("=== memory capacity: empirical connectome minus degree-preserving null ===")
    print("    (positive = the connectome leads, which is the published direction)")
    adv = (g["L4_connectome"] - g[LIT_NULL]).unstack(level="spectral_radius")
    print(adv.round(3).to_string())

    print("\n=== sign of that contrast ===")
    print(np.sign(adv).astype(int).to_string())

    print("\n=== same contrast against the Erdos-Renyi rung ===")
    adv0 = (g["L4_connectome"] - g["L0_erdos_renyi"]).unstack(level="spectral_radius")
    print(adv0.round(3).to_string())


if __name__ == "__main__":
    main()
