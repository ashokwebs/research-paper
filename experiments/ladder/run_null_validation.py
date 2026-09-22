"""
COMPUTATIONAL COUPLING --- Validation of the motif-preserving null ensemble.

A constrained null model is only useful if it does three things at once, and all
three have to be demonstrated rather than assumed:

1. **Hits its target.**  The sampled graphs reproduce the connectome's motif census.
2. **Actually randomises.**  They are not the empirical graph in disguise: edge
   overlap with the original should be near the level of an unconstrained rewiring.
3. **Does not overfit.**  Structure the chain was *not* asked to preserve should not
   be reproduced.  We hold out the bi-fan count (a 4-node motif, excluded from the
   annealing objective) as that check.

Failing (2) would make L3 a copy of L4 and the ladder's top contrast meaningless;
failing (3) would mean the chain is reconstructing the connectome wholesale rather
than isolating its 3-node structure.

A fourth property is reported as evidence rather than claimed as a guarantee.  An
annealed chain is not proven to sample uniformly over the constraint set, so we
measure **pairwise edge overlap between ensemble members**: if the motif-matched
surrogates are as mutually distinct as unconstrained rewirings are, the chain is
exploring the constraint surface rather than collapsing onto one corner of it.  This
is empirical evidence of adequate mixing, not a proof of uniformity, and the paper
states it that way.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.motifs.fast_census import bifan_count, census
from src.nulls.swap_chain import (
    census_scale,
    degree_preserving_rewire,
    motif_preserving_null,
)
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")


def _job(args):
    key, n_rep, sweeps = args
    A = load_target(key)
    real = census(A, include_bifan=True)
    scale = census_scale(A, n_samples=10, seed=0)
    E_real = set(A.iter_edges())

    ov3, ov2, err3, bif3, bif2, conv = [], [], [], [], [], []
    edges3, edges2 = [], []
    for i in range(n_rep):
        B, rep = motif_preserving_null(A, scale_counts=scale, n_sweeps=sweeps, seed=500 + i)
        R = degree_preserving_rewire(A, n_swaps_factor=20.0, seed=900 + i)
        EB, ER = set(B.iter_edges()), set(R.iter_edges())
        edges3.append(EB)
        edges2.append(ER)
        ov3.append(len(E_real & EB) / len(E_real))
        ov2.append(len(E_real & ER) / len(E_real))
        err3.append(max(rep.relative_error().values()))
        conv.append(float(rep.converged))
        bif3.append(bifan_count(B))
        bif2.append(bifan_count(R))

    def _pairwise(sets):
        vals = []
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                vals.append(len(sets[a] & sets[b]) / len(E_real))
        return float(np.mean(vals)) if vals else float("nan")

    return key, dict(
        n_nodes=A.n, n_edges=A.m, n_replicates=n_rep,
        overlap_L3=float(np.mean(ov3)), overlap_L3_sd=float(np.std(ov3, ddof=1)),
        overlap_L2=float(np.mean(ov2)), overlap_L2_sd=float(np.std(ov2, ddof=1)),
        max_rel_err_L3=float(np.mean(err3)),
        converged_fraction=float(np.mean(conv)),
        bifan_real=int(real["bifan"]),
        bifan_L3=float(np.mean(bif3)),
        bifan_L2=float(np.mean(bif2)),
        pairwise_overlap_L3=_pairwise(edges3),
        pairwise_overlap_L2=_pairwise(edges2),
        census_real={k: int(v) for k, v in real.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]

    import multiprocessing as mp

    out = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for key, d in pool.imap_unordered(
                _job, [(k, args.replicates, args.sweeps) for k in keys]):
            out[key] = d
            print(f"[{key}] overlap-with-real L3={d['overlap_L3']:.3f} "
                  f"L2={d['overlap_L2']:.3f} | pairwise L3={d['pairwise_overlap_L3']:.3f} "
                  f"L2={d['pairwise_overlap_L2']:.3f} | max rel err="
                  f"{d['max_rel_err_L3']:.4f} | converged={d['converged_fraction']:.0%} "
                  f"| bifan real={d['bifan_real']:,} L3={d['bifan_L3']:,.0f} "
                  f"L2={d['bifan_L2']:,.0f}", flush=True)

    with open(os.path.join(RESULTS, "null_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] -> {RESULTS}/null_validation.json")


if __name__ == "__main__":
    main()
