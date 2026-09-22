"""
COMPUTATIONAL COUPLING --- Does the connectome advantage depend on where you compare?

The reservoir-computing literature reports that empirical connectomes outperform
degree-preserving rewirings, and reports it **at each network's own optimum** --- at
criticality, or at peak memory capacity.  Our ladder reports the opposite, and evaluates
every network at a **common** operating point, either a matched spectral radius or a
matched firing rate.

Those are different protocols answering different questions, and the difference is not
a detail.  Comparing each network at its own argmax asks *how good can this topology get
when tuned for it*.  Comparing at a common point asks *what does this topology do to a
network held otherwise fixed*.  A topology whose optimum sits at a different gain, or
whose peak is broader, can win the first comparison and lose the second.

This experiment measures both on identical data.  The full ladder is swept across the
gain axis, and the L0-vs-L4 contrast is then read off three ways:

  (a) at a common spectral radius          -- the matched-gain protocol
  (b) at each network's own best radius    -- the literature's protocol
  (c) at each network's peak, per rung     -- the same, resolved per structural rung

If the sign of the contrast differs between (a) and (b), the reported advantage is a
property of the comparison protocol rather than of the topology, and both results in the
literature are correct about different questions.

Usage
-----
    python experiments/ladder/run_gain_sweep_ladder.py [--workers N]
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
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import census_scale
from src.tasks.shd import load_shd_cached
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")

# Spanning the subcritical regime, the critical region where the reservoir literature
# evaluates, and the supercritical regime our main ladder used.
RHO_GRID = (0.80, 0.95, 1.10, 1.30, 1.60, 2.00, 2.50)

_CACHE: Dict[str, object] = {}


def _get_data():
    if "train" not in _CACHE:
        _CACHE["train"] = load_shd_cached("train")
        _CACHE["test"] = load_shd_cached("test")
    return _CACHE["train"], _CACHE["test"]


def _job(args) -> List[Dict]:
    key, rung, instance, scale, sweeps = args
    train, test = _get_data()
    A_real = load_target(key)
    # One surrogate, swept across the whole gain axis. Building the graph once and
    # re-scaling it is not just cheaper: it means the gain curve for a rung is measured
    # on a single realisation, so curvature along rho is not confounded with
    # surrogate-to-surrogate variation.
    s = build_rung(A_real, rung, instance, scale_counts=scale, n_sweeps=sweeps)

    rows = []
    for rho in RHO_GRID:
        t0 = time.time()
        cfg = ReservoirConfig(spectral_radius=rho)
        try:
            res = evaluate_network(s.A, train, test, cfg, node_seed=1000 + instance)
        except Exception as exc:  # noqa: BLE001
            # One gain failing must not lose the other six, nor the 199 other
            # surrogates. The failure is recorded as a row so it is counted rather
            # than silently absent from the sweep.
            rows.append(dict(
                dataset=key, rung=rung, instance=instance, spectral_radius=rho,
                accuracy=float("nan"), val_accuracy=float("nan"),
                mean_rate=float("nan"), synops_per_sample=float("nan"),
                feature_rank_frac=float("nan"), participation_ratio=float("nan"),
                participation_ratio_frac=float("nan"), saturated_frac=float("nan"),
                dead_frac=float("nan"), ok=False, error=repr(exc)[:200],
                seconds=time.time() - t0,
            ))
            continue
        rows.append(dict(
            dataset=key, rung=rung, instance=instance, spectral_radius=rho,
            accuracy=res.accuracy, val_accuracy=res.val_accuracy,
            mean_rate=res.mean_rate, synops_per_sample=res.synops_per_sample,
            feature_rank_frac=res.extra.get("feature_rank_frac", float("nan")),
            # Representation geometry, recorded on the same runs as the accuracy so a
            # mediation question can be asked of them rather than across experiments.
            participation_ratio=res.extra.get("participation_ratio", float("nan")),
            participation_ratio_frac=res.extra.get("participation_ratio_frac", float("nan")),
            saturated_frac=res.extra.get("saturated_frac", float("nan")),
            dead_frac=res.extra.get("dead_frac", float("nan")),
            ok=True, error="",
            seconds=time.time() - t0,
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]

    scales = {}
    for key in keys:
        scales[key] = census_scale(load_target(key), n_samples=10, seed=0)

    jobs = [(k, r, i, scales[k], args.sweeps)
            for k in keys for r in LADDER_RUNGS for i in range(args.instances)]
    print(f"[run] {len(jobs)} surrogates x {len(RHO_GRID)} gains "
          f"= {len(jobs) * len(RHO_GRID)} evaluations", flush=True)

    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, batch in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.extend(batch)
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} surrogates  {el:.0f}s  "
                      f"~{el / n * (len(jobs) - n):.0f}s left", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "gain_sweep_ladder.csv")
    df.to_csv(out, index=False)
    print(f"\n[done] {time.time() - t0:.0f}s -> {out}\n")

    per = df.groupby(["dataset", "rung", "spectral_radius"])["accuracy"].mean()
    print("=== accuracy by rung and gain ===")
    print(per.unstack().round(4).to_string())

    print("\n=== L0 - L4, at each gain (positive = unstructured wins) ===")
    g = per.unstack(level="rung")
    gap = (g["L0_erdos_renyi"] - g["L4_connectome"]).unstack(level="spectral_radius")
    print((100 * gap).round(2).to_string())

    print("\n=== L0 - L4 at each rung's OWN best gain (the literature's protocol) ===")
    best = df.groupby(["dataset", "rung"])["accuracy"].max().unstack()
    print((100 * (best["L0_erdos_renyi"] - best["L4_connectome"])).round(2).to_string())
    print("\nbest gain per rung:")
    idx = df.groupby(["dataset", "rung", "spectral_radius"])["accuracy"].mean()
    print(idx.groupby(["dataset", "rung"]).idxmax().apply(lambda t: t[2]).unstack().to_string())


if __name__ == "__main__":
    main()
