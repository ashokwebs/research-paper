"""
COMPUTATIONAL COUPLING --- Is the spiking non-detection a finding, or an untuned model?

The headline comparison reports that the connectome advantage is present with graded
units and absent with spiking ones.  There is an obvious and serious objection: the
spiking reservoir was run at one threshold, one leak and one input scale, none of which
were tuned, and 62% of its units never fire.  A network that quiet may simply be
misconfigured, and a non-detection in a misconfigured model says nothing about spiking
units -- it says something about our choice of constants.  That is the same error as
reporting an underpowered replication as a null result, which this project has already
made once.

So the objection gets the experiment rather than a paragraph.  We sweep the three
parameters that set the operating regime of a LIF reservoir -- input gain, firing
threshold and membrane leak -- across a range wide enough to take the network from nearly
silent to nearly saturated, and ask two questions at every setting:

  1. Does the silencing go away?  If no setting produces an active network, the spiking
     harness is broken and nothing measured with it counts.
  2. Where the network *is* active, does the connectome advantage appear?  If it does at
     any well-conditioned setting, the headline claim is wrong and becomes a statement
     about one corner of parameter space.  If it does not appear anywhere the network is
     healthy, the claim stands on much firmer ground than a single configuration.

The test is deliberately generous to the alternative hypothesis: it searches for the
setting that would refute us.

Usage
-----
    python experiments/ladder/run_spiking_tuning_control.py [--workers N]
"""

import argparse
import itertools
import os
import sys
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.eval.dynamics import _run, memory_capacity as mc_spiking
from src.eval.rate_reservoir import rescale
from run_unit_model_comparison import load_conn2res, strength_preserving_null  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")

# Input gain moves the network from silent to driven; threshold and leak set how much
# drive is needed and how long it is remembered. The grid spans two orders of magnitude
# in gain because that is what it takes to cross from silence to saturation.
GAIN_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
THRESH_GRID = (0.25, 0.5, 1.0)
LEAK_GRID = (0.7, 0.9)
RHO = 1.0          # the published operating point, held fixed while the rest is swept


def _job(args) -> List[Dict]:
    name, kind, instance, seed = args
    W0 = load_conn2res(name)
    W = W0 if kind == "empirical" else strength_preserving_null(W0, seed=seed)
    Wr = rescale(W, RHO)
    n = Wr.shape[0]

    rng = np.random.default_rng(3000 + instance)
    W_in0 = (rng.normal(0, 1, n) * (rng.random(n) < 0.3)).reshape(-1, 1)

    rows = []
    for gain, thresh, leak in itertools.product(GAIN_GRID, THRESH_GRID, LEAK_GRID):
        W_in = W_in0 * gain
        t0 = time.time()
        mc = mc_spiking(Wr, W_in, leak=leak, thresh=thresh, seed=instance)

        # Activity diagnostics on the same trajectory, so "was the network alive" is
        # answerable for every cell rather than inferred from the capacity.
        rng2 = np.random.default_rng(seed=instance)
        u = rng2.uniform(-1.0, 1.0, size=2030).astype(np.float32)
        X = _run(Wr.astype(np.float32), W_in.astype(np.float32), u[30:],
                 leak, thresh)[200:]
        sd = X.std(axis=0)
        rows.append(dict(
            network=name, kind=kind, instance=instance,
            input_gain=gain, threshold=thresh, leak=leak,
            memory_capacity=mc,
            dead_frac=float((sd < 1e-4).mean()),
            mean_state=float(X.mean()),
            seconds=time.time() - t0,
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--networks", type=str, default="macaque_modha,mouse,rat,drosophila")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    # Human is excluded: its spiking capacity sits at the floor in every configuration we
    # have seen, so it cannot discriminate between hypotheses here.
    names = [n for n in args.networks.split(",") if n]

    jobs = []
    for name in names:
        for i in range(args.instances):
            jobs.append((name, "empirical", i, 7000 + i))
            jobs.append((name, "rewired", i, 7000 + i))
    n_cells = len(GAIN_GRID) * len(THRESH_GRID) * len(LEAK_GRID)
    print(f"[run] {len(jobs)} draws x {n_cells} configurations "
          f"= {len(jobs) * n_cells} evaluations", flush=True)

    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, batch in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.extend(batch)
            if k % 10 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)}  {el:.0f}s  ~{el/k*(len(jobs)-k):.0f}s left",
                      flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "spiking_tuning_control.csv")
    df.to_csv(out, index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {out}\n")

    print("=== did the network come alive? mean silent fraction by configuration ===")
    print(df.groupby(["input_gain", "threshold"])["dead_frac"].mean()
          .unstack().round(3).to_string())
    print("\n=== mean memory capacity by configuration ===")
    print(df.groupby(["input_gain", "threshold"])["memory_capacity"].mean()
          .unstack().round(2).to_string())


if __name__ == "__main__":
    main()
