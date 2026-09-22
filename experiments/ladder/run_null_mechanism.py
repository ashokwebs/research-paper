r"""
COMPUTATIONAL COUPLING --- Why the two "degree-preserving" nulls disagree.

Section \ref{sec:res-nullchoice} establishes that two rewiring constructions answering to
the same published description give the connectome's memory-capacity advantage opposite
significant verdicts.  That is a fact about the measurement, and on its own it is only a
warning.  This experiment asks what the difference *is*, because a mechanism turns the
warning into something checkable in advance.

The hypothesis is that rank-reassignment does not merely randomise: it *installs* a
structural property the connectome does not have.  Sorting the weight multiset onto the
rewired edges by endpoint-degree product manufactures an association between an edge's
weight and its endpoints' degrees --- large weights land on hub-to-hub edges --- whereas
carrying each weight along with its edge leaves whatever association the data had.  A
null that adds a property is not a control for the absence of that property, and a
hub-weighted reservoir is a poor one, so the connectome's advantage over it is inflated.

The hypothesis makes a prediction sharp enough to fail: the two ensembles must coincide
exactly on an unweighted network, because there is then no weight multiset to permute.
One network in the set (macaque) is binary.  If the two nulls disagree there too, this
account is wrong.

Measured per network over independent draws:
  * ``w_deg_r``  -- Pearson correlation between an edge's weight and log endpoint-degree
                    product.  The quantity the hypothesis says is manufactured.
  * ``gini``     -- inequality of the weight multiset.  Identical across arms by
                    construction; reported so that a network with no weight variation is
                    visible as such rather than inferred.
  * ``rho``      -- spectral radius before gain normalisation.
  * ``norm_ratio`` -- $\|A\|_2 / \rho$, the non-normality that makes the spectral radius
                    a poor summary of a matrix's transient gain.

Usage
-----
    python experiments/ladder/run_null_mechanism.py [--instances N]
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from run_unit_model_comparison import (  # noqa: E402
    conn2res_preprocess, load_conn2res, published_null, rank_reassigned_null,
)

RESULTS = os.path.join(ROOT, "results", "ladder")
NETWORKS = ("human", "macaque_modha", "mouse", "rat", "drosophila")


def descriptors(W: np.ndarray) -> Dict[str, float]:
    """The four structural quantities the hypothesis is stated in."""
    A = W != 0
    e = np.argwhere(A)
    w = W[A]
    if w.size == 0:
        return dict(w_deg_r=float("nan"), gini=float("nan"),
                    rho=0.0, norm_ratio=float("nan"))

    deg_out, deg_in = A.sum(axis=1), A.sum(axis=0)
    dp = deg_out[e[:, 0]].astype(float) * deg_in[e[:, 1]]
    # log1p because the degree product is heavy-tailed; the hypothesis is about
    # monotone association, not about its functional form.
    r = (float(np.corrcoef(np.log1p(dp), w)[0, 1])
         if w.std() > 0 and dp.std() > 0 else 0.0)

    ws = np.sort(w)
    n = ws.size
    gini = (float((2 * np.arange(1, n + 1) - n - 1).dot(ws) / (n * ws.sum()))
            if ws.sum() > 0 else 0.0)

    rho = float(np.abs(np.linalg.eigvals(W)).max())
    nrm = float(np.linalg.norm(W, 2))
    return dict(w_deg_r=r, gini=gini, rho=rho,
                norm_ratio=(nrm / rho if rho > 1e-12 else float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    rows: List[Dict] = []
    for name in NETWORKS:
        W0 = conn2res_preprocess(load_conn2res(name))
        rows.append(dict(network=name, arm="empirical", instance=-1, **descriptors(W0)))
        for i in range(args.instances):
            seed = 7000 + i
            rows.append(dict(network=name, arm="bct_randmio", instance=i,
                             **descriptors(published_null(W0, seed=seed))))
            rows.append(dict(network=name, arm="rank_reassigned", instance=i,
                             **descriptors(rank_reassigned_null(W0, seed=seed))))
        print(f"  {name}: done", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "null_mechanism.csv")
    df.to_csv(out, index=False)

    piv = df.groupby(["network", "arm"])["w_deg_r"].mean().unstack("arm")
    print("\n=== weight--degree association, by arm ===")
    print(piv.round(3).to_string())

    # The falsifiable part: on a network with no weight variation the two nulls are the
    # same ensemble, so every descriptor must agree.
    gini = df[df.arm == "empirical"].set_index("network")["gini"]
    binary = [n for n in NETWORKS if abs(float(gini.get(n, 1.0))) < 1e-9]
    print(f"\n  networks with no weight variation (Gini = 0): {binary or 'none'}")

    summary: Dict = dict(n_instances=int(args.instances),
                         binary_networks=binary)
    for net in NETWORKS:
        g = df[df.network == net]
        emp = float(g[g.arm == "empirical"]["w_deg_r"].iloc[0])
        bct = float(g[g.arm == "bct_randmio"]["w_deg_r"].mean())
        rnk = float(g[g.arm == "rank_reassigned"]["w_deg_r"].mean())
        summary[net] = dict(empirical=emp, bct=bct, rank=rnk,
                            bct_shift=bct - emp, rank_shift=rnk - emp,
                            gini=float(g[g.arm == "empirical"]["gini"].iloc[0]))
        print(f"  {net:14s} empirical {emp:+.3f} | randmio {bct:+.3f} "
              f"(shift {bct-emp:+.3f}) | rank {rnk:+.3f} (shift {rnk-emp:+.3f})")

    # Aggregate the two shifts across weighted networks: the claim is that one ensemble
    # leaves the association alone and the other moves it, so the two shifts are the
    # numbers to compare, not the raw correlations.
    wt = [n for n in NETWORKS if n not in binary]
    summary["n_weighted"] = len(wt)
    summary["median_bct_shift"] = float(np.median([summary[n]["bct_shift"] for n in wt]))
    summary["median_rank_shift"] = float(np.median([summary[n]["rank_shift"] for n in wt]))
    summary["max_bct_shift"] = float(np.max([abs(summary[n]["bct_shift"]) for n in wt]))
    print(f"\n  across the {len(wt)} weighted networks: median |shift| "
          f"randmio {summary['median_bct_shift']:+.3f}, "
          f"rank-reassignment {summary['median_rank_shift']:+.3f}")

    with open(os.path.join(RESULTS, "null_mechanism.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
