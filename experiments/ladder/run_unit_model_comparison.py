"""
COMPUTATIONAL COUPLING --- Does connectome topology help, or does that depend on whether
the units have a threshold?

The disagreement this experiment is built to settle
---------------------------------------------------
Suarez et al. report that empirical connectomes have greater linear memory capacity than
degree-preserving rewirings, measured in an echo state network of ``tanh`` units.  Our
ladder finds the opposite, measured in a spiking leaky integrate-and-fire network.  Both
results have been checked; neither is going away.

The two models differ in one respect that interacts directly with degree heterogeneity.
A ``tanh`` unit responds to any input however small and is never silent.  A LIF unit
below threshold emits nothing and contributes no variance to the readout.  A connectome's
degree sequence is heavy-tailed, so in a spiking network its low-degree neurons sit below
threshold while hubs dominate --- and we measure exactly that: the fraction of readout
neurons that never fire rises monotonically along the structural ladder at every gain.

That gives a prediction which is sharp, cheap and decisive.  If the threshold is what
reverses the comparison, then running **the same graphs, the same nulls, the same input,
the same delays and the same readout** through the two unit models should reverse the
sign of the contrast, and the reversal should track the silent fraction.  If the contrast
has the same sign in both, the unit model is not the explanation and the disagreement
lies somewhere else.

Design
------
A 2 x 2: {empirical connectome, degree-preserving null} x {rate units, spiking units},
swept across the gain axis so that neither model is privileged by the operating point.
Run on the regional connectomes distributed with ``conn2res`` --- the same networks the
published claim is made on --- so the replication does not inherit our subgraph
extraction, our binarisation, or our choice of species.

Usage
-----
    python experiments/ladder/run_unit_model_comparison.py [--workers N]
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

from src.eval.dynamics import memory_capacity as memory_capacity_spiking
from src.eval.rate_reservoir import RateConfig, memory_capacity_rate, rescale

RESULTS = os.path.join(ROOT, "results", "ladder")
CONN2RES = os.path.join(ROOT, "data", "raw", "conn2res_data", "data")

# Dense through the critical region, where the published comparison is made.
RHO_GRID = (0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8)


def load_conn2res(name: str) -> np.ndarray:
    """One regional connectome as distributed with conn2res, as a dense weight matrix."""
    import pandas as pd

    if name == "human":
        return np.load(os.path.join(CONN2RES, "human", "consensus_0.npy"))
    path = os.path.join(CONN2RES, name, "conn.csv")
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def conn2res_preprocess(W: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1], then divide by the spectral radius.

    Reproduces ``Conn.scale_and_normalize`` from the published toolbox, and is applied
    before randomisation and before the gain is set so that both arms of every comparison
    are the object the published comparison is made on.

    On a sparse connectome the matrix minimum is the zero of a non-edge, so the min-max
    step reduces to a plain rescaling and the order of operations does not change the
    weight distribution -- only the spectral normalisation is doing work. The shift
    matters only for a matrix with no zero entry. The ordering is kept because it is the
    toolbox's, not because it is load-bearing here.
    """
    lo, hi = W.min(), W.max()
    Ws = (W - lo) / (hi - lo) if hi > lo else W.copy()
    ew = np.linalg.eigvals(Ws)
    r = float(np.abs(ew).max()) if ew.size else 0.0
    return Ws if r <= 1e-12 else Ws / r


def published_null(W: np.ndarray, seed: int, swaps: int = 10) -> np.ndarray:
    """The published null: BCT ``randmio_*_connected``, degree- and weight-preserving.

    Our first version of this reassigned weights to the rewired topology by rank of
    endpoint degree, which preserves the weight *distribution* but not the association
    between a weight and its edge, and does not maintain connectedness. That is a
    different ensemble from the one the result being tested was measured against, and a
    replication that uses its own null is not a replication. This calls the same Brain
    Connectivity Toolbox routine the toolbox calls, with the same default of ten swaps
    per edge, dispatching on symmetry exactly as it does.
    """
    from bct import reference

    np.random.seed(seed % (2 ** 32 - 1))   # BCT draws from the legacy global stream
    if np.allclose(W, W.T):
        out, _ = reference.randmio_und_connected(W, swaps)
    else:
        out, _ = reference.randmio_dir_connected(W, swaps)
    return np.asarray(out, dtype=float)


def rank_reassigned_null(W: np.ndarray, seed: int, n_swaps_factor: float = 20.0
                         ) -> np.ndarray:
    """A second, equally defensible null: rewire, then re-assign the weight values.

    This preserves the binary degree sequence and the *distribution* of weights, but not
    the association between a weight and the edge carrying it: the largest weight goes to
    the rewired edge whose endpoints have the largest degree product. Descriptions in the
    literature -- "preserving density and degree sequence" -- do not usually distinguish
    this from carrying weights along with their edges, and we did not either until the
    two gave opposite answers. It is kept as a first-class condition rather than deleted,
    because the difference between the two is the paper's point.
    """
    rng = np.random.default_rng(seed)
    A = (W != 0)
    edges = np.argwhere(A)
    m = len(edges)
    target = int(n_swaps_factor * m)

    adj = A.copy()
    e = [tuple(x) for x in edges]
    done = 0
    for _ in range(target * 4):
        if done >= target:
            break
        i1, i2 = rng.integers(m), rng.integers(m)
        (u, v), (x, y) = e[i1], e[i2]
        if len({u, v, x, y}) != 4 or adj[u, y] or adj[x, v]:
            continue
        adj[u, v] = adj[x, y] = False
        adj[u, y] = adj[x, v] = True
        e[i1], e[i2] = (u, y), (x, v)
        done += 1

    w_sorted = np.sort(W[A])[::-1]
    new_edges = np.argwhere(adj)
    deg_out, deg_in = adj.sum(axis=1), adj.sum(axis=0)
    score = deg_out[new_edges[:, 0]].astype(float) * deg_in[new_edges[:, 1]]
    order = np.argsort(-score)
    out = np.zeros_like(W)
    k = min(len(w_sorted), len(new_edges))
    out[new_edges[order[:k], 0], new_edges[order[:k], 1]] = w_sorted[:k]
    return out


NULLS = {
    "bct_randmio": published_null,          # what the published toolbox calls
    "rank_reassigned": rank_reassigned_null,  # same description, different ensemble
}


def _job(args) -> List[Dict]:
    name, kind, instance, seed = args
    # Preprocess exactly as the toolbox does, *then* randomise, so both arms are the
    # same object the published comparison is made on.
    W0 = conn2res_preprocess(load_conn2res(name))
    W = W0 if kind == "empirical" else NULLS[kind](W0, seed=seed)
    n = W.shape[0]

    rng = np.random.default_rng(3000 + instance)
    W_in = rng.normal(0, 1, n) * (rng.random(n) < 0.3)

    rows = []
    for rho in RHO_GRID:
        Wr = rescale(W, rho)
        t0 = time.time()

        mc_r, diag_r = memory_capacity_rate(
            Wr, W_in, RateConfig(spectral_radius=rho), seed=instance)
        rows.append(dict(network=name, kind=kind, instance=instance,
                         spectral_radius=rho, unit_model="rate",
                         memory_capacity=mc_r, seconds=time.time() - t0,
                         **{k: diag_r.get(k, float("nan"))
                            for k in ("participation_ratio", "dead_frac",
                                      "saturated_frac")}))

        t1 = time.time()
        mc_s = memory_capacity_spiking(Wr, W_in.reshape(-1, 1), seed=instance)
        # The spiking path's diagnostics are recomputed here on the same states rather
        # than imported, so both rows carry the same three columns.
        from src.eval.dynamics import _run

        rng2 = np.random.default_rng(seed=instance)
        u = rng2.uniform(-1.0, 1.0, size=2000 + 30).astype(np.float32)
        X = _run(Wr.astype(np.float32), W_in.reshape(-1, 1).astype(np.float32),
                 u[30:], 0.9, 1.0)[200:]
        sd = X.std(axis=0)
        Xc = X - X.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(Xc, compute_uv=False)
        lam = sv.astype(np.float64) ** 2
        pr = float((lam.sum() ** 2) / np.sum(lam ** 2)) if lam.sum() > 0 else 0.0
        rows.append(dict(network=name, kind=kind, instance=instance,
                         spectral_radius=rho, unit_model="spiking",
                         memory_capacity=mc_s, seconds=time.time() - t1,
                         participation_ratio=pr,
                         dead_frac=float((sd < 1e-4).mean()),
                         saturated_frac=float((X > 0.99).mean())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--networks", type=str,
                    default="human,macaque_modha,mouse,rat,drosophila")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    names = [n for n in args.networks.split(",") if n]

    # The condition is called "rewired" and not "null" on purpose: pandas reads the
    # bare string "null" as a missing value, which silently drops the entire control arm
    # on read-back while leaving a file that looks complete.
    jobs = []
    for name in names:
        for i in range(args.instances):
            jobs.append((name, "empirical", i, 7000 + i))
            for null_name in NULLS:
                jobs.append((name, null_name, i, 7000 + i))
    print(f"[run] {len(jobs)} network draws x {len(RHO_GRID)} gains x 2 unit models "
          f"= {len(jobs) * len(RHO_GRID) * 2} evaluations", flush=True)

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
    out = os.path.join(RESULTS, "unit_model_comparison.csv")
    df.to_csv(out, index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {out}\n")

    per = df.groupby(["unit_model", "network", "kind", "spectral_radius"])[
        "memory_capacity"].mean().unstack("kind")
    # empirical minus each null, so positive always means "the connectome leads".
    adv = (-per.sub(per["empirical"], axis=0).drop(columns="empirical")
           ).stack().unstack("spectral_radius")
    print("=== memory capacity: empirical MINUS rewired (positive = connectome leads) ===")
    print(adv.round(3).to_string())
    print("\n=== sign ===")
    print(np.sign(adv).astype(int).to_string())
    print("\n=== silent fraction by unit model ===")
    print(df.groupby(["unit_model", "kind"])["dead_frac"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
