"""
COMPUTATIONAL COUPLING --- Size-matched subcircuit extraction.

Cross-species comparison requires networks of equal size: reservoir capacity grows
with neuron count, so a 300-neuron nematode nervous system and a 2,859-neuron fly
central complex cannot be compared directly without confounding phylogeny with
dimensionality.  This module extracts induced subgraphs of a fixed node count by
breadth-first expansion from a seed neuron, which preserves local wiring structure
far better than uniform node sampling (uniform sampling destroys degree structure
and motif density by construction).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..motifs.fast_census import AdjacencySets, from_networkx


def bfs_subgraph(A: AdjacencySets, n_target: int, seed: int = 0) -> AdjacencySets:
    """Induced subgraph on ``n_target`` nodes grown by BFS from a random seed node.

    Expansion follows both successors and predecessors so that the sample is not
    biased towards downstream-only structure.
    """
    rng = np.random.default_rng(seed)
    if A.n <= n_target:
        return A.copy()

    # Seed from a node with above-median total degree so the sample lands inside
    # the connected core rather than on an isolated periphery.
    deg = np.array([len(A.succ[i]) + len(A.pred[i]) for i in range(A.n)])
    candidates = np.flatnonzero(deg >= max(1, np.median(deg[deg > 0])))
    if candidates.size == 0:
        candidates = np.flatnonzero(deg > 0)
    start = int(rng.choice(candidates))

    selected = [start]
    in_set = {start}
    frontier = [start]
    while len(selected) < n_target and frontier:
        nxt: List[int] = []
        for u in frontier:
            nbrs = list(A.succ[u] | A.pred[u])
            rng.shuffle(nbrs)
            for v in nbrs:
                if v not in in_set:
                    in_set.add(v)
                    selected.append(v)
                    nxt.append(v)
                    if len(selected) >= n_target:
                        break
            if len(selected) >= n_target:
                break
        frontier = nxt
    # Pad with the highest-degree unused nodes if BFS exhausted the component.
    if len(selected) < n_target:
        rest = [i for i in np.argsort(-deg) if i not in in_set]
        selected.extend(int(i) for i in rest[: n_target - len(selected)])

    return induced(A, selected)


def induced(A: AdjacencySets, nodes: List[int]) -> AdjacencySets:
    """Induced subgraph on ``nodes``, relabelled to ``0..len(nodes)-1``."""
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    succ: List[set] = [set() for _ in range(n)]
    pred: List[set] = [set() for _ in range(n)]
    for u in nodes:
        iu = idx[u]
        for v in A.succ[u]:
            if v in idx:
                iv = idx[v]
                if iu != iv:
                    succ[iu].add(iv)
                    pred[iv].add(iu)
    sign = np.array([A.sign[u] for u in nodes], dtype=np.int8)
    return AdjacencySets(n=n, succ=succ, pred=pred, sign=sign)


def drop_isolated(A: AdjacencySets) -> AdjacencySets:
    """Remove degree-zero nodes.

    Isolated nodes are not merely inert: in the MICrONS V1 subgraph used here every
    annotated inhibitory soma has zero degree, so leaving them in makes the network
    appear to contain inhibition that cannot participate in any motif.
    """
    keep = [i for i in range(A.n) if A.succ[i] or A.pred[i]]
    return induced(A, keep)


def summarise(A: AdjacencySets) -> Dict[str, float]:
    deg_out = np.array([len(s) for s in A.succ], dtype=float)
    deg_in = np.array([len(s) for s in A.pred], dtype=float)
    n_inh = int((A.sign == -1).sum())
    inh_edges = sum(len(A.succ[i]) for i in range(A.n) if A.sign[i] == -1)
    return {
        "n_nodes": float(A.n),
        "n_edges": float(A.m),
        "density": float(A.m / (A.n * (A.n - 1))) if A.n > 1 else 0.0,
        "mean_degree": float(deg_out.mean()),
        "frac_inhibitory_neurons": float(n_inh / A.n) if A.n else 0.0,
        "frac_inhibitory_edges": float(inh_edges / A.m) if A.m else 0.0,
        "n_isolated": float(int(((deg_in + deg_out) == 0).sum())),
    }
