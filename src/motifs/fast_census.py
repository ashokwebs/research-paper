"""
COMPUTATIONAL COUPLING --- Exact sign-attributed motif census.

This module replaces enumeration-based motif counting with (a) closed-form sparse
linear algebra for whole-graph censuses and (b) exact O(d) incremental updates for
single-edge modifications, which is what makes the annealed motif-preserving null
model of ``src.nulls.swap_chain`` tractable.

Sign convention (Dale's principle)
----------------------------------
Every synapse inherits the valence of its presynaptic neuron, so an edge sign is a
node property, not an independent degree of freedom.  For a feedforward loop
(u -> v, v -> w, u -> w) the product of the three edge signs is

    sign(u) * sign(v) * sign(u) = sign(u)^2 * sign(v) = sign(v),

i.e. **the coherence of an FFL is decided entirely by the valence of its
intermediate node**.  Coherent FFLs are those routed through an excitatory
intermediate; incoherent FFLs are those routed through an inhibitory intermediate.
This identity makes the sign-attributed census exactly as cheap as the topological
one and removes the need to store per-edge signs.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Motif keys produced by :func:`census`.  Keep this ordering stable: the annealed
# null model and the significance-profile code both index feature vectors by it.
MOTIF_KEYS: Tuple[str, ...] = (
    "recip_pair",
    "recip_inhib_pair",
    "ffl_total",
    "cffl",
    "iffl",
    "cycle3",
    "bifan",
)


@dataclass
class AdjacencySets:
    """Adjacency-set representation of a signed digraph, tuned for swap chains.

    Attributes
    ----------
    n : int
        Number of nodes.  Nodes are relabelled to ``0..n-1``.
    succ, pred : list of set
        Successor / predecessor sets.
    sign : ndarray of int8
        Node valence in ``{+1, -1}``; ``0`` marks unknown valence, which is treated
        as excitatory for coherence purposes but recorded separately.

    Notes
    -----
    ``succ`` / ``pred`` are the single source of truth for connectivity.  Callers
    that need stable edge indices (the swap chain) maintain their own index array
    alongside this object; nothing here caches an edge list.
    """

    n: int
    succ: List[set]
    pred: List[set]
    sign: np.ndarray

    @property
    def m(self) -> int:
        return sum(len(s) for s in self.succ)

    def iter_edges(self):
        for u in range(self.n):
            for v in self.succ[u]:
                yield u, v

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return list(self.iter_edges())

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.succ[u]

    def copy(self) -> "AdjacencySets":
        return AdjacencySets(
            n=self.n,
            succ=[set(s) for s in self.succ],
            pred=[set(s) for s in self.pred],
            sign=self.sign.copy(),
        )

    def to_networkx(self):
        import networkx as nx

        G = nx.DiGraph()
        for i in range(self.n):
            G.add_node(i, sign=int(self.sign[i]))
        for u, v in self.iter_edges():
            G.add_edge(u, v, sign=int(self.sign[u]))
        return G

    def to_dense(self, signed: bool = False) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for u, v in self.iter_edges():
            A[u, v] = float(self.sign[u]) if signed else 1.0
        return A


def from_networkx(G) -> AdjacencySets:
    """Build an :class:`AdjacencySets` view of a NetworkX DiGraph."""
    nodes = list(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    succ: List[set] = [set() for _ in range(n)]
    pred: List[set] = [set() for _ in range(n)]
    sign = np.ones(n, dtype=np.int8)
    for u in nodes:
        s = G.nodes[u].get("sign", 1)
        sign[index[u]] = 1 if s is None else int(s)
    for u, v in G.edges():
        iu, iv = index[u], index[v]
        if iu == iv:
            continue  # drop self-loops; parallel edges collapse into the set
        succ[iu].add(iv)
        pred[iv].add(iu)
    return AdjacencySets(n=n, succ=succ, pred=pred, sign=sign)


# --------------------------------------------------------------------------- #
# Whole-graph census
# --------------------------------------------------------------------------- #
def census(A: AdjacencySets, include_bifan: bool = True) -> Dict[str, int]:
    """Exact sign-attributed motif census.

    Complexity is ``O(sum_(u,v) in E  min(d_out(u), d_out(v)))`` for the loop
    motifs, i.e. linear in the number of wedges, and is exact (no sampling).
    """
    succ, pred, sign = A.succ, A.pred, A.sign

    recip = 0
    recip_inh = 0
    ffl = 0
    cffl = 0
    cyc3 = 0

    for u, v in A.iter_edges():
        # 2-node reciprocal motifs (counted once per unordered pair)
        if u < v and u in succ[v]:
            recip += 1
            if sign[u] == -1 and sign[v] == -1:
                recip_inh += 1

        # Feedforward loops with (u, v) as the u->v leg: w in N+(u) & N+(v)
        common = succ[u] & succ[v]
        common.discard(u)
        common.discard(v)
        k = len(common)
        if k:
            ffl += k
            # Coherence is fixed by the intermediate node's valence (see module docstring).
            if sign[v] >= 0:
                cffl += k

        # Directed 3-cycles with (u, v) as one leg; each cycle is found 3 times.
        back = succ[v] & pred[u]
        back.discard(u)
        back.discard(v)
        cyc3 += len(back)

    out: Dict[str, int] = {
        "recip_pair": recip,
        "recip_inhib_pair": recip_inh,
        "ffl_total": ffl,
        "cffl": cffl,
        "iffl": ffl - cffl,
        "cycle3": cyc3 // 3,
        "bifan": bifan_count(A) if include_bifan else 0,
    }
    return out


def bifan_count(A: AdjacencySets) -> int:
    """Exact bi-fan count: number of (u1,u2; v1,v2) with both sources hitting both sinks.

    Computed as ``sum over unordered source pairs of C(|common successors|, 2)``,
    evaluated by accumulating co-occurrence counts through shared sinks.  This is
    exact and avoids the O(N^2) pair scan used previously; cost is
    ``sum_w C(d_in(w), 2)``.
    """
    from collections import defaultdict

    pair_common: Dict[Tuple[int, int], int] = defaultdict(int)
    for w in range(A.n):
        srcs = sorted(A.pred[w])
        d = len(srcs)
        if d < 2:
            continue
        for i in range(d):
            si = srcs[i]
            for j in range(i + 1, d):
                pair_common[(si, srcs[j])] += 1
    total = 0
    for c in pair_common.values():
        if c >= 2:
            total += c * (c - 1) // 2
    return total


def census_vector(counts: Dict[str, int], keys: Sequence[str] = MOTIF_KEYS) -> np.ndarray:
    return np.array([float(counts[k]) for k in keys], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Exact incremental updates (single-edge add / remove)
# --------------------------------------------------------------------------- #
def _edge_delta(A: AdjacencySets, u: int, v: int, present: bool) -> Dict[str, int]:
    """Exact motif-count delta contributed by edge ``(u, v)``.

    ``present=True`` evaluates the contribution of an edge currently in the graph
    (so removing it applies the negated delta); ``present=False`` evaluates the
    contribution an absent edge *would* add.  In both cases the graph passed in
    must be the state in which ``(u, v)`` is absent from all *other* motif legs,
    which is guaranteed because the caller applies one edge modification at a time.

    Bi-fan is intentionally excluded here: it is a 4-node motif whose incremental
    update is not O(d), and the annealed null targets 3-node structure.
    """
    succ, pred, sign = A.succ, A.pred, A.sign

    # For an absent edge we must not let (u,v) itself appear in the neighbour sets.
    su, sv = succ[u], succ[v]
    pu, pv = pred[u], pred[v]

    # (u,v) as the u->v leg: w with u->w and v->w
    c1 = sv & su
    c1.discard(u)
    c1.discard(v)
    # (u,v) as the v->w leg: t with t->u and t->v
    c2 = pu & pv
    c2.discard(u)
    c2.discard(v)
    # (u,v) as the u->w leg: t with u->t and t->v
    c3 = su & pv
    c3.discard(u)
    c3.discard(v)

    n1, n2, n3 = len(c1), len(c2), len(c3)

    # Coherence: for leg (u,v) the intermediate is v; for leg (v,w) with (u,v)
    # playing v->w the intermediate is u... here (u,v) is the second leg of a
    # triple (t, u, v), whose intermediate is u.  For (u,v) as the closing u->w
    # leg of triple (u, t, v) the intermediate is t.
    coh = 0
    if sign[v] >= 0:
        coh += n1
    if sign[u] >= 0:
        coh += n2
    for t in c3:
        if sign[t] >= 0:
            coh += 1

    # Directed 3-cycles through (u,v): w with v->w->u
    cyc = len((sv & pu) - {u, v})

    d = {
        "ffl_total": n1 + n2 + n3,
        "cffl": coh,
        "iffl": (n1 + n2 + n3) - coh,
        "cycle3": cyc,
        "recip_pair": 1 if u in sv else 0,
        "recip_inhib_pair": 1 if (u in sv and sign[u] == -1 and sign[v] == -1) else 0,
        "bifan": 0,
    }
    return d


def add_edge(A: AdjacencySets, u: int, v: int, counts: Dict[str, int]) -> None:
    """Insert ``(u, v)`` and update ``counts`` in place (exact, O(d))."""
    d = _edge_delta(A, u, v, present=False)
    A.succ[u].add(v)
    A.pred[v].add(u)
    for k, val in d.items():
        counts[k] += val


def remove_edge(A: AdjacencySets, u: int, v: int, counts: Dict[str, int]) -> None:
    """Delete ``(u, v)`` and update ``counts`` in place (exact, O(d))."""
    A.succ[u].discard(v)
    A.pred[v].discard(u)
    d = _edge_delta(A, u, v, present=False)
    for k, val in d.items():
        counts[k] -= val
