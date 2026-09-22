"""
COMPUTATIONAL COUPLING --- Degree-preserving swap chains and the motif-preserving null.

Why a new null model
--------------------
The standard connectomics null is the Maslov-Sneppen double-edge swap, which fixes
the joint in/out degree sequence and randomises everything else.  Motif enrichment
is then reported as a Z-score against that ensemble.  That design can establish
*that* a connectome has excess feedforward loops, but it cannot establish that the
excess is what does the computing, because every functional comparison against a
Maslov-Sneppen null simultaneously perturbs the motif census and every higher-order
statistic that co-varies with it.

A note on Dale's principle
--------------------------
Under the convention that a synapse carries the valence of its presynaptic neuron,
a double-edge swap (u->v, x->y) => (u->y, x->v) leaves u and x presynaptic, so every
neuron's outgoing synapses keep a single shared valence.  **Dale's principle is
therefore preserved automatically by any double-edge swap and imposes no additional
constraint on the ensemble.**  Testing whether excitatory/inhibitory *placement*
carries information consequently requires a separate control that permutes valence
labels across neurons (:func:`permute_signs`), not a "Dale-compliant" rewiring.

The motif-preserving null
-------------------------
:func:`motif_preserving_null` runs the same degree-preserving swap chain under a
Metropolis criterion whose energy is the squared deviation of the running 3-node
motif census from a target census, measured in units of the unconstrained-ensemble
standard deviation.  Setting the target to the empirical census yields graphs that
are maximally randomised *subject to reproducing the connectome's motif spectrum*,
which is the missing rung needed to attribute function to motifs rather than to
degrees.  Exact O(d) incremental census updates (``src.motifs.fast_census``) make
the chain tractable on connectome-scale graphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..motifs.fast_census import (
    MOTIF_KEYS,
    AdjacencySets,
    add_edge,
    census,
    remove_edge,
)

# Motif coordinates targeted by the annealed chain.  Bi-fan is excluded because its
# incremental update is not O(d); it is reported post hoc as a held-out statistic,
# which doubles as a check that the chain does not merely overfit its own objective.
ANNEAL_KEYS: Tuple[str, ...] = (
    "recip_pair",
    "recip_inhib_pair",
    "ffl_total",
    "cffl",
    "iffl",
    "cycle3",
)


@dataclass
class SwapChainState:
    """Adjacency plus a stable edge-index array for O(1) uniform edge sampling."""

    A: AdjacencySets
    edge_list: List[Tuple[int, int]]
    edge_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def from_adjacency(cls, A: AdjacencySets) -> "SwapChainState":
        edges = list(A.iter_edges())
        return cls(A=A, edge_list=edges, edge_pos={e: i for i, e in enumerate(edges)})

    def _replace(self, idx: int, new_edge: Tuple[int, int]) -> None:
        old = self.edge_list[idx]
        del self.edge_pos[old]
        self.edge_list[idx] = new_edge
        self.edge_pos[new_edge] = idx


def _propose(state: SwapChainState, rng: np.random.Generator):
    """Sample a valid double-edge swap, or ``None`` if the draw is rejected."""
    m = len(state.edge_list)
    i1 = int(rng.integers(m))
    i2 = int(rng.integers(m))
    if i1 == i2:
        return None
    u, v = state.edge_list[i1]
    x, y = state.edge_list[i2]
    # Reject swaps that would create self-loops or collapse into parallel edges.
    if u == y or x == v or u == x or v == y:
        return None
    if y in state.A.succ[u] or v in state.A.succ[x]:
        return None
    return i1, i2, u, v, x, y


def _propose_reciprocal(state: SwapChainState, rng: np.random.Generator):
    """Sample a swap of two *reciprocal pairs*, moving both directions together.

    Ordinary double-edge swaps destroy reciprocal pairs far more readily than they
    create them, so a chain constrained to preserve a high reciprocal-pair count
    cannot hold it: in a near-symmetric network such as diffusion tractography, where
    almost every edge is reciprocal, the count drifts down by tens of percent however
    stiff the penalty.  Swapping whole pairs -- (u<->v, x<->y) => (u<->y, x<->v) --
    preserves every node's in- and out-degree *and* the reciprocity of the four edges
    involved, which makes the constrained ensemble reachable.
    """
    m = len(state.edge_list)
    A = state.A
    for _ in range(8):   # a few draws; reciprocal edges may be a small minority
        i1 = int(rng.integers(m))
        u, v = state.edge_list[i1]
        if u not in A.succ[v]:
            continue
        i2 = int(rng.integers(m))
        x, y = state.edge_list[i2]
        if x not in A.succ[y]:
            continue
        if len({u, v, x, y}) != 4:
            continue
        if (y in A.succ[u] or u in A.succ[y]
                or v in A.succ[x] or x in A.succ[v]):
            continue
        return u, v, x, y
    return None


def _apply_reciprocal(state: SwapChainState, u, v, x, y, counts) -> None:
    A = state.A
    for a, b in ((u, v), (v, u), (x, y), (y, x)):
        remove_edge(A, a, b, counts)
    for a, b in ((u, y), (y, u), (x, v), (v, x)):
        add_edge(A, a, b, counts)


def _revert_reciprocal(state: SwapChainState, u, v, x, y, counts) -> None:
    A = state.A
    for a, b in ((u, y), (y, u), (x, v), (v, x)):
        remove_edge(A, a, b, counts)
    for a, b in ((u, v), (v, u), (x, y), (y, x)):
        add_edge(A, a, b, counts)


def _reindex_reciprocal(state: SwapChainState, u, v, x, y) -> None:
    for old, new in (((u, v), (u, y)), ((v, u), (y, u)),
                     ((x, y), (x, v)), ((y, x), (v, x))):
        idx = state.edge_pos.get(old)
        if idx is not None:
            state._replace(idx, new)


def _apply(state: SwapChainState, u, v, x, y, counts) -> None:
    A = state.A
    remove_edge(A, u, v, counts)
    remove_edge(A, x, y, counts)
    add_edge(A, u, y, counts)
    add_edge(A, x, v, counts)


def _revert(state: SwapChainState, u, v, x, y, counts) -> None:
    A = state.A
    remove_edge(A, x, v, counts)
    remove_edge(A, u, y, counts)
    add_edge(A, x, y, counts)
    add_edge(A, u, v, counts)


def degree_preserving_rewire(
    A: AdjacencySets,
    n_swaps_factor: float = 20.0,
    seed: int = 0,
) -> AdjacencySets:
    """Maslov-Sneppen rewiring: exact in/out degree sequence, everything else free.

    This is the conventional connectomics null (ladder rung L1/L2 depending on
    whether valence labels are subsequently permuted).
    """
    rng = np.random.default_rng(seed)
    B = A.copy()
    state = SwapChainState.from_adjacency(B)
    target = int(n_swaps_factor * len(state.edge_list))
    done = 0
    attempts = 0
    max_attempts = target * 20
    while done < target and attempts < max_attempts:
        attempts += 1
        prop = _propose(state, rng)
        if prop is None:
            continue
        i1, i2, u, v, x, y = prop
        B.succ[u].discard(v); B.pred[v].discard(u)
        B.succ[x].discard(y); B.pred[y].discard(x)
        B.succ[u].add(y); B.pred[y].add(u)
        B.succ[x].add(v); B.pred[v].add(x)
        state._replace(i1, (u, y))
        state._replace(i2, (x, v))
        done += 1
    return B


def permute_signs(A: AdjacencySets, seed: int = 0) -> AdjacencySets:
    """Randomly permute valence labels across neurons, preserving the E:I ratio.

    Breaks any association between a neuron's valence and its topological position
    while leaving the wiring diagram and the global E:I ratio untouched.
    """
    rng = np.random.default_rng(seed)
    B = A.copy()
    B.sign = B.sign[rng.permutation(B.n)]
    return B


def erdos_renyi_matched(A: AdjacencySets, seed: int = 0) -> AdjacencySets:
    """Erdos-Renyi digraph matched on node count, edge count and E:I ratio."""
    rng = np.random.default_rng(seed)
    n = A.n
    m = A.m
    succ: List[set] = [set() for _ in range(n)]
    pred: List[set] = [set() for _ in range(n)]
    placed = 0
    while placed < m:
        u = int(rng.integers(n))
        v = int(rng.integers(n))
        if u == v or v in succ[u]:
            continue
        succ[u].add(v)
        pred[v].add(u)
        placed += 1
    sign = A.sign[rng.permutation(n)]
    return AdjacencySets(n=n, succ=succ, pred=pred, sign=sign)


# --------------------------------------------------------------------------- #
# Annealed motif-preserving null
# --------------------------------------------------------------------------- #
@dataclass
class AnnealReport:
    """Diagnostics for one annealed chain; reported so runs are auditable."""

    accepted: int
    proposed: int
    energy_start: float
    energy_final: float
    target: Dict[str, float]
    achieved: Dict[str, float]
    scale: Dict[str, float]
    converged: bool

    def relative_error(self) -> Dict[str, float]:
        out = {}
        for k in self.target:
            t = self.target[k]
            out[k] = abs(self.achieved[k] - t) / t if t else 0.0
        return out


def _energy(counts: Dict[str, int], target: np.ndarray, scale: np.ndarray) -> float:
    c = np.array([counts[k] for k in ANNEAL_KEYS], dtype=np.float64)
    return float(np.sum(((c - target) / scale) ** 2))


def motif_preserving_null(
    A: AdjacencySets,
    target_counts: Optional[Dict[str, int]] = None,
    scale_counts: Optional[Dict[str, float]] = None,
    n_sweeps: float = 60.0,
    seed: int = 0,
    t_start: Optional[float] = None,
    t_end: float = 0.01,
    burn_in_factor: float = 10.0,
    tol: float = 0.02,
    init: str = "empirical",
) -> Tuple[AdjacencySets, AnnealReport]:
    """Sample a degree-preserving graph whose 3-node motif census matches a target.

    Parameters
    ----------
    A : AdjacencySets
        Source graph; supplies the degree sequence, valence labels and (by default)
        the target census.
    target_counts : dict, optional
        Census to match.  Defaults to the census of ``A`` -- i.e. a maximally
        randomised graph that nevertheless reproduces the connectome's motifs.
        Passing the *unconstrained* ensemble mean instead yields a motif-ablated
        control at the same degree sequence.
    scale_counts : dict, optional
        Per-motif normalisation, nominally the unconstrained-ensemble standard
        deviation, so that energy is measured in null standard deviations.  Falls
        back to ``max(1, sqrt(target))``.
    n_sweeps : float
        Annealing length in multiples of the edge count.
    tol : float
        Per-motif relative error below which the chain is declared converged.
    init : {"empirical", "random"}
        Where the chain starts.  ``"empirical"`` starts at the source graph, which
        satisfies the constraint exactly, and performs a constrained random walk that
        randomises the wiring while holding the census; convergence is then guaranteed
        and randomisation is what must be demonstrated (by edge overlap, which the
        validation experiment measures).  ``"random"`` starts from a fully rewired
        graph and anneals towards the target; randomisation is then guaranteed and
        convergence is what must be demonstrated.  The two give equivalent ensembles
        where both converge, but ``"random"`` fails on graphs whose census is hard to
        rebuild by chance -- notably near-symmetric networks such as diffusion
        tractography, where almost every edge is reciprocal.

    Returns
    -------
    (AdjacencySets, AnnealReport)
    """
    rng = np.random.default_rng(seed)

    if target_counts is None:
        target_counts = census(A, include_bifan=False)
    target = np.array([float(target_counts[k]) for k in ANNEAL_KEYS])

    if scale_counts is None:
        scale = np.maximum(1.0, np.sqrt(np.maximum(target, 1.0)))
    else:
        scale = np.array(
            [max(1.0, float(scale_counts.get(k, np.sqrt(max(target_counts[k], 1)))))
             for k in ANNEAL_KEYS]
        )

    if init == "random":
        B = degree_preserving_rewire(A, n_swaps_factor=burn_in_factor, seed=seed + 7919)
        # A random start must first climb to the target, so the chain needs a hot
        # phase to escape local minima.
        if t_start is None:
            t_start = 2.0
    elif init == "empirical":
        B = A.copy()
        # Starting on the constraint surface, a hot phase would only destroy the
        # census -- and some motif counts (reciprocal pairs above all) are far easier
        # to destroy than to rebuild, so the damage is not recoverable.  The chain is
        # therefore run as a low-temperature constrained random walk.
        if t_start is None:
            t_start = 0.05
    else:
        raise ValueError(f"unknown init {init!r}")
    state = SwapChainState.from_adjacency(B)
    counts = dict(census(B, include_bifan=False))

    e0 = _energy(counts, target, scale)
    e_cur = e0

    m = len(state.edge_list)
    n_steps = max(1, int(n_sweeps * m))
    accepted = 0
    proposed = 0

    # Mix in paired moves in proportion to how reciprocal the graph is: a network
    # with few reciprocal edges gains nothing from them, while a near-symmetric one
    # cannot hold its reciprocal-pair count without them.
    recip_frac = 2.0 * float(target_counts.get("recip_pair", 0)) / max(A.m, 1)
    p_recip = float(np.clip(recip_frac, 0.0, 0.8))

    # Geometric temperature schedule.
    log_ratio = np.log(t_end / t_start)

    for step in range(n_steps):
        T = t_start * np.exp(log_ratio * step / n_steps)
        use_recip = p_recip > 0 and rng.random() < p_recip

        if use_recip:
            prop = _propose_reciprocal(state, rng)
            if prop is None:
                continue
            proposed += 1
            u, v, x, y = prop
            _apply_reciprocal(state, u, v, x, y, counts)
            e_new = _energy(counts, target, scale)
            delta = e_new - e_cur
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                _reindex_reciprocal(state, u, v, x, y)
                e_cur = e_new
                accepted += 1
            else:
                _revert_reciprocal(state, u, v, x, y, counts)
            continue

        prop = _propose(state, rng)
        if prop is None:
            continue
        proposed += 1
        i1, i2, u, v, x, y = prop

        _apply(state, u, v, x, y, counts)
        e_new = _energy(counts, target, scale)
        delta = e_new - e_cur

        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            state._replace(i1, (u, y))
            state._replace(i2, (x, v))
            e_cur = e_new
            accepted += 1
        else:
            _revert(state, u, v, x, y, counts)

    achieved = {k: float(counts[k]) for k in ANNEAL_KEYS}
    tgt = {k: float(target_counts[k]) for k in ANNEAL_KEYS}
    rel = {
        k: (abs(achieved[k] - tgt[k]) / tgt[k] if tgt[k] else 0.0) for k in ANNEAL_KEYS
    }
    converged = all(v <= tol for v in rel.values())

    report = AnnealReport(
        accepted=accepted,
        proposed=proposed,
        energy_start=e0,
        energy_final=e_cur,
        target=tgt,
        achieved=achieved,
        scale={k: float(s) for k, s in zip(ANNEAL_KEYS, scale)},
        converged=converged,
    )
    return state.A, report


def ensemble_scale(
    A: AdjacencySets, n_samples: int = 20, seed: int = 0, n_swaps_factor: float = 20.0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Mean and standard deviation of the motif census over the unconstrained null.

    Supplies both the Z-score denominator for enrichment testing and the energy
    normalisation for the annealed chain.
    """
    rows = []
    for i in range(n_samples):
        B = degree_preserving_rewire(A, n_swaps_factor=n_swaps_factor, seed=seed + i)
        rows.append(census(B, include_bifan=False))
    keys = list(rows[0].keys())
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}
    return mean, std


def census_scale(
    A: AdjacencySets, n_samples: int = 10, seed: int = 0, n_swaps_factor: float = 20.0
) -> Dict[str, float]:
    """Per-motif energy normalisation for :func:`motif_preserving_null`.

    The annealing energy is measured in units of the *unconstrained-ensemble standard
    deviation*, so that a unit of energy means the same thing for a motif counted in
    the thousands and one counted in the millions.  This function is the single
    definition of that unit; every experiment draws it from here rather than
    recomputing it, because an earlier version of this code applied an additional
    floor of 2% of the empirical count at each call site, which made the constraint
    roughly five times slacker than the paper described it and left the chain unable to
    hold the census on near-symmetric networks (human tractography, worst-case relative
    error 8.6%).  With the floor removed the same chain converges to 0.5% on that
    network at unchanged edge overlap.

    The floor of 1.0 remains and is load-bearing for a different reason: a motif whose
    count is invariant across the unconstrained ensemble has ``std == 0``, and dividing
    by it would make the energy infinite.
    """
    _, std = ensemble_scale(A, n_samples=n_samples, seed=seed,
                            n_swaps_factor=n_swaps_factor)
    keys = census(A, include_bifan=False).keys()
    return {k: max(std.get(k, 1.0), 1.0) for k in keys}
