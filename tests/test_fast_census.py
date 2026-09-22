"""Exactness tests for the sign-attributed motif census and its incremental updates."""

import random

import networkx as nx
import numpy as np
import pytest

from src.motifs.fast_census import (
    MOTIF_KEYS,
    add_edge,
    bifan_count,
    census,
    from_networkx,
    remove_edge,
)


def _random_signed_digraph(n, p, seed):
    rng = random.Random(seed)
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    for u in G.nodes():
        G.nodes[u]["sign"] = rng.choice([1, 1, 1, -1])
    return G


def _brute_force(A):
    """Triple enumeration; the reference the fast census must reproduce."""
    succ, sign = A.succ, A.sign
    ffl = cffl = cyc = recip = recip_i = 0
    for u in range(A.n):
        for v in succ[u]:
            if u < v and u in succ[v]:
                recip += 1
                if sign[u] == -1 and sign[v] == -1:
                    recip_i += 1
            for w in range(A.n):
                if w in (u, v):
                    continue
                if w in succ[v] and w in succ[u]:
                    ffl += 1
                    if sign[v] >= 0:
                        cffl += 1
                if w in succ[v] and u in succ[w]:
                    cyc += 1
    return dict(recip_pair=recip, recip_inhib_pair=recip_i, ffl_total=ffl,
                cffl=cffl, iffl=ffl - cffl, cycle3=cyc // 3)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_census_matches_enumeration(seed):
    G = _random_signed_digraph(24, 0.18, seed)
    A = from_networkx(G)
    got = census(A, include_bifan=False)
    ref = _brute_force(A)
    for k, v in ref.items():
        assert got[k] == v, k


def test_coherence_is_set_by_intermediate_valence():
    """cFFL/iFFL split must follow the sign of the middle node, not edge products."""
    G = nx.DiGraph()
    for i, s in enumerate([1, -1, 1]):
        G.add_node(i, sign=s)
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    c = census(from_networkx(G), include_bifan=False)
    assert c["ffl_total"] == 1
    assert c["iffl"] == 1 and c["cffl"] == 0

    G.nodes[1]["sign"] = 1
    c = census(from_networkx(G), include_bifan=False)
    assert c["cffl"] == 1 and c["iffl"] == 0


def test_incremental_updates_are_exact():
    rng = random.Random(7)
    G = _random_signed_digraph(20, 0.15, 7)
    A = from_networkx(G)
    counts = dict(census(A, include_bifan=False))
    for _ in range(80):
        u, v = rng.randrange(A.n), rng.randrange(A.n)
        if u == v:
            continue
        if v in A.succ[u]:
            remove_edge(A, u, v, counts)
        else:
            add_edge(A, u, v, counts)
        ref = census(A, include_bifan=False)
        for k in MOTIF_KEYS:
            if k == "bifan":
                continue
            assert counts[k] == ref[k], k


def test_bifan_exact_on_known_graph():
    G = nx.DiGraph()
    for i in range(4):
        G.add_node(i, sign=1)
    # Sources 0,1 both project to sinks 2,3 -> exactly one bi-fan.
    G.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 3)])
    assert bifan_count(from_networkx(G)) == 1


def test_edges_derive_from_adjacency_after_mutation():
    """Regression: the census must not read a stale cached edge list."""
    G = nx.DiGraph()
    for i in range(3):
        G.add_node(i, sign=1)
    G.add_edges_from([(0, 1), (1, 2)])
    A = from_networkx(G)
    counts = dict(census(A, include_bifan=False))
    add_edge(A, 0, 2, counts)
    assert census(A, include_bifan=False)["ffl_total"] == 1
    assert counts["ffl_total"] == 1
