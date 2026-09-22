"""Tests for the swap chains and the motif-preserving null model."""

import networkx as nx
import numpy as np
import pytest

from src.motifs.fast_census import census, from_networkx
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import (
    census_scale,
    degree_preserving_rewire,
    ensemble_scale,
    erdos_renyi_matched,
    motif_preserving_null,
    permute_signs,
)


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(3)
    G = nx.gnp_random_graph(120, 0.06, seed=3, directed=True)
    for u in G.nodes():
        G.nodes[u]["sign"] = int(rng.choice([1, 1, 1, -1]))
    return from_networkx(G)


def _degrees(A):
    return (np.array([len(s) for s in A.succ]), np.array([len(s) for s in A.pred]))


def test_rewire_preserves_degrees_and_edge_count(graph):
    B = degree_preserving_rewire(graph, n_swaps_factor=10, seed=0)
    da, ia = _degrees(graph)
    db, ib = _degrees(B)
    assert np.array_equal(da, db) and np.array_equal(ia, ib)
    assert B.m == graph.m


def test_double_edge_swap_preserves_dale_automatically(graph):
    """Edge valence follows the presynaptic node, so swaps cannot violate Dale."""
    B = degree_preserving_rewire(graph, n_swaps_factor=10, seed=1)
    assert np.array_equal(B.sign, graph.sign)
    for u, v in B.iter_edges():
        assert B.sign[u] == graph.sign[u]


def test_rewire_actually_randomises(graph):
    B = degree_preserving_rewire(graph, n_swaps_factor=20, seed=2)
    overlap = len(set(graph.iter_edges()) & set(B.iter_edges())) / graph.m
    assert overlap < 0.5


def test_permute_signs_preserves_ratio_and_topology(graph):
    B = permute_signs(graph, seed=4)
    assert sorted(B.sign.tolist()) == sorted(graph.sign.tolist())
    assert set(B.iter_edges()) == set(graph.iter_edges())


def test_erdos_renyi_matches_size_and_density(graph):
    B = erdos_renyi_matched(graph, seed=5)
    assert B.n == graph.n and B.m == graph.m


def test_motif_preserving_null_hits_target_and_randomises(graph):
    scale = census_scale(graph, n_samples=8, seed=0)
    B, rep = motif_preserving_null(graph, scale_counts=scale, n_sweeps=60, seed=6)

    da, ia = _degrees(graph)
    db, ib = _degrees(B)
    assert np.array_equal(da, db) and np.array_equal(ia, ib)
    assert np.array_equal(B.sign, graph.sign)

    # The default chain starts on the constraint surface, so hitting the target is
    # guaranteed and *randomisation* is the property that has to be demonstrated.
    rel = rep.relative_error()
    assert rel["ffl_total"] < 0.10
    assert rel["cycle3"] < 0.10
    overlap = len(set(graph.iter_edges()) & set(B.iter_edges())) / graph.m
    assert overlap < 0.6


def test_motif_preserving_null_random_start_descends_to_target(graph):
    """The reverse construction: from a fully rewired start the energy must fall.

    This is the property the empirical-start chain cannot show (it begins at energy
    zero), and it is what establishes that the annealing objective is the thing
    driving the census rather than the initial condition.
    """
    scale = census_scale(graph, n_samples=8, seed=0)
    B, rep = motif_preserving_null(
        graph, scale_counts=scale, n_sweeps=60, seed=6, init="random"
    )

    da, ia = _degrees(graph)
    db, ib = _degrees(B)
    assert np.array_equal(da, db) and np.array_equal(ia, ib)
    assert rep.energy_start > 0.0
    assert rep.energy_final < 0.25 * rep.energy_start


def test_ladder_rungs_match_size(graph):
    scale = census_scale(graph, n_samples=5, seed=0)
    for rung in LADDER_RUNGS:
        s = build_rung(graph, rung, 0, scale_counts=scale, n_sweeps=25)
        assert s.A.n == graph.n
        assert s.A.m == graph.m, rung


def test_census_scale_is_the_null_sigma_not_a_fraction_of_the_count(graph):
    """The annealing energy must be measured in null standard deviations.

    An earlier version applied an extra floor of 2% of the empirical count at each
    call site, which is larger than the null sigma for every motif of any size and
    therefore made the constraint several times slacker than the paper described.  The
    chain then could not hold the census on near-symmetric graphs.  This test pins the
    definition so the floor cannot creep back in.
    """
    scale = census_scale(graph, n_samples=8, seed=0)
    _, std = ensemble_scale(graph, n_samples=8, seed=0)
    real = census(graph, include_bifan=False)
    for k, v in scale.items():
        assert v == pytest.approx(max(std[k], 1.0)), k
        # ...and specifically, it is not the old 2%-of-count floor.
        if 0.02 * real[k] > max(std[k], 1.0):
            assert v < 0.02 * real[k], k


def test_motif_preserving_null_holds_a_near_symmetric_census():
    """A graph whose edges are almost all reciprocal is the case that broke the chain.

    Ordinary double-edge swaps destroy reciprocal pairs far more readily than they
    create them, so without the paired move the reciprocal-pair count collapses and the
    chain cannot converge.  Human tractography is the real instance of this; here it is
    reproduced in miniature so the regression is cheap to catch.
    """
    import networkx as nx

    g = nx.gnm_random_graph(60, 240, seed=3, directed=False)
    d = nx.DiGraph()
    for u, v in g.edges():
        d.add_edge(u, v, sign=1)
        d.add_edge(v, u, sign=1)
    A = from_networkx(d)

    scale = census_scale(A, n_samples=8, seed=0)
    B, rep = motif_preserving_null(A, scale_counts=scale, n_sweeps=60, seed=11)
    assert rep.converged, rep.relative_error()
    # ...and it randomised rather than standing still.
    overlap = len(set(A.iter_edges()) & set(B.iter_edges())) / A.m
    assert overlap < 0.8
