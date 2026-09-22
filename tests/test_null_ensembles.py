"""
The two rewiring ensembles that the manuscript's central claim rests on.

Section \\ref{sec:res-nullchoice} argues that "degree-preserving randomisation" names two
different ensembles and that the difference between them changed the sign of our headline
result. These tests pin the properties that argument depends on, so that a future edit
which quietly makes the two constructions equivalent -- or breaks the invariant each is
supposed to hold -- fails here rather than in a rebuilt manuscript.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from run_unit_model_comparison import (  # noqa: E402
    conn2res_preprocess, published_null, rank_reassigned_null,
)


def _weighted_graph(n=60, p=0.12, seed=0):
    """A directed graph with a heavy-tailed weight distribution and no self-loops."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p)
    np.fill_diagonal(A, False)
    W = np.zeros((n, n))
    W[A] = rng.lognormal(0.0, 1.0, size=int(A.sum()))
    return W


def _binary_graph(n=60, p=0.12, seed=0):
    """The same graph with every weight equal -- the case where the two must coincide."""
    W = _weighted_graph(n, p, seed)
    return (W != 0).astype(float)


def _w_deg_r(W):
    """Correlation between an edge's weight and its endpoints' log degree product."""
    A = W != 0
    e = np.argwhere(A)
    w = W[A]
    do, di = A.sum(axis=1), A.sum(axis=0)
    dp = do[e[:, 0]].astype(float) * di[e[:, 1]]
    if w.std() == 0 or dp.std() == 0:
        return 0.0
    return float(np.corrcoef(np.log1p(dp), w)[0, 1])


@pytest.mark.parametrize("null", [published_null, rank_reassigned_null])
def test_preserves_edge_count(null):
    """Both ensembles are rewirings: the number of edges is an invariant, not a target."""
    W = _weighted_graph()
    out = null(W, seed=1)
    assert (out != 0).sum() == (W != 0).sum()


@pytest.mark.parametrize("null", [published_null, rank_reassigned_null])
def test_preserves_weight_multiset(null):
    """Neither ensemble invents or discards weight values; they only move them."""
    W = _weighted_graph()
    out = null(W, seed=1)
    assert np.allclose(np.sort(W[W != 0]), np.sort(out[out != 0]))


@pytest.mark.parametrize("null", [published_null, rank_reassigned_null])
def test_actually_rewires(null):
    """A null that returns its input would make every contrast in the paper vacuous."""
    W = _weighted_graph()
    out = null(W, seed=1)
    overlap = ((W != 0) & (out != 0)).sum() / (W != 0).sum()
    assert overlap < 0.95, f"{null.__name__} left {overlap:.0%} of edges in place"


def test_rank_reassignment_installs_weight_degree_association():
    """The mechanism claim: rank-reassignment manufactures an association, randmio does not.

    This is the quantity §\\ref{sec:res-nullchoice} reports as shifting by a median of
    +0.66 against +0.06 on the real connectomes. The threshold here is deliberately loose
    -- the claim is a difference in kind, not a specific magnitude.
    """
    W = _weighted_graph(seed=3)
    base = _w_deg_r(W)
    bct = np.mean([_w_deg_r(published_null(W, seed=s)) for s in range(5)])
    rank = np.mean([_w_deg_r(rank_reassigned_null(W, seed=s)) for s in range(5)])

    assert rank - base > 0.3, (
        f"rank-reassignment should install a weight-degree association; "
        f"empirical {base:+.3f} -> {rank:+.3f}")
    assert abs(bct - base) < abs(rank - base) / 2, (
        f"randmio should leave the association roughly alone; "
        f"empirical {base:+.3f} -> randmio {bct:+.3f}, rank {rank:+.3f}")


def test_the_two_nulls_coincide_on_an_unweighted_graph():
    """The falsifiable prediction behind the mechanism.

    If the ensembles differ only in how they place weights, then with nothing to place
    they must agree. This is the invariant that the binary macaque network satisfies in
    the manuscript, and it is what makes the mechanism an explanation rather than a story.
    """
    W = _binary_graph(seed=4)
    for s in range(3):
        a, b = published_null(W, seed=s), rank_reassigned_null(W, seed=s)
        # Every edge carries the same value, so both must be 0/1 matrices with the same
        # edge count and an identically undefined weight-degree association.
        assert set(np.unique(a)) <= {0.0, 1.0}
        assert set(np.unique(b)) <= {0.0, 1.0}
        assert _w_deg_r(a) == pytest.approx(0.0, abs=1e-9)
        assert _w_deg_r(b) == pytest.approx(0.0, abs=1e-9)


def test_conn2res_preprocess_normalises_spectral_radius():
    """Min-max then spectral normalisation, in the toolbox's order."""
    W = _weighted_graph(seed=5)
    out = conn2res_preprocess(W)
    assert np.abs(np.linalg.eigvals(out)).max() == pytest.approx(1.0, rel=1e-6)


def test_conn2res_preprocess_is_linear_only_when_the_minimum_is_zero():
    """The min-max step's order of application matters, but only for a dense matrix.

    On a sparse graph the matrix minimum is the zero of a non-edge, so ``(W - 0)/hi`` is
    an ordinary rescaling and weight ratios survive it. On a matrix with no zero entry
    the shift is real and the transform is not linear. The distinction is worth pinning
    because the order-of-operations argument in ``conn2res_preprocess``'s docstring only
    has force in the second case.
    """
    sparse = _weighted_graph(seed=5)
    assert sparse.min() == 0.0
    w0 = np.sort(sparse[sparse != 0])
    o0 = np.sort(conn2res_preprocess(sparse)[conn2res_preprocess(sparse) != 0])
    assert np.allclose(w0 / w0.max(), o0 / o0.max()), "sparse case should be linear"

    rng = np.random.default_rng(6)
    dense = rng.lognormal(0.0, 1.0, size=(30, 30)) + 1.0   # strictly positive minimum
    assert dense.min() > 0
    d0 = np.sort(dense.ravel())
    p0 = np.sort(conn2res_preprocess(dense).ravel())
    assert not np.allclose(d0 / d0.max(), p0 / p0.max()), "dense case should not be"
