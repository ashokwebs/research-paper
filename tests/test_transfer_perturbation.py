"""Tests for the gap-free transfer-map perturbation theory."""

import numpy as np
import pytest

from src.theory.transfer_perturbation import (
    adversarial_edge_ranking,
    apply_pruning,
    hub_edge_ranking,
    magnitude_edge_ranking,
    markov_parameters,
    operational_alignment,
    random_edge_ranking,
    spectral_radius,
    structural_drift,
    worst_case_constant,
)


@pytest.fixture
def system():
    rng = np.random.default_rng(0)
    n = 50
    A = rng.normal(0, 1, (n, n)) * (rng.random((n, n)) < 0.2)
    A *= 0.9 / spectral_radius(A)
    W_in = rng.normal(0, 1, (n, 6))
    W_out = rng.normal(0, 1, (4, n))
    return A, W_in, W_out


def test_markov_parameters_match_explicit_powers(system):
    A, W_in, W_out = system
    H = markov_parameters(A, W_in, W_out, 5)
    for t in range(5):
        assert np.allclose(H[t], W_out @ np.linalg.matrix_power(A, t) @ W_in)


def test_bound_holds_for_random_perturbations(system):
    """gamma must lie in [0, 1]: the realised change cannot exceed the worst case."""
    A, W_in, W_out = system
    rng = np.random.default_rng(1)
    for scale in (1e-4, 1e-3, 1e-2):
        E = rng.normal(0, scale, A.shape)
        m = operational_alignment(A, E, W_in, W_out, horizon=10)
        assert 0.0 <= m.gamma <= 1.0, m.gamma


def test_bound_requires_no_spectral_gap(system):
    """A matrix with an exactly repeated eigenvalue still yields a finite bound."""
    A, W_in, W_out = system
    n = A.shape[0]
    A_deg = np.eye(n) * 0.5           # every eigenvalue identical: gap is zero
    C = worst_case_constant(float(np.linalg.norm(A_deg, 2)), 10,
                            float(np.linalg.norm(W_in, 2)),
                            float(np.linalg.norm(W_out, 2)))
    assert np.isfinite(C) and C > 0


def test_adversarial_ranking_beats_random(system):
    """The first-order maximiser must realise a larger map change than random."""
    A, W_in, W_out = system
    rng = np.random.default_rng(2)
    adv = adversarial_edge_ranking(A, W_in, W_out, horizon=10)
    rnd = random_edge_ranking(A, rng)
    E_adv = apply_pruning(A, adv, 0.1)
    E_rnd = apply_pruning(A, rnd, 0.1)
    g_adv = operational_alignment(A, E_adv, W_in, W_out, horizon=10)
    g_rnd = operational_alignment(A, E_rnd, W_in, W_out, horizon=10)
    # Compare damage per unit perturbation norm, not raw damage.
    assert g_adv.delta_h_norm / g_adv.e_norm > g_rnd.delta_h_norm / g_rnd.e_norm


def test_structural_drift_bounds():
    W = np.array([[0.0, 1.0], [2.0, 0.0]])
    assert structural_drift(W, np.zeros_like(W)) == 0.0
    assert 0.0 < structural_drift(W, -W) <= 1.0


def test_magnitude_ranking_refuses_an_unweighted_graph():
    """Weakest-first pruning must not silently degenerate to index order.

    On a binary connectome every entry has the same magnitude, so ``argsort`` returns
    row-major order and the resulting schedule deletes neuron 0's edges, then neuron
    1's.  That is a structured neuron-ablation attack, and it was previously reported
    under the name of a weight heuristic.  The function now refuses instead.
    """
    rng = np.random.default_rng(0)
    W = (rng.random((40, 40)) < 0.2).astype(float)   # binary, all magnitudes equal
    np.fill_diagonal(W, 0.0)
    with pytest.raises(ValueError, match="unweighted"):
        magnitude_edge_ranking(W)

    # ...and it still works where magnitudes genuinely differ.
    Wv = W * rng.random((40, 40))
    order = magnitude_edge_ranking(Wv)
    mags = np.abs(Wv.ravel()[order])
    assert np.all(np.diff(mags) >= -1e-12)


def test_hub_ranking_targets_high_degree_sources_and_is_not_index_order():
    rng = np.random.default_rng(1)
    W = (rng.random((60, 60)) < 0.15).astype(float)
    np.fill_diagonal(W, 0.0)
    # Give a late-indexed node a large out-degree; a genuine hub ranking must reach it
    # first, whereas an index ordering would reach it last.
    W[55, :] = 1.0
    W[55, 55] = 0.0
    order = hub_edge_ranking(W)
    first_sources = order[:50] // W.shape[1]
    assert (first_sources == 55).sum() > 25


def test_bound_holds_on_a_strongly_non_normal_matrix():
    """The case the spectral-radius version of the constant got wrong.

    A shifted Jordan block has a tiny spectral radius but a large operator norm, so its
    transient growth is enormous while a constant built from ``rho`` collapses to
    roughly the ``t = 1`` term.  ``gamma`` then comes out far above 1, which contradicts
    its definition as the realised fraction of the worst case.  Connectome matrices are
    non-normal in the same direction, if less extremely (we measure ``||A||_2 / rho`` up
    to 2.1), so this is the property the constant must have rather than a corner case.

    Written to fail loudly against the old constant: with ``rho`` the measured gamma
    here is of order 10^2.
    """
    n, horizon = 20, 15
    A = np.diag(np.ones(n - 1), k=1) * 1.6 + np.eye(n) * 1e-3
    assert spectral_radius(A) < 0.01 < 1.0 < np.linalg.norm(A, 2)

    rng = np.random.default_rng(5)
    W_in = rng.normal(0, 1, (n, 3))
    W_out = rng.normal(0, 1, (2, n))
    for scale in (1e-7, 1e-6, 1e-5):
        E = rng.normal(0, scale, (n, n))
        m = operational_alignment(A, E, W_in, W_out, horizon=horizon)
        assert 0.0 <= m.gamma <= 1.0, (m.gamma, scale)


def test_worst_case_constant_uses_operator_norm_not_spectral_radius():
    """Pin the fix: a non-normal matrix must give a strictly larger constant."""
    n = 10
    A = np.diag(np.ones(n - 1), k=1) + np.eye(n) * 0.5
    rho = spectral_radius(A)
    op = float(np.linalg.norm(A, 2))
    assert op > rho
    assert (worst_case_constant(op, 12, 1.0, 1.0)
            > worst_case_constant(rho, 12, 1.0, 1.0))
