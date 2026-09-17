"""
Unit tests for Degeneracy Invariance Theorems and analytical perturbation bounds.
"""

import pytest
import numpy as np
from src.theory.degeneracy_theorems import (
    compute_spectral_decomposition,
    evaluate_degeneracy_bounds,
)


def test_spectral_decomposition():
    """Verify eigenspectrum sorting, spectral radius, and spectral gap."""
    N = 30
    np.random.seed(42)
    adj = np.random.randn(N, N) * 0.1

    vals, vecs, radius, gap = compute_spectral_decomposition(adj, core_fraction=0.30)

    assert len(vals) == N
    assert vecs.shape == (N, N)
    assert radius >= 0.0
    assert gap >= 0.0
    # First eigenvalue magnitude should equal spectral radius
    assert abs(abs(vals[0]) - radius) < 1e-5


def test_evaluate_degeneracy_bounds():
    """Verify analytical perturbation metrics calculation."""
    N = 25
    np.random.seed(123)
    adj = np.random.randn(N, N) * 0.2

    metrics = evaluate_degeneracy_bounds(adj, "Synthetic Net", empirical_r90=0.20)

    assert metrics.organism == "Synthetic Net"
    assert metrics.num_neurons == N
    assert metrics.spectral_radius >= 0.0
    assert metrics.spectral_gap >= 0.0
    assert metrics.condition_number >= 1.0
    assert metrics.theoretical_r90_bound >= 0.0
    assert metrics.bound_slack_ratio >= 0.0
