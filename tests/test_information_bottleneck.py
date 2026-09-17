"""
Unit tests for Information Bottleneck and Rate-Distortion theory module.
Verifies:
  1. Discrete entropy and mutual information estimators.
  2. Gaussian copula parametric estimation.
  3. Theoretical Rate-Distortion bound calculation.
  4. End-to-end network Information Bottleneck evaluation profile.
"""

import pytest
import numpy as np
from src.theory.information_bottleneck import (
    estimate_entropy_discrete,
    estimate_mutual_information_discrete,
    compute_gaussian_copula_mutual_info,
    compute_rate_distortion_bound,
    evaluate_network_information_bottleneck,
    InformationBottleneckProfile,
)


def test_entropy_discrete():
    """Verify Shannon entropy calculation."""
    # Uniform distribution over 4 states -> log2(4) = 2.0 bits
    probs_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    h = estimate_entropy_discrete(probs_uniform)
    assert abs(h - 2.0) < 1e-5

    # Deterministic distribution -> 0.0 bits
    probs_det = np.array([1.0, 0.0, 0.0])
    h_det = estimate_entropy_discrete(probs_det)
    assert abs(h_det - 0.0) < 1e-5


def test_mutual_information_discrete():
    """Verify mutual information between identical and independent variables."""
    # Identical variables
    x = np.random.randint(0, 5, size=200)
    mi_self = estimate_mutual_information_discrete(x, x, num_bins=5)
    assert mi_self > 1.0

    # Independent variables
    np.random.seed(42)
    x_rand = np.random.randint(0, 5, size=500)
    y_rand = np.random.randint(0, 5, size=500)
    mi_indep = estimate_mutual_information_discrete(x_rand, y_rand, num_bins=5)
    assert mi_indep < mi_self


def test_gaussian_copula_mutual_info():
    """Verify Gaussian copula mutual information on correlated linear signals."""
    np.random.seed(42)
    x = np.random.randn(200, 3)
    y = x + 0.05 * np.random.randn(200, 3) # Strongly correlated
    mi = compute_gaussian_copula_mutual_info(x, y)
    assert mi > 2.0


def test_rate_distortion_bound():
    """Verify theoretical Rate-Distortion bound monotonicity and bounds."""
    h_x = 3.0
    h_y = 2.5
    i_xm, i_my = compute_rate_distortion_bound(h_x, h_y)
    assert len(i_xm) == len(i_my)
    # Monotonically increasing
    assert np.all(np.diff(i_my) >= 0.0)
    # Bounded by H(Y)
    assert np.all(i_my <= h_y)


def test_network_information_bottleneck_evaluation():
    """Verify full evaluation profile on synthetic network states."""
    np.random.seed(42)
    N_samples = 150
    T_steps = 20
    D_in = 8
    D_hidden = 32

    X = np.random.randn(N_samples, T_steps, D_in)
    # Hidden states correlate with input
    M = np.repeat(np.mean(X, axis=-1, keepdims=True), D_hidden, axis=-1) + 0.1 * np.random.randn(N_samples, T_steps, D_hidden)
    # Targets correlate with hidden states
    Y = (np.mean(M, axis=(1, 2)) > 0.0).astype(int)

    profile = evaluate_network_information_bottleneck(X, M, Y, organism="C. elegans", architecture="ActiveDendriticCEN")
    assert isinstance(profile, InformationBottleneckProfile)
    assert profile.mutual_info_xm_bits >= 0.0
    assert profile.mutual_info_my_bits >= 0.0
    assert 0.0 <= profile.compression_ratio <= 1.0
