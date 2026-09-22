"""
Unit tests for Self-Organized Criticality and Neuronal Avalanche analysis.
"""

import pytest
import numpy as np
from src.theory.criticality import (
    detect_neuronal_avalanches,
    compute_branching_ratio,
    fit_power_law_mle,
    analyze_network_criticality,
)


def test_avalanche_detection():
    """Verify contiguous avalanche size and duration extraction."""
    # 5 time steps, 3 neurons
    # t=0: [1, 0, 1] -> size 2
    # t=1: [0, 1, 0] -> size 1 (continues avalanche, total size=3, dur=2)
    # t=2: [0, 0, 0] -> quiescent (ends avalanche 1)
    # t=3: [1, 1, 1] -> size 3, dur=1 (avalanche 2)
    # t=4: [0, 0, 0] -> quiescent
    spike_train = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    
    sizes, durations = detect_neuronal_avalanches(spike_train, bin_size=1)
    assert len(sizes) == 2
    assert len(durations) == 2
    assert sizes[0] == 3
    assert durations[0] == 2
    assert sizes[1] == 3
    assert durations[1] == 1


def test_branching_ratio_estimation():
    """Verify least-squares branching ratio estimation on critical branching process."""
    np.random.seed(42)
    T = 1000
    N = 50
    # Simulate a critical branching process where A(t+1) = 1.0 * A(t) + small noise
    act = np.zeros(T)
    act[0] = 10
    for t in range(T - 1):
        if act[t] > 0:
            act[t + 1] = max(1, np.random.poisson(act[t]))
        else:
            act[t + 1] = np.random.poisson(5)
            
    # Build a spike train that matches this activity
    spikes = np.zeros((T, N))
    for t in range(T):
        n_spk = min(N, int(act[t]))
        spikes[t, :n_spk] = 1.0
        
    sigma, sigma_err = compute_branching_ratio(spikes)
    # Since Poisson mean = lambda, A(t+1) ~ A(t), sigma should be very close to 1.0
    assert 0.90 <= sigma <= 1.10
    assert sigma_err >= 0.0


def test_power_law_fitting():
    """Verify MLE power-law exponent fit on synthetic Pareto distributed data."""
    np.random.seed(123)
    # Generate Pareto distributed samples with true exponent alpha = 1.5
    # numpy pareto generates x >= 1 with P(x) ~ x^(-(a+1)), so with a=0.5, exponent is 1.5
    samples = (np.random.pareto(a=0.5, size=2000) + 1).astype(int)
    alpha = fit_power_law_mle(list(samples), x_min=1)
    
    # Should be close to 1.5 +/- 0.3
    assert 1.2 <= alpha <= 1.8


def test_full_criticality_profile():
    """Verify that analyze_network_criticality returns a valid AvalancheProfile."""
    np.random.seed(99)
    spikes = (np.random.rand(500, 20) < 0.05).astype(int)
    profile = analyze_network_criticality(spikes, bin_size=1)
    
    assert profile.num_avalanches > 0
    assert profile.mean_avalanche_size >= 1.0
    assert profile.mean_avalanche_duration >= 1.0
    assert isinstance(profile.is_critical, bool)
