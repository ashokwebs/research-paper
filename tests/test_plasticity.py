"""
Unit tests for Dale-Constrained Motif-Gated STDP (DCMG-STDP) plasticity engine.
"""

import numpy as np
import pytest
from src.learning.motif_gated_stdp import (
    DaleConstrainedMotifGatedSTDP,
    STDPConfig,
    extract_motif_protection_mask,
    run_continual_learning_experiment,
)


def test_motif_protection_mask_extraction():
    # 4 neurons: 0, 1, 2 exc (+1), 3 inh (-1)
    # Form a cFFL: 0 -> 1 -> 2 and 0 -> 2
    N = 4
    adj = np.zeros((N, N), dtype=np.float32)
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    adj[0, 2] = 1.0

    dale_signs = np.array([1, 1, 1, -1], dtype=np.float32)
    mask = extract_motif_protection_mask(adj, dale_signs)

    # cFFL edges should be protected
    assert mask[0, 1] == 1.0
    assert mask[1, 2] == 1.0
    assert mask[0, 2] == 1.0


def test_dcmg_stdp_weight_update_and_protection():
    N = 4
    weights = np.ones((N, N), dtype=np.float32) * 0.5
    dale_signs = np.array([1, 1, -1, -1], dtype=np.float32)
    # Protect synapse (0, 1) but leave (0, 2) unprotected
    motif_mask = np.zeros((N, N), dtype=np.float32)
    motif_mask[0, 1] = 1.0  # 95% protected

    cfg = STDPConfig(core_protection_factor=0.95, a_plus=0.1, a_minus=0.0)
    stdp = DaleConstrainedMotifGatedSTDP(weights, dale_signs, motif_mask=motif_mask, config=cfg)

    # Pre-synaptic spike at 0, post-synaptic spike at 1 and 2
    pre = np.array([1, 0, 0, 0], dtype=np.float32)
    post = np.array([0, 1, 1, 0], dtype=np.float32)

    # First step creates trace
    stdp.step(pre, post, enable_learning=True)
    # Second step applies LTP
    stdp.step(pre, post, enable_learning=True)

    # The unprotected synapse (0, 2) should experience significantly more LTP than protected (0, 1)
    delta_protected = stdp.W[0, 1] - 0.5
    delta_unprotected = stdp.W[0, 2] - 0.5

    assert delta_unprotected > delta_protected * 5.0
    # Weights must stay non-negative (Dale compliance)
    assert np.all(stdp.W >= 0.0)


def test_continual_learning_experiment_runs():
    results = run_continual_learning_experiment(num_samples_per_task=10, time_steps=15, num_neurons=32)
    assert len(results) == 4
    dcmg = next(r for r in results if "DCMG-STDP" in r.method)
    assert dcmg.task_a_retained_acc >= 50.0
