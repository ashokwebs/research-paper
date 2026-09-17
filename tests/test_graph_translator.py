"""
Unit tests for multi-scale invariant motif graph translator.
"""

import numpy as np
import pytest
from src.connectome.graph_translator import (
    InvariantMotifGraphTranslator,
    run_cross_scale_graph_translation,
)


def test_graph_translator_generation_and_dale():
    translator = InvariantMotifGraphTranslator(target_ei_ratio=2.69, seed=42)
    N = 120
    signed_adj, dale_signs, metrics = translator.generate_scaled_connectome(N)

    assert signed_adj.shape == (N, N)
    assert len(dale_signs) == N
    assert metrics.num_neurons == N
    assert metrics.num_synapses > 0

    # Verify Dale compliance: all outgoing connections from neuron u must match dale_signs[u]
    for u in range(N):
        row = signed_adj[u, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            if dale_signs[u] > 0:
                assert np.all(nonzero > 0), f"Neuron {u} is excitatory but has negative weights"
            else:
                assert np.all(nonzero < 0), f"Neuron {u} is inhibitory but has positive weights"


def test_graph_translator_motif_presence():
    translator = InvariantMotifGraphTranslator(seed=42)
    N = 150
    _, _, metrics = translator.generate_scaled_connectome(N)

    assert metrics.num_cffl > 0
    assert metrics.num_iffl > 0
    assert metrics.num_rip > 0
    assert metrics.spectral_radius > 0.0
    assert metrics.spectral_gap > 0.0


def test_cross_scale_scaling_runs():
    sizes = [60, 120]
    results = run_cross_scale_graph_translation(sizes=sizes)
    assert len(results) == 2
    assert results[0].num_neurons == 60
    assert results[1].num_neurons == 120
    assert results[1].num_synapses > results[0].num_synapses
