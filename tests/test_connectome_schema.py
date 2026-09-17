"""
Unit tests for unified connectome graph schema and Dale's principle validation.
"""

import pytest
import numpy as np
from src.connectome.unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata


def test_connectome_graph_initialization():
    g = ConnectomeGraph("TestGraph", "Synthetic", "microcircuit")
    assert g.name == "TestGraph"
    assert g.organism == "Synthetic"
    assert g.num_neurons == 0
    assert g.num_synapses == 0
    assert g.density == 0.0


def test_dales_principle_compliance_and_violation():
    g = ConnectomeGraph("DaleTest", "Synthetic", "microcircuit")

    # Add 2 neurons: 0 is excitatory (+1), 1 is inhibitory (-1)
    g.add_neuron(NeuronMetadata(0, "N0", "excitatory", sign=1))
    g.add_neuron(NeuronMetadata(1, "N1", "inhibitory", sign=-1))
    g.add_neuron(NeuronMetadata(2, "N2", "target", sign=1))

    # Valid connections: 0 -> 2 (+1), 1 -> 2 (-1)
    g.add_synapse(SynapseMetadata(0, 2, weight=1.0, sign=1))
    g.add_synapse(SynapseMetadata(1, 2, weight=1.0, sign=-1))

    is_valid, violations = g.validate_dales_principle()
    assert is_valid is True
    assert violations == 0

    # Intentionally violate Dale's principle: add negative connection from excitatory neuron 0
    g.graph.add_edge(0, 1, sign=-1, weight=1.0)
    is_valid, violations = g.validate_dales_principle()
    assert is_valid is False
    assert violations == 1


def test_adjacency_matrix_extraction():
    g = ConnectomeGraph("AdjTest", "Synthetic", "microcircuit")
    g.add_neuron(NeuronMetadata(10, "N10", "excitatory", sign=1))
    g.add_neuron(NeuronMetadata(20, "N20", "inhibitory", sign=-1))

    g.add_synapse(SynapseMetadata(10, 20, weight=5.0, sign=1))

    adj, nodes = g.get_adjacency_matrix(weighted=True, signed=True)
    assert nodes == [10, 20]
    assert adj.shape == (2, 2)
    assert adj[0, 1] == 5.0
    assert adj[1, 0] == 0.0


def test_subcircuit_extraction():
    g = ConnectomeGraph("SubTest", "Synthetic", "microcircuit")
    for i in range(5):
        g.add_neuron(NeuronMetadata(i, f"N{i}", "excitatory", sign=1))

    g.add_synapse(SynapseMetadata(0, 1, weight=2.0, sign=1))
    g.add_synapse(SynapseMetadata(1, 2, weight=3.0, sign=1))
    g.add_synapse(SynapseMetadata(3, 4, weight=4.0, sign=1))

    sub = g.get_subcircuit({0, 1, 2})
    assert sub.num_neurons == 3
    assert sub.num_synapses == 2
    assert sub.graph.has_edge(0, 1)
    assert sub.graph.has_edge(1, 2)
    assert not sub.graph.has_edge(3, 4)
