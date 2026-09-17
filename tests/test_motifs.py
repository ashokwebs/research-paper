"""
Unit tests for motif discovery and configuration null models.
"""

import pytest
import numpy as np
from src.connectome.unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata
from src.motifs.motif_engine import MotifDiscoveryEngine


def test_motif_counting_canonical_ffl():
    g = ConnectomeGraph("FFLTest", "Synthetic", "motif")
    g.add_neuron(NeuronMetadata(0, "u", "excitatory", sign=1))
    g.add_neuron(NeuronMetadata(1, "v", "excitatory", sign=1))
    g.add_neuron(NeuronMetadata(2, "w", "excitatory", sign=1))

    # Add u->v, v->w, u->w
    g.add_synapse(SynapseMetadata(0, 1, weight=1.0, sign=1))
    g.add_synapse(SynapseMetadata(1, 2, weight=1.0, sign=1))
    g.add_synapse(SynapseMetadata(0, 2, weight=1.0, sign=1))

    engine = MotifDiscoveryEngine(g)
    counts = engine.count_motifs()

    assert counts["FFL_Total"] == 1
    assert counts["cFFL_Coherent"] == 1
    assert counts["iFFL_Incoherent"] == 0
    assert counts["FBL_3Cycle"] == 0


def test_motif_counting_canonical_iffl():
    g = ConnectomeGraph("iFFLTest", "Synthetic", "motif")
    g.add_neuron(NeuronMetadata(0, "u", "excitatory", sign=1))
    g.add_neuron(NeuronMetadata(1, "v", "inhibitory", sign=-1))
    g.add_neuron(NeuronMetadata(2, "w", "excitatory", sign=1))

    # u->w (+), u->v (+), v->w (-)
    g.add_synapse(SynapseMetadata(0, 2, weight=1.0, sign=1))
    g.add_synapse(SynapseMetadata(0, 1, weight=1.0, sign=1))
    g.add_synapse(SynapseMetadata(1, 2, weight=1.0, sign=-1))

    engine = MotifDiscoveryEngine(g)
    counts = engine.count_motifs()

    assert counts["FFL_Total"] == 1
    assert counts["cFFL_Coherent"] == 0
    assert counts["iFFL_Incoherent"] == 1


def test_dales_null_model_preserves_degrees_and_signs():
    g = ConnectomeGraph("NullTest", "Synthetic", "motif")
    for i in range(10):
        sign = 1 if i < 7 else -1
        g.add_neuron(NeuronMetadata(i, f"N{i}", "neuron", sign=sign))

    # Random edges
    np.random.seed(42)
    for i in range(10):
        for j in range(10):
            if i != j and np.random.rand() < 0.3:
                sign = g.node_metadata[i].sign
                g.add_synapse(SynapseMetadata(i, j, weight=1.0, sign=sign))

    engine = MotifDiscoveryEngine(g)
    G_null = engine.generate_dales_null_model(n_swaps_factor=5)

    # In-degree and out-degree sequence preservation
    orig_in = sorted([d for _, d in g.graph.in_degree()])
    null_in = sorted([d for _, d in G_null.in_degree()])
    assert orig_in == null_in

    orig_out = sorted([d for _, d in g.graph.out_degree()])
    null_out = sorted([d for _, d in G_null.out_degree()])
    assert orig_out == null_out

    # Dale's principle preservation in null model
    for u in G_null.nodes():
        src_sign = g.node_metadata[u].sign
        for v in G_null.successors(u):
            edge_sign = G_null[u][v].get("sign", 1)
            assert edge_sign == src_sign
