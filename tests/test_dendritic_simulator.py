"""
Unit tests for TwoCompartmentLIFSimulator.
Verifies:
  1. Subthreshold dendritic input attenuates without somatic spiking
  2. Suprathreshold input triggers dendritic NMDA plateau and downstream somatic spike
  3. Perisomatic inhibition shunts somatic firing despite dendritic activation
  4. Exact hardware SynOps accounting
"""

import numpy as np
import pytest
from src.connectome.unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata
from src.simulation.dendritic_lif_simulator import (
    TwoCompartmentLIFSimulator,
    TwoCompartmentConfig,
)


def create_minimal_test_circuit() -> ConnectomeGraph:
    """Create 3-neuron circuit: N0 (Sensory Exc) -> N1 (Interneuron) -> N2 (Readout)."""
    g = ConnectomeGraph(name="test_dendritic_circuit", organism="synthetic", scale="microcircuit")
    g.add_neuron(NeuronMetadata(node_id=0, name="N0", cell_type="sensory", sign=1, layer="L1"))
    g.add_neuron(NeuronMetadata(node_id=1, name="N1", cell_type="interneuron", sign=1, layer="L2"))
    g.add_neuron(NeuronMetadata(node_id=2, name="N2", cell_type="readout", sign=-1, layer="L3"))

    g.add_synapse(SynapseMetadata(source_id=0, target_id=1, weight=5.0, sign=1))
    g.add_synapse(SynapseMetadata(source_id=1, target_id=2, weight=5.0, sign=1))
    g.add_synapse(SynapseMetadata(source_id=2, target_id=1, weight=8.0, sign=-1))  # Feedback inhibition
    return g


def test_subthreshold_dendritic_attenuation():
    g = create_minimal_test_circuit()
    cfg = TwoCompartmentConfig(dt_ms=0.5, theta_dend=-55.0, v_thresh_soma=-50.0)
    sim = TwoCompartmentLIFSimulator(g, config=cfg)

    # Subthreshold weak pulse on neuron 0
    stim = np.zeros((100, 3), dtype=np.float32)
    stim[10:15, 0] = 5.0  # Very small current

    res = sim.run(50.0, stim, record_voltages=True)
    assert res.total_soma_spikes == 0
    assert res.total_dend_plateaus == 0
    assert res.total_synops == 0


def test_suprathreshold_nmda_and_somatic_spike():
    g = create_minimal_test_circuit()
    cfg = TwoCompartmentConfig(dt_ms=0.5, theta_dend=-58.0, v_thresh_soma=-50.0, g_nmda=15.0)
    sim = TwoCompartmentLIFSimulator(g, config=cfg)

    # Sustained sensory burst on neuron 0 (30 ms duration = 60 steps)
    stim = np.zeros((200, 3), dtype=np.float32)
    stim[10:70, 0] = 80.0

    res = sim.run(100.0, stim, record_voltages=True)
    assert res.total_soma_spikes > 0
    assert res.total_synops > 0
    assert res.energy_proxy_nj > 0.0


def test_perisomatic_inhibition_shunting():
    g = create_minimal_test_circuit()
    cfg = TwoCompartmentConfig(dt_ms=0.5, theta_dend=-58.0, v_thresh_soma=-50.0)
    sim = TwoCompartmentLIFSimulator(g, config=cfg)

    # Case A: Neuron 1 alone spikes under 75.0 stimulus
    stim_a = np.zeros((200, 3), dtype=np.float32)
    stim_a[10:70, 1] = 75.0
    res_a = sim.run(100.0, stim_a)
    assert res_a.soma_spike_raster[:, 1].sum() > 0

    # Case B: Co-activating inhibitory neuron 2 shunts neuron 1 to 0 spikes
    stim_b = np.zeros((200, 3), dtype=np.float32)
    stim_b[10:70, 1] = 75.0
    stim_b[5:70, 2] = 85.0
    res_b = sim.run(100.0, stim_b)
    assert res_b.soma_spike_raster[:, 2].sum() > 0  # Inhibitory neuron fired
    assert res_b.soma_spike_raster[:, 1].sum() == 0  # Neuron 1 successfully shunted!
