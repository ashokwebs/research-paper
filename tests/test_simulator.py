"""
Unit tests for VectorizedLIFSimulator and hardware SynOps tracking.
"""

import pytest
import numpy as np
from src.connectome.unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata
from src.simulation.lif_simulator import VectorizedLIFSimulator, SimulationConfig


def test_lif_single_neuron_spiking_and_refractory():
    g = ConnectomeGraph("SingleNeuron", "Synthetic", "test")
    g.add_neuron(NeuronMetadata(0, "N0", "excitatory", sign=1))

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=10.0, tau_ref_ms=2.0)
    sim = VectorizedLIFSimulator(g, config=cfg)

    # Apply 10 ms pulse of 60 mV (superthreshold)
    stim = np.zeros((40, 1), dtype=np.float32)
    stim[5:25, 0] = 60.0

    res = sim.run(20.0, stim, record_voltages=True)
    assert res.total_spikes >= 1
    # Check that voltages are bounded by reversal potential
    assert res.voltage_trace.min() >= -85.0


def test_synops_counting_exact_out_degree():
    g = ConnectomeGraph("SynOpTest", "Synthetic", "test")
    # Neuron 0 connects to 3 targets (out-degree = 3)
    g.add_neuron(NeuronMetadata(0, "Source", "excitatory", sign=1))
    for i in range(1, 4):
        g.add_neuron(NeuronMetadata(i, f"T{i}", "excitatory", sign=1))
        g.add_synapse(SynapseMetadata(0, i, weight=1.0, sign=1))

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=10.0)
    sim = VectorizedLIFSimulator(g, config=cfg)

    # Excite only Neuron 0
    stim = np.zeros((40, 4), dtype=np.float32)
    stim[5:15, 0] = 60.0

    res = sim.run(20.0, stim)
    spikes_0 = int(res.spike_raster[:, 0].sum())

    # Every spike of neuron 0 must produce exactly 3 SynOps
    assert res.total_synops == spikes_0 * 3
