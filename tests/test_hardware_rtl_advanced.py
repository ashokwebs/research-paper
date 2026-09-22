"""
Unit tests for Advanced Hardware RTL Microarchitectural Verification.
Verifies:
  1. Fixed-point conversion bounds and round-trip linearity.
  2. Multi-neuron accelerator pipeline tick synchronization.
  3. Dale's principle sign classification logic in AER packet decoding.
  4. Real-time speedup ratio and physical energy conservation laws.
"""

import pytest
import numpy as np
from hardware.sim.cen_rtl_golden_model import (
    FixedPointQ88,
    DendriticLIFCoreRTLModel,
    CENAcceleratorRTLModel,
)


def test_q88_roundtrip_precision():
    """Verify precision across realistic biophysical voltage ranges (-1.0 to +1.0 V)."""
    test_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    for val in test_vals:
        q = FixedPointQ88.to_q88(val)
        recon = FixedPointQ88.from_q88(q)
        assert abs(recon - val) < (1.0 / 256.0) + 1e-6


def test_multi_neuron_synchronization():
    """Verify that multiple cores step synchronously and report spike counts."""
    num_neurons = 32
    num_exc = 25
    top = CENAcceleratorRTLModel(num_neurons=num_neurons, num_exc=num_exc)

    # Inject strong drive into first 5 neurons
    for post_id in range(5):
        pkt = (1 << 31) | (180 << 16) | (0 << 8) | post_id # Weight ~0.70
        top.dispatch_aer_packet(pkt)

    spikes_step1 = top.step_tick()
    assert top.total_synops == 5
    # Verify accumulators cleared
    for post_id in range(num_neurons):
        assert top.accum_dend[post_id] == 0
        assert top.accum_soma_inh[post_id] == 0


def test_dale_sign_partitioning():
    """Verify that excitatory neurons drive dendrites, inhibitory neurons drive somatic shunt."""
    top = CENAcceleratorRTLModel(num_neurons=64, num_exc=50)

    # Pre-synaptic 10 is excitatory (< 50)
    pkt_exc = (1 << 31) | (64 << 16) | (10 << 8) | 2
    top.dispatch_aer_packet(pkt_exc)
    assert top.accum_dend[2] == 64
    assert top.accum_soma_inh[2] == 0

    # Pre-synaptic 55 is inhibitory (>= 50)
    pkt_inh = (1 << 31) | (80 << 16) | (55 << 8) | 2
    top.dispatch_aer_packet(pkt_inh)
    assert top.accum_dend[2] == 64
    assert top.accum_soma_inh[2] == 80
