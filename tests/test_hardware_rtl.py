"""
Unit tests for CoupledEventNet Hardware RTL Golden Model.
Verifies synthesizable Verilog microarchitecture logic:
  1. Fixed-point arithmetic precision (Q8.8 representation).
  2. AER packet decoding and Dale's principle sign classification.
  3. Two-compartment dendritic NMDA plateau state machine.
  4. Perisomatic shunting inhibition clamp.
  5. Multi-cycle hardware telemetry counters.
"""

import pytest
import numpy as np
from hardware.sim.cen_rtl_golden_model import (
    FixedPointQ88,
    DendriticLIFCoreRTLModel,
    CENAcceleratorRTLModel,
)


def test_q88_fixed_point_arithmetic():
    """Verify Q8.8 fixed-point conversion and multiplication precision."""
    val1 = 0.5
    val2 = 0.25
    q1 = FixedPointQ88.to_q88(val1)
    q2 = FixedPointQ88.to_q88(val2)

    assert q1 == 128
    assert q2 == 64

    q_prod = FixedPointQ88.mult(q1, q2)
    f_prod = FixedPointQ88.from_q88(q_prod)
    assert abs(f_prod - 0.125) < 0.005


def test_dendritic_core_nmda_plateau_state_machine():
    """Verify that dendritic core triggers and sustains regenerative NMDA plateaus."""
    core = DendriticLIFCoreRTLModel(neuron_id=0)

    # Subthreshold drive: 0.2 in Q8.8
    sub_drive = FixedPointQ88.to_q88(0.2)
    spike, vd, vs = core.step(sub_drive, 0)
    assert not core.nmda_active
    assert vd < core.theta_dend

    # Suprathreshold drive: 0.6 in Q8.8 charges dendrite past theta_dend
    supra_drive = FixedPointQ88.to_q88(0.6)
    spike, vd, vs = core.step(supra_drive, 0)
    assert vd >= core.theta_dend

    # Clock cycle t+1: registered state machine latches nmda_active and sustains plateau
    spike, vd, vs = core.step(0, 0)
    assert core.nmda_active
    assert vd > FixedPointQ88.to_q88(0.4)


def test_dendritic_core_perisomatic_shunting():
    """Verify that perisomatic inhibition suppresses somatic action potentials."""
    core_uninhibited = DendriticLIFCoreRTLModel(neuron_id=1)
    core_inhibited = DendriticLIFCoreRTLModel(neuron_id=2)

    # Prime dendrites to near threshold
    strong_drive = FixedPointQ88.to_q88(1.2)
    core_uninhibited.step(strong_drive, 0)
    core_inhibited.step(strong_drive, FixedPointQ88.to_q88(1.5)) # Strong GABA shunting

    # Uninhibited neuron should transmit axial current to soma; inhibited neuron shunted
    assert core_inhibited.v_soma < core_uninhibited.v_soma


def test_aer_router_and_telemetry_counters():
    """Verify top-level AER packet processing and telemetry counters."""
    top = CENAcceleratorRTLModel(num_neurons=128, num_exc=102)

    # Packet 1: Excitatory pre-synaptic 10 targeting post-synaptic 5 with weight 32
    # [31]=1, [23:16]=32, [14:8]=10, [6:0]=5
    pkt_exc = (1 << 31) | (32 << 16) | (10 << 8) | 5
    top.dispatch_aer_packet(pkt_exc)

    assert top.accum_dend[5] == 32
    assert top.accum_soma_inh[5] == 0
    assert top.total_synops == 1

    # Packet 2: Inhibitory pre-synaptic 110 targeting post-synaptic 5 with weight 64
    pkt_inh = (1 << 31) | (64 << 16) | (110 << 8) | 5
    top.dispatch_aer_packet(pkt_inh)

    assert top.accum_dend[5] == 32
    assert top.accum_soma_inh[5] == 64
    assert top.total_synops == 2

    # Step simulation
    spikes = top.step_tick()
    # After step_tick, accumulators must be cleared
    assert top.accum_dend[5] == 0
    assert top.accum_soma_inh[5] == 0
