"""
COMPUTATIONAL COUPLING — Cycle-Accurate Golden Model for CEN Hardware RTL
Directly models the bit-exact Q8.8 fixed-point arithmetic and state transitions
of the synthesizable Verilog RTL cores:
  - cen_defs.vh
  - dendritic_lif_core.v
  - aer_crossbar_router.v
  - synaptic_accumulator.v
  - cen_accelerator_top.v
"""

from typing import List, Dict, Tuple, Any
import numpy as np


class FixedPointQ88:
    """Emulates 16-bit signed Q8.8 fixed-point arithmetic exactly as in Verilog."""

    @staticmethod
    def to_q88(val: float) -> int:
        ival = int(round(val * 256.0))
        # Clip to signed 16-bit [-32768, 32767]
        return max(-32768, min(32767, ival))

    @staticmethod
    def from_q88(ival: int) -> float:
        # Sign-extend 16-bit
        if ival >= 32768:
            ival -= 65536
        return float(ival) / 256.0

    @staticmethod
    def mult(a: int, b: int) -> int:
        prod = a * b
        res = (prod >> 8) & 0xFFFF
        if res >= 32768:
            res -= 65536
        return res


class DendriticLIFCoreRTLModel:
    """Bit-exact Python behavioral model of dendritic_lif_core.v."""

    def __init__(self, neuron_id: int):
        self.id = neuron_id
        self.v_dend = 0       # Q8.8
        self.v_soma = 0       # Q8.8
        self.refr_counter = 0 # 3-bit
        self.nmda_active = False

        # Constants matching cen_defs.vh
        self.theta_dend = 0x0073    # 0.45
        self.v_thresh_soma = 0x0100 # 1.00
        self.v_plateau = 0x00D9     # 0.85
        self.g_nmda = 0x005A        # 0.35
        self.g_axial = 0x0040       # 0.25
        self.alpha_soma = 0x00F3    # 0.9512
        self.alpha_dend = 0x00F9    # 0.9753
        self.alpha_plateau = 0x00FE # 0.9933
        self.shunt_eff = 0x00F3     # 0.95

    def step(self, i_syn_dend: int, i_syn_soma_inh: int) -> Tuple[bool, int, int]:
        """Execute one 1-ms step_tick."""
        # 1. NMDA state machine
        if self.v_dend >= self.theta_dend:
            self.nmda_active = True
        elif self.v_dend < (self.theta_dend - 0x0020):
            self.nmda_active = False

        # 2. Dendritic decay and drive
        alpha_eff = self.alpha_plateau if self.nmda_active else self.alpha_dend
        v_dend_decayed = FixedPointQ88.mult(alpha_eff, self.v_dend)

        plateau_diff = self.v_plateau - self.v_dend
        nmda_drive = FixedPointQ88.mult(self.g_nmda, plateau_diff) if self.nmda_active else 0

        self.v_dend = v_dend_decayed + i_syn_dend + nmda_drive

        # 3. Axial current to soma
        v_diff = self.v_dend - self.v_soma
        i_axial = FixedPointQ88.mult(self.g_axial, v_diff)

        # 4. Somatic decay and shunting
        v_soma_decayed = FixedPointQ88.mult(self.alpha_soma, self.v_soma)
        shunt_active = (i_syn_soma_inh > 0x0010)

        spike_out = False
        if self.refr_counter > 0:
            self.refr_counter -= 1
            self.v_soma = 0
        else:
            if shunt_active:
                shunt_factor = 0x0100 - self.shunt_eff
                self.v_soma = FixedPointQ88.mult(v_soma_decayed, shunt_factor)
            else:
                v_cand = v_soma_decayed + i_axial
                if v_cand >= self.v_thresh_soma:
                    spike_out = True
                    self.v_soma = 0
                    self.refr_counter = 2
                else:
                    self.v_soma = v_cand

        return spike_out, self.v_dend, self.v_soma


class CENAcceleratorRTLModel:
    """Behavioral top-level model of cen_accelerator_top.v."""

    def __init__(self, num_neurons: int = 128, num_exc: int = 102):
        self.num_neurons = num_neurons
        self.num_exc = num_exc
        self.cores = [DendriticLIFCoreRTLModel(i) for i in range(num_neurons)]
        self.accum_dend = [0] * num_neurons
        self.accum_soma_inh = [0] * num_neurons

        # Telemetry counters
        self.total_synops = 0
        self.total_spikes = 0
        self.total_plateaus = 0

    def dispatch_aer_packet(self, packet_32b: int):
        """Simulate ingress of 32-bit AER packet through crossbar router."""
        pre_idx = (packet_32b >> 8) & 0x7F
        post_idx = packet_32b & 0x7F
        weight_raw = (packet_32b >> 16) & 0xFF
        weight_q88 = weight_raw # Low byte in Q8.8

        is_exc = (pre_idx < self.num_exc)
        if post_idx < self.num_neurons:
            if is_exc:
                self.accum_dend[post_idx] += weight_q88
            else:
                self.accum_soma_inh[post_idx] += weight_q88
            self.total_synops += 1

    def step_tick(self) -> List[int]:
        """Execute one 1-ms step tick across all 128 cores."""
        spikes = []
        for i in range(self.num_neurons):
            spk, vd, vs = self.cores[i].step(self.accum_dend[i], self.accum_soma_inh[i])
            if spk:
                spikes.append(i)
                self.total_spikes += 1
            if self.cores[i].nmda_active:
                self.total_plateaus += 1
            # Clear accumulators
            self.accum_dend[i] = 0
            self.accum_soma_inh[i] = 0
        return spikes
