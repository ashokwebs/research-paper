"""
COMPUTATIONAL COUPLING — Two-Compartment Biophysical LIF Simulator
Implements active dendritic compartmentalization:
  - Dendritic compartment (V_d): Synaptic integration with nonlinear NMDA-like supralinear plateau
  - Somatic compartment (V_s): Forward axial transfer g_c * (V_d - V_s), direct perisomatic inhibition, spike generation
Tracks somatic SynOps, dendritic events, and energy dissipation proxy.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
from ..connectome.unified_schema import ConnectomeGraph


@dataclass
class TwoCompartmentConfig:
    """Biophysical parameters for two-compartment LIF simulation."""
    dt_ms: float = 0.5
    # Somatic properties
    v_thresh_soma: float = -50.0  # mV
    v_rest: float = -70.0         # mV
    v_reset: float = -75.0        # mV
    tau_soma_ms: float = 20.0     # ms
    t_ref_ms: float = 2.0         # ms
    # Dendritic properties
    tau_dend_ms: float = 25.0     # ms
    theta_dend: float = -58.0     # mV: threshold for active dendritic NMDA plateau
    g_nmda: float = 25.0          # mV equivalent boost during active NMDA plateau
    # Axial coupling
    g_c: float = 1.2              # Coupling conductance factor between soma and dendrite
    # Synaptic parameters
    tau_syn_ms: float = 5.0       # ms
    # Energy model
    energy_per_synop_nj: float = 0.020
    energy_per_somatic_spike_nj: float = 0.005
    energy_per_dendritic_spike_nj: float = 0.002


@dataclass
class TwoCompartmentResult:
    """Output results of two-compartment simulation."""
    time_points_ms: np.ndarray
    soma_spike_raster: np.ndarray       # Shape (T, N), bool
    dend_plateau_raster: np.ndarray     # Shape (T, N), bool
    total_soma_spikes: int
    total_dend_plateaus: int
    total_synops: int
    energy_proxy_nj: float
    soma_voltages: Optional[np.ndarray] = None  # Shape (T, N)
    dend_voltages: Optional[np.ndarray] = None  # Shape (T, N)


class TwoCompartmentLIFSimulator:
    """
    Vectorized Two-Compartment Leaky Integrate-and-Fire Simulator.
    Simulates coupled dendritic and somatic dynamics across connectome topologies.
    """

    def __init__(self, connectome: ConnectomeGraph, config: Optional[TwoCompartmentConfig] = None):
        self.connectome = connectome
        self.cfg = config if config is not None else TwoCompartmentConfig()

        self.num_neurons = connectome.num_neurons
        self.adj_matrix, self.node_ids = connectome.get_adjacency_matrix(weighted=True, signed=True)
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        # Out-degree vector for exact SynOp computation
        self.out_degrees = np.count_nonzero(self.adj_matrix, axis=1)

        # Separate excitatory and inhibitory synaptic weight matrices
        # In our convention, adj_matrix[i, j] is connection from i -> j
        self.w_exc = np.maximum(0.0, self.adj_matrix)
        self.w_inh = np.maximum(0.0, -self.adj_matrix)

    def run(
        self,
        duration_ms: float,
        external_stimulus: np.ndarray,
        record_voltages: bool = False,
    ) -> TwoCompartmentResult:
        """
        Run vectorized simulation of coupled soma-dendrite dynamics.

        Parameters
        ----------
        duration_ms : float
            Simulation time in milliseconds.
        external_stimulus : np.ndarray
            Shape (time_steps, N). External input current into dendritic compartments.
        record_voltages : bool
            If True, saves continuous somatic and dendritic membrane potentials.
        """
        dt = self.cfg.dt_ms
        time_steps = int(round(duration_ms / dt))
        N = self.num_neurons

        if external_stimulus.shape[0] != time_steps or external_stimulus.shape[1] != N:
            raise ValueError(
                f"Stimulus shape {external_stimulus.shape} does not match (time_steps={time_steps}, N={N})"
            )

        # Precompute constants
        alpha_soma = float(np.exp(-dt / self.cfg.tau_soma_ms))
        alpha_dend = float(np.exp(-dt / self.cfg.tau_dend_ms))
        alpha_syn = float(np.exp(-dt / self.cfg.tau_syn_ms))
        syn_scale = float(1.0 / max(1e-4, (1.0 - alpha_dend)))
        g_c = self.cfg.g_c

        # State vectors
        v_s = np.full(N, self.cfg.v_rest, dtype=np.float32)
        v_d = np.full(N, self.cfg.v_rest, dtype=np.float32)
        i_syn_exc = np.zeros(N, dtype=np.float32)
        i_syn_inh = np.zeros(N, dtype=np.float32)
        refractory_timer = np.zeros(N, dtype=np.float32)

        # Output rasters
        soma_spikes = np.zeros((time_steps, N), dtype=bool)
        dend_plateaus = np.zeros((time_steps, N), dtype=bool)

        v_s_rec = np.zeros((time_steps, N), dtype=np.float32) if record_voltages else None
        v_d_rec = np.zeros((time_steps, N), dtype=np.float32) if record_voltages else None

        total_synops = 0

        for t in range(time_steps):
            if record_voltages:
                v_s_rec[t, :] = v_s
                v_d_rec[t, :] = v_d

            # 1. Decay synaptic currents
            i_syn_exc *= alpha_syn
            i_syn_inh *= alpha_syn

            # 2. Add external stimulus to dendritic compartment
            i_ext_t = external_stimulus[t, :]

            # 3. Active NMDA-like supralinear plateau in dendrite
            nmda_active = v_d > self.cfg.theta_dend
            dend_plateaus[t, :] = nmda_active
            i_nmda = np.where(nmda_active, self.cfg.g_nmda, 0.0)

            # 4. Dendritic potential update
            # Forward coupling: g_c * (v_s - v_d)
            v_d_drive = i_ext_t + i_syn_exc + i_nmda + g_c * (v_s - v_d)
            v_d = self.cfg.v_rest + (v_d - self.cfg.v_rest) * alpha_dend + v_d_drive * (1.0 - alpha_dend)

            # 5. Somatic potential update
            # Soma receives forward drive from dendrite: g_c * (v_d - v_s)
            # Perisomatic inhibition directly shunts soma: -i_syn_inh
            not_refractory = refractory_timer <= 0.0
            v_s_drive = g_c * (v_d - v_s) - i_syn_inh
            v_s_new = self.cfg.v_rest + (v_s - self.cfg.v_rest) * alpha_soma + v_s_drive * (1.0 - alpha_soma)
            v_s = np.where(not_refractory, v_s_new, self.cfg.v_reset)

            # 6. Somatic spike generation
            spiked = (v_s >= self.cfg.v_thresh_soma) & not_refractory
            soma_spikes[t, :] = spiked

            if np.any(spiked):
                # Somatic reset and refractory timer
                v_s[spiked] = self.cfg.v_reset
                refractory_timer[spiked] = self.cfg.t_ref_ms

                # Synaptic transmission: pre-synaptic somatic spike triggers post-synaptic dendritic currents
                spiking_indices = np.where(spiked)[0]
                total_synops += int(self.out_degrees[spiking_indices].sum())

                # Transmit excitation to post-synaptic dendritic compartments
                # Note: w_exc[i, j] is connection from i -> j, so incoming to j is sum over spiking i
                i_syn_exc += self.w_exc[spiking_indices, :].sum(axis=0) * (syn_scale * 0.15)
                # Transmit perisomatic inhibition directly to soma
                i_syn_inh += self.w_inh[spiking_indices, :].sum(axis=0) * (syn_scale * 0.25)

            # Decrement refractory timers
            refractory_timer = np.maximum(0.0, refractory_timer - dt)

        total_soma_spikes = int(soma_spikes.sum())
        total_dend_plateaus = int(dend_plateaus.sum())

        energy_nj = (
            total_synops * self.cfg.energy_per_synop_nj
            + total_soma_spikes * self.cfg.energy_per_somatic_spike_nj
            + total_dend_plateaus * self.cfg.energy_per_dendritic_spike_nj
        )

        return TwoCompartmentResult(
            time_points_ms=np.arange(time_steps) * dt,
            soma_spike_raster=soma_spikes,
            dend_plateau_raster=dend_plateaus,
            total_soma_spikes=total_soma_spikes,
            total_dend_plateaus=total_dend_plateaus,
            total_synops=total_synops,
            energy_proxy_nj=float(energy_nj),
            soma_voltages=v_s_rec,
            dend_voltages=v_d_rec,
        )
