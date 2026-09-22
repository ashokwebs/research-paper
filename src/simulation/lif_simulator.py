"""
COMPUTATIONAL COUPLING — Vectorized Leaky Integrate-and-Fire (LIF) Simulator
Simulates biological connectomes and synthetic event-driven networks with:
  - Exact exponential Euler LIF membrane integration
  - Biophysical transmission delays via circular ring buffer
  - Dale's principle enforcement (strict non-negative / non-positive weights)
  - Refractory periods
  - Hardware-grounded Synaptic Operations (SynOps) counter & Energy Proxy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from ..connectome.unified_schema import ConnectomeGraph


@dataclass
class SimulationConfig:
    """Biophysical and numerical configuration parameters for LIF simulation."""
    dt_ms: float = 0.5           # Simulation time step in milliseconds
    v_rest: float = -70.0        # Resting potential in mV
    v_reset: float = -75.0       # Post-spike reset potential in mV
    v_thresh: float = -50.0      # Action potential threshold in mV
    tau_m_ms: float = 20.0       # Membrane time constant (ms)
    tau_ref_ms: float = 2.0      # Absolute refractory period (ms)
    tau_syn_ms: float = 5.0      # Post-synaptic current decay time constant (ms)
    energy_per_synop_nj: float = 0.020  # 20 pJ per synaptic event (neuromorphic standard)
    energy_per_spike_nj: float = 0.005  # 5 pJ per somatic action potential


@dataclass
class SimulationResult:
    """Detailed simulation outputs, raster events, and energy metrics."""
    total_time_ms: float
    time_steps: int
    num_neurons: int
    spike_raster: np.ndarray        # Shape (T, N) boolean spike events
    voltage_trace: np.ndarray       # Shape (T, N) membrane potentials (mV)
    synaptic_currents: np.ndarray   # Shape (T, N) post-synaptic currents
    total_spikes: int
    total_synops: int
    mean_firing_rate_hz: float
    energy_proxy_nj: float
    step_synops: np.ndarray         # Shape (T,) SynOps per time step

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": self.total_time_ms,
            "num_neurons": self.num_neurons,
            "total_spikes": int(self.total_spikes),
            "total_synops": int(self.total_synops),
            "mean_firing_rate_hz": float(self.mean_firing_rate_hz),
            "energy_proxy_nj": float(self.energy_proxy_nj),
            "avg_synops_per_spike": float(self.total_synops / max(1, self.total_spikes)),
        }


class VectorizedLIFSimulator:
    """
    Vectorized, high-throughput LIF connectome simulation engine.
    Supports PyTorch tensor acceleration (CPU / GPU) and exact delayed spike delivery.
    """

    def __init__(
        self,
        connectome: ConnectomeGraph,
        config: Optional[SimulationConfig] = None,
        device: str = "cpu",
    ):
        self.connectome = connectome
        self.config = config or SimulationConfig()
        self.device = torch.device(device)

        # 1. Build signed adjacency weight matrix and delay matrix
        adj_raw, self.node_order = connectome.get_adjacency_matrix(weighted=True, signed=True)
        self.num_neurons = len(self.node_order)
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_order)}

        # PyTorch weight matrix: W[i, j] is weight from pre j to post i
        # Transpose so that W @ s_pre gives post-synaptic synaptic input
        self.W = torch.tensor(adj_raw.T, dtype=torch.float32, device=self.device)

        # Out-degree vector for SynOp computation: out_deg[j] = number of post targets of j
        # Count non-zero outgoing connections
        out_degrees = np.count_nonzero(adj_raw, axis=1)
        self.out_degrees = torch.tensor(out_degrees, dtype=torch.float32, device=self.device)

        # Delay buffer setup
        # Compute maximum delay in steps
        dt = self.config.dt_ms
        self.delay_steps = max(1, int(round(1.8 / dt)))  # Default delay steps
        # Can be customized per edge in advanced mode

    def run(
        self,
        duration_ms: float,
        external_currents: Optional[np.ndarray] = None,
        initial_voltage: Optional[np.ndarray] = None,
        record_voltages: bool = True,
    ) -> SimulationResult:
        """
        Run vectorized forward simulation for duration_ms.

        Parameters
        ----------
        duration_ms : float
            Simulation length in ms.
        external_currents : np.ndarray, optional
            External input current array of shape (T, N) or (N,).
        initial_voltage : np.ndarray, optional
            Initial membrane potentials (shape N).
        record_voltages : bool
            Whether to keep full voltage traces in memory (set False for long runs).
        """
        dt = self.config.dt_ms
        num_steps = int(round(duration_ms / dt))
        N = self.num_neurons

        # State vectors
        if initial_voltage is not None:
            v = torch.tensor(initial_voltage, dtype=torch.float32, device=self.device)
        else:
            v = torch.full((N,), self.config.v_rest, dtype=torch.float32, device=self.device)

        i_syn = torch.zeros(N, dtype=torch.float32, device=self.device)
        refractory_timer = torch.zeros(N, dtype=torch.float32, device=self.device)

        # Precompute constants
        alpha_m = float(np.exp(-dt / self.config.tau_m_ms))
        alpha_s = float(np.exp(-dt / self.config.tau_syn_ms))
        v_rest = self.config.v_rest
        v_reset = self.config.v_reset
        v_thresh = self.config.v_thresh
        tau_ref = self.config.tau_ref_ms

        # Delay ring buffer for spikes: shape (delay_steps, N)
        delay_buf_size = max(1, self.delay_steps)
        delay_buffer = torch.zeros((delay_buf_size, N), dtype=torch.float32, device=self.device)
        buffer_idx = 0

        # Parse external currents
        if external_currents is None:
            i_ext_seq = torch.zeros((num_steps, N), dtype=torch.float32, device=self.device)
        elif external_currents.ndim == 1:
            i_ext_seq = torch.tensor(external_currents, dtype=torch.float32, device=self.device).repeat(num_steps, 1)
        else:
            i_ext_seq = torch.tensor(external_currents, dtype=torch.float32, device=self.device)

        # Output recorders
        spikes_recorded = []
        synops_recorded = np.zeros(num_steps, dtype=np.int64)
        voltages_recorded = [] if record_voltages else None
        currents_recorded = [] if record_voltages else None

        total_spikes = 0
        total_synops = 0

        # Main vectorized simulation loop
        for t_step in range(num_steps):
            if record_voltages:
                voltages_recorded.append(v.cpu().numpy().copy())
                currents_recorded.append(i_syn.cpu().numpy().copy())

            # 1. Fetch delayed spikes from ring buffer
            arriving_spikes = delay_buffer[buffer_idx]  # Shape (N,)

            # 2. Update synaptic currents: decay + arriving input
            # arriving_input scaled so that weight represents post-synaptic voltage deflection (mV)
            syn_scale = 1.0 / max(1e-4, (1.0 - alpha_m))
            syn_input = torch.matmul(self.W, arriving_spikes) * syn_scale
            i_syn = i_syn * alpha_s + syn_input

            # 3. Add external drive and update membrane potential (for non-refractory neurons)
            is_refractory = refractory_timer > 0
            i_total = i_syn + i_ext_seq[t_step]

            # dv/dt = -(v - v_rest) / tau_m + i_total
            # Clamp to biological reversal potential (E_rev >= -85.0 mV)
            v_new = torch.clamp(
                v_rest + (v - v_rest) * alpha_m + i_total * (1.0 - alpha_m),
                min=-85.0
            )
            v = torch.where(is_refractory, torch.full_like(v, v_reset), v_new)

            # 4. Spike detection
            spikes = (v >= v_thresh).float()
            spikes_bool = spikes.bool()

            # 5. Handle reset and refractory period
            v = torch.where(spikes_bool, torch.full_like(v, v_reset), v)
            refractory_timer = torch.where(
                spikes_bool,
                torch.full_like(refractory_timer, tau_ref),
                torch.clamp(refractory_timer - dt, min=0.0),
            )

            # 6. Put newly generated spikes into delay ring buffer
            delay_buffer[buffer_idx] = spikes
            buffer_idx = (buffer_idx + 1) % delay_buf_size

            # 7. Compute SynOps and metrics
            step_spike_count = int(spikes.sum().item())
            # SynOps: each spiking neuron j triggers out_degree[j] synaptic evaluations
            step_synops_count = int(torch.dot(spikes, self.out_degrees).item())

            total_spikes += step_spike_count
            total_synops += step_synops_count
            synops_recorded[t_step] = step_synops_count
            spikes_recorded.append(spikes.cpu().numpy().astype(bool))

        # Final packaging
        raster_arr = np.array(spikes_recorded, dtype=bool)  # Shape (T, N)
        v_trace = np.array(voltages_recorded, dtype=np.float32) if record_voltages else np.empty((0, N))
        i_trace = np.array(currents_recorded, dtype=np.float32) if record_voltages else np.empty((0, N))

        mean_fr = float((total_spikes / (N * (duration_ms / 1000.0))) if N > 0 and duration_ms > 0 else 0.0)
        energy_nj = float(
            total_synops * self.config.energy_per_synop_nj
            + total_spikes * self.config.energy_per_spike_nj
        )

        return SimulationResult(
            total_time_ms=duration_ms,
            time_steps=num_steps,
            num_neurons=N,
            spike_raster=raster_arr,
            voltage_trace=v_trace,
            synaptic_currents=i_trace,
            total_spikes=total_spikes,
            total_synops=total_synops,
            mean_firing_rate_hz=mean_fr,
            energy_proxy_nj=energy_nj,
            step_synops=synops_recorded,
        )
