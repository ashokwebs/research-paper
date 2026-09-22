"""
COMPUTATIONAL COUPLING — Active Dendritic CoupledEventNet (AD-CEN)
An end-to-end differentiable, two-compartment spiking neural network that couples
conserved biological microcircuit motifs with active non-linear dendritic dynamics:
  - Dendritic Compartment: High-impedance passive attenuation + regenerative NMDA plateaus
  - Axial Conductance: Forward electrotonic coupling (V_d -> V_s)
  - Somatic Compartment: Perisomatic shunting inhibition and surrogate-gradient action potentials
  - Motif Routing: Coherent FFL, Incoherent FFL, and Reciprocal Inhibitory Pairs
Solves the long-horizon working memory retention barrier while maintaining ultra-sparse SynOps.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architecture.coupled_event_net import SurrogateHeaviside, CENConfig


@dataclass
class ActiveDendriticConfig:
    """Hyperparameters for Two-Compartment Active Dendritic CoupledEventNet."""
    input_dim: int = 10
    num_neurons: int = 128
    output_dim: int = 2
    ei_ratio: float = 4.0          # ~80% Excitatory, ~20% Inhibitory
    density: float = 0.10          # Sparse recurrent connectivity density
    cffl_fraction: float = 0.35    # Coherent Feedforward Loops
    iffl_fraction: float = 0.25    # Incoherent Feedforward Loops
    rip_fraction: float = 0.15     # Reciprocal Inhibitory Pairs
    dt_ms: float = 1.0             # Timestep (ms)
    tau_soma_ms: float = 20.0      # Somatic membrane time constant (ms)
    tau_dend_ms: float = 40.0      # Dendritic passive time constant (ms)
    tau_plateau_ms: float = 150.0  # NMDA plateau active decay time constant (ms)
    tau_syn_ms: float = 5.0        # Synaptic conductance decay (ms)
    theta_dend: float = 0.45       # Dendritic NMDA activation threshold (normalized)
    g_nmda: float = 0.35           # Regenerative NMDA plateau conductance
    v_plateau: float = 0.85        # NMDA plateau target voltage
    g_axial: float = 0.25          # Forward axial conductance (dendrite -> soma)
    v_thresh_soma: float = 1.0     # Somatic action potential threshold
    v_reset_soma: float = 0.0      # Somatic reset potential
    v_rest: float = 0.0            # Baseline resting potential
    shunting_efficiency: float = 0.95  # Perisomatic inhibitory shunting clamp factor


class ActiveDendriticCEN(nn.Module):
    """
    Two-Compartment Active Dendritic CoupledEventNet (AD-CEN).
    Combines sparse biological motif topology with active dendritic plateaus for
    robust working memory retention across temporal delays exceeding membrane time constants.
    """

    def __init__(self, config: ActiveDendriticConfig):
        super().__init__()
        self.cfg = config
        N = config.num_neurons
        D_in = config.input_dim
        D_out = config.output_dim

        # 1. Partition into Excitatory and Inhibitory subpopulations
        n_exc = int(round(N * (config.ei_ratio / (1.0 + config.ei_ratio))))
        n_inh = N - n_exc
        self.n_exc = n_exc
        self.n_inh = n_inh

        # Valence signs: +1 for E, -1 for I
        node_signs = np.ones(N, dtype=np.float32)
        node_signs[n_exc:] = -1.0
        self.register_buffer("node_signs", torch.tensor(node_signs, dtype=torch.float32))

        # Dale sign matrix (N, N): pre-synaptic j determines sign
        dale_matrix = np.tile(node_signs, (N, 1))
        self.register_buffer("dale_matrix", torch.tensor(dale_matrix, dtype=torch.float32))

        # 2. Build motif-structured recurrent adjacency mask M_conn
        cen_cfg = CENConfig(
            input_dim=D_in,
            num_neurons=N,
            output_dim=D_out,
            ei_ratio=config.ei_ratio,
            density=config.density,
            cffl_fraction=config.cffl_fraction,
            iffl_fraction=config.iffl_fraction,
            rip_fraction=config.rip_fraction,
        )
        # Reuse robust motif builder
        from src.architecture.coupled_event_net import CoupledEventNet
        temp_cen = CoupledEventNet(cen_cfg)
        self.register_buffer("adj_mask", temp_cen.adj_mask.clone())
        self.register_buffer("out_degs", temp_cen.out_degs.clone())

        # 3. Separate Exc and Inh projections
        # Excitatory recurrent connections target dendritic compartments
        # Inhibitory recurrent connections target both dendrites (distal) and soma (perisomatic shunting)
        is_exc_pre = (self.node_signs > 0).float()  # Shape (N,)
        is_inh_pre = (self.node_signs < 0).float()  # Shape (N,)
        self.register_buffer("is_exc_pre", is_exc_pre)
        self.register_buffer("is_inh_pre", is_inh_pre)

        # Learnable parameters
        self.w_in = nn.Parameter(torch.empty(N, D_in))
        nn.init.kaiming_uniform_(self.w_in, a=np.sqrt(5))

        self.w_rec_mag = nn.Parameter(torch.empty(N, N))
        nn.init.uniform_(self.w_rec_mag, 0.05, 0.25)

        # Output linear readout from somatic spikes / states
        self.w_out = nn.Linear(N, D_out)

        # Continuous decay factors
        self.alpha_soma = float(np.exp(-config.dt_ms / config.tau_soma_ms))
        self.alpha_dend = float(np.exp(-config.dt_ms / config.tau_dend_ms))
        self.alpha_plateau = float(np.exp(-config.dt_ms / config.tau_plateau_ms))
        self.alpha_syn = float(np.exp(-config.dt_ms / config.tau_syn_ms))

    def get_effective_recurrent_weights(self) -> torch.Tensor:
        """Dale-compliant recurrent connectivity matrix."""
        w_mag = F.relu(self.w_rec_mag)
        return w_mag * self.adj_mask * self.dale_matrix

    def forward(
        self,
        x_seq: torch.Tensor,
        return_dynamics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Two-compartment forward pass over temporal sequence.

        Parameters
        ----------
        x_seq : torch.Tensor
            Shape (Batch, Time, D_in).
        return_dynamics : bool
            If True, logs temporal voltage traces and spike rasters.

        Returns
        -------
        logits : torch.Tensor
            Output logits of shape (Batch, D_out).
        stats : dict
            Diagnostic metrics (SynOps, somatic spikes, dendritic plateau activations).
        """
        batch_size, time_steps, _ = x_seq.shape
        N = self.cfg.num_neurons
        device = x_seq.device

        W_rec = self.get_effective_recurrent_weights()  # Shape (N, N)

        # Split W_rec into excitatory (to dendrite) and inhibitory (to soma & dendrite)
        # Pre-synaptic excitatory: W_rec * is_exc_pre (along cols)
        W_rec_exc = W_rec * self.is_exc_pre.unsqueeze(0)  # Shape (N, N)
        W_rec_inh = -W_rec * self.is_inh_pre.unsqueeze(0) # Non-negative inhibitory magnitudes

        # State initialization
        v_dend = torch.full((batch_size, N), self.cfg.v_rest, device=device)
        v_soma = torch.full((batch_size, N), self.cfg.v_rest, device=device)
        i_syn_dend = torch.zeros((batch_size, N), device=device)
        i_syn_soma_inh = torch.zeros((batch_size, N), device=device)
        s_soma_prev = torch.zeros((batch_size, N), device=device)

        readout_acc = torch.zeros((batch_size, self.cfg.output_dim), device=device)
        total_soma_spikes = 0
        total_plateau_steps = 0
        total_synops = 0

        v_dend_hist = [] if return_dynamics else None
        v_soma_hist = [] if return_dynamics else None
        soma_spikes_hist = [] if return_dynamics else None

        surrogate_fn = SurrogateHeaviside.apply

        for t in range(time_steps):
            x_t = x_seq[:, t, :]  # Shape (Batch, D_in)

            # 1. Incoming synaptic currents
            i_in = F.linear(x_t, self.w_in)  # Targets dendritic arbor
            i_rec_exc = F.linear(s_soma_prev, W_rec_exc)  # Excitatory to dendrite
            i_rec_inh = F.linear(s_soma_prev, W_rec_inh)  # Inhibitory to perisomatic region

            i_syn_dend = i_syn_dend * self.alpha_syn + (i_in + i_rec_exc)
            i_syn_soma_inh = i_syn_soma_inh * self.alpha_syn + i_rec_inh

            # 2. Dendritic Compartment Dynamics with Regenerative NMDA Plateaus
            # Sigmoidal activation for NMDA voltage-dependent magnesium unblock
            nmda_gate = torch.sigmoid((v_dend - self.cfg.theta_dend) * 12.0)
            
            # Active plateau regenerative current pulls voltage towards v_plateau
            i_nmda = self.cfg.g_nmda * nmda_gate * (self.cfg.v_plateau - v_dend)

            # Plateau decay is much slower (tau_plateau = 150ms) than passive decay (tau_dend = 40ms)
            effective_alpha_d = self.alpha_dend * (1.0 - nmda_gate) + self.alpha_plateau * nmda_gate

            v_dend = effective_alpha_d * v_dend + (1.0 - effective_alpha_d) * self.cfg.v_rest + i_syn_dend + i_nmda

            # 3. Forward Axial Conductance to Soma
            i_axial = self.cfg.g_axial * (v_dend - v_soma)

            # 4. Somatic Compartment Dynamics with Perisomatic Shunting Inhibition
            # Shunting factor scales down somatic voltage towards resting potential
            shunt = torch.clamp(self.cfg.shunting_efficiency * torch.tanh(i_syn_soma_inh * 2.0), 0.0, 0.99)

            # Somatic voltage update with axial input and shunting
            v_soma_decayed = self.alpha_soma * v_soma * (1.0 - shunt) + i_axial * (1.0 - shunt)
            
            # Reset after previous somatic spike
            v_soma = v_soma_decayed - (self.cfg.v_thresh_soma - self.cfg.v_reset_soma) * s_soma_prev

            # 5. Somatic Action Potential Generation
            s_soma = surrogate_fn(v_soma, self.cfg.v_thresh_soma)
            s_soma_prev = s_soma

            # Accumulate readout
            readout_acc = readout_acc + self.w_out(s_soma)

            # Metrics
            n_spikes_t = s_soma.sum()
            total_soma_spikes = total_soma_spikes + n_spikes_t
            total_plateau_steps = total_plateau_steps + (nmda_gate > 0.5).float().sum()
            total_synops = total_synops + (s_soma @ self.out_degs).sum()

            if return_dynamics:
                v_dend_hist.append(v_dend.detach().cpu())
                v_soma_hist.append(v_soma.detach().cpu())
                soma_spikes_hist.append(s_soma.detach().cpu())

        # Average readout over time
        logits = readout_acc / float(time_steps)

        n_synops_val = float(total_synops.item()) if isinstance(total_synops, torch.Tensor) else float(total_synops)
        n_spikes_val = float(total_soma_spikes.item()) if isinstance(total_soma_spikes, torch.Tensor) else float(total_soma_spikes)
        avg_synops = n_synops_val / float(batch_size)
        mean_fr = float(n_spikes_val / (batch_size * N * (time_steps * self.cfg.dt_ms / 1000.0)))
        energy_proxy_nj = float(n_synops_val * 0.020 + n_spikes_val * 0.005)

        stats = {
            "total_synops": n_synops_val,
            "total_spikes": n_spikes_val,
            "total_soma_spikes": n_spikes_val,
            "total_plateau_activations": float(total_plateau_steps.item()) if isinstance(total_plateau_steps, torch.Tensor) else float(total_plateau_steps),
            "synops_per_sample": avg_synops,
            "avg_synops_per_sample": avg_synops,
            "spikes_per_sample": n_spikes_val / float(batch_size),
            "mean_firing_rate_hz": mean_fr,
            "energy_proxy_nj": energy_proxy_nj,
        }

        if return_dynamics:
            stats["v_dend"] = torch.stack(v_dend_hist, dim=1)
            stats["v_soma"] = torch.stack(v_soma_hist, dim=1)
            stats["spikes"] = torch.stack(soma_spikes_hist, dim=1)

        return logits, stats
