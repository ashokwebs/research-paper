"""
COMPUTATIONAL COUPLING — Standardized Architecture Baselines
Implements comparative baselines evaluated under identical conditions:
  1. Dense SNN: Unconstrained recurrent weights (violates Dale's principle, no motif structure)
  2. Connectome Reservoir (conn2res): Fixed biological connectome topology as reservoir
  3. NCP Spiking Baseline: 4-layer sensory -> inter -> command -> motor hierarchy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional
from .coupled_event_net import SurrogateHeaviside


class DenseSNN(nn.Module):
    """
    Dense Recurrent Spiking Neural Network baseline.
    Recurrent weights are unconstrained real values (allows Dale violations).
    """

    def __init__(self, input_dim: int, num_neurons: int, output_dim: int, dt_ms: float = 1.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.dt_ms = dt_ms

        self.w_in = nn.Linear(input_dim, num_neurons)
        self.w_rec = nn.Linear(num_neurons, num_neurons, bias=False)
        self.w_out = nn.Linear(num_neurons, output_dim)

        # Decay constants
        self.alpha_m = float(np.exp(-dt_ms / 20.0))
        self.alpha_s = float(np.exp(-dt_ms / 5.0))
        self.v_thresh = 1.0

        # Out-degree is all-to-all: every neuron connects to num_neurons - 1
        self.register_buffer("out_degs", torch.full((num_neurons,), num_neurons - 1, dtype=torch.float32))

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, time_steps, _ = x_seq.shape
        device = x_seq.device
        N = self.num_neurons

        v = torch.zeros((batch_size, N), device=device)
        i_syn = torch.zeros((batch_size, N), device=device)
        readout_acc = torch.zeros((batch_size, self.w_out.out_features), device=device)

        total_spikes = 0
        total_synops = 0

        for t in range(time_steps):
            x_t = x_seq[:, t, :]
            i_in = self.w_in(x_t)

            if t == 0:
                s_prev = torch.zeros((batch_size, N), device=device)
            i_rec = self.w_rec(s_prev)

            i_syn = i_syn * self.alpha_s + (i_in + i_rec)
            v_new = v * self.alpha_m + i_syn * (1.0 - self.alpha_m)

            s = SurrogateHeaviside.apply(v_new, self.v_thresh)
            v = v_new * (1.0 - s)

            out_t = self.w_out(v)
            readout_acc = readout_acc * 0.95 + out_t

            step_spikes = int(s.sum().item())
            step_synops = int(torch.sum(s @ self.out_degs).item())

            total_spikes += step_spikes
            total_synops += step_synops
            s_prev = s

        stats = {
            "total_spikes": total_spikes,
            "total_synops": total_synops,
            "avg_synops_per_sample": float(total_synops / batch_size),
            "energy_proxy_nj": float(total_synops * 0.020 + total_spikes * 0.005),
        }
        return readout_acc, stats


class ConnectomeReservoirNet(nn.Module):
    """
    Connectome Reservoir baseline (Suárez, Misic et al. Nature Communications 2024 / conn2res).
    Uses actual biological connectome adjacency matrix as a frozen, fixed recurrent reservoir.
    Only the output readout layer is trained.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        input_dim: int,
        output_dim: int,
        spectral_radius: float = 0.95,
        dt_ms: float = 1.0,
    ):
        super().__init__()
        N = adjacency_matrix.shape[0]
        self.num_neurons = N
        self.dt_ms = dt_ms

        # Normalize spectral radius
        adj = adjacency_matrix.copy().astype(np.float32)
        # Compute eigenvalues
        try:
            eigs = np.linalg.eigvals(adj)
            max_eig = np.max(np.abs(eigs))
            if max_eig > 1e-6:
                adj = (adj / max_eig) * spectral_radius
        except Exception:
            adj = adj / (np.max(np.abs(adj)) + 1e-6) * spectral_radius

        # Out-degree vector
        out_degs = np.count_nonzero(adjacency_matrix, axis=1)
        self.register_buffer("out_degs", torch.tensor(out_degs, dtype=torch.float32))

        # Frozen recurrent weight buffer
        self.register_buffer("W_rec", torch.tensor(adj.T, dtype=torch.float32))

        # Fixed random input projection
        w_in_init = np.random.uniform(-0.5, 0.5, size=(N, input_dim)).astype(np.float32)
        self.register_buffer("W_in", torch.tensor(w_in_init, dtype=torch.float32))

        # Trainable readout layer
        self.w_out = nn.Linear(N, output_dim)

        self.alpha_m = float(np.exp(-dt_ms / 20.0))
        self.alpha_s = float(np.exp(-dt_ms / 5.0))
        self.v_thresh = 1.0

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, time_steps, _ = x_seq.shape
        device = x_seq.device
        N = self.num_neurons

        v = torch.zeros((batch_size, N), device=device)
        i_syn = torch.zeros((batch_size, N), device=device)
        readout_acc = torch.zeros((batch_size, self.w_out.out_features), device=device)

        total_spikes = 0
        total_synops = 0

        for t in range(time_steps):
            x_t = x_seq[:, t, :]
            i_in = F.linear(x_t, self.W_in)

            if t == 0:
                s_prev = torch.zeros((batch_size, N), device=device)
            i_rec = F.linear(s_prev, self.W_rec)

            i_syn = i_syn * self.alpha_s + (i_in + i_rec)
            v_new = v * self.alpha_m + i_syn * (1.0 - self.alpha_m)

            s = SurrogateHeaviside.apply(v_new, self.v_thresh)
            v = v_new * (1.0 - s)

            out_t = self.w_out(v)
            readout_acc = readout_acc * 0.95 + out_t

            step_spikes = int(s.sum().item())
            step_synops = int(torch.sum(s @ self.out_degs).item())

            total_spikes += step_spikes
            total_synops += step_synops
            s_prev = s

        stats = {
            "total_spikes": total_spikes,
            "total_synops": total_synops,
            "avg_synops_per_sample": float(total_synops / batch_size),
            "energy_proxy_nj": float(total_synops * 0.020 + total_spikes * 0.005),
        }
        return readout_acc, stats
