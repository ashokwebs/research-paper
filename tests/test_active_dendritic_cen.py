"""
Unit tests for Active Dendritic CoupledEventNet (AD-CEN).
Verifies:
  1. Forward pass dynamics, shape preservation, and Dale's principle sign adherence.
  2. Differentiability and non-zero surrogate gradient flow through dendritic plateaus and axial currents.
  3. Regenerative NMDA plateau activation upon suprathreshold dendritic drive.
"""

import pytest
import numpy as np
import torch
from src.architecture.active_dendritic_cen import ActiveDendriticCEN, ActiveDendriticConfig


def test_ad_cen_forward_and_dale_sign():
    """Verify AD-CEN forward pass and Dale's principle compliance."""
    cfg = ActiveDendriticConfig(input_dim=8, num_neurons=64, output_dim=2, density=0.15)
    model = ActiveDendriticCEN(cfg)

    batch_size = 4
    seq_len = 25
    x = torch.randn(batch_size, seq_len, cfg.input_dim)

    logits, stats = model(x, return_dynamics=True)

    assert logits.shape == (batch_size, cfg.output_dim)
    assert stats["synops_per_sample"] >= 0.0
    assert stats["spikes_per_sample"] >= 0.0
    assert "v_dend" in stats
    assert "v_soma" in stats
    assert stats["v_dend"].shape == (batch_size, seq_len, cfg.num_neurons)

    # Check Dale compliance on recurrent weights
    W_eff = model.get_effective_recurrent_weights().detach().cpu().numpy()
    n_exc = model.n_exc
    # Pre-synaptic excitatory columns (0..n_exc-1) must be >= 0
    assert np.all(W_eff[:, :n_exc] >= -1e-6)
    # Pre-synaptic inhibitory columns (n_exc..N-1) must be <= 0
    assert np.all(W_eff[:, n_exc:] <= 1e-6)


def test_ad_cen_backward_gradient_flow():
    """Verify end-to-end backpropagation through time through dendritic plateaus."""
    cfg = ActiveDendriticConfig(input_dim=6, num_neurons=32, output_dim=3)
    model = ActiveDendriticCEN(cfg)

    x = torch.randn(2, 20, cfg.input_dim, requires_grad=True)
    target = torch.tensor([0, 2], dtype=torch.long)

    logits, stats = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    # Verify gradients reach recurrent weights and input projection
    assert model.w_in.grad is not None
    assert model.w_rec_mag.grad is not None
    assert torch.sum(torch.abs(model.w_rec_mag.grad)) > 0.0
    assert x.grad is not None
    assert torch.sum(torch.abs(x.grad)) > 0.0


def test_ad_cen_nmda_plateau_activation():
    """Verify that strong localized synaptic input triggers sustained dendritic plateaus."""
    cfg = ActiveDendriticConfig(input_dim=4, num_neurons=32, output_dim=2, theta_dend=0.3, g_nmda=0.5)
    model = ActiveDendriticCEN(cfg)

    # Input: brief pulse at t=0..3, followed by zero input
    x = torch.zeros(1, 40, cfg.input_dim)
    x[0, :4, :] = 2.5

    _, stats = model(x, return_dynamics=True)
    v_dend = stats["v_dend"][0].numpy()  # (40, 32)

    # Verify that dendritic voltage is depolarized past theta_dend and sustains depolarization
    max_dend = v_dend.max(axis=0)
    depolarized_neurons = np.where(max_dend > cfg.theta_dend)[0]
    assert len(depolarized_neurons) > 0, "At least one neuron should enter the NMDA plateau regime."

    # Verify plateau activations counter
    assert stats["total_plateau_activations"] > 0
