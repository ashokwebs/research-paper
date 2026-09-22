"""
Unit tests for CoupledEventNet PyTorch module and baselines.
"""

import pytest
import torch
from src.architecture.coupled_event_net import CoupledEventNet, CENConfig
from src.architecture.baselines import DenseSNN


def test_cen_forward_and_dale_sign_matrix():
    cfg = CENConfig(input_dim=8, num_neurons=64, output_dim=3, density=0.15)
    model = CoupledEventNet(cfg)

    # Check that effective recurrent weights strictly satisfy Dale's principle
    W_eff = model.get_effective_recurrent_weights()
    assert W_eff.shape == (64, 64)

    # Column j corresponds to pre-synaptic neuron j
    for j in range(64):
        col = W_eff[:, j]
        sign_j = model.node_signs[j].item()
        if sign_j > 0:
            assert torch.all(col >= 0.0)
        else:
            assert torch.all(col <= 0.0)

    # Test forward pass
    x = torch.randn(4, 20, 8)  # Batch 4, Time 20, Dim 8
    logits, stats = model(x)
    assert logits.shape == (4, 3)
    assert stats["total_synops"] >= 0
    assert stats["avg_synops_per_sample"] >= 0.0


def test_cen_backward_gradient_flow():
    cfg = CENConfig(input_dim=4, num_neurons=32, output_dim=2)
    model = CoupledEventNet(cfg)

    x = torch.randn(2, 10, 4)
    y = torch.tensor([0, 1])

    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    # Verify gradients flow into recurrent weight magnitudes and input weights
    assert model.w_in.grad is not None
    assert model.w_rec_mag.grad is not None
    assert not torch.all(model.w_in.grad == 0.0)
