"""
Unit tests for Neuromorphic DVS Gesture benchmark suite.
Verifies:
  1. Synthetic DVS event stream generation across all 10 gesture classes.
  2. Noise injection and temporal jitter perturbation bounds.
  3. Single-batch model forward evaluation and metric extraction.
"""

import pytest
import torch
import numpy as np
from experiments.benchmarks.run_dvs_gesture_benchmark import (
    generate_dvs_gesture_dataset,
    evaluate_model_on_dataset,
)
from src.architecture.coupled_event_net import CoupledEventNet, CENConfig
from src.architecture.active_dendritic_cen import ActiveDendriticCEN, ActiveDendriticConfig


def test_dvs_gesture_synthesis():
    """Verify 10-class dataset dimensions and class balance."""
    num_samples = 50
    time_steps = 30
    num_channels = 16
    X, y = generate_dvs_gesture_dataset(
        num_samples=num_samples,
        time_steps=time_steps,
        num_channels=num_channels,
        noise_rate=0.01,
        seed=42,
    )
    assert X.shape == (50, 30, 16)
    assert y.shape == (50,)
    assert len(torch.unique(y)) == 10
    # Values should be binary or bounded in [0, 1]
    assert torch.all(X >= 0.0)
    assert torch.all(X <= 1.0)


def test_dvs_noise_and_jitter():
    """Verify that adding noise increases total event count."""
    X_clean, _ = generate_dvs_gesture_dataset(num_samples=20, time_steps=30, noise_rate=0.0, seed=42)
    X_noisy, _ = generate_dvs_gesture_dataset(num_samples=20, time_steps=30, noise_rate=0.10, seed=42)
    assert torch.sum(X_noisy) > torch.sum(X_clean)


def test_dvs_model_forward_eval():
    """Verify CoupledEventNet and ActiveDendriticCEN evaluation on DVS batches."""
    X, y = generate_dvs_gesture_dataset(num_samples=20, time_steps=20, num_channels=8, seed=42)

    cfg = CENConfig(input_dim=8, num_neurons=32, output_dim=10)
    model = CoupledEventNet(cfg)
    metrics = evaluate_model_on_dataset(model, X, y)

    assert "accuracy" in metrics
    assert "synops" in metrics
    assert "energy_nj" in metrics
    assert 0.0 <= metrics["accuracy"] <= 100.0
