"""
COMPUTATIONAL COUPLING — Long-Horizon Delayed-Match-to-Sample with Distractors (DMTS-D)
Evaluates temporal working memory retention and distractor rejection across long horizons (200 time steps / 100-200 ms).
Tests:
  1. CoupledEventNet (Full, motif-routed)
  2. Dense SNN (fully connected recurrent LIF)
  3. Connectome Reservoir (conn2res / fixed C. elegans connectome)
  4. CEN w/o RIP (ablating mutual inhibition attractor memory)
  5. CEN w/o cFFL (ablating feedforward noise filter)
"""

import os
import sys
import json
import time
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connectome import load_celegans_connectome
from src.architecture.coupled_event_net import CoupledEventNet, CENConfig
from src.architecture.baselines import DenseSNN, ConnectomeReservoirNet


def generate_dmts_dataset(
    num_samples: int = 400,
    time_steps: int = 150,
    num_channels: int = 8,
    noise_prob: float = 0.04,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Delayed-Match-to-Sample with Distractors (DMTS-D):
      - Cue Period (t = 10..25): Cue A (Ch 0) or Cue B (Ch 1)
      - Delay / Distractor Period (t = 25..120, 95 steps): Continuous Poisson distractor bursts on Ch 4..7
      - Probe Period (t = 120..135): Probe A (Ch 0) or Probe B (Ch 1)
      - Target: 1 if Cue == Probe (Match), 0 if Cue != Probe (Non-Match)
    """
    np.random.seed(seed)
    X = np.random.binomial(1, noise_prob, size=(num_samples, time_steps, num_channels)).astype(np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        cue_is_A = np.random.rand() > 0.5
        probe_is_A = np.random.rand() > 0.5

        # Cue presentation (t = 10..25)
        cue_ch = 0 if cue_is_A else 1
        X[i, 10:25, cue_ch] = 1.0

        # Strong distractor bursts during delay (t = 40..60 and t = 80..100 on distractor channels 4..7)
        X[i, 40:60, 4:6] = np.random.binomial(1, 0.5, size=(20, 2))
        X[i, 80:100, 6:8] = np.random.binomial(1, 0.5, size=(20, 2))

        # Probe presentation (t = 120..135)
        probe_ch = 0 if probe_is_A else 1
        X[i, 120:135, probe_ch] = 1.0

        is_match = 1 if (cue_is_A == probe_is_A) else 0
        y[i] = is_match

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    """Evaluate accuracy (%), average SynOps per sample, and energy proxy (nJ)."""
    model.eval()
    with torch.no_grad():
        logits, stats = model(X)
        preds = torch.argmax(logits, dim=1)
        acc = float((preds == y).float().mean().item()) * 100.0
        avg_synops = stats["avg_synops_per_sample"]
        energy_nj = stats["energy_proxy_nj"]
    return acc, avg_synops, energy_nj


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.008,
    batch_size: int = 32,
) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    N = X_train.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for b in range(0, N, batch_size):
            idx = perm[b:b + batch_size]
            bx, by = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    return model


def run_long_horizon_benchmark():
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Long-Horizon Delayed-Match-to-Sample (DMTS-D) Benchmark")
    print("=" * 80)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    g_ce = load_celegans_connectome()
    adj_ce, _ = g_ce.get_adjacency_matrix(weighted=True, signed=True)

    # Generate train and test sets (150 time steps = 75 ms horizon at dt=0.5)
    print("\n[+] Generating Long-Horizon DMTS-D Dataset (T=150 steps, 95-step distractor delay)...")
    X_train, y_train = generate_dmts_dataset(num_samples=500, time_steps=150, num_channels=8, seed=42)
    X_test, y_test = generate_dmts_dataset(num_samples=300, time_steps=150, num_channels=8, seed=123)

    models_config = {
        "CoupledEventNet (Full)": CoupledEventNet(
            CENConfig(
                input_dim=8,
                num_neurons=64,
                output_dim=2,
                cffl_fraction=0.35,
                iffl_fraction=0.25,
                rip_fraction=0.20,
            )
        ),
        "Dense SNN": DenseSNN(
            input_dim=8,
            num_neurons=64,
            output_dim=2,
        ),
        "Connectome Reservoir": ConnectomeReservoirNet(
            adjacency_matrix=adj_ce,
            input_dim=8,
            output_dim=2,
            spectral_radius=0.90,
        ),
        "CEN w/o RIP": CoupledEventNet(
            CENConfig(
                input_dim=8,
                num_neurons=64,
                output_dim=2,
                cffl_fraction=0.35,
                iffl_fraction=0.25,
                rip_fraction=0.0,
            )
        ),
        "CEN w/o cFFL": CoupledEventNet(
            CENConfig(
                input_dim=8,
                num_neurons=64,
                output_dim=2,
                cffl_fraction=0.0,
                iffl_fraction=0.25,
                rip_fraction=0.20,
            )
        ),
    }

    benchmark_results = []
    print("\n[*] Training and Evaluating Models across 3 Independent Seeds...")

    for name, model_init in models_config.items():
        seed_accs = []
        seed_synops = []
        seed_energies = []

        for seed in [42, 101, 777]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Rebuild model for each seed
            if name == "CoupledEventNet (Full)":
                model = CoupledEventNet(CENConfig(input_dim=8, num_neurons=64, output_dim=2, cffl_fraction=0.35, iffl_fraction=0.25, rip_fraction=0.20))
            elif name == "Dense SNN":
                model = DenseSNN(input_dim=8, num_neurons=64, output_dim=2)
            elif name == "Connectome Reservoir":
                model = ConnectomeReservoirNet(adjacency_matrix=adj_ce, input_dim=8, output_dim=2, spectral_radius=0.90)
            elif name == "CEN w/o RIP":
                model = CoupledEventNet(CENConfig(input_dim=8, num_neurons=64, output_dim=2, cffl_fraction=0.35, iffl_fraction=0.25, rip_fraction=0.0))
            elif name == "CEN w/o cFFL":
                model = CoupledEventNet(CENConfig(input_dim=8, num_neurons=64, output_dim=2, cffl_fraction=0.0, iffl_fraction=0.25, rip_fraction=0.20))

            trained = train_model(model, X_train, y_train, epochs=25, lr=0.01)
            acc, synops, energy = evaluate_model(trained, X_test, y_test)
            seed_accs.append(acc)
            seed_synops.append(synops)
            seed_energies.append(energy)

        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs))
        mean_synops = float(np.mean(seed_synops))
        mean_energy = float(np.mean(seed_energies))

        res = {
            "model": name,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_synops_per_sample": mean_synops,
            "mean_energy_nj": mean_energy,
        }
        benchmark_results.append(res)
        print(f"  {name:<25}: Acc = {mean_acc:6.2f} +/- {std_acc:4.2f}% | SynOps = {mean_synops:8.1f} | Energy = {mean_energy:8.2f} nJ")

    # Save to disk
    df = pd.DataFrame(benchmark_results)
    csv_path = os.path.join(results_dir, "long_horizon_dmts_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved DMTS benchmark CSV to {csv_path}")

    json_path = os.path.join(results_dir, "long_horizon_dmts_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"[+] Saved DMTS benchmark JSON to {json_path}")

    print("\n" + "=" * 80)
    print("LONG-HORIZON DMTS BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_long_horizon_benchmark()
