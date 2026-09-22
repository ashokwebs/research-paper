"""
COMPUTATIONAL COUPLING — Dynamic Stress, Jitter, and Channel Dropout Benchmark
Rigorously evaluates noise filtering, temporal resilience, and energy scaling across:
  1. CoupledEventNet (CEN, full motif-routed)
  2. Dense SNN (fully connected recurrent LIF)
  3. Connectome Reservoir (conn2res, fixed biological connectome)
  4. Ablated CEN (without cFFL coincidence detection)
  5. Ablated CEN (without RIP mutual inhibition)

Stress conditions evaluated:
  - Temporal Spike Jitter: sigma_t in [0, 2, 4, 8, 12, 16] ms
  - Background Poisson Distractor Rate: p_noise in [0.02, 0.05, 0.10, 0.15, 0.20]
  - Synaptic Transmission Dropout: p_fail in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
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


def generate_synthetic_dataset(
    num_samples: int = 400,
    time_steps: int = 50,
    num_channels: int = 10,
    num_classes: int = 4,
    noise_prob: float = 0.05,
    jitter_sigma: float = 0.0,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate temporal pattern discrimination dataset with controllable jitter and noise."""
    np.random.seed(seed)
    X = np.random.binomial(1, noise_prob, size=(num_samples, time_steps, num_channels)).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))

    for i in range(num_samples):
        cls = y[i]
        # Calculate jitter offset (integer time steps)
        jitter = int(round(np.random.normal(0, jitter_sigma))) if jitter_sigma > 0 else 0

        if cls == 0:
            # Synchronous coincidence across ch 0, 1, 2
            t_start = max(0, min(time_steps - 3, 15 + jitter))
            X[i, t_start:t_start + 3, 0:3] = 1.0
        elif cls == 1:
            # Sequential cascade
            t1 = max(0, min(time_steps - 3, 10 + jitter))
            t2 = max(0, min(time_steps - 3, 20 + jitter))
            t3 = max(0, min(time_steps - 3, 30 + jitter))
            X[i, t1:t1 + 3, 3] = 1.0
            X[i, t2:t2 + 3, 4] = 1.0
            X[i, t3:t3 + 3, 5] = 1.0
        elif cls == 2:
            # High-frequency transient burst
            t_start = max(0, min(time_steps - 10, 5 + jitter))
            X[i, t_start:t_start + 10, 6:8] = np.random.binomial(1, 0.8, size=(10, 2))
        elif cls == 3:
            # Sustained low-frequency rate
            t_start = max(0, min(time_steps - 30, 10 + jitter))
            X[i, t_start:t_start + 30, 8:10] = np.random.binomial(1, 0.25, size=(30, 2))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def evaluate_under_dropout(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    dropout_p: float = 0.0,
) -> Tuple[float, float, float]:
    """Evaluate accuracy, avg synops, and energy proxy under optional channel dropout."""
    model.eval()
    with torch.no_grad():
        if dropout_p > 0.0:
            # Mask input spikes with channel dropout
            drop_mask = torch.bernoulli(torch.full_like(X, 1.0 - dropout_p))
            X_eval = X * drop_mask
        else:
            X_eval = X

        logits, stats = model(X_eval)
        preds = torch.argmax(logits, dim=1)
        acc = float((preds == y).float().mean().item()) * 100.0
        avg_synops = stats["avg_synops_per_sample"]
        energy_nj = stats["energy_proxy_nj"]
    return acc, avg_synops, energy_nj


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 25,
    lr: float = 0.01,
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


def build_models(g_ce) -> Dict[str, nn.Module]:
    """Instantiate the 5 comparison models."""
    adj_ce, _ = g_ce.get_adjacency_matrix(weighted=True, signed=True)
    models = {
        "CoupledEventNet (Full)": CoupledEventNet(
            CENConfig(
                input_dim=10,
                num_neurons=64,
                output_dim=4,
                cffl_fraction=0.35,
                iffl_fraction=0.25,
                rip_fraction=0.15,
            )
        ),
        "Dense SNN": DenseSNN(
            input_dim=10,
            num_neurons=64,
            output_dim=4,
        ),
        "Connectome Reservoir": ConnectomeReservoirNet(
            adjacency_matrix=adj_ce,
            input_dim=10,
            output_dim=4,
            spectral_radius=0.90,
        ),
        "CEN w/o cFFL": CoupledEventNet(
            CENConfig(
                input_dim=10,
                num_neurons=64,
                output_dim=4,
                cffl_fraction=0.0,
                iffl_fraction=0.25,
                rip_fraction=0.15,
            )
        ),
        "CEN w/o RIP": CoupledEventNet(
            CENConfig(
                input_dim=10,
                num_neurons=64,
                output_dim=4,
                cffl_fraction=0.35,
                iffl_fraction=0.25,
                rip_fraction=0.0,
            )
        ),
    }
    return models


def run_stress_benchmark():
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Jitter, Noise, and Channel Dropout Stress Test")
    print("=" * 80)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    g_ce = load_celegans_connectome()

    # Training data (clean baseline)
    X_train, y_train = generate_synthetic_dataset(num_samples=400, noise_prob=0.03, jitter_sigma=0.0, seed=42)
    # Validation data (for sanity check)
    X_val, y_val = generate_synthetic_dataset(num_samples=200, noise_prob=0.03, jitter_sigma=0.0, seed=101)

    # Train all models on clean data
    print("\n[*] Training 5 architecture variants on baseline temporal dataset...")
    trained_models = {}
    for name, model in build_models(g_ce).items():
        t0 = time.time()
        print(f"    Training {name}...")
        trained = train_model(model, X_train, y_train, epochs=25, lr=0.01)
        val_acc, _, _ = evaluate_under_dropout(trained, X_val, y_val)
        print(f"      Validation Accuracy: {val_acc:.2f}% ({time.time()-t0:.2f}s)")
        trained_models[name] = trained

    # --- Sweep 1: Temporal Spike Jitter ---
    jitter_values = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    print("\n[*] Running Sweep 1: Temporal Spike Jitter (sigma_t = 0..16 ms)...")
    jitter_records = []

    for jitter in jitter_values:
        X_test, y_test = generate_synthetic_dataset(
            num_samples=250, noise_prob=0.03, jitter_sigma=jitter, seed=202
        )
        for name, model in trained_models.items():
            acc, synops, energy = evaluate_under_dropout(model, X_test, y_test)
            jitter_records.append({
                "model": name,
                "jitter_sigma_ms": jitter,
                "accuracy": acc,
                "synops_per_sample": synops,
                "energy_nj": energy,
            })

    # --- Sweep 2: Background Poisson Noise ---
    noise_values = [0.01, 0.05, 0.10, 0.15, 0.20]
    print("\n[*] Running Sweep 2: Background Poisson Distractor Noise (p = 0.01..0.20)...")
    noise_records = []

    for noise_p in noise_values:
        X_test, y_test = generate_synthetic_dataset(
            num_samples=250, noise_prob=noise_p, jitter_sigma=0.0, seed=303
        )
        for name, model in trained_models.items():
            acc, synops, energy = evaluate_under_dropout(model, X_test, y_test)
            noise_records.append({
                "model": name,
                "noise_prob": noise_p,
                "accuracy": acc,
                "synops_per_sample": synops,
                "energy_nj": energy,
            })

    # --- Sweep 3: Channel Dropout / Synaptic Failure ---
    dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("\n[*] Running Sweep 3: Synaptic Transmission Dropout (p_fail = 0.0..0.5)...")
    dropout_records = []

    X_test_clean, y_test_clean = generate_synthetic_dataset(
        num_samples=250, noise_prob=0.03, jitter_sigma=0.0, seed=404
    )
    for drop_p in dropout_values:
        for name, model in trained_models.items():
            acc, synops, energy = evaluate_under_dropout(model, X_test_clean, y_test_clean, dropout_p=drop_p)
            dropout_records.append({
                "model": name,
                "dropout_prob": drop_p,
                "accuracy": acc,
                "synops_per_sample": synops,
                "energy_nj": energy,
            })

    # Save outputs
    pd.DataFrame(jitter_records).to_csv(os.path.join(results_dir, "stress_jitter_sweep.csv"), index=False)
    pd.DataFrame(noise_records).to_csv(os.path.join(results_dir, "stress_noise_sweep.csv"), index=False)
    pd.DataFrame(dropout_records).to_csv(os.path.join(results_dir, "stress_dropout_sweep.csv"), index=False)

    detailed_json = os.path.join(results_dir, "stress_benchmark_detailed.json")
    with open(detailed_json, "w") as f:
        json.dump({
            "jitter_sweep": jitter_records,
            "noise_sweep": noise_records,
            "dropout_sweep": dropout_records,
        }, f, indent=2)
    print(f"\n[+] Saved stress benchmark datasets to {results_dir}")

    # Print summary table at extreme conditions
    print("\n" + "=" * 80)
    print("STRESS BENCHMARK SUMMARY AT ADVERSE OPERATING CONDITIONS")
    print("=" * 80)
    print(f"{'Model':<25} | {'Clean Acc':<10} | {'Jitter 8ms':<10} | {'Noise 15%':<10} | {'Drop 30%':<10} | {'SynOps/s':<10} | {'Energy (nJ)':<10}")
    print("-" * 100)

    for name in trained_models.keys():
        clean_acc = [r["accuracy"] for r in jitter_records if r["model"] == name and r["jitter_sigma_ms"] == 0.0][0]
        jit8_acc = [r["accuracy"] for r in jitter_records if r["model"] == name and r["jitter_sigma_ms"] == 8.0][0]
        noi15_acc = [r["accuracy"] for r in noise_records if r["model"] == name and r["noise_prob"] == 0.15][0]
        drop30_acc = [r["accuracy"] for r in dropout_records if r["model"] == name and r["dropout_prob"] == 0.3][0]
        clean_ops = [r["synops_per_sample"] for r in jitter_records if r["model"] == name and r["jitter_sigma_ms"] == 0.0][0]
        clean_energy = [r["energy_nj"] for r in jitter_records if r["model"] == name and r["jitter_sigma_ms"] == 0.0][0]

        print(f"{name:<25} | {clean_acc:9.1f}% | {jit8_acc:9.1f}% | {noi15_acc:9.1f}% | {drop30_acc:9.1f}% | {clean_ops:10.1f} | {clean_energy:10.2f}")
    print("=" * 100)


if __name__ == "__main__":
    run_stress_benchmark()
