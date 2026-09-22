"""
COMPUTATIONAL COUPLING — CoupledEventNet (CEN) vs SNN / Reservoir Baselines
Benchmarks CoupledEventNet against:
  1. Dense SNN (unconstrained recurrent weights, Dale-violating)
  2. Connectome Reservoir (conn2res / fixed biological connectome topology)
on multi-class temporal spike pattern discrimination.
Evaluates:
  - Task Accuracy (%)
  - Synaptic Operations (SynOps)
  - Synaptic Energy Proxy (nJ)
  - Spike Sparsity (%)
  - Noise Robustness (Accuracy Degradation under Jitter)
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


def generate_temporal_coincidence_dataset(
    num_samples: int = 400,
    time_steps: int = 50,
    num_channels: int = 10,
    num_classes: int = 4,
    noise_prob: float = 0.05,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesize temporal pattern discrimination dataset:
      - Class 0: Coincident bursts on channels (0, 1, 2) at t=15
      - Class 1: Sequential staggered pulses on channels (3, 4, 5) at t=10, 20, 30
      - Class 2: High-frequency transient burst on channels (6, 7) at t=5..15
      - Class 3: Sustained low-frequency rate on channels (8, 9) across t=10..40
    All embedded in background Poisson noise spikes.
    """
    np.random.seed(seed)
    X = np.random.binomial(1, noise_prob, size=(num_samples, time_steps, num_channels)).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))

    for i in range(num_samples):
        cls = y[i]
        if cls == 0:
            # Coincident synchronous pulse across ch 0, 1, 2 at t=15..17
            X[i, 15:18, 0:3] = 1.0
        elif cls == 1:
            # Sequential cascade
            X[i, 10:13, 3] = 1.0
            X[i, 20:23, 4] = 1.0
            X[i, 30:33, 5] = 1.0
        elif cls == 2:
            # High-frequency burst
            X[i, 5:15, 6:8] = np.random.binomial(1, 0.8, size=(10, 2))
        elif cls == 3:
            # Sustained low-frequency rate
            X[i, 10:40, 8:10] = np.random.binomial(1, 0.25, size=(30, 2))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, int, float, float]:
    """Evaluate accuracy, total SynOps, average SynOps per sample, and energy proxy."""
    model.eval()
    with torch.no_grad():
        logits, stats = model(X)
        preds = torch.argmax(logits, dim=1)
        acc = float((preds == y).float().mean().item()) * 100.0
        synops = stats["total_synops"]
        avg_synops = stats["avg_synops_per_sample"]
        energy_nj = stats["energy_proxy_nj"]
    return acc, synops, avg_synops, energy_nj


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
            idx = perm[b : b + batch_size]
            bx, by = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    return model


def run_architecture_benchmark():
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — CoupledEventNet (CEN) Comparative Benchmark")
    print("=" * 80)

    # 1. Prepare Dataset
    X_train, y_train = generate_temporal_coincidence_dataset(num_samples=400, seed=42)
    X_test, y_test = generate_temporal_coincidence_dataset(num_samples=200, seed=123)
    X_noisy = X_test + torch.randn_like(X_test) * 0.5  # Noisy test set for robustness evaluation

    num_neurons = 128
    input_dim = 10
    output_dim = 4

    results = []

    # Model 1: CoupledEventNet (CEN)
    print("\n[+] Instantiating and training CoupledEventNet (CEN)...")
    cen_cfg = CENConfig(
        input_dim=input_dim,
        num_neurons=num_neurons,
        output_dim=output_dim,
        ei_ratio=4.0,
        density=0.12,
        cffl_fraction=0.35,
        iffl_fraction=0.25,
        rip_fraction=0.15,
    )
    cen_model = CoupledEventNet(cen_cfg)
    t0 = time.time()
    cen_model = train_model(cen_model, X_train, y_train, epochs=25, lr=0.015)
    cen_time = time.time() - t0

    acc_cen, synops_cen, avg_synops_cen, e_cen = evaluate_model(cen_model, X_test, y_test)
    acc_cen_noisy, _, _, _ = evaluate_model(cen_model, X_noisy, y_test)

    print(f"    CEN Test Accuracy: {acc_cen:.2f}% | Noisy Acc: {acc_cen_noisy:.2f}%")
    print(f"    CEN SynOps/Sample: {avg_synops_cen:.1f} | Energy Proxy: {e_cen:.2f} nJ")

    results.append({
        "Model": "CoupledEventNet (CEN)",
        "Dale_Compliant": True,
        "Wiring_Structure": "Motif-Constrained (cFFL/iFFL/RIP)",
        "Test_Accuracy_%": acc_cen,
        "Noisy_Accuracy_%": acc_cen_noisy,
        "Noise_Drop_%": acc_cen - acc_cen_noisy,
        "SynOps_Per_Sample": avg_synops_cen,
        "Energy_Proxy_nJ": e_cen,
        "Training_Time_s": cen_time,
    })

    # Model 2: Dense SNN Baseline
    print("\n[+] Instantiating and training Dense SNN (Dale-Violating)...")
    dense_model = DenseSNN(input_dim=input_dim, num_neurons=num_neurons, output_dim=output_dim)
    t0 = time.time()
    dense_model = train_model(dense_model, X_train, y_train, epochs=25, lr=0.01)
    dense_time = time.time() - t0

    acc_dense, synops_dense, avg_synops_dense, e_dense = evaluate_model(dense_model, X_test, y_test)
    acc_dense_noisy, _, _, _ = evaluate_model(dense_model, X_noisy, y_test)

    print(f"    Dense SNN Test Accuracy: {acc_dense:.2f}% | Noisy Acc: {acc_dense_noisy:.2f}%")
    print(f"    Dense SNN SynOps/Sample: {avg_synops_dense:.1f} | Energy Proxy: {e_dense:.2f} nJ")

    results.append({
        "Model": "Dense SNN (Standard)",
        "Dale_Compliant": False,
        "Wiring_Structure": "Dense Recurrent (All-to-All)",
        "Test_Accuracy_%": acc_dense,
        "Noisy_Accuracy_%": acc_dense_noisy,
        "Noise_Drop_%": acc_dense - acc_dense_noisy,
        "SynOps_Per_Sample": avg_synops_dense,
        "Energy_Proxy_nJ": e_dense,
        "Training_Time_s": dense_time,
    })

    # Model 3: Connectome Reservoir Baseline (conn2res with C. elegans subcircuit)
    print("\n[+] Instantiating and training Connectome Reservoir (conn2res / C. elegans)...")
    ce_conn = load_celegans_connectome()
    adj_ce, _ = ce_conn.get_adjacency_matrix(weighted=True, signed=True)
    # Take 128x128 submatrix
    adj_sub = adj_ce[:num_neurons, :num_neurons]

    res_model = ConnectomeReservoirNet(
        adjacency_matrix=adj_sub,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    t0 = time.time()
    res_model = train_model(res_model, X_train, y_train, epochs=25, lr=0.02)
    res_time = time.time() - t0

    acc_res, synops_res, avg_synops_res, e_res = evaluate_model(res_model, X_test, y_test)
    acc_res_noisy, _, _, _ = evaluate_model(res_model, X_noisy, y_test)

    print(f"    Connectome Reservoir Test Accuracy: {acc_res:.2f}% | Noisy Acc: {acc_res_noisy:.2f}%")
    print(f"    Connectome Reservoir SynOps/Sample: {avg_synops_res:.1f} | Energy Proxy: {e_res:.2f} nJ")

    results.append({
        "Model": "Connectome Reservoir (conn2res)",
        "Dale_Compliant": True,
        "Wiring_Structure": "Fixed C. elegans Connectome",
        "Test_Accuracy_%": acc_res,
        "Noisy_Accuracy_%": acc_res_noisy,
        "Noise_Drop_%": acc_res - acc_res_noisy,
        "SynOps_Per_Sample": avg_synops_res,
        "Energy_Proxy_nJ": e_res,
        "Training_Time_s": res_time,
    })

    # Summary Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY COMPARISON:")
    print(df.to_string(index=False))
    print("=" * 80)

    # Save to disk
    res_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(res_dir, exist_ok=True)
    csv_file = os.path.join(res_dir, "cen_benchmark_table.csv")
    json_file = os.path.join(res_dir, "cen_benchmark_results.json")

    df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Saved benchmark results to:\n    - {csv_file}\n    - {json_file}")
    return results


if __name__ == "__main__":
    run_architecture_benchmark()
