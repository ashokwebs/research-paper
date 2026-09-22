"""
COMPUTATIONAL COUPLING — CoupledEventNet (CEN) Systematic Ablation Study
Investigates the causal contribution of each biological motif and constraint:
  1. Full CEN (cFFL + iFFL + RIP, Dale-compliant)
  2. Ablate cFFL (cffl_fraction = 0)
  3. Ablate iFFL (iffl_fraction = 0)
  4. Ablate RIP (rip_fraction = 0)
  5. Random Erdős-Rényi Sparse Control (Same density & E/I, zero motif bias)
  6. Dale-Violating Sparse Control (Random sign flips allowed)
Evaluates across multiple seeds with mean and standard deviation.
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
import torch.nn.functional as F
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.architecture.coupled_event_net import CoupledEventNet, CENConfig, SurrogateHeaviside
from experiments.architecture.run_coupled_event_net_benchmark import (
    generate_temporal_coincidence_dataset,
    evaluate_model,
    train_model,
)


class RandomSparseSNN(nn.Module):
    """
    Random Erdős-Rényi Sparse Control with matching density and Dale compliance,
    but with zero biological motif bias.
    """
    def __init__(self, cfg: CENConfig):
        super().__init__()
        self.cfg = cfg
        N = cfg.num_neurons
        D_in = cfg.input_dim
        D_out = cfg.output_dim

        n_exc = int(round(N * (cfg.ei_ratio / (1.0 + cfg.ei_ratio))))
        self.register_buffer("node_signs", torch.tensor([1.0]*n_exc + [-1.0]*(N - n_exc), dtype=torch.float32))
        dale_matrix = np.tile(self.node_signs.cpu().numpy(), (N, 1))
        self.register_buffer("dale_matrix", torch.tensor(dale_matrix, dtype=torch.float32))

        # Uniform random Erdős-Rényi mask with exact same density
        adj_mask = (np.random.rand(N, N) < cfg.density).astype(np.float32)
        np.fill_diagonal(adj_mask, 0.0)
        self.register_buffer("adj_mask", torch.tensor(adj_mask, dtype=torch.float32))
        self.register_buffer("out_degs", torch.tensor(adj_mask.sum(axis=0), dtype=torch.float32))

        self.w_in = nn.Parameter(torch.empty(N, D_in))
        nn.init.kaiming_uniform_(self.w_in, a=np.sqrt(5))
        self.w_rec_mag = nn.Parameter(torch.empty(N, N))
        nn.init.uniform_(self.w_rec_mag, 0.05, 0.25)
        self.w_out = nn.Linear(N, D_out)

        self.alpha_m = float(np.exp(-cfg.dt_ms / cfg.tau_m_ms))
        self.alpha_s = float(np.exp(-cfg.dt_ms / cfg.tau_syn_ms))

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, time_steps, _ = x_seq.shape
        N = self.cfg.num_neurons
        device = x_seq.device

        W_rec = F.relu(self.w_rec_mag) * self.adj_mask * self.dale_matrix

        v = torch.full((batch_size, N), self.cfg.v_rest, device=device)
        i_syn = torch.zeros((batch_size, N), device=device)
        readout_acc = torch.zeros((batch_size, self.cfg.output_dim), device=device)

        total_spikes = 0
        total_synops = 0

        for t in range(time_steps):
            x_t = x_seq[:, t, :]
            i_in = F.linear(x_t, self.w_in)
            s_prev = torch.zeros((batch_size, N), device=device) if t == 0 else s_prev
            i_rec = F.linear(s_prev, W_rec)

            i_syn = i_syn * self.alpha_s + (i_in + i_rec)
            v_new = self.cfg.v_rest + (v - self.cfg.v_rest) * self.alpha_m + i_syn * (1.0 - self.alpha_m)
            s = SurrogateHeaviside.apply(v_new, self.cfg.v_thresh)
            v = v_new * (1.0 - s) + self.cfg.v_reset * s

            out_t = self.w_out(v)
            readout_acc = readout_acc * 0.95 + out_t

            total_spikes += int(s.sum().item())
            total_synops += int(torch.sum(s @ self.out_degs).item())
            s_prev = s

        return readout_acc, {
            "total_spikes": total_spikes,
            "total_synops": total_synops,
            "avg_synops_per_sample": float(total_synops / batch_size),
            "energy_proxy_nj": float(total_synops * 0.020 + total_spikes * 0.005),
        }


def run_ablation_experiments(seeds: List[int] = [42, 123, 456]):
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Systematic Ablation Study of CoupledEventNet")
    print(f"Random Seeds: {seeds}")
    print("=" * 80)

    conditions = [
        ("Full CEN (cFFL+iFFL+RIP)", lambda: CoupledEventNet(CENConfig(output_dim=4, density=0.12, cffl_fraction=0.35, iffl_fraction=0.25, rip_fraction=0.15))),
        ("Ablate cFFL (No Persistence)", lambda: CoupledEventNet(CENConfig(output_dim=4, density=0.12, cffl_fraction=0.00, iffl_fraction=0.35, rip_fraction=0.25))),
        ("Ablate iFFL (No Adaptation)", lambda: CoupledEventNet(CENConfig(output_dim=4, density=0.12, cffl_fraction=0.50, iffl_fraction=0.00, rip_fraction=0.25))),
        ("Ablate RIP (No Mutual Inh)", lambda: CoupledEventNet(CENConfig(output_dim=4, density=0.12, cffl_fraction=0.50, iffl_fraction=0.35, rip_fraction=0.00))),
        ("Random Erdős-Rényi Sparse", lambda: RandomSparseSNN(CENConfig(output_dim=4, density=0.12))),
    ]

    all_results = []

    for cond_name, model_fn in conditions:
        print(f"\n[+] Testing Condition: {cond_name}...")
        test_accs = []
        noisy_accs = []
        synops_list = []
        energy_list = []

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)

            X_train, y_train = generate_temporal_coincidence_dataset(num_samples=400, seed=s)
            X_test, y_test = generate_temporal_coincidence_dataset(num_samples=200, seed=s + 1000)
            X_noisy = X_test + torch.randn_like(X_test) * 0.5

            model = model_fn()
            model = train_model(model, X_train, y_train, epochs=25, lr=0.015)

            acc, syn, avg_syn, e_nj = evaluate_model(model, X_test, y_test)
            acc_n, _, _, _ = evaluate_model(model, X_noisy, y_test)

            test_accs.append(acc)
            noisy_accs.append(acc_n)
            synops_list.append(avg_syn)
            energy_list.append(e_nj)

        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs))
        mean_noisy = float(np.mean(noisy_accs))
        std_noisy = float(np.std(noisy_accs))
        mean_syn = float(np.mean(synops_list))
        std_syn = float(np.std(synops_list))
        mean_e = float(np.mean(energy_list))

        print(f"    Test Acc: {mean_acc:.2f} ± {std_acc:.2f}% | Noisy Acc: {mean_noisy:.2f} ± {std_noisy:.2f}%")
        print(f"    SynOps:   {mean_syn:.1f} ± {std_syn:.1f} | Energy Proxy: {mean_e:.2f} nJ")

        all_results.append({
            "Condition": cond_name,
            "Mean_Test_Acc_%": mean_acc,
            "Std_Test_Acc_%": std_acc,
            "Mean_Noisy_Acc_%": mean_noisy,
            "Std_Noisy_Acc_%": std_noisy,
            "Noise_Degradation_%": mean_acc - mean_noisy,
            "Mean_SynOps": mean_syn,
            "Std_SynOps": std_syn,
            "Energy_Proxy_nJ": mean_e,
        })

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY:")
    print(df.to_string(index=False))
    print("=" * 80)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    csv_path = os.path.join(results_dir, "cen_ablation_study.csv")
    json_path = os.path.join(results_dir, "cen_ablation_study.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[✓] Saved ablation study results to:\n    - {csv_path}\n    - {json_path}")
    return all_results


if __name__ == "__main__":
    run_ablation_experiments()
