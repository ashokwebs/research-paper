"""
Experiment: Self-Organized Criticality and Neuronal Avalanche Scaling Benchmark.

Compares:
1. ActiveDendriticCEN (Ours, 2-compartment with motifs and NMDA plateaus)
2. CoupledEventNet (Ours, 1-compartment with motifs)
3. Degenerate Frontier Remodeled Connectome (Pareto Knee, 50% pruned)
4. Raw Biological Connectome (C. elegans)
5. Dense Recurrent SNN (Standard Point LIF, dense unconstrained)
6. Random Rewired Control (Erdos-Renyi with Dale constraints)

Analyzes:
- Branching ratio sigma (via least-squares autoregressive operator)
- Neuronal avalanche size exponent tau
- Neuronal avalanche duration exponent alpha
- Crackling noise scaling relation gamma = (alpha - 1) / (tau - 1)
- Validation of Theorem 2 (Criticality Preservation under Degeneracy Pruning)
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.architecture.coupled_event_net import CoupledEventNet, CENConfig
from src.architecture.active_dendritic_cen import ActiveDendriticCEN, ActiveDendriticConfig
from src.connectome.celegans_loader import load_celegans_connectome
from src.theory.criticality import analyze_network_criticality, AvalancheProfile


def run_criticality_experiment() -> Dict[str, Any]:
    print("=" * 105)
    print("RUNNING SELF-ORGANIZED CRITICALITY & NEURONAL AVALANCHE BENCHMARK")
    print("=" * 105)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    T_steps = 3000
    N_neurons = 100
    dt_ms = 1.0
    alpha_leak = np.exp(-dt_ms / 20.0)
    
    # 1. Load biological connectome
    celegans_g = load_celegans_connectome()
    adj_bio, node_ids = celegans_g.get_adjacency_matrix(weighted=True, signed=True)
    W_bio = adj_bio[:N_neurons, :N_neurons].copy()
    # Normalize baseline spectral radius to near-critical regime (~1.05)
    ev_max = np.max(np.abs(np.linalg.eigvals(W_bio)))
    W_bio = W_bio * (1.05 / max(1e-4, ev_max))
    
    # 2. Setup architectures
    models_to_test = {}
    
    # (A) ActiveDendriticCEN
    cfg_ad = ActiveDendriticConfig(input_dim=10, num_neurons=N_neurons, output_dim=2)
    ad_cen = ActiveDendriticCEN(config=cfg_ad)
    
    # (B) CoupledEventNet
    cfg_cen = CENConfig(input_dim=10, num_neurons=N_neurons, output_dim=2)
    cen = CoupledEventNet(config=cfg_cen)
    
    # (C) Degenerate Frontier Remodeled Connectome (50% pruned)
    W_pruned = W_bio.copy()
    threshold = np.percentile(np.abs(W_pruned[W_pruned != 0]), 50)
    W_pruned[np.abs(W_pruned) < threshold] = 0.0
    # Rescale to preserve spectral radius of the invariant core
    ev_pruned = np.max(np.abs(np.linalg.eigvals(W_pruned)))
    W_pruned = W_pruned * (1.02 / max(1e-4, ev_pruned))
    
    # (D) Dense Recurrent SNN (unconstrained, non-Dale)
    W_dense = np.random.randn(N_neurons, N_neurons) * (1.15 / np.sqrt(N_neurons))
    
    # (E) Random Rewired Control (Erdos-Renyi with Dale valences)
    p_conn = np.count_nonzero(W_bio) / (N_neurons * N_neurons)
    W_rand = (np.random.rand(N_neurons, N_neurons) < p_conn).astype(float)
    signs = np.array([1.0 if np.random.rand() < 0.8 else -1.0 for _ in range(N_neurons)])
    W_rand = W_rand * signs[:, None] * (0.9 / np.sqrt(N_neurons * p_conn + 1e-4))
    
    # Create pulsed sensory stimulus with quiescent intervals to observe discrete cascades
    stimulus = np.zeros((T_steps, N_neurons))
    pulse_times = np.arange(10, T_steps - 30, 35)
    for pt in pulse_times:
        n_inj = np.random.randint(2, 6)
        active_nodes = np.random.choice(N_neurons, size=n_inj, replace=False)
        stimulus[pt, active_nodes] = 1.0
        
    results: Dict[str, AvalancheProfile] = {}
    
    # Helper simulator for graph matrices
    def simulate_graph_spikes(W_mat: np.ndarray, stim: np.ndarray, syn_gain: float = 1.6) -> np.ndarray:
        T, N = stim.shape
        v = np.zeros(N)
        spikes = np.zeros((T, N))
        v_thresh = 1.0
        v_reset = 0.0
        
        for t in range(T):
            syn_input = (W_mat.T @ spikes[t-1]) * syn_gain if t > 0 else np.zeros(N)
            v = alpha_leak * v + syn_input + stim[t] * 1.2
            fired = v >= v_thresh
            spikes[t, fired] = 1.0
            v[fired] = v_reset
        return spikes

    # --- Test 1: ActiveDendriticCEN ---
    print("\n[1/6] Simulating ActiveDendriticCEN (Ours)...")
    stim_t = torch.zeros(1, T_steps, 10)
    for pt in pulse_times:
        stim_t[0, pt, :4] = 2.8
    with torch.no_grad():
        _, stats_ad = ad_cen(stim_t, return_dynamics=True)
        spikes_ad_np = stats_ad["spikes"][0].cpu().numpy()
    prof_ad = analyze_network_criticality(spikes_ad_np, bin_size=1)
    results["ActiveDendriticCEN (Ours)"] = prof_ad
    
    # --- Test 2: CoupledEventNet ---
    print("[2/6] Simulating CoupledEventNet (Ours)...")
    with torch.no_grad():
        # Scale input projection for CEN to evoke clear cascades
        cen.w_in.data = cen.w_in.data * 2.5
        _, stats_cen = cen(stim_t, return_dynamics=True)
        spikes_cen_np = np.array([s[0] for s in stats_cen["spikes_history"]])
    prof_cen = analyze_network_criticality(spikes_cen_np, bin_size=1)
    results["CoupledEventNet (Ours)"] = prof_cen
    
    # --- Test 3: Degenerate Frontier Remodeled Connectome ---
    print("[3/6] Simulating Degenerate Frontier Remodeled Connectome (50% Pruned)...")
    spikes_pruned = simulate_graph_spikes(W_pruned, stimulus, syn_gain=1.75)
    prof_pruned = analyze_network_criticality(spikes_pruned, bin_size=1)
    results["Degenerate Pruned (50% Knee)"] = prof_pruned
    
    # --- Test 4: Raw Biological Connectome ---
    print("[4/6] Simulating Raw Biological Connectome (C. elegans)...")
    spikes_bio = simulate_graph_spikes(W_bio, stimulus, syn_gain=1.75)
    prof_bio = analyze_network_criticality(spikes_bio, bin_size=1)
    results["Raw Connectome (C. elegans)"] = prof_bio
    
    # --- Test 5: Dense Recurrent SNN ---
    print("[5/6] Simulating Dense Recurrent SNN (Unconstrained)...")
    spikes_dense = simulate_graph_spikes(W_dense, stimulus, syn_gain=1.20)
    prof_dense = analyze_network_criticality(spikes_dense, bin_size=1)
    results["Dense Recurrent SNN"] = prof_dense
    
    # --- Test 6: Random Rewired Control ---
    print("[6/6] Simulating Random Rewired Control (Erdos-Renyi)...")
    spikes_rand = simulate_graph_spikes(W_rand, stimulus, syn_gain=1.20)
    prof_rand = analyze_network_criticality(spikes_rand, bin_size=1)
    results["Random Rewired Control"] = prof_rand

    # Print comparative table
    print("\n" + "=" * 105)
    print(f"{'Architecture':<32} | {'Branching (sigma)':<20} | {'Size (tau)':<10} | {'Dur (alpha)':<10} | {'Scaling gamma':<12} | {'Dynamic Regime'}")
    print("-" * 105)
    
    summary_data = []
    for arch, p in results.items():
        sigma_str = f"{p.branching_ratio:.3f} +/- {p.branching_ratio_std:.3f}"
        if p.is_critical:
            crit_str = "CRITICAL (Edge-of-Chaos)"
        elif p.branching_ratio > 1.05:
            crit_str = "SUPERCRITICAL (Runaway)"
        else:
            crit_str = "SUBCRITICAL (Quenched)"
            
        print(f"{arch:<32} | {sigma_str:<20} | {p.tau_size_exponent:<10.2f} | {p.alpha_duration_exponent:<10.2f} | {p.crackling_scaling_gamma:<12.2f} | {crit_str}")
        summary_data.append({
            "architecture": arch,
            "branching_ratio": round(p.branching_ratio, 4),
            "branching_ratio_std": round(p.branching_ratio_std, 4),
            "num_avalanches": p.num_avalanches,
            "mean_avalanche_size": round(p.mean_avalanche_size, 2),
            "max_avalanche_size": p.max_avalanche_size,
            "mean_avalanche_duration": round(p.mean_avalanche_duration, 2),
            "max_avalanche_duration": p.max_avalanche_duration,
            "tau_size_exponent": round(p.tau_size_exponent, 3),
            "alpha_duration_exponent": round(p.alpha_duration_exponent, 3),
            "crackling_scaling_gamma": round(p.crackling_scaling_gamma, 3),
            "dynamic_regime": crit_str,
            "is_critical": p.is_critical,
        })
    print("=" * 105)
    
    # Save output files
    os.makedirs("results", exist_ok=True)
    out_json = "results/criticality_avalanche_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved benchmark metrics to {out_json}")
    
    out_csv = "results/criticality_avalanche_benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"Saved tabular metrics to {out_csv}")
    
    return {"status": "SUCCESS", "results": summary_data}


if __name__ == "__main__":
    run_criticality_experiment()
