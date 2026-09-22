"""
Experiment: Multi-Core Neuromorphic Network-on-Chip (NoC) Mesh Scaling Benchmark.

Compares:
1. Motif-Aware Spectral Partitioning (Ours)
2. Uniform Sequential Partitioning (Standard Baseline)

Across Network Cardinalities:
- N = 128  (2x2 Mesh = 4 Cores)
- N = 512  (4x4 Mesh = 16 Cores)
- N = 1024 (4x4 Mesh = 16 Cores)
- N = 2048 (8x8 Mesh = 64 Cores)

Quantifies:
- Inter-core packet ratio (traffic escaping local core)
- Mean router hops per packet
- Packet latency (clock cycles)
- Dynamic NoC routing energy (pJ / spike)
- Peak link contention
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hardware.noc.noc_mesh import NeuromorphicNoCMesh, NoCBenchmarkResult
from src.connectome.graph_translator import InvariantMotifGraphTranslator


def run_noc_benchmark() -> Dict[str, Any]:
    print("=" * 105)
    print("RUNNING MULTI-CORE NEUROMORPHIC NETWORK-ON-CHIP (NoC) SCALING BENCHMARK")
    print("=" * 105)
    
    np.random.seed(42)
    T_steps = 1000
    p_spike = 0.02  # 20 Hz event rate at 1 ms bins
    
    configurations = [
        {"N": 128,  "mesh_dim": 2, "name": "128-Neuron (2x2 Mesh, 4 Cores)"},
        {"N": 512,  "mesh_dim": 4, "name": "512-Neuron (4x4 Mesh, 16 Cores)"},
        {"N": 1024, "mesh_dim": 4, "name": "1024-Neuron (4x4 Mesh, 16 Cores)"},
        {"N": 2048, "mesh_dim": 8, "name": "2048-Neuron (8x8 Mesh, 64 Cores)"},
    ]
    
    results: List[Dict[str, Any]] = []
    translator = InvariantMotifGraphTranslator(target_ei_ratio=2.69, seed=42)
    
    for cfg in configurations:
        N = cfg["N"]
        mesh_dim = cfg["mesh_dim"]
        label = cfg["name"]
        print(f"\n--- Testing {label} ---")
        
        # 1. Synthesize scalable Dale-compliant connectome with cFFL/iFFL/RIP motifs
        adj_matrix, dale_signs, metrics = translator.generate_scaled_connectome(N)
        signed_adj = adj_matrix * dale_signs[:, None]
        
        # 2. Generate synthetic spike train
        spike_train = (np.random.rand(T_steps, N) < p_spike).astype(int)
        
        # 3. Strategy A: Uniform Sequential Partitioning
        noc_uniform = NeuromorphicNoCMesh(mesh_dim=mesh_dim, core_capacity=max(32, N // (mesh_dim**2)))
        noc_uniform.partition_neurons_uniform(N)
        res_uniform = noc_uniform.simulate_event_traffic(spike_train, signed_adj, strategy_name="Uniform Partitioning")
        
        # 4. Strategy B: Motif-Aware Spectral Partitioning
        noc_motif = NeuromorphicNoCMesh(mesh_dim=mesh_dim, core_capacity=max(32, N // (mesh_dim**2)))
        noc_motif.partition_neurons_motif_aware(signed_adj, motif_mask=None)
        res_motif = noc_motif.simulate_event_traffic(spike_train, signed_adj, strategy_name="Motif-Aware Partitioning (Ours)")
        
        # Calculate improvements
        traffic_reduction = (1.0 - res_motif.inter_core_ratio / max(1e-5, res_uniform.inter_core_ratio)) * 100.0
        energy_advantage = res_uniform.total_noc_energy_pj / max(1e-5, res_motif.total_noc_energy_pj)
        latency_reduction = (1.0 - res_motif.mean_latency_cycles / max(1e-5, res_uniform.mean_latency_cycles)) * 100.0
        
        print(f"  [Uniform] Inter-Core Ratio: {res_uniform.inter_core_ratio*100:.1f}%, Mean Hops: {res_uniform.mean_hops_per_packet:.2f}, Latency: {res_uniform.mean_latency_cycles:.1f} cycles, Energy: {res_uniform.total_noc_energy_pj/1000:.2f} nJ")
        print(f"  [Motif]   Inter-Core Ratio: {res_motif.inter_core_ratio*100:.1f}%, Mean Hops: {res_motif.mean_hops_per_packet:.2f}, Latency: {res_motif.mean_latency_cycles:.1f} cycles, Energy: {res_motif.total_noc_energy_pj/1000:.2f} nJ")
        print(f"  --> Traffic Reduction: {traffic_reduction:.1f}%, Energy Advantage: {energy_advantage:.2f}x, Latency Speedup: {latency_reduction:.1f}%")
        
        for res, strat in [(res_uniform, "Uniform"), (res_motif, "Motif-Aware (Ours)")]:
            results.append({
                "network_label": label,
                "num_neurons": N,
                "mesh_dimension": f"{mesh_dim}x{mesh_dim}",
                "total_cores": mesh_dim**2,
                "strategy": strat,
                "total_spikes": res.total_spikes,
                "intra_core_events": res.intra_core_events,
                "inter_core_packets": res.inter_core_packets,
                "inter_core_ratio_pct": round(res.inter_core_ratio * 100.0, 2),
                "mean_hops_per_packet": round(res.mean_hops_per_packet, 2),
                "mean_latency_cycles": round(res.mean_latency_cycles, 2),
                "max_latency_cycles": res.max_latency_cycles,
                "total_noc_energy_nj": round(res.total_noc_energy_pj / 1000.0, 4),
                "energy_per_spike_pj": round(res.mean_energy_per_spike_pj, 3),
                "peak_link_contention": int(res.link_utilization_peak),
            })
            
    # Save results
    os.makedirs("results", exist_ok=True)
    out_json = "results/neuromorphic_noc_scaling.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved NoC scaling benchmark to {out_json}")
    
    out_csv = "results/neuromorphic_noc_scaling.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved tabular results to {out_csv}")
    
    return {"status": "SUCCESS", "results": results}


if __name__ == "__main__":
    run_noc_benchmark()
