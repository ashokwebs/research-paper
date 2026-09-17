"""
COMPUTATIONAL COUPLING — Comparative Cross-Species Degeneracy Frontier
Compares the degeneracy frontier across three phylogenetic scales with cellular-resolution connectomes:
  1. C. elegans (100-neuron motor command subcircuit)
  2. Drosophila melanogaster (100-neuron Central Complex heading steering subcircuit)
  3. Mus musculus (100-neuron Pinky100 V1 Layer 2/3 microcircuit)

Quantifies:
  - Causal single-edge ablation profile (Invariants vs Regulatory vs Scaffolding)
  - Degeneracy volume and critical drift radius R_crit (max Delta_struct maintaining F_task >= 0.90)
  - SynOps reduction potential along the Pareto frontier
"""

import os
import sys
import json
import time
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connectome import load_celegans_connectome, load_drosophila_subcircuit, load_mouse_v1_connectome
from src.connectome.unified_schema import ConnectomeGraph
from src.simulation.lif_simulator import VectorizedLIFSimulator, SimulationConfig
from src.remodeling.pareto_remodeler import (
    CausalLesionAnalyzer,
    ParetoRemodeler,
    EdgeLesionResult,
    ParetoCandidate,
)


def extract_standardized_subcircuit(graph: ConnectomeGraph, target_size: int = 100) -> ConnectomeGraph:
    """Extract the top target_size most densely connected neurons to form a standardized subcircuit."""
    degs = sorted(graph.graph.degree(), key=lambda x: x[1], reverse=True)
    top_nodes = set(x[0] for x in degs[:target_size])
    return graph.get_subcircuit(top_nodes, name_suffix=f"top_{target_size}")


def evaluate_subcircuit_temporal_throughput(graph: ConnectomeGraph) -> Tuple[float, int]:
    """
    Evaluates circuit on a standardized temporal stimulus propagation & coincidence task:
    Delivers two synchronized high-frequency bursts to input sensors (first 15% nodes)
    and evaluates transmission fidelity and noise-free integration at readouts (last 20% nodes).
    """
    N = graph.num_neurons
    if N == 0:
        return 0.0, 0

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=20.0)
    sim = VectorizedLIFSimulator(graph, config=cfg)

    n_in = max(2, int(N * 0.15))
    n_out = max(2, int(N * 0.20))
    time_steps = 100  # 50 ms

    stim = np.zeros((time_steps, N), dtype=np.float32)
    # Deliver two coordinated sensory pulses
    stim[10:25, :n_in] = 75.0
    stim[45:60, :n_in] = 75.0

    res = sim.run(50.0, stim, record_voltages=False)

    target_spikes = int(res.spike_raster[:, -n_out:].sum())
    total_spikes = res.total_spikes

    # Expected readout: coordinated bursting without runaway epileptic activity
    expected_spikes = n_out * 4.0
    if target_spikes == 0:
        score = 0.0
    elif total_spikes > N * 20:
        # Penalize epileptic runaway discharge
        score = max(0.05, 1.0 - (total_spikes - N * 20) / (N * 25))
    else:
        score = min(1.0, float(target_spikes / max(1.0, expected_spikes)))

    return float(score), int(res.total_synops)


def run_comparative_degeneracy_study():
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Cross-Species Degeneracy Frontier Mapping")
    print("=" * 80)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load biological connectomes
    print("\n[+] Ingesting Connectomes for Cross-Species Comparison...")
    g_ce = load_celegans_connectome()
    g_cx = load_drosophila_subcircuit("cx", min_synapses=3)
    g_mouse = load_mouse_v1_connectome(min_synapses=1)

    species_circuits = {
        "C. elegans": extract_standardized_subcircuit(g_ce, 100),
        "Drosophila CX": extract_standardized_subcircuit(g_cx, 100),
        "Mouse V1 L2/3": extract_standardized_subcircuit(g_mouse, 100),
    }

    eval_fn = lambda g: evaluate_subcircuit_temporal_throughput(g)[0]
    eval_fn_with_ops = evaluate_subcircuit_temporal_throughput

    lesion_summary = []
    degeneracy_summary = []
    all_pareto_fronts = {}

    for species, circuit in species_circuits.items():
        print(f"\n{'='*40}\n[*] Processing Organism: {species}")
        print(f"    Neurons: {circuit.num_neurons}, Synapses: {circuit.num_synapses}, Density: {circuit.density:.4f}")

        # Baseline evaluation
        base_f, base_ops = eval_fn_with_ops(circuit)
        print(f"    Baseline Task F_task: {base_f:.4f}, Baseline SynOps: {base_ops:,}")

        # --- A. Causal Single-Edge Ablation ---
        print(f"    -> Running Causal Single-Edge Lesion Analysis (60 edges)...")
        analyzer = CausalLesionAnalyzer(circuit, eval_fn=eval_fn)
        t0 = time.time()
        lesion_results = analyzer.analyze_edges(max_edges=60, invariant_threshold=0.30, scaffolding_threshold=0.02)
        lesion_time = time.time() - t0

        n_inv = sum(1 for r in lesion_results if r.category == "invariant")
        n_reg = sum(1 for r in lesion_results if r.category == "regulatory")
        n_scaff = sum(1 for r in lesion_results if r.category == "scaffolding")
        total_tested = len(lesion_results)

        mean_drop = float(np.mean([r.relative_drop for r in lesion_results]))
        max_drop = float(np.max([r.relative_drop for r in lesion_results])) if lesion_results else 0.0

        print(f"    Lesions completed in {lesion_time:.2f}s:")
        print(f"      Invariants (drop >= 30%):   {n_inv:2d} ({n_inv/total_tested*100:.1f}%)")
        print(f"      Regulatory (2% - 30% drop): {n_reg:2d} ({n_reg/total_tested*100:.1f}%)")
        print(f"      Scaffolding (drop <= 2%):   {n_scaff:2d} ({n_scaff/total_tested*100:.1f}%)")
        print(f"      Mean Relative Drop:         {mean_drop*100:.2f}% (Max: {max_drop*100:.2f}%)")

        lesion_summary.append({
            "organism": species,
            "num_neurons": circuit.num_neurons,
            "num_synapses": circuit.num_synapses,
            "synaptic_density": circuit.density,
            "edges_tested": total_tested,
            "invariants_count": n_inv,
            "invariants_pct": float(n_inv / total_tested * 100),
            "regulatory_count": n_reg,
            "regulatory_pct": float(n_reg / total_tested * 100),
            "scaffolding_count": n_scaff,
            "scaffolding_pct": float(n_scaff / total_tested * 100),
            "mean_relative_drop": mean_drop,
            "max_relative_drop": max_drop,
        })

        # --- B. Pareto Degeneracy Frontier Remodeling ---
        print(f"    -> Running Multi-Objective Pareto Remodeling (12 gens, pop 16)...")
        remodeler = ParetoRemodeler(circuit, eval_fn=eval_fn_with_ops, degree_slack=0.25)
        t0 = time.time()
        all_explored = remodeler.run_evolution(
            generations=12,
            population_size=16,
            mutation_rate=0.08,
            seed=42,
        )
        pareto_front = [c for c in all_explored if c.is_pareto_optimal]
        opt_time = time.time() - t0

        # Metrics on Pareto Front
        # R_90: max structural drift with F_task >= 0.90 * baseline
        # R_80: max structural drift with F_task >= 0.80 * baseline
        qualifying_90 = [c for c in pareto_front if c.task_performance >= 0.90 * base_f]
        qualifying_80 = [c for c in pareto_front if c.task_performance >= 0.80 * base_f]
        r_90 = float(max([c.structural_distance for c in qualifying_90])) if qualifying_90 else 0.0
        r_80 = float(max([c.structural_distance for c in qualifying_80])) if qualifying_80 else 0.0

        # Maximum SynOps reduction preserving task competence (F_task >= 0.90 * baseline)
        if qualifying_90:
            min_ops = min(c.total_synops for c in qualifying_90)
            synops_reduction_pct = float((base_ops - min_ops) / max(1, base_ops) * 100)
        else:
            synops_reduction_pct = 0.0

        print(f"    Pareto Optimization completed in {opt_time:.2f}s:")
        print(f"      Frontier size: {len(pareto_front)} candidates")
        print(f"      Critical Drift Radius R_90 (F >= 90%): {r_90:.4f}")
        print(f"      Critical Drift Radius R_80 (F >= 80%): {r_80:.4f}")
        print(f"      Max SynOps Reduction at F >= 90%:       {synops_reduction_pct:.1f}%")

        front_records = []
        for c in pareto_front:
            front_records.append({
                "structural_distance": c.structural_distance,
                "task_performance": c.task_performance,
                "total_synops": c.total_synops,
                "generation": c.generation,
                "dales_valid": c.dales_valid,
            })
        all_pareto_fronts[species] = front_records

        degeneracy_summary.append({
            "organism": species,
            "baseline_score": base_f,
            "baseline_synops": base_ops,
            "critical_radius_r90": r_90,
            "critical_radius_r80": r_80,
            "synops_reduction_pct": synops_reduction_pct,
            "pareto_front_size": len(pareto_front),
            "optimization_time_sec": opt_time,
        })

    # Save outputs
    lesion_df = pd.DataFrame(lesion_summary)
    lesion_csv_path = os.path.join(results_dir, "cross_species_causal_lesion_summary.csv")
    lesion_df.to_csv(lesion_csv_path, index=False)
    print(f"\n[+] Saved lesion summary to {lesion_csv_path}")

    deg_df = pd.DataFrame(degeneracy_summary)
    deg_csv_path = os.path.join(results_dir, "cross_species_degeneracy_frontier_summary.csv")
    deg_df.to_csv(deg_csv_path, index=False)
    print(f"[+] Saved degeneracy summary to {deg_csv_path}")

    combined_json_path = os.path.join(results_dir, "cross_species_degeneracy_detailed.json")
    with open(combined_json_path, "w") as f:
        json.dump({
            "lesion_summary": lesion_summary,
            "degeneracy_summary": degeneracy_summary,
            "pareto_fronts": all_pareto_fronts,
        }, f, indent=2)
    print(f"[+] Saved detailed cross-species JSON to {combined_json_path}")
    print("\n" + "=" * 80)
    print("CROSS-SPECIES DEGENERACY EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_comparative_degeneracy_study()
