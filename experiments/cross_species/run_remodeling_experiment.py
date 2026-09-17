"""
COMPUTATIONAL COUPLING — Phase 2: Remodeling along the Degeneracy Frontier
Executes:
  1. Causal edge lesioning to segregate Computational Invariants from Developmental Scaffolding.
  2. Multi-objective evolutionary search exploring the Degeneracy Frontier
     (maximizing structural drift Delta_struct while maintaining task competence F_task
      under Dale's principle and degree bounds).
Saves structured outputs to results/.
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

from src.connectome import load_celegans_connectome, load_mouse_v1_connectome
from src.connectome.unified_schema import ConnectomeGraph
from src.simulation.lif_simulator import VectorizedLIFSimulator, SimulationConfig
from src.remodeling.pareto_remodeler import (
    CausalLesionAnalyzer,
    ParetoRemodeler,
    EdgeLesionResult,
    ParetoCandidate,
)


def evaluate_circuit_coincidence_task(graph: ConnectomeGraph) -> Tuple[float, int]:
    """
    Evaluates circuit on a standardized sensory-to-command propagation task:
    Requires propagating sensory bursts from input sensors (e.g. ASH / sensory)
    to command interneurons (e.g. AVA / command).
    Returns (task_performance in [0, 1], total_synops).
    """
    N = graph.num_neurons
    if N == 0:
        return 0.0, 0

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=20.0)
    sim = VectorizedLIFSimulator(graph, config=cfg)

    # First 15% of nodes as sensory inputs, last 20% as readout targets
    n_in = max(2, int(N * 0.15))
    n_out = max(2, int(N * 0.20))
    time_steps = 100  # 50 ms

    stim = np.zeros((time_steps, N), dtype=np.float32)
    # Deliver two coordinated sensory bursts
    stim[10:25, :n_in] = 70.0
    stim[40:55, :n_in] = 70.0

    res = sim.run(50.0, stim, record_voltages=False)

    # Readout targets are the last n_out nodes
    target_spikes = int(res.spike_raster[:, -n_out:].sum())
    total_spikes = res.total_spikes

    # Ideal response: downstream targets fire cleanly (2-8 spikes per target)
    expected_spikes = n_out * 4.0
    if target_spikes == 0:
        score = 0.0
    elif total_spikes > N * 15:
        # Runaway epileptic penalty
        score = max(0.1, 1.0 - (total_spikes - N * 15) / (N * 20))
    else:
        score = min(1.0, float(target_spikes / max(1.0, expected_spikes)))

    return float(score), int(res.total_synops)


def run_phase2_experiments():
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Phase 2: Degeneracy Frontier & Causal Lesioning")
    print("=" * 80)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load biological circuit (C. elegans)
    print("\n[+] Loading C. elegans Connectome...")
    g_ce = load_celegans_connectome()
    print(f"    Loaded {g_ce.num_neurons} neurons, {g_ce.num_synapses} synapses")

    # 2. Causal Edge Lesioning
    print("\n[*] Running Causal Single-Edge Ablation (Invariants vs Scaffolding)...")
    eval_fn = lambda g: evaluate_circuit_coincidence_task(g)[0]
    analyzer = CausalLesionAnalyzer(g_ce, eval_fn=eval_fn)

    t0 = time.time()
    lesion_results = analyzer.analyze_edges(max_edges=80, invariant_threshold=0.30, scaffolding_threshold=0.02)
    lesion_time = time.time() - t0

    # Tally categories
    invariants = [r for r in lesion_results if r.category == "invariant"]
    regulatory = [r for r in lesion_results if r.category == "regulatory"]
    scaffolding = [r for r in lesion_results if r.category == "scaffolding"]

    print(f"    Lesion analysis completed in {lesion_time:.2f}s ({len(lesion_results)} edges tested)")
    print(f"    - Computational Invariants: {len(invariants)} ({len(invariants)/len(lesion_results)*100:.1f}%)")
    print(f"    - Regulatory Connections:  {len(regulatory)} ({len(regulatory)/len(lesion_results)*100:.1f}%)")
    print(f"    - Developmental Scaffolding: {len(scaffolding)} ({len(scaffolding)/len(lesion_results)*100:.1f}%)")

    # 3. Evolutionary Search along the Degeneracy Frontier
    print("\n[*] Mapping the Degeneracy Frontier via Multi-Objective Remodeling...")
    # Use induced subcircuit of 80 nodes for evolutionary remodeling
    nodes_sub = sorted(list(g_ce.graph.nodes()))[:80]
    g_sub = g_ce.get_subcircuit(set(nodes_sub), name_suffix="sub80")
    print(f"    Using 80-neuron functional subcircuit ({g_sub.num_synapses} synapses) for Pareto evolution...")

    remodeler = ParetoRemodeler(
        base_graph=g_sub,
        eval_fn=evaluate_circuit_coincidence_task,
        degree_slack=0.20,
    )

    t0 = time.time()
    pareto_candidates = remodeler.run_evolution(
        population_size=16,
        generations=8,
        mutation_rate=0.08,
        seed=42,
    )
    evo_time = time.time() - t0

    pareto_optimal = [c for c in pareto_candidates if c.is_pareto_optimal]
    max_drift_competent = max(
        [c.structural_distance for c in pareto_candidates if c.task_performance >= 0.85 * remodeler.base_score],
        default=0.0
    )

    print(f"    Evolution completed in {evo_time:.2f}s ({len(pareto_candidates)} total candidates evaluated)")
    print(f"    - Pareto-Optimal Solutions: {len(pareto_optimal)}")
    print(f"    - Baseline Task Performance: {remodeler.base_score:.3f} | Baseline SynOps: {remodeler.base_ops}")
    print(f"    - Max Structural Drift at >= 85% Competence: Delta_struct = {max_drift_competent:.3f}")

    # Display Top Pareto Solutions
    pareto_sorted = sorted(pareto_optimal, key=lambda c: c.structural_distance)
    print("\n    Top Degeneracy Frontier Candidates:")
    print(f"    {'Gen':>4} {'Delta_struct':>14} {'F_task':>10} {'SynOps':>10} {'Dale Valid':>12}")
    print("    " + "-" * 54)
    for c in pareto_sorted[:10]:
        print(f"    {c.generation:>4} {c.structural_distance:>14.3f} {c.task_performance:>10.3f} {c.total_synops:>10} {str(c.dales_valid):>12}")

    # 4. Save Structured Results
    results_payload = {
        "causal_lesioning": {
            "total_tested": len(lesion_results),
            "num_invariants": len(invariants),
            "num_regulatory": len(regulatory),
            "num_scaffolding": len(scaffolding),
            "invariants_fraction": float(len(invariants) / len(lesion_results)),
            "scaffolding_fraction": float(len(scaffolding) / len(lesion_results)),
            "sample_invariants": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "weight": r.weight,
                    "sign": r.sign,
                    "relative_drop": r.relative_drop,
                }
                for r in invariants[:5]
            ],
        },
        "degeneracy_frontier": {
            "num_candidates": len(pareto_candidates),
            "num_pareto_optimal": len(pareto_optimal),
            "base_score": float(remodeler.base_score),
            "base_synops": int(remodeler.base_ops),
            "max_structural_drift_at_85pct": float(max_drift_competent),
            "pareto_front": [
                {
                    "generation": c.generation,
                    "structural_distance": float(c.structural_distance),
                    "task_performance": float(c.task_performance),
                    "total_synops": int(c.total_synops),
                    "dales_valid": bool(c.dales_valid),
                }
                for c in pareto_sorted
            ]
        }
    }

    out_file = os.path.join(results_dir, "phase2_remodeling_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[✓] Saved Phase 2 remodeling results to: {out_file}")

    return results_payload


if __name__ == "__main__":
    run_phase2_experiments()
