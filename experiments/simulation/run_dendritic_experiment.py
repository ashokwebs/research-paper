"""
COMPUTATIONAL COUPLING — Active Dendritic Non-Linearity vs Point Neuron LIF Benchmark
Evaluates how two-compartment active dendritic integration with NMDA supralinear plateaus
and perisomatic inhibition affects noise immunity, spurious somatic spiking, and SynOps dissipation
compared to standard single-compartment LIF point neurons.
Evaluated across C. elegans, Drosophila CX, and Mouse V1 L2/3 connectomes.
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
from src.simulation.dendritic_lif_simulator import TwoCompartmentLIFSimulator, TwoCompartmentConfig


def extract_standardized_subcircuit(graph: ConnectomeGraph, target_size: int = 80) -> ConnectomeGraph:
    """Extract top connected neurons for standardized biophysical comparison."""
    degs = sorted(graph.graph.degree(), key=lambda x: x[1], reverse=True)
    top_nodes = set(x[0] for x in degs[:target_size])
    return graph.get_subcircuit(top_nodes, name_suffix=f"top_{target_size}")


def run_dendritic_experiment():
    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Two-Compartment Active Dendritic vs Point LIF Benchmark")
    print("=" * 80)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load biological circuits
    print("\n[+] Loading Connectome Subcircuits...")
    g_ce = extract_standardized_subcircuit(load_celegans_connectome(), 80)
    g_cx = extract_standardized_subcircuit(load_drosophila_subcircuit("cx", min_synapses=3), 80)
    g_mouse = extract_standardized_subcircuit(load_mouse_v1_connectome(min_synapses=1), 80)

    circuits = {
        "C. elegans": g_ce,
        "Drosophila CX": g_cx,
        "Mouse V1 L2/3": g_mouse,
    }

    # Simulation parameters
    duration_ms = 100.0
    dt_ms = 0.5
    time_steps = int(duration_ms / dt_ms)  # 200 steps

    noise_rates = [0.0, 5.0, 15.0, 30.0, 50.0]  # Spurious noise current amplitude
    experiment_records = []

    for name, circuit in circuits.items():
        N = circuit.num_neurons
        print(f"\n[*] Evaluating Circuit: {name} (N={N}, E={circuit.num_synapses})")

        # Configure simulators
        lif_cfg = SimulationConfig(dt_ms=dt_ms, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=20.0)
        point_sim = VectorizedLIFSimulator(circuit, config=lif_cfg)

        dend_cfg = TwoCompartmentConfig(
            dt_ms=dt_ms,
            v_thresh_soma=-50.0,
            v_rest=-70.0,
            v_reset=-75.0,
            theta_dend=-58.0,
            g_nmda=25.0,
            g_c=1.2,
        )
        dend_sim = TwoCompartmentLIFSimulator(circuit, config=dend_cfg)

        # Sensory input subset: first 10 neurons
        n_in = 10

        for noise_amp in noise_rates:
            np.random.seed(42)
            stim = np.zeros((time_steps, N), dtype=np.float32)

            # Signal: two coordinated 20 ms pulses on sensory channels (t=15..35 ms and t=60..80 ms)
            stim[30:70, :n_in] = 75.0
            stim[120:160, :n_in] = 75.0

            # Distractor noise: continuous low-amplitude stochastic fluctuations across all neurons
            if noise_amp > 0:
                noise = np.random.normal(0, noise_amp, size=(time_steps, N)).astype(np.float32)
                stim += np.maximum(0.0, noise)

            # Run Point LIF
            res_point = point_sim.run(duration_ms, stim, record_voltages=False)

            # Run Two-Compartment Dendritic LIF
            res_dend = dend_sim.run(duration_ms, stim, record_voltages=False)

            # Metrics
            energy_ratio = (
                res_point.energy_proxy_nj / max(1e-6, res_dend.energy_proxy_nj)
                if res_dend.energy_proxy_nj > 0 else 1.0
            )
            synops_ratio = (
                res_point.total_synops / max(1, res_dend.total_synops)
                if res_dend.total_synops > 0 else 1.0
            )

            record = {
                "organism": name,
                "noise_amp": noise_amp,
                "point_soma_spikes": int(res_point.total_spikes),
                "point_synops": int(res_point.total_synops),
                "point_energy_nj": float(res_point.energy_proxy_nj),
                "dend_soma_spikes": int(res_dend.total_soma_spikes),
                "dend_plateaus": int(res_dend.total_dend_plateaus),
                "dend_synops": int(res_dend.total_synops),
                "dend_energy_nj": float(res_dend.energy_proxy_nj),
                "synops_saving_factor": float(synops_ratio),
                "energy_saving_factor": float(energy_ratio),
            }
            experiment_records.append(record)

            print(f"  Noise Amp {noise_amp:4.1f}: Point SynOps={res_point.total_synops:6d} ({res_point.energy_proxy_nj:7.2f} nJ) | "
                  f"Dend SynOps={res_dend.total_synops:6d} ({res_dend.energy_proxy_nj:7.2f} nJ) | "
                  f"Energy Reduction: {energy_ratio:5.2f}x")

    # Save results
    df = pd.DataFrame(experiment_records)
    csv_path = os.path.join(results_dir, "dendritic_vs_point_lif_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved dendritic comparison CSV to {csv_path}")

    json_path = os.path.join(results_dir, "dendritic_vs_point_lif_results.json")
    with open(json_path, "w") as f:
        json.dump(experiment_records, f, indent=2)
    print(f"[+] Saved dendritic comparison JSON to {json_path}")

    print("\n" + "=" * 80)
    print("DENDRITIC COMPARTMENTALIZATION EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_dendritic_experiment()
