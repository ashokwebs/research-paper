"""
COMPUTATIONAL COUPLING — Dynamic Biophysical Motif Benchmark
Evaluates Level 3 of the Cross-Scale Motif Framework:
  1. Coherent Feedforward Loop (cFFL): Persistence detection & high-frequency noise filtering
  2. Incoherent Feedforward Loop (iFFL): Adaptive pulse generation & fold-change detection
  3. Reciprocal Inhibitory Pair (RIP): Bistable switching & winner-take-all attractor dynamics
  4. Bi-Fan Hub: Combinatorial coincidence gating
Compares performance, latency, and SynOps against uncoupled null configurations.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connectome.unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata
from src.simulation.lif_simulator import VectorizedLIFSimulator, SimulationConfig


def benchmark_cffl_persistence_detection():
    """
    Benchmark Coherent FFL (u -> v -> w, u -> w, all excitatory)
    against a simple linear chain (u -> v -> w) on noise rejection vs sustained signal.
    """
    print("[*] Benchmarking cFFL Persistence Detection & Noise Filtering...")

    # Build cFFL graph
    g_cffl = ConnectomeGraph("cFFL_Circuit", "Synthetic", "motif")
    for i, name in enumerate(["u_sensor", "v_inter", "w_output"]):
        g_cffl.add_neuron(NeuronMetadata(i, name, "excitatory", sign=1))

    # u->v, v->w, u->w (subthreshold individual weights, superthreshold combined)
    g_cffl.add_synapse(SynapseMetadata(0, 1, weight=18.0, sign=1, delay_ms=2.0))
    g_cffl.add_synapse(SynapseMetadata(1, 2, weight=18.0, sign=1, delay_ms=2.0))
    g_cffl.add_synapse(SynapseMetadata(0, 2, weight=18.0, sign=1, delay_ms=4.0))

    # Build linear chain control (lacks direct u->w feedforward shortcut)
    g_chain = ConnectomeGraph("Chain_Control", "Synthetic", "motif")
    for i, name in enumerate(["u_sensor", "v_inter", "w_output"]):
        g_chain.add_neuron(NeuronMetadata(i, name, "excitatory", sign=1))
    g_chain.add_synapse(SynapseMetadata(0, 1, weight=18.0, sign=1, delay_ms=2.0))
    g_chain.add_synapse(SynapseMetadata(1, 2, weight=18.0, sign=1, delay_ms=2.0))

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=15.0)
    sim_cffl = VectorizedLIFSimulator(g_cffl, config=cfg)
    sim_chain = VectorizedLIFSimulator(g_chain, config=cfg)

    # Test 1: Transient noise spike (5 ms pulse)
    # Test 2: Sustained stimulus (30 ms pulse)
    time_steps = 200  # 100 ms total
    noise_stim = np.zeros((time_steps, 3), dtype=np.float32)
    noise_stim[20:30, 0] = 35.0  # 5 ms transient burst at u

    sustained_stim = np.zeros((time_steps, 3), dtype=np.float32)
    sustained_stim[20:80, 0] = 35.0  # 30 ms sustained drive at u

    res_cffl_noise = sim_cffl.run(100.0, noise_stim)
    res_cffl_sust = sim_cffl.run(100.0, sustained_stim)

    res_chain_noise = sim_chain.run(100.0, noise_stim)
    res_chain_sust = sim_chain.run(100.0, sustained_stim)

    # Output spikes at w (neuron 2)
    cffl_noise_spikes = int(res_cffl_noise.spike_raster[:, 2].sum())
    cffl_sust_spikes = int(res_cffl_sust.spike_raster[:, 2].sum())

    chain_noise_spikes = int(res_chain_noise.spike_raster[:, 2].sum())
    chain_sust_spikes = int(res_chain_sust.spike_raster[:, 2].sum())

    print(f"    cFFL:  Noise Response = {cffl_noise_spikes} spikes | Sustained Response = {cffl_sust_spikes} spikes")
    print(f"    Chain: Noise Response = {chain_noise_spikes} spikes | Sustained Response = {chain_sust_spikes} spikes")

    return {
        "motif": "cFFL",
        "cffl_noise_rejection": bool(cffl_noise_spikes == 0 and cffl_sust_spikes > 0),
        "cffl_sustained_spikes": cffl_sust_spikes,
        "cffl_synops": int(res_cffl_sust.total_synops),
        "chain_sustained_spikes": chain_sust_spikes,
        "chain_synops": int(res_chain_sust.total_synops),
    }


def benchmark_iffl_adaptation():
    """
    Benchmark Incoherent FFL (u -> w (+), u -> v (+), v -> w (-))
    evaluating adaptive pulse generation and steady-state suppression under continuous step drive.
    """
    print("[*] Benchmarking iFFL Adaptive Pulse Generation...")

    g_iffl = ConnectomeGraph("iFFL_Circuit", "Synthetic", "motif")
    g_iffl.add_neuron(NeuronMetadata(0, "u_input", "excitatory", sign=1))
    g_iffl.add_neuron(NeuronMetadata(1, "v_interneuron", "inhibitory", sign=-1))
    g_iffl.add_neuron(NeuronMetadata(2, "w_output", "excitatory", sign=1))

    # u -> w direct fast excitation (delay 1.0 ms)
    g_iffl.add_synapse(SynapseMetadata(0, 2, weight=25.0, sign=1, delay_ms=1.0))
    # u -> v excitation (delay 1.0 ms)
    g_iffl.add_synapse(SynapseMetadata(0, 1, weight=25.0, sign=1, delay_ms=1.0))
    # v -> w delayed feedforward inhibition (delay 2.5 ms, strong inhibitory weight)
    g_iffl.add_synapse(SynapseMetadata(1, 2, weight=35.0, sign=-1, delay_ms=2.5))

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=20.0)
    sim_iffl = VectorizedLIFSimulator(g_iffl, config=cfg)

    # Continuous step stimulus for 100 ms
    time_steps = 200
    step_stim = np.zeros((time_steps, 3), dtype=np.float32)
    step_stim[20:180, 0] = 40.0  # Sustained input to u

    res = sim_iffl.run(100.0, step_stim)

    # Peak response in transient window (20..60 ms) vs steady state (60..100 ms)
    transient_spikes = int(res.spike_raster[20:80, 2].sum())
    steady_spikes = int(res.spike_raster[80:180, 2].sum())
    adaptation_index = (transient_spikes - steady_spikes) / max(1, transient_spikes + steady_spikes)

    print(f"    iFFL: Transient Spikes = {transient_spikes} | Steady-State Spikes = {steady_spikes} | Adaptation Index = {adaptation_index:.2f}")

    return {
        "motif": "iFFL",
        "transient_spikes": transient_spikes,
        "steady_state_spikes": steady_spikes,
        "adaptation_index": float(adaptation_index),
        "total_synops": int(res.total_synops),
    }


def benchmark_reciprocal_inhibition_bistability():
    """
    Benchmark Reciprocal Inhibitory Pair (A <-> B, both inhibitory)
    evaluating bistable categorical switching and state maintenance.
    """
    print("[*] Benchmarking Reciprocal Inhibitory Pair (RIP) Bistable Switching...")

    g_rip = ConnectomeGraph("RIP_Circuit", "Synthetic", "motif")
    g_rip.add_neuron(NeuronMetadata(0, "Neuron_A", "inhibitory", sign=-1))
    g_rip.add_neuron(NeuronMetadata(1, "Neuron_B", "inhibitory", sign=-1))

    # Mutual inhibition
    g_rip.add_synapse(SynapseMetadata(0, 1, weight=30.0, sign=-1, delay_ms=1.5))
    g_rip.add_synapse(SynapseMetadata(1, 0, weight=30.0, sign=-1, delay_ms=1.5))

    cfg = SimulationConfig(dt_ms=0.5, v_thresh=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m_ms=25.0)
    sim_rip = VectorizedLIFSimulator(g_rip, config=cfg)

    # Two-phase stimulus test:
    # Phase 1: Input drive to A (t=5..30 ms), B un-driven
    # Phase 2: Input drive to B (t=75..100 ms), A un-driven
    time_steps = 300  # 150 ms
    stim = np.zeros((time_steps, 2), dtype=np.float32)
    stim[10:60, 0] = 60.0    # Input to A
    stim[150:200, 1] = 60.0  # Input to B

    res = sim_rip.run(150.0, stim)

    # Spikes in Phase 1 (steps 0..140): Expect A active, B suppressed
    phase1_A = int(res.spike_raster[:140, 0].sum())
    phase1_B = int(res.spike_raster[:140, 1].sum())

    # Spikes in Phase 2 (steps 140..300): Expect B active, A suppressed
    phase2_A = int(res.spike_raster[140:, 0].sum())
    phase2_B = int(res.spike_raster[140:, 1].sum())

    switch_success = (phase1_A > phase1_B) and (phase2_B > phase2_A)
    print(f"    Phase 1 (A-state): A={phase1_A}, B={phase1_B} | Phase 2 (B-state): A={phase2_A}, B={phase2_B} | Switch Valid: {switch_success}")

    return {
        "motif": "RIP",
        "phase1_winner": "A" if phase1_A > phase1_B else "Tie",
        "phase2_winner": "B" if phase2_B > phase2_A else "Tie",
        "bistable_switch_successful": bool(switch_success),
        "total_synops": int(res.total_synops),
    }


def run_all_dynamic_benchmarks():
    results = {}
    results["cFFL"] = benchmark_cffl_persistence_detection()
    results["iFFL"] = benchmark_iffl_adaptation()
    results["RIP"] = benchmark_reciprocal_inhibition_bistability()

    out_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "dynamic_motif_benchmark_results.json"
    )
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved dynamic motif benchmark results to: {out_file}")
    return results


if __name__ == "__main__":
    run_all_dynamic_benchmarks()
