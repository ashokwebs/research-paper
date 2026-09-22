"""
COMPUTATIONAL COUPLING — Multi-Scale Graph Translation Benchmark Runner
Evaluates scaling of invariant motif projection across network cardinalities (N=100 to N=2,000):
  - Preserves empirical E/I balance (2.69:1 from Google H01)
  - Tracks motif scaling (cFFL, iFFL, RIP)
  - Evaluates spectral radius and spectral gap stability
  - Computes theoretical Davis-Kahan bounds across scale
Saves to results/graph_translation_scaling.json and .csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connectome.graph_translator import run_cross_scale_graph_translation, ScaledNetworkMetrics


def main():
    print("=" * 115)
    print("RUNNING MULTI-SCALE INVARIANT GRAPH TRANSLATION BENCHMARK")
    print("=" * 115)

    sizes = [100, 250, 500, 1000, 2000]
    metrics_list = run_cross_scale_graph_translation(sizes=sizes)
    os.makedirs("results", exist_ok=True)

    res_dicts = [
        {
            "num_neurons": m.num_neurons,
            "num_synapses": m.num_synapses,
            "density": m.density,
            "ei_ratio": m.ei_ratio,
            "num_cffl": m.num_cffl,
            "num_iffl": m.num_iffl,
            "num_rip": m.num_rip,
            "spectral_radius": m.spectral_radius,
            "spectral_gap": m.spectral_gap,
            "theoretical_r90_bound": m.theoretical_r90_bound,
            "estimated_synops_per_sample": m.estimated_synops_per_sample,
            "estimated_energy_nj": m.estimated_energy_nj,
        }
        for m in metrics_list
    ]

    with open("results/graph_translation_scaling.json", "w") as f:
        json.dump(res_dicts, f, indent=2)

    df = pd.DataFrame(res_dicts)
    df.to_csv("results/graph_translation_scaling.csv", index=False)
    print("Saved results to results/graph_translation_scaling.json and .csv")

    print("-" * 115)
    print(f"{'Neurons (N)':<12} | {'Synapses (E)':<13} | {'Density':<8} | {'cFFL':<8} | {'iFFL':<8} | {'RIP':<6} | {'Spec Rad':<9} | {'Theor R90':<10} | {'Est SynOps'}")
    print("-" * 115)
    for m in metrics_list:
        print(f"{m.num_neurons:<12} | {m.num_synapses:<13} | {m.density:>7.4f} | {m.num_cffl:>7} | {m.num_iffl:>7} | {m.num_rip:>5} | {m.spectral_radius:>8.2f}  | {m.theoretical_r90_bound:>9.5f} | {m.estimated_synops_per_sample:>9.1f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
