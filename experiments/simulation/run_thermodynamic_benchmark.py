"""
COMPUTATIONAL COUPLING — Thermodynamic & Landauer Efficiency Benchmark Runner
Executes information-theoretic analysis and Landauer bound proximity:
  - Shannon Mutual Information I(Y; Y_hat)
  - Dynamic Energy Dissipation (nJ)
  - Bits per Nanojoule and Bits per Joule
  - Landauer Proximity Ratio
Saves to results/thermodynamic_scaling_benchmark.json and .csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.theory.thermodynamics import (
    run_thermodynamic_cross_architecture_comparison,
    evaluate_thermodynamic_efficiency,
    BOLTZMANN_CONSTANT,
    LANDAUER_LIMIT_JOULES,
    LANDAUER_LIMIT_NJ,
)


def main():
    print("=" * 115)
    print("RUNNING THERMODYNAMIC & LANDAUER EFFICIENCY BENCHMARK")
    print("=" * 115)

    results = run_thermodynamic_cross_architecture_comparison()
    os.makedirs("results", exist_ok=True)

    res_dicts = [
        {
            "model_name": m.model_name,
            "accuracy_pct": m.accuracy_pct,
            "mutual_information_bits": m.mutual_information_bits,
            "energy_proxy_nj": m.energy_proxy_nj,
            "energy_per_bit_nj": m.energy_per_bit_nj,
            "bits_per_nanojoule": m.bits_per_nanojoule,
            "bits_per_joule": m.bits_per_joule,
            "landauer_proximity_ratio": m.landauer_proximity_ratio,
            "energy_advantage_vs_dense": m.energy_advantage_vs_dense,
        }
        for m in results
    ]

    with open("results/thermodynamic_scaling_benchmark.json", "w") as f:
        json.dump(res_dicts, f, indent=2)

    df = pd.DataFrame(res_dicts)
    df.to_csv("results/thermodynamic_scaling_benchmark.csv", index=False)
    print("Saved results to results/thermodynamic_scaling_benchmark.json and .csv")

    print("-" * 115)
    print(f"{'Model Architecture':<32} | {'Acc (%)':<7} | {'MI (bits)':<9} | {'Energy (nJ)':<11} | {'Bits / nJ':<10} | {'Landauer Ratio':<15} | {'Advantage'}")
    print("-" * 115)
    for m in results:
        print(f"{m.model_name:<32} | {m.accuracy_pct:>6.2f}% | {m.mutual_information_bits:>8.3f}  | {m.energy_proxy_nj:>10.2f}  | {m.bits_per_nanojoule:>9.4f}  | {m.landauer_proximity_ratio:>14.2e}  | {m.energy_advantage_vs_dense:>8.1f}x")
    print("=" * 115)


if __name__ == "__main__":
    main()
