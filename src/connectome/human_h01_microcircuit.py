"""
COMPUTATIONAL COUPLING — Human Neocortical Microcircuit (Google H01) Invariant Extraction
Analyzes empirical human synaptic distributions from Lichtman Lab / Google H01:
  - Laminar distribution of excitatory vs inhibitory synapses across Layers 1--6
  - Morphological segregation: Distal dendritic excitation vs perisomatic inhibition
  - Empirical E/I balance ratios and connectomic scaling laws in human cortex
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def analyze_h01_laminar_organization(
    h01_dir: str = "upstream/human/h01",
) -> Dict[str, Any]:
    """
    Extract laminar synaptic density and E/I ratios across human cortical layers.
    """
    csv_path = os.path.join(h01_dir, "goog14r0s5c3_e_i_ratios_feb_2023.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"H01 E/I csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Clean layer names
    df["clean_layer"] = df["cortical layer"].astype(str).str.strip()
    # Filter valid cortical layers
    valid_layers = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6"]
    df_cortical = df[df["clean_layer"].isin(valid_layers)].copy()

    laminar_summary = []
    for layer in valid_layers:
        sub = df_cortical[df_cortical["clean_layer"] == layer]
        if len(sub) == 0:
            continue

        n_cells = len(sub)
        total_exc = float(sub["excitatory synapses"].sum())
        total_inh = float(sub["inhibitory synapses"].sum())
        total_syn = total_exc + total_inh

        ei_ratio = total_exc / max(1.0, total_inh)
        exc_fraction = total_exc / max(1.0, total_syn)
        inh_fraction = total_inh / max(1.0, total_syn)

        mean_exc_per_cell = float(sub["excitatory synapses"].mean())
        mean_inh_per_cell = float(sub["inhibitory synapses"].mean())

        # Cell type counts
        cell_types = sub["cell type"].value_counts().to_dict()

        row = {
            "layer": layer,
            "neuron_count": n_cells,
            "total_exc_synapses": int(total_exc),
            "total_inh_synapses": int(total_inh),
            "total_synapses": int(total_syn),
            "ei_ratio": float(ei_ratio),
            "exc_fraction": float(exc_fraction),
            "inh_fraction": float(inh_fraction),
            "mean_exc_per_cell": float(mean_exc_per_cell),
            "mean_inh_per_cell": float(mean_inh_per_cell),
            "dominant_cell_type": max(cell_types, key=cell_types.get) if cell_types else "Unknown",
        }
        laminar_summary.append(row)

    # Global human cortical metrics
    overall_exc = sum(r["total_exc_synapses"] for r in laminar_summary)
    overall_inh = sum(r["total_inh_synapses"] for r in laminar_summary)
    global_ei_ratio = overall_exc / max(1.0, overall_inh)

    return {
        "dataset": "Google H01 Human Temporal Neocortex (1 mm^3)",
        "total_analyzed_neurons": sum(r["neuron_count"] for r in laminar_summary),
        "total_analyzed_synapses": overall_exc + overall_inh,
        "global_ei_ratio": global_ei_ratio,
        "laminar_profiles": laminar_summary,
    }


def save_h01_analysis_results(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    stats = analyze_h01_laminar_organization()

    df = pd.DataFrame(stats["laminar_profiles"])
    csv_path = os.path.join(output_dir, "human_h01_laminar_synapse_distribution.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved H01 Laminar Distribution CSV to {csv_path}")

    json_path = os.path.join(output_dir, "human_h01_laminar_synapse_distribution.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[+] Saved H01 Laminar Distribution JSON to {json_path}")

    print("\n" + "=" * 90)
    print("HUMAN H01 NEOCORTICAL LAMINAR SYNAPSE ANALYSIS")
    print("=" * 90)
    print(f"{'Layer':<10} | {'Neurons':<8} | {'Exc Syn':<10} | {'Inh Syn':<10} | {'E/I Ratio':<10} | {'Exc %':<8} | {'Inh %':<8}")
    print("-" * 90)
    for r in stats["laminar_profiles"]:
        print(f"{r['layer']:<10} | {r['neuron_count']:<8} | {r['total_exc_synapses']:<10} | {r['total_inh_synapses']:<10} | {r['ei_ratio']:<10.2f} | {r['exc_fraction']*100:<7.1f}% | {r['inh_fraction']*100:<7.1f}%")
    print("=" * 90)
    print(f"Overall Human Neocortical E/I Ratio: {stats['global_ei_ratio']:.2f} ({(stats['global_ei_ratio']/(1+stats['global_ei_ratio']))*100:.1f}% Excitatory)")
    print("=" * 90)


if __name__ == "__main__":
    save_h01_analysis_results()
