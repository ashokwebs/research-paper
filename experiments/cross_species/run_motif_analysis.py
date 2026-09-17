"""
COMPUTATIONAL COUPLING — Phase 1: Cross-Species Motif Discovery & Statistical Census
Executes comparative motif significance profiling across:
  - C. elegans (whole nervous system, Cook 2019)
  - Drosophila melanogaster (Antennal Lobe & Central Complex, FlyWire v630)
  - Mus musculus (Primary Visual Cortex V1 L2/3, MICrONS Pinky100)
  - Homo sapiens (HCP Visual Network & Whole-Brain Consensus 1,015)
Saves structured JSON and tabular CSV reports to results/.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connectome import (
    load_celegans_connectome,
    load_drosophila_subcircuit,
    load_mouse_v1_connectome,
    load_human_hcp_macroscale,
)
from src.motifs.motif_engine import MotifDiscoveryEngine, ConnectomeSignificanceReport


def run_cross_species_motif_census(n_null_samples: int = 25, seed: int = 42):
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("COMPUTATIONAL COUPLING — Phase 1: Cross-Species Motif Census")
    print(f"Configuration Null Samples per Circuit: {n_null_samples} | Seed: {seed}")
    print("=" * 80)

    datasets = [
        ("C. elegans (Whole Nervous System)", lambda: load_celegans_connectome()),
        ("Drosophila (FlyWire Central Complex)", lambda: load_drosophila_subcircuit("cx", min_synapses=5)),
        ("Drosophila (FlyWire Antennal Lobe)", lambda: load_drosophila_subcircuit("al", min_synapses=5)),
        ("Mouse (MICrONS Pinky100 V1 L2/3)", lambda: load_mouse_v1_connectome(min_synapses=1)),
        ("Human (HCP Macroscale Visual Network)", lambda: load_human_hcp_macroscale(network="VIS")),
        ("Human (HCP Macroscale Whole Brain)", lambda: load_human_hcp_macroscale()),
    ]

    all_reports = {}
    table_rows = []

    for label, loader_fn in datasets:
        print(f"\n[+] Loading and analyzing: {label}...")
        t0 = time.time()
        g = loader_fn()
        load_time = time.time() - t0
        print(f"    Graph loaded in {load_time:.2f}s: {g.num_neurons} neurons, {g.num_synapses} synapses")

        engine = MotifDiscoveryEngine(g)
        t1 = time.time()
        report = engine.compute_significance_profile(
            n_null_samples=n_null_samples,
            n_swaps_factor=3,
            seed=seed,
        )
        analysis_time = time.time() - t1
        print(f"    Motif census & null profiling completed in {analysis_time:.2f}s")
        print(report.summary_table())

        # Record for JSON
        rep_dict = {
            "graph_name": report.graph_name,
            "organism": report.organism,
            "num_neurons": report.num_neurons,
            "num_synapses": report.num_synapses,
            "density": float(g.density),
            "ei_ratio": float(g.summary_dict()["ei_ratio"]),
            "motifs": {
                m_name: {
                    "real_count": m_stats.real_count,
                    "null_mean": m_stats.null_mean,
                    "null_std": m_stats.null_std,
                    "z_score": m_stats.z_score,
                    "p_value": m_stats.p_value,
                    "significance_profile": m_stats.significance_profile,
                }
                for m_name, m_stats in report.motifs.items()
            }
        }
        all_reports[label] = rep_dict

        # Record for CSV
        for m_name, m_stats in report.motifs.items():
            table_rows.append({
                "dataset": label,
                "organism": report.organism,
                "num_neurons": report.num_neurons,
                "num_synapses": report.num_synapses,
                "density": float(g.density),
                "motif_name": m_name,
                "real_count": m_stats.real_count,
                "null_mean": m_stats.null_mean,
                "null_std": m_stats.null_std,
                "z_score": m_stats.z_score,
                "p_value": m_stats.p_value,
                "significance_profile": m_stats.significance_profile,
            })

    # Save to JSON
    json_path = os.path.join(results_dir, "motif_significance_cross_species.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n[✓] Saved structured JSON report to: {json_path}")

    # Save to CSV
    csv_path = os.path.join(results_dir, "motif_significance_cross_species.csv")
    df = pd.DataFrame(table_rows)
    df.to_csv(csv_path, index=False)
    print(f"[✓] Saved flat tabular CSV to: {csv_path}")

    return all_reports


if __name__ == "__main__":
    run_cross_species_motif_census(n_null_samples=25, seed=42)
