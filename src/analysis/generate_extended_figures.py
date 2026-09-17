"""
COMPUTATIONAL COUPLING — Extended Publication Figures Generator (Figures 6 - 9)
Generates publication-quality 300-DPI vector-ready plots:
  - Figure 6: Cross-Species Degeneracy Frontier & Invariant Segregation
  - Figure 7: Stress Resilience Landscape (Temporal Jitter, Noise, Channel Dropout)
  - Figure 8: Active Dendritic Compartmentalization vs Point Neuron LIF
  - Figure 9: Long-Horizon Delayed-Match-to-Sample Working Memory
Saves to results/figures/, paper/figures/, and artifact directory.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
PAPER_FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper", "figures")
ARTIFACT_DIR = "/home/charizard/.gemini/antigravity-cli/brain/a942bac5-f639-4fc9-b377-70bb4e1abbe0"

os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


def save_fig(fig, filename: str):
    """Save high-resolution figures to results, paper, and artifact directories."""
    paths = [
        os.path.join(RESULTS_DIR, "figures", filename),
        os.path.join(PAPER_FIG_DIR, filename),
        os.path.join(ARTIFACT_DIR, filename),
    ]
    for p in paths:
        try:
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"[+] Saved figure: {p}")
        except Exception as e:
            print(f"[-] Could not save to {p}: {e}")
    plt.close(fig)


def generate_figure_6():
    """Figure 6: Cross-Species Degeneracy Frontier & Causal Invariant Segregation."""
    detailed_path = os.path.join(RESULTS_DIR, "cross_species_degeneracy_detailed.json")
    if not os.path.exists(detailed_path):
        print("[-] Skipping Fig 6: data not found")
        return

    with open(detailed_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Pareto Frontier Trajectories
    ax = axes[0]
    palette = {"C. elegans": "#1f77b4", "Drosophila CX": "#ff7f0e", "Mouse V1 L2/3": "#2ca02c"}
    markers = {"C. elegans": "o", "Drosophila CX": "s", "Mouse V1 L2/3": "^"}

    for species, front in data["pareto_fronts"].items():
        d_struct = [c["structural_distance"] for c in front]
        f_task = [c["task_performance"] for c in front]
        # Sort by structural distance
        sorted_idx = np.argsort(d_struct)
        ax.plot(
            np.array(d_struct)[sorted_idx],
            np.array(f_task)[sorted_idx],
            marker=markers.get(species, "o"),
            color=palette.get(species, "blue"),
            linewidth=2,
            label=species,
            markersize=6,
        )

    ax.set_xlabel("Structural Drift $\\Delta_{\\mathrm{struct}}(G_0, G')$")
    ax.set_ylabel("Normalized Task Competence $F_{\\mathrm{task}}$")
    ax.set_title("A. Cross-Species Degeneracy Frontiers")
    ax.axhline(0.90, color="gray", linestyle="--", alpha=0.6, label="90% Retention Threshold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower left", frameon=True)

    # Panel B: SynOps Reduction Along the Frontier
    ax = axes[1]
    deg_summary = data["degeneracy_summary"]
    species_names = [d["organism"] for d in deg_summary]
    synops_red = [d["synops_reduction_pct"] for d in deg_summary]
    crit_r90 = [d["critical_radius_r90"] for d in deg_summary]

    x = np.arange(len(species_names))
    width = 0.35
    b1 = ax.bar(x - width/2, synops_red, width, label="SynOps Reduction (%)", color="#3498db")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + width/2, crit_r90, width, label="Critical Radius $R_{90}$", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(species_names)
    ax.set_ylabel("Max SynOps Reduction (%)", color="#3498db")
    ax2.set_ylabel("Critical Drift Radius $R_{90}$", color="#e74c3c")
    ax.set_title("B. Remodeling Capacity & Energy Gain")
    ax.set_ylim(0, 70)
    ax2.set_ylim(0, 0.35)
    ax.grid(False)

    # Panel C: Edge Neutrality / Scaffolding Distribution
    ax = axes[2]
    lesion_summary = data["lesion_summary"]
    scaffolding_pcts = [d["scaffolding_pct"] for d in lesion_summary]
    regulatory_pcts = [d["regulatory_pct"] for d in lesion_summary]
    invariants_pcts = [d["invariants_pct"] for d in lesion_summary]

    ax.bar(species_names, scaffolding_pcts, label="Neutral Scaffolding (<2% drop)", color="#2ecc71")
    ax.bar(species_names, regulatory_pcts, bottom=scaffolding_pcts, label="Regulatory (2-30% drop)", color="#f39c12")
    ax.bar(species_names, invariants_pcts, bottom=np.array(scaffolding_pcts)+np.array(regulatory_pcts), label="Invariants (>30% drop)", color="#e74c3c")

    ax.set_ylabel("Fraction of Evaluated Synapses (%)")
    ax.set_title("C. Single-Edge Causal Ablation Profile")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    save_fig(fig, "figure6_cross_species_degeneracy.png")


def generate_figure_7():
    """Figure 7: Stress Resilience Landscape Under Adverse Conditions."""
    j_path = os.path.join(RESULTS_DIR, "stress_jitter_sweep.csv")
    n_path = os.path.join(RESULTS_DIR, "stress_noise_sweep.csv")
    d_path = os.path.join(RESULTS_DIR, "stress_dropout_sweep.csv")

    if not (os.path.exists(j_path) and os.path.exists(n_path) and os.path.exists(d_path)):
        print("[-] Skipping Fig 7: stress datasets not found")
        return

    df_j = pd.read_csv(j_path)
    df_n = pd.read_csv(n_path)
    df_d = pd.read_csv(d_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    palette = {
        "CoupledEventNet (Full)": "#2ecc71",
        "Dense SNN": "#e74c3c",
        "Connectome Reservoir": "#3498db",
        "CEN w/o cFFL": "#9b59b6",
        "CEN w/o RIP": "#e67e22",
    }

    # Panel A: Temporal Jitter
    ax = axes[0, 0]
    for model, group in df_j.groupby("model"):
        ax.plot(group["jitter_sigma_ms"], group["accuracy"], marker="o", linewidth=2, label=model, color=palette.get(model))
    ax.set_xlabel("Temporal Spike Jitter $\\sigma_t$ (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A. Robustness to Temporal Spike Jitter")
    ax.set_ylim(90, 102)
    ax.legend(loc="lower left", frameon=True)

    # Panel B: Poisson Distractor Noise
    ax = axes[0, 1]
    for model, group in df_n.groupby("model"):
        ax.plot(group["noise_prob"] * 100, group["accuracy"], marker="s", linewidth=2, label=model, color=palette.get(model))
    ax.set_xlabel("Background Noise Probability (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B. Robustness to Poisson Background Noise")
    ax.set_ylim(50, 105)

    # Panel C: Synaptic Channel Dropout
    ax = axes[1, 0]
    for model, group in df_d.groupby("model"):
        ax.plot(group["dropout_prob"] * 100, group["accuracy"], marker="^", linewidth=2, label=model, color=palette.get(model))
    ax.set_xlabel("Synaptic Dropout Probability $p_{\\mathrm{fail}}$ (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("C. Robustness to Synaptic Transmission Failure")
    ax.set_ylim(85, 102)

    # Panel D: Energy Dissipation Proxy (Log scale)
    ax = axes[1, 1]
    # Use clean baseline energy
    clean_j = df_j[df_j["jitter_sigma_ms"] == 0.0]
    models = clean_j["model"].tolist()
    energies = [max(1e-2, e) for e in clean_j["energy_nj"].tolist()]
    colors = [palette.get(m, "gray") for m in models]

    bars = ax.barh(models, energies, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Synaptic Energy Proxy per Sample (nJ, log scale)")
    ax.set_title("D. Baseline Energy Dissipation Under Clean Signal")

    for bar, val in zip(bars, energies):
        ax.text(val * 1.3, bar.get_y() + bar.get_height()/2, f"{val:.2f} nJ", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "figure7_stress_resilience.png")


def generate_figure_8():
    """Figure 8: Active Dendritic Compartmentalization vs Point Neuron LIF."""
    csv_path = os.path.join(RESULTS_DIR, "dendritic_vs_point_lif_results.csv")
    if not os.path.exists(csv_path):
        print("[-] Skipping Fig 8: dendritic results not found")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    palette = {"C. elegans": "#1f77b4", "Drosophila CX": "#ff7f0e", "Mouse V1 L2/3": "#2ca02c"}

    # Panel A: Somatic Spikes vs Noise Amplitude
    ax = axes[0]
    for org, group in df.groupby("organism"):
        ax.plot(group["noise_amp"], group["point_soma_spikes"], linestyle="--", marker="x", color=palette[org], label=f"{org} (Point LIF)")
        ax.plot(group["noise_amp"], group["dend_soma_spikes"], linestyle="-", marker="o", color=palette[org], linewidth=2, label=f"{org} (Active Dend)")
    ax.set_xlabel("Noise Current Amplitude (mV)")
    ax.set_ylabel("Total Somatic Spikes")
    ax.set_title("A. Spurious Somatic Spikes Under Noise")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    # Panel B: SynOps Dissipation vs Noise
    ax = axes[1]
    for org, group in df.groupby("organism"):
        ax.plot(group["noise_amp"], group["point_synops"], linestyle="--", marker="x", color=palette[org])
        ax.plot(group["noise_amp"], group["dend_synops"], linestyle="-", marker="s", color=palette[org], linewidth=2)
    ax.set_xlabel("Noise Current Amplitude (mV)")
    ax.set_ylabel("Total Synaptic Operations (SynOps)")
    ax.set_title("B. Synaptic Operations (Dashed: Point, Solid: Dend)")
    ax.set_yscale("log")

    # Panel C: Energy Reduction Factor (Point / Dend)
    ax = axes[2]
    for org, group in df.groupby("organism"):
        ax.plot(group["noise_amp"], group["energy_saving_factor"], marker="^", color=palette[org], linewidth=2.5, label=org)
    ax.set_xlabel("Noise Current Amplitude (mV)")
    ax.set_ylabel("Energy Reduction Factor (Fold, $E_{\\mathrm{point}} / E_{\\mathrm{dend}}$)")
    ax.set_title("C. Energy Advantage of Dendritic Thresholding")
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_fig(fig, "figure8_dendritic_scaling.png")


def generate_figure_9():
    """Figure 9: Long-Horizon DMTS Working Memory Benchmark."""
    csv_path = os.path.join(RESULTS_DIR, "long_horizon_dmts_benchmark.csv")
    if not os.path.exists(csv_path):
        print("[-] Skipping Fig 9: DMTS dataset not yet ready")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    palette = {
        "CoupledEventNet (Full)": "#2ecc71",
        "Dense SNN": "#e74c3c",
        "Connectome Reservoir": "#3498db",
        "CEN w/o RIP": "#e67e22",
        "CEN w/o cFFL": "#9b59b6",
    }
    colors = [palette.get(m, "gray") for m in df["model"]]

    # Panel A: Accuracy Across 3 Seeds
    ax = axes[0]
    bars = ax.bar(df["model"], df["mean_accuracy"], yerr=df["std_accuracy"], capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("A. Working Memory Retention (150 steps, 95-step Distractor Delay)")
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylim(40, 105)

    for bar, acc, std in zip(bars, df["mean_accuracy"], df["std_accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, acc + std + 1.5, f"{acc:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Panel B: SynOps vs Energy Proxy
    ax = axes[1]
    bars2 = ax.bar(df["model"], df["mean_synops_per_sample"], color=colors, alpha=0.85)
    ax.set_ylabel("Synaptic Operations per Sample")
    ax.set_title("B. Hardware SynOps During Working Memory Retention")
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_yscale("log")

    plt.tight_layout()
    save_fig(fig, "figure9_dmts_benchmark.png")


if __name__ == "__main__":
    generate_figure_6()
    generate_figure_7()
    generate_figure_8()
    generate_figure_9()
