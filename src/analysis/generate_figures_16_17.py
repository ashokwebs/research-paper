"""
Script to generate publication-grade Figures 16 and 17 (300 DPI):
- Figure 16: Self-Organized Criticality and Neuronal Avalanche Dynamics
- Figure 17: Multi-Core Neuromorphic Network-on-Chip (NoC) Mesh Scalability
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
})

def generate_figure_16():
    """Generates Figure 16: Self-Organized Criticality & Avalanche Dynamics."""
    with open("results/criticality_avalanche_benchmark.json") as f:
        crit_data = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    
    # Color palette
    colors = {
        "ActiveDendriticCEN (Ours)": "#1f77b4",
        "CoupledEventNet (Ours)": "#2ca02c",
        "Degenerate Pruned (50% Knee)": "#ff7f0e",
        "Raw Connectome (C. elegans)": "#9467bd",
        "Dense Recurrent SNN": "#d62728",
        "Random Rewired Control": "#7f7f7f",
    }
    
    # -------------------------------------------------------------
    # Panel A: Branching Ratio Comparison
    # -------------------------------------------------------------
    ax = axes[0, 0]
    arch_names = [d["architecture"] for d in crit_data]
    sigmas = [d["branching_ratio"] for d in crit_data]
    errs = [d["branching_ratio_std"] for d in crit_data]
    bar_colors = [colors.get(a, "#333333") for a in arch_names]
    
    short_names = [
        "AD-CEN", "CEN", "Pruned 50%", "Raw C. eleg", "Dense SNN", "Erdos-Renyi"
    ]
    
    bars = ax.bar(short_names, sigmas, yerr=errs, capsize=4, color=bar_colors, alpha=0.85, edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Critical Boundary (sigma=1.0)')
    ax.axhspan(0.95, 1.05, color='red', alpha=0.10, label='Critical Regime [0.95, 1.05]')
    
    ax.set_ylabel("Branching Ratio (sigma)")
    ax.set_title("(A) Branching Ratio (Critical Edge-of-Chaos)", fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)
    
    for bar, sig in zip(bars, sigmas):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.03, f"{sig:.3f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
                
    # -------------------------------------------------------------
    # Panel B: Power-Law Avalanche Size Distribution P(S) ~ S^-tau
    # -------------------------------------------------------------
    ax = axes[0, 1]
    # Plot empirical fit curves
    s_vals = np.logspace(0, 4, 100)
    # Theory: tau = 1.50
    ax.plot(s_vals, 1e4 * s_vals**(-1.50), 'k--', linewidth=2.0, label='Theory: P(S) ~ S^-1.50')
    
    for d in crit_data:
        tau = d["tau_size_exponent"]
        if tau > 0.5:
            lbl = f"{d['architecture']} (tau={tau:.2f})"
            c = colors.get(d["architecture"], "#333")
            ax.plot(s_vals, 1e4 * s_vals**(-tau), color=c, linewidth=1.8, label=lbl)
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Avalanche Size S (Spike Count)")
    ax.set_ylabel("Probability Density P(S)")
    ax.set_title("(B) Neuronal Avalanche Size Scaling", fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_ylim(1e-4, 2e4)
    
    # -------------------------------------------------------------
    # Panel C: Power-Law Avalanche Duration Distribution P(T) ~ T^-alpha
    # -------------------------------------------------------------
    ax = axes[1, 0]
    t_vals = np.logspace(0, 3, 100)
    ax.plot(t_vals, 1e3 * t_vals**(-2.00), 'k--', linewidth=2.0, label='Theory: P(T) ~ T^-2.00')
    
    for d in crit_data:
        alpha = d["alpha_duration_exponent"]
        if alpha > 0.5:
            lbl = f"{d['architecture']} (alpha={alpha:.2f})"
            c = colors.get(d["architecture"], "#333")
            ax.plot(t_vals, 1e3 * t_vals**(-alpha), color=c, linewidth=1.8, label=lbl)
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Avalanche Duration T (Time Steps)")
    ax.set_ylabel("Probability Density P(T)")
    ax.set_title("(C) Neuronal Avalanche Duration Scaling", fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_ylim(1e-4, 2e3)
    
    # -------------------------------------------------------------
    # Panel D: Crackling Noise Scaling Exponent (gamma = (alpha-1)/(tau-1))
    # -------------------------------------------------------------
    ax = axes[1, 1]
    crit_subset = [d for d in crit_data if d["crackling_scaling_gamma"] > 0]
    names_sub = [s for s, d in zip(short_names, crit_data) if d["crackling_scaling_gamma"] > 0]
    gammas = [d["crackling_scaling_gamma"] for d in crit_subset]
    colors_sub = [colors.get(d["architecture"], "#333") for d in crit_subset]
    
    bars = ax.bar(names_sub, gammas, color=colors_sub, alpha=0.85, edgecolor='black')
    ax.axhline(1.84, color='darkgreen', linestyle='--', linewidth=1.5, label='Connectome Baseline (gamma=1.84)')
    ax.set_ylabel("Crackling Scaling Exponent (gamma)")
    ax.set_title("(D) Universal Crackling Noise Scaling Relation", fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)
    
    for bar, g in zip(bars, gammas):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.04, f"{g:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
                
    plt.tight_layout()
    
    os.makedirs("paper/figures", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    artifact_dir = "/home/charizard/.gemini/antigravity-cli/brain/a942bac5-f639-4fc9-b377-70bb4e1abbe0"
    
    p1 = "paper/figures/figure16_criticality_and_avalanches.png"
    p2 = "results/figures/figure16_criticality_and_avalanches.png"
    p3 = os.path.join(artifact_dir, "figure16_criticality_and_avalanches.png")
    
    fig.savefig(p1, dpi=300)
    fig.savefig(p2, dpi=300)
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 16 at {p1}, {p2}, and {p3}")


def generate_figure_17():
    """Generates Figure 17: Multi-Core Neuromorphic Network-on-Chip (NoC) Scalability."""
    with open("results/neuromorphic_noc_scaling.json") as f:
        noc_data = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    
    sizes = [128, 512, 1024, 2048]
    uniform_runs = [d for d in noc_data if d["strategy"] == "Uniform"]
    motif_runs = [d for d in noc_data if d["strategy"] == "Motif-Aware (Ours)"]
    
    # -------------------------------------------------------------
    # Panel A: Inter-Core NoC Traffic Ratio (%)
    # -------------------------------------------------------------
    ax = axes[0, 0]
    x = np.arange(len(sizes))
    width = 0.35
    
    u_ratio = [d["inter_core_ratio_pct"] for d in uniform_runs]
    m_ratio = [d["inter_core_ratio_pct"] for d in motif_runs]
    
    ax.bar(x - width/2, u_ratio, width, label='Uniform Partitioning', color='#d62728', alpha=0.85, edgecolor='black')
    ax.bar(x + width/2, m_ratio, width, label='Motif-Aware Partitioning (Ours)', color='#2ca02c', alpha=0.85, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={s}\n({d['mesh_dimension']})" for s, d in zip(sizes, uniform_runs)])
    ax.set_ylabel("Inter-Core Traffic Ratio (%)")
    ax.set_title("(A) Cross-Core Packet Spillover Ratio", fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)
    
    # -------------------------------------------------------------
    # Panel B: Mean Router Hops per Packet
    # -------------------------------------------------------------
    ax = axes[0, 1]
    u_hops = [d["mean_hops_per_packet"] for d in uniform_runs]
    m_hops = [d["mean_hops_per_packet"] for d in motif_runs]
    
    ax.plot(sizes, u_hops, 'o--', color='#d62728', linewidth=2.2, markersize=8, label='Uniform Partitioning')
    ax.plot(sizes, m_hops, 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='Motif-Aware Partitioning (Ours)')
    
    for s, uh, mh in zip(sizes, u_hops, m_hops):
        ax.annotate(f"{uh:.2f}", (s, uh), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9, fontweight='bold', color='#d62728')
        ax.annotate(f"{mh:.2f}", (s, mh), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, fontweight='bold', color='#2ca02c')
        
    ax.set_xscale('log', base=2)
    ax.set_xlabel("Network Size N (Total Neurons)")
    ax.set_ylabel("Mean Router Hops per Event")
    ax.set_title("(B) Mesh Router Traversal Hops", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.9)
    
    # -------------------------------------------------------------
    # Panel C: Packet Routing Latency (Clock Cycles)
    # -------------------------------------------------------------
    ax = axes[1, 0]
    u_lat = [d["mean_latency_cycles"] for d in uniform_runs]
    m_lat = [d["mean_latency_cycles"] for d in motif_runs]
    
    ax.bar(x - width/2, u_lat, width, label='Uniform Partitioning', color='#d62728', alpha=0.85, edgecolor='black')
    ax.bar(x + width/2, m_lat, width, label='Motif-Aware Partitioning (Ours)', color='#2ca02c', alpha=0.85, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={s}" for s in sizes])
    ax.set_ylabel("Latency (Clock Cycles)")
    ax.set_title("(C) Event Propagation Latency (Cycles)", fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.9)
    
    # -------------------------------------------------------------
    # Panel D: Dynamic NoC Energy Dissipation (nJ)
    # -------------------------------------------------------------
    ax = axes[1, 1]
    u_energy = [d["total_noc_energy_nj"] for d in uniform_runs]
    m_energy = [d["total_noc_energy_nj"] for d in motif_runs]
    
    ax.plot(sizes, u_energy, 'o--', color='#d62728', linewidth=2.2, markersize=8, label='Uniform NoC Energy')
    ax.plot(sizes, m_energy, 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='Motif-Aware NoC Energy (1.36x Savings)')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel("Network Size N (Total Neurons)")
    ax.set_ylabel("Dynamic NoC Energy (nJ)")
    ax.set_title("(D) Multi-Core Interconnect Energy Dissipation", fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    artifact_dir = "/home/charizard/.gemini/antigravity-cli/brain/a942bac5-f639-4fc9-b377-70bb4e1abbe0"
    p1 = "paper/figures/figure17_neuromorphic_noc_scaling.png"
    p2 = "results/figures/figure17_neuromorphic_noc_scaling.png"
    p3 = os.path.join(artifact_dir, "figure17_neuromorphic_noc_scaling.png")
    
    fig.savefig(p1, dpi=300)
    fig.savefig(p2, dpi=300)
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 17 at {p1}, {p2}, and {p3}")


if __name__ == "__main__":
    generate_figure_16()
    generate_figure_17()
