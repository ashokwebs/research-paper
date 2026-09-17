"""
COMPUTATIONAL COUPLING — Figure 14 & Figure 15 Generator (300 DPI)
Generates publication-quality figures:
  - Figure 14: Continual Learning & Catastrophic Forgetting under Motif-Gated Plasticity (DCMG-STDP)
  - Figure 15: Thermodynamic Landauer Proximity & Cross-Scale Invariant Graph Translation
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DARK_BLUE = "#0f2350"
BRAND_BLUE = "#1e5ab4"
ACCENT_RED = "#d9381e"
ACCENT_GREEN = "#2ca02c"
PURPLE = "#7b1fa2"
GOLD = "#d4af37"
GRAY = "#7f8c8d"


def plot_figure_14(continual_data_path: str, save_paths: list[str]):
    """Figure 14: Continual Learning & Catastrophic Forgetting Benchmark."""
    with open(continual_data_path, "r") as f:
        data = json.load(f)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # Panel A: Task A Initial vs Retained Accuracy
    ax_a = fig.add_subplot(gs[0, 0])
    methods = [
        "DCMG-STDP\n(Ours)",
        "Unconstrained\nSTDP",
        "Dense SNN\n(BPTT)",
        "Connectome\nReservoir",
    ]
    x = np.arange(len(methods))
    width = 0.35

    init_acc = [d["task_a_initial_acc"] for d in data]
    ret_acc = [d["task_a_retained_acc"] for d in data]

    rects1 = ax_a.bar(x - width/2, init_acc, width, label="Task A Initial", color=BRAND_BLUE, alpha=0.85)
    rects2 = ax_a.bar(x + width/2, ret_acc, width, label="Task A Retained", color=ACCENT_RED, alpha=0.85)

    ax_a.set_ylabel("Classification Accuracy (%)")
    ax_a.set_title("(A) Task A Retention After Sequential Learning")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(methods, fontsize=8)
    ax_a.set_ylim(0, 110)
    ax_a.axhline(50, color="gray", linestyle="--", alpha=0.6, label="Chance Level (50%)")
    ax_a.legend(loc="lower right", frameon=True)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")

    for rect in rects1:
        h = rect.get_height()
        ax_a.text(rect.get_x() + rect.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax_a.text(rect.get_x() + rect.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: Catastrophic Forgetting Index (%)
    ax_b = fig.add_subplot(gs[0, 1])
    forgetting = [d["forgetting_pct"] for d in data]
    colors_f = [ACCENT_GREEN, ACCENT_GREEN, ACCENT_RED, ACCENT_GREEN]
    bars_b = ax_b.bar(methods, forgetting, color=colors_f, alpha=0.85, width=0.55)

    ax_b.set_ylabel("Catastrophic Forgetting (% Acc Loss)")
    ax_b.set_title("(B) Catastrophic Forgetting Metric (Lower is Better)")
    ax_b.set_xticklabels(methods, fontsize=8)
    ax_b.set_ylim(-2, 55)
    ax_b.grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars_b:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel C: Backward Transfer (BWT %)
    ax_c = fig.add_subplot(gs[1, 0])
    bwt = [d["backward_transfer_pct"] for d in data]
    colors_c = [ACCENT_GREEN if b >= 0 else ACCENT_RED for b in bwt]
    bars_c = ax_c.bar(methods, bwt, color=colors_c, alpha=0.85, width=0.55)

    ax_c.set_ylabel("Backward Transfer (% BWT)")
    ax_c.set_title("(C) Backward Transfer (Retained - Initial)")
    ax_c.set_xticklabels(methods, fontsize=8)
    ax_c.axhline(0, color="black", linewidth=1.0)
    ax_c.set_ylim(-55, 10)
    ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars_c:
        h = bar.get_height()
        y_pos = h - 4.5 if h < 0 else h + 1.0
        ax_c.text(bar.get_x() + bar.get_width()/2., y_pos, f"{h:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel D: Task B Accuracy vs Energy Proxy (nJ)
    ax_d = fig.add_subplot(gs[1, 1])
    task_b_acc = [d["task_b_acc"] for d in data]
    energy = [d["energy_proxy_nj"] for d in data]

    scatter_colors = [ACCENT_GREEN, BRAND_BLUE, ACCENT_RED, PURPLE]
    for i, m in enumerate(methods):
        ax_d.scatter(energy[i], task_b_acc[i], s=180, color=scatter_colors[i], label=m.replace("\n", " "), zorder=4)
        ax_d.text(energy[i] * 1.08, task_b_acc[i] - 0.8, m.replace("\n", " "), fontsize=8)

    ax_d.set_xscale("log")
    ax_d.set_xlabel("Dynamic Energy Proxy (nJ, log scale)")
    ax_d.set_ylabel("Task B Accuracy (%)")
    ax_d.set_title("(D) Task B Acquisition vs Energy Proxy")
    ax_d.set_ylim(85, 102)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Figure 14: Continual Learning Without Catastrophic Forgetting via DCMG-STDP", fontsize=13, fontweight="bold", y=0.98)
    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300)
    plt.close(fig)
    print("Saved Figure 14 to", save_paths)


def plot_figure_15(thermo_path: str, scaling_path: str, save_paths: list[str]):
    """Figure 15: Thermodynamic Landauer Proximity & Cross-Scale Graph Translation."""
    with open(thermo_path, "r") as f:
        t_data = json.load(f)
    with open(scaling_path, "r") as f:
        s_data = json.load(f)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # Panel A: Bits per Nanojoule Efficiency
    ax_a = fig.add_subplot(gs[0, 0])
    models = ["AD-CEN\n(Ours)", "CEN\n(Ours)", "Connectome\nReservoir", "Dense SNN\n(Standard)"]
    bits_per_nj = [d["bits_per_nanojoule"] for d in t_data[:4]]
    colors_a = [ACCENT_GREEN, BRAND_BLUE, PURPLE, ACCENT_RED]

    bars_a = ax_a.bar(models, bits_per_nj, color=colors_a, alpha=0.85, width=0.55)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("Thermodynamic Efficiency (Bits / nJ, log)")
    ax_a.set_title("(A) Information Transmitted per Unit Energy")
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars_a:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., h * 1.3, f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: Proximity to Landauer Thermodynamic Bound
    ax_b = fig.add_subplot(gs[0, 1])
    all_models = ["AD-CEN", "CEN", "conn2res", "Dense SNN", "Jetson GPU"]
    landauer_ratios = [d["landauer_proximity_ratio"] for d in t_data]
    colors_b = [ACCENT_GREEN, BRAND_BLUE, PURPLE, ACCENT_RED, GOLD]

    bars_b = ax_b.bar(all_models, landauer_ratios, color=colors_b, alpha=0.85, width=0.55)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("Distance to Landauer Bound (x k_B T ln 2)")
    ax_b.set_title("(B) Thermodynamic Landauer Distance (Lower is Better)")
    ax_b.grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars_b:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h * 1.5, f"{h:.1e}", ha="center", va="bottom", fontsize=7)

    # Panel C: Multi-Scale Graph Cardinality Scaling (N=100 to N=2,000)
    ax_c = fig.add_subplot(gs[1, 0])
    N_vals = [d["num_neurons"] for d in s_data]
    E_vals = [d["num_synapses"] for d in s_data]
    cffl_vals = [d["num_cffl"] for d in s_data]
    iffl_vals = [d["num_iffl"] for d in s_data]

    ax_c.plot(N_vals, E_vals, "o-", color=DARK_BLUE, label="Synapses (E)", linewidth=2.2)
    ax_c.plot(N_vals, cffl_vals, "s--", color=BRAND_BLUE, label="cFFL Motifs", linewidth=2.0)
    ax_c.plot(N_vals, iffl_vals, "^:", color=ACCENT_RED, label="iFFL Motifs", linewidth=2.0)

    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Number of Neurons (N, log scale)")
    ax_c.set_ylabel("Count (log scale)")
    ax_c.set_title("(C) Invariant Motif Projection Across Network Scales")
    ax_c.legend(loc="upper left", frameon=True)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # Panel D: Spectral Radius & Gap Stability Across Scale
    ax_d = fig.add_subplot(gs[1, 1])
    spec_rad = [d["spectral_radius"] for d in s_data]
    theor_r90 = [d["theoretical_r90_bound"] * 1000.0 for d in s_data]

    ax_d.plot(N_vals, spec_rad, "o-", color=ACCENT_GREEN, label="Spectral Radius (rho)", linewidth=2.2)
    ax_d.set_xlabel("Number of Neurons (N, log scale)")
    ax_d.set_ylabel("Spectral Radius", color=ACCENT_GREEN)
    ax_d.tick_params(axis="y", labelcolor=ACCENT_GREEN)
    ax_d.set_ylim(5, 12)
    ax_d.set_xscale("log")
    ax_d.grid(True, linestyle=":", alpha=0.5)

    ax_d2 = ax_d.twinx()
    ax_d2.plot(N_vals, theor_r90, "s--", color=PURPLE, label="Theor R90 (x10^-3)", linewidth=2.0)
    ax_d2.set_ylabel("Theoretical R90 (x 10^-3)", color=PURPLE)
    ax_d2.tick_params(axis="y", labelcolor=PURPLE)
    ax_d2.set_ylim(-0.05, 0.4)

    ax_d.set_title("(D) Spectral Radius & Dynamic Stability Across Scale")

    plt.suptitle("Figure 15: Thermodynamic Landauer Proximity & Cross-Scale Graph Scaling", fontsize=13, fontweight="bold", y=0.98)
    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300)
    plt.close(fig)
    print("Saved Figure 15 to", save_paths)


def main():
    thermo_path = "results/thermodynamic_scaling_benchmark.json"
    continual_path = "results/continual_learning_benchmark.json"
    scaling_path = "results/graph_translation_scaling.json"
    artifact_dir = "/home/charizard/.gemini/antigravity-cli/brain/a942bac5-f639-4fc9-b377-70bb4e1abbe0"

    f14_paths = [
        "results/figures/figure14_continual_learning_stdp.png",
        "paper/figures/figure14_continual_learning_stdp.png",
        os.path.join(artifact_dir, "figure14_continual_learning_stdp.png"),
    ]
    f15_paths = [
        "results/figures/figure15_thermodynamics_and_scaling.png",
        "paper/figures/figure15_thermodynamics_and_scaling.png",
        os.path.join(artifact_dir, "figure15_thermodynamics_and_scaling.png"),
    ]

    plot_figure_14(continual_path, f14_paths)
    plot_figure_15(thermo_path, scaling_path, f15_paths)


if __name__ == "__main__":
    main()
