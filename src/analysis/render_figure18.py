"""
Renders Figure 18 (DVS 10-Gesture Benchmark) cleanly from saved benchmark results.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def render_figure18():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(root, "results", "dvs_gesture_benchmark.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    models_dict = {
        "ActiveDendriticCEN": {"color": "#1b9e77", "marker": "o"},
        "CoupledEventNet": {"color": "#d95f02", "marker": "s"},
        "Dense SNN": {"color": "#7570b3", "marker": "^"},
        "Connectome Reservoir": {"color": "#e7298a", "marker": "D"},
    }

    noise_rates = data["noise_sweep"]["rates"]
    noise_results = data["noise_sweep"]["accuracies"]
    jitter_levels = data["jitter_sweep"]["jitters_ms"]
    jitter_results = data["jitter_sweep"]["accuracies"]
    summary = {item["Model"]: item for item in data["summary"]}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Noise sweep
    ax_a = axes[0, 0]
    for name, cfg in models_dict.items():
        ax_a.plot(noise_rates, noise_results[name], label=name, color=cfg["color"], marker=cfg["marker"], linewidth=2.2, markersize=7)
    ax_a.set_title("A. Robustness to Dynamic Shot Noise (Dark Current)", fontsize=12, fontweight="bold", pad=10)
    ax_a.set_xlabel(r"Poisson Shot Noise Rate ($\lambda$ events/step/pixel)", fontsize=10.5)
    ax_a.set_ylabel("10-Gesture Classification Accuracy (%)", fontsize=10.5)
    ax_a.set_ylim(20, 105)
    ax_a.axhline(10.0, color="gray", linestyle=":", label="Chance Level (10%)")
    ax_a.legend(frameon=True, fontsize=9, loc="lower left")
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Jitter sweep
    ax_b = axes[0, 1]
    for name, cfg in models_dict.items():
        ax_b.plot(jitter_levels, jitter_results[name], label=name, color=cfg["color"], marker=cfg["marker"], linewidth=2.2, markersize=7)
    ax_b.set_title("B. Phase Invariance under Timestamp Temporal Jitter", fontsize=12, fontweight="bold", pad=10)
    ax_b.set_xlabel("Temporal Jitter Gaussian Standard Deviation (ms)", fontsize=10.5)
    ax_b.set_ylabel("10-Gesture Classification Accuracy (%)", fontsize=10.5)
    ax_b.set_ylim(20, 105)
    ax_b.legend(frameon=True, fontsize=9, loc="lower left")
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Energy Dissipation per Inference
    ax_c = axes[1, 0]
    names = list(models_dict.keys())
    energies = [summary[m]["Energy_Proxy_nJ"] for m in names]
    colors = [models_dict[m]["color"] for m in names]
    bars = ax_c.bar(["AD-CEN", "CEN", "Dense SNN", "Conn. Res."], energies, color=colors, width=0.55, edgecolor="black", linewidth=1.0)
    ax_c.set_yscale("log")
    ax_c.set_title("C. Calibrated Energy Dissipation per Inference (Proxy)", fontsize=12, fontweight="bold", pad=10)
    ax_c.set_ylabel("Energy per Sample (nJ, log scale)", fontsize=10.5)
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., h * 1.25, f"{h:,.1f} nJ", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel D: Latency vs Energy Pareto Trade-off
    ax_d = axes[1, 1]
    for name, cfg in models_dict.items():
        lat = summary[name]["Decision_Latency_ms"]
        en = summary[name]["Energy_Proxy_nJ"]
        acc = summary[name]["Clean_Accuracy_Pct"]
        ax_d.scatter(lat, en, s=220, color=cfg["color"], edgecolors="black", linewidth=1.5, label=f"{name} ({acc:.1f}% acc)", zorder=4)
    ax_d.set_yscale("log")
    ax_d.set_title("D. Decision Latency vs Energy Dissipation Trade-Off", fontsize=12, fontweight="bold", pad=10)
    ax_d.set_xlabel("Decision Latency (ms)", fontsize=10.5)
    ax_d.set_ylabel("Energy per Inference (nJ, log scale)", fontsize=10.5)
    ax_d.legend(frameon=True, fontsize=9, loc="center right")
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    out_paths = [
        os.path.join(root, "results", "figures", "figure18_dvs_gesture_benchmark.png"),
        os.path.join(root, "paper", "figures", "figure18_dvs_gesture_benchmark.png"),
        "/home/charizard/.gemini/antigravity-cli/brain/a942bac5-f639-4fc9-b377-70bb4e1abbe0/figure18_dvs_gesture_benchmark.png",
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[✓] Saved Figure 18 to: {p}")
    plt.close()

if __name__ == "__main__":
    render_figure18()
