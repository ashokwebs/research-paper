"""Regenerate every figure in the manuscript from raw result files."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.analysis.figures_ladder import (  # noqa: E402
    figure_dynamics,
    figure_ladder,
    figure_motifs,
    figure_operating_point,
    figure_pruning,
    figure_rate_matched,
)
from src.analysis.figure_null_choice import figure_null_choice  # noqa: E402

RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figures")

LABELS = {
    "celegans": "C. elegans",
    "drosophila_cx": "Drosophila CX",
    "drosophila_al": "Drosophila AL",
    "mouse_v1": "Mouse V1 L2/3",
    "human_hcp": "Human HCP",
}


def _controls():
    p = os.path.join(RES, "ladder", "protocol_controls.json")
    if not os.path.exists(p):
        return 0.05, None
    with open(p) as f:
        d = json.load(f)
    zr = d.get("sensory_motor_disjoint::zero_recurrence", {}).get("mean")
    return d.get("chance", 0.05), zr


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    chance, zero_rec = _controls()

    jobs = [
        # The headline figure. Keyed on the two-null comparison; its mechanism panel
        # needs the descriptor file, checked inside so a missing partner skips the whole
        # figure rather than emitting a half-built one.
        (os.path.join(RES, "ladder", "unit_model_comparison.csv"),
         lambda p: figure_null_choice(
             p, os.path.join(RES, "ladder", "null_mechanism.csv"),
             os.path.join(FIGS, "fig_null_choice.png"))
         if os.path.exists(os.path.join(RES, "ladder", "null_mechanism.csv")) else None),
        (os.path.join(RES, "ladder", "ladder_raw.csv"),
         lambda p: figure_ladder(p, os.path.join(FIGS, "fig_ladder.png"), LABELS,
                                 chance=chance, zero_recurrence=zero_rec)),
        (os.path.join(RES, "ladder", "dynamics_ladder.csv"),
         lambda p: figure_dynamics(p, os.path.join(FIGS, "fig_dynamics.png"), LABELS)),
        (os.path.join(RES, "degeneracy", "pruning_frontier.csv"),
         lambda p: figure_pruning(p, os.path.join(FIGS, "fig_pruning.png"), LABELS,
                                  chance=chance)),
        (os.path.join(RES, "motifs", "motif_significance.csv"),
         lambda p: figure_motifs(p, os.path.join(FIGS, "fig_motifs.png"), LABELS)),
        # Keyed on the rate-matched file; the matched-gain ladder is its companion and
        # is checked inside the lambda so a missing partner skips rather than crashes.
        # Keyed on the gain sweep; the memory sweep is its companion and is checked
        # inside, so a missing partner degrades to a one-panel figure rather than
        # failing.
        (os.path.join(RES, "ladder", "gain_sweep_ladder.csv"),
         lambda p: figure_operating_point(
             p, os.path.join(RES, "ladder", "memory_capacity_sweep.csv"),
             os.path.join(FIGS, "fig_operating_point.png"), LABELS)),
        (os.path.join(RES, "ladder", "ladder_rate_matched.csv"),
         lambda p: figure_rate_matched(
             os.path.join(RES, "ladder", "ladder_raw.csv"), p,
             os.path.join(FIGS, "fig_rate_matched.png"), LABELS, chance=chance)
         if os.path.exists(os.path.join(RES, "ladder", "ladder_raw.csv")) else None),
    ]
    for path, fn in jobs:
        if os.path.exists(path):
            fn(path)
            print(f"[figure] from {os.path.relpath(path, ROOT)}")
        else:
            print(f"[skip] missing {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
