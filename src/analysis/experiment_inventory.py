"""
COMPUTATIONAL COUPLING --- The canonical list of experiments in this project.

One spec, consumed by both the manuscript's table and the prose documents. It lives in
its own module because the previous arrangement -- a list in the LaTeX table generator and
a hand-written list of commands in ``info.md`` -- had already drifted: the documents were
missing two experiments and quoted a test count from an earlier revision. A reader who
wants to know what was run should not have to reconcile three lists.

Each entry names the script that produces the result, the file it writes, and the unit its
rows are counted in. Sizes are always read from the output file at build time, never
recorded here, so an experiment that has not been run reports as not run rather than as a
number someone remembered.
"""

import json
import os
from typing import Dict, List, NamedTuple, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")


class Experiment(NamedTuple):
    section: str        # the manuscript section reporting it, as a \ref key
    title: str          # a short name, for the documents
    description: str    # one line, plain enough to read outside the field
    script: str         # the command that produces the result
    result: str         # path under results/
    unit: str           # what one row of that file is


EXPERIMENTS: List[Experiment] = [
    Experiment(
        "sec:res-motifs", "Motif enrichment",
        "Are the classic subgraphs really over-represented? Exact sign-attributed counts "
        "against 500 degree-preserving surrogates per network.",
        "experiments/motifs/run_motif_significance.py",
        "motifs/motif_significance.csv", "tests"),
    Experiment(
        "sec:res-null", "Motif-preserving null ensemble",
        "Validation of the annealed chain: does it hit the target census, does it "
        "actually randomise, and does it leave untargeted structure alone?",
        "experiments/ladder/run_null_validation.py",
        "ladder/null_validation.json", "networks"),
    Experiment(
        "sec:res-robust", "Sampler robustness",
        "The same ensemble redrawn over a factor-of-eight range of chain lengths and from "
        "both possible starting points, ruling out two ways a sampler could fake the result.",
        "experiments/ladder/run_null_robustness.py",
        "ladder/null_robustness.csv", "draws"),
    Experiment(
        "sec:res-protocol", "Protocol controls",
        "What does the benchmark measure? Includes the reservoir with its recurrent "
        "matrix set to zero, under both readout protocols.",
        "experiments/ladder/run_protocol_controls.py",
        "ladder/protocol_controls.json", "conditions"),
    Experiment(
        "sec:res-ladder", "Structural control ladder",
        "L0 density to L4 real connectome on spoken digits, each rung adding exactly one "
        "structural property; accuracy, memory capacity, kernel and generalisation rank.",
        "experiments/ladder/run_structural_ladder.py",
        "ladder/ladder_raw.csv", "runs"),
    Experiment(
        "sec:res-rate", "The ladder at matched firing rate",
        "The same decomposition with every network recalibrated to a common firing rate, "
        "because matching spectral radius does not match activity.",
        "experiments/ladder/run_rate_matched_ladder.py",
        "ladder/ladder_rate_matched.csv", "runs"),
    Experiment(
        "sec:res-ladder", "Extraction check",
        "Does taking a size-matched subgraph of a larger connectome bias the degree "
        "result? Run because the ladder depends on the extraction being neutral.",
        "experiments/ladder/run_extraction_check.py",
        "ladder/extraction_check.csv", "networks"),
    Experiment(
        "sec:res-nullchoice", "Two null ensembles, two unit models",
        "The headline experiment. Both readings of \"degree-preserving randomisation\" "
        "against both unit models across nine gains, on the original toolbox's own connectomes.",
        "experiments/ladder/run_unit_model_comparison.py",
        "ladder/unit_model_comparison.csv", "evaluations"),
    Experiment(
        "sec:res-nullchoice", "Why the two nulls differ",
        "Structural descriptors of both ensembles against the real graph, including the "
        "prediction that they must coincide on an unweighted network.",
        "experiments/ladder/run_null_mechanism.py",
        "ladder/null_mechanism.csv", "graphs"),
    Experiment(
        "sec:res-unit", "Spiking tuning control",
        "Input gain x threshold x leak, searching for the configuration that would refute "
        "our own spiking result rather than the one that confirms it.",
        "experiments/ladder/run_spiking_tuning_control.py",
        "ladder/spiking_tuning_control.csv", "evaluations"),
    Experiment(
        "sec:res-operating", "Gain sweep of the ladder",
        "The whole decomposition across the critical region, because a comparison at one "
        "gain is a statement about that gain.",
        "experiments/ladder/run_gain_sweep_ladder.py",
        "ladder/gain_sweep_ladder.csv", "evaluations"),
    Experiment(
        "sec:res-operating", "Memory-capacity sweep",
        "Locating the published advantage on the gain axis at matched spectral radius.",
        "experiments/ladder/run_memory_capacity_sweep.py",
        "ladder/memory_capacity_sweep.csv", "evaluations"),
    Experiment(
        "sec:res-dynamics", "Intrinsic dynamics",
        "The same decomposition measured on task-independent probes, so the ordering does "
        "not depend on one benchmark.",
        "experiments/ladder/run_dynamics_ladder.py",
        "ladder/dynamics_ladder.csv", "runs"),
    Experiment(
        "sec:res-degeneracy", "Pruning frontier",
        "Random, hub-targeted and adversarial edge removal, testing whether the "
        "operational alignment predicts how much damage a direction does.",
        "experiments/degeneracy/run_pruning_frontier.py",
        "degeneracy/pruning_frontier.csv", "prunings"),
    Experiment(
        "sec:res-hardware", "Open-PDK synthesis",
        "Real logic synthesis and technology mapping of the accompanying RTL, reporting "
        "only what the open flow can establish.",
        "experiments/hardware/run_open_synthesis.py",
        "hardware/open_synthesis.csv", "designs"),
]


def size(exp: Experiment) -> Optional[int]:
    """Rows in the experiment's output file, or None if it has not been run."""
    path = os.path.join(RES, *exp.result.split("/"))
    if not os.path.exists(path):
        return None
    if path.endswith(".csv"):
        import pandas as pd

        return len(pd.read_csv(path, keep_default_na=False, na_values=[""]))
    with open(path) as f:
        obj = json.load(f)
    return len(obj) if isinstance(obj, (list, dict)) else 1


def sizes() -> Dict[str, Optional[int]]:
    return {e.result: size(e) for e in EXPERIMENTS}


def markdown_table() -> str:
    """The inventory as a markdown table, for the prose documents."""
    rows = ["| # | Experiment | What it asks | Size |",
            "|---|---|---|---|"]
    for i, e in enumerate(EXPERIMENTS, 1):
        n = size(e)
        amount = f"{n:,} {e.unit}" if n is not None else "*not run*"
        rows.append(f"| {i} | **{e.title}**<br>`{e.script}` | {e.description} | {amount} |")
    return "\n".join(rows)


def command_list() -> str:
    """The experiments as a runnable sequence, in dependency order."""
    width = max(len(e.script) for e in EXPERIMENTS)
    return "\n".join(f"python {e.script.ljust(width)}  # {e.title}"
                     for e in EXPERIMENTS)


if __name__ == "__main__":
    print(markdown_table())
    print()
    missing = [e.title for e in EXPERIMENTS if size(e) is None]
    print(f"{len(EXPERIMENTS) - len(missing)}/{len(EXPERIMENTS)} experiments have results")
    if missing:
        print("not run:", ", ".join(missing))
