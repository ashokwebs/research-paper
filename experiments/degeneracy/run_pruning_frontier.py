"""
COMPUTATIONAL COUPLING --- Degeneracy frontier under random, heuristic and
adversarial pruning.

The claim under test is Proposition 2 of ``src.theory.transfer_perturbation``: the
amount of structural damage a recurrent circuit tolerates is set not by the size of
the perturbation but by its **operational alignment** ``gamma`` -- the fraction of
the worst-case sensitivity of the input-output map that the perturbation direction
actually realises.  The prediction is that the tolerated pruning fraction scales
as ``1 / gamma``.

This is falsifiable in a way that the usual degeneracy demonstration is not.
Showing that a connectome survives *random* pruning establishes only that random
directions are benign.  We therefore additionally construct the maximally damaging
perturbation of the same cardinality, by ranking synapses on their exact
first-order influence on the horizon-aggregated input-output map, and check whether
its measured ``gamma`` predicts its measured survival curve.  If a connectome
tolerated adversarial and random pruning equally, the proposition would be false.

Every pruned network is renormalised to the original spectral radius before
evaluation, so the measured degradation reflects lost structure rather than lost
gain, and ``gamma`` is computed on exactly the perturbation that is applied.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.eval.reservoir import (
    ReservoirConfig,
    _ridge_fit,
    _ridge_predict,
    make_input_matrix,
    rescale_spectral_radius,
    sensory_motor_split,
    signed_matrix,
    simulate,
)
from src.tasks.shd import load_shd_cached
from src.theory.transfer_perturbation import (
    adversarial_edge_ranking,
    apply_pruning,
    davis_kahan_radius,
    hub_edge_ranking,
    operational_alignment,
    random_edge_ranking,
    spectral_radius,
    structural_drift,
)

sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "degeneracy")
# The grid runs to 0.98 because random pruning survives 0.9 on several networks: a
# grid that stopped there would report a tolerated fraction censored at its own maximum
# and the random-versus-adversarial ratio would be a property of the grid.
FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
STRATEGIES = ("random", "hub", "adversarial")
HORIZON = 15

_CACHE: Dict[str, object] = {}


def _get_data():
    if "train" not in _CACHE:
        _CACHE["train"] = load_shd_cached("train")
        _CACHE["test"] = load_shd_cached("test")
    return _CACHE["train"], _CACHE["test"]


def _accuracy(W, W_in, readout_nodes, cfg, train, test) -> Dict[str, float]:
    F_tr, _, _ = simulate(train.X, W, W_in, cfg, readout_nodes=readout_nodes)
    F_te, syn_te, spk_te = simulate(test.X, W, W_in, cfg, readout_nodes=readout_nodes)
    n_classes = int(max(train.y.max(), test.y.max())) + 1
    Y = np.eye(n_classes)[train.y]
    mu, sd = F_tr.mean(axis=0), F_tr.std(axis=0) + 1e-8
    Ftr = (F_tr - mu) / sd
    Fte = (F_te - mu) / sd
    Wout = _ridge_fit(Ftr, Y, 1.0)
    acc = float((_ridge_predict(Fte, Wout).argmax(axis=1) == test.y).mean())
    return dict(accuracy=acc, synops=float(syn_te), spikes=float(spk_te))


def _job(args) -> List[Dict]:
    key, strategy, seed, cfg_kw = args
    cfg = ReservoirConfig(**cfg_kw)
    train, test = _get_data()

    A_struct = load_target(key)
    W0 = signed_matrix(A_struct)
    rho0 = spectral_radius(W0)
    W0n, _ = rescale_spectral_radius(W0, cfg.spectral_radius)

    input_nodes, readout_nodes = sensory_motor_split(A_struct.n, cfg, seed=2000 + seed)
    W_in = make_input_matrix(train.X.shape[2], A_struct.n, cfg,
                             seed=2000 + seed, input_nodes=input_nodes)
    # The readout observes only the motor population; that indicator is the correct
    # output operator for the sensitivity ranking.
    W_out = np.zeros((len(readout_nodes), A_struct.n))
    W_out[np.arange(len(readout_nodes)), readout_nodes] = 1.0
    # Input operator in state space: columns are the drive each channel injects.
    B = W_in.T                        # (n_neurons, n_channels)

    rng = np.random.default_rng(7000 + seed)
    if strategy == "random":
        ranking = random_edge_ranking(W0n, rng)
    elif strategy == "hub":
        # Weight-magnitude pruning is undefined on these graphs -- every edge has the
        # same magnitude -- so the structured-but-task-blind comparison is a
        # degree-targeted attack instead.  See transfer_perturbation.py.
        ranking = hub_edge_ranking(W0n)
    elif strategy == "adversarial":
        ranking = adversarial_edge_ranking(W0n, B, W_out, horizon=HORIZON)
    else:
        raise ValueError(strategy)

    gap, dk_radius = davis_kahan_radius(W0n, k=5)
    rows = []
    for frac in FRACTIONS:
        E_raw = apply_pruning(W0n, ranking, frac)
        W_pruned = W0n + E_raw
        # Renormalise so that only structure, not gain, differs from baseline.
        W_eval, rho_pruned = rescale_spectral_radius(W_pruned, cfg.spectral_radius)
        E_eff = W_eval - W0n
        align = operational_alignment(W0n, E_eff, B, W_out, horizon=HORIZON)
        perf = _accuracy(W_eval, W_in, readout_nodes, cfg, train, test)
        rows.append(dict(
            dataset=key, strategy=strategy, seed=seed, fraction=frac,
            drift=structural_drift(W0n, E_eff),
            gamma=align.gamma,
            delta_h_norm=align.delta_h_norm,
            e_norm=align.e_norm,
            worst_case_bound=align.worst_case,
            spectral_gap_k5=gap,
            davis_kahan_admissible=dk_radius,
            rho_native=rho0,
            rho_pruned_pre_rescale=rho_pruned,
            **perf,
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--spectral-radius", type=float, default=1.30)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    cfg_kw = dict(spectral_radius=args.spectral_radius)
    keys = [k for k in args.datasets.split(",") if k]

    jobs = [(k, s, seed, cfg_kw)
            for k in keys for s in STRATEGIES for seed in range(args.seeds)]
    print(f"[run] {len(jobs)} pruning sweeps on {args.workers} workers", flush=True)

    import multiprocessing as mp

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, part in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.extend(part)
            el = time.time() - t0
            print(f"  {n}/{len(jobs)}  {el:.0f}s, ~{el/n*(len(jobs)-n):.0f}s left", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "pruning_frontier.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {RESULTS}/pruning_frontier.csv")
    piv = df.groupby(["dataset", "strategy", "fraction"])["accuracy"].mean().unstack(0)
    print(piv.to_string())


if __name__ == "__main__":
    main()
