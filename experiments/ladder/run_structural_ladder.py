"""
COMPUTATIONAL COUPLING --- Structural control ladder on Spiking Heidelberg Digits.

Asks, for each connectome, how much of its task performance as a fixed recurrent
reservoir is attributable to each structural property in turn: density, degree
sequence, valence placement, 3-node motif census, and residual wiring.  Every rung
is matched on node count, edge count, spectral radius and input projection, so a
difference between adjacent rungs isolates exactly one property.

Usage
-----
    python experiments/ladder/run_structural_ladder.py [--quick] [--workers N]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.connectome.celegans_loader import load_celegans_connectome
from src.connectome.drosophila_loader import load_drosophila_subcircuit
from src.connectome.human_loader import load_human_hcp_macroscale
from src.connectome.mouse_loader import load_mouse_v1_connectome
from src.connectome.subcircuits import bfs_subgraph, drop_isolated, summarise
from src.eval.reservoir import ReservoirConfig, evaluate_network
from src.motifs.fast_census import census, from_networkx
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import census_scale, ensemble_scale
from src.tasks.shd import load_shd_cached

N_NODES = 400          # common reservoir size across species
RESULTS = os.path.join(ROOT, "results", "ladder")


DATASETS = {
    "celegans": dict(
        label="C. elegans (whole nervous system)",
        organism="Caenorhabditis elegans",
        signed=True,
        loader=lambda: load_celegans_connectome(),
    ),
    "drosophila_cx": dict(
        label="Drosophila central complex",
        organism="Drosophila melanogaster",
        signed=True,
        loader=lambda: load_drosophila_subcircuit(subcircuit="central_complex"),
    ),
    "drosophila_al": dict(
        label="Drosophila antennal lobe",
        organism="Drosophila melanogaster",
        signed=True,
        loader=lambda: load_drosophila_subcircuit(subcircuit="antennal_lobe"),
    ),
    "mouse_v1": dict(
        label="Mouse V1 L2/3 (MICrONS Pinky100)",
        organism="Mus musculus",
        signed=False,   # every annotated inhibitory soma is disconnected in this release
        loader=lambda: load_mouse_v1_connectome(),
    ),
    "human_hcp": dict(
        label="Human HCP macroscale",
        organism="Homo sapiens",
        signed=False,   # diffusion tractography carries no synaptic valence
        loader=lambda: load_human_hcp_macroscale(),
    ),
}


_TARGET_CACHE: Dict[tuple, object] = {}


def load_target(key: str, n_nodes: int = N_NODES, seed: int = 0):
    """Load a connectome and reduce it to a size-matched, isolate-free subcircuit.

    Cached per worker process: the FlyWire release is hundreds of megabytes and is
    otherwise re-parsed on every job, which dominates runtime for experiments that
    draw many surrogates from the same connectome.
    """
    ck = (key, n_nodes, seed)
    hit = _TARGET_CACHE.get(ck)
    if hit is not None:
        return hit.copy()
    g = DATASETS[key]["loader"]()
    A = drop_isolated(from_networkx(g.graph))
    if A.n > n_nodes:
        A = drop_isolated(bfs_subgraph(A, n_nodes, seed=seed))
    _TARGET_CACHE[ck] = A
    return A.copy()


# --------------------------------------------------------------------------- #
# Worker entry points (module level so they pickle for multiprocessing)
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, object] = {}


def _get_data():
    if "train" not in _CACHE:
        _CACHE["train"] = load_shd_cached("train")
        _CACHE["test"] = load_shd_cached("test")
    return _CACHE["train"], _CACHE["test"]


def _job(args) -> Dict:
    key, rung, instance, scale, n_sweeps, cfg_kw = args
    t0 = time.time()
    A_real = load_target(key)
    sample = build_rung(
        A_real, rung, instance,
        scale_counts=scale,
        n_sweeps=n_sweeps,
    )
    build_s = time.time() - t0

    train, test = _get_data()
    cfg = ReservoirConfig(**cfg_kw)
    # The instance index doubles as the seed for the sensory/motor node split and
    # the input projection, so every rung sees the same 15 input codings and the
    # variance sources are matched across rungs.
    t1 = time.time()
    res = evaluate_network(sample.A, train, test, cfg, node_seed=1000 + instance)
    row = dict(
        dataset=key,
        rung=rung,
        instance=instance,
        accuracy=res.accuracy,
        val_accuracy=res.val_accuracy,
        synops_per_sample=res.synops_per_sample,
        n_readout_nodes=res.extra.get("n_readout_nodes", 0.0),
        spikes_per_sample=res.spikes_per_sample,
        mean_rate=res.mean_rate,
        ridge_alpha=res.alpha,
        n_neurons=res.n_neurons,
        n_edges=res.n_edges,
        spectral_radius_pre=res.spectral_radius_pre,
        build_seconds=build_s,
        eval_seconds=time.time() - t1,
    )
    row.update({f"diag_{k}": v for k, v in sample.diagnostics.items()})
    return row


def build_parser() -> argparse.ArgumentParser:
    """The experiment's argument parser.

    Exposed separately so that the manuscript's macro generator can read the protocol
    defaults from here rather than restating them, which is how a paper ends up
    describing an experiment nobody ran.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 instances, short anneal")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--instances", type=int, default=15)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--spectral-radius", type=float, default=1.30)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    n_instances = 3 if args.quick else args.instances
    n_sweeps = 25.0 if args.quick else args.sweeps
    keys = [k for k in args.datasets.split(",") if k]

    os.makedirs(RESULTS, exist_ok=True)
    cfg_kw = dict(spectral_radius=args.spectral_radius)

    # Pre-compute the annealing energy normalisation once per dataset.
    scales: Dict[str, Dict[str, float]] = {}
    meta: Dict[str, Dict] = {}
    for key in keys:
        A = load_target(key)
        scales[key] = census_scale(A, n_samples=10, seed=0)
        _, std = ensemble_scale(A, n_samples=10, seed=0)
        meta[key] = dict(
            label=DATASETS[key]["label"],
            organism=DATASETS[key]["organism"],
            signed=DATASETS[key]["signed"],
            summary=summarise(A),
            real_census=census(A, include_bifan=False),
            null_census_std=std,
        )
        print(f"[setup] {key}: {meta[key]['summary']}", flush=True)

    jobs = []
    for key in keys:
        for rung in LADDER_RUNGS:
            for i in range(n_instances):
                jobs.append((key, rung, i, scales[key], n_sweeps, cfg_kw))

    print(f"[run] {len(jobs)} jobs on {args.workers} workers", flush=True)
    t0 = time.time()
    rows: List[Dict] = []

    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(row)
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el:.0f}s elapsed, "
                      f"~{el / n * (len(jobs) - n):.0f}s left", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["dataset", "rung", "instance"])
    df.to_csv(os.path.join(RESULTS, "ladder_raw.csv"), index=False)
    with open(os.path.join(RESULTS, "ladder_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)

    print(f"\n[done] {time.time() - t0:.0f}s -> {RESULTS}/ladder_raw.csv")
    print(df.groupby(["dataset", "rung"])["accuracy"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
