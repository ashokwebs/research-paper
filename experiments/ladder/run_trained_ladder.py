"""
COMPUTATIONAL COUPLING --- The structural control ladder with trained weights.

Each rung supplies the recurrent connectivity mask and the valence pattern; synaptic
magnitudes are learned by surrogate-gradient BPTT.  This tests the inductive-bias
version of the motif hypothesis, which a fixed reservoir cannot address: if a
connectome's 3-node structure makes a network easier to train, the empirical and
motif-matched rungs should outperform the degree-matched rung at identical density.
"""

import argparse
import os
import sys
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.eval.reservoir import ReservoirConfig, sensory_motor_split
from src.eval.trained_snn import TrainConfig, train_and_eval
from src.motifs.fast_census import census
from src.nulls.ladder import LADDER_RUNGS, build_rung
from src.nulls.swap_chain import census_scale
from src.tasks.shd import load_shd_cached
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")

# An extra condition beyond the ladder rungs: the same trained network with an empty
# recurrent mask.  It is built from the L4 rung so that everything except the recurrent
# synapses matches the empirical condition.
ZERO_RUNG = "Z_zero_recurrence"


def to_mask(A) -> np.ndarray:
    M = np.zeros((A.n, A.n), dtype=np.float32)
    for u, v in A.iter_edges():
        M[u, v] = 1.0
    return M


def _job(args) -> Dict:
    key, rung, seed, sweeps, scale, tcfg_kw, T, C = args
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    train = load_shd_cached("train", n_steps=T, n_channels=C)
    test = load_shd_cached("test", n_steps=T, n_channels=C)

    A_real = load_target(key)
    s = build_rung(A_real, "L4_connectome" if rung == ZERO_RUNG else rung,
                   seed, scale_counts=scale, n_sweeps=sweeps)
    cfg = TrainConfig(hidden=s.A.n, seed=seed, **tcfg_kw)

    # Same sensory/motor split as the fixed-reservoir harness, and the same node seed,
    # so the encoder and readout populations are identical across rungs and identical
    # to the ones the reservoir ladder used.
    rcfg = ReservoirConfig()
    input_nodes, readout_nodes = sensory_motor_split(s.A.n, rcfg, seed=1000 + seed)

    mask = to_mask(s.A)
    if rung == ZERO_RUNG:
        # The diagnostic: no recurrent synapses at all.  If this scores near the
        # connectomes, the trained benchmark is not testing recurrent structure and
        # nothing else in the table means anything.
        mask = np.zeros_like(mask)

    t0 = time.time()
    res = train_and_eval(
        mask, s.A.sign,
        train.X.astype(np.float32), train.y,
        test.X.astype(np.float32), test.y,
        cfg, n_classes=20,
        input_nodes=input_nodes, readout_nodes=readout_nodes,
    )
    return dict(dataset=key, rung=rung, seed=seed,
                train_seconds=time.time() - t0, **res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sweeps", type=float, default=120.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--channels", type=int, default=140)
    ap.add_argument("--datasets", type=str, default="celegans,drosophila_cx")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]

    scales = {}
    for key in keys:
        A = load_target(key)
        scales[key] = census_scale(A, n_samples=10, seed=0)

    tcfg_kw = dict(epochs=args.epochs)
    conditions = list(LADDER_RUNGS) + [ZERO_RUNG]
    jobs = [(k, r, s, args.sweeps, scales[k], tcfg_kw, args.steps, args.channels)
            for k in keys for r in conditions for s in range(args.seeds)]
    print(f"[run] {len(jobs)} training runs on {args.workers} workers", flush=True)

    import multiprocessing as mp
    import pandas as pd

    t0 = time.time()
    rows: List[Dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(row)
            el = time.time() - t0
            print(f"  {n}/{len(jobs)} {row['dataset']}/{row['rung']} "
                  f"acc={row['accuracy']:.4f} synops={row['synops_per_sample']:.0f} "
                  f"({el:.0f}s, ~{el/n*(len(jobs)-n):.0f}s left)", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS, "trained_ladder.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "trained_ladder.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> {RESULTS}/trained_ladder.csv")
    print(df.groupby(["dataset", "rung"])[["accuracy", "synops_per_sample"]]
            .agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
