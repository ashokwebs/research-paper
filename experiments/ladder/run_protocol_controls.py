"""
COMPUTATIONAL COUPLING --- Protocol sanity controls for the reservoir benchmark.

Produces the two numbers that license every structural comparison in this paper:

1. **Zero-recurrence control.**  With the recurrent matrix set to zero the network
   has no wiring at all.  Under the naive protocol (every neuron driven, every neuron
   decoded) this control scores as well as any connectome, which means that protocol
   cannot detect topology.  Under the sensory/motor-disjoint protocol it must fall to
   chance.  Reporting both is what distinguishes a benchmark that is sensitive to
   recurrent structure from one that merely appears to be.

2. **Gain sweep.**  Accuracy as a function of spectral radius, establishing the
   operating point at which the ladder is run and showing it is not an artefact of a
   single gain.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.eval.reservoir import ReservoirConfig, evaluate_network
from src.tasks.shd import load_shd_cached
from run_structural_ladder import load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")


def zeroed(A):
    B = A.copy()
    for i in range(B.n):
        B.succ[i] = set()
        B.pred[i] = set()
    return B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="celegans")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    train, test = load_shd_cached("train"), load_shd_cached("test")
    A = load_target(args.dataset)
    A0 = zeroed(A)
    chance = 1.0 / (int(max(train.y.max(), test.y.max())) + 1)

    out = {"dataset": args.dataset, "chance": chance, "n_seeds": args.seeds}

    # --- 1. Zero-recurrence control, both protocols -------------------------
    for tag, kw in [
        # Every neuron driven and every neuron decoded.  disjoint_io=False is required:
        # with it left on, the two populations cannot both be the whole network and the
        # readout set comes back empty, which scores chance and inverts the control.
        #
        # input_density is lowered to 0.10 so that the *total* number of input
        # connections roughly matches the disjoint protocol (1.0 x 0.10 x N against
        # 0.15 x 0.50 x N).  Without that the naive arm would also receive several times
        # more input, and the comparison would confound protocol with input strength --
        # which is the kind of unmatched contrast this paper is about.
        ("naive_all_to_all", dict(frac_input_nodes=1.0, frac_readout_nodes=1.0,
                                  input_density=0.10, disjoint_io=False)),
        ("sensory_motor_disjoint", dict()),
    ]:
        for net_tag, net in [("zero_recurrence", A0), ("connectome", A)]:
            accs = []
            for s in range(args.seeds):
                cfg = ReservoirConfig(spectral_radius=1.30, **kw)
                r = evaluate_network(net, train, test, cfg, node_seed=1000 + s)
                accs.append(r.accuracy)
            out[f"{tag}::{net_tag}"] = dict(
                mean=float(np.mean(accs)), std=float(np.std(accs, ddof=1)),
                values=[float(a) for a in accs],
            )
            print(f"{tag:26s} {net_tag:16s} {np.mean(accs):.4f} "
                  f"+- {np.std(accs, ddof=1):.4f}", flush=True)

    # --- 2. Gain sweep ------------------------------------------------------
    sweep = []
    for rho in (0.6, 0.8, 0.95, 1.1, 1.3, 1.6, 2.0):
        accs, rates = [], []
        for s in range(args.seeds):
            cfg = ReservoirConfig(spectral_radius=rho)
            r = evaluate_network(A, train, test, cfg, node_seed=1000 + s)
            accs.append(r.accuracy)
            rates.append(r.mean_rate)
        sweep.append(dict(spectral_radius=rho,
                          accuracy_mean=float(np.mean(accs)),
                          accuracy_std=float(np.std(accs, ddof=1)),
                          mean_rate=float(np.mean(rates))))
        print(f"rho={rho:4.2f}  acc={np.mean(accs):.4f}  rate={np.mean(rates):.4f}",
              flush=True)
    out["gain_sweep"] = sweep

    with open(os.path.join(RESULTS, "protocol_controls.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] -> {RESULTS}/protocol_controls.json")


if __name__ == "__main__":
    main()
