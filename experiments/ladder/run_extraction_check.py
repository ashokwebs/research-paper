"""
COMPUTATIONAL COUPLING --- Does the subcircuit extraction manufacture the degree effect?

The ladder's largest effect is the one carried by the degree sequence, and the networks
it is measured on are not whole connectomes: where a release is larger than the common
reservoir size, we extract an induced subgraph by breadth-first expansion from a seed
neuron of above-median degree.  Growing outward from a hub is exactly the kind of
sampling that could concentrate high-degree nodes and so inflate the degree
heterogeneity of the sample relative to the network it came from --- which would make
"imposing the empirical degree sequence hurts" partly a statement about our extraction.

This script measures that directly, as the coefficient of variation of total degree in
the full release against the extracted subgraph.  A ratio above one would mean the
extraction concentrates heterogeneity and the effect is partly an artefact of sampling;
a ratio at or below one means the extraction cannot be what produces it.

Writes ``results/ladder/extraction_check.csv``.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.connectome.subcircuits import drop_isolated
from src.motifs.fast_census import from_networkx
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")


def degree_cv(A) -> float:
    """Coefficient of variation of total (in + out) degree."""
    d = np.array([len(A.succ[i]) + len(A.pred[i]) for i in range(A.n)], dtype=float)
    return float(d.std() / d.mean()) if d.mean() > 0 else float("nan")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    import pandas as pd

    rows = []
    for key in DATASETS:
        g = DATASETS[key]["loader"]()
        A_full = drop_isolated(from_networkx(g.graph))
        A_sub = load_target(key)
        cv_full, cv_sub = degree_cv(A_full), degree_cv(A_sub)
        rows.append(dict(
            dataset=key,
            n_full=A_full.n, m_full=A_full.m, degree_cv_full=cv_full,
            n_subgraph=A_sub.n, m_subgraph=A_sub.m, degree_cv_subgraph=cv_sub,
            cv_ratio=cv_sub / cv_full if cv_full else float("nan"),
            # True when no extraction happened: the release already fits.
            whole_network=bool(A_sub.n == A_full.n),
        ))
        print(f"[{key}] full n={A_full.n} CV={cv_full:.3f} | "
              f"sub n={A_sub.n} CV={cv_sub:.3f} | ratio={rows[-1]['cv_ratio']:.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "extraction_check.csv")
    df.to_csv(out, index=False)
    print(f"\n[done] -> {out}")
    print(f"max CV ratio: {df.cv_ratio.max():.3f} "
          f"({'extraction concentrates heterogeneity' if df.cv_ratio.max() > 1.0 else 'extraction does not concentrate heterogeneity'})")


if __name__ == "__main__":
    main()
