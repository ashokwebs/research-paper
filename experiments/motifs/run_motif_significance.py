"""
COMPUTATIONAL COUPLING --- Cross-species motif enrichment, recomputed.

Corrections relative to earlier revisions of this project
---------------------------------------------------------
1. **p-values.**  Enrichment was previously reported as ``p < 0.0001`` from 25 null
   samples.  The smallest attainable empirical p-value with ``K`` samples is
   ``1/(K+1)``, so 25 samples cannot resolve anything below 0.038.  We use
   ``K = 500`` and report ``p = (1 + #{null >= observed}) / (K + 1)``, stating the
   resolution floor explicitly rather than quoting unattainable values.
2. **Sign-attributed motifs are only computed where valence exists.**  Coherent and
   incoherent feedforward loops require synaptic sign.  Diffusion tractography
   (HCP) carries none, and in the MICrONS Pinky100 release every annotated
   inhibitory soma is disconnected, so both were previously scored with all signs
   set to excitatory -- which makes "coherent FFL enrichment" identical to plain
   FFL enrichment by construction.  Those entries are now reported as topological
   only and flagged.
3. **Fan-in / fan-out are not reported as motifs.**  Double-edge swaps preserve the
   degree sequence, so these counts are invariant by construction and their
   Z-scores are identically zero; including them inflates the motif table.
4. **Bi-fan counts are exact.**  The previous sampling estimator for large graphs
   produced a null standard deviation dominated by its own sampling noise.
5. **Directed 3-cycles are counted correctly.**  The previous enumeration applied an
   ordering filter that undercounted cycles by roughly a factor of two.

The scientific claim that survives is stated in terms of the *significance profile*:
the normalised vector of motif Z-scores, compared across species.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

from src.motifs.fast_census import MOTIF_KEYS, census
from src.nulls.swap_chain import degree_preserving_rewire
from run_structural_ladder import DATASETS, load_target  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "motifs")

# Motifs whose definition requires synaptic valence.
SIGNED_MOTIFS = {"cffl", "iffl", "recip_inhib_pair"}
REPORT_KEYS = [k for k in MOTIF_KEYS]


def _null_job(args):
    key, i, factor = args
    A = load_target(key)
    B = degree_preserving_rewire(A, n_swaps_factor=factor, seed=100_000 + i)
    return census(B, include_bifan=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", type=int, default=500)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--swap-factor", type=float, default=20.0)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keys = [k for k in args.datasets.split(",") if k]
    K = args.nulls
    p_floor = 1.0 / (K + 1)

    import multiprocessing as mp
    import pandas as pd

    rows: List[Dict] = []
    profiles: Dict[str, Dict[str, float]] = {}

    for key in keys:
        A = load_target(key)
        signed = DATASETS[key]["signed"]
        real = census(A, include_bifan=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            nulls = pool.map(_null_job, [(key, i, args.swap_factor) for i in range(K)])
        print(f"[{key}] {K} nulls in {time.time()-t0:.0f}s", flush=True)

        z_by_motif: Dict[str, float] = {}
        for m in REPORT_KEYS:
            vals = np.array([c[m] for c in nulls], dtype=float)
            mu, sd = float(vals.mean()), float(vals.std(ddof=1))
            obs = float(real[m])
            z = (obs - mu) / sd if sd > 1e-12 else 0.0
            n_ge = int((vals >= obs).sum())
            n_le = int((vals <= obs).sum())
            p = (1 + (n_ge if z >= 0 else n_le)) / (K + 1)
            applicable = signed or (m not in SIGNED_MOTIFS)
            z_by_motif[m] = z if applicable else float("nan")
            rows.append(dict(
                dataset=key,
                label=DATASETS[key]["label"],
                organism=DATASETS[key]["organism"],
                motif=m,
                requires_valence=m in SIGNED_MOTIFS,
                valence_available=bool(signed),
                interpretable=bool(applicable),
                n_nodes=A.n, n_edges=A.m,
                observed=obs, null_mean=mu, null_std=sd,
                z_score=z, p_value=p, p_floor=p_floor, n_nulls=K,
            ))
        # Normalised significance profile over the motifs that are interpretable
        # for this dataset (Milo et al. 2004 convention).
        usable = {m: v for m, v in z_by_motif.items() if not np.isnan(v)}
        norm = np.sqrt(sum(v * v for v in usable.values())) or 1.0
        profiles[key] = {m: v / norm for m, v in usable.items()}

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "motif_significance.csv"), index=False)

    # Cross-species similarity of significance profiles, over the motif coordinates
    # that both members of a pair can actually express.
    sim = {}
    for a in profiles:
        for b in profiles:
            shared = sorted(set(profiles[a]) & set(profiles[b]))
            if len(shared) >= 3:
                va = np.array([profiles[a][m] for m in shared])
                vb = np.array([profiles[b][m] for m in shared])
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                sim[f"{a}|{b}"] = float(va @ vb / denom) if denom else 0.0
    with open(os.path.join(RESULTS, "significance_profiles.json"), "w") as f:
        json.dump(dict(profiles=profiles, cosine_similarity=sim,
                       n_nulls=K, p_floor=p_floor), f, indent=2)

    print(f"\n[done] -> {RESULTS}/motif_significance.csv")
    show = df[df.interpretable][["dataset", "motif", "observed", "null_mean", "z_score", "p_value"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
