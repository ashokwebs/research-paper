"""
COMPUTATIONAL COUPLING --- The structural control ladder.

Claims of the form "connectome wiring confers a computational advantage" are only
as strong as the control the connectome is compared against.  Comparing a real
connectome to an Erdos-Renyi graph confounds every structural property at once.
This module builds a nested sequence of surrogate ensembles in which exactly one
structural property is added per rung, so that the contribution of each property
can be read off as a difference between adjacent rungs:

===== ==================================== =========================================
Rung  Preserved                             Isolated by comparing with the rung below
===== ==================================== =========================================
L0    N, edge count, E:I ratio              --
L1    + joint in/out degree sequence        degree heterogeneity
L2    + valence-topology coupling           where inhibition sits in the graph
L3    + 3-node motif census                 motif structure
L4    the empirical connectome              residual higher-order wiring
===== ==================================== =========================================

Every rung has identical node count and edge count by construction, and the
evaluation harness additionally matches spectral radius and input weights, so a
difference between rungs cannot be explained by size, density or gain.

L3 is the rung that does not exist in the literature; it is produced by the
annealed motif-preserving chain in :mod:`src.nulls.swap_chain`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..motifs.fast_census import AdjacencySets, census
from .swap_chain import (
    census_scale,
    degree_preserving_rewire,
    erdos_renyi_matched,
    motif_preserving_null,
    permute_signs,
)

LADDER_RUNGS: Tuple[str, ...] = (
    "L0_erdos_renyi",
    "L1_degree",
    "L2_degree_valence",
    "L3_degree_valence_motif",
    "L4_connectome",
)

RUNG_LABEL: Dict[str, str] = {
    "L0_erdos_renyi": "Erdos-Renyi (density)",
    "L1_degree": "+ degree sequence",
    "L2_degree_valence": "+ valence placement",
    "L3_degree_valence_motif": "+ motif census",
    "L4_connectome": "Empirical connectome",
}


@dataclass
class LadderSample:
    rung: str
    instance: int
    A: AdjacencySets
    diagnostics: Dict[str, float]


def build_rung(
    A_real: AdjacencySets,
    rung: str,
    instance: int,
    scale_counts: Optional[Dict[str, float]] = None,
    target_counts: Optional[Dict[str, int]] = None,
    n_sweeps: float = 120.0,
    rewire_factor: float = 20.0,
) -> LadderSample:
    """Draw one surrogate from the requested rung."""
    seed = 10_000 * (LADDER_RUNGS.index(rung) + 1) + instance
    diag: Dict[str, float] = {}

    if rung == "L4_connectome":
        A = A_real.copy()
    elif rung == "L0_erdos_renyi":
        A = erdos_renyi_matched(A_real, seed=seed)
    elif rung == "L1_degree":
        # Degree sequence preserved, but valence labels shuffled across neurons so
        # that no information about *which* neurons are inhibitory survives.
        A = permute_signs(
            degree_preserving_rewire(A_real, n_swaps_factor=rewire_factor, seed=seed),
            seed=seed + 1,
        )
    elif rung == "L2_degree_valence":
        A = degree_preserving_rewire(A_real, n_swaps_factor=rewire_factor, seed=seed)
    elif rung == "L3_degree_valence_motif":
        A, rep = motif_preserving_null(
            A_real,
            target_counts=target_counts,
            scale_counts=scale_counts,
            n_sweeps=n_sweeps,
            seed=seed,
        )
        diag["anneal_converged"] = float(rep.converged)
        diag["anneal_accept_rate"] = rep.accepted / max(rep.proposed, 1)
        diag["anneal_energy_final"] = rep.energy_final
        diag["anneal_max_rel_err"] = max(rep.relative_error().values())
    else:
        raise ValueError(f"unknown rung {rung!r}")

    c = census(A, include_bifan=False)
    for k, v in c.items():
        diag[f"motif_{k}"] = float(v)
    diag["n_nodes"] = float(A.n)
    diag["n_edges"] = float(A.m)
    return LadderSample(rung=rung, instance=instance, A=A, diagnostics=diag)


def build_ladder(
    A_real: AdjacencySets,
    n_instances: int = 20,
    rungs: Sequence[str] = LADDER_RUNGS,
    n_scale_samples: int = 20,
    n_sweeps: float = 120.0,
    rewire_factor: float = 20.0,
    progress: Optional[Callable[[str, int], None]] = None,
) -> List[LadderSample]:
    """Build the full ladder.  The empirical rung has a single (deterministic) member."""
    scale = census_scale(A_real, n_samples=n_scale_samples,
                         n_swaps_factor=rewire_factor)

    out: List[LadderSample] = []
    for rung in rungs:
        k = 1 if rung == "L4_connectome" else n_instances
        for i in range(k):
            if progress:
                progress(rung, i)
            out.append(
                build_rung(
                    A_real, rung, i,
                    scale_counts=scale,
                    n_sweeps=n_sweeps,
                    rewire_factor=rewire_factor,
                )
            )
    return out
