"""
COMPUTATIONAL COUPLING --- Statistical analysis of the structural control ladder.

Reports, per connectome, the accuracy attributable to each structural property as
the difference between adjacent rungs, with bootstrap confidence intervals and
Welch tests, and applies a Holm-Bonferroni correction across the full family of
rung comparisons.  Effect sizes are reported as Hedges' g alongside p-values,
because with 15 instances per rung a statistically detectable difference can still
be operationally negligible, and that distinction is the point of the analysis.
"""

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.nulls.ladder import LADDER_RUNGS, RUNG_LABEL  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "ladder")


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardised mean difference."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / sp
    J = 1.0 - 3.0 / (4 * (na + nb) - 9)
    return float(d * J)


def boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
            seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the difference of means."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - \
                   rng.choice(b, len(b), replace=True).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def min_detectable_effect(n: int, alpha: float, power: float = 0.8) -> float:
    """Smallest standardised mean difference two groups of ``n`` can resolve.

    A null result is only as strong as the effect it could have detected, so this is
    reported alongside the contrasts rather than left for a reader to reconstruct.
    Computed from the noncentral t distribution at the given two-sided ``alpha``, which
    for the corrected figure is the Holm worst case (``0.05 / m``) rather than 0.05.
    """
    df = 2 * n - 2
    crit = stats.t.ppf(1 - alpha / 2, df)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        d = 0.5 * (lo + hi)
        ncp = d * np.sqrt(n / 2.0)
        achieved = (1 - stats.nct.cdf(crit, df, ncp)) + stats.nct.cdf(-crit, df, ncp)
        if achieved < power:
            lo = d
        else:
            hi = d
    return 0.5 * (lo + hi)


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(RESULTS, "ladder_raw.csv"))
    ap.add_argument("--metric", default="accuracy")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    metric = args.metric

    # --- Per-rung summary ---------------------------------------------------
    summary = (df.groupby(["dataset", "rung"])[metric]
                 .agg(mean="mean", std="std", n="count")
                 .reset_index())
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["label"] = summary["rung"].map(RUNG_LABEL)
    summary.to_csv(os.path.join(RESULTS, f"ladder_summary_{metric}.csv"), index=False)

    # --- Adjacent-rung contrasts (the structural decomposition) -------------
    rows: List[Dict] = []
    for ds, g in df.groupby("dataset"):
        by = {r: g[g.rung == r][metric].to_numpy() for r in LADDER_RUNGS}
        for lo, hi in zip(LADDER_RUNGS[:-1], LADDER_RUNGS[1:]):
            a, b = by.get(hi), by.get(lo)
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            lo_ci, hi_ci = boot_ci(a, b, seed=abs(hash((ds, lo, hi))) % 2**31)
            rows.append(dict(
                dataset=ds, contrast=f"{hi} - {lo}",
                property_added=RUNG_LABEL[hi],
                mean_hi=float(a.mean()), mean_lo=float(b.mean()),
                delta=float(a.mean() - b.mean()),
                ci_low=lo_ci, ci_high=hi_ci,
                hedges_g=hedges_g(a, b), t=float(t), p_raw=float(p),
                n_hi=len(a), n_lo=len(b),
            ))
    # Also compare the empirical connectome directly against the motif-matched null,
    # which is the comparison that isolates everything beyond 3-node structure.
    contrasts = pd.DataFrame(rows)
    if len(contrasts):
        contrasts["p_holm"] = holm(contrasts["p_raw"].tolist())
        contrasts["significant_holm_05"] = contrasts["p_holm"] < 0.05
        contrasts.to_csv(os.path.join(RESULTS, f"ladder_contrasts_{metric}.csv"), index=False)

    # --- Power ---------------------------------------------------------------
    # What the design could have resolved, so that a small measured effect can be
    # distinguished from an underpowered one.
    if len(contrasts):
        n_per = int(df.groupby(["dataset", "rung"]).size().min())
        m_fam = int(len(contrasts))
        power = dict(
            metric=metric,
            n_per_group=n_per,
            n_contrasts=m_fam,
            mde_uncorrected=float(min_detectable_effect(n_per, 0.05)),
            mde_holm_worst_case=float(min_detectable_effect(n_per, 0.05 / m_fam)),
            power=0.8,
        )
        with open(os.path.join(RESULTS, f"power_{metric}.json"), "w") as f:
            json.dump(power, f, indent=2)
        print(f"\n=== Power ({metric}) ===")
        print(f"n = {n_per} per rung, {m_fam} contrasts; 80% power to detect "
              f"d = {power['mde_uncorrected']:.2f} uncorrected, "
              f"d = {power['mde_holm_worst_case']:.2f} at the Holm worst case")

    # --- Annealing diagnostics ---------------------------------------------
    diag_cols = [c for c in df.columns if c.startswith("diag_anneal")]
    diag = None
    if diag_cols:
        d3 = df[df.rung == "L3_degree_valence_motif"]
        diag = d3.groupby("dataset")[diag_cols].mean().reset_index()
        diag.to_csv(os.path.join(RESULTS, "anneal_diagnostics.csv"), index=False)

    print("=== Per-rung means ===")
    print(summary.pivot(index="rung", columns="dataset", values="mean").to_string())
    if len(contrasts):
        print("\n=== Adjacent-rung contrasts ===")
        cols = ["dataset", "property_added", "delta", "ci_low", "ci_high",
                "hedges_g", "p_raw", "p_holm", "significant_holm_05"]
        print(contrasts[cols].to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    if diag is not None:
        print("\n=== Annealed null convergence (L3) ===")
        print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
