"""
COMPUTATIONAL COUPLING --- How much does the answer depend on which null you build?

Two rewiring procedures both satisfy the description that appears in the literature ---
"randomised while preserving density and degree sequence" --- and they are not the same
ensemble:

* ``bct_randmio`` carries each weight along with its edge through degree-preserving
  double-edge swaps, maintaining connectedness.  This is what the published toolbox
  calls.
* ``rank_reassigned`` rewires the binary topology and then re-assigns the same multiset
  of weight values to the new edges by rank of endpoint degree.  It preserves the degree
  sequence and the weight distribution, and destroys the association between a weight and
  the edge carrying it.

We wrote the second one first, believing it to implement the first.  It produced a clean,
significant, entirely spurious result.  This file measures the size of that difference,
because it is the paper's central point and because we are the worked example.

Reported per network and unit model: the empirical connectome's advantage over each null,
paired by instance, with the disagreement between the two nulls stated as a quantity
rather than a remark.
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results", "ladder")
CRITICAL = 1.0
NULLS = ("bct_randmio", "rank_reassigned")


def _load() -> pd.DataFrame:
    # "null" is in pandas' default NA list; read condition labels as literal strings.
    df = pd.read_csv(os.path.join(RESULTS, "unit_model_comparison.csv"),
                     keep_default_na=False, na_values=[""])
    for c in ("memory_capacity", "spectral_radius", "dead_frac",
              "participation_ratio", "saturated_frac"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def contrasts(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Empirical minus each null, paired by instance, per network and unit model."""
    rows: List[Dict] = []
    sub = df[np.isclose(df.spectral_radius, rho)]
    for (unit, net), g in sub.groupby(["unit_model", "network"]):
        emp = g[g.kind == "empirical"].sort_values("instance")["memory_capacity"].to_numpy()
        for null in NULLS:
            nul = g[g.kind == null].sort_values("instance")["memory_capacity"].to_numpy()
            k = min(len(emp), len(nul))
            if k < 3:
                continue
            d = emp[:k] - nul[:k]
            p = float(stats.ttest_rel(emp[:k], nul[:k]).pvalue) if d.std() > 0 else 1.0
            rows.append(dict(
                unit_model=unit, network=net, null=null,
                empirical=float(emp[:k].mean()), null_cap=float(nul[:k].mean()),
                diff=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / np.sqrt(k)),
                p=p, n=int(k),
                verdict=("leads" if (p < 0.05 and d.mean() > 0)
                         else "trails" if (p < 0.05 and d.mean() < 0) else "n.s."),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    df = _load()
    have = set(df.kind.unique())
    if not set(NULLS) <= have:
        print(f"[skip] need both nulls, have {sorted(have)}")
        return

    print(f"[null-sensitivity] {len(df)} evaluations, {df.network.nunique()} networks, "
          f"{df.instance.nunique()} instances\n")

    c = contrasts(df, CRITICAL)
    print(f"=== empirical minus null at rho = {CRITICAL}, by null construction ===")
    print(c[["unit_model", "network", "null", "empirical", "null_cap", "diff", "ci",
             "p", "verdict"]].round(4).to_string(index=False))

    # The headline: how often do the two nulls disagree about the direction?
    piv = c.pivot_table(index=["unit_model", "network"], columns="null",
                        values="diff")
    ver = c.pivot_table(index=["unit_model", "network"], columns="null",
                        values="verdict", aggfunc="first")
    print("\n=== difference in the measured advantage, by null ===")
    print(piv.round(3).to_string())
    print("\n=== verdict, by null ===")
    print(ver.to_string())

    both = piv.dropna()
    sign_flip = int((np.sign(both[NULLS[0]]) != np.sign(both[NULLS[1]])).sum())
    gap = (both[NULLS[1]] - both[NULLS[0]]).abs()
    verdict_flip = 0
    for idx in ver.dropna().index:
        a, b = ver.loc[idx, NULLS[0]], ver.loc[idx, NULLS[1]]
        if {a, b} == {"leads", "trails"}:
            verdict_flip += 1

    print(f"\n  cells where the two nulls give opposite SIGNS:    "
          f"{sign_flip} of {len(both)}")
    print(f"  cells where they give opposite SIGNIFICANT verdicts: "
          f"{verdict_flip} of {len(ver.dropna())}")
    print(f"  median absolute disagreement: {gap.median():.3f} capacity units")
    print(f"  largest:                      {gap.max():.3f}")

    # How often does each null, on its own, reproduce the published direction?
    summary: Dict = dict(n_cells=int(len(both)), n_sign_flip=sign_flip,
                         n_verdict_flip=verdict_flip,
                         median_abs_disagreement=float(gap.median()),
                         max_abs_disagreement=float(gap.max()))
    for null in NULLS:
        sub = c[c.null == null]
        for unit in ("rate", "spiking"):
            u = sub[sub.unit_model == unit]
            summary[f"{null}_{unit}_leads"] = int((u.verdict == "leads").sum())
            summary[f"{null}_{unit}_trails"] = int((u.verdict == "trails").sum())
            summary[f"{null}_{unit}_n"] = int(len(u))
            summary[f"{null}_{unit}_median"] = float(u["diff"].median())
            print(f"  {null:16s} {unit:8s}: leads {int((u.verdict=='leads').sum())}"
                  f"/{len(u)}, trails {int((u.verdict=='trails').sum())}/{len(u)}, "
                  f"median {u['diff'].median():+.3f}")

    with open(os.path.join(RESULTS, "null_sensitivity.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[done] -> {RESULTS}/null_sensitivity.json")


if __name__ == "__main__":
    main()
