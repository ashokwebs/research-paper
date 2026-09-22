"""
COMPUTATIONAL COUPLING --- Statistics for the unit-model comparison.

The claim under test is a sign reversal, so the analysis has to establish two things
that a difference in means does not: that the contrast is significant *in each
direction*, and that the thing separating the two directions is the silent fraction
rather than anything else that differs between the models.

Reported per network and then aggregated, because the networks differ by an order of
magnitude in size and pooling them would let the largest dominate a median.
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
CRITICAL = 1.0          # the operating point the published comparison is made at


def contrast(df: pd.DataFrame, unit: str, rho: float, null: str) -> List[Dict]:
    """Empirical minus null, paired by instance, per network.

    ``null`` names which rewiring ensemble to compare against. It is an argument and not
    a constant because the two ensembles we ran do not agree, and a function that can
    only express one of them would hide that.
    """
    out = []
    sub = df[(df.unit_model == unit) & np.isclose(df.spectral_radius, rho)]
    for net, g in sub.groupby("network"):
        a = g[g.kind == "empirical"].sort_values("instance")["memory_capacity"].to_numpy()
        b = g[g.kind == null].sort_values("instance")["memory_capacity"].to_numpy()
        k = min(len(a), len(b))
        if k < 3:
            continue
        d = a[:k] - b[:k]
        t = stats.ttest_rel(a[:k], b[:k])
        out.append(dict(
            network=net, unit_model=unit, spectral_radius=float(rho),
            mean_diff=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / np.sqrt(k)),
            p=float(t.pvalue), n=int(k),
            empirical=float(a[:k].mean()), null=float(b[:k].mean()),
            significant=bool(t.pvalue < 0.05),
            direction=int(np.sign(d.mean())) if t.pvalue < 0.05 else 0,
        ))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    # Default to the ensemble the published toolbox builds, so the headline comparison
    # is the one that replicates the published protocol rather than our first guess at it.
    ap.add_argument("--null", default="bct_randmio")
    ap.add_argument("--out", default="unit_model_stats.json")
    args = ap.parse_args()
    null = args.null

    path = os.path.join(RESULTS, "unit_model_comparison.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    # keep_default_na=False: the control arm is a literal string, and pandas' default
    # NA list contains "null". Reading it as missing drops half the experiment.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["kind"] = df["kind"].replace({"null": "rewired"})
    if null not in set(df.kind.unique()):
        print(f"[abort] null {null!r} not in file; have "
              f"{sorted(set(df.kind.unique()))}")
        return
    for c in ("memory_capacity", "participation_ratio", "dead_frac",
              "saturated_frac", "spectral_radius"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"[unit-model] {len(df)} evaluations, {df.network.nunique()} networks, "
          f"{df.instance.nunique()} instances, {df.spectral_radius.nunique()} gains\n")

    # --- The headline: same graphs, same nulls, two unit models -------------
    rows = []
    for unit in ("rate", "spiking"):
        rows += contrast(df, unit, CRITICAL, null)
    res = pd.DataFrame(rows)
    print(f"=== empirical minus {null} at rho = {CRITICAL} "
          f"(the published operating point) ===")
    print(res[["network", "unit_model", "empirical", "null", "mean_diff", "ci", "p",
               "significant"]].round(4).to_string(index=False))

    summary = {"null": null}
    for unit in ("rate", "spiking"):
        u = res[res.unit_model == unit]
        pos = int(((u.direction > 0)).sum())
        neg = int(((u.direction < 0)).sum())
        summary[unit] = dict(n_networks=int(len(u)), n_sig_positive=pos,
                             n_sig_negative=neg,
                             median_diff=float(u.mean_diff.median()))
        print(f"\n  {unit:8s}: connectome significantly LEADS in {pos} of {len(u)}, "
              f"significantly TRAILS in {neg} of {len(u)}, "
              f"median difference {u.mean_diff.median():+.3f}")

    # --- Does the sign reverse within a network? ---------------------------
    piv = res.pivot(index="network", columns="unit_model", values="direction")
    if {"rate", "spiking"} <= set(piv.columns):
        rev = ((piv["rate"] > 0) & (piv["spiking"] < 0)).sum()
        print("\n=== sign reversal within a network (rate leads, spiking trails) ===")
        print(piv.to_string())
        print(f"\n  reversal in {int(rev)} of {len(piv)} networks")
        summary["n_reversal"] = int(rev)
        summary["n_networks"] = int(len(piv))

    # --- The proposed mediator ---------------------------------------------
    print("\n=== silent fraction, by unit model (all gains, all networks) ===")
    dead = df.groupby(["unit_model", "kind"])["dead_frac"].agg(["mean", "std"])
    print(dead.round(4).to_string())
    d_rate = float(df[df.unit_model == "rate"]["dead_frac"].mean())
    d_spk = float(df[df.unit_model == "spiking"]["dead_frac"].mean())
    print(f"\n  rate units silent: {d_rate:.1%}   spiking units silent: {d_spk:.1%}")
    summary["dead_frac_rate"] = d_rate
    summary["dead_frac_spiking"] = d_spk

    # Within the spiking model, does the connectome silence more than its null, and does
    # that gap track the capacity gap? This is the mechanism claim, restricted to the
    # model in which silencing can occur at all.
    spk = df[(df.unit_model == "spiking") & df.kind.isin(["empirical", null])]
    per = spk.groupby(["network", "kind", "spectral_radius"])[
        ["memory_capacity", "dead_frac"]].mean().unstack("kind")
    gaps = pd.DataFrame({
        "cap_gap": per[("memory_capacity", "empirical")] - per[("memory_capacity", null)],
        "dead_gap": per[("dead_frac", "empirical")] - per[("dead_frac", null)],
    }).dropna()
    if len(gaps) > 3 and gaps.dead_gap.std() > 0:
        r = stats.pearsonr(gaps.dead_gap, gaps.cap_gap)
        print("\n=== within the spiking model: does extra silencing cost capacity? ===")
        print(f"  silencing gap vs capacity gap: r = {r.statistic:+.3f} "
              f"(p = {r.pvalue:.2g}) over {len(gaps)} (network, gain) cells")
        summary["spiking_gap_r"] = float(r.statistic)
        summary["spiking_gap_p"] = float(r.pvalue)
        summary["spiking_gap_n"] = int(len(gaps))

    # --- Across the whole gain axis, not only at criticality ---------------
    print("\n=== direction at every gain (+1 connectome leads, -1 trails, 0 n.s.) ===")
    allg = []
    for unit in ("rate", "spiking"):
        for rho in sorted(df.spectral_radius.unique()):
            allg += contrast(df, unit, rho, null)
    ag = pd.DataFrame(allg)
    if len(ag):
        print(ag.pivot_table(index=["unit_model", "network"],
                             columns="spectral_radius",
                             values="direction").to_string())
        for unit in ("rate", "spiking"):
            u = ag[ag.unit_model == unit]
            summary[f"{unit}_frac_positive_all_gains"] = float((u.direction > 0).mean())
            summary[f"{unit}_frac_negative_all_gains"] = float((u.direction < 0).mean())

    with open(os.path.join(RESULTS, args.out), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[done] -> {RESULTS}/{args.out}")


if __name__ == "__main__":
    main()
