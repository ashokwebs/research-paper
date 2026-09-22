"""
COMPUTATIONAL COUPLING --- Why connectome structure costs accuracy, and why the cost
changes sign with gain.

The ladder establishes *that* every structural property which moves reservoir accuracy
moves it down, and the gain sweep establishes that the connectome-versus-null contrast
reverses along the gain axis in some networks.  Both are descriptions.  Neither says what
the structure is doing to the network, and without that they are a pair of facts about
one benchmark rather than an account of anything.

The hypothesis tested here is specific and mechanical.  A linear readout can only use
variance that is spread across directions, so what a reservoir supplies it with is an
*effective dimensionality*: the participation ratio

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

of its state covariance.  A spiking network loses dimensions in exactly two ways --- a
neuron pinned on carries no variance, and a neuron that never fires carries none either
--- and degree heterogeneity acts on both.  In a graph with a heavy-tailed degree
sequence, low-degree neurons sit below threshold while hubs dominate; in a homogeneous
graph, drive is spread evenly and every neuron sits in the same part of its range.

That gives a prediction with a sign, and a second one that explains the reversal:

  H1  Connectome structure reduces PR relative to a degree-matched surrogate, and the
      accuracy cost of each ladder rung is *mediated* by that reduction: conditioning on
      PR should shrink the rung effect substantially.

  H2  The two topologies lose dimensions at opposite ends of the gain axis.  A
      homogeneous graph saturates uniformly, so its PR collapses at high gain; a
      heterogeneous graph has a broad spread of effective drive, so some of its neurons
      are always in their graded range and its PR is flatter in rho.  Where the
      Erdos-Renyi rung's PR falls below the connectome's, accuracy should cross too ---
      and the crossing gains should agree.

H2 is the falsifiable one: it predicts *where* the reversal happens from a quantity
measured independently of accuracy.  If the PR crossing and the accuracy crossing occur
at unrelated gains, the mechanism is wrong and this file says so.

Usage
-----
    python experiments/ladder/analyse_mechanism.py
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
# The mediator named in H1 before the data was seen. It is kept as the primary test
# even though it fails, because swapping in whichever covariate turns out to work and
# presenting that as the hypothesis is how a post-hoc finding gets dressed as a
# prediction.
GEOM = "participation_ratio"

# Identified *after* looking at the descriptive tables, and reported as such throughout.
GEOM_POSTHOC = "dead_frac"


def _load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS, "gain_sweep_ladder.csv"))
    if "ok" in df.columns:
        df = df[df["ok"].astype(bool)]
    return df.dropna(subset=[GEOM, "accuracy"])


def mediation(df: pd.DataFrame, mediator: str = GEOM) -> List[Dict]:
    """How much of each rung's accuracy effect survives conditioning on dimensionality.

    Within a network and a gain, regress accuracy on a rung indicator, then on the rung
    indicator plus PR, and compare the rung coefficient. A large drop means the rung's
    effect runs *through* the dimensionality it produces; no drop means dimensionality is
    a bystander. This is a mediation statement about a measured covariate, not a causal
    identification -- rung and PR are not independently manipulable -- and it is reported
    as an attenuation, not as a causal fraction.
    """
    rows = []
    base = "L2_degree_valence"
    for (ds, rho), g in df.groupby(["dataset", "spectral_radius"]):
        for rung in ("L0_erdos_renyi", "L1_degree", "L3_degree_valence_motif",
                     "L4_connectome"):
            sub = g[g.rung.isin([base, rung])]
            if sub.rung.nunique() < 2 or len(sub) < 8:
                continue
            x = (sub.rung == rung).to_numpy(float)
            y = sub["accuracy"].to_numpy(float)
            z = sub[mediator].to_numpy(float)
            if np.std(x) == 0 or np.std(z) == 0:
                continue
            b_raw = np.polyfit(x, y, 1)[0]
            # Partial out PR: regress both y and x on z, then relate the residuals.
            ry = y - np.polyval(np.polyfit(z, y, 1), z)
            rx = x - np.polyval(np.polyfit(z, x, 1), z)
            b_adj = np.polyfit(rx, ry, 1)[0] if np.std(rx) > 1e-12 else np.nan
            # An attenuation ratio is meaningless where there is no effect to attenuate:
            # dividing a small adjusted coefficient by a small raw one is dividing noise
            # by noise. Only cells whose raw rung effect is resolvable against the
            # within-cell spread contribute to the summary.
            se = float(np.std(y, ddof=1) / np.sqrt(len(y))) if len(y) > 2 else np.inf
            resolvable = abs(b_raw) > 2.0 * se
            rows.append(dict(
                dataset=ds, spectral_radius=float(rho), rung=rung,
                effect_raw=float(b_raw), effect_given_pr=float(b_adj),
                attenuation=(float(1 - abs(b_adj) / abs(b_raw))
                             if resolvable and b_raw else np.nan),
                resolvable=bool(resolvable),
            ))
    return rows


def crossings(df: pd.DataFrame, mediator: str = GEOM) -> List[Dict]:
    """Where accuracy crosses, and where dimensionality crosses, per network.

    H2 predicts these coincide. Both are located by linear interpolation of the
    connectome-minus-Erdos-Renyi contrast through zero, on the same grid.
    """
    out = []
    for ds, g in df.groupby("dataset"):
        per = g.groupby(["rung", "spectral_radius"])[["accuracy", mediator]].mean()
        try:
            acc = (per.xs("L4_connectome")["accuracy"]
                   - per.xs("L0_erdos_renyi")["accuracy"])
            geo = (per.xs("L4_connectome")[mediator]
                   - per.xs("L0_erdos_renyi")[mediator])
        except KeyError:
            continue

        def _zero(series):
            x, y = series.index.to_numpy(float), series.to_numpy(float)
            s = np.sign(y)
            for i in range(len(y) - 1):
                if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
                    t = y[i] / (y[i] - y[i + 1])
                    return float(x[i] + t * (x[i + 1] - x[i]))
            return float("nan")

        out.append(dict(dataset=ds, acc_crossing=_zero(acc), geom_crossing=_zero(geo)))
    return out


def main() -> None:
    df = _load()
    print(f"[mechanism] {len(df)} evaluations, {df.dataset.nunique()} networks, "
          f"{df.spectral_radius.nunique()} gains\n")

    # --- The descriptive picture first -------------------------------------
    print("=== effective dimensionality (participation ratio) by rung and gain ===")
    pr = df.groupby(["dataset", "rung", "spectral_radius"])[GEOM].mean()
    print(pr.groupby(["rung", "spectral_radius"]).mean().unstack().round(2).to_string())

    print("\n=== dead and saturated fraction by rung and gain (pooled) ===")
    for col in ("dead_frac", "saturated_frac"):
        if col in df:
            print(f"\n{col}:")
            print(df.groupby(["rung", "spectral_radius"])[col].mean()
                  .unstack().round(3).to_string())

    # --- H1: does PR mediate the rung effects? ------------------------------
    med = pd.DataFrame(mediation(df))
    print("\n=== H1: attenuation of each rung's accuracy effect, conditioning on PR ===")
    if len(med):
        print(med.groupby("rung")[["effect_raw", "effect_given_pr", "attenuation"]]
              .median().round(4).to_string())
        n_res = int(med["resolvable"].sum())
        print(f"\ncells with a resolvable rung effect: {n_res} of {len(med)}")
        overall = float(med["attenuation"].median()) if n_res else float("nan")
        if n_res:
            print(f"median attenuation over those cells: {overall:.1%}")
        else:
            print("no resolvable effects; mediation is not estimable here")
    else:
        overall = float("nan")
        n_res = 0

    # --- Does PR predict accuracy at all? -----------------------------------
    print("\n=== PR vs accuracy, within network and gain ===")
    rs = []
    for (ds, rho), g in df.groupby(["dataset", "spectral_radius"]):
        if len(g) > 5 and g[GEOM].std() > 0:
            rs.append(stats.spearmanr(g[GEOM], g["accuracy"]).statistic)
    rs = [r for r in rs if np.isfinite(r)]
    print(f"median Spearman rho = {np.median(rs):+.3f} over {len(rs)} (network, gain) cells")
    print(f"positive in {sum(r > 0 for r in rs)} of {len(rs)}")

    # --- H2: do the crossings coincide? -------------------------------------
    cr = pd.DataFrame(crossings(df))
    print("\n=== H2: accuracy crossing vs dimensionality crossing ===")
    print(cr.round(3).to_string(index=False))
    both = cr.dropna()
    if len(both) >= 2:
        r = stats.pearsonr(both.acc_crossing, both.geom_crossing)
        print(f"\nPearson r = {r.statistic:+.3f} (p = {r.pvalue:.3f}) over "
              f"{len(both)} networks with both crossings")
    else:
        print("\nToo few networks have both crossings to correlate them.")

    # --- The post-hoc mediator ---------------------------------------------
    # PR was the pre-specified mediator and it fails. The descriptive tables point
    # instead at the fraction of readout neurons that never fire, which is monotone in
    # the ladder at every gain. That was found by looking, so it is tested here and
    # labelled post hoc wherever it is reported; it is a hypothesis for a future
    # experiment to pre-register, not a confirmed mechanism.
    print("\n" + "=" * 72)
    print(f"POST HOC mediator: {GEOM_POSTHOC} (identified after inspecting the data)")
    print("=" * 72)
    med2 = pd.DataFrame(mediation(df, GEOM_POSTHOC))
    att2 = float("nan")
    if len(med2):
        print(med2.groupby("rung")[["effect_raw", "effect_given_pr", "attenuation"]]
              .median().round(4).to_string())
        n_res2 = int(med2["resolvable"].sum())
        att2 = float(med2["attenuation"].median()) if n_res2 else float("nan")
        print(f"\nresolvable cells: {n_res2} of {len(med2)}; "
              f"median attenuation: {att2:.1%}")

    rs2 = []
    for (ds, rho), g in df.groupby(["dataset", "spectral_radius"]):
        if len(g) > 5 and g[GEOM_POSTHOC].std() > 0:
            rs2.append(stats.spearmanr(g[GEOM_POSTHOC], g["accuracy"]).statistic)
    rs2 = [r for r in rs2 if np.isfinite(r)]
    print(f"\n{GEOM_POSTHOC} vs accuracy: median Spearman = {np.median(rs2):+.3f}, "
          f"negative in {sum(r < 0 for r in rs2)} of {len(rs2)} cells")

    cr2 = pd.DataFrame(crossings(df, GEOM_POSTHOC))
    print(f"\ncrossing of accuracy vs crossing of {GEOM_POSTHOC}:")
    print(cr2.round(3).to_string(index=False))
    both2 = cr2.dropna()
    r2 = None
    if len(both2) >= 3:
        r2 = stats.pearsonr(both2.acc_crossing, both2.geom_crossing)
        print(f"Pearson r = {r2.statistic:+.3f} (p = {r2.pvalue:.3f}) over {len(both2)}")

    # --- The relationship at the level of the structural manipulation ------
    # Instance-level mediation is diluted by surrogate-to-surrogate noise in the
    # mediator. The structural variable is the rung, so the question "does silencing
    # track the ladder" is asked at the rung mean, where it belongs.
    print("\n" + "=" * 72)
    print("RUNG-LEVEL: does silencing track the accuracy ladder?")
    print("=" * 72)
    rung_r = []
    for (ds, rho), g in df.groupby(["dataset", "spectral_radius"]):
        pm = g.groupby("rung")[["accuracy", GEOM_POSTHOC]].mean()
        if len(pm) >= 5 and pm[GEOM_POSTHOC].std() > 0:
            rung_r.append(stats.pearsonr(pm[GEOM_POSTHOC], pm["accuracy"]).statistic)
    rung_r = np.array([r for r in rung_r if np.isfinite(r)])
    print(f"  median r across rungs = {np.median(rung_r):+.3f}, "
          f"negative in {(rung_r < 0).sum()} of {len(rung_r)} (network, gain) cells")
    pooled = df.groupby(["dataset", "spectral_radius", "rung"])[
        ["accuracy", GEOM_POSTHOC]].mean().reset_index()
    r_pool = stats.pearsonr(pooled[GEOM_POSTHOC], pooled["accuracy"])
    print(f"  pooled over {len(pooled)} cells: r = {r_pool.statistic:+.3f} "
          f"(p = {r_pool.pvalue:.2g})")

    # Does the silencing gap predict the accuracy gap, gain by gain?
    per2 = df.groupby(["dataset", "rung", "spectral_radius"])[
        ["accuracy", GEOM_POSTHOC]].mean().unstack("rung")
    gaps = pd.DataFrame({
        "acc_gap": per2[("accuracy", "L4_connectome")]
        - per2[("accuracy", "L0_erdos_renyi")],
        "dead_gap": per2[(GEOM_POSTHOC, "L4_connectome")]
        - per2[(GEOM_POSTHOC, "L0_erdos_renyi")],
    }).dropna()
    r_gap = stats.pearsonr(gaps.dead_gap, gaps.acc_gap)
    slope = float(np.polyfit(gaps.dead_gap, gaps.acc_gap, 1)[0])
    print(f"\n  silencing gap vs accuracy gap: r = {r_gap.statistic:+.3f} "
          f"(p = {r_gap.pvalue:.2g}) over {len(gaps)} cells")
    print(f"  slope = {100*slope:+.1f} accuracy points per unit of dead fraction")

    out = dict(
        n_rows=int(len(df)),
        median_attenuation=overall,
        n_resolvable_cells=int(n_res),
        median_spearman_pr_accuracy=float(np.median(rs)) if rs else None,
        n_spearman_positive=int(sum(r > 0 for r in rs)),
        n_spearman_cells=int(len(rs)),
        crossings=cr.to_dict("records"),
        mediation_by_rung=(med.groupby("rung")["attenuation"].median().to_dict()
                           if len(med) else {}),
        posthoc_mediator=GEOM_POSTHOC,
        posthoc_median_attenuation=att2,
        posthoc_median_spearman=float(np.median(rs2)) if rs2 else None,
        posthoc_n_spearman_negative=int(sum(r < 0 for r in rs2)),
        posthoc_n_spearman_cells=int(len(rs2)),
        posthoc_crossings=cr2.to_dict("records") if len(cr2) else [],
        posthoc_crossing_r=(float(r2.statistic) if r2 is not None else None),
        posthoc_crossing_p=(float(r2.pvalue) if r2 is not None else None),
        rung_level_median_r=float(np.median(rung_r)),
        rung_level_n_negative=int((rung_r < 0).sum()),
        rung_level_n_cells=int(len(rung_r)),
        rung_level_pooled_r=float(r_pool.statistic),
        rung_level_pooled_p=float(r_pool.pvalue),
        gap_r=float(r_gap.statistic),
        gap_p=float(r_gap.pvalue),
        gap_slope_pp=float(100 * slope),
        gap_n_cells=int(len(gaps)),
    )
    with open(os.path.join(RESULTS, "mechanism.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] -> {RESULTS}/mechanism.json")


if __name__ == "__main__":
    main()
