"""
COMPUTATIONAL COUPLING --- Figures for the structural control ladder.

Design notes
------------
The ladder rungs are an *ordered* sequence (each adds one property to the previous),
so they are encoded with a single-hue sequential ramp rather than categorical hues.
Datasets are shown as small multiples rather than five overlaid series, because five
categorical hues cannot be separated reliably under colour-vision deficiency when all
pairs appear together.  Every panel carries the two reference levels that make the
numbers interpretable: the chance rate, and the zero-recurrence control that fixes
what "no contribution from wiring" looks like on this benchmark.
"""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sequential ramp for the ordered ladder rungs (light -> dark, single hue).
RUNG_RAMP = ["#c6dbf0", "#8fbce4", "#5a9ad9", "#2a78d6", "#14508f"]
# Categorical slots 1-3 from the validated reference palette (all-pairs safe).
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e2de"

SHORT_RUNG = {
    "L0_erdos_renyi": "L0\ndensity",
    "L1_degree": "L1\n+degree",
    "L2_degree_valence": "L2\n+valence",
    "L3_degree_valence_motif": "L3\n+motifs",
    "L4_connectome": "L4\nreal",
}
RUNG_ORDER = list(SHORT_RUNG)


def _style(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figure_ladder(
    raw_csv: str,
    out_path: str,
    dataset_labels: Dict[str, str],
    chance: float = 0.05,
    zero_recurrence: Optional[float] = None,
    metric: str = "accuracy",
) -> None:
    """Small multiples of rung means with 95% CIs, plus the contrast forest plot."""
    df = pd.read_csv(raw_csv)
    datasets = [d for d in dataset_labels if d in set(df.dataset)]

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, len(datasets), height_ratios=[1.0, 0.95], hspace=0.55,
                          wspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.10)

    # --- Row 1: per-dataset ladder -----------------------------------------
    ymin, ymax = 1.0, 0.0
    stats: Dict[str, Dict[str, tuple]] = {}
    for d in datasets:
        sub = df[df.dataset == d]
        s = {}
        for r in RUNG_ORDER:
            v = sub[sub.rung == r][metric].to_numpy()
            if len(v):
                m = v.mean()
                ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                s[r] = (m, ci)
                ymin, ymax = min(ymin, m - ci), max(ymax, m + ci)
        stats[d] = s

    pad = 0.06 * (ymax - ymin)
    lo, hi = max(0.0, ymin - pad), ymax + pad

    for j, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, j])
        _style(ax)
        s = stats[d]
        xs = np.arange(len(RUNG_ORDER))
        ms = np.array([s[r][0] if r in s else np.nan for r in RUNG_ORDER])
        cs = np.array([s[r][1] if r in s else np.nan for r in RUNG_ORDER])
        ax.plot(xs, ms, color=MUTED, lw=1.2, zorder=1)
        for i, r in enumerate(RUNG_ORDER):
            if r not in s:
                continue
            ax.errorbar(i, ms[i], yerr=cs[i], color=RUNG_RAMP[i], lw=2.0,
                        capsize=3, capthick=1.4, zorder=3)
            ax.plot(i, ms[i], "o", ms=7, color=RUNG_RAMP[i],
                    markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        # Reference levels are drawn only when they fall inside the plotted range;
        # otherwise they are stated in the panel so the scale is never misread as
        # spanning them.
        for lvl, style in ((chance, ":"), (zero_recurrence, "--")):
            if lvl is not None and lo <= lvl <= hi:
                ax.axhline(lvl, color=MUTED, ls=style, lw=1.0, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT_RUNG[r] for r in RUNG_ORDER], fontsize=7)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        ax.set_title(dataset_labels[d], fontsize=8.5, color=INK, pad=6)
        if j == 0:
            ax.set_ylabel("SHD test accuracy", fontsize=9, color=INK)
        else:
            ax.tick_params(labelleft=False)
    if zero_recurrence is not None:
        fig.text(0.07, 0.945,
                 f"chance {100*chance:.1f}%   ·   zero-recurrence control "
                 f"{100*zero_recurrence:.1f}%   (both below every panel's range)",
                 fontsize=7.5, color=INK2, ha="left", va="bottom")

    # --- Row 2: contrast forest plot ---------------------------------------
    cpath = raw_csv.replace("ladder_raw.csv", f"ladder_contrasts_{metric}.csv")
    if os.path.exists(cpath):
        cdf = pd.read_csv(cpath)
        ax = fig.add_subplot(gs[1, :])
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.8)

        labels: List[str] = []
        y = 0
        ymap = []
        for d in datasets:
            sub = cdf[cdf.dataset == d]
            for _, row in sub.iterrows():
                ymap.append((y, row, d))
                labels.append(f"{dataset_labels[d].split('(')[0].strip()}  ·  "
                              f"{row.property_added}")
                y += 1
            y += 0.6
        ax.axvline(0.0, color=INK2, lw=1.0, zorder=1)
        for yy, row, d in ymap:
            sig = bool(row.significant_holm_05)
            col = CAT[1] if sig else MUTED
            ax.plot([row.ci_low, row.ci_high], [yy, yy], color=col, lw=2.0, zorder=2,
                    solid_capstyle="round")
            ax.plot(row.delta, yy, "o", ms=6, color=col,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_yticks([yy for yy, _, _ in ymap])
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:+.0f}"))
        ax.set_xlabel("Change in SHD accuracy when the property is added, "
                      "percentage points (95% bootstrap CI)", fontsize=9, color=INK)
        ax.text(0.995, 1.04, "filled = significant after Holm correction",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=INK2)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_rate_matched(
    matched_gain_csv: str,
    rate_matched_csv: str,
    out_path: str,
    dataset_labels: Dict[str, str],
    chance: float = 0.05,
) -> None:
    """The ladder at matched gain against the ladder at a matched firing rate.

    Colour encodes the *matching regime*, not the dataset or the rung, because the
    comparison the figure exists to make is between the two regimes within a network.
    Datasets are small multiples for the same reason they are in
    :func:`figure_ladder`.  Two hues are used, which is the only count that separates
    reliably under every common colour-vision deficiency.

    The bottom row shows the confound rather than asserting it: mean firing rate by
    rung under each regime.  Under matched gain it varies by rung; under matched rate
    it sits on the target by construction, which is what makes the top-right panel a
    topology comparison rather than an activity comparison.
    """
    mg = pd.read_csv(matched_gain_csv)
    rm = pd.read_csv(rate_matched_csv)
    # Surrogates whose bisection never reached the target rate are not rate-matched;
    # plotting them on the matched-rate series would draw a comparison that was not
    # made.  Same rule as the table.
    if "calib_ok" in rm.columns:
        rm = rm[rm.calib_ok.astype(bool)]
    datasets = [d for d in dataset_labels
                if d in set(mg.dataset) and d in set(rm.dataset)]
    if not datasets:
        return

    GAIN, RATE = CAT[1], CAT[0]          # orange = matched gain, blue = matched rate
    fig = plt.figure(figsize=(2.25 * len(datasets) + 0.6, 5.6))
    gs = fig.add_gridspec(2, len(datasets), height_ratios=[1.0, 0.8], hspace=0.42,
                          wspace=0.24, left=0.085, right=0.985, top=0.86, bottom=0.11)

    def _series(df, d, col):
        out = []
        for r in RUNG_ORDER:
            v = df[(df.dataset == d) & (df.rung == r)][col].to_numpy()
            if len(v):
                ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                out.append((v.mean(), ci))
            else:
                out.append((np.nan, np.nan))
        return np.array(out)

    xs = np.arange(len(RUNG_ORDER))

    def _draw(ax, arr, color):
        ax.plot(xs, arr[:, 0], "-o", ms=4.5, lw=1.5, color=color,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.fill_between(xs, arr[:, 0] - arr[:, 1], arr[:, 0] + arr[:, 1],
                        color=color, alpha=0.18, lw=0, zorder=2)

    # Shared accuracy scale across all panels, so panel-to-panel height is comparable.
    acc = {(t, d): _series(df, d, "accuracy")
           for t, df in (("gain", mg), ("rate", rm)) for d in datasets}
    band = np.concatenate([np.r_[a[:, 0] - a[:, 1], a[:, 0] + a[:, 1]]
                           for a in acc.values()])
    alo, ahi = np.nanmin(band), np.nanmax(band)
    apad = 0.07 * (ahi - alo)

    rate = {(t, d): _series(df, d, "mean_rate")
            for t, df in (("gain", mg), ("rate", rm)) for d in datasets}
    rband = np.concatenate([a[:, 0] for a in rate.values()])
    rlo, rhi = np.nanmin(rband), np.nanmax(rband)
    rpad = 0.10 * (rhi - rlo)
    target = float(rm["target_rate"].iloc[0])

    for j, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, j])
        _style(ax)
        _draw(ax, acc[("gain", d)], GAIN)
        _draw(ax, acc[("rate", d)], RATE)
        ax.set_xticks(xs)
        ax.set_xticklabels([r.split("_")[0] for r in RUNG_ORDER], fontsize=7.5)
        ax.set_ylim(alo - apad, ahi + apad)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        ax.set_title(dataset_labels[d], fontsize=8.5, color=INK, pad=6)
        if j == 0:
            ax.set_ylabel("SHD test accuracy", fontsize=9, color=INK)
        else:
            ax.tick_params(labelleft=False)

        ax = fig.add_subplot(gs[1, j])
        _style(ax)
        ax.axhline(target, color=MUTED, ls="--", lw=1.0, zorder=0)
        _draw(ax, rate[("gain", d)], GAIN)
        _draw(ax, rate[("rate", d)], RATE)
        ax.set_xticks(xs)
        ax.set_xticklabels([r.split("_")[0] for r in RUNG_ORDER], fontsize=7.5)
        ax.set_ylim(rlo - rpad, rhi + rpad)
        if j == 0:
            ax.set_ylabel("Mean firing rate", fontsize=9, color=INK)
        else:
            ax.tick_params(labelleft=False)

    handles = [
        matplotlib.lines.Line2D([], [], color=GAIN, marker="o", ms=4.5, lw=1.5,
                                markeredgecolor="white", label="matched gain"),
        matplotlib.lines.Line2D([], [], color=RATE, marker="o", ms=4.5, lw=1.5,
                                markeredgecolor="white", label="matched rate"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.085, 0.985),
               ncol=2, frameon=False, fontsize=8.5, labelcolor=INK,
               handlelength=1.6, columnspacing=1.6)
    fig.text(0.085, 0.905,
             f"dashed line, the common target rate {target:.2f}; bands are 95% CIs",
             fontsize=7.2, color=INK2, ha="left", va="bottom")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_pruning(
    csv_path: str,
    out_path: str,
    dataset_labels: Dict[str, str],
    chance: float = 0.05,
) -> None:
    """Survival curves under each pruning strategy, and the gamma-R90 relation."""
    df = pd.read_csv(csv_path)
    datasets = [d for d in dataset_labels if d in set(df.dataset)]
    strategies = ["random", "hub", "adversarial"]
    strat_label = {"random": "random", "hub": "degree-targeted",
                   "adversarial": "adversarial"}

    # Accuracy retained relative to each network's own unpruned baseline.  The bottom
    # panel pools across networks whose baselines run from roughly 26% to 43%, and
    # pooling raw accuracies there would plot network identity as much as perturbation
    # effect.
    base = (df[df.fraction == 0].groupby(["dataset", "seed"])["accuracy"]
            .mean().rename("baseline"))
    df = df.merge(base, on=["dataset", "seed"], how="left")
    df["retained"] = df["accuracy"] / df["baseline"]

    fig = plt.figure(figsize=(11.0, 5.6))
    gs = fig.add_gridspec(2, len(datasets), height_ratios=[1.0, 0.85],
                          hspace=0.5, wspace=0.28,
                          left=0.07, right=0.98, top=0.90, bottom=0.12)

    for j, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, j])
        _style(ax)
        sub = df[df.dataset == d]
        for k, s in enumerate(strategies):
            g = sub[sub.strategy == s].groupby("fraction")["accuracy"]
            m, sd, n = g.mean(), g.std(ddof=1), g.count()
            ci = 1.96 * sd / np.sqrt(n.clip(lower=1))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
            ax.plot(m.index, m.values, color=CAT[k], lw=2.0,
                    marker="osD"[k], ms=5, markeredgecolor="white",
                    markeredgewidth=1.0, label=strat_label[s], zorder=3)
            ax.fill_between(m.index, m - ci, m + ci, color=CAT[k], alpha=0.16, lw=0)
        ax.axhline(chance, color=MUTED, ls=":", lw=1.0)
        ax.set_xlabel("fraction of synapses removed", fontsize=8)
        ax.set_title(dataset_labels[d], fontsize=8.5, color=INK, pad=6)
        if j == 0:
            ax.set_ylabel("SHD test accuracy", fontsize=9, color=INK)
            ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # Row 2: the claim the theory actually makes, tested within each network.
    #
    # An earlier version pooled retained accuracy against gamma across all networks on
    # one axis.  That cannot work: gamma is a realised fraction of a worst-case constant
    # C(T, ||A||_2) that differs between networks by five orders of magnitude, so the
    # pooled scatter plotted network identity along x and showed no relation at all.
    # The prediction is ordinal and within-network -- higher gamma, lower tolerance --
    # so that is what is drawn, one panel per network, three points per panel.
    fmax = float(df["fraction"].max())
    for j, d in enumerate(datasets):
        ax = fig.add_subplot(gs[1, j])
        _style(ax)
        sub = df[df.dataset == d]
        pts = []
        for k, st in enumerate(strategies):
            g = sub[sub.strategy == st]
            if not len(g):
                continue
            gam = float(g[g.fraction > 0]["gamma"].median())
            tol, cens = [], False
            for seed, gg in g.groupby("seed"):
                gg = gg.sort_values("fraction")
                base = float(gg[gg.fraction == 0]["accuracy"].iloc[0])
                ok = gg[gg.accuracy >= 0.9 * base]["fraction"]
                v = float(ok.max()) if len(ok) else 0.0
                tol.append(v)
                cens |= v >= fmax - 1e-9
            pts.append((gam, float(np.mean(tol)), CAT[k], st, cens))
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=MUTED, lw=1.0, zorder=1)
            for gam, tol, col, st, cens in pts:
                ax.plot(gam, tol, "osD"["random hub adversarial".split().index(st)],
                        ms=8, color=col, markeredgecolor="white", markeredgewidth=1.2,
                        zorder=3)
                if cens:
                    ax.annotate("", xy=(gam, min(1.02, tol + 0.06)), xytext=(gam, tol),
                                arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.08)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        # gamma spans only a small range within a panel, so the default log locator
        # emits a dozen overlapping minor labels.  Two ticks, at the extremes, are the
        # only ones that carry information here.
        if pts:
            lo_g, hi_g = pts[0][0], pts[-1][0]
            ax.set_xlim(lo_g / 1.6, hi_g * 1.6)
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([lo_g, hi_g]))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: (f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
                              if v > 0 else "")))
            ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel(r"median $\gamma$", fontsize=8)
        if j == 0:
            ax.set_ylabel("tolerated fraction\n(retains 90% accuracy)",
                          fontsize=8.5, color=INK)
        else:
            ax.tick_params(labelleft=False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_dynamics(
    csv_path: str,
    out_path: str,
    dataset_labels: Dict[str, str],
    metrics: Sequence[str] = ("memory_capacity", "kernel_rank", "capability"),
    metric_labels: Optional[Dict[str, str]] = None,
) -> None:
    """One panel per dynamical probe; rung on x, one line per network."""
    df = pd.read_csv(csv_path)
    datasets = [d for d in dataset_labels if d in set(df.dataset)]
    labels = metric_labels or {
        "memory_capacity": "linear memory capacity",
        "kernel_rank": "kernel rank",
        "capability": "kernel $-$ generalisation rank",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 3.1))
    xs = np.arange(len(RUNG_ORDER))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        _style(ax)
        for j, d in enumerate(datasets):
            sub = df[df.dataset == d]
            ms, cs = [], []
            for r in RUNG_ORDER:
                v = sub[sub.rung == r][metric].to_numpy()
                ms.append(v.mean() if len(v) else np.nan)
                cs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            ms, cs = np.array(ms), np.array(cs)
            # Datasets are small multiples elsewhere; here they are distinguished by
            # marker as well as hue so identity never rests on colour alone.
            ax.errorbar(xs, ms, yerr=cs, lw=1.6, capsize=2.5,
                        color=RUNG_RAMP[min(j, len(RUNG_RAMP) - 1)] if j >= len(CAT)
                        else CAT[j],
                        marker="osD^v"[j % 5], ms=5, markeredgecolor="white",
                        markeredgewidth=1.0,
                        label=dataset_labels[d].split("(")[0].strip())
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT_RUNG[r] for r in RUNG_ORDER], fontsize=6.5)
        ax.set_title(labels.get(metric, metric), fontsize=9, color=INK, pad=6)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_motifs(
    csv_path: str,
    out_path: str,
    dataset_labels: Dict[str, str],
    alpha: float = 0.05,
) -> None:
    """Motif enrichment by network, showing only entries the data can support.

    Colour encodes the *outcome* of each test --- enriched, depleted, or
    indistinguishable from the degree-preserving null --- rather than the network,
    because the finding this figure exists to convey is that the outcome is not the
    same across networks.  Three hues separate reliably under every common colour-vision
    deficiency; five, one per network, would not, which is why the networks are small
    multiples here for the same reason they are in :func:`figure_ladder`.

    Bars are drawn only where the dataset can support the motif's definition: coherent
    and incoherent feedforward loops need synaptic valence, which tractography does not
    carry and which the mouse release does not supply in connected form.
    """
    df = pd.read_csv(csv_path)
    df = df[df.interpretable]
    order = ["ffl_total", "cffl", "iffl", "recip_pair", "recip_inhib_pair",
             "cycle3", "bifan"]
    pretty = {"ffl_total": "FFL", "cffl": "cFFL", "iffl": "iFFL",
              "recip_pair": "recip.", "recip_inhib_pair": "recip.\ninhib.",
              "cycle3": "3-cycle", "bifan": "bi-fan"}
    datasets = [d for d in dataset_labels if d in set(df.dataset)]
    if not datasets:
        return

    UP, DOWN, NS = CAT[0], CAT[1], MUTED

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(2.15 * len(datasets) + 0.5, 3.1), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, datasets):
        _style(ax)
        sub = df[df.dataset == d].set_index("motif")
        vals, cols, pos, ticks = [], [], [], []
        for i, mkey in enumerate(order):
            ticks.append(i)
            if mkey not in sub.index:
                continue
            z = float(sub.loc[mkey, "z_score"])
            pv = float(sub.loc[mkey, "p_value"])
            vals.append(z)
            pos.append(i)
            cols.append(NS if pv > alpha else (UP if z > 0 else DOWN))
        ax.bar(pos, vals, width=0.72, color=cols, edgecolor="white", linewidth=0.6,
               zorder=3)
        ax.axhline(0, color=INK2, lw=1.0, zorder=2)
        ax.set_yscale("symlog", linthresh=5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([pretty[m] for m in order], fontsize=6.4, rotation=90)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_title(dataset_labels[d], fontsize=8.5, color=INK, pad=6)
    axes[0].set_ylabel("enrichment $Z$ (symlog)", fontsize=9, color=INK)

    handles = [
        matplotlib.patches.Patch(color=UP, label="enriched"),
        matplotlib.patches.Patch(color=DOWN, label="depleted"),
        matplotlib.patches.Patch(color=NS, label=f"n.s. ($p>{alpha:g}$)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10),
               ncol=3, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_operating_point(
    gain_csv: str,
    memory_csv: str,
    out_path: str,
    dataset_labels: Dict[str, str],
) -> None:
    """Where on the gain axis the connectome-versus-null contrast changes sign.

    This is the figure the paper turns on, so it is drawn as the contrast itself rather
    than as two performance curves the reader has to subtract by eye. Zero is the line
    that matters: above it the unstructured or degree-matched null leads, below it the
    connectome does. A published result is a single point on one of these curves.

    Colour encodes nothing here -- the networks are small multiples -- and the shaded
    band marks the critical region where the reservoir literature makes its comparison.
    """
    fig = plt.figure(figsize=(11.0, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.52,
                          left=0.08, right=0.98, top=0.90, bottom=0.11)

    panels = []
    if os.path.exists(gain_csv):
        g = pd.read_csv(gain_csv)
        # Cells whose evaluation raised are recorded as rows rather than dropped, so
        # they must be filtered here; a NaN would otherwise poison the whole mean.
        if "ok" in g.columns:
            g = g[g["ok"].astype(bool)]
        per = g.groupby(["dataset", "rung", "spectral_radius"])["accuracy"].mean()
        w = per.unstack(level="rung")
        panels.append((
            (w["L4_connectome"] - w["L0_erdos_renyi"]).unstack(level="spectral_radius"),
            "SHD accuracy:\nconnectome $-$ Erdős–Rényi",
            lambda v, _: f"{100*v:+.0f}",
        ))
    if os.path.exists(memory_csv):
        mdf = pd.read_csv(memory_csv)
        per = mdf.groupby(["dataset", "rung", "spectral_radius"])["memory_capacity"].mean()
        w = per.unstack(level="rung")
        panels.append((
            (w["L4_connectome"] - w["L2_degree_valence"]).unstack(level="spectral_radius"),
            "Memory capacity:\nconnectome $-$ degree-matched null",
            lambda v, _: f"{v:+.1f}",
        ))
    if not panels:
        return

    for row, (mat, ylab, fmt) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        _style(ax)
        xs = np.array(sorted(mat.columns.astype(float)))
        # The critical region, where the published comparison is made.
        ax.axvspan(0.95, 1.15, color=CAT[1], alpha=0.10, lw=0, zorder=0)
        ax.axhline(0.0, color=INK2, lw=1.2, zorder=2)
        # Five networks on one axis is past what a single hue ramp separates, so shape
        # carries the identity and the ramp only orders them.
        marks = ["o", "s", "D", "^", "v"]
        for k, d in enumerate([x for x in dataset_labels if x in mat.index]):
            y = mat.loc[d, xs].to_numpy(dtype=float)
            ax.plot(xs, y, "-", marker=marks[k % len(marks)], ms=5.0, lw=1.5,
                    color=RUNG_RAMP[min(k, 4)],
                    markeredgecolor="white", markeredgewidth=0.9,
                    label=dataset_labels[d] if row == 0 else None, zorder=3)
        ax.set_xlabel(r"spectral radius $\rho$", fontsize=9, color=INK)
        ax.set_ylabel(ylab, fontsize=8.5, color=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=7.5)
        if row == 0:
            ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center",
                      bbox_to_anchor=(0.5, 1.30), labelcolor=INK2)
        # Label the two protocols on the axis itself.
        ax.text(1.05, ax.get_ylim()[1], "criticality\n(published protocol)",
                fontsize=6.8, color=CAT[1], ha="center", va="top")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
