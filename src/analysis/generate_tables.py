"""
COMPUTATIONAL COUPLING --- Generate LaTeX tables from raw result files.

Tables are emitted as standalone ``.tex`` fragments included by the manuscript, so
that the numbers in the paper and the numbers in ``results/`` cannot drift apart.
"""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables_generated")

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "ladder"))

DS_ORDER = ["celegans", "drosophila_cx", "drosophila_al", "mouse_v1", "human_hcp"]
DS_SHORT = {
    "celegans": r"\textit{C. elegans}",
    "drosophila_cx": r"\textit{Drosophila} CX",
    "drosophila_al": r"\textit{Drosophila} AL",
    "mouse_v1": r"Mouse V1 L2/3",
    "human_hcp": r"Human HCP",
}
RUNG_SHORT = {
    "L0_erdos_renyi": "L0 density",
    "L1_degree": "L1 $+$degree",
    "L2_degree_valence": "L2 $+$valence",
    "L3_degree_valence_motif": "L3 $+$motifs",
    "L4_connectome": "L4 connectome",
}


def _w(name: str, body: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print(f"[table] {name}")


def _placeholder(name: str, source: str) -> None:
    """Emit a visible stub so the manuscript still builds when a result is missing.

    A build that fails is easy to notice; a build that silently drops a table is not,
    and a build that reuses a stale table is worse than either.
    """
    _w(name, "\\begin{table}[t]\\centering\\small\n"
             "\\textbf{[table pending: " + source.replace("_", "\\_") + " not found]}\n"
             "\\end{table}\n")
    print(f"[table] {name} (placeholder; missing {source})")


def table_datasets() -> None:
    path = os.path.join(RES, "ladder", "ladder_meta.json")
    if not os.path.exists(path):
        _placeholder("table_datasets.tex", os.path.relpath(path, ROOT))
        return
    with open(path) as f:
        meta = json.load(f)
    rows = []
    for k in DS_ORDER:
        if k not in meta:
            continue
        m, s = meta[k], meta[k]["summary"]
        rows.append(
            f"{DS_SHORT[k]} & {m['organism'].split()[0][0]}. "
            f"{m['organism'].split()[-1]} & {int(s['n_nodes']):,} & {int(s['n_edges']):,} & "
            f"{s['density']*100:.2f} & {s['mean_degree']:.1f} & "
            f"{s['frac_inhibitory_neurons']*100:.1f} & "
            f"{'yes' if m['signed'] else r'\textbf{no}'} \\\\"
        )
    body = r"""\begin{table}[t]
\centering\small
\caption{\textbf{Size-matched subcircuits and the analyses each supports.}
Networks are induced subgraphs grown by breadth-first expansion and stripped of
degree-zero nodes. The final column records whether synaptic valence is available:
in the MICrONS release used here every annotated inhibitory soma is disconnected,
and diffusion tractography carries no valence at all, so neither network can support
sign-attributed motif analysis.}
\label{tab:datasets}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrrrrrc}
\toprule
\textbf{Network} & \textbf{Species} & \textbf{$N$} & \textbf{$|E|$} &
\textbf{Dens.\ (\%)} & \textbf{$\bar{k}$} & \textbf{Inh.\ (\%)} & \textbf{Valence} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    _w("table_datasets.tex", body)


def table_ladder() -> None:
    p = os.path.join(RES, "ladder", "ladder_raw.csv")
    if not os.path.exists(p):
        _placeholder("table_ladder.tex", os.path.relpath(p, ROOT))
        return
    df = pd.read_csv(p)
    ds = [d for d in DS_ORDER if d in set(df.dataset)]
    lines = []
    for rung, short in RUNG_SHORT.items():
        cells = []
        for d in ds:
            v = df[(df.dataset == d) & (df.rung == rung)]["accuracy"].to_numpy()
            if len(v) > 1:
                ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                cells.append(f"{100*v.mean():.1f}\\,$\\pm$\\,{100*ci:.1f}")
            elif len(v):
                cells.append(f"{100*v.mean():.1f}")
            else:
                cells.append("--")
        lines.append(f"{short} & " + " & ".join(cells) + r" \\")
    header = " & ".join(DS_SHORT[d] for d in ds)
    body = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{SHD test accuracy (\%) by ladder rung}, mean $\pm$ 95\% CI over
15 surrogates per rung. Every rung is matched on node count, edge count, spectral
radius and input projection; adjacent rungs differ in exactly one structural
property. Chance is 5.0\%; a zero-recurrence network scores \ZeroRecSplit{} under
this protocol.}
\label{tab:ladder_acc}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + "r" * len(ds) + r"""}
\toprule
\textbf{Rung} & """ + header + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    _w("table_ladder.tex", body)


def table_rate_matched() -> None:
    """The ladder re-run with each network calibrated to a common firing rate.

    Reported beside the calibrated spectral radius, because the radius each topology
    needs in order to reach the common rate is itself the finding: the connectomes sit
    at a lower-activity operating point than their surrogates at equal gain.
    """
    p = os.path.join(RES, "ladder", "ladder_rate_matched.csv")
    if not os.path.exists(p):
        _placeholder("table_rate_matched.tex", os.path.relpath(p, ROOT))
        return
    df = pd.read_csv(p)
    # Surrogates whose bisection never reached the target rate are not rate-matched.
    # Reporting their accuracy in a matched-rate table would report a comparison that
    # was not made, so they are dropped and the cell is marked.
    if "calib_ok" in df.columns:
        matched = df[df.calib_ok.astype(bool)]
    else:
        matched = df
    ds = [d for d in DS_ORDER if d in set(df.dataset)]
    n_unmatched = int(len(df) - len(matched))
    lines = []
    for rung, short in RUNG_SHORT.items():
        cells = []
        for d in ds:
            sub = matched[(matched.dataset == d) & (matched.rung == rung)]
            total = int(((df.dataset == d) & (df.rung == rung)).sum())
            v = sub["accuracy"].to_numpy()
            if len(v) == 0:
                cells.append(r"$\dagger$")
            elif len(v) > 1:
                ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                mark = r"$^{\ast}$" if len(v) < total else ""
                cells.append(f"{100*v.mean():.1f}\\,$\\pm$\\,{100*ci:.1f}{mark}")
            else:
                cells.append(f"{100*v.mean():.1f}" + (r"$^{\ast}$" if len(v) < total else ""))
        lines.append(f"{short} & " + " & ".join(cells) + r" \\")
    # A second block: the spectral radius each topology needed to hit the target rate.
    rho_lines = []
    for rung, short in RUNG_SHORT.items():
        cells = []
        for d in ds:
            v = df[(df.dataset == d) & (df.rung == rung)]["calibrated_rho"].to_numpy()
            cells.append(f"{v.mean():.2f}" if len(v) else "--")
        rho_lines.append(f"{short} & " + " & ".join(cells) + r" \\")
    n_total = int(len(df))
    header = " & ".join(DS_SHORT[d] for d in ds)
    target = df["target_rate"].iloc[0]
    n_inst = int(df.groupby(["dataset", "rung"]).size().max())
    body = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{The ladder at a matched operating point.} Each network is calibrated
individually, by bisection on its own spectral radius, to a common mean firing rate of
""" + f"{target:.2f}" + r"""; """ + f"{n_inst} surrogate" + ("s" if n_inst != 1 else "") + r""" per rung.
\textbf{Top:} SHD test accuracy (\%), mean $\pm$ 95\% CI, over the surrogates that
reached the target. \textbf{Bottom:} the spectral radius each topology required, averaged
over all surrogates including those that did not. """ + str(n_unmatched) + r""" of
""" + str(n_total) + r""" surrogates never reached the target rate: the bisection
saturates at the top of its bracket and the network still fires below target.
$^{\ast}$ marks a cell where some surrogates were excluded on that ground and
$\dagger$ one where all were --- the \textit{Drosophila} central complex is 66\%
inhibitory, so raising its gain raises its inhibition and its attainable rate is capped.
Accuracy differences that survive in Table~\ref{tab:ladder_acc} but not here were
differences in activity level rather than in wiring.}
\label{tab:rate_matched}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + "r" * len(ds) + r"""}
\toprule
\textbf{Rung} & """ + header + r""" \\
\midrule
\multicolumn{""" + str(len(ds) + 1) + r"""}{l}{\emph{Test accuracy (\%)}} \\
""" + "\n".join(lines) + r"""
\midrule
\multicolumn{""" + str(len(ds) + 1) + r"""}{l}{\emph{Calibrated spectral radius $\rho$}} \\
""" + "\n".join(rho_lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    _w("table_rate_matched.tex", body)


def table_unit_model() -> None:
    """The 2x2x2: connectome vs. two rewiring ensembles, in two unit models, at rho = 1.

    Both nulls are shown side by side and both arms as levels rather than only as
    differences. The side-by-side is the point of the table: the two columns answer to
    the same description in the literature, and where they disagree the disagreement is
    visible in one row rather than distributed over a paragraph.
    """
    import numpy as np
    from scipy import stats as _st

    p = os.path.join(RES, "ladder", "unit_model_comparison.csv")
    if not os.path.exists(p):
        _placeholder("table_unit_model.tex", os.path.relpath(p, ROOT))
        return
    # "null" is in pandas' default NA list; read the label as a literal string.
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])
    df["kind"] = df["kind"].replace({"null": "rewired"})
    for c in ("memory_capacity", "spectral_radius"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isclose(df.spectral_radius, 1.0)]

    kinds = set(df.kind.unique())
    # Fall back to the single-null layout if this file predates the two-null run, so a
    # stale CSV produces an honest narrower table rather than a crash or empty cells.
    nulls = [n for n in ("bct_randmio", "rank_reassigned") if n in kinds]
    if not nulls:
        nulls = [n for n in ("rewired",) if n in kinds]
    if not nulls:
        _placeholder("table_unit_model.tex", os.path.relpath(p, ROOT))
        return

    head = {"bct_randmio": r"\textbf{BCT} \texttt{randmio}",
            "rank_reassigned": r"\textbf{Rank-reassigned}",
            "rewired": r"\textbf{Rewired}"}
    pretty = {"human": "Human", "macaque_modha": "Macaque", "mouse": "Mouse",
              "rat": "Rat", "drosophila": r"\textit{Drosophila}"}

    def cell(a, b):
        """Null level and paired difference, or em-dashes when the cell is unusable."""
        k = min(len(a), len(b))
        if k < 3:
            return "--- & ---"
        d = a[:k] - b[:k]
        pv = float(_st.ttest_rel(a[:k], b[:k]).pvalue) if d.std() > 0 else 1.0
        star = r"$^{\ast}$" if pv < 0.05 else ""
        ci = 1.96 * d.std(ddof=1) / np.sqrt(k)
        return f"{b[:k].mean():.2f} & {d.mean():+.2f}\\,$\\pm$\\,{ci:.2f}{star}"

    lines = []
    for unit, label in (("rate", r"\emph{Graded-rate units} (\texttt{tanh})"),
                        ("spiking", r"\emph{Spiking units} (LIF)")):
        ncol = 2 + 2 * len(nulls)
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{{label}}} \\\\")
        for net in ("human", "macaque_modha", "mouse", "rat", "drosophila"):
            g = df[(df.unit_model == unit) & (df.network == net)]
            a = g[g.kind == "empirical"].sort_values("instance")[
                "memory_capacity"].to_numpy()
            if len(a) < 3:
                continue
            cells = [cell(a, g[g.kind == n].sort_values("instance")[
                "memory_capacity"].to_numpy()) for n in nulls]
            lines.append(f"\\quad {pretty[net]} & {a.mean():.2f} & "
                         + " & ".join(cells) + r" \\")

    spec = "lr" + "rr" * len(nulls)
    colhead = " & ".join(rf"\multicolumn{{2}}{{c}}{{{head[n]}}}" for n in nulls)
    subhead = " & ".join([r"\textbf{Capacity} & \textbf{Difference}"] * len(nulls))
    # Six columns do not fit a 10pt column, so this one spans the page.
    body = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{The connectome advantage depends on which rewiring it is compared
against, and on what a unit is.} Empirical connectome against two degree-preserving
rewiring ensembles at $\rho = 1$, the operating point of the published comparison, on the
five regional connectomes distributed with \texttt{conn2res}. \texttt{randmio} carries
each weight along with its edge; rank-reassignment rewires the topology and then
re-assigns the same multiset of weights by endpoint-degree rank. Both satisfy
``degree-preserving randomisation''. Identical graphs, inputs, delays and readout; ten
paired instances; paired $t$-test, $\ast$ marks $p < 0.05$. Absolute capacities are shown
because they establish whether a cell is informative: the human spiking row sits at the
floor and carries nothing, the other four do not.}
\label{tab:unitmodel}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + spec + r"""}
\toprule
& \textbf{Connectome} & """ + colhead + r""" \\
\cmidrule(lr){3-""" + str(2 + 2 * len(nulls)) + r"""}
\textbf{Network} & \textbf{Capacity} & """ + subhead + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    _w("table_unit_model.tex", body)


def table_contrasts() -> None:
    p = os.path.join(RES, "ladder", "ladder_contrasts_accuracy.csv")
    if not os.path.exists(p):
        _placeholder("table_contrasts.tex", os.path.relpath(p, ROOT))
        return
    df = pd.read_csv(p)
    lines = []
    for d in DS_ORDER:
        sub = df[df.dataset == d]
        for _, r in sub.iterrows():
            star = r"$\ast$" if r["significant_holm_05"] else ""
            lines.append(
                f"{DS_SHORT[d]} & {r['property_added']} & "
                f"{100*r['delta']:+.2f} & "
                f"[{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}] & "
                f"{r['hedges_g']:+.2f} & {r['p_holm']:.3f}{star} \\\\"
            )
    body = r"""\begin{table}[t]
\centering\small
\caption{\textbf{Contribution of each structural property}, as the difference between
adjacent ladder rungs in SHD accuracy (percentage points), with percentile-bootstrap
95\% CIs, Hedges' $g$, and Holm-corrected $p$-values across the whole family of
contrasts. $\ast$ marks $p_{\mathrm{Holm}} < 0.05$.}
\label{tab:contrasts}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrlrr}
\toprule
\textbf{Network} & \textbf{Property added} & \textbf{$\Delta$ (pp)} &
\textbf{95\% CI} & \textbf{$g$} & \textbf{$p_{\mathrm{Holm}}$} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    _w("table_contrasts.tex", body)


def table_null_validation() -> None:
    p = os.path.join(RES, "ladder", "null_validation.json")
    if not os.path.exists(p):
        _placeholder("table_null_validation.tex", os.path.relpath(p, ROOT))
        return
    with open(p) as f:
        d = json.load(f)
    lines = []
    for k in DS_ORDER:
        if k not in d:
            continue
        v = d[k]
        lines.append(
            f"{DS_SHORT[k]} & {v['max_rel_err_L3']*100:.1f} & "
            f"{v['converged_fraction']*100:.0f} & "
            f"{v['overlap_L3']*100:.1f} & {v['overlap_L2']*100:.1f} & "
            f"{v.get('pairwise_overlap_L3', float('nan'))*100:.1f} & "
            f"{v.get('pairwise_overlap_L2', float('nan'))*100:.1f} & "
            f"{v['bifan_real']:,} & {v['bifan_L3']:,.0f} & {v['bifan_L2']:,.0f} \\\\"
        )
    body = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Validation of the motif-preserving ensemble.} The chain must hit its
target census, genuinely randomise, and not reproduce structure it was not asked to
preserve. Columns 2--3: worst-case relative error across the six targeted motif counts,
and the fraction of chains meeting the 2\% convergence criterion. Columns 4--5: edge
overlap with the empirical connectome for the motif-matched (L3) and degree-matched
(L2) ensembles --- L3 is close to the unconstrained level, so it is not the
connectome in disguise. Columns 6--7: mean pairwise overlap \emph{between} ensemble
members, as evidence of mixing. L3 members are somewhat less mutually distinct than L2
members, which is expected because the constrained set is a subset of the unconstrained
one; what matters is that they remain far from identical, so the chain explores the
constraint surface rather than collapsing onto one region of it. This is empirical
evidence of adequate mixing, not a proof of uniform sampling. Columns 8--10: bi-fan
counts, a 4-node motif deliberately excluded from the objective, reproduced neither in
value nor in a consistent direction, confirming the chain constrains 3-node structure
specifically rather than reconstructing the graph.}
\label{tab:nullval}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
& \multicolumn{2}{c}{\textbf{Target match}} & \multicolumn{2}{c}{\textbf{Overlap w/ real (\%)}}
& \multicolumn{2}{c}{\textbf{Pairwise overlap (\%)}}
& \multicolumn{3}{c}{\textbf{Held-out bi-fan count}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}
\textbf{Network} & \textbf{max err (\%)} & \textbf{conv.\ (\%)} & \textbf{L3} & \textbf{L2}
& \textbf{L3} & \textbf{L2} & \textbf{empirical} & \textbf{L3} & \textbf{L2} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    _w("table_null_validation.tex", body)


def table_synthesis() -> None:
    p = os.path.join(RES, "hardware", "open_synthesis.csv")
    if not os.path.exists(p):
        _placeholder("table_synthesis.tex", os.path.relpath(p, ROOT))
        return
    df = pd.read_csv(p)
    lines = []
    for _, r in df.iterrows():
        if not r.get("ok", False):
            continue
        lines.append(
            f"\\texttt{{{str(r['top']).replace('_', chr(92)+'_')}}} & "
            f"{r.get('cells', float('nan')):,.0f} & "
            f"{r.get('flip_flops', float('nan')):,.0f} & "
            f"{r.get('cell_area_um2', float('nan')):,.0f} \\\\"
        )
    body = r"""\begin{table}[t]
\centering\small
\caption{\textbf{Logic synthesis of the accompanying RTL} with Yosys and ABC onto the
Nangate45 Open Cell Library (typical corner). These are the quantities this flow can
establish. No die area, power or energy-per-operation figure is reported: the flow
performs no placement, routing, clock-tree synthesis or parasitic extraction. No clock
frequency is reported either, because the toolchain ships no static timing analyser. The
commercial-process figures reported in earlier revisions of this project were never
measured and are withdrawn.}
\label{tab:synthesis}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
\textbf{Module} & \textbf{Cells} & \textbf{Flip-flops} &
\textbf{Cell area ($\mu$m$^2$)} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    _w("table_synthesis.tex", body)


def table_experiments() -> None:
    """Every experiment in the project, with its size counted from its own output file.

    The list itself comes from ``experiment_inventory``, which the prose documents also
    read, so the manuscript and the documentation cannot disagree about what was run. The
    counts are read from the result files at build time rather than written down, and an
    experiment whose output is missing appears as missing rather than as a remembered
    number.
    """
    from src.analysis.experiment_inventory import EXPERIMENTS, size

    # LaTeX-safe rendering of the shared descriptions, which are written as plain prose.
    def esc(t: str) -> str:
        for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                     ('"', "''"), ("x ", r"$\times$ ")):
            t = t.replace(a, b)
        return t

    lines = []
    for e in EXPERIMENTS:
        n = size(e)
        amount = f"{n:,} {e.unit}" if n is not None else r"\emph{not run}"
        lines.append(rf"\S\ref{{{e.section}}} & {esc(e.description)} & {amount} \\")

    body = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Every experiment in this study.} Counts are read from each experiment's
own output file at build time rather than transcribed, so the table cannot disagree with
what was run; an experiment whose output is absent is reported as absent. This list is the
same one the accompanying documentation prints, generated from a single source. The unit
differs by row --- a ``run'' is one network at one rung trained to convergence, an
``evaluation'' one network at one operating point probed for one metric --- so the totals
are not comparable across rows and are not summed.}
\label{tab:experiments}
\begin{tabular}{@{}l p{0.70\textwidth} r@{}}
\toprule
\textbf{Section} & \textbf{Experiment} & \textbf{Size} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    _w("table_experiments.tex", body)


def main() -> None:
    table_datasets()
    table_experiments()
    table_ladder()
    table_unit_model()
    table_contrasts()
    table_rate_matched()
    table_null_validation()
    table_synthesis()


if __name__ == "__main__":
    main()
