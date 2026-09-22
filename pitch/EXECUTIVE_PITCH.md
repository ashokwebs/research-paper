# Computational Coupling — Executive Summary

**Four choices decide whether a connectome beats its rewiring.**

*Computational coupling* is the extent to which what a network computes is attributable
to how it is wired. It is never measured directly — only as the gap between a network's
score and the scores of surrogate networks, under some unit model, at some operating
point, through some readout. This project shows that each of those four choices can flip
the sign of that gap on identical data, so a coupling reported as a single signed number
is under-determined by its own methods section.

**Ashok Raj Pasala** · **Snigdha Gorai**
Under the guidance of **Jeevana Jyothi P.**
School of Computer Science and Engineering, VIT-AP University,
Amaravati 522237, Andhra Pradesh, India

Version 5.0 · September 2026

---

> **This document replaces a withdrawn one.** Version 4 of this project circulated a pitch
> whose hardware figures — a TSMC 28 nm node, a die area, 116.71 pJ/SynOp, a 455× energy
> reduction, a "physical tapeout" — were never measured by any tool. They came from a
> Python model with those values written in as constants. Every one is retracted. The
> retracted document is kept at `paper/superseded_v4/pitch/` behind its own notice, and
> `AUDIT.md` gives the evidence for each withdrawal.
>
> This version claims nothing that is not produced by a script in this repository from a
> result file also in this repository.

---

## 0. The headline

The reservoir-computing literature reports that empirical connectomes outperform
degree-preserving rewirings on memory capacity. That result is increasingly cited as a
reason to build **neuromorphic hardware** on connectome-derived wiring.

**We tried to check it and found that the comparison is not one experiment.** It is a
composition of four choices, each normally reported in a subordinate clause, each of which
can flip the sign of the answer on identical data. We measured all four on the five
connectomes distributed with the original authors' own toolbox.

![The connectome's measured advantage over two rewirings that answer to the same
description in the literature. Blue and orange differ only in what happened to the edge
weights during randomisation. In four of five networks they point in opposite
directions.](paper/figures/fig_null_choice.png)

*(A) The result. (B) The cause: one rewiring installs a weight-degree association the
real graphs do not have. (C) The test that could have refuted (B) — on the one network
with no weight variation, the two procedures must agree, and they do, to 0.008.*

**The first one caught us.** "Degree-preserving randomisation" names at least two
different ensembles — one where each weight travels with its edge (what the published
toolbox actually calls), one where the weight multiset is re-assigned to the rewired graph
afterwards. Both match the phrase. We built the second believing it was the first:

| null ensemble | connectome beats it, graded units | median difference |
|---|---|---|
| BCT `randmio` — the published one | **1 of 5** lead, **3 of 5** trail | −0.51 |
| rank-reassigned — ours, by mistake | **5 of 5** lead, all significant | +2.13 |

Across 10 (network, unit model) cells the two nulls give opposite **signs** in 6 and
opposite **significant verdicts** in 3 — one says the connectome wins at p < 0.05, the
other says it loses at p < 0.05, same graphs, same operating point.

Our version produced a clean, large, highly significant result in the expected direction.
Nothing in its output indicated it was answering a different question. It was caught by
reading the toolbox source, not by anything in the numbers.

**We can say why, and the explanation was falsifiable.** Rank-reassignment does not merely
randomise — it *installs* a weight–degree association the data does not have (+0.66 median
shift, against +0.06 for the published ensemble). That predicts the two must agree exactly
on a network with no weight variation. One of the five is binary. There they agree to
**0.008 capacity units**.

This is not a claim that the published result is wrong — our harness differs in other
respects too. It is a demonstration that a replication which builds its null from a
published description is not a replication, and cannot tell which hypothesis it tested.

**The other three choices**: the unit model (graded vs spiking — capacity compresses 5×,
and neuromorphic hardware is the spiking one), the operating point (the contrast changes
sign along the gain axis in 4 of 5 networks, significant on both sides), and the readout
protocol (under an all-to-all readout, a reservoir with its recurrent matrix **set to
zero** scores 0.596 against the connectome's 0.596 — the wiring is not being measured at
all).

**One mechanism we proposed changed sign when the null was corrected, and we report
that too.** We expected a threshold to silence a heavy-tailed graph's low-degree neurons
differentially. Against our mistaken null the silencing gap did not predict the capacity
gap, and we wrote the prediction up as failed. Against the published null it does —
r = −0.58, p = 2.5×10⁻⁵. A wrong control manufactured a clean positive result in one
place and destroyed a real one in another, from the same code.

## 1. The problem

Connectomes contain statistically overrepresented subgraphs — feedforward loops,
reciprocal inhibitory pairs. A large literature argues these *motifs* are what make
biological wiring computationally effective, and a neuromorphic literature builds
architectures on that premise.

The argument has never been properly tested, for a reason that is methodological rather
than empirical. Establishing that a structural property **causes** a computational
advantage needs a control that differs in that property **and nothing else**. The
standard comparison — a connectome against a degree-preserving rewiring, or against an
Erdős–Rényi graph — does not have that form. Rewiring a connectome changes its motif
census *together with* its clustering, its spectrum, and every higher-order statistic
that co-varies with them. A performance difference cannot be attributed to motifs.

The control that would settle it does not exist: an ensemble that randomises a connectome
while **holding its motif census fixed**.

## 2. What we built

**A motif-preserving null ensemble.** Degree-, valence- and census-matched surrogates,
sampled by an annealed Metropolis chain over exact, incrementally updated motif counts.
On *C. elegans* it reproduces all six targeted motif counts to within **0.3%** while
retaining only **13.9%** edge overlap with the empirical graph — against **11.6%** for an
unconstrained rewiring. It converges on all five networks we test, including diffusion
tractography, where almost every edge is reciprocal and an ordinary swap chain cannot
hold the census at all.

**A five-rung control ladder.** Density → degree sequence → valence placement → motif
census → the empirical connectome. Each rung adds exactly one structural property, so
each property's contribution is the difference between adjacent rungs, with node count,
edge count, spectral radius and input projection matched throughout.

**Two controls that catch silent failures.** Both are cheap, both are transferable, and
both changed our own conclusions:

| Control | What it exposes |
|---|---|
| **Zero-recurrence** | Under an all-to-all protocol, a reservoir with its recurrent matrix **set to zero** scores 59.6% — against 59.6% for the connectome. The wiring is not used, and any structural comparison run in that regime measures the input projection. Under the disjoint protocol we use, the same control falls to 4.9% (chance) while the connectome reaches 35.2%. |
| **Matched firing rate** | Matching spectral radius does not match activity level. Erdős–Rényi surrogates run **1.77×** hotter than the connectomes they are matched to, and rate predicts accuracy (*r* = 0.38). A matched-gain comparison confounds topology with operating point. |

**A gap-free perturbation bound.** The Davis–Kahan theorem conventionally invoked for
degeneracy is inapplicable to measured connectomes — their spectral gaps make it vacuous.
We bound the finite-horizon input–output map instead, which needs no gap, and extract the
**operational alignment** γ of a perturbation.

## 3. What we found

Evaluated as fixed reservoirs on Spiking Heidelberg Digits, across five networks from
four species:

**Every structural property that measurably moves accuracy moves it down.**

| Property added | Significant | Median effect |
|---|---|---|
| Degree sequence | 3 / 5 networks | **−4.65 pp** |
| Motif census | 2 / 5 networks | **−3.50 pp** |
| Valence placement | 1 / 5 networks | −0.82 pp |
| Residual wiring | 2 / 5 networks | +1.06 pp |

Adding a connectome's exact 3-node motif spectrum to a degree- and valence-matched
surrogate **costs** accuracy, by about as much as the degree sequence does. That is the
opposite sign to the one the motif hypothesis predicts. Task-independent dynamical probes
(memory capacity, kernel rank) reproduce the ordering and the direction.

Recomputing enrichment itself, with 500 surrogates and exact counts, already qualifies
the premise: of 29 interpretable tests, 23 motifs are enriched, **3 are significantly
depleted** — coherent feedforward loops in the *Drosophila* central complex, *Z* = −10.9 —
and 3 are indistinguishable from the null. Mouse V1 sits at its null throughout. Motif
enrichment is real but not conserved across phylogeny.

## 4. Why this is worth something

**The operating-point result generalises to a whole class of published comparison.**
Any study that compares a network against a null at one dynamical regime — and most do —
inherits a dependence it has not reported. The fix is one sweep rather than one
evaluation. That is cheap, and it converts a claim about a point into a claim about a
regime.

**The null ensemble is a reusable instrument, though not a new idea.** Motif-constrained
ensembles are standard in statistical network science (ERGMs, the dk-series, and
Fischer et al.'s Wang–Landau sampler, which has a sampling guarantee ours does not). What
we supply is the sign-attributed version at connectome scale and its use as a *control
rung*. Any field that infers function from subgraph overrepresentation faces the same
attribution problem — gene regulatory networks, food webs, software dependency graphs.

**The controls generalise further than the finding.** The zero-recurrence control is four
lines and would invalidate a category of published comparison. We would encourage it in
any study comparing connectome-derived reservoirs, and we report it on our own work
first.

**A negative result, properly bounded, is worth more than an unbounded positive one.** We
state what the design could have detected (80% power at *d* = 1.54 after correction), we
state where our own prediction failed, and we state where two of our measurements
disagree rather than choosing the one that reads better.

## 5. What we are not claiming

- **Not** that connectome wiring is bad for computation. It is a poor fixed reservoir for
  one classification task with a linear readout, which is not the objective any nervous
  system is under selection for.
- **Not** that motif structure is functionally irrelevant. Our design is powered for
  *d* = 1.54; it does not exclude moderate effects, and it does not test motifs as an
  inductive bias for **learning**, which is a different claim requiring a trained ladder
  we have not run to publication standard.
- **Not** any silicon result beyond what an open flow establishes. We report gate count,
  flip-flop count and mapped pre-layout cell area from Yosys/Nangate45 — 590,116 cells,
  25,644 flip-flops, 749,263 µm² for the top macro. **No die area, no power, no energy
  per operation, no clock frequency.** The toolchain performs no place-and-route and
  ships no static timing analyser.

## 6. Where our own prediction failed

We predicted tolerance to structural damage would scale as 1/γ. Tested two ways:

- **Ordering — holds.** γ ranks the perturbation directions correctly in 3 of 5 networks.
  Tolerated pruning falls from **83%** for random removal to **64%** along the adversarial
  direction at identical cardinality.
- **Scaling — fails.** The product γ·‖E‖max should be constant across directions. It
  varies by a factor of **2.2**.

We report the failure. γ identifies which direction is dangerous and does not predict by
how much. The conclusion that matters for the literature needs only the ordering:
robustness to *random* ablation is much weaker evidence of degeneracy than it appears,
because a random direction is nearly orthogonal to any fixed functional subspace.

## 7. Verification

Everything in the manuscript regenerates from raw result files:

```bash
./src/analysis/rebuild_paper.sh
```

It reports unresolved numbers, undefined references and LaTeX errors, because a clean
exit status is not evidence of anything — `pdflatex` exits 0 on a hard error by design.
Current state: **0 unresolved, 0 errors, 0 undefined references, 108 tests passing.**

No number in the manuscript is transcribed by hand. A missing result appears in the PDF
as a visible placeholder rather than as a stale value.

## 8. The record

`AUDIT.md` documents 34 entries: **15** defects in the retracted v4 manuscript, **12**
found in the **replacement** analysis while building it — our own, recorded on the same
terms — and **7** where a correction changed a conclusion, including our own partly
falsified prediction and two conclusions that reversed outright when we found that our
null ensemble was not the published one.

One pattern runs through nearly all of them, and it is the most transferable thing in
this project:

> **Every defect produced a plausible number and no error.** A loader that swallowed its
> exception returned an empty graph. An `argsort` over equal values returned index order.
> A null model imposed a constraint it did not impose. A protocol control decoded nothing
> and scored chance. `pdflatex` reported a duplicate macro and exited 0 with a complete
> PDF. None was caught by a result looking wrong. Each was caught by checking a recorded
> diagnostic, or by reading the code against the prose.

That is why the convergence diagnostics, the protocol control and the held-out statistic
are reported in the paper rather than kept in the repository.

---

**Repository:** analysis code, null-model implementations, experiment drivers, the
synthesis flow, the audit and the retracted v4 material.
**Manuscript:** `paper/computational_coupling_v5.pdf` (27 pages).
**Full technical detail:** `info.md`.
