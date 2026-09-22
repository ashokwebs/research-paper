# Computational Coupling — Project Information

**Four choices decide whether a connectome beats its rewiring.**

*Computational coupling* is the extent to which what a network computes is attributable
to how it is wired. It is never measured directly — only as the gap between a network's
score and the scores of surrogate networks, under some unit model, at some operating
point, through some readout. This project shows that each of those four choices can flip
the sign of that gap on identical data, so a coupling reported as a single signed number
is under-determined by its own methods section.

**Authors** — Ashok Raj Pasala · Snigdha Gorai · under the guidance of Jeevana Jyothi P.

**Affiliation** — School of Computer Science and Engineering, VIT-AP University,
Amaravati 522237, Andhra Pradesh, India

**Version** — 5.0, supersedes and retracts v4 · **Licence** — MIT

**Manuscript** — `paper/computational_coupling_v5.pdf` (27 pages, two-column) ·
**Record of withdrawals** — `AUDIT.md`

This document is the single reference for the project: what it asks, how it is built,
what it found, what it does not claim, and how to reproduce every number in it. The
manuscript is the scientific statement; this is the operational one.

---

## 1. The question, and why it was open

Connectomes contain statistically overrepresented subgraphs. A large literature holds
that these *motifs* are what make biological wiring computationally effective; a
neuromorphic literature builds architectures on that premise.

Testing the claim needs a control that differs from a connectome **in its motif structure
and in nothing else**. No such control existed. Rewiring a connectome perturbs its motif
census together with every statistic that co-varies with it — clustering, spectrum,
higher-order structure — so a performance difference against a rewired null cannot be
attributed to motifs in particular.

Sampling the ensemble that would settle it is hard: it requires moving through the space
of degree-constrained graphs under a constraint on a *global* subgraph count, with the
census recomputed at every proposal.

Three further confounds recur in this literature. All three appear in our own v4 work,
and each is a failure that produces no error and no warning:

1. **A "Dale-compliant" null imposes no constraint.** When synapses inherit their
   presynaptic neuron's valence, Dale's principle is automatically preserved by any
   double-edge swap. Such a null is the standard Maslov–Sneppen ensemble and cannot test
   whether the *placement* of inhibition carries information.
2. **An all-to-all reservoir does not use its wiring.** With every neuron driven and every
   neuron decoded, the task is solved by the input projection and the leak. We measure
   59.6% for a **zeroed** recurrent matrix against 59.6% for the connectome. Under the
   sensory/motor-disjoint protocol used throughout this work the same control falls to
   4.9% — chance — while the connectome reaches 35.2%. That gap is the diagnostic that
   the benchmark is sensitive to recurrent structure at all.
3. **Matched spectral radius is not a matched operating point.** Erdős–Rényi surrogates
   settle at 1.77× the firing rate of the connectomes they are matched to, and rate
   predicts accuracy (*r* = 0.38). A matched-gain comparison confounds topology with
   activity level, in the direction that flatters the unstructured control.

## 2. What is new here

**1. The connectome advantage does not survive a firing threshold** —
`experiments/ladder/run_unit_model_comparison.py`,
`experiments/ladder/run_null_mechanism.py`

The reservoir literature reports that empirical connectomes beat degree-preserving
rewirings on memory capacity, and that result is increasingly invoked as motivation for
neuromorphic architectures. We ran identical graphs, identical inputs, delays and readout
through two reservoirs differing **only** in their units, against **two** rewiring
ensembles that both answer to "degree-preserving randomisation", on the five regional
connectomes distributed with the original toolbox — 2,700 evaluations, 5 networks × 10
paired instances × 9 gains × 2 unit models × (connectome + 2 nulls).

**The null choice changes the answer.** "Degree-preserving randomisation" names at least
two ensembles: one where a weight travels with its edge (BCT `randmio_*_connected`, what
`conn2res` calls), one where the weight multiset is re-assigned to the rewired graph by
endpoint-degree rank. We wrote the second believing it was the first.

| null ensemble | connectome beats it at ρ=1, graded units | median difference |
|---|---|---|
| BCT `randmio` (the published one) | **1 of 5** leads, **3 of 5** trail | −0.51 |
| rank-reassigned (ours, mistakenly) | **5 of 5** lead, all significant | +2.13 |

Of the 10 (network, unit model) cells, the two nulls give opposite **signs** in 6 and
opposite **significant verdicts** in 3. Median disagreement 0.73 capacity units, max 6.98.

![The connectome's measured advantage over two rewirings that answer to the same
description in the literature. Blue and orange differ only in what happened to the edge
weights during randomisation. In four of five networks they point in opposite
directions.](paper/figures/fig_null_choice.png)

*(A) The result. (B) The cause: one rewiring installs a weight-degree association the
real graphs do not have. (C) The test that could have refuted (B) — on the one network
with no weight variation, the two procedures must agree, and they do, to 0.008.*

**Why, and the prediction that could have refuted it.** Rank-reassignment does not only
randomise — it *installs* a weight–degree association the data lacks, shifting that
correlation by a median of **+0.66** across the four weighted networks against **+0.06**
for weight-carrying rewiring. So on a network with **no weight variation** the two must
coincide exactly. Macaque is binary. There they shift the association by 0.000 and their
measured advantages differ by 0.008 capacity units — the one cell where the choice
provably cannot matter is the one cell where it does not.

This is **not** a refutation of the published result: our harness also differs in its
input projection, delay set and ridge solver. It shows that the null choice *alone* moves
the result by more than the effect being reported, which is a sufficient explanation for a
reproduction failure without anything else being wrong.

| units | connectome beats `randmio` at ρ=1 | across the gain axis | silent units |
|---|---|---|---|
| graded `tanh` | **1 of 5** leads, 3 trail | 29% of cells | 0.6% |
| spiking LIF | **0 of 5** | 9% of cells | 60.7% |

**One mechanism changed sign when the null was corrected.** We predicted a threshold
would silence a heavy-tailed graph's low-degree neurons differentially. Silencing is
large (60.7% vs 0.6%). Against the rank-reassigned null the silencing gap did *not*
predict the capacity gap and we reported the prediction as failed; against `randmio` it
**does** — r = −0.58, p = 2.5×10⁻⁵ over 45 (network, gain) cells, in the predicted
direction. We report both, because a wrong control manufactured a clean positive result
in one place and destroyed a real one in another, from the same code and the same graphs.
Thresholding also compresses capacity for every topology alike, by a factor of 4.9, and
that part is not differential at all.

**A floor, declared.** The human network's spiking capacities are 0.049 vs 0.019 — both
effectively zero, so it carries nothing either way. The claim rests on the other four,
which retain capacities of 1.0–1.9.

**2. The contrast also changes sign along the gain axis** —
`experiments/ladder/run_gain_sweep_ladder.py`

**3. A motif-preserving null ensemble** — `src/nulls/swap_chain.py`

Degree-, valence- and census-matched surrogates, sampled by an annealed Metropolis chain
over exact, incrementally updated motif counts.

**We do not claim the idea.** Motif-constrained ensembles are standard in statistical
network science: Milo et al. proposed a simulated-annealing null preserving lower-order
subgraph counts; exponential random graph models condition on triad counts by design
(Frank & Strauss 1986; Robins et al. 2007); the dk-series is a nested hierarchy of nulls
constraining progressively higher-order structure (Mahadevan et al. 2006; Orsini et al.
2015); and Fischer et al. (*PRL* 2015) sample motif-constrained ensembles at fixed degree
sequence by Wang–Landau multicanonical sampling, which unlike annealing comes with a
sampling guarantee. What we contribute is an implementation for the **sign-attributed**
census at connectome scale — exact coherent/incoherent FFL counts under Dale's principle,
O(d) incremental updates, and a move set that survives near-total reciprocity — and the
use of that ensemble as a **control rung** rather than as an enrichment null. Where a
formally exact sampler is available it should be preferred.

Two things about the construction matter, because the obvious implementation does not
work:

- *Where the chain starts decides which property is guaranteed.* Starting from a fully
  rewired graph guarantees randomisation and leaves convergence to be shown; starting
  from the connectome under a census constraint guarantees convergence and leaves
  randomisation to be shown. We use the empirical start and demonstrate randomisation by
  edge overlap, because the random start fails outright on networks whose census is hard
  to rebuild by chance.
- *Ordinary swaps destroy reciprocal pairs faster than they create them.* On a
  near-symmetric network the reciprocal-pair count drifts down however stiff the penalty.
  We add a move that swaps two reciprocal **pairs** as units, preserving both degrees and
  reciprocity.

**4. A five-rung structural control ladder** — `src/nulls/ladder.py`

| Rung | Preserves | Isolated by comparing with the rung below |
|---|---|---|
| L0 | N, edge count, E:I ratio | — |
| L1 | + joint in/out degree sequence | degree heterogeneity |
| L2 | + valence–topology coupling | where inhibition sits |
| L3 | + 3-node motif census | motif structure |
| L4 | the empirical connectome | residual higher-order wiring |

**5. Two controls that are usually missing** — `experiments/ladder/`

The zero-recurrence control, and the ladder re-run with every network individually
recalibrated to a common firing rate.

**6. A gap-free perturbation bound** — `src/theory/transfer_perturbation.py`

Davis–Kahan is inapplicable to measured connectomes: the smallest gap we measure is
4.4 × 10⁻², admitting perturbations of norm at most 2.2 × 10⁻² — less than removing one
synapse. We bound the finite-horizon input–output map instead, which needs no gap, and
extract the **operational alignment** γ of a perturbation.

**7. An honest hardware report** — `experiments/hardware/run_open_synthesis.py`

Real logic synthesis with Yosys + ABC onto the Nangate45 open cell library, reporting
only what such a flow establishes.

## 3. Data

| Network | Source | *N* | \|E\| | Valence |
|---|---|---|---|---|
| *C. elegans* whole nervous system | Cook et al. 2019 | 300 | 5,044 | yes |
| *Drosophila* central complex | FlyWire v630 | 400 | 7,856 | yes |
| *Drosophila* antennal lobe | FlyWire v630 | 400 | 16,468 | yes |
| Mouse V1 L2/3 | MICrONS Pinky100 | 334 | 1,734 | **no** |
| Human, regional | HCP consensus tractography | 400 | 6,092 | **no** |

Four public releases spanning three phyla, yielding five networks. Two carry synaptic
valence and two do not, and that constrains which questions each can answer:

- **Mouse.** The release annotates 34 inhibitory somata, but every one is **disconnected**
  in the accompanying synapse table — all 1,736 synapses are excitatory→excitatory.
  Treated as topology-only. Reporting coherent feedforward loops here would be reporting
  plain feedforward loops under another name.
- **Human HCP.** Diffusion tractography carries no synaptic valence. Treated as
  topology-only, and labelled a *regional* network rather than a neuronal connectome.
- **Human microscale is not analysed.** The H01 release provides cell and synapse tables,
  but the connectivity export needed to build a wiring diagram is not in the distribution
  we worked from. A census of a synapse table is not a connectome analysis. (v4 claimed
  this tier; see §12.)

**Task data.** Spiking Heidelberg Digits (Cramer et al. 2022): 20 classes, 700 cochlear
channels, 12 speakers, canonical speaker-held-out split. Binned to 100 time steps over
1 s, channels pooled to 140 bands. A real spike-based benchmark, replacing the synthetic
tasks of v4 which saturated near 100% for every architecture.

**Subcircuit extraction.** Where a release exceeds the common reservoir size, we take an
induced subgraph by breadth-first expansion from an above-median-degree seed, then drop
isolates. Growing from a hub could inflate degree heterogeneity — the property the
largest effect is attributed to — so we measured it: the ratio of subgraph to
full-release degree CV is at most **1.00** across all five networks, as low as **0.79**
among the three genuinely subsampled. Extraction biases the degree result *against*
itself, if at all.

## 4. Method

### Exact sign-attributed motif census

Under Dale's principle the sign product around a feedforward loop
(*u*→*v*, *v*→*w*, *u*→*w*) is σ*ᵤ*·σ*ᵥ*·σ*ᵤ* = σ*ᵥ*.

> **A feedforward loop is coherent if and only if its *intermediate* neuron is
> excitatory.** Coherence is a property of one node, not of a product over three edges.

This makes the sign-attributed census exactly as cheap as the topological one. All counts
are exact; none are estimated. Single-edge insertions and deletions admit exact *O(d)*
updates of every 3-node count, which is what makes the annealed null tractable.

### The annealed chain

Energy is the squared deviation of the running census from a target, **measured in units
of the unconstrained-ensemble standard deviation** — over reciprocal pairs, mutually
inhibitory pairs, FFL, cFFL, iFFL and 3-cycles, across 120 × |E| proposals. Bi-fan is
deliberately excluded and reported post hoc as a **held-out statistic**.

The normalisation lives in exactly one place (`census_scale`). An earlier version
restated it at ten call sites with an extra floor of 2% of the empirical count, which made
the constraint roughly five times slacker than the prose described and left the chain
unable to hold the census on human tractography. See `AUDIT.md` §D2.

### Evaluation protocol

Fixed reservoir, trained ridge readout only. Discrete-time LIF units, leak 0.90,
threshold 1. Held fixed across every network in a comparison: node count, edge count,
spectral radius (ρ = 1.30) and input projection.

**Sensory/motor disjoint.** The input reaches 15% of neurons; the readout sees a
**disjoint** 35%. Task-relevant signal must traverse the recurrent graph to be decodable.
The zero-recurrence control then falls to chance, which is the diagnostic that the
benchmark is sensitive to recurrent structure at all.

**Statistics.** Welch's *t* with Holm correction across the full 20-contrast family,
percentile-bootstrap CIs, Hedges' *g*. 15 surrogates per rung. Rung L4 is one graph, so
its replicates carry one variance component where L0–L3 carry two; a contrast involving
L4 supports inference about *this* connectome rather than connectomes as a class.

**Power.** 80% power to resolve *d* = 1.06 uncorrected, *d* = **1.54** at the Holm worst
case. Stated so a small measured effect can be told from an underpowered design.

## 5. Every experiment we ran

<!-- BEGIN GENERATED: experiment-inventory -->
*15 of 15 experiments have results on disk. Sizes are counted from each experiment's own output file when this document is built, so they cannot drift from what was actually run. The unit differs by row: a "run" is one network at one rung trained to convergence, an "evaluation" one network at one operating point probed for one metric, so the column does not sum to anything meaningful.*

| # | Experiment | What it asks | Size |
|---|---|---|---|
| 1 | **Motif enrichment**<br>`experiments/motifs/run_motif_significance.py` | Are the classic subgraphs really over-represented? Exact sign-attributed counts against 500 degree-preserving surrogates per network. | 35 tests |
| 2 | **Motif-preserving null ensemble**<br>`experiments/ladder/run_null_validation.py` | Validation of the annealed chain: does it hit the target census, does it actually randomise, and does it leave untargeted structure alone? | 5 networks |
| 3 | **Sampler robustness**<br>`experiments/ladder/run_null_robustness.py` | The same ensemble redrawn over a factor-of-eight range of chain lengths and from both possible starting points, ruling out two ways a sampler could fake the result. | 200 draws |
| 4 | **Protocol controls**<br>`experiments/ladder/run_protocol_controls.py` | What does the benchmark measure? Includes the reservoir with its recurrent matrix set to zero, under both readout protocols. | 8 conditions |
| 5 | **Structural control ladder**<br>`experiments/ladder/run_structural_ladder.py` | L0 density to L4 real connectome on spoken digits, each rung adding exactly one structural property; accuracy, memory capacity, kernel and generalisation rank. | 375 runs |
| 6 | **The ladder at matched firing rate**<br>`experiments/ladder/run_rate_matched_ladder.py` | The same decomposition with every network recalibrated to a common firing rate, because matching spectral radius does not match activity. | 250 runs |
| 7 | **Extraction check**<br>`experiments/ladder/run_extraction_check.py` | Does taking a size-matched subgraph of a larger connectome bias the degree result? Run because the ladder depends on the extraction being neutral. | 5 networks |
| 8 | **Two null ensembles, two unit models**<br>`experiments/ladder/run_unit_model_comparison.py` | The headline experiment. Both readings of "degree-preserving randomisation" against both unit models across nine gains, on the original toolbox's own connectomes. | 2,700 evaluations |
| 9 | **Why the two nulls differ**<br>`experiments/ladder/run_null_mechanism.py` | Structural descriptors of both ensembles against the real graph, including the prediction that they must coincide on an unweighted network. | 105 graphs |
| 10 | **Spiking tuning control**<br>`experiments/ladder/run_spiking_tuning_control.py` | Input gain x threshold x leak, searching for the configuration that would refute our own spiking result rather than the one that confirms it. | 1,728 evaluations |
| 11 | **Gain sweep of the ladder**<br>`experiments/ladder/run_gain_sweep_ladder.py` | The whole decomposition across the critical region, because a comparison at one gain is a statement about that gain. | 1,400 evaluations |
| 12 | **Memory-capacity sweep**<br>`experiments/ladder/run_memory_capacity_sweep.py` | Locating the published advantage on the gain axis at matched spectral radius. | 2,500 evaluations |
| 13 | **Intrinsic dynamics**<br>`experiments/ladder/run_dynamics_ladder.py` | The same decomposition measured on task-independent probes, so the ordering does not depend on one benchmark. | 250 runs |
| 14 | **Pruning frontier**<br>`experiments/degeneracy/run_pruning_frontier.py` | Random, hub-targeted and adversarial edge removal, testing whether the operational alignment predicts how much damage a direction does. | 540 prunings |
| 15 | **Open-PDK synthesis**<br>`experiments/hardware/run_open_synthesis.py` | Real logic synthesis and technology mapping of the accompanying RTL, reporting only what the open flow can establish. | 4 designs |
<!-- END GENERATED: experiment-inventory -->

Three of these exist only because they could have refuted us, and two of them partly did:
the **spiking tuning control** (#10) searched for the configuration that would overturn
our own spiking result and found enough to force the claim down from "absent" to "sporadic
and inconsistent in sign"; the **pruning frontier** (#14) tested a scaling prediction of
our own perturbation bound and falsified it, leaving only the ordering; and the
**extraction check** (#7) tested whether our own subgraph sampling was producing the
degree result. We regard running them before submission rather than after as the
difference between a finding and a correction.

## 6. Results

### Motif enrichment, recomputed

500 degree-preserving surrogates, exact counts, resolution floor stated. Enrichment is
real but **not conserved**:

| Outcome | Count |
|---|---|
| Significantly enriched | 23 / 29 |
| Significantly **depleted** | 3 / 29 |
| Indistinguishable from null | 3 / 29 |

Coherent feedforward loops are **depleted** in the *Drosophila* central complex
(*Z* = −10.9). Mouse V1 sits at its null throughout (no motif above *Z* = +2.7). The
normalised significance profiles of the two fly circuits agree with each other no better
than either agrees with the worm.

### Null ensemble validation

| Network | Max rel. error | Converged | Overlap L3 | Overlap L2 |
|---|---|---|---|---|
| *C. elegans* | 0.25% | 100% | 13.9% | 11.6% |
| *Drosophila* CX | 0.20% | 100% | 20.5% | 19.9% |
| *Drosophila* AL | 0.11% | 100% | 43.9% | 40.0% |
| Mouse V1 | 0.85% | 100% | 10.6% | 10.7% |
| Human HCP | 0.50% | 100% | 11.4% | 7.8% |

The chain hits its target, randomises to near the unconstrained level, and does **not**
reproduce the held-out bi-fan count — nor displace it in a consistent direction
(overshoot in the worm, undershoot in the fly CX), which is what distinguishes L3 from a
partial reconstruction of the connectome.

**The constraint is binding.** L2 and L3 are separated by tens to hundreds of null
standard deviations in the constrained coordinates — except in mouse V1, where the census
sits near its null by construction and the contrast is correspondingly uninformative.

**Not a sampler artefact.** Over an eightfold range of chain lengths accuracy moves by a
median of 1.04 pp, non-monotonically in 4 of 5 networks. The opposite construction
(random start, annealing towards the census) converges on 4 of 5 and agrees to within
1.07 pp where it does.

### The structural decomposition

**Every structural property that measurably moves accuracy moves it down.**

| Property added | Significant | Median | Largest | Hedges' *g* |
|---|---|---|---|---|
| Degree sequence | 3 / 5 | **−4.65 pp** | −8.02 pp | −2.38 |
| Motif census | 2 / 5 | **−3.50 pp** | −5.94 pp | −2.06 |
| Valence placement | 1 / 5 | −0.82 pp | −3.16 pp | −1.14 |
| Residual wiring | 2 / 5 | +1.06 pp | −7.60 pp | −2.04 |

The motif contrast is **not** negligible — it is comparable in size to the degree
sequence — but its sign is negative. Imposing a connectome's motif structure on a degree-
and valence-matched surrogate **costs** accuracy on this task. That is the opposite sign
to the one the motif hypothesis predicts.

### At a matched operating point

Recalibrating each network to a common mean firing rate of 0.07 does **not** rescue the
connectomes: the Erdős–Rényi rung still leads by 9.00–13.95 pp, marginally more than at
matched gain (7.96–11.74 pp). 67–91% of the decline is spent at the step that imposes the
degree sequence.

27 of 250 calibrations never reached the target, 25 of them in the *Drosophila* central
complex, whose edges are 66% inhibitory — raising its gain raises its inhibition, so its
attainable rate is capped by its own valence organisation. Those cells are excluded from
every matched-rate statistic and marked in the table rather than averaged in.

### Dynamical probes

Task-independent measures (memory capacity, kernel rank, generalisation rank) reproduce
the ordering: the Erdős–Rényi rung has higher memory capacity in **5 of 5** networks
(median 1.41×) and higher kernel rank in 5 of 5 (1.21×). The connectomes are not trading
task accuracy for capability the benchmark fails to reward.

**A disagreement we do not resolve.** The probes and the task metric agree on direction
and on the degree sequence carrying the largest share, but disagree on the motif rung:
comparable to degree on accuracy, smallest in the ladder on the probes. The manuscript
states the divergence and offers a conjecture labelled as one.

### Degeneracy

Tolerated pruning (largest removal retaining 90% of unpruned accuracy):

| Direction | Tolerated | Median γ, relative to random |
|---|---|---|
| Uniformly random | **83%** (lower bound) | 1.0× |
| Degree-targeted | 73% | 2.6× |
| Adversarial (first-order) | 64% | 3.0× |

All three strategies **improve** accuracy at low fractions, peaking around 40–60% removal.
These reservoirs are over-connected for the task, consistent with degree heterogeneity
being costly here.

### Hardware

Yosys + ABC onto Nangate45, typical corner:

| Module | Cells | Flip-flops | Cell area (µm²) |
|---|---|---|---|
| `dendritic_lif_core` | 4,401 | 70 | 5,301 |
| `synaptic_accumulator` | 19,649 | 8,192 | 63,361 |
| `aer_crossbar_router` | 39 | 17 | 124 |
| `cen_accelerator_top` | 590,116 | 25,644 | 749,263 |

**No die area, no power, no energy per operation, no clock frequency.** The flow performs
no placement, routing, clock-tree synthesis or parasitic extraction, and the toolchain
ships no static timing analyser.

## 7. What this does not claim

- **Not** that connectome wiring is bad for computation. It is a poor fixed reservoir for
  a 20-class classification with a linear readout — one task, one regime, and not the
  objective any nervous system is under selection for. The uniform negative sign is
  informative about the probe: a linear decoder rewards distributed, high-dimensional
  state, and connectome structure concentrates and correlates activity.
- **Not** that motif structure is functionally irrelevant. The design resolves *d* = 1.54
  after correction; non-significant contrasts do not exclude moderate effects.
- **Not** anything about **learnability**. Motif structure has been argued to act as an
  inductive bias, which is a claim about learning trajectories, not about the function
  computed by a frozen network. Testing it needs a trained ladder (§13).
- **Not** any silicon figure beyond gate count, flip-flop count and pre-layout cell area.
- **Not** a microscale human result.

## 8. Where our own predictions failed

Recorded because narrowing a claim quietly after seeing the data is the behaviour this
project exists to correct.

**The 1/γ scaling.** Proposition 2 predicts tolerance scales as 1/γ. The **ordering**
holds — γ ranks the directions correctly in 3 of 5 networks. The **scaling** does not: the
product γ·‖E‖max should be constant across directions and varies by a factor of 2.2. γ
identifies which direction is dangerous and does not predict by how much.

**γ is not comparable between networks.** It is a realised fraction of a worst-case
constant depending on ‖A‖₂ᵀ, and its median spans ~7×10⁵ across our five networks. Pooled
γ medians — which our first analysis reported, and our first figure plotted — are not a
quantity at all.

**The rate-matching expectation.** We introduced the matched-rate ladder expecting it to
explain the Erdős–Rényi advantage. It does not; the gap widens slightly. The resulting
claim is stronger and less comfortable than the one we set out to make.

**The operating-point expectation, twice.** A one-instance probe suggested the
connectome-versus-null memory-capacity contrast turned positive exactly at criticality
and negative on both sides — a clean crossing that would have reproduced and located the
published result. The paper was rewritten around it. At ten paired instances the pattern
was scatter, not a crossing: 0 of 5 significant. The obvious next move, reporting a failed
replication, would have been wrong too — the design is underpowered there by roughly a
factor of five. The honest report is a non-detection with its power stated. See
`AUDIT.md` §E4.

**The motif artefact expectation.** The human motif contrast was significant under a
sampler that never converged on that network. We expected it to vanish once the sampler
was fixed, and recorded that expectation before the re-run. It did not vanish — it
persists at *g* = −2.06, joined by *C. elegans* at −1.46. Had it vanished, this paper
would have reported a clean null on motifs; instead it reports a real effect pointing the
opposite way to the hypothesis under test.

## 9. Repository layout

```
src/motifs/fast_census.py            exact sign-attributed census, O(d) incremental updates
src/nulls/swap_chain.py              degree-preserving chains + the annealed motif-preserving null
src/nulls/ladder.py                  the five-rung structural control ladder
src/theory/transfer_perturbation.py  gap-free bound, operational alignment, edge rankings
src/eval/reservoir.py                reservoir protocol (sensory/motor disjoint)
src/eval/dynamics.py                 memory capacity, kernel & generalisation rank
src/eval/trained_snn.py              surrogate-gradient training of masked recurrent SNNs
src/tasks/shd.py                     Spiking Heidelberg Digits loader
src/analysis/                        figure, table and LaTeX-macro generation
src/analysis/rebuild_paper.sh        the whole path from results/ to the compiled PDF
experiments/                         experiment drivers, one per result
hardware/rtl/                        Verilog sources
paper/                               the current manuscript and its generated inputs
pitch/                               the current executive summary
AUDIT.md                             record of withdrawn claims and of defects in the replacement
info.md                              this document
paper/superseded_v4/                 the retracted v4 manuscript, figures and pitch dossier
results/superseded_v4/               the result files those claims were drawn from
```

Other packages under `src/` (`architecture/`, `benchmarks/`, `learning/`, `remodeling/`,
`simulation/`) and the matching directories under `experiments/` belong to v4 and are not
exercised by any current result. They are kept because `AUDIT.md` cites them as the origin
of specific withdrawn numbers; deleting them would make the retraction harder to check.

## 10. Reproducing everything

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Task data (~350 MB)
mkdir -p data/raw/shd && cd data/raw/shd
curl -L -O https://zenkelab.org/datasets/shd_train.h5.gz
curl -L -O https://zenkelab.org/datasets/shd_test.h5.gz && gunzip *.gz && cd -
```

Then the experiments. This list is generated from the same source as the table in
§5, so it cannot fall behind what the repository actually contains:

<!-- BEGIN GENERATED: experiment-commands -->
```bash
python experiments/motifs/run_motif_significance.py      # Motif enrichment
python experiments/ladder/run_null_validation.py         # Motif-preserving null ensemble
python experiments/ladder/run_null_robustness.py         # Sampler robustness
python experiments/ladder/run_protocol_controls.py       # Protocol controls
python experiments/ladder/run_structural_ladder.py       # Structural control ladder
python experiments/ladder/run_rate_matched_ladder.py     # The ladder at matched firing rate
python experiments/ladder/run_extraction_check.py        # Extraction check
python experiments/ladder/run_unit_model_comparison.py   # Two null ensembles, two unit models
python experiments/ladder/run_null_mechanism.py          # Why the two nulls differ
python experiments/ladder/run_spiking_tuning_control.py  # Spiking tuning control
python experiments/ladder/run_gain_sweep_ladder.py       # Gain sweep of the ladder
python experiments/ladder/run_memory_capacity_sweep.py   # Memory-capacity sweep
python experiments/ladder/run_dynamics_ladder.py         # Intrinsic dynamics
python experiments/degeneracy/run_pruning_frontier.py    # Pruning frontier
python experiments/hardware/run_open_synthesis.py        # Open-PDK synthesis
```
<!-- END GENERATED: experiment-commands -->

Then the statistics, figures, tables, macros and three LaTeX passes:

```bash
./src/analysis/rebuild_paper.sh
```

`rebuild_paper.sh` reports unresolved numbers, undefined references, LaTeX errors and
overfull boxes at the end. A clean exit status is not evidence of anything — `pdflatex`
exits 0 on a hard error by design, and a duplicate `\newcommand` builds a complete PDF
while silently keeping whichever definition came first.

Every number in the manuscript is emitted into `paper/generated_numbers.tex`; none is
transcribed by hand, and a missing result appears as a visible `??` rather than as a
stale value. Protocol constants are read from the experiment's own argument parser and
from the task loader's signature, not restated.

## 11. Testing

```bash
.venv/bin/pytest          # 108 tests
```

Covers the census against brute-force enumeration, exactness of incremental updates,
degree and valence preservation in every null, convergence and randomisation of the
annealed chain, the reservoir protocol controls, and the perturbation bounds.

**Every defect in `AUDIT.md` §D is pinned by a regression test that was verified to fail
against the old code before being kept.** A test that passes against both versions pins
nothing, and two of ours originally did.

## 12. The audit, and the pattern in it

`AUDIT.md` documents 34 entries: **15** in the retracted v4 manuscript (**A–C**), **12**
found in the *replacement* analysis while building it (**D–D12**) — our own, on the same
terms — and **7** where a correction changed a conclusion (**E–E7**), including two that
reversed when we discovered our null ensemble was not the published one.

One pattern runs through nearly all of them:

> **Every defect produced a plausible number and no error.** A loader wrapped in
> `except Exception: pass` returned a zero-edge graph. `argsort` over constant magnitudes
> returned index order, turning "weakest-first pruning" into whole-neuron ablation. A
> null model imposed a constraint it did not impose. An energy scale was five times
> slacker than the prose claimed. A bound was stated in the spectral radius where the
> operator norm was required — in a paper arguing that eigenvalue reasoning fails on these
> networks. A protocol control decoded nothing and scored chance. `pdflatex` reported a
> duplicate macro and exited 0 with a complete PDF.

None was caught by a result looking wrong. Each was caught by checking a recorded
diagnostic, or by reading the code against the prose. That is why the convergence
diagnostics, the protocol control and the held-out statistic are reported in the paper
rather than kept in the repository.

## 13. Relationship to v4

This work **supersedes and retracts** the previous manuscript. Withdrawn:

- **Commercial silicon figures that were never measured.** "TSMC 28nm HPC+", a die area,
  116.71 pJ/SynOp, 2,857× real-time acceleration, a "physical tapeout". No Cadence tool,
  no TSMC library and no synthesis output ever existed in the repository; the numbers came
  from a Python model with those values as literals. The RTL had never been through any
  synthesis tool, and did not compile.
- **A human microscale tier built on an empty graph.** `load_human_h01_microcircuit()`
  returned a graph with **zero edges**; the loader wrapped the failure in
  `except Exception: pass`. The "26.5M synapses" figure is a sum over a column in a cell
  table.
- **Sign-attributed motifs for datasets with no usable inhibition.** Mouse and human
  entries where cFFL equals FFL by construction.
- **p-values below the resolution of the null that produced them.** `p < 0.0001` from
  K = 25 surrogates, where the smallest attainable value is 1/(K+1) ≈ 0.038.
- **An ablation table whose own control row contradicted its conclusion.** The
  Erdős–Rényi control was *better* than the motif-structured model on the very metric
  used to argue for motifs.
- **A "455× energy reduction"** comparing a near-silent network against a dense one firing
  freely, at different accuracies.

The retracted documents are at `paper/superseded_v4/` and their data at
`results/superseded_v4/`, labelled. The v4 **executive pitch dossier** — the artefact most
likely to be forwarded — carries its own retraction notice at
`paper/superseded_v4/pitch/RETRACTED.md`.

## 14. Open work

**The trained ladder.** `experiments/ladder/run_trained_ladder.py` exists and is **not**
run or reported. It tests motifs as an inductive bias for learning, which the
fixed-reservoir design cannot address. Its harness has been corrected — it previously used
the all-to-all protocol this paper rejects — and its firing-rate penalty retuned, but it
needs a zero-recurrence control that separates from the connectomes before any number
from it means anything. One trap is documented: with only a fraction of neurons driven, a
firing-rate penalty strong enough for an all-to-all network makes total silence the global
optimum, and a collapsed network returns a perfectly flat ladder indistinguishable from
the result reported here.

**A weighted null.** All weights here are binary up to valence. Real synapses have graded
strengths correlating with topological position. The annealed chain generalises to
weighted targets without modification; the degree-preserving move set does not.

**An exact sampler.** Annealing is not proven to sample uniformly over the constraint set.
We show the conclusion survives an eightfold variation in chain length and does not depend
on the starting point, which rules out two specific ways an annealed sampler could
manufacture the result — but a formally exact sampler for a census-constrained
degree-preserving ensemble remains open.

**A better-powered motif contrast.** A design powered for *d* ≈ 0.5 at this correction
would need of order a hundred surrogates per rung.

**γ against measured biological perturbation.** The prediction is that developmental and
activity-dependent rewiring occupies the low-γ regime that makes degeneracy cheap.
