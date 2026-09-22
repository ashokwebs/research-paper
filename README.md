# Computational Coupling

**Four choices decide whether a connectome beats its rewiring.**

*Computational coupling* is the extent to which what a network computes is attributable
to how it is wired. It is never measured directly — only as the gap between a network's
score and the scores of surrogate networks, under some unit model, at some operating
point, through some readout. This project shows that each of those four choices can flip
the sign of that gap on identical data, so a coupling reported as a single signed number
is under-determined by its own methods section.

**Author:** Ashok Raj Pasala
**Co-author:** Snigdha Gorai
**Guide:** Jeevana Jyothi P.
School of Computer Science and Engineering, VIT-AP University,
Amaravati 522237, Andhra Pradesh, India

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-v5%20%2827pp%29-success.svg)](paper/computational_coupling_v5.pdf)
[![Audit](https://img.shields.io/badge/audit-withdrawn%20claims-critical.svg)](AUDIT.md)

**Paper:** *Computational Coupling: Four Choices Decide Whether a Connectome Beats Its
Rewiring* — [`paper/computational_coupling_v5.pdf`](paper/computational_coupling_v5.pdf)

> **This version supersedes and retracts an earlier one.** Every withdrawn claim and the
> evidence against it is in [`AUDIT.md`](AUDIT.md). See
> [Relationship to the previous version](#relationship-to-the-previous-version-of-this-project).

---

## Contents

- [What this project asks](#what-this-project-asks)
- [What is new here](#what-is-new-here)
- [Relationship to the previous version](#relationship-to-the-previous-version-of-this-project)
- [Data](#data) · [Layout](#layout) · [Every experiment in this study](#every-experiment-in-this-study)
- [Reproducing](#reproducing) · [Tests](#tests) · [Documents](#documents)
- [Citing](#citing) · [License](#license)

---

## What this project asks

Connectomes contain statistically overrepresented subgraphs, and it is widely argued that
these *motifs* are what make biological wiring computationally effective. The standard
evidence is a comparison: score a real connectome on a task, score a rewired version of
it, and credit the wiring with the difference.

That difference is the quantity we call **computational coupling**. Writing it down makes
its dependencies visible:

> κ(W; N, U, ρ, R) — the standardised gap between the capacity of network `W` and the
> capacity of surrogates drawn from ensemble `N`, measured with unit model `U`, at
> operating point `ρ`, through readout protocol `R`.

This is the Z-score the field already reports. Naming it is useful because the name
carries its four arguments. "The connectome beats its rewiring" is the claim κ > 0 — and
κ is a function of four choices besides the connectome, each normally reported in a
subordinate clause, if at all.

**We varied all four, and each one can flip the sign of κ on identical data.** A coupling
reported as a single signed number is therefore under-determined by its own methods
section.

The project also builds the control that the motif argument actually needs — an ensemble
differing from a connectome **in its motif census and in nothing else** — because rewiring
perturbs the census together with every statistic that co-varies with it, so a difference
against an ordinary rewired null cannot be attributed to motifs in particular.

| Choice | Setting A | Setting B | What the choice alone does |
|---|---|---|---|
| **Null ensemble** `N` | BCT `randmio`: weights travel with edges | Rank-reassigned: same weights, re-placed by degree rank | Opposite signs in 6 of 10 cells; opposite **significant** verdicts in 3 |
| **Unit model** `U` | Graded `tanh`, never silent | LIF, silent below threshold | Capacity compressed **4.9×**; 60.7% of spiking units never fire |
| **Operating point** `ρ` | At criticality, ρ = 1 | Above it, ρ > 1 | Contrast changes sign in 3 of 5 networks, significant on both sides |
| **Readout** `R` | All-to-all: every neuron driven and decoded | Sensory/motor-disjoint | Under A, deleting the recurrent matrix costs **nothing** (59.6% vs 59.6%) |

The fourth row is the limiting case: under an all-to-all readout the coupling is zero by
construction, because no change to the wiring could have moved the number.

## What is new here

**1. "Degree-preserving randomisation" names two different ensembles, and they disagree**
(`experiments/ladder/run_unit_model_comparison.py`,
`experiments/ladder/run_null_mechanism.py`).
The reservoir literature reports that empirical connectomes beat degree-preserving
rewirings on memory capacity — a result increasingly invoked as motivation for
neuromorphic hardware. The phrase describing the control does not determine an ensemble:
a weight can travel with its edge (BCT `randmio_*_connected`, what `conn2res` calls) or be
re-assigned to the rewired graph afterwards. We built the second believing it was the
first. Across 10 (network, unit model) cells the two give the connectome's advantage
opposite **signs** in 6 and opposite **significant verdicts** in 3; against `randmio` the
connectome leads in **1 of 5** networks with graded units, against rank-reassignment in
**5 of 5**, median +2.13. The cause is that rank-reassignment *installs* a weight–degree
association the data lacks (+0.66 median shift against +0.06), which predicts the two must
agree exactly on an unweighted network — and on the one binary network in the set they
agree to 0.008 capacity units. This is not a refutation of the published finding; it shows
the null choice alone moves the result by more than the effect being reported.

**1b. What a unit is also matters** (same experiment).
Graded `tanh` units are never silent; sub-threshold LIF units contribute nothing.
Capacity compresses by a factor of **4.9** between the two, and **60.7%** of spiking units
never fire against **0.6%** of graded ones. Our predicted mechanism — *differential*
silencing — was reported as refuted against our mistaken null and is **supported** against
the published one (r = −0.58, p = 2.5×10⁻⁵ over 45 cells). We report both, and the audit
records why the conclusion moved.

**2. The contrast also changes sign along the gain axis** (`experiments/ladder/run_gain_sweep_ladder.py`).
The reservoir literature reports that empirical connectomes beat degree-preserving
rewirings on memory capacity, measured at criticality. We sweep the gain axis and find the
contrast **changes sign** along it — in 3 of 5 networks on task accuracy, significant on
both sides, swinging up to 23.4 percentage points within a network. At criticality the
*unstructured* graph leads in all five. We also attempted the published memory-capacity
comparison directly and did not detect the advantage, but report that as a
**non-detection rather than a refutation**, because our design there resolves only an
effect about half the size of the null's own capacity. A single-operating-point
comparison establishes a fact about that operating point; reported without the curve it
is indistinguishable from a fact about the topology.

**3. A motif-preserving null ensemble** (`src/nulls/swap_chain.py`).
Degree-, valence- and census-matched surrogates, sampled by an annealed Metropolis
chain over exact, incrementally updated motif counts. It reproduces a connectome's
motif spectrum while randomising everything else, and it does *not* reproduce the
held-out 4-node statistic it was not asked to preserve.

**4. A structural control ladder** (`src/nulls/ladder.py`).
Five nested rungs, each adding exactly one structural property to the one below:
density → degree sequence → valence placement → motif census → the empirical
connectome. The contribution of each property is the difference between adjacent
rungs, with everything else matched by construction.

**5. Two controls that are usually missing** (`experiments/ladder/`).
A reservoir comparison is only a topology comparison if the recurrent matrix is used
and if the networks are compared at the same operating point. We report the
zero-recurrence control for the first, and re-run the whole ladder with every network
individually recalibrated to a common firing rate for the second — matching spectral
radius equalises gain, not activity level, and activity level both predicts accuracy
and sets the synaptic-operation count.

**6. A gap-free perturbation theory** (`src/theory/transfer_perturbation.py`).
The Davis–Kahan bound usually invoked for degeneracy is inapplicable to measured
connectomes — their spectral gaps make it vacuous. We bound the finite-horizon
input–output map instead, which needs no gap, and derive the **operational alignment**
γ of a perturbation, with the falsifiable prediction that tolerance scales as 1/γ. The
prediction is tested against a constructed maximally-damaging perturbation, not only
against random ablation.

**7. An honest hardware report** (`experiments/hardware/run_open_synthesis.py`).
Real logic synthesis of the accompanying RTL with Yosys + ABC onto the Nangate45 open
cell library, reporting only what such a flow establishes.

## Relationship to the previous version of this project

This work **supersedes and retracts** the previous manuscript in this repository. The
earlier version reported commercial 28 nm silicon figures that were never measured, a
human microscale tier whose connectivity graph was empty, sign-attributed motif results
for datasets containing no usable inhibition, p-values below the resolution of the null
ensemble that produced them, and an ablation table whose own control row contradicted
the conclusion drawn from it.

Every withdrawn claim and the evidence against it is documented in
**[`AUDIT.md`](AUDIT.md)**. It is kept in the repository deliberately: the corrected
analysis is only interpretable against a record of what it replaces. The retracted
documents themselves are in **[`paper/superseded_v4/`](paper/superseded_v4/)** and their
data in **[`results/superseded_v4/`](results/superseded_v4/)**, moved aside and
labelled, because they previously sat beside the current manuscript and its results with
nothing to indicate they had been withdrawn. That includes the v4 **executive pitch
dossier**, every hardware number in which is fabricated; it carries its own retraction
notice because a pitch is the artefact most likely to be forwarded. There is no
replacement pitch.

One finding from that audit is worth repeating here, because it is easy to reproduce
by accident. In a reservoir where **every neuron is both driven and decoded**, zeroing
the recurrent matrix entirely leaves benchmark accuracy unchanged — the wiring is not
used. Any structural comparison run in that regime measures the input projection. The
protocol used here drives a sensory subset and decodes from a disjoint motor subset, and
reports the zero-recurrence control as the diagnostic that the benchmark is sensitive to
topology at all.

A second finding from the audit has the same shape. Matching spectral radius across
networks does **not** match the rate at which they fire, Erdős–Rényi surrogates run
roughly twice as hot as the connectomes they are matched to at the same gain, and
firing rate predicts accuracy. A comparison at matched gain therefore confounds
topology with operating point — and, like the protocol failure, it produces entirely
plausible numbers with nothing in the output to indicate a problem.

Removing that confound did not give us the result we expected. At a matched firing rate
the Erdős–Rényi control still beats the connectome, by slightly *more* than at matched
gain. On this benchmark connectome wiring is measurably worse than an unstructured graph
of the same size and density, and every structural property that moves accuracy at all
moves it downward — including the motif census itself, which costs a median 3.5
percentage points against 4.7 for the degree sequence. The motif hypothesis predicts the
opposite sign. We report it that way.

---

## Data

| Network | Source | Valence available |
|---|---|---|
| *C. elegans* whole nervous system | Cook et al. 2019 | yes |
| *Drosophila* central complex, antennal lobe | FlyWire v630 | yes |
| Mouse V1 L2/3 | MICrONS Pinky100 | **no** — every annotated inhibitory soma is disconnected in this release |
| Human, regional | HCP group consensus tractography | **no** — tractography carries no synaptic sign |

Task data: **Spiking Heidelberg Digits** (Cramer et al. 2022), a real spike-based
benchmark, replacing the synthetic pattern tasks used previously (which saturated near
100% for every architecture and so could not discriminate between them).

## Layout

```
src/motifs/fast_census.py        exact sign-attributed motif census, O(d) incremental updates
src/nulls/swap_chain.py          degree-preserving chains + the annealed motif-preserving null
src/nulls/ladder.py              the five-rung structural control ladder
src/theory/transfer_perturbation.py  gap-free bound, operational alignment, adversarial ranking
src/eval/reservoir.py            reservoir protocol (sensory/motor disjoint)
src/eval/dynamics.py             memory capacity, kernel & generalisation rank
src/eval/trained_snn.py          surrogate-gradient training of masked recurrent SNNs
src/tasks/shd.py                 Spiking Heidelberg Digits loader
experiments/                     experiment drivers (one per result)
src/analysis/                    figure, table and LaTeX-macro generation
hardware/rtl/                    Verilog sources
AUDIT.md                         record of withdrawn claims
paper/superseded_v4/             the retracted v4 manuscript, figures and pitch dossier
results/superseded_v4/           the result files those claims were drawn from
```

The listing above is the code the current paper uses. Other packages under `src/`
(`architecture/`, `benchmarks/`, `learning/`, `remodeling/`, `simulation/`) and the
matching directories under `experiments/` belong to v4 and are not exercised by any
current result. They are kept because `AUDIT.md` cites them as the origin of specific
withdrawn numbers — the energy and area constants of §A1 are literals in
`hardware/sim/cen_rtl_golden_model.py`, and the synthetic benchmark of §B2 is generated
by `experiments/benchmarks/run_dvs_gesture_benchmark.py` — and deleting them would make
the retraction harder to check rather than easier.

## Every experiment in this study

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

## Reproducing

```bash
git clone https://github.com/Ashok-clg/computational-coupling.git
cd computational-coupling
python -m venv .venv && .venv/bin/pip install -e .

# Task data (~350 MB)
mkdir -p data/raw/shd && cd data/raw/shd
curl -L -O https://zenkelab.org/datasets/shd_train.h5.gz
curl -L -O https://zenkelab.org/datasets/shd_test.h5.gz && gunzip *.gz && cd -
```

Then every experiment. This list is generated from the same source as the
inventory above and the table in the manuscript, so all three agree by
construction:

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

Then the statistics, figures, tables, macros and three LaTeX passes. This
reports unresolved numbers, undefined references and LaTeX errors at the end,
because "the build succeeded" is not evidence of anything on its own --
pdflatex exits 0 on a hard error by design:

```bash
./src/analysis/rebuild_paper.sh
```

`experiments/ladder/run_trained_ladder.py` is deliberately **not** in that list. It
trains the recurrent weights rather than freezing them, which is the natural next
experiment and is discussed as such in the paper, but it has not been run to publication
standard and nothing in the manuscript reports it. Its harness has been corrected — it
previously used the all-to-all protocol the paper rejects — and its firing-rate penalty
retuned, but it still needs a zero-recurrence control that separates from the
connectomes before any number from it means anything. See `AUDIT.md` §D6.

Every number in the manuscript is emitted into `paper/generated_numbers.tex` by
`generate_numbers.py`; none is transcribed by hand, and a missing result appears in the
PDF as a visible `??` rather than as a stale value.

## Tests

```bash
.venv/bin/pytest
```

The suite covers the census against brute-force enumeration, exactness of the
incremental updates, degree and valence preservation in every null, convergence and
randomisation of the annealed chain, the reservoir protocol controls, and the
perturbation bounds.

## Documents

| File | What it is |
|---|---|
| [`info.md`](info.md) | **Everything about the project** — question, method, results, what is not claimed, layout, reproduction, open work |
| [`AUDIT.md`](AUDIT.md) | Every withdrawn claim and the evidence against it, including defects found in the replacement analysis |
| [`pitch/EXECUTIVE_PITCH.md`](pitch/EXECUTIVE_PITCH.md) | Executive summary (v5 — replaces a withdrawn one) |
| `paper/computational_coupling_v5.pdf` | The manuscript |
| [`research/`](research/) | Planning documents from the project's start, superseded; kept so the retraction is checkable against the plan |
| [`paper/superseded_v4/`](paper/superseded_v4/) | The retracted v4 manuscript, figures and pitch dossier |

**A note on what is and is not in this repository.** The retracted v4 manuscript,
monograph, supplement and pitch dossier are kept as **LaTeX sources** under
[`paper/superseded_v4/`](paper/superseded_v4/), so every claim `AUDIT.md` cites can be
checked against what was actually written. Their compiled PDFs are not published here:
their title pages carry the authors' contact details, and `build_docs.sh` already refuses
to render fresh copies of withdrawn claims (see its header). The retraction notices in
those directories are tracked and rendered.

A PDF of each Markdown document is built alongside it by `./src/analysis/build_docs.sh`,
which first runs `check_doc_numbers.py` to verify that every figure quoted in `info.md`
and the pitch still matches `paper/generated_numbers.tex`. Those two documents are written
for people rather than compiled from macros, so they are the one place a number could go
stale against a re-run; the check refuses to render a PDF from prose that disagrees with
the results. The retracted v4 pitch and manuscript are deliberately **not** rendered —
their retraction notices are.

## Citing

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it as the
**"Cite this repository"** button in the sidebar.

```bibtex
@misc{pasala2026coupling,
  title  = {Computational Coupling: Four Choices Decide Whether a Connectome
            Beats Its Rewiring},
  author = {Pasala, Ashok Raj and Gorai, Snigdha and P., Jeevana Jyothi},
  year   = {2026},
  note   = {School of Computer Science and Engineering, VIT-AP University,
            Amaravati 522237, Andhra Pradesh, India},
  howpublished = {\url{https://github.com/Ashok-clg/computational-coupling}}
}
```

If you cite the null-ensemble result, please say **which** ensemble you mean — and cite
the code that generates it, not a description of it. That is the point of the paper and of
this repository: the phrase "degree-preserving randomisation" does not determine one.

## License

MIT — see [LICENSE](LICENSE).
