# Pre-revision audit

This file records defects found in this project, the evidence for each, and what was
done about it. It exists so that the difference between the previous and current
manuscripts is traceable, and so that no corrected claim silently disappears.

It covers two rounds. **Sections A–C** are defects in the state of the repository at
commit `7c14647` ("docs: add Supplementary Information document and update README to
v4.0") — the v4 manuscript, whose central claims are withdrawn. **Sections D–D8** are
defects found in the *replacement* analysis while building it, which is to say our own,
and they are recorded on the same terms. **Sections E–E2** record where the corrections
changed a conclusion, including one place where a prediction this work makes was partly
falsified by this work's own test.

Within A–C, findings are ordered by severity. D onwards is chronological.

One pattern runs through nearly all of them and is the most transferable thing here:
every defect produced a plausible number and no error. A loader that swallowed its
exception returned an empty graph; an `argsort` over equal values returned index order;
a null model imposed a constraint it did not impose; a protocol control decoded nothing
and scored chance; `pdflatex` reported a duplicate macro and exited 0 with a complete
PDF. None was caught by a result looking wrong. Each was caught by checking a recorded
diagnostic, or by reading the code against the prose.

---

## A. Claims that were not supported by anything in the repository

### A1. Commercial silicon synthesis was reported but never run

**Claimed** (`paper/main.tex`, §"Synthesizable Neuromorphic Silicon Microarchitecture
& Physical Tapeout"): *"we designed, modeled, and physically synthesized the
ActiveDendriticCEN microarchitecture ... Standard-cell synthesis using Cadence Genus
and Innovus on TSMC 28nm HPC+ demonstrates a compact core macro die area of
0.201 mm² for 128 PEs ... dynamic efficiency of 116.71 pJ/SynOp"*.

**Evidence.** No Cadence tool, no TSMC library and no synthesis output exists in the
repository. The numbers originate in `experiments/hardware/run_rtl_cycle_benchmark.py`,
which runs a hand-written Python "golden model" (`hardware/sim/cen_rtl_golden_model.py`)
with energy and area constants supplied as literals. No logic synthesis, placement,
routing, parasitic extraction or power analysis was ever performed. The Verilog in
`hardware/rtl/` had never been passed through any synthesis tool at all.

**Severity.** Highest. Fabricated process-node, die-area and energy figures are a
research-integrity failure independent of whether the rest of the work is sound.

These figures were not confined to the manuscript. They were also the headline numbers
of the executive pitch dossier, the artefact most likely to be forwarded to someone
outside the project. It now sits in `paper/superseded_v4/pitch/` behind its own
retraction notice, and there is no replacement pitch.

**Resolution.** All TSMC/Cadence/tapeout claims removed. Replaced with
`experiments/hardware/run_open_synthesis.py`, which performs real logic synthesis and
technology mapping of the same RTL against the open Nangate45 cell library, reports
only what that flow can establish (gate count, flip-flop count, mapped pre-layout cell
area), and states explicitly that no place-and-route, parasitics or power signoff were
performed. No pJ/SynOp figure is claimed, and no clock frequency either: the open
toolchain ships no static timing analyser, so a critical path would have to be guessed
rather than measured.

### A2. "2,857× real-time acceleration" and "116.7 pJ/SynOp"

Both derive from the same Python model as A1: wall-clock of a NumPy loop divided by
simulated biological time, and a literal energy-per-operation constant. Neither is a
hardware measurement. Removed.

### A3. Human microscale connectome was never used

**Claimed.** H01 is presented throughout as one of four phylogenetic tiers, with a
"Human H01 Neocortex Census: 33,122 neurons, 26.5M synapses".

**Evidence.** `load_human_h01_microcircuit()` returns a graph with **zero edges**.
Its edge construction reads `gt_pair_data_5000nm_cutoff_with_skel_dists.json`, which
contains 293 synapse *pairs* used as proofreading ground truth for a merge-error
model, whose segment IDs do not intersect the cell table's IDs. The loader wraps the
whole block in `except Exception: pass`, so the failure was silent. The "26.5M
synapses" figure is a sum over a per-cell attribute column in a table, not a
connectivity graph: no human microscale network was analysed anywhere in the project.

**Resolution.** The human microscale tier is withdrawn. Human data is represented
only by HCP macroscale tractography, explicitly labelled as a regional network rather
than a neuronal connectome.

---

## B. Results contradicted by the repository's own data

### B1. The motif ablation table refutes the paper's central claim

`paper/main.tex` Table 5 reports, over 3 seeds:

| Condition | Test acc (%) | Mean SynOps |
|---|---|---|
| Full CEN | 99.83 ± 0.24 | 2.46 ± 1.61 |
| Ablate cFFL | 99.67 ± 0.24 | 3.12 ± 1.89 |
| Erdős–Rényi | 99.83 ± 0.24 | **1.85 ± 0.88** |

The accompanying text states that ablating cFFL *"produces a 26.8% surge in spurious
synaptic operations, confirming its causal role"* and cites `p < 0.05`.

**Problems.** (i) The Erdős–Rényi control is *better* than the full motif-structured
model on the very metric used to argue for motifs. (ii) The 26.8% difference is far
smaller than the reported standard deviations; with n = 3 it cannot support `p < 0.05`,
and no test statistic is given. (iii) Every accuracy is ≥ 99.67%, i.e. the task is
saturated and cannot discriminate architectures.

**Resolution.** The claim is withdrawn. The underlying question — does motif structure
confer a computational advantage over matched controls — is now the subject of the
paper, addressed with a nested control ladder, 15 instances per condition, effect
sizes, bootstrap CIs and multiplicity correction.

### B2. Benchmarks were synthetic and saturated

The "Neuromorphic DVS 10-Gesture Benchmark" is not DVS data. It is generated by
`generate_dvs_gesture_dataset()` in `experiments/benchmarks/run_dvs_gesture_benchmark.py`
from `np.random`, on 16 channels over 50 time steps. The "temporal pattern
discrimination" benchmark is likewise synthetic and reports 100.0% accuracy for three
of four architectures.

**Resolution.** Replaced with the Spiking Heidelberg Digits dataset (Cramer et al.,
IEEE TNNLS 2022): 20 classes, 700 cochlear channels, 12 speakers, with the canonical
speaker-held-out test split.

### B3. The mouse tier contains no inhibition

The MICrONS Pinky100 subgraph used (`soma_valence_v185.csv` +
`soma_subgraph_synapses_spines_v185.csv`) annotates 34 inhibitory somata, but **every
one of them has degree zero**: all 1,736 synapses in the file are excitatory→excitatory.
Consequently `iFFL = 0`, `RIP = 0` and `cFFL = FFL_total` identically, and the reported
"Mouse cFFL Z = +2.81" is a purely topological result presented as sign-attributed.

### B4. Human sign-attributed motifs are vacuous by construction

`load_human_hcp_macroscale()` assigns `sign = 1` to every region ("macroscale fiber
tracts are predominantly glutamatergic"). Diffusion tractography carries no synaptic
valence. Every human "coherent FFL" count therefore equals the plain FFL count, and
the reported `Z = +675.84` for human cFFL is the FFL Z-score relabelled.

---

## C. Statistical and methodological errors

### C1. Unattainable p-values
All motif p-values were reported as `p < 0.0001` from `K = 25` null samples. The
smallest attainable empirical p-value is `1/(K+1) ≈ 0.038`. Recomputed with `K = 500`
and reported with the resolution floor stated.

### C2. Table 1 does not match the underlying data
The manuscript's *C. elegans* FFL row lists null mean 10,960.6 and `Z = +70.95`;
`results/motif_significance_cross_species.csv` gives 11,551.3 and `Z = +73.93` for
the same quantity. The manuscript row duplicates the cFFL row's null statistics.

### C3. "Dale-compliant null model" imposes no constraint
The null is described as preserving degrees *and* Dale's principle, and E/I structure
is argued to be non-random on that basis. But when a synapse carries its presynaptic
neuron's valence, a double-edge swap `(u→v, x→y) → (u→y, x→v)` leaves `u` and `x`
presynaptic, so Dale's principle is preserved **automatically by any double-edge
swap**. The ensemble is exactly the standard Maslov–Sneppen ensemble, and it cannot
test whether valence placement is non-random. A separate valence-permutation control
is required and is now included as ladder rung L1→L2.

### C4. Non-motifs reported as motifs
Fan-in and fan-out counts are invariant under degree-preserving rewiring, so their
Z-scores are identically 0 with null standard deviation 0. They were tabulated as
motifs with `p = 1.0`.

### C5. Bi-fan nulls dominated by estimator noise
For graphs above 1,500 nodes, bi-fan counts used a 2,000-pair sampling estimator. The
resulting null standard deviation for *Drosophila* (394,750 against a mean of
2,539,798) is sampling noise, not ensemble variance, so the reported `Z = +14.10` is
an artefact. Bi-fan is now counted exactly.

### C6. Directed 3-cycles undercounted
The enumeration applied an index-ordering filter inside a loop that had already fixed
two of the three nodes, undercounting cycles by roughly a factor of two (*C. elegans*:
3,510 reported vs 7,128 exact).

### C7. Davis–Kahan bound reported as a finding
Table 8 reports that the empirical drift radius exceeds the Davis–Kahan bound by
218× to 19,190×, presented as evidence that "developmental rewiring is
non-adversarial". The measured spectral gaps are 1e-4 or smaller, which makes the
bound vacuous: it is a statement about the unsuitability of eigenvector perturbation
theory for near-degenerate spectra, not about connectomes. Replaced by a gap-free
bound on the finite-horizon input–output map, plus a falsifiable prediction tested
against adversarially constructed perturbations.

### C8. Energy comparisons were not matched
The headline "455× energy reduction" compares a network firing at a near-silent rate
against a dense network firing freely, at different accuracies. SynOps is a function
of firing rate and out-degree; without matching activity level or accuracy the ratio
measures sparsity, not architecture.

---

## D. The protocol under which no topology comparison can work

Not a defect of the original work, but recorded because it invalidated the first
version of the replacement experiment: when every neuron receives the stimulus and
every neuron feeds the readout, a reservoir with its **recurrent matrix set to zero**
scores identically to any connectome. Re-measured with the corrected harness (see D5):
59.65% ± 1.78 for a zeroed recurrent matrix against 59.57% ± 1.12 for the *C. elegans*
connectome. Under that protocol no topology comparison can detect anything.

The corrected protocol drives a sensory subset and decodes from a disjoint motor
subset; the zero-matrix control then falls to chance (4.95% against a 5.0% chance
level) while the connectome reaches 35.2%, which is the diagnostic that the benchmark
is sensitive to recurrent structure at all. Any structural comparison
lacking this control should be treated as uninformative.

## D2. The annealing energy was measured in the wrong units

Not a defect of the original work either, but it invalidated the first run of the
replacement null model and is recorded for the same reason as D.

The manuscript defines the annealing energy as the squared deviation of the running
motif census from its target, *measured in units of the unconstrained-ensemble standard
deviation* `s_m`. The code did not do that. Every call site computed

```python
scale = {k: max(std[k], 0.02 * real[k], 1.0) for k in real}
```

and the `0.02 * real[k]` term dominates for every motif in every network we use — for
the human HCP network it is roughly five times the null standard deviation. The energy
was therefore measured in units about five times coarser than the paper claimed, which
makes the penalty for drifting off the census correspondingly weaker.

**Consequence.** On near-symmetric networks the chain could not hold its target at all.
Human HCP is 100% reciprocal (3,046 reciprocal pairs over 6,092 edges), and ordinary
double-edge swaps destroy reciprocal pairs far more readily than they create them; with
a slack penalty the chain drifted to a worst-case relative error of **8.6%** and never
met the 2% convergence criterion, on any replicate. The L3 rung for that network was
consequently not a motif-matched surrogate at all, and the ladder's human row was
uninterpretable.

**Diagnosis.** Two separate causes, fixed separately. The move set could not maintain
reciprocity, which is addressed by a move that swaps two reciprocal *pairs* as units and
so preserves both degrees and reciprocity (`_propose_reciprocal`). That alone took the
error from 18.7% to 8.6% — an improvement, but still a failure. The residual was the
units: with `scale` set to what the paper says (`max(std, 1)`), the same chain converges
to **0.49%** worst-case error on human HCP and **0.19%** on *C. elegans*, at unchanged
edge overlap with the empirical graph (0.110 vs 0.111 for HCP), so the fix costs no
randomisation.

**Resolution.** The scale is now computed in exactly one place,
`src.nulls.swap_chain.census_scale`, rather than being restated at ten call sites where
it had already drifted from the definition. Two regression tests pin it: one asserts the
scale equals the null sigma and is not the old fraction-of-count floor, and one runs the
chain on a synthetic near-symmetric graph and requires both convergence and
randomisation. The floor of 1.0 is retained and is load-bearing for a different reason:
a motif with zero variance across the unconstrained ensemble would otherwise divide by
zero.

**Reading.** The general lesson is the one this audit keeps arriving at. The failure was
silent: a non-converged chain returns a graph, the experiment runs to completion, and the
resulting accuracy is entirely plausible. It was caught only because convergence was
recorded as a diagnostic per surrogate and checked, which is why those diagnostics are
reported in the paper rather than kept in the repository.

## D3. "Weakest-first pruning" was neuron ablation in index order

The degeneracy experiment compared three pruning directions: uniformly random,
weight-magnitude ("weakest-first", the standard pruning heuristic), and the first-order
maximally damaging ranking. The middle arm was not what it was called.

`signed_matrix()` builds the recurrent matrix from the connectome with entries `±1` —
these graphs are unweighted, and valence is the only thing an entry carries. Rescaling
to a common spectral radius multiplies every entry by the same constant. Every nonzero
entry therefore has **identical magnitude**, and `np.argsort` over a constant array
returns row-major index order. "Prune the weakest 30% of synapses" was executing as
"delete every outgoing synapse of neurons 0 through ~0.3n" — a structured whole-neuron
ablation, which is a far more damaging and completely different perturbation than the
weight heuristic named in the text.

**Consequence.** The reported tolerated fraction for that arm is a neuron-lesion result
presented as a synaptic-weight result. It also made the comparison incoherent: the
"weakest-first" arm was *more* damaging than the arm constructed to be maximally
damaging, which reads as a failure of the adversarial construction when it is an
artefact of the mislabelled control.

**Resolution.** `magnitude_edge_ranking` now raises rather than returning a degenerate
ordering when every edge magnitude is equal, so the failure cannot recur silently. The
second arm is replaced by `hub_edge_ranking`, a degree-targeted attack that concentrates
removal on the highest-out-degree neurons with random tie-breaking within a degree
class — a structured but task-blind direction, which is the comparison an unweighted
graph actually admits. The manuscript states why weight-magnitude pruning is not
reported. Two regression tests cover both halves.

**Reading.** The same shape as the others: an array operation with a well-defined but
unintended behaviour on degenerate input, producing a plausible number with no error
raised anywhere.

## D4. The gap-free bound was stated in the wrong norm

Found while checking the replacement theory against its own code. Proposition 1 bounds
the first-order movement of the finite-horizon transfer map by

    ||ΔH_t||_F ≤ t · ρ^(t-1) · ||E||_F · ||W_in|| · ||W_out||,

and its proof bounded the `t` expansion terms by `||A||^(t-1) ≤ ρ^(t-1)`. That
inequality runs the other way: for any matrix `ρ(A) ≤ ||A||`, with equality only for
normal matrices. The step is invalid, and the bound as stated does not follow.

**It is not a technicality on this data.** Connectome matrices are markedly non-normal.
Rescaled to the common operating radius ρ = 1.30 we measure `||A||₂ / ρ` of 1.2 for
*C. elegans* and 2.1 for mouse V1. Over a horizon of 15 the constant was therefore
understated by up to `2.1^14`. Since the operational alignment γ is defined as the
realised fraction of that constant, an understated constant inflates γ, and γ can then
exceed 1 — which contradicts its definition and, more importantly, is exactly the
regime the bound exists to describe: transient amplification in a non-normal recurrent
network is where eigenvalue-based reasoning fails, which is the paper's whole argument
against Davis–Kahan.

**Resolution.** Both the proposition and `worst_case_constant` now use the operator
norm `a = ||A||₂`. The proof is correct as stated in that norm and still requires no
spectral gap. A remark in the methods records why the spectral radius cannot be
substituted. The change rescales γ by a constant per network and so does not affect the
ordering of perturbation strategies, which is what the falsifiable prediction concerns.

**Test coverage.** The existing test asserted `0 ≤ γ ≤ 1` on a random matrix, which
passed under both versions — a random dense matrix is close enough to normal that the
error does not surface. The regression test now uses a shifted Jordan block, where the
old constant yields γ ≈ 79, and it was checked to fail against the previous code before
being kept.

## D5. The all-to-all protocol control decoded nothing

The protocol control is the argument that licenses every structural comparison in the
paper: under a naive protocol where every neuron is driven and every neuron decoded, a
reservoir with its recurrent matrix zeroed scores as well as a connectome, so that
regime measures the input projection rather than the wiring. The rebuilt
`run_protocol_controls.py` was not measuring that.

`sensory_motor_split` builds the two populations by slicing one permutation,
`perm[:n_in]` and `perm[n_in:n_in+n_out]`, which is disjoint by construction. The naive
arm requested `frac_input_nodes = frac_readout_nodes = 1.0`, so `n_in = n`, and
`perm[n:2n]` is **empty**. With no readout population the ridge had no features, and
both conditions scored chance:

```
naive_all_to_all       zero_recurrence  0.0495
naive_all_to_all       connectome       0.0495      <- should be ~0.60 for both
sensory_motor_disjoint zero_recurrence  0.0495      <- correct: chance
sensory_motor_disjoint connectome       0.3515      <- correct
```

**Consequence.** Read at face value the table says the naive protocol *cannot* solve the
task, which is the opposite of the finding, and it says so with a number that is
perfectly plausible for a broken reservoir. The disjoint arm — the one the whole paper
runs on — was correct throughout, so no headline result depended on this; the control
that justifies the protocol did.

**Resolution.** `ReservoirConfig` gains an explicit `disjoint_io` flag. With it off, the
sensory and motor sets are drawn independently and may overlap, which is what
"all-to-all" means. With it on, requesting more than the whole population now raises
instead of silently returning an empty set. Three tests cover the all-to-all case, the
refusal, and the default. The control was re-run and the manuscript reports the
re-measured value.

## D6. The trained-ladder harness used the protocol the paper rejects

Found while preparing to run the trained ladder, which the discussion names as the
natural next experiment. It is not a defect in any reported result — the trained ladder
has never been run for publication — but it would have been one the moment it was.

`MaskedRSNN` built its encoder as `randn(n_in, n)` and its readout as `Linear(n, n_out)`:
every neuron driven, every neuron decoded. That is precisely the all-to-all protocol the
paper's own control shows cannot detect recurrent structure, and training the recurrent
weights does not repair it — the encoder and readout can still solve the task between
them without the signal traversing the graph. Had the ladder been run and reported in
that configuration, it would have produced a flat table for the same reason the v4
reservoir comparison did.

**Resolution.** The encoder and readout are now restricted to the same disjoint
sensory and motor populations as the fixed-reservoir harness, using the same node seed,
with both restrictions held as fixed buffers so training cannot widen them. A
`Z_zero_recurrence` condition is added alongside the five rungs — the same trained
network with an empty recurrent mask — as the diagnostic that the trained benchmark is
sensitive to recurrent structure at all.

**A second trap found immediately after.** With the protocol corrected, every condition
scored chance with zero firing. The cause is the L1 firing-rate penalty: at the
inherited value of 1.0 — chosen when every neuron received direct drive — silence is the
global minimum once only ~15% of neurons are driven. Measured on *C. elegans* at four
epochs: `reg_rate` 1.0 gives mean rate 0.000 and accuracy 0.049 (chance); 0.1 gives
0.088 and 0.130; 0.0 gives 0.244 and 0.219. A collapsed network returns a perfectly
flat ladder, which is indistinguishable from the null result the paper reports, and is
the most dangerous possible failure mode for this particular experiment. The default is
now 0.02 with the reasoning recorded at the definition, and a test asserts it stays out
of the collapse regime.

**Status.** The trained ladder remains **unrun and unreported**. Tuning it to the
standard the rest of the paper is held to — a zero-recurrence control that separates
from the connectomes, and an epoch budget at which accuracy has stopped moving —
is outstanding work, and the discussion continues to list it as the natural next
experiment rather than as a result.

## D7. Two numbers, one macro name, and a PDF that built anyway

Small, but the same shape as everything else here, and worth one paragraph because it
went undetected by a pipeline built specifically to catch this class of problem.

Adding a per-property summary of the ladder contrasts introduced a second definition of
`\MotifEffectMax`, `\MotifEffectMaxG` and `\NMotifSig` alongside the existing ones, from
a different computation: `-8.38%` against `8.38%` for the same quantity. A duplicate
`\newcommand` is a hard LaTeX error, but `pdflatex -interaction=nonstopmode` reports it,
keeps the first definition and carries on, exiting 0 with a complete 15-page PDF. The
rebuild script checked for undefined references, unresolved `??` placeholders and the
page count — none of which this trips.

**Resolution.** `Macros.add` now raises when a name is defined twice, so the conflict
surfaces where it is created rather than in a log nobody reads. `rebuild_paper.sh`
additionally counts and prints lines beginning `!` from the LaTeX log, and prints them
when there are any. The duplicate emission was removed.

**Reading.** A checker only catches what it was told to look for, and "the build
succeeded" was never evidence of anything: `pdflatex` in nonstopmode exits 0 on errors
by design. This is the fifth distinct failure in this project where the exit status was
clean and the output plausible.

---

## D8. Protocol constants were transcribed into the manuscript by hand

The manuscript's premise is that no number in it is transcribed: every quantity is
emitted from a result file by `src/analysis/generate_numbers.py`. Six were not. The
reservoir's spectral radius, the reservoir size, the annealing sweep count, the instance
count, and the SHD time-step and channel counts were written as literals in the
generator itself:

```python
m.add("SpectralRadius", 1.30, "{:.2f}")
m.add("Tsteps", 100, "{}")
...
```

All six happened to match the code when checked. That is the whole problem: they match
until someone changes a default, after which the paper describes an experiment nobody
ran, with nothing in any output to indicate it. The generator was the one place in the
project specifically built to prevent this.

**Resolution.** `run_structural_ladder` now exposes `build_parser()` so the defaults can
be read without running the experiment; the SHD constants come from
`inspect.signature(load_shd_cached)`; and the instance count is taken from
`ladder_raw.csv`, so it reports what was actually run rather than what the default says.

## E. What this second round of corrections changed in the conclusions

The v4 audit (sections A–C) withdrew claims. This round (D2–D7) mostly corrected the
machinery that replaced them, but two findings moved as a result and are recorded here
so the change is traceable rather than silent.

**The motif premise is weaker than v4 stated, and weaker than the literature's framing.**
Recomputed against K = 500 degree-preserving surrogates with exact counts, enrichment is
real but not conserved: of 29 interpretable (network, motif) tests, 23 are enriched, 3
are significantly *depleted* and 3 are indistinguishable from the null. Coherent
feedforward loops are depleted in the *Drosophila* central complex (Z = −10.9), and mouse
V1 L2/3 sits at its null throughout. v4 reported these as uniformly and massively
overrepresented across phylogeny.

**The matched-rate control did not do what we expected, and we report it as it came out.**
The firing-rate confound is real — Erdős–Rényi surrogates run 1.77× hotter than the
connectomes they are matched to at equal gain, and rate predicts accuracy (r = 0.38) —
and we introduced the matched-rate ladder expecting it to explain the unstructured
control's advantage. It does not. At a matched operating point the Erdős–Rényi rung still
leads the connectome, by marginally more than at matched gain. The resulting claim is
therefore stronger and less comfortable than the one we set out to make: on this
benchmark connectome wiring is measurably *worse* than an unstructured graph of the same
size and density, and 67–91% of that cost is attributable to the degree sequence rather
than to anything the motif census adds.

Both are reported in the manuscript in those terms. Neither was the result we expected
when the experiment was designed, which is the only reason this section exists.

## E2. Our own falsifiable prediction was partly falsified

Recorded here because the alternative is to quietly narrow the claim, which is the
behaviour this audit exists to correct.

Proposition 2 predicts that the tolerated perturbation magnitude scales as `1/γ`, where
γ is the operational alignment of the perturbation direction. The manuscript presented
this as the falsifiable content of the degeneracy theory, and it is: the test either
comes out or it does not.

**What held.** Within each network, γ is larger for structured perturbation directions
than for random ones (2.6× for degree-targeted removal, 3.0× for the first-order
maximally damaging ranking), and the tolerated fraction runs in the opposite order —
83% for random, 73% for degree-targeted, 64% for adversarial. Network by network, γ
orders the three strategies correctly in 3 of 5.

**What did not.** The proposition asserts a scaling, not an ordering, so the product
`γ · ||E||_F^max` should be constant across directions within a network. It varies by a
median factor of 2.2. γ identifies which direction is more damaging and does not
correctly predict by how much.

**A related misuse, ours.** γ is a realised fraction of a worst-case constant `C(T, a)`
that depends on `||A||₂^T`, and the median γ spans a factor of roughly 7×10⁵ across our
five networks. Pooled γ medians across networks — which the first version of this
analysis reported, and which the figure's lower panel plotted — are not a meaningful
quantity. All comparisons are now made within a network and then aggregated, and the
figure normalises accuracy to each network's own unpruned baseline before pooling.

**Also found, and worth stating because it changes how the survival curves read.** All
three pruning strategies *improve* accuracy at low fractions, peaking around 40–60% of
synapses removed. These reservoirs are over-connected for the task, which is consistent
with the ladder's finding that degree heterogeneity is costly here. It also means the
tolerated fractions are conservative, and that several random-pruning runs still retained
90% of unpruned accuracy at the largest fraction tested, so that figure is a lower bound
and is reported as one.

## E3. The motif effect is real, and it has the wrong sign

The first version of the structural ladder ran with the mis-scaled annealing energy of
D2, so its L3 rung was not a motif-matched surrogate on the networks where the chain
failed to converge — human HCP worst, at 18.7% relative error. Under that broken sampler
the human motif contrast came out significant (`g = -2.68`). The natural expectation was
that it would disappear once the sampler converged, and that expectation was recorded
before the re-run.

It did not disappear. With the corrected scale the chain converges on all five networks
(100% of surrogates, worst-case relative error 0.48%), and the human motif contrast is
still significant at `g = -2.06`, joined by *C. elegans* at `g = -1.46`. Across the
ladder the motif census is now significant in **2 of 5** networks with a median effect of
**-3.50 percentage points**, against **-4.65** for the degree sequence. The two are
comparable in size.

**What changed in the manuscript.** The earlier draft described the motif contrast as the
smallest in the ladder and the benchmark as resolving "the structure furthest from the
motif hypothesis". That is wrong and has been rewritten. The finding is stronger and
less comfortable than the one it replaces: adding a connectome's motif census to a degree-
and valence-matched surrogate does not fail to help, it measurably *hurts*, and does so
by about as much as the degree sequence does. Every structural property in the ladder
that moves accuracy at all moves it downward.

**A disagreement we are not resolving.** The task metric and the task-independent
dynamical probes agree on the direction of every effect and on the degree sequence
carrying the largest share, but they disagree about the motif rung: it is comparable to
the degree contrast on accuracy and the smallest contrast in the ladder on memory
capacity and kernel rank. The manuscript states the divergence and offers a conjecture
labelled as one, rather than choosing whichever measure supports a cleaner story.

**Reading.** Recorded because the expectation was wrong in the direction that would have
been convenient. Had the human contrast vanished, the paper would have reported a clean
null on motifs; instead it reports a real effect pointing the opposite way to the
hypothesis under test, which is a different claim requiring different caveats.

## D9. The manuscript quoted a smoke test

Caught within minutes of being introduced, and recorded because the mechanism is the one
this project keeps meeting.

Two new experiments (`run_gain_sweep_ladder.py`, `run_memory_capacity_sweep.py`) were
smoke-tested at one network and one instance before being launched properly. The smoke
tests write to the same paths as the real runs. The macro generator read them, the
manuscript quoted them, and the build reported **zero unresolved numbers** — because from
the generator's point of view nothing was missing. A one-instance, one-network result and
a publication run differ in no way that any existing check could see: same filename, same
columns, same plausible magnitudes.

**Resolution.** `generate_numbers.py` now carries the expected shape of every result file
it reads — how many networks, how many instances — and prints `UNDERPOWERED INPUTS` naming
any file that falls short, listing what the manuscript is therefore quoting.
`rebuild_paper.sh` reports the count alongside LaTeX errors and unresolved numbers.

**Reading.** The existing checks all asked *is a number missing?* None asked *is this
number worth quoting?* Those are different questions, and only the first had ever been
automated. Provenance is not visible in a value.

## D10. A rank collapse killed the gain sweep, and the trigger was not isolated

The gain-axis sweep aborted after 80 of 200 surrogates with
`numpy.linalg.LinAlgError: Singular matrix`, raised from the ridge readout.

**Diagnosis, partial.** The readout appended an unpenalised column of ones to the design
and solved the resulting Gram system. That construction is singular whenever a feature
column is collinear with the intercept, and high gain produces exactly that: measured on
*C. elegans*, the effective rank of the readout features falls from 0.96 of their number
at ρ = 1.30 to 0.73 at ρ = 2.50 as the reservoir saturates and every neuron fires at
every step.

**What was not established.** We could not reproduce the raise. Constant columns,
all-constant designs, dead columns, exactly duplicated columns, NaN and Inf entries, and
float32 precision loss at the real problem's dimensions were each run against the old
implementation and none of them raised. The failing cell's parameters were not recorded,
because the exception aborted the pool before any row was written — which is itself the
defect worth fixing.

**Resolution.** Two changes, only the second of which depends on knowing the cause.

The readout now fits on centred features and recovers the intercept in closed form. This
is the textbook handling of an unpenalised intercept and it makes intercept collinearity
impossible by construction, so it removes that class of failure whether or not it was the
class encountered. A test pins the property and states plainly that it does not pin the
original trigger.

More importantly, the sweep no longer aborts. A cell that raises is recorded as a row
with `ok=False` and the exception text, so a single bad evaluation costs one point on one
curve instead of the other 199 surrogates. The effective feature rank is recorded for
every cell, so saturation is visible as a number rather than inferred from a strange
accuracy.

**Reading.** The instinct on hitting an exception is to find and fix the cause. Here the
more valuable change was to make the failure non-fatal and self-documenting, because that
converts an unreproducible crash into a recorded data point. If it recurs, it will arrive
with its parameters attached.

## E4. A one-instance smoke test suggested a result that ten instances did not support

The clearest case in this file of a number being believed before it was earned, and the
sequence is worth recording because at no point did anything look wrong.

**What happened.** Searching the literature turned up that the reservoir-computing
result we appeared to contradict (Suárez et al., *Nat Commun* 2024) is measured at
criticality, ρ = 1, while our ladder ran at ρ = 1.30. A one-instance, one-network probe
of memory capacity across the gain axis showed the connectome-versus-null contrast
turning positive at exactly ρ = 1.00–1.15 and negative on both sides. That is a clean,
striking, publishable pattern, and it was used to rewrite the paper's title, abstract,
introduction, results and discussion around the claim that we reproduce the published
direction at the published operating point and reverse it nearby.

**What the real run showed.** At ten paired instances across all five networks, the
connectome leads at ρ = 1.00 in 2 of 5 by the sign of the mean and in **0 of 5** once the
paired confidence interval is applied. The sign pattern across the gain grid is scatter,
not a crossing. The one-instance pattern was noise.

**And the correction to the correction.** The obvious next move — reporting a failed
replication — would also have been wrong. The design's minimum detectable effect at that
operating point is a median of 1.04 capacity units, **46% of the null network's own
capacity**. An advantage of any interesting size sits inside our confidence intervals. On
one network of five (mouse V1) the variance is low enough to resolve 8%, and there the
difference is +0.05 ± 0.11. So the honest report is a **non-detection on an underpowered
design**, not evidence against the published result, and the manuscript says which of the
two it is and lists the implementation differences that could account for it.

**What survived.** The accuracy sweep, which was run at eight paired instances and
checked with paired confidence intervals before being written up: the
connectome-versus-null contrast changes sign along the gain axis in 3 of 5 networks,
significant on both sides of the crossing, with a within-network swing of up to 23.4
percentage points. At criticality the unstructured rung leads significantly in 5 of 5.
That is the paper's central claim and it is the one the data supports.

**Reading.** Three framings of this paper were written and two were discarded, each time
because a measurement came back differently. That is the process working. The failure
mode to guard against is not being wrong early — it is the smoke test and the real run
writing to the same filename, which is what allowed a one-instance result into the
manuscript in the first place (§D9), and the pull of a clean story, which is what made it
attractive enough not to question for an hour.

## D11. Pandas read the control condition as missing

The unit-model experiment labelled its two arms `"empirical"` and `"null"`. Pandas'
default `na_values` list contains the bare string `"null"`, so on read-back the entire
control arm became `NaN`: `df.kind.unique()` returned `['empirical', nan]`, every paired
contrast came back empty, and the analysis raised a `KeyError` on a DataFrame with no
rows.

The 1800-evaluation file on disk was correct throughout. Only the read was wrong, and it
was wrong in a way that removed half the experiment while leaving a file of the right
size with the right column names.

**Resolution.** The condition is renamed `"rewired"` in the experiment, so the value can
never be parsed as missing whatever reads it. The analysis additionally passes
`keep_default_na=False`, maps the old spelling forward so existing files still load, and
coerces the numeric columns explicitly.

**Reading.** It surfaced immediately here because the downstream code crashed. Had the
analysis instead computed a mean over the surviving arm, it would have produced a
plausible number for a comparison that had silently lost its control.

## E5. The mechanism failed; the finding did not

The unit-model experiment was built to test a specific account: that a firing threshold
plus a heavy-tailed degree sequence silences a connectome's low-degree neurons, that a
`tanh` unit cannot be silent, and that this is why the two literatures disagree.

**The finding is clean.** On the five regional connectomes distributed with `conn2res`,
with identical graphs, identical degree-preserving rewirings, identical inputs, delays
and readout, and paired over ten instances:

- **rate units:** the empirical connectome leads its rewiring in **45 of 45**
  (network, gain) cells, every one significant. The published result replicates
  completely, on the original authors' own data, across the whole gain axis.
- **spiking units:** the advantage is absent. At ρ = 1 the connectome leads
  significantly in **0 of 5** networks and trails significantly in one.

**The mechanism is wrong.** Silencing does not differ between the empirical connectome
and its rewiring in the spiking model — 61.8% against 62.9% silent, and the silencing gap
does not predict the capacity gap (r = −0.03, p = 0.84). What differs is the *level*:
0.9% silent in rate units against 62.4% in spiking ones, with total capacity falling from
4.4–8.5 to 0.6–2.0. Thresholding compresses the measurement for every topology alike; it
does not penalise connectome structure specifically.

So the honest statement is narrower than the one the experiment was designed to support:
the advantage does not survive the change of unit model, and we do not have an account of
why beyond the collapse in measurable capacity.

**A floor to declare.** For human the spiking capacity is 0.049 against 0.019 — both
effectively zero, so that network contributes nothing either way. The other four retain
capacities of 1.0–1.9 and are informative.

**Reading.** The prediction was specific enough to be wrong, and it was. The result it was
built to explain stands on its own, and is reported without the explanation rather than
with a weaker one retrofitted to it.

## E6. The tuning control weakened our own headline, and we kept the weaker version

The unit-model result invited one obvious objection: the spiking reservoir was run at a
single threshold, leak and input scale, none of them tuned, with 62% of its units silent.
A non-detection in a misconfigured model is not a finding, and this project had already
made that mistake once (§E4).

So the objection was given an experiment designed to refute us: 36 spiking configurations
spanning input gain 0.5–16 and threshold 0.25–1.0, 1,728 evaluations, asking whether the
connectome advantage appears anywhere the network is computing.

**It does, in a few places.** Of 144 (network, configuration) cells, 80 are healthy by the
pre-stated criterion. Among those the connectome leads significantly in **3** and trails
significantly in **7**, median difference −0.16.

**What changed.** The claim was written as "with spiking units the advantage is absent."
That is not what the data supports and it has been rewritten: the advantage is *sporadic
and inconsistent in sign* — present in 3 of 80 well-conditioned settings, reversed in more
of them. Against 45 of 45 cells in the graded model, significant in every network at every
gain, the contrast still carries the paper, but it is a weaker and more accurate statement
than the one it replaces. The title changed with it, from "does not survive a firing
threshold" to "is a property of the unit model it is measured in."

**What the control also established.** The configuration used for the headline table is
not a pathological corner: it sits within 20% of the best capacity found anywhere in the
sweep, and raising input gain sixteenfold does not clear the silencing. The spiking
harness is not broken.

**Reading.** The control was run because a reviewer would have demanded it, and it cost us
the stronger version of our own headline. Running it before submission rather than after
is the whole difference between a finding and a correction.

## D12. The null ensemble was a reimplementation, not the published one

**What the code did.** `strength_preserving_null` rewired the binary topology by
double-edge swaps and then re-assigned the original multiset of weight values to the new
edges in order of endpoint-degree product. This preserves the degree sequence exactly and
the weight distribution exactly, and it was written from the published description
"randomised while preserving density and degree sequence."

**What the published comparison uses.** `conn2res` calls the Brain Connectivity Toolbox
routines `randmio_und_connected` / `randmio_dir_connected`, in which each weight travels
with the edge carrying it and swaps that disconnect the graph are rejected. These are
different ensembles. Ours does not preserve the association between a weight and its
edge; theirs does.

**Evidence that it mattered.** Both nulls were run over the full design — 5 networks ×
10 paired instances × 9 gains × 2 unit models, 2,700 evaluations. Of the 10
(network, unit model) cells, the two give the connectome's advantage **opposite signs in
6** and **opposite significant verdicts in 3**: one ensemble says the connectome leads at
p < 0.05, the other says it trails at p < 0.05, on the same graphs at the same operating
point. Median absolute disagreement 0.73 capacity units, maximum 6.98.

**Mechanism, with the prediction that could have refuted it.** Rank-reassignment does not
only randomise; it *installs* a weight–degree association the data does not have. Across
the four weighted networks it shifts the weight–degree correlation by a median of +0.66,
against +0.06 for `randmio` (never above 0.16). A reservoir whose largest weights sit on
hub-to-hub edges is a poor one, so the connectome's advantage over it is inflated. The
prediction: on a network with no weight variation the two constructions are the same
ensemble and must agree. Macaque is binary (Gini = 0). There the two nulls shift the
association by 0.000 and the measured advantages differ by 0.008 capacity units.

**Severity.** Highest of the D series. It is not a coding error — the code did exactly
what it was written to do — and no test could have caught it, because both ensembles are
valid samplers of correctly-specified constraints. It was found only by reading the
toolbox source.

**Resolution.** `published_null` now calls BCT directly and `bctpy` is a declared
dependency. The original is retained as `rank_reassigned_null`, a first-class condition
rather than a deleted mistake, and the comparison between the two is now
§\ref{sec:res-nullchoice} of the manuscript.

---

## E7. Two conclusions reversed when the null was corrected

Both were stated in the previous draft and both are withdrawn in the form they were
stated. They are recorded together because they changed for the same reason.

**E7a. "With graded units the published result replicates completely."** Measured against
the rank-reassigned ensemble the connectome led in 5 of 5 networks, every one significant,
median +2.13. Against `randmio` it leads in 1 of 5 and trails significantly in 3, median
−0.51. The manuscript no longer claims a replication. It claims something narrower: that
the null choice alone moves the result by more than the reported effect, which is a
sufficient explanation for a reproduction failure without the original being wrong. Our
harness also differs from the published one in its input projection, delay set and ridge
solver, and we say so rather than attributing the whole gap to the null.

**E7b. "Our explanation for this is wrong, and we report it as wrong."** The previous
draft reported the differential-silencing mechanism as falsified: against the
rank-reassigned null the silencing gap did not predict the capacity gap. Against `randmio`
it does — r = −0.585, p = 2.5 × 10⁻⁵ over 45 (network, gain) cells, in the predicted
direction. The manuscript now reports the supported version, states that we previously
reported the opposite, and keeps the claim modest: differential silencing accounts for
part of the spiking arm's behaviour and leaves most of the variance unexplained.

**Reading.** E7b is the uncomfortable one. A wrong control did not merely add noise; it
manufactured a clean positive result in E7a and destroyed a real one in E7b, in the same
dataset, from the same code. Neither direction of error announced itself.
