# COMPUTATIONAL COUPLING — Research Question Specification

> **Status: planning document, superseded.** This was written at the start of the project
> (September 2026), before any experiment was run, and it records the research plan and
> novelty analysis as they stood then. It is **not** a statement of findings. Several of
> its premises did not survive: the human H01 microscale tier was withdrawn when its
> connectivity graph turned out to be empty, the CoupledEventNet / neuromorphic-efficiency
> direction was withdrawn along with the fabricated silicon figures supporting it, and the
> central motif hypothesis it sets out was tested and came out with the opposite sign.
>
> For what the project actually found, see [`../info.md`](../info.md) and the manuscript.
> For what was withdrawn and why, see [`../AUDIT.md`](../AUDIT.md). This file is kept
> because the retraction is only checkable against the plan it replaced.

---


**Project:** COMPUTATIONAL COUPLING  
**Lead Autonomous Research Engineer:** Antigravity  
**Version:** 1.0 (Milestone 0 Consensus)  
**Date:** September 17, 2026  

---

## 1. Primary Research Question (Extended Cross-Scale Formulation)

> **Can neural computational principles identified across biological systems remain meaningful across multiple scales of organization—from complete invertebrate nervous systems (*C. elegans*) and insect whole brains (*Drosophila*) to mammalian cortical circuits (*Mus musculus*) and human cortical micro-connectivity (H01) and whole-brain macro-connectivity (HCP)? Specifically: which structural motifs constitute functionally necessary "computational invariants", how far can these circuits be structurally remodeled under biological constraints while preserving dynamic computation, and can their abstractions produce event-driven computing architectures that achieve superior energy-delay efficiency?**

---

## 2. Biological Scale Ladder

We explicitly distinguish the different scales and modalities of biological neural organization:
1. **System A — *C. elegans*:** ~302 neurons, complete whole-animal nervous-system connectome (Cook et al. 2019, Witvliet et al. 2021).
2. **System B — *Drosophila melanogaster*:** ~139k neurons, whole adult brain connectome (FlyWire Consortium, Nature 2024).
3. **System C — *Mus musculus*:** Large cortical microcircuit (~1,000 proofread cells in Layer 2/3 of 1 mm³ volume), combining structural connectivity + in vivo calcium activity + visual stimuli (MICrONS Consortium, Nature 2025).
4. **System D (Micro) — *Homo sapiens* (H01):** Small cortical volume (~1 mm³ temporal cortex), resolving single neurons (~16k) and synapses (~150M), establishing human synaptic micro-connectivity and laminar E/I distributions (Shapson-Coe et al., Science 2024).
5. **System D (Macro) — *Homo sapiens* (HCP):** Whole-brain macro-scale connectivity, resolving regional tractography and functional co-activation across 180–1015 parcels (Van Essen et al. 2012, Glasser et al. 2016).

*Critical Scientific Qualification:* A complete human neuron-level connectome is **NOT AVAILABLE**. Whole-brain HCP data represents region-to-region tractography (voxel scale ~1.25–2 mm), while H01 represents a local 1 mm³ micro-fragment. We do NOT treat these as directly equivalent, but as distinct hierarchical tiers.

---

## 3. Definition: Cross-Scale Neural Computational Motif

A structural motif candidate must satisfy progressively rigorous, falsifiable criteria to qualify as a **Cross-Scale Neural Computational Motif**:

- **Level 1 (Structural Existence):** The motif exists structurally within the biological connectome graph.
- **Level 2 (Statistical Overrepresentation):** The motif occurs at a significantly higher frequency than in degree-preserving randomized null configuration models ($Z > 2.0$, $p < 0.01$).
- **Level 3 (Computational Contribution):** Embedding the motif within an event-driven dynamical system measurably improves performance on a dynamic signal-processing or learning task (e.g., coincidence detection, duration tuning, SNR).
- **Level 4 (Causal Necessity / Invariance):** Perturbing or lesioning the motif's constituent edges degrades dynamic task performance by $>50\%$, demonstrating that it is an invariant computational bottleneck rather than redundant scaffolding.
- **Level 5 (Architectural Transferability):** A computational abstraction of the motif remains functional and advantageous when deployed outside the original biological context in an event-driven artificial neural network.
- **Level 6 (Cross-Scale Generalization):** Structurally analogous or functionally homologous configurations are detected across multiple phylogenetic levels (e.g., conserved from *C. elegans* command circuits and fly optic lobes to mouse V1 and human H01/HCP).

---

## 4. Decomposition into Core Sub-Questions

### Sub-Question 1 (Causal Motif Functionality Across Scales):
*Do conserved motifs (Coherent/Incoherent Feed-Forward Loops, Reciprocal Inhibition, Disinhibitory Triads) execute distinct, quantifiable temporal operations across invertebrate (*C. elegans*, fly) and mammalian/human cortical microcircuits (mouse V1, human H01), and do they survive Level 1–6 validation?*

### Sub-Question 2 (The Structural Remodeling & Degeneracy Frontier):
*Given an empirical biological neural circuit $G_{\text{orig}}$ at each scale, how structurally divergent can a rewired circuit $G'$ become ($\max \Delta_{\text{struct}}(G_{\text{orig}}, G')$) under strict biological constraints (Dale's principle, neurotransmitter conservation, degree caps, delay bounds) while maintaining task performance $F_{\text{task}}(G') \ge (1 - \epsilon) F_{\text{task}}(G_{\text{orig}})$?*

### Sub-Question 3 (Event-Driven Architecture Transfer & Neuromorphic Advantage):
*Can the abstracted cross-scale computational invariants be synthesized into an event-driven computing architecture (CoupledEventNet / CEN) that demonstrates Pareto superiority (accuracy vs. SynOps / latency) over conventional dense SNNs, connectome reservoirs (`conn2res`), and Neural Circuit Policies (NCPs) on temporal event-stream benchmarks?*

---

## 5. Scope Boundaries & Forbidden Claims

1. **Forbidden Over-Claim:** "The complete human brain connectome is available." (Only regional macro-connectivity [HCP] and a 1 mm³ micro-fragment [H01] exist).
2. **Forbidden Over-Claim:** "Human brains work like fly brains" or "All brains use the exact same algorithm." (We test specific structural-dynamical invariants, acknowledging distinct ecological adaptations).
3. **Forbidden Over-Claim:** "Brain-inspired computers will replace binary digital computers." (All claims must be limited to measured metrics: SynOps, event sparsity, latency, and noise robustness on event-stream tasks).
