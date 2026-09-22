# NOVELTY REPORT — COMPUTATIONAL COUPLING

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
**Audit Date:** September 17, 2026  
**Status:** Milestone 0 Initial Audit Complete  

---

## 1. Executive Summary & Novelty Classification

We conducted a comprehensive, skeptical audit of the literature across computational neuroscience, network connectomics, machine learning, and neuromorphic engineering to determine whether the proposed research direction is genuinely novel.

| Dimension / Candidate Sub-Goal | Classification | Rationale & Status |
| :--- | :---: | :--- |
| **A. Cross-species static connectome motif extraction** | **RED** | **Essentially already done.** Lin et al. (*Nature* 2024) compared 3-node motif distributions, rich-club, and reciprocity across FlyWire (Drosophila), C. elegans, zebrafish, and mouse Pinky100. We must NOT claim novelty for extracting or comparing static graph motifs across these organisms. |
| **B. Drosophila whole-brain spiking neural simulation** | **RED** | **Essentially already done.** Shiu et al. (*Nature* 2024) already constructed a whole-brain Leaky Integrate-and-Fire (LIF) model with 139,255 neurons and ~50M synapses, matching biological sensorimotor pathways. We must NOT claim novelty for whole-brain Drosophila LIF simulation. |
| **C. C. elegans connectome-based AI architectures** | **RED** | **Essentially already done.** Hasani, Lechner et al. (*Nat Mach Intell* 2020, 2021) abstracted C. elegans 4-layer topology into Neural Circuit Policies (NCPs) for control and autonomous driving. |
| **D. Connectome-informed reservoir computing** | **RED** | **Essentially already done.** Suárez, Misic et al. (*Nat Commun* 2024) published `conn2res`, evaluating memory and cognitive tasks across empirical C. elegans, mouse, and human connectome reservoirs. |
| **E. Algorithmic synaptic rewiring in artificial neural nets** | **RED** | **Essentially already done.** Bellec et al. (DEEP R, *ICLR* 2018) and Chen et al. (Net2Net, *ICLR* 2016) already developed stochastic synaptic rewiring and function-preserving network morphisms for ANNs. |
| **F. Concept of neural degeneracy** | **RED** | **Essentially already done.** Marder, Prinz et al. (*Nat Neurosci* 2004, 2011) established that disparate circuit parameters and topologies produce identical target dynamics. |
| **G. Human whole-brain macro-scale graph analysis (HCP)** | **RED** | **Essentially already done.** Thousands of studies have analyzed degree distributions, rich-club, modularity, and tractography networks across HCP datasets. |
| **H. Connectome-constrained deep networks on sensory tasks** | **YELLOW** | Lappalainen et al. (*Nature* 2024) constrained recurrent networks with Drosophila optic lobe wiring to predict motion detection. Related, but their wiring was strictly static (non-remodeled) and executed synchronously via dense BPTT, not asynchronous event-driven compute. |
| **I. Event-driven functional abstraction of cross-species conserved motifs** | **GREEN** | **No close equivalent found.** Translating the conserved 3- and 4-node motif types (e.g. coherent/incoherent feed-forward loops, reciprocal inhibitory feedback, disinhibitory gating) identified across C. elegans, FlyWire, and MICrONS into modular *asynchronous event-driven computing primitives* (temporal coincidence detectors, homeostatic gain gates, transient delay lines) and benchmarking them against standard SNNs. |
| **J. Biologically constrained structural remodeling & the degeneracy frontier ($\Delta_{\text{struct}}$ vs. $F_{\text{task}}$)** | **GREEN** | **No close equivalent found.** Systematically remodeling empirical connectome subcircuits under strict biological constraints (Dale's E/I sign preservation, neurotransmitter identity, bounded delay, synaptic degree caps) to trace the multi-objective Pareto frontier: $\max \Delta_{\text{struct}}$ subject to preserving dynamic task performance and minimizing event/energy proxy. |
| **K. Human H01 micro-connectome cross-validation of computational invariants** | **GREEN** | **No close equivalent found.** Validating whether the non-rewirable computational invariants discovered in model organisms survive in the nanoscale H01 human cortical micro-connectome (Shapson-Coe et al., Science 2024). |

---

## 2. Closest Works & Precise Prior Art Analysis

### 1. Lin, Yang et al. (*Nature* 2024) — "Network statistics of the whole-brain connectome of Drosophila"
- **What it did:** Calculated degree distributions, rich-club curves, clustering coefficients, and 3-node motif counts for FlyWire v630, C. elegans (Cook 2019), zebrafish, and mouse Pinky100.
- **What it left unresolved:** It treated connectomes purely as static topological graphs. It did not simulate, test, or evaluate any computational or cognitive tasks, dynamic signal processing, or event-driven execution.
- **What our work does differently:** We evaluate dynamic *computation*—testing whether these topological structures execute specific temporal and event-driven functions, and whether they can be abstracted into machine learning architectures.

### 2. Shiu et al. (*Nature* 2024) — "A Drosophila computational brain model reveals sensorimotor processing"
- **What it did:** Implemented a whole-brain LIF model in Python/PyTorch for FlyWire v630 using predicted neurotransmitter signs (Ach, Glu, GABA) to replicate feeding and grooming reflexes.
- **What it left unresolved:** Kept the empirical connectome 100% rigid. Did not ask how the network could be remodeled, did not extract reusable computational modules, and did not evaluate artificial event-stream tasks or energy metrics.
- **What our work does differently:** We use Shiu's LIF model as an upstream verification baseline, but focus on *remodeling* subcircuits and extracting transferable event-driven compute blocks.

### 3. Shapson-Coe et al. (*Science* 2024) — "A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution" (H01)
- **What it did:** Reconstructed a 1 mm³ volume of human anterior temporal cortex at 4x4x33 nm resolution, identifying ~57k cells, ~150M synapses, axonal whorls, multi-synaptic connections (up to 50 synapses per pair), and laminar E/I distributions.
- **What it left unresolved:** It provided structural connectomics and morphological annotations without functional activity, sensory stimulus co-registration, or computational circuit remodeling.
- **What our work does differently:** We use H01 as a human micro-connectome validation tier to test whether the computational invariants discovered in model organisms persist in human cortical wiring.

### 4. Suárez, Misic et al. (*Nature Communications* 2024) — "Connectome-based reservoir computing with the conn2res toolbox"
- **What it did:** Evaluated empirical connectomes (C. elegans, mouse, human HCP) as fixed reservoirs for cognitive tasks (XOR, memory capacity, speech classification) using linear readout regression.
- **What it left unresolved:** Connectome topologies were treated as black-box reservoirs with fixed adjacency matrices; no structural remodeling, no evolutionary search, no event-driven asynchronous spike primitives, and no causal motif lesioning.
- **What our work does differently:** We isolate individual motifs, remodel connectivity under biological constraints, and translate them into event-driven spiking architectures rather than relying solely on reservoir readouts.

### 5. Lechner, Hasani et al. (*Nature Machine Intelligence* 2020, 2021) — "Neural Circuit Policies" (NCPs) & "Liquid Time-Constant Networks" (LTCs)
- **What it did:** Designed a 19-neuron recurrent continuous-time architecture inspired by the 4-layer C. elegans sensorimotor layout for autonomous control.
- **What it left unresolved:** Hand-crafted design; used continuous-time ODE solvers requiring dense integration; did not draw from empirical large-scale connectomes (Drosophila / Mouse / Human) and did not study structural remodeling.
- **What our work does differently:** We ground our architectures in empirical multi-species connectomes (C. elegans, Drosophila, Mouse, Human) and deploy event-driven spiking primitives rather than continuous ODEs.

### 6. Bellec et al. (*ICLR* 2018) — "Deep Rewiring: Training very sparse neural networks"
- **What it did:** Algorithmically rewired connections during backpropagation in ANNs using random walk / temperature-dependent sampling to maintain extreme parameter sparsity.
- **What it left unresolved:** Completely unconstrained by biology; connections were arbitrary and agnostic to Dale's principle, neurotransmitter signs, or biological motifs.
- **What our work does differently:** We formulate *biologically constrained* remodeling: preserving Dale's principle (E/I sign segregation), cell-type identities, and delay distributions, explicitly tracing the structural distance from biological truth.

---

## 3. Claims We MUST NOT Make (Forbidden Over-Claims)

1. **DO NOT CLAIM:** *"The complete human brain connectome is available."*  
   **Fact:** Only regional macroscale tractography (HCP) and a 1 mm³ micro-fragment (H01) exist. A complete neuron-level human connectome does not exist.
2. **DO NOT CLAIM:** *"We are the first to compare connectomes across C. elegans, Drosophila, and mouse."*  
   **Fact:** Lin et al. (*Nature* 2024) already published this cross-species comparison.
3. **DO NOT CLAIM:** *"We are the first to simulate the Drosophila FlyWire connectome as a spiking network."*  
   **Fact:** Shiu et al. (*Nature* 2024) already published a whole-brain LIF model.
4. **DO NOT CLAIM:** *"We discovered that biological connectomes have overrepresented feed-forward loops."*  
   **Fact:** Milo et al. (*Science* 2002) and Sporns & Kötter (*PLoS Biol* 2004) proved this over two decades ago.
5. **DO NOT CLAIM:** *"Human brains compute identically to flies or worms."*  
   **Fact:** Scientifically false; different organisms possess unique ecological specializations. We test for conserved computational *primitives*, not universal equivalence.
6. **DO NOT CLAIM:** *"We invented function-preserving rewiring or structural degeneracy."*  
   **Fact:** Marder & Prinz (*Nat Neurosci* 2004) established biological degeneracy; Net2Net (Chen 2016) established functional preservation in deep learning.
7. **DO NOT CLAIM:** *"Brains are fundamentally better than binary computers" or "We created a digital brain/AGI."*  
   **Fact:** Scientifically unjustified and irrelevant to the research questions.

---

## 4. The Defensible, High-Impact Research Question

### Primary Research Question:
> **Can neural computational principles identified across biological systems remain meaningful across multiple scales of organization—from invertebrate nervous systems (*C. elegans*) and insect whole brains (*Drosophila*) to mammalian cortical circuits (*Mus musculus*) and human cortical micro-connectivity (H01) and macro-connectivity (HCP)? Specifically: which structural motifs constitute functionally necessary "computational invariants", how far can these circuits be structurally remodeled under biological constraints while preserving dynamic computation, and can their abstractions produce event-driven computing architectures that achieve superior energy-delay efficiency?**

### Precise Novel Contributions:
1. **The Biological Degeneracy Frontier Across Scales ($\Delta_{\text{struct}}$ vs. $F_{\text{task}}$):**
   First quantitative mapping of how far empirical biological circuits (C. elegans, Drosophila subcircuits, Mouse V1, and human H01) can be structurally deformed under Dale's principle and cell-type constraints before dynamic event-driven computation collapses.
2. **Identification of Computational Invariants vs. Biological Scaffolding:**
   Causal lesioning and permutation experiments identifying which specific multi-node motifs are computationally non-negotiable (destroying temporal memory or coincidence detection when broken) versus which connections are redundant developmental scaffolding.
3. **Human Micro-Connectome Cross-Validation (H01):**
   First verification of whether computational invariant motifs identified in model organisms persist in nanoscale human cortical micro-connectivity.
4. **Cross-Scale Event-Driven Architecture (CoupledEventNet / CEN):**
   A modular, event-driven neural computing architecture composed of abstracted cross-scale motifs, benchmarked against conventional SNNs, LIF reservoirs, and NCPs for energy proxy (synaptic operations), spike sparsity, and temporal classification latency.
