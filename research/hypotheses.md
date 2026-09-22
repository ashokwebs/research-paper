# COMPUTATIONAL COUPLING — Experimental Hypotheses

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
**Version:** 1.0  
**Date:** September 17, 2026  

---

## Hypothesis 1: Functional Superiority of Conserved Motifs
*Conserved structural motifs (specifically Type-1 Coherent Feed-Forward Loops and Reciprocal Inhibitory Feedback) identified across C. elegans, Drosophila, and mouse cortical connectomes exhibit statistically significant advantages in temporal coincidence detection, dynamic range, and persistence over degree-matched random configuration null models under event-driven spiking dynamics.*
- **Null Hypothesis ($H_{1,0}$):** The dynamical performance (signal-to-noise ratio, temporal memory capacity, spike efficiency) of empirical motifs is statistically indistinguishable from degree-preserving randomized graphs ($p \ge 0.05$, two-tailed permutation test).
- **Falsification Criterion:** If randomized null graphs achieve equal or superior dynamical memory capacity and coincidence selectivity, $H_1$ is rejected.

---

## Hypothesis 2: Structural Degeneracy & The Remodeling Frontier
*Biological connectome subcircuits exhibit a structural degeneracy regime under biological constraints: an empirical circuit $G_{\text{orig}}$ can be topologically rewired by a graph edit distance $\Delta_{\text{struct}} \ge 0.35$ (35% synaptic turnover) while preserving $\ge 95\%$ of baseline task performance ($F_{\text{task}}(G') \ge 0.95 F_{\text{task}}(G_{\text{orig}})$), provided Dale's E/I sign segregation and core feedback invariants are preserved.*
- **Null Hypothesis ($H_{2,0}$):** Any topological remodeling beyond $\Delta_{\text{struct}} > 0.10$ results in catastrophic degradation of task accuracy ($F_{\text{task}} < 0.80 F_{\text{task}}(G_{\text{orig}})$).
- **Falsification Criterion:** If the multi-objective Pareto front collapses immediately upon structural modification under biological constraints, $H_2$ is rejected.

---

## Hypothesis 3: Bimodal Causal Invariance vs. Developmental Scaffolding
*Synaptic connections in empirical connectome subcircuits do not contribute uniformly to dynamic task computation; rather, their causal importance distribution is bimodal: a small subset of connections embedded in conserved motifs act as "computational invariants" (lesioning causes $>50\%$ performance collapse), whereas the majority of connections represent "developmental scaffolding" that can be pruned or rewired with $<2\%$ performance impact.*
- **Null Hypothesis ($H_{3,0}$):** The distribution of causal lesion impacts across all synapses is unimodal or uniform, showing no segregation between motif-embedded connections and peripheral connections.
- **Falsification Criterion:** If lesioning motif connections versus non-motif connections shows no statistically significant difference in performance collapse, $H_3$ is rejected.

---

## Hypothesis 4: Neuromorphic Event-Driven Efficiency of CoupledEventNet
*An event-driven neural architecture synthesized from cross-species computational invariants (CoupledEventNet / CEN) will achieve at least a $3\times$ reduction in estimated Synaptic Operations (SynOps) and event count compared to conventional dense SNN baselines (standard LIF networks) and connectome reservoirs (`conn2res`) at equivalent or superior temporal classification accuracy on sequential benchmarks.*
- **Null Hypothesis ($H_{4,0}$):** CoupledEventNet requires equal or greater SynOps / spike events, or suffers a statistically significant decrease in accuracy ($>5\%$ drop) relative to standard SNN/LIF baselines.
- **Falsification Criterion:** If CEN does not demonstrate a statistically significant reduction in SynOps and event count at matched accuracy ($p < 0.01$), $H_4$ is rejected.

---

## Hypothesis 5: Human Micro-Connectome Motif Invariance (H01 Cross-Validation)
*The causal computational motifs validated in C. elegans, Drosophila, and mouse cortex (specifically Feed-Forward Coincidence Loops and Reciprocal Inhibitory Feedback) will exhibit statistically significant overrepresentation in the H01 human cortical micro-connectome relative to degree-preserving randomized null models ($Z > 2.5$, $p < 0.01$), confirming their role as cross-scale biological invariants.*
- **Null Hypothesis ($H_{5,0}$):** Motif frequencies in H01 human temporal cortex do not differ significantly from randomized configuration nulls, or display a fundamentally divergent motif significance profile unrelated to non-human models.
- **Falsification Criterion:** If empirical H01 subgraphs show no statistical enrichment of FFL or reciprocal inhibitory motifs ($Z \le 1.0$), $H_5$ is rejected.

---

## Hypothesis 6: Scale Separation (Synaptic Micro-Motifs vs. Macro-Scale Routing)
*Network motifs that govern event-driven computation at the synaptic micro-scale (H01 / MICrONS / Drosophila) do NOT scale self-similarly to the whole-brain macro-scale (HCP tractography); rather, biological nervous systems exhibit scale separation where microcircuits optimize temporal coincidence and gain control (local FFL / reciprocal inhibition), whereas macro-networks optimize global modular routing and communication latency (rich-club hubs / small-world path length).*
- **Null Hypothesis ($H_{6,0}$):** The network motif profile and functional dynamics of whole-brain HCP diffusion tractography are identical to nanoscale synaptic micro-connectomes.
- **Falsification Criterion:** If 3-node motif distributions and dynamical memory profiles are isomorphic between H01 and HCP, $H_6$ is rejected.

