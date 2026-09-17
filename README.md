# COMPUTATIONAL COUPLING

> **Computationally analyzing biological neural wiring across four phylogenetic levels (*C. elegans*, *Drosophila melanogaster*, *Mus musculus*, and *Homo sapiens*), identifying conserved computational structures, remodeling biological neural architectures under biophysical constraints, and synthesizing these principles into an ultra-low-energy event-driven architecture: CoupledEventNet (CEN).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Connectomes: Nematode | Fly | Mouse | Human](https://img.shields.io/badge/Connectomes-Worm%20%7C%20Fly%20%7C%20Mouse%20%7C%20Human-green.svg)](#the-four-biological-systems)

---

## 1. Project Overview & Research Highlights

**COMPUTATIONAL COUPLING** is an end-to-end computational neuroscience and neuromorphic AI research project. Rather than treating biological connectomes as static graphs or arbitrary reservoir matrices, we systematically isolate **non-rewirable computational invariants** from **degenerate developmental scaffolding**, proving that sparse biological motifs provide an actionable blueprint for ultra-low-power neuromorphic hardware.

### Key Scientific Discoveries
1. **Universal Cross-Scale Motif Conservation:**
   Using degree-preserving configuration null models respecting Dale's principle, we demonstrate that **Coherent and Incoherent Feedforward Loops (cFFL, iFFL)** and **Reciprocal Inhibitory Pairs (RIP)** are profoundly enriched ($Z > 2.0$ to $Z = +754.9$, $p < 0.0001$) across *C. elegans* (whole animal), *Drosophila* (Central Complex and Antennal Lobe), *Mus musculus* (V1 Layer 2/3), and *Homo sapiens* (HCP 1,015-region macroscale network).
2. **Biophysical Function of Validated Motifs:**
   - **cFFL:** Acts as a persistence detector, achieving 100% rejection of transient noise spikes while transmitting sustained signals.
   - **iFFL:** Generates adaptive pulses with complete steady-state suppression (**Adaptation Index = 1.00**).
   - **RIP:** Implements bistable winner-take-all categorical attractor switching.
3. **The Degeneracy Frontier:**
   Multi-objective evolutionary remodeling demonstrates that biological neural subcircuits can drift structurally by **$\Delta_{\text{struct}} = 0.177$** while maintaining **100% task competence ($F_{\text{task}} = 1.000$)** and reducing synaptic operations by **27.6%** (from 11,376 down to 8,230 SynOps).
4. **CoupledEventNet (CEN) Hardware Breakthrough:**
   Abstracting these invariant motifs into **CoupledEventNet (CEN)** yields 96.50% test accuracy on temporal pattern discrimination while requiring only **3.52 SynOps per inference sample (14.31 nJ energy proxy)**. This represents a **455-fold reduction in energy proxy** compared to standard dense recurrent SNNs (1,623.7 SynOps, 6,507.6 nJ) and an **11-fold reduction** compared to biological connectome reservoirs (`conn2res`, 38.97 SynOps, 159.2 nJ).

---

## 2. Comparative Neuromorphic Benchmark

| Model Architecture | Dale Compliant | Structural Recurrent Topology | Test Accuracy (%) | Noisy Accuracy (%) | SynOps / Sample | Energy Proxy (nJ) | Energy Advantage vs SNN |
|---|---|---|---|---|---|---|---|
| **CoupledEventNet (CEN)** | **Yes** | **Motif-Constrained (cFFL/iFFL/RIP)** | **96.50%** | **82.00%** | **3.52** | **14.31 nJ** | **455x lower energy** |
| **Dense SNN (Standard)** | No | Dense Recurrent (All-to-All) | 97.50% | 83.00% | 1,623.70 | 6,507.57 nJ | Baseline (1.0x) |
| **Connectome Reservoir (`conn2res`)** | Yes | Fixed *C. elegans* Connectome | 98.50% | 82.50% | 38.97 | 159.17 nJ | 41x lower energy |

---

## 3. Repository Architecture

```
computational-coupling/
├── README.md                           # Master project guide
├── LICENSE                             # MIT License
├── pyproject.toml                      # Package configuration
│
├── research/                           # Research specifications & audits
│   ├── prior_art_matrix.csv            # 18-column systematic prior art comparison
│   ├── novelty_report.md               # Scientific novelty & forbidden claims audit
│   ├── research_question.md            # Deconstructed research questions & 6-level framework
│   ├── hypotheses.md                   # Falsifiable hypotheses H1–H6
│   └── literature_review.md            # Comprehensive theoretical synthesis
│
├── src/                                # Core computational modules
│   ├── connectome/                     # Unified graph schema & organism loaders
│   │   ├── unified_schema.py           # ConnectomeGraph, NeuronMetadata, SynapseMetadata
│   │   ├── celegans_loader.py          # Cook et al. (2019) adult hermaphrodite loader
│   │   ├── drosophila_loader.py        # FlyWire v630 subcircuit loader (CX, AL)
│   │   ├── mouse_loader.py             # MICrONS Pinky100 V1 L2/3 loader
│   │   └── human_loader.py             # H01 microcircuit & HCP macroscale loader
│   ├── motifs/                         # Motif discovery & statistical significance
│   │   └── motif_engine.py             # Sign-attributed census & Dale-preserving null models
│   ├── simulation/                     # Biophysical simulation engine
│   │   └── lif_simulator.py            # Vectorized LIF with delays, Dale clamps, SynOps
│   ├── remodeling/                     # Biological remodeling & degeneracy frontier
│   │   └── pareto_remodeler.py         # Multi-objective Pareto evolutionary search & lesioning
│   ├── architecture/                   # Neuromorphic architectures
│   │   ├── coupled_event_net.py        # CoupledEventNet (CEN) PyTorch module
│   │   └── baselines.py                # Dense SNN and conn2res reservoir baselines
│   └── analysis/                       # Visualization & figures
│       └── generate_paper_figures.py   # Publication-quality figure generation
│
├── experiments/                        # Reproducible experimental pipelines
│   ├── cross_species/
│   │   ├── run_motif_analysis.py       # Phase 1: Cross-species motif significance census
│   │   ├── run_dynamic_motif_benchmark.py # Level 3: Dynamic biophysical motif benchmark
│   │   └── run_remodeling_experiment.py # Phase 2: Degeneracy frontier evolutionary search
│   └── architecture/
│       └── run_coupled_event_net_benchmark.py # Phase 3: CEN comparative benchmark
│
├── results/                            # Experimental outputs & figures
│   ├── motif_significance_cross_species.csv # Flat CSV of motif metrics
│   ├── motif_significance_cross_species.json# Structured JSON report
│   ├── phase2_remodeling_results.json  # Pareto frontier candidates & lesion data
│   ├── cen_benchmark_table.csv         # Neuromorphic benchmark comparison table
│   └── figures/                        # High-resolution 300 DPI publication figures
│       ├── figure1_cross_species_motifs.png
│       ├── figure2_degeneracy_frontier.png
│       ├── figure3_dynamic_motifs.png
│       └── figure4_cen_benchmark.png
│
└── paper/                              # Publication manuscript
    ├── computational_coupling_manuscript.md # Complete research paper manuscript
    ├── figures/                        # Manuscript figures
    └── tables/                         # Formatted research tables
```

---

## 4. Quickstart & Verification

### Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r <(python3 -c "
import tomli
with open('pyproject.toml', 'rb') as f:
    data = tomli.load(f)
print('\n'.join(data['project']['dependencies']))
")
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run Experiments
```bash
# 1. Phase 1: Cross-species motif discovery & significance census
python3 experiments/cross_species/run_motif_analysis.py

# 2. Level 3: Dynamic biophysical motif operations (cFFL, iFFL, RIP)
python3 experiments/cross_species/run_dynamic_motif_benchmark.py

# 3. Phase 2: Degeneracy frontier & causal edge lesioning
python3 experiments/cross_species/run_remodeling_experiment.py

# 4. Phase 3: CoupledEventNet (CEN) comparative benchmark
python3 experiments/architecture/run_coupled_event_net_benchmark.py

# 5. Generate publication figures
python3 src/analysis/generate_paper_figures.py
```

---

## 5. Citation

```bibtex
@article{computational_coupling_2026,
  title={Computational Coupling: Conserved Cross-Scale Neural Motifs and Degeneracy Frontiers in Biological Wiring Inform Low-Energy Event-Driven Computing},
  author={COMPUTATIONAL COUPLING Research Initiative},
  journal={Working Paper / Technical Report},
  year={2026}
}
```
