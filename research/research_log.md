# COMPUTATIONAL COUPLING — Autonomous Research Log

**Project Working Name:** COMPUTATIONAL COUPLING  
**Lead Autonomous Research Engineer:** Antigravity (Advanced Agentic Coding / DeepMind Team)  
**Date Initialized:** September 17, 2026  
**Hardware Profile:** 13th Gen Intel Core i7-13620H (16 vCPUs), 16 GB RAM (7.9 GB free), NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, CUDA 12.4), 268 GB Disk Available.

---

## Log Entry 001: Initial Setup & Environment Verification
- Initialized workspace structure at `/home/charizard/brain/computational-coupling/`.
- Created required top-level and subdirectories:
  `research/`, `sources/` (`papers/`, `bib/`, `datasets/`), `upstream/` (`openworm/`, `flywire/`, `mouse/`, `other/`), `data/` (`raw/`, `processed/`, `metadata/`), `src/` (`connectome/`, `analysis/`, `motifs/`, `remodeling/`, `simulation/`, `architecture/`, `benchmarks/`), `experiments/` (`celegans/`, `drosophila/`, `mouse/`, `cross_species/`, `architecture/`), `baselines/`, `results/` (`raw/`, `processed/`, `figures/`), `paper/` (`figures/`, `tables/`, `supplementary/`).
- System verification:
  - Python 3.13.5 with NumPy 2.2.4, Pandas 3.0.1, SciPy 1.15.3.
  - Dedicated virtual environment planned for dependencies: `networkx`, `torch` (CUDA 12.4), `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`, `h5py`.

---

## Log Entry 002: Authoritative Repository Acquisition
Cloned authoritative repositories into `upstream/` cleanly with `--depth 1`, preserving original commits and read-only upstream integrity:

1. **C. elegans (System A):**
   - `upstream/openworm/ConnectomeToolbox`: Commit `b9c0b4a7bc2ccf47d3ce7aac624e1b3e2ea86254`
     - Organization: OpenWorm
     - Purpose: Unified Python interface to published C. elegans connectome datasets (Cook et al. 2019, Witvliet et al. 2021, White et al. 1986, Ripoll-Sánchez et al. 2023).
     - Data assets present: `SI 5 Connectome adjacency matrices, corrected July 2020.xlsx`, `herm_full_edgelist.csv`, `all_cell_info.csv`, `witvliet_2020_*.xlsx`, neuropeptide and monoamine connectomes.
   - `upstream/openworm/CElegansNeuroML`: Commit `b36380a36d2a6dda0f03c946c433524b25ea2268`
     - Organization: OpenWorm
     - Purpose: NeuroML definitions and morphological reconstructions of individual C. elegans neurons.

2. **Drosophila melanogaster (System B):**
   - `upstream/flywire/FlyConnectome`: Commit `9b1790b5553082e1f57921790564c7aae7c9c99a`
     - Organization: Seung Lab / Princeton
     - Purpose: Core access libraries and schemas for the FlyWire whole-brain connectome.
   - `upstream/flywire/flywire_annotations`: Commit `8587524c1748ce5ef2080822a2fc890fc03bf597`
     - Organization: FlyConnectome Consortium / Schlegel et al. 2024
     - Purpose: Cell typing, hierarchical annotations, neurotransmitter predictions (GABA, acetylcholine, glutamate, dopamine, serotonin, octopamine).
   - `upstream/flywire/flywire-network-analysis`: Commit `ee4944c03429edf857a6b10bcf98bea7dcef35b8`
     - Organization: Murthy Lab / Seung Lab (Lin, Yang et al. Nature 2024)
     - Purpose: Network statistics, 3-node motif analyses, rich-club, degree percolation, cross-animal comparison scripts.
     - Data assets present: `v630_data/connections.csv.gz` (26 MB compressed), `classification.csv.gz`, `neurons.csv.gz`, `neuropil_synapse_table.csv.gz`.
   - `upstream/flywire/Drosophila_brain_model`: Commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
     - Organization: Philip K. Shiu / Murthy & Seung Labs / Scott Lab (Nature 2024)
     - Purpose: Leaky integrate-and-fire (LIF) whole-brain simulation pipeline with verified biophysical parameters ($v_0 = -52\text{ mV}$, $v_{\text{th}} = -45\text{ mV}$, $\tau_{\text{mbr}} = 20\text{ ms}$, $\tau_{\text{syn}} = 5\text{ ms}$, $t_{\text{dly}} = 1.8\text{ ms}$).
     - Data assets present: `2023_03_23_connectivity_630_final.parquet` (86 MB), `Connectivity_783.parquet` (100 MB).

3. **Mouse Cortical Circuit (System C - MICrONS / Allen Institute):**
   - `upstream/mouse/MicronsBinder`: Commit `3f53e35e2bfd3063469dcbab0e1dedce5a82e3ca`
     - Organization: Allen Institute for Brain Science
     - Purpose: Multi-modal connectomics tutorials and access pipelines for MICrONS Phase 1 (Pinky100 Layer 2/3) and Phase 3 (1 mm³ cortical volume).
     - Data links: Zenodo records `3710459` and `6363348` containing proofread synaptic connectivity, soma valences, functional calcium imaging responses, and motif analysis data.
   - `upstream/mouse/CAVEclient`: Commit `c57c15f55b6fe9c6dff5e8c6199df57b77b59c2d`
     - Organization: CAVEconnectome / Seung Lab
     - Purpose: Programmatic client for CAVE (Connectome Annotation Versioning Engine) live query API.

4. **Relevant Computational Baselines:**
   - `upstream/other/conn2res`: Commit `0304677...`
     - Organization: Network Neuroscience Lab (Bratislav Misic / Laura Suárez, Nature Communications 2024)
     - Purpose: Connectome-informed reservoir computing toolbox.
   - `upstream/other/ncps`: Commit `d90cf1f...`
     - Organization: Mathias Lechner / Ramin Hasani (Nature Machine Intelligence 2020)
     - Purpose: Neural Circuit Policies (NCPs) derived from C. elegans for continuous-time RNN control.

---

## Log Entry 003: Systematic Prior Art & Literature Landscape Analysis
Extensive search conducted across Nature, Science, Cell, PNAS, bioRxiv, arXiv, IEEE, ACM, OpenAlex, and GitHub. Key prior-art findings:

- **Finding 1 (Cross-Species Connectome Motifs):**
  Lin et al. (Nature 2024) already performed static graph motif counting (3-node subgraphs: FFL, RFB, mutual inhibition) comparing FlyWire (Drosophila), C. elegans (Cook et al.), larval zebrafish, and mouse Pinky100.
  *Implication:* We cannot claim novelty for simply counting or comparing graph motifs across these organisms.
  *Open Gap:* Lin et al. did NOT evaluate whether these motifs perform distinct dynamic computations in event-driven systems or whether they enhance computational efficiency.

- **Finding 2 (Drosophila Whole-Brain Simulation):**
  Shiu et al. (Nature 2024) built a full-brain Leaky Integrate-and-Fire (LIF) model of all 139k neurons and 50M synapses in FlyWire v630, verifying sensorimotor paths (antennal grooming, proboscis extension).
  Lappalainen et al. (Nature 2024) trained connectome-constrained recurrent networks of the visual system to predict neural activity and motion detection.
  *Implication:* We cannot claim to be the first to build a spiking/LIF simulation of the fly connectome or to train connectome-constrained networks on visual tasks.

- **Finding 3 (Connectome-based AI / Reservoir Computing):**
  Hasani & Lechner (2020, 2021) abstracted C. elegans 4-layer topology into Neural Circuit Policies (NCPs) for control.
  Suárez, Misic et al. (Nature Communications 2024) created `conn2res` for reservoir computing on human and mouse connectomes.
  *Implication:* We cannot claim to be the first to map biological connectomes into reservoirs or RNN policies.

- **Finding 4 (The Unexplored Frontier — Computational Coupling & Remodeling):**
  Existing work treats connectomes as either:
  1. Static descriptive graphs (Lin et al. 2024).
  2. Fixed anatomical constraints where only weights are optimized (Lappalainen et al. 2024, Shiu et al. 2024, Hasani et al. 2020).
  3. Reservoir topologies without biological rewiring rules (Suárez et al. 2024).
  
  **No prior work has performed biologically constrained structural remodeling on empirical cross-species connectomes to trace the Pareto frontier between structural novelty ($\Delta_{\text{struct}}$) and behavioral preservation ($F_{\text{task}}$) in event-driven spiking architectures.**
  Specifically:
  - How structurally divergent can a biological neural circuit become while preserving its dynamic temporal computation under strict biological constraints (Dale's E/I sign conservation, bounded delay, synaptic sparsity)?
  - Which cross-species conserved motifs act as "computational invariants" (non-rewirable functional bottlenecks) versus "redundant scaffolding" (rewirable without loss)?
  - Does an event-driven architecture synthesized from these invariant motifs provide measurable improvements in energy proxy (synaptic operations per inference), temporal latency, and noise robustness over standard SNN/LIF and reservoir baselines?
