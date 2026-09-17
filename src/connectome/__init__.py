"""
COMPUTATIONAL COUPLING — Connectome Ingestion and Standardization Package
Exposes unified graph data structures and organism loaders for:
  - C. elegans (Cook et al. Nature 2019)
  - Drosophila melanogaster (FlyWire v630 / Nature 2024)
  - Mus musculus (MICrONS Pinky100 V1 L2/3 / Nature 2025)
  - Homo sapiens (H01 Science 2024 microcircuit + HCP macroscale tractography)
"""

from .unified_schema import (
    ConnectomeGraph,
    NeuronMetadata,
    SynapseMetadata,
)
from .celegans_loader import load_celegans_connectome
from .drosophila_loader import load_drosophila_subcircuit
from .mouse_loader import load_mouse_v1_connectome
from .human_loader import load_human_h01_microcircuit, load_human_hcp_macroscale

__all__ = [
    "ConnectomeGraph",
    "NeuronMetadata",
    "SynapseMetadata",
    "load_celegans_connectome",
    "load_drosophila_subcircuit",
    "load_mouse_v1_connectome",
    "load_human_h01_microcircuit",
    "load_human_hcp_macroscale",
]
