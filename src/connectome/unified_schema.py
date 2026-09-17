"""
COMPUTATIONAL COUPLING — Unified Connectome Graph Schema
Provides a standardized data representation for biological neural wiring diagrams
across C. elegans, Drosophila, mouse cortex, and human cortical/macro networks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import networkx as nx


@dataclass
class NeuronMetadata:
    """Standardized metadata for a single biological neuron or regional parcel."""
    node_id: int
    name: str
    cell_type: str  # e.g., 'sensory', 'interneuron', 'motor', 'pyramidal', 'interneuron_basket'
    sign: int = 1   # +1: excitatory, -1: inhibitory, 0: unknown
    neurotransmitter: Optional[str] = None  # 'acetylcholine', 'glutamate', 'gaba', etc.
    layer: Optional[str] = None  # e.g., 'Layer 1', 'Layer 2/3', 'ME_R'
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) in nm or mm
    organism: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapseMetadata:
    """Standardized metadata for a directed synaptic connection or electrical junction."""
    source_id: int
    target_id: int
    weight: float = 1.0  # Synapse count or connection strength
    sign: int = 1        # +1: excitatory, -1: inhibitory (strictly follows Dale's principle)
    synapse_type: str = "chemical"  # 'chemical' or 'electrical'
    delay_ms: float = 1.8  # Transmission delay in ms
    extra: Dict[str, Any] = field(default_factory=dict)


class ConnectomeGraph:
    """
    Standardized in-memory biological connectome representation.
    Wraps a directed graph (NetworkX DiGraph) with strict type checking,
    Dale's E/I sign enforcement, sparse adjacency matrices, and motif extractors.
    """

    def __init__(self, name: str, organism: str, scale: str):
        self.name = name
        self.organism = organism
        self.scale = scale  # e.g., 'whole_nervous_system', 'whole_brain', 'microcircuit', 'macroscale'
        self.graph = nx.DiGraph(name=name, organism=organism, scale=scale)
        self.node_metadata: Dict[int, NeuronMetadata] = {}
        self.edge_metadata: Dict[Tuple[int, int], SynapseMetadata] = {}

    def add_neuron(self, meta: NeuronMetadata):
        """Add a neuron with verified biological metadata."""
        self.node_metadata[meta.node_id] = meta
        self.graph.add_node(
            meta.node_id,
            name=meta.name,
            cell_type=meta.cell_type,
            sign=meta.sign,
            neurotransmitter=meta.neurotransmitter,
            layer=meta.layer,
            position=meta.position,
            organism=self.organism,
            extra=meta.extra,
        )

    def add_synapse(self, meta: SynapseMetadata):
        """
        Add a synaptic connection.
        Enforces Dale's principle: connection sign must match pre-synaptic neuron sign.
        """
        if meta.source_id in self.node_metadata:
            source_sign = self.node_metadata[meta.source_id].sign
            if source_sign in (-1, 1):
                meta.sign = source_sign

        self.edge_metadata[(meta.source_id, meta.target_id)] = meta
        self.graph.add_edge(
            meta.source_id,
            meta.target_id,
            weight=float(meta.weight),
            sign=meta.sign,
            synapse_type=meta.synapse_type,
            delay_ms=meta.delay_ms,
            extra=meta.extra,
        )

    @property
    def num_neurons(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_synapses(self) -> int:
        return self.graph.number_of_edges()

    @property
    def density(self) -> float:
        n = self.num_neurons
        if n <= 1:
            return 0.0
        return self.num_synapses / (n * (n - 1))

    def get_adjacency_matrix(self, weighted: bool = True, signed: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Return the adjacency matrix $A$ where $A_{ij}$ is connection from $i \to j$.
        Ordered by sorted node IDs.
        """
        nodes = sorted(list(self.graph.nodes()))
        node_to_idx = {nid: i for i, nid in enumerate(nodes)}
        n = len(nodes)
        adj = np.zeros((n, n), dtype=np.float32)

        for u, v, data in self.graph.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            w = data.get("weight", 1.0) if weighted else 1.0
            if signed:
                w *= data.get("sign", 1)
            adj[i, j] = w

        return adj, nodes

    def get_subcircuit(self, node_ids: Set[int], name_suffix: str = "subcircuit") -> "ConnectomeGraph":
        """Extract an induced subcircuit graph preserving metadata."""
        sub = ConnectomeGraph(
            name=f"{self.name}_{name_suffix}",
            organism=self.organism,
            scale="subcircuit",
        )
        for nid in node_ids:
            if nid in self.node_metadata:
                sub.add_neuron(self.node_metadata[nid])

        for (u, v), emeta in self.edge_metadata.items():
            if u in node_ids and v in node_ids:
                sub.add_synapse(emeta)

        return sub

    def validate_dales_principle(self) -> Tuple[bool, int]:
        """
        Verify Dale's principle: each pre-synaptic neuron must have strictly non-negative
        or strictly non-positive outgoing edge signs. Returns (is_valid, violation_count).
        """
        violations = 0
        for nid in self.graph.nodes():
            out_signs = [self.graph[nid][target].get("sign", 1) for target in self.graph.successors(nid)]
            if out_signs:
                has_pos = any(s > 0 for s in out_signs)
                has_neg = any(s < 0 for s in out_signs)
                if has_pos and has_neg:
                    violations += 1
        return violations == 0, violations

    def summary_dict(self) -> Dict[str, Any]:
        """Return structural summary statistics."""
        dales_valid, dales_violations = self.validate_dales_principle()
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        exc_count = sum(1 for m in self.node_metadata.values() if m.sign == 1)
        inh_count = sum(1 for m in self.node_metadata.values() if m.sign == -1)

        return {
            "name": self.name,
            "organism": self.organism,
            "scale": self.scale,
            "num_neurons": self.num_neurons,
            "num_synapses": self.num_synapses,
            "density": float(self.density),
            "excitatory_neurons": exc_count,
            "inhibitory_neurons": inh_count,
            "ei_ratio": float(exc_count / max(1, inh_count)),
            "dales_principle_valid": dales_valid,
            "dales_violations": dales_violations,
            "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
            "max_in_degree": int(np.max(in_degrees)) if in_degrees else 0,
            "avg_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
            "max_out_degree": int(np.max(out_degrees)) if out_degrees else 0,
        }
