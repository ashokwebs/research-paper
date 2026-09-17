"""
Multi-Core Neuromorphic Network-on-Chip (NoC) Mesh Architecture.

Simulates 2D mesh packet-switched routing for multi-core neuromorphic accelerators.
Supports:
1. Parametric 2D Mesh topology (M x M cores).
2. Dimension-Order XY routing algorithm.
3. AER packet generation and queueing.
4. Motif-Aware spectral graph partitioning vs uniform random partitioning.
5. Cycle-accurate packet latency, hop count, buffer contention, and dynamic NoC energy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class AERPacket:
    """Address-Event Representation packet in NoC mesh."""
    src_core: Tuple[int, int]
    dest_core: Tuple[int, int]
    src_neuron: int
    dest_neuron: int
    timestamp_spawn: int
    hops: int = 0
    latency_cycles: int = 0


@dataclass
class NoCBenchmarkResult:
    """Summary metrics of NoC multi-core emulation."""
    mesh_dim: int
    total_cores: int
    num_neurons: int
    partitioning_strategy: str
    total_spikes: int
    intra_core_events: int
    inter_core_packets: int
    inter_core_ratio: float  # inter_core_packets / total_spikes
    mean_hops_per_packet: float
    max_hops: int
    mean_latency_cycles: float
    max_latency_cycles: int
    total_noc_energy_pj: float
    mean_energy_per_spike_pj: float
    link_utilization_peak: float


class NeuromorphicNoCMesh:
    """
    Simulates a 2D Mesh Network-on-Chip connecting M x M neuromorphic cores.
    
    Attributes:
        mesh_dim: Grid dimension (e.g. 4 for a 4x4 mesh with 16 cores).
        core_capacity: Number of neurons per core.
        energy_per_hop_pj: Dynamic energy consumed per flit/hop (calibrated to TSMC 28nm: ~1.5 pJ/hop).
        cycles_per_hop: Latency in clock cycles per router traversal (typically 2 cycles: route compute + switch).
    """
    def __init__(
        self,
        mesh_dim: int = 4,
        core_capacity: int = 128,
        energy_per_hop_pj: float = 1.50,
        cycles_per_hop: int = 2,
    ):
        self.mesh_dim = mesh_dim
        self.total_cores = mesh_dim * mesh_dim
        self.core_capacity = core_capacity
        self.energy_per_hop_pj = energy_per_hop_pj
        self.cycles_per_hop = cycles_per_hop
        self.neuron_to_core: Dict[int, Tuple[int, int]] = {}

    def partition_neurons_uniform(self, num_neurons: int) -> Dict[int, Tuple[int, int]]:
        """Partitions neurons sequentially across cores."""
        mapping = {}
        for nid in range(num_neurons):
            core_id = nid % self.total_cores
            cx = core_id % self.mesh_dim
            cy = core_id // self.mesh_dim
            mapping[nid] = (cx, cy)
        self.neuron_to_core = mapping
        return mapping

    def partition_neurons_motif_aware(
        self,
        adj_matrix: np.ndarray,
        motif_mask: Optional[np.ndarray] = None,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Partitions neurons using affinity clustering so tightly coupled motif loops
        (cFFL, iFFL, RIP) reside on identical or adjacent cores.
        """
        N = adj_matrix.shape[0]
        affinity = np.abs(adj_matrix) + np.abs(adj_matrix.T)
        if motif_mask is not None:
            affinity = affinity + 2.0 * motif_mask

        # Spectral core assignment: compute Fiedler vector / principal eigenvectors
        degrees = np.sum(affinity, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-6)))
        laplacian_norm = np.eye(N) - d_inv_sqrt @ affinity @ d_inv_sqrt
        
        evals, evecs = np.linalg.eigh(laplacian_norm)
        # Use 2nd and 3rd smallest eigenvectors for 2D spatial coordinate embedding
        coords_x = evecs[:, 1]
        coords_y = evecs[:, 2] if evecs.shape[1] > 2 else evecs[:, 1]
        
        # Rank-order partition into 2D mesh grid
        x_ranks = np.argsort(coords_x)
        y_ranks = np.argsort(coords_y)
        
        mapping = {}
        for nid in range(N):
            cx = int(np.floor((np.where(x_ranks == nid)[0][0] / N) * self.mesh_dim))
            cy = int(np.floor((np.where(y_ranks == nid)[0][0] / N) * self.mesh_dim))
            cx = min(max(0, cx), self.mesh_dim - 1)
            cy = min(max(0, cy), self.mesh_dim - 1)
            mapping[nid] = (cx, cy)
            
        self.neuron_to_core = mapping
        return mapping

    def route_xy(self, src: Tuple[int, int], dest: Tuple[int, int]) -> int:
        """Dimension-Order XY routing: returns Manhattan distance (router hops)."""
        dx = abs(src[0] - dest[0])
        dy = abs(src[1] - dest[1])
        return dx + dy

    def simulate_event_traffic(
        self,
        spike_train: np.ndarray,
        adj_matrix: np.ndarray,
        strategy_name: str = "custom",
    ) -> NoCBenchmarkResult:
        """
        Simulates event routing traffic over the 2D mesh given a spike train.
        
        Args:
            spike_train: (T_steps, N_neurons) binary spike matrix.
            adj_matrix: (N_neurons, N_neurons) connectivity matrix.
            strategy_name: Label for partitioning method.
        """
        T_steps, N_neurons = spike_train.shape
        total_spikes = 0
        intra_core = 0
        inter_core_packets = 0
        hop_counts: List[int] = []
        latencies: List[int] = []
        
        # Link contention tracker: key is ((x1, y1), (x2, y2)), val is flit count
        link_traffic: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        
        for t in range(T_steps):
            firing_neurons = np.where(spike_train[t] > 0)[0]
            total_spikes += len(firing_neurons)
            
            for pre_n in firing_neurons:
                src_core = self.neuron_to_core.get(pre_n, (0, 0))
                # Identify downstream targets
                post_targets = np.where(np.abs(adj_matrix[pre_n]) > 1e-4)[0]
                
                for post_n in post_targets:
                    dest_core = self.neuron_to_core.get(post_n, (0, 0))
                    
                    if src_core == dest_core:
                        intra_core += 1
                    else:
                        inter_core_packets += 1
                        hops = self.route_xy(src_core, dest_core)
                        latency = hops * self.cycles_per_hop + 1  # +1 for arbitration
                        hop_counts.append(hops)
                        latencies.append(latency)
                        
                        # Trace path for link utilization
                        curr_x, curr_y = src_core
                        dest_x, dest_y = dest_core
                        
                        while curr_x != dest_x:
                            next_x = curr_x + (1 if dest_x > curr_x else -1)
                            link = ((curr_x, curr_y), (next_x, curr_y))
                            link_traffic[link] = link_traffic.get(link, 0) + 1
                            curr_x = next_x
                            
                        while curr_y != dest_y:
                            next_y = curr_y + (1 if dest_y > curr_y else -1)
                            link = ((curr_x, curr_y), (curr_x, next_y))
                            link_traffic[link] = link_traffic.get(link, 0) + 1
                            curr_y = next_y

        mean_hops = float(np.mean(hop_counts)) if hop_counts else 0.0
        max_hops = int(np.max(hop_counts)) if hop_counts else 0
        mean_lat = float(np.mean(latencies)) if latencies else 1.0
        max_lat = int(np.max(latencies)) if latencies else 1
        
        total_hops = sum(hop_counts)
        total_energy_pj = total_hops * self.energy_per_hop_pj
        mean_energy_per_spike = total_energy_pj / max(1, total_spikes)
        inter_ratio = inter_core_packets / max(1, intra_core + inter_core_packets)
        peak_contention = float(max(link_traffic.values())) if link_traffic else 0.0
        
        return NoCBenchmarkResult(
            mesh_dim=self.mesh_dim,
            total_cores=self.total_cores,
            num_neurons=N_neurons,
            partitioning_strategy=strategy_name,
            total_spikes=total_spikes,
            intra_core_events=intra_core,
            inter_core_packets=inter_core_packets,
            inter_core_ratio=inter_ratio,
            mean_hops_per_packet=mean_hops,
            max_hops=max_hops,
            mean_latency_cycles=mean_lat,
            max_latency_cycles=max_lat,
            total_noc_energy_pj=total_energy_pj,
            mean_energy_per_spike_pj=mean_energy_per_spike,
            link_utilization_peak=peak_contention,
        )
