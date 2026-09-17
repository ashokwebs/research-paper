"""
Unit tests for Neuromorphic Multi-Core Network-on-Chip (NoC) Mesh Architecture.
"""

import pytest
import numpy as np
from hardware.noc.noc_mesh import NeuromorphicNoCMesh, AERPacket, NoCBenchmarkResult


def test_dimension_order_xy_routing():
    """Verify Dimension-Order XY routing Manhattan distance calculation."""
    noc = NeuromorphicNoCMesh(mesh_dim=4)
    # Core (0, 0) to Core (3, 3) -> 3 + 3 = 6 hops
    assert noc.route_xy((0, 0), (3, 3)) == 6
    # Same core -> 0 hops
    assert noc.route_xy((1, 2), (1, 2)) == 0
    # Adjacent in X -> 1 hop
    assert noc.route_xy((1, 2), (2, 2)) == 1
    # Adjacent in Y -> 1 hop
    assert noc.route_xy((1, 2), (1, 3)) == 1


def test_mesh_partitioning():
    """Verify uniform and motif-aware spectral core assignments."""
    N = 64
    mesh_dim = 4
    noc = NeuromorphicNoCMesh(mesh_dim=mesh_dim)
    
    # Uniform partitioning
    uniform_map = noc.partition_neurons_uniform(N)
    assert len(uniform_map) == N
    for nid, (cx, cy) in uniform_map.items():
        assert 0 <= cx < mesh_dim
        assert 0 <= cy < mesh_dim
        
    # Spectral motif-aware partitioning
    np.random.seed(42)
    adj = (np.random.rand(N, N) < 0.1).astype(float)
    motif_map = noc.partition_neurons_motif_aware(adj)
    assert len(motif_map) == N
    for nid, (cx, cy) in motif_map.items():
        assert 0 <= cx < mesh_dim
        assert 0 <= cy < mesh_dim


def test_noc_traffic_simulation():
    """Verify packet traffic, latency, and energy calculation."""
    N = 32
    T = 100
    mesh_dim = 2
    noc = NeuromorphicNoCMesh(mesh_dim=mesh_dim, energy_per_hop_pj=1.5, cycles_per_hop=2)
    noc.partition_neurons_uniform(N)
    
    np.random.seed(123)
    spike_train = (np.random.rand(T, N) < 0.05).astype(int)
    adj_matrix = (np.random.rand(N, N) < 0.15).astype(float)
    
    res = noc.simulate_event_traffic(spike_train, adj_matrix, strategy_name="Test")
    assert isinstance(res, NoCBenchmarkResult)
    assert res.total_spikes > 0
    assert res.inter_core_packets >= 0
    assert res.mean_hops_per_packet >= 0.0
    assert res.mean_latency_cycles >= 1.0
    assert res.total_noc_energy_pj >= 0.0
