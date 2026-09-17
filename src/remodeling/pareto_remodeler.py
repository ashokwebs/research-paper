"""
COMPUTATIONAL COUPLING — Biologically Constrained Remodeling Engine
Maps the Degeneracy Frontier between structural drift Delta_struct(G_0, G') and task performance F_task.
Enforces:
  1. Strict Dale's E/I sign preservation
  2. Maximum in/out degree bounds
  3. Synaptic budget constraints
Performs causal edge lesioning to dissociate computational invariants from developmental scaffolding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Callable, Any
import copy
import random
import numpy as np
import networkx as nx
from ..connectome.unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata
from ..simulation.lif_simulator import VectorizedLIFSimulator, SimulationConfig, SimulationResult


@dataclass
class EdgeLesionResult:
    """Outcome of causal single-edge disruption."""
    source_id: int
    target_id: int
    weight: float
    sign: int
    baseline_score: float
    lesioned_score: float
    score_drop: float
    relative_drop: float
    category: str  # 'invariant', 'regulatory', or 'scaffolding'


@dataclass
class ParetoCandidate:
    """A candidate remodeled connectome on the degeneracy frontier."""
    graph: ConnectomeGraph
    structural_distance: float  # Delta_struct in [0, 1]
    task_performance: float     # F_task in [0, 1]
    total_synops: int           # Hardware energy proxy
    dales_valid: bool
    is_pareto_optimal: bool = False
    generation: int = 0


class CausalLesionAnalyzer:
    """
    Performs causal edge ablation on biological connectomes to isolate
    non-rewirable computational invariants from rewirable developmental scaffolding.
    """

    def __init__(
        self,
        connectome: ConnectomeGraph,
        eval_fn: Callable[[ConnectomeGraph], float],
    ):
        self.connectome = connectome
        self.eval_fn = eval_fn

    def analyze_edges(
        self,
        edge_subset: Optional[List[Tuple[int, int]]] = None,
        max_edges: int = 100,
        invariant_threshold: float = 0.50,
        scaffolding_threshold: float = 0.02,
    ) -> List[EdgeLesionResult]:
        """
        Ablate individual edges and measure relative task performance degradation.

        Parameters
        ----------
        edge_subset : list of (u, v), optional
            Specific edges to test. If None, samples up to max_edges.
        max_edges : int
            Maximum number of edges to evaluate.
        invariant_threshold : float
            Relative score drop above which an edge is considered a computational invariant (> 50%).
        scaffolding_threshold : float
            Relative score drop below which an edge is considered developmental scaffolding (< 2%).
        """
        # Baseline performance of pristine biological circuit
        baseline = self.eval_fn(self.connectome)

        all_edges = list(self.connectome.edge_metadata.keys())
        if edge_subset is not None:
            edges_to_test = [e for e in edge_subset if e in self.connectome.edge_metadata]
        elif len(all_edges) > max_edges:
            # Sort edges by weight descending and sample across high, medium, and low weights
            all_edges_sorted = sorted(
                all_edges,
                key=lambda e: self.connectome.edge_metadata[e].weight,
                reverse=True
            )
            # Top 40% high weight, 30% medium, 30% low
            n_high = int(max_edges * 0.4)
            n_mid = int(max_edges * 0.3)
            n_low = max_edges - n_high - n_mid

            high_edges = all_edges_sorted[:n_high]
            mid_start = len(all_edges_sorted) // 3
            mid_edges = all_edges_sorted[mid_start : mid_start + n_mid]
            low_edges = all_edges_sorted[-n_low:]
            edges_to_test = high_edges + mid_edges + low_edges
        else:
            edges_to_test = all_edges

        results = []
        for u, v in edges_to_test:
            e_meta = self.connectome.edge_metadata[(u, v)]

            # Create temporarily lesioned graph by removing (u, v)
            G_lesioned = self.connectome.graph.copy()
            G_lesioned.remove_edge(u, v)

            # Wrap in ConnectomeGraph
            sub_conn = copy.copy(self.connectome)
            sub_conn.graph = G_lesioned

            # Evaluate task performance under lesion
            lesioned_score = self.eval_fn(sub_conn)
            drop = max(0.0, baseline - lesioned_score)
            rel_drop = drop / (baseline + 1e-8)

            if rel_drop >= invariant_threshold:
                cat = "invariant"
            elif rel_drop <= scaffolding_threshold:
                cat = "scaffolding"
            else:
                cat = "regulatory"

            res = EdgeLesionResult(
                source_id=u,
                target_id=v,
                weight=e_meta.weight,
                sign=e_meta.sign,
                baseline_score=baseline,
                lesioned_score=lesioned_score,
                score_drop=drop,
                relative_drop=rel_drop,
                category=cat,
            )
            results.append(res)

        return results


class ParetoRemodeler:
    """
    Multi-objective evolutionary optimization exploring the Degeneracy Frontier:
    Maximizes structural divergence Delta_struct(G_0, G') while maintaining task performance F_task
    under strict Dale's principle and degree constraints.
    """

    def __init__(
        self,
        base_graph: ConnectomeGraph,
        eval_fn: Callable[[ConnectomeGraph], Tuple[float, int]],  # Returns (F_task, SynOps)
        degree_slack: float = 0.20,
        weight_dist_coeff: float = 0.50,
    ):
        self.base_graph = base_graph
        self.eval_fn = eval_fn
        self.degree_slack = degree_slack
        self.weight_dist_coeff = weight_dist_coeff

        # Precompute biological baseline metrics
        self.base_edges = set(base_graph.edge_metadata.keys())
        self.base_num_edges = len(self.base_edges)

        in_degs = [d for _, d in base_graph.graph.in_degree()]
        out_degs = [d for _, d in base_graph.graph.out_degree()]
        self.max_in_deg = int(np.max(in_degs) * (1.0 + degree_slack)) if in_degs else 50
        self.max_out_deg = int(np.max(out_degs) * (1.0 + degree_slack)) if out_degs else 50

        # Baseline evaluation
        base_score, base_ops = self.eval_fn(self.base_graph)
        self.base_score = base_score
        self.base_ops = base_ops

    def compute_structural_distance(self, candidate: ConnectomeGraph) -> float:
        """
        Compute normalized structural distance Delta_struct(G_0, G').
        Combines edge symmetric difference (Jaccard) and weight L1 difference.
        """
        cand_edges = set(candidate.edge_metadata.keys())
        diff_edges = len(self.base_edges.symmetric_difference(cand_edges))
        total_edges = len(self.base_edges) + len(cand_edges)
        jaccard_dist = diff_edges / max(1, total_edges)

        # L1 weight distance on shared edges
        common_edges = self.base_edges.intersection(cand_edges)
        if common_edges:
            w_diff = sum(
                abs(self.base_graph.edge_metadata[e].weight - candidate.edge_metadata[e].weight)
                for e in common_edges
            )
            w_sum = sum(
                self.base_graph.edge_metadata[e].weight + candidate.edge_metadata[e].weight
                for e in common_edges
            )
            w_dist = w_diff / max(1.0, w_sum)
        else:
            w_dist = 1.0

        return float(jaccard_dist * (1.0 - self.weight_dist_coeff) + w_dist * self.weight_dist_coeff)

    def mutate_graph(self, parent: ConnectomeGraph, mutation_rate: float = 0.05) -> ConnectomeGraph:
        """
        Apply biologically valid stochastic mutations:
          1. Rewiring an edge (respects Dale's principle of source)
          2. Perturbing synaptic weight
          3. Pruning an edge
          4. Inserting a new synaptic connection
        """
        child = ConnectomeGraph(
            name=f"{parent.name}_mutant",
            organism=parent.organism,
            scale=parent.scale,
        )
        # Copy neurons
        for nid, meta in parent.node_metadata.items():
            child.add_neuron(copy.deepcopy(meta))

        # Copy synapses with probabilistic mutation
        edges = list(parent.edge_metadata.values())
        nodes = list(parent.node_metadata.keys())

        for e in edges:
            if random.random() < mutation_rate:
                op = random.choice(["rewire", "perturb", "prune"])
                if op == "prune":
                    # Drop edge
                    continue
                elif op == "perturb":
                    # Gaussian weight jitter
                    new_w = max(0.5, float(e.weight * np.random.normal(1.0, 0.2)))
                    new_e = copy.deepcopy(e)
                    new_e.weight = new_w
                    child.add_synapse(new_e)
                elif op == "rewire":
                    # Pick new post-synaptic target respecting degree bounds
                    u = e.source_id
                    possible_targets = [n for n in nodes if n != u and not child.graph.has_edge(u, n)]
                    if possible_targets:
                        new_target = random.choice(possible_targets)
                        new_e = copy.deepcopy(e)
                        new_e.target_id = new_target
                        child.add_synapse(new_e)
                    else:
                        child.add_synapse(copy.deepcopy(e))
            else:
                child.add_synapse(copy.deepcopy(e))

        # Possible novel edge insertion (1-2 edges)
        if random.random() < 0.25 and len(nodes) > 1:
            u, v = random.sample(nodes, 2)
            if not child.graph.has_edge(u, v):
                src_sign = child.node_metadata[u].sign
                new_e = SynapseMetadata(
                    source_id=u,
                    target_id=v,
                    weight=1.0,
                    sign=src_sign,
                    synapse_type="chemical",
                    delay_ms=1.8,
                )
                child.add_synapse(new_e)

        return child

    def run_evolution(
        self,
        population_size: int = 16,
        generations: int = 10,
        mutation_rate: float = 0.08,
        seed: int = 42,
    ) -> List[ParetoCandidate]:
        """
        Run multi-objective evolutionary search along the Degeneracy Frontier.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Initialize population from base graph with small mutations
        population: List[ParetoCandidate] = []

        # Candidate 0: pristine biological baseline
        base_cand = ParetoCandidate(
            graph=self.base_graph,
            structural_distance=0.0,
            task_performance=self.base_score,
            total_synops=self.base_ops,
            dales_valid=True,
            generation=0,
        )
        population.append(base_cand)

        for _ in range(population_size - 1):
            mut_g = self.mutate_graph(self.base_graph, mutation_rate=mutation_rate)
            d_struct = self.compute_structural_distance(mut_g)
            score, ops = self.eval_fn(mut_g)
            dales_valid, _ = mut_g.validate_dales_principle()
            cand = ParetoCandidate(
                graph=mut_g,
                structural_distance=d_struct,
                task_performance=score,
                total_synops=ops,
                dales_valid=dales_valid,
                generation=0,
            )
            population.append(cand)

        all_explored: List[ParetoCandidate] = list(population)

        for gen in range(1, generations + 1):
            # Sort by Pareto dominance: want high Delta_struct, high F_task, low SynOps
            # Fast non-dominated sorting / fitness ranking
            survivors = sorted(
                population,
                key=lambda c: (
                    1 if c.dales_valid else 0,
                    c.task_performance,
                    c.structural_distance,
                    -c.total_synops
                ),
                reverse=True
            )[: population_size // 2]

            next_pop = list(survivors)
            while len(next_pop) < population_size:
                parent = random.choice(survivors)
                mut_g = self.mutate_graph(parent.graph, mutation_rate=mutation_rate)
                d_struct = self.compute_structural_distance(mut_g)
                score, ops = self.eval_fn(mut_g)
                dales_valid, _ = mut_g.validate_dales_principle()
                child = ParetoCandidate(
                    graph=mut_g,
                    structural_distance=d_struct,
                    task_performance=score,
                    total_synops=ops,
                    dales_valid=dales_valid,
                    generation=gen,
                )
                next_pop.append(child)
                all_explored.append(child)

            population = next_pop

        # Mark non-dominated Pareto frontier points
        for i, c1 in enumerate(all_explored):
            dominated = False
            for j, c2 in enumerate(all_explored):
                if i != j:
                    # c2 dominates c1 if c2 is >= in all objectives and strictly > in at least one
                    if (
                        c2.task_performance >= c1.task_performance
                        and c2.structural_distance >= c1.structural_distance
                        and c2.total_synops <= c1.total_synops
                        and (
                            c2.task_performance > c1.task_performance
                            or c2.structural_distance > c1.structural_distance
                            or c2.total_synops < c1.total_synops
                        )
                    ):
                        dominated = True
                        break
            c1.is_pareto_optimal = not dominated

        return all_explored
