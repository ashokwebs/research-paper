"""
COMPUTATIONAL COUPLING — Motif Discovery & Statistical Significance Engine
Extracts directed 3-node and 4-node topological and sign-attributed computational motifs.
Implements Maslov-Sneppen degree-preserving configuration null models respecting Dale's principle.
Computes Z-scores, empirical p-values, and Normalized Significance Profiles (SP).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
import random
import numpy as np
import networkx as nx
from ..connectome.unified_schema import ConnectomeGraph


@dataclass
class MotifStats:
    """Statistical summary for a specific motif type."""
    motif_name: str
    real_count: int
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    significance_profile: float = 0.0


@dataclass
class ConnectomeSignificanceReport:
    """Full motif census and significance profile for a connectome graph."""
    graph_name: str
    organism: str
    num_neurons: int
    num_synapses: int
    num_null_samples: int
    motifs: Dict[str, MotifStats] = field(default_factory=dict)

    def summary_table(self) -> str:
        lines = [
            f"=== MOTIF SIGNIFICANCE REPORT: {self.graph_name} ({self.organism}) ===",
            f"Neurons: {self.num_neurons} | Synapses: {self.num_synapses} | Null Samples: {self.num_null_samples}",
            f"{'Motif Type':<32} {'Real':>8} {'Null Mean':>10} {'Null Std':>10} {'Z-Score':>10} {'p-value':>10} {'Norm SP':>10}",
            "-" * 96,
        ]
        for name, stats in self.motifs.items():
            lines.append(
                f"{name:<32} {stats.real_count:>8} {stats.null_mean:>10.2f} {stats.null_std:>10.2f} "
                f"{stats.z_score:>10.2f} {stats.p_value:>10.4f} {stats.significance_profile:>10.3f}"
            )
        return "\n".join(lines)


class MotifDiscoveryEngine:
    """
    High-performance motif discovery engine for directed, sign-attributed connectome graphs.
    """

    def __init__(self, graph: ConnectomeGraph):
        self.connectome = graph
        self.nx_graph = graph.graph

    def count_motifs(self, G: Optional[nx.DiGraph] = None) -> Dict[str, int]:
        """
        Count directed 3-node and 4-node computational motifs in the given DiGraph.
        If G is None, uses self.nx_graph.
        """
        if G is None:
            G = self.nx_graph

        counts = {
            "FFL_Total": 0,               # All Feedforward Loops (u->v, v->w, u->w)
            "cFFL_Coherent": 0,           # Coherent FFL (sign product > 0: persistence / delay detector)
            "iFFL_Incoherent": 0,         # Incoherent FFL (sign product < 0: pulse generator / adaptation)
            "FBL_3Cycle": 0,              # Directed 3-cycle feedback loop (u->v->w->u)
            "Reciprocal_Pair": 0,         # 2-node reciprocal connection (u <-> v)
            "Reciprocal_Inhibitory_Pair": 0, # Mutual inhibition (u <-> v, both inhibitory: bistable switch)
            "Disinhibitory_Triad": 0,     # u -> v -> w where v, w are inhibitory interneurons
            "Bi_Fan": 0,                  # 4-node combinatorial coincidence (u1, u2 -> v1, v2)
            "Fan_In": 0,                  # 2 inputs converging to 1 target (u -> w <- v)
            "Fan_Out": 0,                 # 1 source diverging to 2 targets (u <- v -> w)
        }

        nodes = list(G.nodes())
        node_set = set(nodes)

        # 1. Reciprocal pairs (2-node)
        seen_recip = set()
        for u in nodes:
            u_succ = set(G.successors(u))
            u_sign = G.nodes[u].get("sign", 1) if "sign" in G.nodes[u] else 1

            for v in u_succ:
                if u < v and G.has_edge(v, u):
                    counts["Reciprocal_Pair"] += 1
                    v_sign = G.nodes[v].get("sign", 1) if "sign" in G.nodes[v] else 1
                    if u_sign == -1 and v_sign == -1:
                        counts["Reciprocal_Inhibitory_Pair"] += 1

        # 2. Directed 3-node motifs
        # Precompute successors and predecessors for fast set operations
        succ_dict = {u: set(G.successors(u)) for u in nodes}
        pred_dict = {u: set(G.predecessors(u)) for u in nodes}

        # Fan-in and Fan-out
        for w in nodes:
            preds = list(pred_dict[w])
            lp = len(preds)
            if lp >= 2:
                counts["Fan_In"] += (lp * (lp - 1)) // 2

            succs = list(succ_dict[w])
            ls = len(succs)
            if ls >= 2:
                counts["Fan_Out"] += (ls * (ls - 1)) // 2

        # Feedforward Loops and 3-Cycles
        for u in nodes:
            u_succ = succ_dict[u]
            u_sign = G.nodes[u].get("sign", 1)

            for v in u_succ:
                if v == u:
                    continue
                v_succ = succ_dict[v]
                v_sign = G.nodes[v].get("sign", 1)

                # Feedforward Loop: u->v, v->w, and u->w
                # Common nodes that are successors of both u and v
                ffl_targets = u_succ & v_succ
                for w in ffl_targets:
                    if w != u and w != v:
                        counts["FFL_Total"] += 1

                        # Edge signs
                        e_uv = G[u][v].get("sign", u_sign)
                        e_vw = G[v][w].get("sign", v_sign)
                        e_uw = G[u][w].get("sign", u_sign)

                        sign_prod = e_uv * e_vw * e_uw
                        if sign_prod > 0:
                            counts["cFFL_Coherent"] += 1
                        else:
                            counts["iFFL_Incoherent"] += 1

                # 3-Cycle Feedback Loop: u->v->w->u (w must be in v_succ and in pred_dict[u])
                fbl_targets = v_succ & pred_dict[u]
                for w in fbl_targets:
                    if u < v and v < w:  # Count each 3-cycle exactly once
                        counts["FBL_3Cycle"] += 1

                # Disinhibitory Triad: u -> v -> w where v is inhibitory and w is inhibitory
                if v_sign == -1:
                    for w in v_succ:
                        if w != u and w != v:
                            w_sign = G.nodes[w].get("sign", 1)
                            if w_sign == -1:
                                counts["Disinhibitory_Triad"] += 1

        # 3. 4-Node Bi-Fan Motifs: (u1, u2) -> (v1, v2)
        # For each pair of nodes (u1, u2), count common successors
        node_list = nodes
        n_nodes = len(node_list)
        # If network is very large, sample pairs to prevent O(N^2) explosion
        if n_nodes <= 1500:
            for i in range(n_nodes):
                u1 = node_list[i]
                succ1 = succ_dict[u1]
                if len(succ1) < 2:
                    continue
                for j in range(i + 1, n_nodes):
                    u2 = node_list[j]
                    succ2 = succ_dict[u2]
                    common = succ1 & succ2
                    # Exclude self-references
                    common.discard(u1)
                    common.discard(u2)
                    k = len(common)
                    if k >= 2:
                        counts["Bi_Fan"] += (k * (k - 1)) // 2
        else:
            # Sample 2,000 random pairs to estimate bi-fan frequency scaled proportionally
            sample_size = 2000
            sampled_bifans = 0
            all_pairs_total = (n_nodes * (n_nodes - 1)) // 2
            for _ in range(sample_size):
                u1, u2 = random.sample(node_list, 2)
                common = (succ_dict[u1] & succ_dict[u2]) - {u1, u2}
                k = len(common)
                if k >= 2:
                    sampled_bifans += (k * (k - 1)) // 2
            counts["Bi_Fan"] = int(sampled_bifans * (all_pairs_total / sample_size))

        return counts

    def generate_dales_null_model(self, G: Optional[nx.DiGraph] = None, n_swaps_factor: int = 5) -> nx.DiGraph:
        """
        Generate a degree-preserving configuration null model using double-edge swaps.
        Strictly preserves:
          - Node in-degree sequence k_in(i)
          - Node out-degree sequence k_out(i)
          - Pre-synaptic Dale's E/I sign enforcement
        """
        if G is None:
            G = self.nx_graph

        G_null = G.copy()
        edges = list(G_null.edges())
        num_edges = len(edges)
        if num_edges < 4:
            return G_null

        target_swaps = num_edges * n_swaps_factor
        successful_swaps = 0
        max_attempts = target_swaps * 10
        attempts = 0

        while successful_swaps < target_swaps and attempts < max_attempts:
            attempts += 1
            i1 = random.randrange(num_edges)
            i2 = random.randrange(num_edges)
            if i1 == i2:
                continue

            u, v = edges[i1]
            x, y = edges[i2]

            # Prevent self-loops and multi-edges
            if u == y or x == v or u == x or v == y:
                continue
            if G_null.has_edge(u, y) or G_null.has_edge(x, v):
                continue

            # Check pre-synaptic signs to preserve Dale's principle
            sign_u = G_null.nodes[u].get("sign", 1)
            sign_x = G_null.nodes[x].get("sign", 1)

            # Retrieve edge attributes
            attr_uv = dict(G_null[u][v])
            attr_xy = dict(G_null[x][y])

            # Swap edges: (u->v, x->y) becomes (u->y, x->v)
            G_null.remove_edge(u, v)
            G_null.remove_edge(x, y)

            attr_uy = attr_uv.copy()
            attr_uy["sign"] = sign_u

            attr_xv = attr_xy.copy()
            attr_xv["sign"] = sign_x

            G_null.add_edge(u, y, **attr_uy)
            G_null.add_edge(x, v, **attr_xv)

            edges[i1] = (u, y)
            edges[i2] = (x, v)
            successful_swaps += 1

        return G_null

    def compute_significance_profile(
        self,
        n_null_samples: int = 30,
        n_swaps_factor: int = 5,
        seed: int = 42,
    ) -> ConnectomeSignificanceReport:
        """
        Compute empirical motif counts, null model distribution, Z-scores, p-values,
        and normalized significance profiles.
        """
        random.seed(seed)
        np.random.seed(seed)

        real_counts = self.count_motifs()
        null_counts_list: Dict[str, List[int]] = {m: [] for m in real_counts}

        for _ in range(n_null_samples):
            G_rand = self.generate_dales_null_model(n_swaps_factor=n_swaps_factor)
            counts_rand = self.count_motifs(G_rand)
            for m, val in counts_rand.items():
                null_counts_list[m].append(val)

        motifs_dict: Dict[str, MotifStats] = {}
        z_scores_sq = 0.0

        for m, real_val in real_counts.items():
            rand_vals = np.array(null_counts_list[m], dtype=np.float64)
            null_mean = float(np.mean(rand_vals))
            null_std = float(np.std(rand_vals))

            # Z-score
            denom = null_std if null_std > 1e-6 else 1.0
            z = float((real_val - null_mean) / denom)

            # Empirical p-value
            if z >= 0:
                p_val = float(np.mean(rand_vals >= real_val))
            else:
                p_val = float(np.mean(rand_vals <= real_val))

            motifs_dict[m] = MotifStats(
                motif_name=m,
                real_count=real_val,
                null_mean=null_mean,
                null_std=null_std,
                z_score=z,
                p_value=p_val,
            )
            z_scores_sq += z ** 2

        # Normalized Significance Profile: SP_i = Z_i / sqrt(sum Z_j^2)
        norm_factor = np.sqrt(z_scores_sq) if z_scores_sq > 1e-6 else 1.0
        for m in motifs_dict:
            motifs_dict[m].significance_profile = float(motifs_dict[m].z_score / norm_factor)

        report = ConnectomeSignificanceReport(
            graph_name=self.connectome.name,
            organism=self.connectome.organism,
            num_neurons=self.connectome.num_neurons,
            num_synapses=self.connectome.num_synapses,
            num_null_samples=n_null_samples,
            motifs=motifs_dict,
        )

        return report
