"""
COMPUTATIONAL COUPLING — Mouse Cortical Microcircuit Loader
Loads the MICrONS Pinky100 proofread Layer 2/3 primary visual cortex (V1) dataset.
Source: Turner et al. / Schneider-Mizell et al. / Allen Institute & Princeton (Nature 2025 / Zenodo 3710459).
Provides soma valence annotations ('e' vs 'i') and proofread synaptic contacts.
"""

import os
import csv
from collections import defaultdict
from typing import Optional, Dict, Any, Tuple
import numpy as np
from .unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata


DEFAULT_MOUSE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "mouse"
)


def load_mouse_v1_connectome(
    data_dir: Optional[str] = None,
    min_synapses: int = 1,
    weight_mode: str = "synapse_count",  # "synapse_count" or "cleft_volume" or "spine_volume"
) -> ConnectomeGraph:
    """
    Load the MICrONS Pinky100 Layer 2/3 primary visual cortex connectome.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing soma_valence_v185.csv and soma_subgraph_synapses_spines_v185.csv.
    min_synapses : int
        Minimum number of synaptic contacts required between two neurons to form an edge.
    weight_mode : str
        Metric for edge weights: 'synapse_count', 'cleft_volume', or 'spine_volume'.

    Returns
    -------
    ConnectomeGraph
        Standardized connectome graph object with biophysical metadata.
    """
    if data_dir is None:
        data_dir = DEFAULT_MOUSE_DATA_DIR

    valence_file = os.path.join(data_dir, "soma_valence_v185.csv")
    synapse_file = os.path.join(data_dir, "soma_subgraph_synapses_spines_v185.csv")

    if not os.path.exists(valence_file):
        raise FileNotFoundError(f"Mouse valence file not found at {valence_file}")
    if not os.path.exists(synapse_file):
        raise FileNotFoundError(f"Mouse synapse file not found at {synapse_file}")

    graph = ConnectomeGraph(
        name="Mouse_MICrONS_Pinky100_V1",
        organism="Mus musculus",
        scale="microcircuit",
    )

    # 1. Parse soma valence metadata
    root_to_int_id: Dict[str, int] = {}
    with open(valence_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            root_id = str(row["pt_root_id"]).strip()
            cell_type_raw = str(row["cell_type"]).strip().lower()
            sign = 1 if cell_type_raw == "e" else (-1 if cell_type_raw == "i" else 0)
            cell_type = "pyramidal" if sign == 1 else "interneuron"

            try:
                pos_x = float(row.get("soma_x_nm", 0.0))
                pos_y = float(row.get("soma_y_nm", 0.0))
                pos_z = float(row.get("soma_z_nm", 0.0))
                pos = (pos_x, pos_y, pos_z)
            except (ValueError, TypeError):
                pos = None

            int_id = idx
            root_to_int_id[root_id] = int_id

            meta = NeuronMetadata(
                node_id=int_id,
                name=f"M_{root_id}",
                cell_type=cell_type,
                sign=sign,
                neurotransmitter="glutamate" if sign == 1 else "gaba",
                layer="Layer 2/3",
                position=pos,
                organism="Mus musculus",
                extra={
                    "pt_root_id": root_id,
                    "soma_id": row.get("id", ""),
                },
            )
            graph.add_neuron(meta)

    # 2. Parse and aggregate synaptic connections
    # Multiple physical synapses can occur between the same pre and post neuron
    pair_synapses = defaultdict(list)
    with open(synapse_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre_root = str(row["pre_root_id"]).strip()
            post_root = str(row["post_root_id"]).strip()

            if pre_root not in root_to_int_id or post_root not in root_to_int_id:
                continue

            cleft_vx = float(row.get("cleft_vx", 1.0) or 1.0)
            spine_vol = float(row.get("spine_vol_um3", 0.0) or 0.0)
            pair_synapses[(pre_root, post_root)].append((cleft_vx, spine_vol))

    # 3. Add edges into graph
    for (pre_root, post_root), syn_list in pair_synapses.items():
        syn_count = len(syn_list)
        if syn_count < min_synapses:
            continue

        u = root_to_int_id[pre_root]
        v = root_to_int_id[post_root]
        pre_meta = graph.node_metadata[u]
        sign = pre_meta.sign

        if weight_mode == "cleft_volume":
            weight = sum(c[0] for c in syn_list)
        elif weight_mode == "spine_volume":
            weight = sum(c[1] for c in syn_list)
        else:  # synapse_count
            weight = float(syn_count)

        avg_spine = float(np.mean([c[1] for c in syn_list]))

        syn_meta = SynapseMetadata(
            source_id=u,
            target_id=v,
            weight=weight,
            sign=sign,
            synapse_type="chemical",
            delay_ms=1.8,  # Typical rodent cortical synaptic transmission latency
            extra={
                "synapse_count": syn_count,
                "mean_spine_vol_um3": avg_spine,
            },
        )
        graph.add_synapse(syn_meta)

    return graph
