"""
COMPUTATIONAL COUPLING — Drosophila Melanogaster Connectome Loader
Loads FlyWire v630 connectome subcircuits and whole-brain annotations.
Source: Dorkenwald et al., Schlegel et al., Lin et al. (Nature 2024 / FlyWire).
Supports functional subcircuits:
  - Central Complex (CX): Heading direction, navigation, ring attractor (~2.8k neurons)
  - Antennal Lobe (AL): Olfactory representation and processing (~1.1k neurons)
  - Mushroom Body (MB): Associative learning and memory (~5.6k neurons)
  - Optic Lobe (Optic): Visual motion and feature extraction
"""

import os
import gzip
from collections import defaultdict
from typing import Optional, Set, List, Dict, Any
import pandas as pd
import numpy as np
from .unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata


DEFAULT_FLYWIRE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "upstream", "flywire", "flywire-network-analysis", "v630_data"
)

# Drosophila neurotransmitter valence convention:
# ACH = Acetylcholine (Excitatory +1)
# GABA = GABA (Inhibitory -1)
# GLUT = Glutamate (Inhibitory -1 in insect central brain via GluCl channels; Liu & Wilson 2013)
# DA / SER / OCT = Monoaminergic neuromodulators (default +1)
NT_SIGN_MAP = {
    "ACH": 1,
    "GLUT": -1,
    "GABA": -1,
    "DA": 1,
    "SER": 1,
    "OCT": 1,
    "UNKNOWN": 1,
}


def load_drosophila_subcircuit(
    subcircuit: str = "central_complex",
    data_dir: Optional[str] = None,
    min_synapses: int = 5,
    max_neurons: Optional[int] = None,
) -> ConnectomeGraph:
    """
    Load a standardized functional subcircuit from the FlyWire v630 Drosophila connectome.

    Parameters
    ----------
    subcircuit : str
        Predefined subcircuit: 'central_complex' (or 'cx'), 'antennal_lobe' (or 'al'),
        'mushroom_body' (or 'mb'), or 'optic'.
    data_dir : str, optional
        Path to v630_data directory containing classification.csv.gz and connections.csv.gz.
    min_synapses : int
        Minimum synapse count threshold (standard FlyWire threshold is >= 5).
    max_neurons : int, optional
        Optional cap on number of neurons for fast prototyping.

    Returns
    -------
    ConnectomeGraph
        Standardized connectome graph object.
    """
    if data_dir is None:
        data_dir = DEFAULT_FLYWIRE_DATA_DIR

    class_file = os.path.join(data_dir, "classification.csv.gz")
    conn_file = os.path.join(data_dir, "connections.csv.gz")

    if not os.path.exists(class_file):
        raise FileNotFoundError(f"FlyWire classification file not found at {class_file}")
    if not os.path.exists(conn_file):
        raise FileNotFoundError(f"FlyWire connections file not found at {conn_file}")

    # 1. Determine cell filtering by subcircuit
    sub = subcircuit.lower().strip()
    class_df = pd.read_csv(class_file, compression="gzip")

    if sub in ("central_complex", "cx"):
        target_df = class_df[class_df["class"] == "CX"]
        circuit_name = "FlyWire_v630_CentralComplex"
    elif sub in ("antennal_lobe", "al"):
        target_df = class_df[class_df["class"].isin(["ALPN", "ALLN"])]
        circuit_name = "FlyWire_v630_AntennalLobe"
    elif sub in ("mushroom_body", "mb"):
        target_df = class_df[class_df["class"].isin(["Kenyon_Cell", "MBON", "DAN"])]
        circuit_name = "FlyWire_v630_MushroomBody"
    elif sub in ("optic", "optic_lobe"):
        target_df = class_df[class_df["super_class"] == "optic"]
        if max_neurons is None:
            # Optic lobe is massive (73k cells); default to 3,000 for subcircuit analysis if unspecified
            max_neurons = 3000
        circuit_name = "FlyWire_v630_OpticLobe"
    else:
        raise ValueError(f"Unknown Drosophila subcircuit: '{subcircuit}'. Choose 'cx', 'al', 'mb', or 'optic'.")

    if max_neurons is not None and len(target_df) > max_neurons:
        target_df = target_df.iloc[:max_neurons]

    target_roots: Set[int] = set(target_df["root_id"].astype(int))

    # 2. Build graph and register neurons
    graph = ConnectomeGraph(
        name=circuit_name,
        organism="Drosophila melanogaster",
        scale="subcircuit",
    )

    root_to_int_id: Dict[int, int] = {}
    for idx, row in target_df.reset_index(drop=True).iterrows():
        root_id = int(row["root_id"])
        int_id = idx
        root_to_int_id[root_id] = int_id

        cell_class = str(row.get("class", ""))
        super_class = str(row.get("super_class", ""))
        cell_type = str(row.get("cell_type", cell_class))

        # Initial sign unknown until connections determine primary NT
        meta = NeuronMetadata(
            node_id=int_id,
            name=f"D_{root_id}",
            cell_type=f"{super_class}_{cell_class}",
            sign=1,  # Default excitatory, refined below
            organism="Drosophila melanogaster",
            extra={
                "root_id": root_id,
                "super_class": super_class,
                "class": cell_class,
                "cell_type": cell_type,
                "side": str(row.get("side", "")),
            },
        )
        graph.add_neuron(meta)

    # 3. Read connections for target roots
    pre_nt_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_synapses: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(
        lambda: {"syn_count": 0, "nt_types": defaultdict(int), "neuropils": set()}
    )

    for chunk in pd.read_csv(conn_file, compression="gzip", chunksize=500000):
        matched = chunk[chunk["pre_root_id"].isin(target_roots) & chunk["post_root_id"].isin(target_roots)]
        for _, row in matched.iterrows():
            pre_r = int(row["pre_root_id"])
            post_r = int(row["post_root_id"])
            syn_count = int(row["syn_count"])
            nt = str(row.get("nt_type", "UNKNOWN")).upper()
            neuropil = str(row.get("neuropil", ""))

            u = root_to_int_id[pre_r]
            v = root_to_int_id[post_r]

            pre_nt_votes[u][nt] += syn_count
            p_data = pair_synapses[(u, v)]
            p_data["syn_count"] += syn_count
            p_data["nt_types"][nt] += syn_count
            if neuropil:
                p_data["neuropils"].add(neuropil)

    # 4. Resolve pre-synaptic Dale valence
    for u, votes in pre_nt_votes.items():
        if votes:
            majority_nt = max(votes.items(), key=lambda x: x[1])[0]
            sign = NT_SIGN_MAP.get(majority_nt, 1)
            graph.node_metadata[u].sign = sign
            graph.node_metadata[u].neurotransmitter = majority_nt.lower()
            graph.graph.nodes[u]["sign"] = sign
            graph.graph.nodes[u]["neurotransmitter"] = majority_nt.lower()

    # 5. Add synapses to graph
    for (u, v), p_data in pair_synapses.items():
        total_syn = p_data["syn_count"]
        if total_syn < min_synapses:
            continue

        pre_sign = graph.node_metadata[u].sign
        syn_meta = SynapseMetadata(
            source_id=u,
            target_id=v,
            weight=float(total_syn),
            sign=pre_sign,
            synapse_type="chemical",
            delay_ms=2.0,  # Drosophila central synaptic latency
            extra={
                "syn_count": total_syn,
                "neuropils": list(p_data["neuropils"]),
            },
        )
        graph.add_synapse(syn_meta)

    return graph
