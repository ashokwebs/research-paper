"""
COMPUTATIONAL COUPLING — C. elegans Connectome Loader
Loads the Cook et al. (Nature 2019) adult hermaphrodite connectome from ConnectomeToolbox.
Maps cell classifications, chemical synapses, gap junctions, and Dale's E/I signs.
"""

import os
import csv
from typing import Optional, Set
from .unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata


# Canonical neurotransmitter mapping for C. elegans
GABA_NEURONS = {
    "DD1", "DD2", "DD3", "DD4", "DD5", "DD6",
    "VD1", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9", "VD10", "VD11", "VD12", "VD13",
    "RMEL", "RMER", "RMED", "RMEV", "AVL", "DVB", "RIS"
}

GLUTAMATE_NEURONS = {
    "ASEL", "ASER", "AWC", "AWCL", "AWCR", "ASH", "ASHL", "ASHR", "AFDL", "AFDR",
    "ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM",
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
    "AIAL", "AIAR", "AIBL", "AIBR"
}

DEFAULT_CELEGANS_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "upstream", "openworm", "ConnectomeToolbox", "cect", "data"
)


def load_celegans_connectome(
    data_dir: Optional[str] = None,
    include_electrical: bool = True,
    synapse_threshold: int = 1,
    neurons_only: bool = True,
) -> ConnectomeGraph:
    """
    Load the complete adult hermaphrodite C. elegans connectome (Cook et al. Nature 2019).

    Parameters
    ----------
    data_dir : str, optional
        Path to cect/data directory containing herm_full_edgelist.csv and all_cell_info.csv
    include_electrical : bool
        Whether to include undirected electrical gap junctions as bidirectional edges
    synapse_threshold : int
        Minimum synapse weight threshold (default 1)
    neurons_only : bool
        If True, exclude non-neuronal muscle/endorgan targets

    Returns
    -------
    ConnectomeGraph
        Standardized connectome graph object
    """
    if data_dir is None:
        data_dir = DEFAULT_CELEGANS_DATA_DIR

    edgelist_path = os.path.join(data_dir, "herm_full_edgelist.csv")
    cell_info_path = os.path.join(data_dir, "all_cell_info.csv")

    if not os.path.exists(edgelist_path):
        raise FileNotFoundError(f"C. elegans edgelist not found at {edgelist_path}")

    # 1. Load cell metadata
    cell_metadata = {}
    if os.path.exists(cell_info_path):
        with open(cell_info_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cname = row.get("Cell name", "").strip()
                if cname:
                    cell_metadata[cname] = {
                        "type": row.get("Type", "unknown").strip(),
                        "classification": row.get("Classification", "").strip(),
                        "lineage": row.get("Lineage", "").strip(),
                    }

    # 2. Read edge list and identify unique cell names
    raw_edges = []
    cell_names = set()

    with open(edgelist_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["Source"].strip()
            tgt = row["Target"].strip()
            weight = float(row.get("Weight", 1.0))
            stype = row.get("Type", "chemical").strip().lower()

            if weight < synapse_threshold:
                continue

            if neurons_only:
                # Muscles and endorgans in C. elegans start with lowercase (e.g. pm1, mu_bod)
                if src[0].islower() or tgt[0].islower():
                    continue

            cell_names.add(src)
            cell_names.add(tgt)
            raw_edges.append((src, tgt, weight, stype))

    # 3. Create standardized graph
    graph = ConnectomeGraph(
        name="CElegans_Cook2019_Hermaphrodite",
        organism="Caenorhabditis elegans",
        scale="whole_nervous_system",
    )

    name_to_id = {name: i for i, name in enumerate(sorted(cell_names))}

    for name, nid in name_to_id.items():
        meta = cell_metadata.get(name, {})
        ctype = meta.get("classification", meta.get("type", "neuron"))

        # Determine neurotransmitter and sign
        if name in GABA_NEURONS or any(name.startswith(g) for g in ("DD", "VD", "RME")):
            nt = "gaba"
            sign = -1
        elif name in GLUTAMATE_NEURONS:
            nt = "glutamate"
            sign = 1
        else:
            nt = "acetylcholine"
            sign = 1  # Majority of remaining nematode neurons are cholinergic / excitatory

        neuron_meta = NeuronMetadata(
            node_id=nid,
            name=name,
            cell_type=ctype,
            sign=sign,
            neurotransmitter=nt,
            organism="Caenorhabditis elegans",
            extra=meta,
        )
        graph.add_neuron(neuron_meta)

    # 4. Add synapses
    for src, tgt, weight, stype in raw_edges:
        u, v = name_to_id[src], name_to_id[tgt]
        src_sign = graph.node_metadata[u].sign

        syn_meta = SynapseMetadata(
            source_id=u,
            target_id=v,
            weight=weight,
            sign=src_sign,
            synapse_type=stype,
            delay_ms=1.5,
        )
        graph.add_synapse(syn_meta)

        if stype == "electrical" and include_electrical:
            # Add reverse gap junction edge
            tgt_sign = graph.node_metadata[v].sign
            rev_syn = SynapseMetadata(
                source_id=v,
                target_id=u,
                weight=weight,
                sign=tgt_sign,
                synapse_type="electrical",
                delay_ms=0.5,  # Electrical gap junctions have minimal delay
            )
            graph.add_synapse(rev_syn)

    return graph
