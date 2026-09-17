"""
COMPUTATIONAL COUPLING — Human Neural Connectivity Loader
Provides loaders for both human neural datasets:
  1. Microscale (H01): Shapson-Coe et al. (Science 2024), 1 mm³ human temporal cortex,
     nanoscale synaptic connectome fragment, proofread E/I cell classifications and layers.
  2. Macroscale (HCP): Human Connectome Project 1,015-region group consensus structural
     connectome matrix from diffusion tractography (Lausanne parcellation, Yeo RSNs).
"""

import os
import json
import csv
from typing import Optional, Set, List, Dict, Any, Union
import numpy as np
import pandas as pd
from .unified_schema import ConnectomeGraph, NeuronMetadata, SynapseMetadata


DEFAULT_H01_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "upstream", "human", "h01"
)

DEFAULT_HCP_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "conn2res_data", "data", "human"
)

# E/I cell type mapping based on Shapson-Coe et al. 2024
H01_CELLTYPE_TO_EI = {
    "pyramidal neuron": 1,
    "excitatory/spiny neuron with atypical tree": 1,
    "spiny stellate neuron": 1,
    "interneuron": -1,
}


def load_human_h01_microcircuit(
    h01_dir: Optional[str] = None,
    layers: Optional[List[str]] = None,
    neurons_only: bool = True,
    max_neurons: Optional[int] = None,
) -> ConnectomeGraph:
    """
    Load human temporal cortex microcircuit data from the H01 release.

    Parameters
    ----------
    h01_dir : str, optional
        Path to upstream/human/h01 directory.
    layers : list of str, optional
        Filter to specific cortical layers, e.g. ['Layer 2', 'Layer 3']. If None, all layers.
    neurons_only : bool
        If True, exclude glial cells (oligodendrocytes, astrocytes, microglia).
    max_neurons : int, optional
        Optional cap on number of neurons.

    Returns
    -------
    ConnectomeGraph
        Standardized connectome graph object.
    """
    if h01_dir is None:
        h01_dir = DEFAULT_H01_DATA_DIR

    cell_json_path = os.path.join(h01_dir, "agglo_20200916c3_cell_data.json")
    ei_csv_path = os.path.join(h01_dir, "goog14r0s5c3_e_i_ratios_feb_2023.csv")
    gt_pairs_path = os.path.join(h01_dir, "gt_pair_data_5000nm_cutoff_with_skel_dists.json")

    if not os.path.exists(cell_json_path):
        raise FileNotFoundError(f"H01 cell json not found at {cell_json_path}")
    if not os.path.exists(ei_csv_path):
        raise FileNotFoundError(f"H01 E/I csv not found at {ei_csv_path}")

    # 1. Load E/I ratio and layer mapping
    ei_df = pd.read_csv(ei_csv_path)
    ei_map = {}
    for _, row in ei_df.iterrows():
        seg_id = str(row["segment id"])
        ei_map[seg_id] = {
            "exc_synapses": int(row["excitatory synapses"]),
            "inh_synapses": int(row["inhibitory synapses"]),
            "layer": str(row["cortical layer"]).strip(),
            "cell_type": str(row["cell type"]).strip(),
        }

    # 2. Load cell positions and morphology from JSON
    with open(cell_json_path, "r", encoding="utf-8") as f:
        cell_data = json.load(f)

    graph = ConnectomeGraph(
        name="Human_H01_TemporalCortex_Microcircuit",
        organism="Homo sapiens",
        scale="microcircuit",
    )

    seg_to_node_id: Dict[str, int] = {}
    node_idx = 0

    for cell in cell_data:
        agglo_seg = str(cell.get("agglo_seg", ""))
        raw_type = str(cell.get("type", "")).lower()

        # Check if cell is a neuron
        if neurons_only and "neuron" not in raw_type:
            continue

        ei_info = ei_map.get(agglo_seg, {})
        layer = ei_info.get("layer", "Unknown")

        if layers is not None and layer not in layers:
            continue

        sign = H01_CELLTYPE_TO_EI.get(raw_type, 1)

        try:
            pos = (
                float(cell.get("true_x", 0.0)),
                float(cell.get("true_y", 0.0)),
                float(cell.get("true_z", 0.0)),
            )
        except (ValueError, TypeError):
            pos = None

        meta = NeuronMetadata(
            node_id=node_idx,
            name=f"H01_{agglo_seg}",
            cell_type=raw_type,
            sign=sign,
            neurotransmitter="glutamate" if sign == 1 else "gaba",
            layer=layer,
            position=pos,
            organism="Homo sapiens",
            extra={
                "agglo_seg": agglo_seg,
                "base_seg": cell.get("base_seg", ""),
                "soma_cubic_um": cell.get("soma_cubic_um", 0),
                "exc_synapses": ei_info.get("exc_synapses", 0),
                "inh_synapses": ei_info.get("inh_synapses", 0),
            },
        )
        graph.add_neuron(meta)
        seg_to_node_id[agglo_seg] = node_idx
        node_idx += 1

        if max_neurons is not None and node_idx >= max_neurons:
            break

    # 3. Load ground truth proofread synaptic pairs if present
    if os.path.exists(gt_pairs_path):
        try:
            with open(gt_pairs_path, "r", encoding="utf-8") as f:
                gt_pairs = json.load(f)

            for pair in gt_pairs:
                pre_id = str(pair.get("pre_agglo_id", ""))
                post_id = str(pair.get("post_agglo_id", ""))

                if pre_id in seg_to_node_id and post_id in seg_to_node_id:
                    u = seg_to_node_id[pre_id]
                    v = seg_to_node_id[post_id]
                    pre_sign = graph.node_metadata[u].sign
                    dist_nm = float(pair.get("dist_nm", 0.0))

                    syn_meta = SynapseMetadata(
                        source_id=u,
                        target_id=v,
                        weight=1.0,
                        sign=pre_sign,
                        synapse_type="chemical",
                        delay_ms=2.0,  # Human neocortical transmission latency
                        extra={"dist_nm": dist_nm},
                    )
                    graph.add_synapse(syn_meta)
        except Exception:
            pass

    return graph


def load_human_hcp_macroscale(
    data_dir: Optional[str] = None,
    network: Optional[str] = None,
    weight_threshold: float = 0.001,
) -> ConnectomeGraph:
    """
    Load the Human Connectome Project (HCP) 1,015-region group consensus structural connectome.

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing consensus_0.npy, coords.npy, and rsn_mapping.npy.
    network : str, optional
        Filter to specific Yeo resting-state network:
        'VIS' (Visual), 'SM' (Somatomotor), 'DA' (Dorsal Attention), 'VA' (Ventral Attention),
        'LIM' (Limbic), 'FP' (Frontoparietal), 'DMN' (Default Mode), 'subctx' (Subcortex).
    weight_threshold : float
        Threshold below which structural tractography streamline density is set to 0.

    Returns
    -------
    ConnectomeGraph
        Standardized macroscale connectome graph.
    """
    if data_dir is None:
        data_dir = DEFAULT_HCP_DATA_DIR

    mat_path = os.path.join(data_dir, "consensus_0.npy")
    coords_path = os.path.join(data_dir, "coords.npy")
    rsn_path = os.path.join(data_dir, "rsn_mapping.npy")
    hemi_path = os.path.join(data_dir, "hemiid.npy")

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"HCP consensus matrix not found at {mat_path}")

    adj = np.load(mat_path)
    coords = np.load(coords_path) if os.path.exists(coords_path) else None
    rsn = np.load(rsn_path) if os.path.exists(rsn_path) else None
    hemi = np.load(hemi_path) if os.path.exists(hemi_path) else None

    n = adj.shape[0]
    name_suffix = f"_{network}" if network else "_WholeBrain"

    graph = ConnectomeGraph(
        name=f"Human_HCP_Consensus1015{name_suffix}",
        organism="Homo sapiens",
        scale="macroscale",
    )

    # Determine node subset
    if network is not None and rsn is not None:
        selected_nodes = [i for i in range(n) if str(rsn[i]).strip() == network.strip()]
    else:
        selected_nodes = list(range(n))

    node_to_new_id = {old: new for new, old in enumerate(selected_nodes)}

    for old_id in selected_nodes:
        new_id = node_to_new_id[old_id]
        rsn_name = str(rsn[old_id]).strip() if rsn is not None else "Unknown"
        h = int(hemi[old_id]) if hemi is not None else 0
        hemi_str = "LH" if h == 0 else "RH"
        pos = tuple(coords[old_id]) if coords is not None else None

        meta = NeuronMetadata(
            node_id=new_id,
            name=f"HCP_{hemi_str}_{rsn_name}_{old_id}",
            cell_type=f"parcel_{rsn_name}",
            sign=1,  # Macroscale fiber tracts are predominantly glutamatergic excitatory projections
            layer="Macroscale_Parcel",
            position=pos,
            organism="Homo sapiens",
            extra={
                "original_parcel_id": old_id,
                "rsn": rsn_name,
                "hemisphere": hemi_str,
            },
        )
        graph.add_neuron(meta)

    # Add edges
    for i in selected_nodes:
        for j in selected_nodes:
            w = adj[i, j]
            if w >= weight_threshold:
                u = node_to_new_id[i]
                v = node_to_new_id[j]

                # Estimated inter-regional transmission delay based on Euclidean distance
                if coords is not None:
                    dist_mm = np.linalg.norm(coords[i] - coords[j])
                    delay_ms = max(1.0, float(dist_mm / 10.0))  # ~10 m/s myelinated conduction velocity
                else:
                    delay_ms = 5.0

                syn_meta = SynapseMetadata(
                    source_id=u,
                    target_id=v,
                    weight=float(w),
                    sign=1,
                    synapse_type="macro_tract",
                    delay_ms=delay_ms,
                    extra={"streamline_density": float(w)},
                )
                graph.add_synapse(syn_meta)

    return graph
