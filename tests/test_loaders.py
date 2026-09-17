"""
Unit and integration tests for multi-organism connectome loaders.
"""

import pytest
from src.connectome import (
    load_celegans_connectome,
    load_drosophila_subcircuit,
    load_mouse_v1_connectome,
    load_human_hcp_macroscale,
    load_human_h01_microcircuit,
)


def test_load_celegans():
    g = load_celegans_connectome()
    assert g.organism == "Caenorhabditis elegans"
    assert g.num_neurons == 300
    assert g.num_synapses > 4000
    valid, violations = g.validate_dales_principle()
    assert valid is True
    assert violations == 0


def test_load_drosophila_al():
    g = load_drosophila_subcircuit("al", min_synapses=5)
    assert g.organism == "Drosophila melanogaster"
    assert g.num_neurons > 1000
    assert g.num_synapses > 30000
    valid, violations = g.validate_dales_principle()
    assert valid is True
    assert violations == 0


def test_load_mouse_v1():
    g = load_mouse_v1_connectome()
    assert g.organism == "Mus musculus"
    assert g.num_neurons == 456
    assert g.num_synapses > 1500
    valid, violations = g.validate_dales_principle()
    assert valid is True
    assert violations == 0


def test_load_human_hcp():
    g = load_human_hcp_macroscale(network="VIS")
    assert g.organism == "Homo sapiens"
    assert g.num_neurons == 151
    assert g.num_synapses > 1000
    valid, violations = g.validate_dales_principle()
    assert valid is True


def test_load_human_h01():
    g = load_human_h01_microcircuit(max_neurons=100)
    assert g.organism == "Homo sapiens"
    assert g.num_neurons == 100
    valid, violations = g.validate_dales_principle()
    assert valid is True
