"""
Unit tests for information-theoretic and Landauer thermodynamic efficiency modules.
"""

import numpy as np
import pytest
from src.theory.thermodynamics import (
    compute_mutual_information,
    evaluate_thermodynamic_efficiency,
    run_thermodynamic_cross_architecture_comparison,
    LANDAUER_LIMIT_JOULES,
)


def test_mutual_information_perfect_and_random():
    # Perfect classification: 4 classes
    y_true = np.array([0, 1, 2, 3] * 25)
    y_pred_perf = y_true.copy()
    mi_perf = compute_mutual_information(y_true, y_pred_perf, num_classes=4)
    # Shannon entropy of uniform 4 classes is log2(4) = 2.0 bits
    assert np.isclose(mi_perf, 2.0, atol=1e-3)

    # Random guessing
    np.random.seed(42)
    y_pred_rand = np.random.randint(0, 4, size=len(y_true))
    mi_rand = compute_mutual_information(y_true, y_pred_rand, num_classes=4)
    assert mi_rand < 0.20  # Near 0 bits


def test_evaluate_thermodynamic_efficiency_landauer():
    y_true = np.array([0, 1] * 50)
    y_pred = y_true.copy()
    energy_nj = 10.0

    metrics = evaluate_thermodynamic_efficiency(
        model_name="TestModel",
        accuracy_pct=100.0,
        y_true=y_true,
        y_pred=y_pred,
        energy_proxy_nj=energy_nj,
    )

    assert metrics.accuracy_pct == 100.0
    assert metrics.mutual_information_bits > 0.99
    assert metrics.bits_per_nanojoule > 0.09
    assert metrics.landauer_proximity_ratio > 1.0  # Must be above Landauer thermodynamic minimum


def test_cross_architecture_thermodynamic_comparison():
    results = run_thermodynamic_cross_architecture_comparison()
    assert len(results) >= 4

    names = [r.model_name for r in results]
    assert any("ActiveDendriticCEN" in name for name in names)
    assert any("CoupledEventNet" in name for name in names)
    assert any("Dense SNN" in name for name in names)

    # Verify that ActiveDendriticCEN achieves highest bit-per-nJ efficiency
    ad_cen = next(r for r in results if "ActiveDendriticCEN" in r.model_name)
    dense = next(r for r in results if "Dense SNN" in r.model_name)
    assert ad_cen.bits_per_nanojoule > dense.bits_per_nanojoule * 100.0
