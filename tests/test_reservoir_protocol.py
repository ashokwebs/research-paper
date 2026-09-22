"""
Tests for the reservoir harness, focused on the controls the comparison depends on.

The most important of these is that the sensory and motor populations are disjoint.
Under an all-to-all protocol a zero recurrent matrix performs as well as a connectome,
which silently voids any structural comparison; these tests pin the corrected
behaviour so the protocol cannot regress.
"""

import numpy as np
import pytest

from src.eval.reservoir import (
    ReservoirConfig,
    make_input_matrix,
    rescale_spectral_radius,
    sensory_motor_split,
    signed_matrix,
    simulate,
)
from src.motifs.fast_census import AdjacencySets


def _ring(n=40, sign_period=7, all_excitatory=False):
    succ = [set() for _ in range(n)]
    pred = [set() for _ in range(n)]
    for i in range(n):
        j = (i + 1) % n
        succ[i].add(j)
        pred[j].add(i)
    if all_excitatory:
        sign = np.ones(n, dtype=np.int8)
    else:
        sign = np.array([-1 if (i % sign_period == 0) else 1 for i in range(n)],
                        dtype=np.int8)
    return AdjacencySets(n=n, succ=succ, pred=pred, sign=sign)


def test_sensory_and_motor_populations_are_disjoint():
    cfg = ReservoirConfig()
    for seed in range(5):
        sens, motor = sensory_motor_split(200, cfg, seed=seed)
        assert len(set(sens.tolist()) & set(motor.tolist())) == 0
        assert len(sens) == round(cfg.frac_input_nodes * 200)
        assert len(motor) == round(cfg.frac_readout_nodes * 200)


def test_input_matrix_touches_only_sensory_nodes():
    cfg = ReservoirConfig()
    sens, motor = sensory_motor_split(100, cfg, seed=0)
    W = make_input_matrix(12, 100, cfg, seed=0, input_nodes=sens)
    non_sensory = np.setdiff1d(np.arange(100), sens)
    assert np.all(W[:, non_sensory] == 0.0)
    assert np.any(W[:, sens] != 0.0)


def test_motor_population_is_silent_without_recurrence():
    """The control that licenses every structural comparison in the paper."""
    cfg = ReservoirConfig()
    A = _ring(60)
    sens, motor = sensory_motor_split(A.n, cfg, seed=0)
    W_in = make_input_matrix(8, A.n, cfg, seed=0, input_nodes=sens)
    X = (np.random.default_rng(0).random((16, 20, 8)) < 0.4).astype(np.uint8)

    W_zero = np.zeros((A.n, A.n))
    feats, synops, spikes = simulate(X, W_zero, W_in * 5.0, cfg, readout_nodes=motor)
    # With no recurrence, nothing can reach the motor population.
    assert np.allclose(feats, 0.0)
    assert synops == 0.0

    W = signed_matrix(A)
    W, _ = rescale_spectral_radius(W, 1.3)
    feats_r, synops_r, _ = simulate(X, W, W_in * 5.0, cfg, readout_nodes=motor)
    assert synops_r > 0.0
    assert np.any(feats_r != 0.0)


def test_spectral_radius_rescaling_is_exact():
    A = _ring(50)
    W = signed_matrix(A)
    for target in (0.5, 0.95, 1.3):
        Ws, rho_pre = rescale_spectral_radius(W, target)
        rho = np.max(np.abs(np.linalg.eigvals(Ws)))
        assert rho == pytest.approx(target, rel=1e-6)
        assert rho_pre > 0


def test_signed_matrix_respects_dale():
    A = _ring(30, sign_period=3)
    W = signed_matrix(A)
    for u in range(A.n):
        row = W[u][W[u] != 0]
        if row.size:
            assert np.all(np.sign(row) == np.sign(float(A.sign[u])))


def test_synops_counts_spikes_times_outdegree():
    cfg = ReservoirConfig(v_threshold=0.0)   # every neuron spikes every step
    # All-excitatory ring: with any inhibitory neuron, its target is driven below
    # threshold on the following step and the closed-form count no longer applies.
    A = _ring(10, all_excitatory=True)
    W = signed_matrix(A)
    X = np.zeros((2, 3, 4), dtype=np.uint8)
    W_in = np.zeros((4, A.n))
    _, synops, spikes = simulate(X, W, W_in, cfg)
    # Each of 10 neurons has out-degree 1 and fires at all 3 steps.
    assert spikes == pytest.approx(30.0)
    assert synops == pytest.approx(30.0)


def test_all_to_all_protocol_decodes_every_neuron():
    """The naive protocol must actually decode something.

    Requesting frac_input = frac_readout = 1.0 under the disjoint split used to slice
    past the end of the permutation and return an empty readout set.  The control then
    scored chance -- which reads as "the naive protocol is fine", the exact opposite of
    what it was built to demonstrate.
    """
    cfg = ReservoirConfig(frac_input_nodes=1.0, frac_readout_nodes=1.0,
                          disjoint_io=False)
    inp, out = sensory_motor_split(200, cfg, seed=1)
    assert len(inp) == 200 and len(out) == 200


def test_disjoint_split_refuses_an_impossible_request():
    cfg = ReservoirConfig(frac_input_nodes=0.8, frac_readout_nodes=0.8)
    with pytest.raises(ValueError, match="disjoint"):
        sensory_motor_split(200, cfg, seed=1)


def test_default_split_is_disjoint_and_non_empty():
    inp, out = sensory_motor_split(200, ReservoirConfig(), seed=1)
    assert len(inp) > 0 and len(out) > 0
    assert not (set(inp.tolist()) & set(out.tolist()))


def test_trained_snn_rate_penalty_is_not_in_the_collapse_regime():
    """The trained-ladder default must not make silence the global optimum.

    Under the sensory/motor protocol only ~15% of neurons receive direct drive, so a
    strong L1 firing-rate penalty is minimised by firing nothing at all: every
    condition then scores chance and the table reads as a clean null result. The
    default carried over from the all-to-all harness (1.0) was in that regime.
    """
    from src.eval.trained_snn import TrainConfig

    assert TrainConfig().reg_rate <= 0.1


def test_ridge_fit_survives_degenerate_features():
    """Ridge must return a finite fit when feature columns are constant or dead.

    Honest scope note: the gain sweep died at high rho with "LinAlgError: Singular
    matrix", and centring the features was the fix, but we did not isolate the exact
    input that triggered it -- constant columns, dead columns, identical columns, NaN
    and float32 precision loss were each checked against the old code and none
    reproduced it. What centring does guarantee is that the intercept can no longer be
    collinear with the design, which removes that whole class of singularity by
    construction. This test pins the property, not the original trigger; the sweep now
    also records a failed cell as a row rather than aborting, so if the trigger recurs
    it will be captured with its exact parameters instead of guessed at again.
    """
    from src.eval.reservoir import _ridge_fit, _ridge_predict

    rng = np.random.default_rng(0)
    n, d, k = 200, 6, 3
    F = rng.normal(size=(n, d))
    F[:, 2] = 1.0            # constant feature: collinear with the intercept
    F[:, 5] = 0.0            # and a dead one
    Y = np.eye(k)[rng.integers(0, k, n)]

    W = _ridge_fit(F, Y, alpha=1e-2)
    assert W.shape == (d + 1, k)
    assert np.isfinite(W).all()
    assert np.isfinite(_ridge_predict(F, W)).all()


def test_ridge_fit_matches_sklearn_style_solution_on_well_posed_input():
    """Centring must not change the fit where the old path was already valid."""
    from src.eval.reservoir import _ridge_fit, _ridge_predict

    rng = np.random.default_rng(1)
    n, d, k = 300, 8, 4
    F = rng.normal(size=(n, d))
    Y = np.eye(k)[rng.integers(0, k, n)]
    alpha = 0.5

    W = _ridge_fit(F, Y, alpha)
    # Reference: explicit centred ridge, intercept recovered in closed form.
    fm, ym = F.mean(0, keepdims=True), Y.mean(0, keepdims=True)
    Fc, Yc = F - fm, Y - ym
    Wref = np.linalg.solve(Fc.T @ Fc + alpha * np.eye(d), Fc.T @ Yc)
    ref = np.concatenate([Wref, ym - fm @ Wref], axis=0)
    assert np.allclose(W, ref, atol=1e-10)
    assert np.allclose(_ridge_predict(F, W), _ridge_predict(F, ref), atol=1e-10)
