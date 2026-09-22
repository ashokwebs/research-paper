"""
COMPUTATIONAL COUPLING --- The graded-rate reservoir, for comparison against the spiking one.

Why this module exists
----------------------
The connectome-reservoir literature and this project disagree about whether connectome
topology helps, and they use different units.  Suarez et al. evaluate an echo state
network of ``tanh`` units; we evaluate a spiking leaky integrate-and-fire network.  Those
are not two implementations of one model.  A ``tanh`` unit responds to any input, however
small, and is never silent; a LIF unit below threshold emits nothing and contributes no
variance at all.

That difference interacts with degree heterogeneity in a way that could reverse the sign
of the whole comparison, so it cannot be left as an incidental implementation choice.
This module implements the rate model exactly as ``conn2res`` does, so the two unit
classes can be run on identical graphs, identical nulls, identical inputs and identical
readouts, and the unit model isolated as the only difference.

The update is theirs:

    x[t] = tanh( x[t-1] W + u[t-1] W_in )                              (no leak)
    x[t] = (1-a) x[t-1] + a tanh( x[t-1] W + u[t-1] W_in )             (leak a)

with the recurrent matrix rescaled to a target spectral radius, which is the alpha the
published comparison is made at.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RateConfig:
    """Echo state network parameters, matching the conn2res defaults we compare against."""

    spectral_radius: float = 1.0
    leak_rate: Optional[float] = None   # None reproduces the memoryless tanh update
    input_gain: float = 1.0
    washout: int = 200


def rescale(W: np.ndarray, target: float) -> np.ndarray:
    """Rescale to a target spectral radius; a nilpotent matrix is returned unchanged."""
    ev = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(ev))) if ev.size else 0.0
    return W.copy() if rho <= 1e-12 else W * (target / rho)


def simulate_rate(
    W: np.ndarray,
    W_in: np.ndarray,
    u: np.ndarray,
    cfg: RateConfig,
) -> np.ndarray:
    """Run the ESN over a scalar input stream; returns states of shape ``(T, N)``."""
    T = len(u)
    n = W.shape[0]
    X = np.zeros((T + 1, n), dtype=np.float64)
    Wi = W_in.reshape(1, -1) * cfg.input_gain
    a = cfg.leak_rate
    for t in range(1, T + 1):
        drive = X[t - 1] @ W + u[t - 1] * Wi[0]
        act = np.tanh(drive)
        X[t] = act if a is None else (1.0 - a) * X[t - 1] + a * act
    return X[1:]


def memory_capacity_rate(
    W: np.ndarray,
    W_in: np.ndarray,
    cfg: RateConfig,
    max_delay: int = 30,
    n_steps: int = 2000,
    seed: int = 0,
    ridge: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    r"""Jaeger's linear memory capacity, ``MC = sum_k r^2(u_{t-k}, \hat u_{t-k})``.

    Deliberately the same estimator, delays, washout and ridge as the spiking path, so
    that a difference between the two is a difference between the unit models and not
    between two ways of measuring memory.

    Returns the capacity and the same representation diagnostics the spiking harness
    records, so the two can be compared on identical quantities. ``dead_frac`` is
    included even though a ``tanh`` unit is essentially never exactly silent --- that it
    comes out near zero here, while the spiking network's is large, is the comparison.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=n_steps + max_delay)
    X = simulate_rate(W, W_in, u[max_delay:], cfg)[cfg.washout:]
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)

    G = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    try:
        Ginv_Xt = np.linalg.solve(G, Xb.T)
    except np.linalg.LinAlgError:
        return 0.0, {}

    mc = 0.0
    for k in range(1, max_delay + 1):
        target = u[max_delay - k: max_delay - k + n_steps][cfg.washout:]
        if target.var() <= 1e-12:
            continue
        pred = Xb @ (Ginv_Xt @ target)
        r = np.corrcoef(pred, target)[0, 1]
        if np.isfinite(r):
            mc += float(r ** 2)

    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    pr = float((tot ** 2) / np.sum(lam ** 2)) if tot > 0 else 0.0
    # "Active" for a graded unit means its state actually varies; a unit whose output is
    # constant carries no more information than a silent spiking one.
    sd = X.std(axis=0)
    return mc, {
        "participation_ratio": pr,
        "participation_ratio_frac": float(pr / max(X.shape[1], 1)),
        "dead_frac": float((sd < 1e-4).mean()),
        "mean_abs_state": float(np.abs(X).mean()),
        "saturated_frac": float((np.abs(X) > 0.99).mean()),
    }
