"""
COMPUTATIONAL COUPLING --- Task-independent dynamical probes of a reservoir.

Task accuracy is a blunt instrument for detecting structural effects: a readout can
compensate for a great deal, and a benchmark can saturate.  These three measures probe
the reservoir's dynamics directly and are the standard currency of reservoir theory, so
a structural property that matters should show up here even when its effect on
end-task accuracy is small.  Reporting them alongside the task result guards against
concluding "no effect" from an insensitive measure.

* **Linear memory capacity** (Jaeger): how much of the recent input history is linearly
  recoverable from the state, summed over delays.
* **Kernel rank** (Legenstein & Maass): the effective dimensionality of the state
  response to *distinct* inputs -- the reservoir's capacity to separate.
* **Generalisation rank**: the same measure for *similar* inputs -- low is good, since
  it means noise is not being expanded into new dimensions.  The difference
  ``kernel - generalisation`` is the standard computational-capability index.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DynamicsResult:
    memory_capacity: float
    kernel_rank: float
    generalisation_rank: float
    capability: float
    mean_rate: float

    def as_dict(self) -> Dict[str, float]:
        return dict(
            memory_capacity=self.memory_capacity,
            kernel_rank=self.kernel_rank,
            generalisation_rank=self.generalisation_rank,
            capability=self.capability,
            mean_rate=self.mean_rate,
        )


def _run(W_rec: np.ndarray, W_in: np.ndarray, u: np.ndarray,
         leak: float, thresh: float) -> np.ndarray:
    """Simulate the LIF reservoir on a scalar input stream; returns states (T, N)."""
    n = W_rec.shape[0]
    T = len(u)
    v = np.zeros(n, dtype=np.float32)
    s = np.zeros(n, dtype=np.float32)
    trace = np.zeros(n, dtype=np.float32)
    out = np.empty((T, n), dtype=np.float32)
    Wr = W_rec.astype(np.float32)
    Wi = W_in.astype(np.float32).ravel()
    for t in range(T):
        I = u[t] * Wi + s @ Wr
        v = leak * v * (1.0 - s) + I
        s = (v >= thresh).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        # Low-pass the spike train so a linear readout has a graded state to work with.
        trace = 0.8 * trace + s
        out[t] = trace
    return out


def memory_capacity(
    W_rec: np.ndarray,
    W_in: np.ndarray,
    max_delay: int = 30,
    n_steps: int = 2000,
    washout: int = 200,
    leak: float = 0.9,
    thresh: float = 1.0,
    seed: int = 0,
    ridge: float = 1e-6,
) -> float:
    r"""Jaeger's linear memory capacity, ``MC = sum_k r^2(u_{t-k}, \hat u_{t-k})``."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=n_steps + max_delay).astype(np.float32)
    X = _run(W_rec, W_in, u[max_delay:], leak, thresh)[washout:]
    X = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)

    G = X.T @ X + ridge * np.eye(X.shape[1], dtype=np.float32)
    try:
        Ginv_Xt = np.linalg.solve(G, X.T)
    except np.linalg.LinAlgError:
        return 0.0

    mc = 0.0
    for k in range(1, max_delay + 1):
        target = u[max_delay - k: max_delay - k + n_steps][washout:]
        w = Ginv_Xt @ target
        pred = X @ w
        vt = target.var()
        if vt <= 1e-12:
            continue
        r = np.corrcoef(pred, target)[0, 1]
        if np.isfinite(r):
            mc += float(r ** 2)
    return mc


def separation_ranks(
    W_rec: np.ndarray,
    W_in: np.ndarray,
    n_streams: int = 200,
    n_steps: int = 220,
    washout: int = 20,
    jitter: float = 0.01,
    leak: float = 0.9,
    thresh: float = 1.0,
    seed: int = 0,
) -> tuple:
    """Kernel and generalisation rank from the final states of many input streams."""
    rng = np.random.default_rng(seed)
    states_distinct = []
    states_similar = []
    base = rng.uniform(-1.0, 1.0, size=n_steps).astype(np.float32)
    for i in range(n_streams):
        u = rng.uniform(-1.0, 1.0, size=n_steps).astype(np.float32)
        states_distinct.append(_run(W_rec, W_in, u, leak, thresh)[-1])
        v = base + rng.normal(0.0, jitter, size=n_steps).astype(np.float32)
        states_similar.append(_run(W_rec, W_in, v, leak, thresh)[-1])
    Kd = np.stack(states_distinct)
    Ks = np.stack(states_similar)
    # Effective rank at a tolerance scaled to the largest singular value.
    def erank(M):
        s = np.linalg.svd(M, compute_uv=False)
        return float((s > s.max() * 1e-3).sum()) if s.size and s.max() > 0 else 0.0
    return erank(Kd), erank(Ks)


def probe(
    W_rec: np.ndarray,
    W_in: np.ndarray,
    leak: float = 0.9,
    thresh: float = 1.0,
    seed: int = 0,
    max_delay: int = 30,
) -> DynamicsResult:
    """Run all three probes on one reservoir."""
    mc = memory_capacity(W_rec, W_in, max_delay=max_delay, leak=leak,
                         thresh=thresh, seed=seed)
    kr, gr = separation_ranks(W_rec, W_in, leak=leak, thresh=thresh, seed=seed)
    rng = np.random.default_rng(seed + 1)
    u = rng.uniform(-1.0, 1.0, size=500).astype(np.float32)
    n = W_rec.shape[0]
    v = np.zeros(n, dtype=np.float32); s = np.zeros(n, dtype=np.float32)
    spikes = 0.0
    Wr = W_rec.astype(np.float32); Wi = W_in.astype(np.float32).ravel()
    for t in range(len(u)):
        I = u[t] * Wi + s @ Wr
        v = leak * v * (1.0 - s) + I
        s = (v >= thresh).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        spikes += float(s.sum())
    return DynamicsResult(
        memory_capacity=mc, kernel_rank=kr, generalisation_rank=gr,
        capability=kr - gr, mean_rate=spikes / (n * len(u)),
    )
