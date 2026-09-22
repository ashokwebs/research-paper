"""
COMPUTATIONAL COUPLING --- Gap-free perturbation theory for connectome remodelling.

Why not Davis-Kahan
-------------------
Degeneracy -- the capacity of a circuit to be rewired substantially without losing
function -- is naturally posed as a matrix-perturbation question, and the obvious
tool is the Davis-Kahan sin-theta theorem bounding the rotation of an invariant
subspace.  Applied to real connectomes it is vacuous.  Davis-Kahan controls the
rotation of an eigenspace by ``||E|| / delta`` where ``delta`` is the spectral gap
isolating that eigenspace, and measured gaps in connectome spectra are of order
1e-4 or smaller, so the bound permits essentially no perturbation while the
networks empirically tolerate rewiring of 15-25% of their structure.  Reporting
that ratio as a finding ("the empirical radius exceeds the bound by 19,000x")
describes a limitation of the tool, not a property of brains: near-degenerate
spectra have unstable eigenvectors, and a bound on eigenvector rotation therefore
says nothing useful about function.

What actually has to be stable
------------------------------
A reservoir's task behaviour is a function of its finite-horizon input-output map,
not of any individual eigenvector.  For a linear readout over a horizon of ``T``
steps that map is the sequence of Markov parameters

    H_t = W_out A^t W_in,        t = 0, ..., T-1,

and a bound on how ``{H_t}`` moves under ``A -> A + E`` requires **no spectral gap
assumption at all**.  Expanding ``(A+E)^t - A^t`` into its ``t`` first-order terms
and bounding each factor gives Proposition 1 below.  This is elementary, but it is
the right object: it is finite whenever the spectral radius is bounded, it degrades
gracefully rather than discontinuously, and it exposes the quantity that actually
separates benign from damaging perturbations.

Operational alignment
---------------------
The bound of Proposition 1 is worst-case over the *direction* of ``E``.  Real
remodelling is not adversarial.  We therefore define the **operational alignment**

    gamma(E) = ||Delta H||_F / ( C(T, rho) ||E||_F ),      gamma in [0, 1],

the fraction of the worst case that a given perturbation direction actually
realises.  ``C`` is stated in the operator norm; using the spectral radius instead is
valid only for normal matrices and connectomes are not normal.  ``gamma`` is directly measurable, and it converts the qualitative claim
"biological rewiring is non-adversarial" into a number that can be compared across
perturbation strategies.  The falsifiable prediction is that the tolerated
perturbation magnitude scales as ``1 / gamma``: perturbations chosen to maximise
``gamma`` should destroy function at a small fraction of the structural drift that
random or biologically-patterned pruning survives.  :func:`adversarial_edge_ranking`
constructs exactly that maximiser, which is what makes the prediction testable.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Markov parameters and the gap-free bound
# --------------------------------------------------------------------------- #
def markov_parameters(A: np.ndarray, W_in: np.ndarray, W_out: np.ndarray,
                      horizon: int) -> np.ndarray:
    """Return ``H`` of shape (horizon, n_out, n_in) with ``H[t] = W_out A^t W_in``."""
    n_out = W_out.shape[0]
    n_in = W_in.shape[1]
    H = np.empty((horizon, n_out, n_in), dtype=np.float64)
    M = W_in.copy()
    for t in range(horizon):
        H[t] = W_out @ M
        M = A @ M
    return H


def spectral_radius(A: np.ndarray) -> float:
    ev = np.linalg.eigvals(A)
    return float(np.max(np.abs(ev))) if ev.size else 0.0


def worst_case_constant(op_norm: float, horizon: int,
                        norm_in: float, norm_out: float) -> float:
    """``C(T, ||A||)`` of Proposition 1.

    Expanding ``(A+E)^t - A^t = sum_{s<t} A^s E A^(t-1-s) + O(||E||^2)`` and bounding
    each of the ``t`` terms submultiplicatively gives
    ``||Delta H_t||_F <= t ||A||_2^(t-1) ||E||_F ||W_in|| ||W_out||``, so the
    horizon-aggregated constant is
    ``C = ||W_in|| ||W_out|| sqrt( sum_t t^2 ||A||_2^(2t-2) )``.

    The constant is in the **operator norm**, not the spectral radius.  An earlier
    version used ``rho`` here, which requires ``||A|| <= rho`` -- the inequality runs
    the other way for any non-normal matrix, and connectome matrices are strongly
    non-normal (we measure ``||A||_2 / rho`` up to 2.1).  Using ``rho`` understates the
    worst case by up to that ratio raised to the horizon, which would let the measured
    ``gamma`` exceed 1 and break the interpretation of ``gamma`` as a fraction of the
    worst case.

    Like the spectral-radius version it requires no spectral gap and is finite for
    every ``A``.
    """
    t = np.arange(horizon, dtype=np.float64)
    if op_norm <= 0:
        terms = np.where(t == 1, 1.0, 0.0)
    else:
        terms = (t ** 2) * op_norm ** (2 * np.maximum(t - 1, 0))
    return float(norm_in * norm_out * np.sqrt(terms.sum()))


@dataclass
class AlignmentMeasurement:
    """Realised versus worst-case sensitivity of the input-output map."""

    delta_h_norm: float
    e_norm: float
    worst_case: float
    gamma: float
    rho: float
    horizon: int

    def as_dict(self) -> Dict[str, float]:
        return dict(
            delta_h_norm=self.delta_h_norm,
            e_norm=self.e_norm,
            worst_case_bound=self.worst_case,
            gamma=self.gamma,
            spectral_radius=self.rho,
            horizon=self.horizon,
        )


def operational_alignment(
    A: np.ndarray,
    E: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
    horizon: int = 20,
) -> AlignmentMeasurement:
    """Measure ``gamma(E)``: the fraction of the worst-case bound that ``E`` realises."""
    rho = spectral_radius(A)
    op = float(np.linalg.norm(A, 2))
    H0 = markov_parameters(A, W_in, W_out, horizon)
    H1 = markov_parameters(A + E, W_in, W_out, horizon)
    d = float(np.linalg.norm(H1 - H0))
    e = float(np.linalg.norm(E))
    C = worst_case_constant(op, horizon,
                            float(np.linalg.norm(W_in, 2)),
                            float(np.linalg.norm(W_out, 2)))
    gamma = d / (C * e) if C > 0 and e > 0 else 0.0
    return AlignmentMeasurement(
        delta_h_norm=d, e_norm=e, worst_case=C * e, gamma=gamma,
        rho=rho, horizon=horizon,
    )


def davis_kahan_radius(A: np.ndarray, k: int) -> Tuple[float, float]:
    """Spectral gap at rank ``k`` and the perturbation size Davis-Kahan admits.

    Retained only so the paper can report, on identical inputs, how far the
    classical bound is from being usable.
    """
    ev = np.linalg.eigvals(A)
    mag = np.sort(np.abs(ev))[::-1]
    if len(mag) <= k:
        return 0.0, 0.0
    delta = float(mag[k - 1] - mag[k])
    return delta, delta / 2.0


# --------------------------------------------------------------------------- #
# Perturbation strategies
# --------------------------------------------------------------------------- #
def random_edge_ranking(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flat indices of nonzero entries in uniformly random order."""
    idx = np.flatnonzero(W.ravel() != 0)
    rng.shuffle(idx)
    return idx


def magnitude_edge_ranking(W: np.ndarray) -> np.ndarray:
    """Nonzero entries ordered weakest-first (the standard pruning heuristic).

    Only meaningful on a weighted matrix.  On a binary connectome every entry has the
    same magnitude, ``argsort`` falls back to row-major index order, and the resulting
    "weakest-first" schedule silently becomes *delete neuron 0's edges, then neuron
    1's* --- a structured neuron-ablation attack reported under the name of a weight
    heuristic.  We refuse that rather than return it, because it produced a plausible
    and completely misdescribed survival curve.
    """
    flat = W.ravel()
    idx = np.flatnonzero(flat != 0)
    mags = np.abs(flat[idx])
    if np.ptp(mags) <= 1e-12 * max(float(mags.max()), 1.0):
        raise ValueError(
            "weight-magnitude pruning is undefined on an unweighted graph: every "
            "edge has the same magnitude, so the ranking would degenerate to index "
            "order. Use hub_edge_ranking or adversarial_edge_ranking instead."
        )
    return idx[np.argsort(mags)]


def hub_edge_ranking(W: np.ndarray) -> np.ndarray:
    """Nonzero entries ordered by the out-degree of their source node, largest first.

    The structured-but-not-adversarial comparison an unweighted graph admits: it
    concentrates the perturbation on the highest-degree neurons, which is where a
    topology-aware but task-blind attacker would put it.  Ties are broken at random
    within a degree class so the ranking is not secretly an index ordering.
    """
    deg = (W != 0).sum(axis=1).astype(np.float64)
    idx = np.flatnonzero(W.ravel() != 0)
    src = idx // W.shape[1]
    # Deterministic jitter, seeded from the matrix itself, to break degree ties.
    rng = np.random.default_rng(abs(int(deg.sum())) % (2**31))
    key = -deg[src] + rng.random(len(idx)) * 0.5
    return idx[np.argsort(key, kind="stable")]


def adversarial_edge_ranking(
    A: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
    horizon: int = 20,
) -> np.ndarray:
    """Order edges by first-order influence on the input-output map, largest first.

    The exact first-order sensitivity of the horizon-aggregated map to entry
    ``(i, j)`` is obtained by differentiating ``sum_t ||W_out A^t W_in||_F^2``:

        dS/dA_ij = 2 sum_t sum_{s<t} [ (A^s W_in) (H_t)^T W_out A^(t-1-s) ]_ji

    which we evaluate by accumulating the forward states ``A^s W_in`` and the
    adjoint states ``(W_out A^(t-1-s))^T H_t``.  Selecting the largest-magnitude
    entries yields the maximally damaging structured perturbation of a given
    cardinality to first order -- the empirical upper end of ``gamma``.
    """
    n = A.shape[0]
    # Forward states F_s = A^s W_in and adjoint states L_j = W_out A^j, both built
    # by iteration so no matrix power is ever formed explicitly.
    fwd = [W_in.astype(np.float64)]
    for _ in range(horizon):
        fwd.append(A @ fwd[-1])
    adj = [W_out.astype(np.float64)]
    for _ in range(horizon):
        adj.append(adj[-1] @ A)

    grad = np.zeros((n, n), dtype=np.float64)
    for t in range(1, horizon):
        H_t = W_out @ fwd[t]                       # (n_out, n_in)
        for s in range(t):
            left = adj[t - 1 - s]                  # (n_out, n)
            grad += (left.T @ H_t) @ fwd[s].T
    grad *= 2.0

    mask = A != 0
    score = np.where(mask, np.abs(grad), -np.inf)
    idx = np.flatnonzero(mask.ravel())
    order = idx[np.argsort(-score.ravel()[idx])]
    return order


def apply_pruning(W: np.ndarray, ranking: np.ndarray, fraction: float) -> np.ndarray:
    """Zero the first ``fraction`` of the ranked entries; returns the perturbation ``E``."""
    n_cut = int(round(fraction * len(ranking)))
    E = np.zeros_like(W)
    if n_cut:
        flat_E = E.ravel()
        flat_W = W.ravel()
        cut = ranking[:n_cut]
        flat_E[cut] = -flat_W[cut]
    return E


def structural_drift(W: np.ndarray, E: np.ndarray) -> float:
    """Symmetric structural distance used throughout the project.

    Half the Jaccard-style edge-set distance plus half the relative weight change,
    so that a fully rewired network scores 1 and an untouched one scores 0.
    """
    W1 = W + E
    e0 = W != 0
    e1 = W1 != 0
    sym = np.logical_xor(e0, e1).sum()
    denom_e = e0.sum() + e1.sum()
    term_edges = sym / denom_e if denom_e else 0.0
    denom_w = np.abs(W).sum() + np.abs(W1).sum()
    term_w = np.abs(W - W1).sum() / denom_w if denom_w else 0.0
    return float(0.5 * term_edges + 0.5 * term_w)
