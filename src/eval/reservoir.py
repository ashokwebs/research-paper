"""
COMPUTATIONAL COUPLING --- Connectome reservoir evaluation harness.

The recurrent weight matrix is taken directly from a (real or surrogate) connectome
and is never trained; only a ridge readout is fitted.  This is the conn2res protocol
(Suarez et al. 2024) and is what makes a structural comparison interpretable: any
difference in task performance between network families is attributable to wiring,
because nothing else differs.

Controls that the comparison depends on
--------------------------------------
* **Matched spectral radius.**  Every network is rescaled to a common spectral
  radius before simulation.  Without this, families differ in effective gain and
  the comparison measures normalisation rather than topology.
* **Shared input projection.**  All families in a comparison receive the same
  seeded input matrix, so input coding is held fixed.
* **Matched size and edge count.**  Enforced by construction in the null ladder.
* **Matched readout capacity.**  The same ridge regulariser grid is searched for
  every family, selected on a held-out validation split, never on test.

Why input and readout populations are disjoint
----------------------------------------------
If every neuron receives the stimulus and every neuron feeds the readout, the task
is solved by the input projection and the per-neuron leak alone: a reservoir with
its recurrent matrix zeroed scores exactly as well as any connectome, and the
comparison measures nothing.  We therefore drive a small *sensory* subset of
neurons and read out from a **disjoint** *motor* subset, so that task-relevant
signal must traverse the recurrent graph to be decodable at all.  The zero-matrix
control then collapses to chance, which is the diagnostic that the benchmark is
actually sensitive to topology.  The same node index sets are used for every
network family in a comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ReservoirConfig:
    """Discrete-time LIF reservoir parameters."""

    spectral_radius: float = 0.95
    leak: float = 0.90              # membrane retention per step
    v_threshold: float = 1.0
    v_reset: float = 0.0
    input_scale: float = 1.0
    input_density: float = 0.50     # fraction of (channel, sensory neuron) connections
    frac_input_nodes: float = 0.15  # sensory population size
    frac_readout_nodes: float = 0.35  # motor population size, disjoint from sensory
    disjoint_io: bool = True        # False reproduces the naive all-to-all protocol
    refractory: int = 0
    readout: str = "rate+final"     # features handed to the ridge readout
    ridge_alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
    seed: int = 0


@dataclass
class ReservoirResult:
    accuracy: float
    val_accuracy: float
    synops_per_sample: float
    spikes_per_sample: float
    mean_rate: float
    alpha: float
    n_neurons: int
    n_edges: int
    spectral_radius_pre: float
    extra: Dict[str, float] = field(default_factory=dict)


def signed_matrix(A, weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Dense signed recurrent matrix ``W[i, j]`` = weight of edge i -> j."""
    W = np.zeros((A.n, A.n), dtype=np.float64)
    for u, v in A.iter_edges():
        W[u, v] = float(A.sign[u]) if A.sign[u] != 0 else 1.0
    return W


def rescale_spectral_radius(W: np.ndarray, target: float) -> Tuple[np.ndarray, float]:
    """Rescale ``W`` to a target spectral radius; returns the rescaled matrix and the original radius."""
    ev = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(ev))) if ev.size else 0.0
    if rho <= 1e-12:
        return W.copy(), rho
    return W * (target / rho), rho


def sensory_motor_split(n_neurons: int, cfg: ReservoirConfig,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sensory (driven) and motor (decoded) node index sets.

    Disjoint by default, which is what makes the benchmark sensitive to recurrent
    structure: task-relevant signal has to traverse the graph to be decodable.  With
    ``cfg.disjoint_io = False`` the two populations are drawn independently and may
    overlap, which reproduces the naive all-to-all protocol used in earlier work and in
    much of the literature -- the regime in which a zeroed recurrent matrix scores as
    well as any connectome.

    Disjointness needs ``frac_input_nodes + frac_readout_nodes <= 1``.  Asking for more
    than the whole population used to return an *empty* readout set by slicing past the
    end of the permutation, so the "all-to-all" control decoded nothing and scored
    chance, which looks exactly like the opposite of the effect it was built to show.
    We raise instead.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n_in = max(1, int(round(cfg.frac_input_nodes * n_neurons)))
    n_out = max(1, int(round(cfg.frac_readout_nodes * n_neurons)))

    if not cfg.disjoint_io:
        return (np.sort(rng.permutation(n_neurons)[:n_in]),
                np.sort(rng.permutation(n_neurons)[:n_out]))

    if n_in + n_out > n_neurons:
        raise ValueError(
            f"disjoint sensory/motor populations need frac_input_nodes + "
            f"frac_readout_nodes <= 1, got {cfg.frac_input_nodes} + "
            f"{cfg.frac_readout_nodes} on {n_neurons} neurons "
            f"({n_in} + {n_out}). Set disjoint_io=False for the all-to-all protocol."
        )
    perm = rng.permutation(n_neurons)
    return np.sort(perm[:n_in]), np.sort(perm[n_in:n_in + n_out])


def make_input_matrix(n_channels: int, n_neurons: int, cfg: ReservoirConfig,
                      seed: Optional[int] = None,
                      input_nodes: Optional[np.ndarray] = None) -> np.ndarray:
    """Random input projection onto the sensory population only.

    Columns outside the sensory set are exactly zero, so the stimulus enters the
    network at a restricted set of nodes and must propagate from there.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    if input_nodes is None:
        input_nodes, _ = sensory_motor_split(n_neurons, cfg, seed=seed)
    W = np.zeros((n_channels, n_neurons), dtype=np.float64)
    sub = rng.normal(0.0, 1.0, size=(n_channels, len(input_nodes)))
    sub *= (rng.random((n_channels, len(input_nodes))) < cfg.input_density)
    W[:, input_nodes] = sub
    return W * cfg.input_scale


def simulate(
    X: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray,
    cfg: ReservoirConfig,
    batch_size: int = 512,
    readout_nodes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Run the LIF reservoir over a spike tensor.

    Parameters
    ----------
    X : ndarray (n_samples, n_steps, n_channels)
        Input spike counts.

    Returns
    -------
    features : ndarray (n_samples, n_features)
    synops_per_sample : float
        Total synaptic operations, counted as sum over time of the out-degree of
        every neuron that spiked -- the standard event-driven energy proxy.
    spikes_per_sample : float
    """
    n_samples, n_steps, _ = X.shape
    n_neurons = W_rec.shape[0]
    out_degree = (W_rec != 0).sum(axis=1).astype(np.float64)

    feats = []
    total_synops = 0.0
    total_spikes = 0.0

    for start in range(0, n_samples, batch_size):
        xb = X[start:start + batch_size].astype(np.float32)
        B = xb.shape[0]
        v = np.zeros((B, n_neurons), dtype=np.float32)
        s = np.zeros((B, n_neurons), dtype=np.float32)
        rate = np.zeros((B, n_neurons), dtype=np.float32)
        Win32 = W_in.astype(np.float32)
        Wrec32 = W_rec.astype(np.float32)

        for t in range(n_steps):
            I = xb[:, t, :] @ Win32 + s @ Wrec32
            v = cfg.leak * v * (1.0 - s) + I
            s = (v >= cfg.v_threshold).astype(np.float32)
            v = np.where(s > 0, cfg.v_reset, v)
            rate += s
            total_synops += float((s @ out_degree.astype(np.float32)).sum())
            total_spikes += float(s.sum())

        rate /= n_steps
        r_sel = rate if readout_nodes is None else rate[:, readout_nodes]
        v_sel = v if readout_nodes is None else v[:, readout_nodes]
        if cfg.readout == "rate":
            f = r_sel
        elif cfg.readout == "rate+final":
            f = np.concatenate([r_sel, v_sel], axis=1)
        else:
            raise ValueError(cfg.readout)
        feats.append(f)

    features = np.concatenate(feats, axis=0)
    return features, total_synops / n_samples, total_spikes / n_samples




def _representation_geometry(F: np.ndarray, n_rate_cols: int) -> Dict[str, float]:
    """How many dimensions the reservoir's representation actually occupies, and why.

    A linear readout can only use variance that is spread across directions. Matrix rank
    counts directions with *any* variance and so is nearly blind to a representation
    whose variance is concentrated in a handful of them; the participation ratio

        PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

    over the eigenvalues of the feature covariance counts them by how much variance each
    carries, and is the standard measure of effective dimensionality for exactly this
    reason. It equals the number of features when variance is spread evenly and falls to
    one when a single direction dominates.

    ``saturated_frac`` and ``dead_frac`` record the two ways a spiking reservoir loses
    dimensions: neurons pinned on, and neurons that never fire. Both are per-neuron rates
    over the evaluation set, so a representation can be diagnosed rather than just
    scored.
    """
    Fc = F - F.mean(axis=0, keepdims=True)
    # Eigenvalues of the covariance, via singular values of the centred design.
    sv = np.linalg.svd(Fc, compute_uv=False)
    lam = sv.astype(np.float64) ** 2
    tot = lam.sum()
    pr = float((tot ** 2) / np.sum(lam ** 2)) if tot > 0 else 0.0
    # The leading block of the feature matrix is the time-averaged firing rate of each
    # readout neuron, so per-neuron saturation is read straight off it.
    rates = F[:, :n_rate_cols].mean(axis=0) if n_rate_cols > 0 else np.empty(0)
    return {
        "participation_ratio": pr,
        "participation_ratio_frac": float(pr / max(F.shape[1], 1)),
        "saturated_frac": float((rates > 0.9).mean()) if rates.size else float("nan"),
        "dead_frac": float((rates < 0.01).mean()) if rates.size else float("nan"),
    }


def _ridge_fit(F: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge, fitted on centred features so the intercept is separable.

    The intercept must not be penalised, but appending an unpenalised column of ones to
    the design makes the Gram matrix exactly singular whenever any feature is constant --
    a constant column is collinear with the intercept. That happens for real: at high
    gain the reservoir saturates, every neuron fires at every step, the rate features go
    to one, and the solve raises ``LinAlgError: Singular matrix``. Centring the features
    removes the collinearity by construction, because the centred design is orthogonal to
    the constant column, and the intercept is then recovered in closed form.

    Rank deficiency in the centred design is still possible and is still fine: ridge with
    ``alpha > 0`` is well posed on it. Saturation is a property of the operating point
    worth knowing about rather than an error, so it is reported through the feature rank
    recorded by :func:`evaluate_network` rather than raised here.
    """
    f_mean = F.mean(axis=0, keepdims=True)
    y_mean = Y.mean(axis=0, keepdims=True)
    Fc = F - f_mean
    Yc = Y - y_mean

    G = Fc.T @ Fc
    G.flat[:: G.shape[0] + 1] += alpha
    W = np.linalg.solve(G, Fc.T @ Yc)
    intercept = y_mean - f_mean @ W
    return np.concatenate([W, intercept], axis=0)


def _ridge_predict(F: np.ndarray, Wout: np.ndarray) -> np.ndarray:
    Fb = np.concatenate([F, np.ones((F.shape[0], 1), dtype=F.dtype)], axis=1)
    return Fb @ Wout


def evaluate_network(
    A,
    train,
    test,
    cfg: ReservoirConfig,
    W_in: Optional[np.ndarray] = None,
    val_fraction: float = 0.2,
    val_seed: int = 12345,
    node_seed: Optional[int] = None,
) -> ReservoirResult:
    """Fit a ridge readout on one network and report held-out accuracy and energy."""
    W = signed_matrix(A)
    W, rho_pre = rescale_spectral_radius(W, cfg.spectral_radius)

    input_nodes, readout_nodes = sensory_motor_split(A.n, cfg, seed=node_seed)
    if W_in is None:
        W_in = make_input_matrix(train.X.shape[2], A.n, cfg,
                                 seed=node_seed, input_nodes=input_nodes)

    F_tr, synops_tr, spk_tr = simulate(train.X, W, W_in, cfg, readout_nodes=readout_nodes)
    F_te, synops_te, spk_te = simulate(test.X, W, W_in, cfg, readout_nodes=readout_nodes)

    n_classes = int(max(train.y.max(), test.y.max())) + 1
    Y = np.eye(n_classes, dtype=np.float64)[train.y]

    rng = np.random.default_rng(val_seed)
    idx = rng.permutation(len(train.y))
    n_val = int(val_fraction * len(idx))
    vi, ti = idx[:n_val], idx[n_val:]

    # Standardise features using training-fold statistics only.
    mu = F_tr[ti].mean(axis=0)
    sd = F_tr[ti].std(axis=0) + 1e-8
    Ftr = ((F_tr - mu) / sd).astype(np.float64)
    Fte = ((F_te - mu) / sd).astype(np.float64)

    best = (-1.0, cfg.ridge_alphas[0], None)
    for a in cfg.ridge_alphas:
        Wout = _ridge_fit(Ftr[ti], Y[ti], a)
        pred = _ridge_predict(Ftr[vi], Wout).argmax(axis=1)
        acc = float((pred == train.y[vi]).mean())
        if acc > best[0]:
            best = (acc, a, Wout)
    val_acc, alpha, _ = best

    # Refit on the full training split at the selected regulariser.
    Wout = _ridge_fit(Ftr, Y, alpha)
    pred = _ridge_predict(Fte, Wout).argmax(axis=1)
    acc = float((pred == test.y).mean())

    n_edges = int((W != 0).sum())
    return ReservoirResult(
        accuracy=acc,
        val_accuracy=val_acc,
        synops_per_sample=float(synops_te),
        spikes_per_sample=float(spk_te),
        mean_rate=float(spk_te / (A.n * test.X.shape[1])),
        alpha=float(alpha),
        n_neurons=A.n,
        n_edges=n_edges,
        spectral_radius_pre=rho_pre,
        extra={"n_input_nodes": float(len(input_nodes)),
               "n_readout_nodes": float(len(readout_nodes)),
               # Effective rank of the readout features, as a fraction of their number.
               # It falls towards zero as the reservoir saturates and every neuron fires
               # at every step, which is a property of the operating point rather than a
               # failure: a comparison made where the state carries no information is
               # not a comparison of topologies. Recorded so that regime can be excluded
               # by name instead of being noticed as a strange accuracy.
               "feature_rank_frac": float(
                   np.linalg.matrix_rank(Fte - Fte.mean(axis=0, keepdims=True))
                   / max(Fte.shape[1], 1)),
               **_representation_geometry(Fte, len(readout_nodes)),
               },
    )


def calibrate_gain_for_rate(
    A,
    X: np.ndarray,
    cfg: ReservoirConfig,
    target_rate: float,
    W_in: Optional[np.ndarray] = None,
    node_seed: Optional[int] = None,
    rho_lo: float = 0.2,
    rho_hi: float = 6.0,
    tol: float = 0.05,
    max_iter: int = 14,
) -> Tuple[float, float]:
    """Find the spectral radius at which a network fires at a target mean rate.

    Matching spectral radius is *not* sufficient to make a structural comparison fair.
    At equal spectral radius, different topologies settle at different activity levels
    -- in our measurements an Erdos-Renyi surrogate fires roughly twice as often as the
    connectome it is matched to -- and firing rate is itself correlated with task
    accuracy and determines synaptic-operation count directly.  A comparison at equal
    spectral radius therefore confounds topology with operating point.

    This routine calibrates each network individually to a common firing rate by
    bisection on the spectral radius, so that the topology comparison can be repeated
    at a matched operating point.  Returns ``(spectral_radius, achieved_rate)``.
    """
    W0 = signed_matrix(A)
    input_nodes, readout_nodes = sensory_motor_split(A.n, cfg, seed=node_seed)
    if W_in is None:
        W_in = make_input_matrix(X.shape[2], A.n, cfg, seed=node_seed,
                                 input_nodes=input_nodes)

    def rate_at(rho: float) -> float:
        W, _ = rescale_spectral_radius(W0, rho)
        _, _, spikes = simulate(X, W, W_in, cfg, readout_nodes=readout_nodes)
        return spikes / (A.n * X.shape[1])

    lo, hi = rho_lo, rho_hi
    r_lo, r_hi = rate_at(lo), rate_at(hi)
    # If the target lies outside the achievable band, return the closer endpoint.  The
    # caller must check the returned rate against the target: a network whose activity
    # saturates below the target -- a strongly inhibitory circuit, for instance, where
    # raising the gain raises inhibition too -- is *not* rate-matched, and averaging it
    # in with the matched ones reports a comparison that was never made.
    if r_lo > target_rate:
        return lo, r_lo
    if r_hi < target_rate:
        return hi, r_hi

    best = (hi, r_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r = rate_at(mid)
        best = (mid, r)
        if abs(r - target_rate) <= tol * target_rate:
            break
        if r < target_rate:
            lo = mid
        else:
            hi = mid
    return best
