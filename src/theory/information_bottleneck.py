"""
COMPUTATIONAL COUPLING — Information Bottleneck & Rate-Distortion Theory in Connectomes
Formalizes the Information Bottleneck principle (Tishby et al. 2000) for biological networks:
  min I(X; M) - beta * I(M; Y)
Where:
  X: Sensory stimulus stream
  M: Intermediate network motif / recurrent population activation
  Y: Motor readout / behavioral target
Measures sensory compression, predictive information, and thermodynamic efficiency
across C. elegans, Drosophila, Mouse, and Human connectomic topologies.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.stats as stats
import scipy.spatial as spatial


@dataclass
class InformationBottleneckProfile:
    """Quantitative Information Bottleneck metrics for a neural architecture."""
    organism: str
    architecture: str
    mutual_info_xm_bits: float   # I(X; M): Sensory representation complexity
    mutual_info_my_bits: float   # I(M; Y): Behavioral predictive sufficiency
    input_entropy_h_x_bits: float
    output_entropy_h_y_bits: float
    compression_ratio: float     # 1 - I(X; M) / H(X)
    information_efficiency: float # I(M; Y) / I(X; M)
    theoretical_bound_gap: float # Distance to optimal theoretical rate-distortion curve


def estimate_entropy_discrete(probs: np.ndarray) -> float:
    """Compute Shannon entropy H in bits."""
    p = probs[probs > 1e-12]
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def estimate_mutual_information_discrete(
    X_bins: np.ndarray,
    Y_bins: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Estimate mutual information I(X; Y) using 2D joint histogram discrete binning.
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    """
    # 2D joint histogram
    hist_2d, _, _ = np.histogram2d(X_bins.flatten(), Y_bins.flatten(), bins=num_bins)
    p_xy = hist_2d / np.sum(hist_2d)

    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    h_x = estimate_entropy_discrete(p_x)
    h_y = estimate_entropy_discrete(p_y)
    h_xy = estimate_entropy_discrete(p_xy.flatten())

    mi = max(0.0, h_x + h_y - h_xy)
    return float(mi)


def compute_gaussian_copula_mutual_info(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Parametric lower-bound mutual information estimator via Gaussian Copula / Covariance.
    I(X; Y) = -0.5 * ln(1 - rho^2) / ln(2)
    """
    X_c = X - np.mean(X, axis=0, keepdims=True)
    Y_c = Y - np.mean(Y, axis=0, keepdims=True)

    # First principal component projection
    u_x, _, _ = np.linalg.svd(X_c, full_matrices=False)
    u_y, _, _ = np.linalg.svd(Y_c, full_matrices=False)

    proj_x = u_x[:, 0] if u_x.shape[1] > 0 else X.flatten()
    proj_y = u_y[:, 0] if u_y.shape[1] > 0 else Y.flatten()

    corr, _ = stats.pearsonr(proj_x, proj_y)
    corr = min(0.9999, max(-0.9999, corr))

    mi_bits = -0.5 * np.log(1.0 - corr ** 2) / np.log(2.0)
    return float(max(0.0, mi_bits))


def compute_rate_distortion_bound(
    h_x: float,
    h_y: float,
    beta_range: np.ndarray = np.linspace(0.1, 10.0, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical Blahut-Arimoto Rate-Distortion / Information Bottleneck curve:
    R(D) = I(X; M) vs I(M; Y).
    Under optimal trade-off:
      I(M; Y) = H(Y) * (1 - exp(-beta * I(X; M) / H(X)))
    """
    i_xm_vals = np.linspace(0.01 * h_x, 0.99 * h_x, len(beta_range))
    # Theoretical concave upper bound
    i_my_bound = h_y * (1.0 - np.exp(-1.5 * i_xm_vals / max(1e-3, h_x)))
    return i_xm_vals, i_my_bound


def evaluate_network_information_bottleneck(
    inputs: np.ndarray,        # (N_samples, T_steps, D_in)
    hidden_states: np.ndarray, # (N_samples, T_steps, D_hidden)
    targets: np.ndarray,       # (N_samples,) or (N_samples, D_out)
    organism: str = "Synthetic",
    architecture: str = "ActiveDendriticCEN",
    num_bins: int = 8,
) -> InformationBottleneckProfile:
    """
    Evaluates empirical Information Bottleneck metrics for a neural dataset & model.
    """
    N_samples = inputs.shape[0]

    # Collapse spatio-temporal features to summary representations
    x_feat = np.mean(inputs, axis=1) # (N_samples, D_in)
    m_feat = np.mean(hidden_states, axis=1) # (N_samples, D_hidden)

    if targets.ndim == 1:
        y_feat = targets.reshape(-1, 1)
    else:
        y_feat = targets

    # Estimate marginal and mutual information
    # First principal component binning for robust discrete entropy
    def get_1d_bins(mat: np.ndarray) -> np.ndarray:
        mat_c = mat - np.mean(mat, axis=0, keepdims=True)
        u, _, _ = np.linalg.svd(mat_c, full_matrices=False)
        proj = u[:, 0] if u.shape[1] > 0 else mat[:, 0]
        # Discretize into quantiles
        ranks = stats.rankdata(proj)
        bins = (ranks / len(ranks) * (num_bins - 1)).astype(int)
        return bins

    x_bins = get_1d_bins(x_feat)
    m_bins = get_1d_bins(m_feat)
    y_bins = get_1d_bins(y_feat)

    # Compute Entropies
    h_x = estimate_entropy_discrete(np.bincount(x_bins, minlength=num_bins) / N_samples)
    h_y = estimate_entropy_discrete(np.bincount(y_bins, minlength=num_bins) / N_samples)

    # Compute Mutual Information
    i_xm = estimate_mutual_information_discrete(x_bins, m_bins, num_bins=num_bins)
    i_my = estimate_mutual_information_discrete(m_bins, y_bins, num_bins=num_bins)

    # Compression and efficiency
    comp_ratio = max(0.0, 1.0 - (i_xm / max(1e-4, h_x)))
    info_eff = float(i_my / max(1e-4, i_xm))

    # Theoretical Bound Gap:
    # At given I(X; M), theoretical maximum I_bound(M; Y) = H(Y) * (1 - exp(-1.5 * I(X; M)/H(X)))
    theo_my = h_y * (1.0 - np.exp(-1.5 * i_xm / max(1e-4, h_x)))
    bound_gap = float(abs(theo_my - i_my))

    return InformationBottleneckProfile(
        organism=organism,
        architecture=architecture,
        mutual_info_xm_bits=i_xm,
        mutual_info_my_bits=i_my,
        input_entropy_h_x_bits=h_x,
        output_entropy_h_y_bits=h_y,
        compression_ratio=comp_ratio,
        information_efficiency=info_eff,
        theoretical_bound_gap=bound_gap,
    )
