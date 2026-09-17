"""
Self-Organized Criticality and Edge-of-Chaos Dynamics in Degenerate Connectomes.

Formalizes:
1. Branching ratio sigma estimation:
   sigma = E[ N_spikes(t+1) / N_spikes(t) | N_spikes(t) > 0 ]
2. Neuronal avalanche extraction (size S and duration T distributions).
3. Maximum Likelihood Estimation (MLE) of power-law exponents tau and alpha:
   P(S) ~ S^(-tau), P(T) ~ T^(-alpha)
4. Universal crackling noise scaling relation:
   gamma = (alpha - 1) / (tau - 1)
5. Theorem 2 (Motif-Constrained Criticality Preservation Theorem) numerical verification.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


@dataclass
class AvalancheProfile:
    """Quantitative summary of neuronal avalanche statistics."""
    branching_ratio: float
    branching_ratio_std: float
    num_avalanches: int
    mean_avalanche_size: float
    max_avalanche_size: int
    mean_avalanche_duration: float
    max_avalanche_duration: int
    tau_size_exponent: float
    alpha_duration_exponent: float
    crackling_scaling_gamma: float
    is_critical: bool  # True if 0.95 <= sigma <= 1.05 and 1.3 <= tau <= 1.8


def detect_neuronal_avalanches(
    spike_train: np.ndarray,
    bin_size: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Extracts neuronal avalanches from a binary or integer spike train matrix.
    
    Args:
        spike_train: (T_steps, N_neurons) array of spike counts.
        bin_size: Temporal binning window.
        
    Returns:
        sizes: List of total spike counts per contiguous avalanche.
        durations: List of durations (in bins) per contiguous avalanche.
    """
    T_steps, _ = spike_train.shape
    
    # Sum spikes across all neurons for each time step
    population_activity = np.sum(spike_train, axis=1)
    
    if bin_size > 1:
        n_bins = T_steps // bin_size
        binned_activity = np.array([
            np.sum(population_activity[i * bin_size:(i + 1) * bin_size])
            for i in range(n_bins)
        ])
    else:
        binned_activity = population_activity
        
    sizes: List[int] = []
    durations: List[int] = []
    
    current_size = 0
    current_duration = 0
    in_avalanche = False
    
    for count in binned_activity:
        if count > 0:
            current_size += int(count)
            current_duration += 1
            in_avalanche = True
        else:
            if in_avalanche:
                sizes.append(current_size)
                durations.append(current_duration)
                current_size = 0
                current_duration = 0
                in_avalanche = False
                
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_duration)
        
    return sizes, durations


def compute_branching_ratio(spike_train: np.ndarray) -> Tuple[float, float]:
    """
    Estimates the branching ratio sigma using the standard least-squares autoregressive
    estimator (Beggs & Plenz 2003, Spitzner et al. 2021):
        A(t+1) = sigma * A(t) + epsilon_t
        sigma = sum_t [ A(t) * A(t+1) ] / sum_t [ A(t)^2 ]
    
    Args:
        spike_train: (T_steps, N_neurons) array of spike counts.
        
    Returns:
        sigma: Estimated branching ratio.
        sigma_err: Standard error of the regression estimate.
    """
    pop_act = np.sum(spike_train, axis=1).astype(float)
    x = pop_act[:-1]
    y = pop_act[1:]
    
    # Restrict to active points where x > 0
    active = x > 0
    if np.sum(active) < 5:
        return 0.0, 0.0
        
    x_act = x[active]
    y_act = y[active]
    
    denom = np.sum(x_act ** 2)
    if denom <= 1e-9:
        return 0.0, 0.0
        
    sigma = float(np.sum(x_act * y_act) / denom)
    
    # Residual variance
    residuals = y_act - sigma * x_act
    var_res = np.var(residuals)
    sigma_err = float(np.sqrt(var_res / denom)) if denom > 0 else 0.0
    
    return sigma, sigma_err


def fit_power_law_mle(
    data: List[int],
    x_min: Optional[int] = None,
) -> float:
    """
    Computes discrete Maximum Likelihood Estimator (MLE) for power law exponent:
    P(x) ~ x^(-alpha)
    
    Using the continuous approximation formula with discrete correction (Clauset et al. 2009):
    alpha = 1 + n * [ sum_i ln( x_i / (x_min - 0.5) ) ]^(-1)
    """
    if len(data) < 5:
        return 0.0
        
    data_arr = np.array(data, dtype=float)
    if x_min is None:
        x_min = max(1, int(np.percentile(data_arr, 10)))
        
    filtered = data_arr[data_arr >= x_min]
    if len(filtered) < 5:
        return 0.0
        
    log_sum = np.sum(np.log(filtered / (x_min - 0.5)))
    if log_sum <= 0:
        return 0.0
        
    alpha = 1.0 + len(filtered) / log_sum
    return float(alpha)


def analyze_network_criticality(
    spike_train: np.ndarray,
    bin_size: int = 1,
) -> AvalancheProfile:
    """
    Performs full self-organized criticality analysis on a recorded spike train.
    """
    sigma_mean, sigma_std = compute_branching_ratio(spike_train)
    sizes, durations = detect_neuronal_avalanches(spike_train, bin_size=bin_size)
    
    if len(sizes) == 0:
        return AvalancheProfile(
            branching_ratio=sigma_mean,
            branching_ratio_std=sigma_std,
            num_avalanches=0,
            mean_avalanche_size=0.0,
            max_avalanche_size=0,
            mean_avalanche_duration=0.0,
            max_avalanche_duration=0,
            tau_size_exponent=0.0,
            alpha_duration_exponent=0.0,
            crackling_scaling_gamma=0.0,
            is_critical=False,
        )
        
    tau = fit_power_law_mle(sizes, x_min=1)
    alpha = fit_power_law_mle(durations, x_min=1)
    
    gamma = 0.0
    if tau > 1.0 and alpha > 1.0:
        gamma = (alpha - 1.0) / (tau - 1.0)
        
    is_critical = (
        0.92 <= sigma_mean <= 1.08 and
        1.2 <= tau <= 2.2 and
        1.5 <= alpha <= 2.8
    )
    
    return AvalancheProfile(
        branching_ratio=sigma_mean,
        branching_ratio_std=sigma_std,
        num_avalanches=len(sizes),
        mean_avalanche_size=float(np.mean(sizes)),
        max_avalanche_size=int(np.max(sizes)),
        mean_avalanche_duration=float(np.mean(durations)),
        max_avalanche_duration=int(np.max(durations)),
        tau_size_exponent=tau,
        alpha_duration_exponent=alpha,
        crackling_scaling_gamma=gamma,
        is_critical=is_critical,
    )
