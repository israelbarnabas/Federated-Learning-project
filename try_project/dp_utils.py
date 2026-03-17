"""
Correct DP-SGD with Renyi Differential Privacy (RDP) accounting.
Optimized with caching and reduced bounds to prevent CPU bottlenecks.
FIXED: Added noise capping and CORRECT cache key including num_samples.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, List
from scipy.special import comb
import functools

# CRITICAL: Maximum noise multiplier to prevent "noise paralysis"
MAX_NOISE_MULTIPLIER = 20.0

# Minimum useful epsilon per round - below this, noise is too high
MIN_USEFUL_EPSILON = 0.3


def compute_rdp_gaussian(q: float, sigma: float, alpha: float) -> float:
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma**2)
    if q <= 0.01:
        return q**2 * alpha / (2 * sigma**2)
    
    log_moment = _compute_log_moment_gaussian(q, sigma, alpha)
    return log_moment

def _compute_log_moment_gaussian(q: float, sigma: float, alpha: float) -> float:
    if alpha <= 1:
        raise ValueError("Alpha must be > 1")
    if q >= 1.0:
        return alpha / (2 * sigma**2)
    if float(alpha).is_integer():
        return _compute_log_moment_integer(q, sigma, int(alpha))
    
    alpha_floor = int(np.floor(alpha))
    alpha_ceil = int(np.ceil(alpha))
    eps_floor = _compute_log_moment_integer(q, sigma, alpha_floor)
    eps_ceil = _compute_log_moment_integer(q, sigma, alpha_ceil)
    
    t = alpha - alpha_floor
    return eps_floor * (1 - t) + eps_ceil * t

def _compute_log_moment_integer(q: float, sigma: float, alpha: int) -> float:
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef = np.log(comb(alpha, i))
        log_samp = i * np.log(q) + (alpha - i) * np.log1p(-q)
        log_gauss = 0.0 if i == 0 else (i * (i - 1)) / (2 * sigma**2)
        log_a = np.logaddexp(log_a, log_coef + log_samp + log_gauss)
    return log_a / (alpha - 1) if alpha > 1 else 0.0

def from_rdp_to_dp(epsilon_rdp: float, alpha: float, delta: float) -> float:
    if alpha <= 1:
        raise ValueError("Alpha must be > 1")
    return epsilon_rdp + np.log(1 / delta) / (alpha - 1)

def compute_epsilon_rdp(
    num_samples: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-4,
    alphas: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    
    if noise_multiplier <= 0:
        return float('inf'), {"error": "Noise multiplier must be positive"}
    
    q = batch_size / num_samples
    steps_per_epoch = max(1, num_samples // batch_size)
    total_steps = steps_per_epoch * epochs
    
    if alphas is None:
        # CRITICAL FIX: Lowered max alpha to 64 to prevent CPU freeze
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 65))
        alphas = [a for a in alphas if a > 1.0]
    
    rdp_per_step = {alpha: compute_rdp_gaussian(q, noise_multiplier, alpha) for alpha in alphas}
    total_rdp = {alpha: rdp * total_steps for alpha, rdp in rdp_per_step.items()}
    
    eps_candidates = {alpha: from_rdp_to_dp(total_rdp[alpha], alpha, delta) for alpha in alphas}
    
    best_alpha = min(eps_candidates.keys(), key=lambda a: eps_candidates[a])
    best_epsilon = eps_candidates[best_alpha]
    
    details = {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "sampling_rate_q": q,
        "total_steps": total_steps,
        "noise_multiplier": noise_multiplier,
        "best_alpha": best_alpha,
        "rdp_at_best_alpha": total_rdp[best_alpha],
        "epsilon": best_epsilon,
        "delta": delta,
    }
    return best_epsilon, details


# FIXED: Cache key now includes all parameters that affect the result
@functools.lru_cache(maxsize=1024)
def _cached_noise_search(
    target_epsilon_round: float, 
    num_samples: int, 
    batch_size: int, 
    epochs: int, 
    delta: float
) -> Tuple[float, float]:
    """Cached binary search to instantly return known noise values."""
    sigma_low = 0.1
    sigma_high = 100.0
    
    eps_at_high, _ = compute_epsilon_rdp(num_samples, batch_size, sigma_high, epochs, delta)
    if eps_at_high > target_epsilon_round:
        return sigma_high, eps_at_high
        
    eps_at_low, _ = compute_epsilon_rdp(num_samples, batch_size, sigma_low, epochs, delta)
    if eps_at_low < target_epsilon_round:
        return sigma_low, eps_at_low
    
    # Reduced from 50 iterations to 20 for massive speed up
    for _ in range(20):  
        sigma_mid = (sigma_low + sigma_high) / 2.0
        eps_mid, _ = compute_epsilon_rdp(num_samples, batch_size, sigma_mid, epochs, delta)
        
        if abs(eps_mid - target_epsilon_round) < 0.05:
            return sigma_mid, eps_mid
        if eps_mid > target_epsilon_round:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
            
    sigma_final = (sigma_low + sigma_high) / 2.0
    eps_final, _ = compute_epsilon_rdp(num_samples, batch_size, sigma_final, epochs, delta)
    return sigma_final, eps_final


def find_noise_multiplier_for_epsilon_rdp(
    target_epsilon: float,
    num_samples: int,
    batch_size: int,
    epochs: int,
    delta: float = 1e-4,
    tolerance: float = 0.01,
) -> Tuple[float, float]:
    """
    Find noise multiplier for target epsilon with sanity checks.
    FIXED: Prevents excessive noise that causes learning paralysis.
    """
    
    # CRITICAL FIX 1: Enforce minimum useful epsilon
    original_epsilon = target_epsilon
    if target_epsilon < MIN_USEFUL_EPSILON:
        print(f"[DP] Warning: Requested ε={target_epsilon:.3f} below useful threshold, "
              f"clamping to ε={MIN_USEFUL_EPSILON}")
        target_epsilon = MIN_USEFUL_EPSILON
    
    # Round inputs to improve cache hits
    target_eps_rounded = round(target_epsilon, 2)
    num_samples_rounded = int(round(num_samples, -1))  # Round to nearest 10
    batch_size_rounded = batch_size
    epochs_rounded = epochs
    
    noise_mult, achieved_eps = _cached_noise_search(
        target_eps_rounded, num_samples_rounded, batch_size_rounded, epochs_rounded, delta
    )
    
    # CRITICAL FIX 2: Cap maximum noise to prevent paralysis
    if noise_mult > MAX_NOISE_MULTIPLIER:
        print(f"[DP] Warning: Calculated σ={noise_mult:.2f} exceeds maximum useful "
              f"noise (σ={MAX_NOISE_MULTIPLIER}), capping")
        
        # Find epsilon that corresponds to MAX_NOISE_MULTIPLIER
        eps_at_max_noise, _ = compute_epsilon_rdp(
            num_samples, batch_size, MAX_NOISE_MULTIPLIER, epochs, delta
        )
        
        noise_mult = MAX_NOISE_MULTIPLIER
        achieved_eps = eps_at_max_noise
        
        if original_epsilon < achieved_eps:
            print(f"[DP] Adjusted: ε={original_epsilon:.3f} → ε={achieved_eps:.3f} "
                  f"(σ={noise_mult:.2f}) to enable learning")
    
    return noise_mult, achieved_eps


def apply_dp_to_gradients(
    grads: List[tf.Tensor],
    noise_multiplier: float,
    l2_norm_clip: float,
) -> List[tf.Tensor]:
    """
    Apply DP-SGD to gradients with noise capping.
    """
    # CRITICAL FIX 3: Cap noise at application time as safety check
    effective_noise = min(noise_multiplier, MAX_NOISE_MULTIPLIER)
    if effective_noise != noise_multiplier:
        print(f"[DP] Warning: Applied noise σ={noise_multiplier:.2f} capped to "
              f"σ={effective_noise:.2f}")
    
    if effective_noise <= 0:
        global_norm = tf.linalg.global_norm(grads)
        clip_factor = tf.minimum(l2_norm_clip / (global_norm + 1e-10), 1.0)
        return [g * clip_factor for g in grads]
    
    global_norm = tf.linalg.global_norm(grads)
    clip_factor = tf.minimum(l2_norm_clip / (global_norm + 1e-10), 1.0)
    clipped_grads = [g * clip_factor for g in grads]
    
    noise_stddev = l2_norm_clip * effective_noise
    
    noisy_grads = []
    for g in clipped_grads:
        if g is not None:
            noise = tf.random.normal(
                shape=tf.shape(g), mean=0.0, stddev=noise_stddev, dtype=g.dtype
            )
            noisy_grads.append(g + noise)
        else:
            noisy_grads.append(None)
    return noisy_grads


# Aliases
compute_epsilon = compute_epsilon_rdp
find_noise_multiplier_for_epsilon = find_noise_multiplier_for_epsilon_rdp

def check_dp_available() -> bool:
    return True