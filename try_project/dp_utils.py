"""
Standalone Differential Privacy utilities for Federated Learning.
Pure TensorFlow implementation - no tensorflow-privacy dependency.
Implements DP-SGD with manual privacy accounting.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, List
import math


def compute_epsilon_simple(
    num_samples: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-4,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute privacy budget using simplified moments accountant.
    
    CORRECTED: Higher noise = lower epsilon (better privacy)
    """
    if noise_multiplier <= 0:
        return float('inf'), {"error": "Noise multiplier must be positive"}
    
    # Compute key parameters
    steps_per_epoch = max(1, num_samples // batch_size)
    total_steps = steps_per_epoch * epochs
    sampling_rate = batch_size / num_samples
    
    # Standard deviation of Gaussian noise
    sigma = noise_multiplier
    
    # Simplified privacy accounting: use the standard Gaussian mechanism composition
    # For sampled Gaussian mechanism: epsilon ≈ sqrt(2 * log(1.25/δ)) * (1/σ) * sqrt(T) * q
    # where q = sampling_rate, T = total_steps
    
    # This is a simplified but directionally correct formula
    # Higher sigma (more noise) → lower epsilon (better privacy)
    log_term = math.sqrt(2 * math.log(1.25 / delta))
    composition_factor = math.sqrt(total_steps) * sampling_rate
    
    epsilon = log_term * composition_factor / sigma
    
    details = {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "sampling_rate": sampling_rate,
        "noise_multiplier": noise_multiplier,
        "sigma": sigma,
        "epochs": epochs,
        "delta": delta,
        "log_term": log_term,
        "composition_factor": composition_factor,
        "epsilon": epsilon,
        "method": "simplified_gaussian_composition",
    }
    
    return epsilon, details


def find_noise_multiplier_for_epsilon(
    target_epsilon: float,
    num_samples: int,
    batch_size: int,
    epochs: int,
    delta: float = 1e-4,
    tolerance: float = 0.5,
    max_iterations: int = 100,
) -> Tuple[float, float]:
    """
    Find noise multiplier for target epsilon.
    CORRECTED: Direct computation instead of binary search.
    """
    # For target epsilon, we need: sigma = (log_term * composition_factor) / epsilon
    # So higher target epsilon → lower sigma (less noise needed)
    
    # Get the constant factor by computing with sigma=1
    eps_at_1, details = compute_epsilon_simple(num_samples, batch_size, 1.0, epochs, delta)
    log_term = details["log_term"]
    composition_factor = details["composition_factor"]
    
    # Direct computation: sigma = (log_term * composition_factor) / target_epsilon
    optimal_sigma = (log_term * composition_factor) / target_epsilon
    
    # Verify
    achieved_eps, _ = compute_epsilon_simple(num_samples, batch_size, optimal_sigma, epochs, delta)
    
    return optimal_sigma, achieved_eps


def apply_dp_to_gradients(
    grads: List[tf.Tensor],
    noise_multiplier: float,
    l2_norm_clip: float,
) -> List[tf.Tensor]:
    """
    Apply differential privacy to gradients: clip then add noise.
    
    Args:
        grads: List of gradient tensors
        noise_multiplier: Sigma for Gaussian noise
        l2_norm_clip: Maximum L2 norm for gradients
    
    Returns:
        List of DP-processed gradients
    """
    if noise_multiplier <= 0:
        # Just clip, no noise
        global_norm = tf.linalg.global_norm(grads)
        clip_factor = tf.minimum(l2_norm_clip / (global_norm + 1e-10), 1.0)
        return [g * clip_factor for g in grads]
    
    # Compute global L2 norm
    global_norm = tf.linalg.global_norm(grads)
    
    # Clip gradients
    clip_factor = tf.minimum(l2_norm_clip / (global_norm + 1e-10), 1.0)
    clipped_grads = [g * clip_factor for g in grads]
    
    # Add Gaussian noise
    noise_stddev = l2_norm_clip * noise_multiplier
    noisy_grads = []
    for g in clipped_grads:
        if g is not None:
            noise = tf.random.normal(
                shape=tf.shape(g),
                mean=0.0,
                stddev=noise_stddev,
                dtype=g.dtype
            )
            noisy_grads.append(g + noise)
        else:
            noisy_grads.append(None)
    
    return noisy_grads


def create_dp_optimizer(
    noise_multiplier: float,
    l2_norm_clip: float,
    num_microbatches: int,
    learning_rate: float = 0.001,
    optimizer_type: str = "sgd",
) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer for DP-SGD.
    
    NOTE: Gradient clipping and noise are applied in the training loop
    via apply_dp_to_gradients(), not in the optimizer itself.
    This ensures compatibility with Keras 3.x / TF 2.16.
    """
    # Always return standard optimizer - DP operations done in training loop
    if optimizer_type.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)


class PrivacyAccountant:
    """
    Track privacy budget consumption across FL rounds.
    """
    
    def __init__(self, target_epsilon: float, target_delta: float, num_samples: int):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.num_samples = num_samples
        self.rounds_consumed: List[Dict] = []
        self.total_epsilon_spent = 0.0
    
    def spend_budget(self, epsilon_round: float, round_num: int):
        """Record epsilon spent in a round."""
        self.rounds_consumed.append({
            "round": round_num,
            "epsilon": epsilon_round,
        })
        self.total_epsilon_spent += epsilon_round
    
    def get_remaining_budget(self) -> float:
        """Get remaining epsilon budget."""
        return max(0, self.target_epsilon - self.total_epsilon_spent)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.total_epsilon_spent >= self.target_epsilon
    
    def get_summary(self) -> Dict[str, Any]:
        """Get privacy consumption summary."""
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "total_epsilon_spent": self.total_epsilon_spent,
            "remaining_budget": self.get_remaining_budget(),
            "budget_exhausted": self.is_budget_exhausted(),
            "rounds_accounted": len(self.rounds_consumed),
        }


def check_dp_available() -> bool:
    """Always returns True since we use pure TF implementation."""
    return True


# Aliases for compatibility
compute_epsilon = compute_epsilon_simple