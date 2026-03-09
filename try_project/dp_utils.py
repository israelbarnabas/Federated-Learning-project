"""
Differential Privacy utilities for Federated Learning.
Implements DP-SGD with moments accountant for privacy budget tracking.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
import warnings

# Try to import tensorflow-privacy, provide fallback if unavailable
try:
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
        DPKerasSGDOptimizer,
        DPKerasAdamOptimizer,
    )
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
    TF_PRIVACY_AVAILABLE = True
except ImportError:
    TF_PRIVACY_AVAILABLE = False
    warnings.warn("tensorflow-privacy not installed. DP features will be unavailable.")


def check_dp_available() -> bool:
    """Check if tensorflow-privacy is available."""
    return TF_PRIVACY_AVAILABLE


def compute_epsilon(
    num_samples: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-4,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute privacy budget (epsilon) using moments accountant.
    
    Args:
        num_samples: Total number of samples in dataset
        batch_size: Training batch size
        noise_multiplier: Noise multiplier (sigma)
        epochs: Number of training epochs
        delta: Privacy parameter (target failure probability)
    
    Returns:
        Tuple of (epsilon, details_dict)
    """
    if not TF_PRIVACY_AVAILABLE:
        raise RuntimeError("tensorflow-privacy required for epsilon computation")
    
    if noise_multiplier <= 0:
        return float('inf'), {"error": "Noise multiplier must be positive"}
    
    # Compute using TensorFlow Privacy's library
    eps = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        n=num_samples,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta,
    )
    
    details = {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "noise_multiplier": noise_multiplier,
        "epochs": epochs,
        "delta": delta,
        "epsilon": float(eps[0]) if isinstance(eps, tuple) else float(eps),
    }
    
    return details["epsilon"], details


def find_noise_multiplier_for_epsilon(
    target_epsilon: float,
    num_samples: int,
    batch_size: int,
    epochs: int,
    delta: float = 1e-4,
    tolerance: float = 0.1,
    max_iterations: int = 50,
) -> Tuple[float, float]:
    """
    Binary search to find noise multiplier for target epsilon.
    
    Args:
        target_epsilon: Desired privacy budget
        num_samples, batch_size, epochs, delta: DP parameters
        tolerance: Acceptable epsilon difference
        max_iterations: Search iterations
    
    Returns:
        Tuple of (noise_multiplier, achieved_epsilon)
    """
    if not TF_PRIVACY_AVAILABLE:
        raise RuntimeError("tensorflow-privacy required")
    
    # Binary search bounds
    low, high = 0.1, 100.0
    
    for i in range(max_iterations):
        mid = (low + high) / 2
        eps, _ = compute_epsilon(num_samples, batch_size, mid, epochs, delta)
        
        if abs(eps - target_epsilon) < tolerance:
            return mid, eps
        
        if eps > target_epsilon:
            # Need more noise
            low = mid
        else:
            # Need less noise
            high = mid
    
    # Return best found
    final_eps, _ = compute_epsilon(num_samples, batch_size, mid, epochs, delta)
    return mid, final_eps


def create_dp_optimizer(
    noise_multiplier: float,
    l2_norm_clip: float,
    num_microbatches: int,
    learning_rate: float = 0.001,
    optimizer_type: str = "sgd",
) -> tf.keras.optimizers.Optimizer:
    """
    Create a differentially private optimizer.
    
    Args:
        noise_multiplier: Noise multiplier (sigma) for DP
        l2_norm_clip: Gradient clipping bound (C)
        num_microbatches: Number of microbatches for gradient computation
        learning_rate: Learning rate
        optimizer_type: "sgd" or "adam"
    
    Returns:
        DP-wrapped optimizer
    """
    if not TF_PRIVACY_AVAILABLE:
        raise RuntimeError("tensorflow-privacy required for DP optimizer")
    
    if noise_multiplier <= 0:
        raise ValueError("Noise multiplier must be positive for DP")
    
    if optimizer_type.lower() == "sgd":
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate,
        )
    elif optimizer_type.lower() == "adam":
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


class PrivacyAccountant:
    """
    Track privacy budget consumption across FL rounds.
    Implements simple composition for now (can upgrade to advanced composition).
    """
    
    def __init__(self, target_epsilon: float, target_delta: float, num_samples: int):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.num_samples = num_samples
        self.rounds_consumed: list[Dict] = []
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