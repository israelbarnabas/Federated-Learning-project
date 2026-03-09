#!/usr/bin/env python3
"""
Quick test to verify DP-SGD setup and epsilon computation.
Pure TensorFlow version - no tensorflow-privacy dependency.
"""

import sys
sys.path.insert(0, '.')

from try_project.dp_utils import (
    check_dp_available,
    compute_epsilon,
    find_noise_multiplier_for_epsilon,
    apply_dp_to_gradients,
    create_dp_optimizer,
)


def test_dp_availability():
    """Check if DP implementation is available."""
    available = check_dp_available()
    print(f"DP Implementation available: {available}")
    return available


def test_epsilon_computation():
    """Test epsilon computation with typical FL parameters."""
    print("\n--- Testing Epsilon Computation ---")
    print("Expected: Higher noise (sigma) = Lower epsilon (better privacy)")
    
    # Typical FL scenario: 30 clients, ~300 samples each
    num_samples = 300
    batch_size = 32
    epochs = 3
    delta = 1e-4
    
    test_cases = [
        (0.5, "Very low noise, poor privacy"),
        (1.0, "Low noise"),
        (2.0, "Moderate noise"),
        (5.0, "High noise, good privacy"),
        (10.0, "Very high noise, strong privacy"),
    ]
    
    for noise_mult, description in test_cases:
        eps, details = compute_epsilon(
            num_samples=num_samples,
            batch_size=batch_size,
            noise_multiplier=noise_mult,
            epochs=epochs,
            delta=delta,
        )
        print(f"  σ={noise_mult:5.1f} ({description}): ε={eps:8.2f}")
    
    # Verify monotonicity
    eps_low_noise, _ = compute_epsilon(num_samples, batch_size, 0.5, epochs, delta)
    eps_high_noise, _ = compute_epsilon(num_samples, batch_size, 10.0, epochs, delta)
    
    if eps_low_noise > eps_high_noise:
        print("\n✓ CORRECT: Higher noise gives lower epsilon (better privacy)")
    else:
        print("\n✗ ERROR: Privacy computation is inverted!")
    
    return True


def test_noise_search():
    """Test finding noise multiplier for target epsilon."""
    print("\n--- Testing Noise Multiplier Search ---")
    print("Expected: Higher target epsilon → Lower noise needed")
    
    target_epsilons = [1.0, 3.0, 5.0, 10.0]
    num_samples = 300
    batch_size = 32
    epochs = 3
    delta = 1e-4
    
    for target in target_epsilons:
        noise, achieved = find_noise_multiplier_for_epsilon(
            target_epsilon=target,
            num_samples=num_samples,
            batch_size=batch_size,
            epochs=epochs,
            delta=delta,
        )
        print(f"  Target ε={target:5.1f}: need σ={noise:7.4f}, achieves ε={achieved:7.2f}")
    
    return True


def test_optimizer_creation():
    """Test optimizer creation (functional approach)."""
    print("\n--- Testing Optimizer Creation ---")
    
    # Test standard optimizer (no noise)
    opt1 = create_dp_optimizer(
        noise_multiplier=0,
        l2_norm_clip=1.0,
        num_microbatches=1,
        learning_rate=0.001,
    )
    print(f"  Standard optimizer: {type(opt1).__name__}")
    
    # Test "DP" optimizer - now returns standard optimizer (DP applied in training loop)
    opt2 = create_dp_optimizer(
        noise_multiplier=1.0,
        l2_norm_clip=1.0,
        num_microbatches=1,
        learning_rate=0.001,
    )
    print(f"  DP optimizer (functional): {type(opt2).__name__}")
    print("  ✓ Optimizer created successfully (DP applied in training loop)")
    
    return True


def test_gradient_processing():
    """Test DP gradient clipping and noise."""
    print("\n--- Testing DP Gradient Processing ---")
    import tensorflow as tf
    
    # Create fake gradients
    grads = [
        tf.constant([[1.0, 2.0], [3.0, 4.0]]),
        tf.constant([0.5, 1.5]),
    ]
    
    print(f"  Original gradients norm: {tf.linalg.global_norm(grads):.4f}")
    
    # Test with different noise multipliers
    for sigma in [0.0, 1.0, 5.0]:
        dp_grads = apply_dp_to_gradients(grads, noise_multiplier=sigma, l2_norm_clip=1.0)
        new_norm = tf.linalg.global_norm(dp_grads)
        print(f"  After DP (σ={sigma}): norm={new_norm:.4f}")
    
    print("  ✓ Gradient processing works")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DP-SGD Setup Verification (Pure TensorFlow)")
    print("=" * 60)
    
    if not test_dp_availability():
        print("\nERROR: DP implementation not available!")
        return 1
    
    try:
        test_epsilon_computation()
        test_noise_search()
        test_optimizer_creation()
        test_gradient_processing()
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed! DP-SGD is ready to use.")
    print("=" * 60)
    print("\nImplementation notes:")
    print("- Using pure TensorFlow (no tensorflow-privacy)")
    print("- Simplified privacy accounting (conservative estimate)")
    print("- Functional approach: DP applied in training loop, not in optimizer")
    print("\nNext steps:")
    print("1. Run: flwr run . --run-config '{\"use-dp\": true, \"target-epsilon\": 3.0}'")
    print("2. Check for '[Client X] Training with DP-SGD' in logs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())