#!/usr/bin/env python3
"""
Quick test to verify DP-SGD setup and epsilon computation.
"""

import sys
sys.path.insert(0, '.')

from try_project.dp_utils import (
    check_dp_available,
    compute_epsilon,
    find_noise_multiplier_for_epsilon,
)


def test_dp_availability():
    """Check if tensorflow-privacy is installed."""
    available = check_dp_available()
    print(f"TensorFlow Privacy available: {available}")
    return available


def test_epsilon_computation():
    """Test epsilon computation with typical FL parameters."""
    print("\n--- Testing Epsilon Computation ---")
    
    # Typical FL scenario: 30 clients, ~300 samples each
    num_samples = 300
    batch_size = 32
    epochs = 3
    delta = 1e-4
    
    test_cases = [
        (0.5, "Very noisy, high privacy"),
        (1.0, "Standard noise"),
        (2.0, "Moderate noise"),
        (5.0, "Low noise, better utility"),
    ]
    
    for noise_mult, description in test_cases:
        eps, details = compute_epsilon(
            num_samples=num_samples,
            batch_size=batch_size,
            noise_multiplier=noise_mult,
            epochs=epochs,
            delta=delta,
        )
        print(f"  σ={noise_mult:.1f} ({description}): ε={eps:.2f}")
    
    return True


def test_noise_search():
    """Test finding noise multiplier for target epsilon."""
    print("\n--- Testing Noise Multiplier Search ---")
    
    target_epsilons = [1.0, 3.0, 5.0]
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
        print(f"  Target ε={target}: need σ={noise:.4f}, achieves ε={achieved:.2f}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DP-SGD Setup Verification")
    print("=" * 60)
    
    if not test_dp_availability():
        print("\nERROR: tensorflow-privacy not installed!")
        print("Install with: pip install tensorflow-privacy>=0.15.0")
        return 1
    
    try:
        test_epsilon_computation()
        test_noise_search()
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed! DP-SGD is ready to use.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install tensorflow-privacy if not already done")
    print("2. Run: flwr run . --run-config '{\"use-dp\": true, \"target-epsilon\": 3.0}'")
    print("3. Check that DP metrics appear in client logs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())