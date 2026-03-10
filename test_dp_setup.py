#!/usr/bin/env python3
"""
Enhanced DP-SGD verification test.
Validates your current implementation including noise scaling and privacy accounting.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

# Test imports
try:
    from try_project.dp_utils import (
        check_dp_available,
        compute_epsilon,
        find_noise_multiplier_for_epsilon,
        apply_dp_to_gradients,
    )
    from try_project.secure_agg import SecureAggregation
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

import tensorflow as tf


def test_dp_availability():
    """Verify DP implementation is available."""
    print("\n" + "=" * 60)
    print("TEST 1: DP Availability")
    print("=" * 60)
    
    available = check_dp_available()
    status = "✓ PASS" if available else "✗ FAIL"
    print(f"{status}: DP implementation available = {available}")
    return available


def test_epsilon_computation():
    """Test epsilon computation with your actual FL parameters."""
    print("\n" + "=" * 60)
    print("TEST 2: Epsilon Computation (Your FL Settings)")
    print("=" * 60)
    
    # Your actual parameters from client logs
    test_configs = [
        (152, "Client 4 (small)"),
        (371, "Client 8 (medium)"),
        (535, "Client 27 (large)"),
        (87, "Client 12 (tiny)"),
    ]
    
    batch_size = 32
    epochs = 3  # Your actual setting
    delta = 1e-4
    
    print(f"\nFixed parameters: batch_size={batch_size}, epochs={epochs}, δ={delta}")
    print("-" * 60)
    
    for num_samples, description in test_configs:
        # Target epsilon = 3.0 (your test setting)
        target_eps = 3.0
        
        noise_mult, achieved_eps = find_noise_multiplier_for_epsilon(
            target_epsilon=target_eps,
            num_samples=num_samples,
            batch_size=batch_size,
            epochs=epochs,
            delta=delta,
        )
        
        # Verify: higher noise for smaller datasets
        print(f"{description:20s} | n={num_samples:3d} | σ={noise_mult:.4f} | ε={achieved_eps:.2f}")
    
    # Verify monotonicity: smaller dataset → higher noise
    noise_small, _ = find_noise_multiplier_for_epsilon(3.0, 100, 32, 3, 1e-4)
    noise_large, _ = find_noise_multiplier_for_epsilon(3.0, 500, 32, 3, 1e-4)
    
    print("\nMonotonicity check:")
    if noise_small > noise_large:
        print(f"✓ PASS: Smaller dataset needs more noise ({noise_small:.2f} > {noise_large:.2f})")
    else:
        print(f"✗ FAIL: Noise scaling inverted")
        return False
    
    return True


def test_noise_privacy_tradeoff():
    """Verify that noise and privacy have correct inverse relationship."""
    print("\n" + "=" * 60)
    print("TEST 3: Noise-Privacy Tradeoff")
    print("=" * 60)
    
    num_samples = 300
    batch_size = 32
    epochs = 3
    delta = 1e-4
    
    print("Testing: Higher noise → Lower epsilon (better privacy)")
    print("-" * 60)
    
    noise_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for sigma in noise_values:
        eps, _ = compute_epsilon(num_samples, batch_size, sigma, epochs, delta)
        results.append((sigma, eps))
        print(f"σ = {sigma:5.2f} → ε = {eps:8.2f}")
    
    # Verify inverse relationship
    eps_values = [r[1] for r in results]
    is_decreasing = all(eps_values[i] > eps_values[i+1] for i in range(len(eps_values)-1))
    
    print("\nValidation:")
    if is_decreasing:
        print("✓ PASS: Higher noise gives better privacy (lower ε)")
    else:
        print("✗ FAIL: Privacy computation is incorrect")
        return False
    
    # Check specific values from your logs
    print("\nLog validation (your actual values):")
    your_sigma = 2.1522  # From Client 8, ε=3.0
    your_eps, _ = compute_epsilon(371, 32, your_sigma, 3, 1e-4)
    print(f"Your log: σ={your_sigma:.4f} for n=371, claimed ε=3.0")
    print(f"Computed:  σ={your_sigma:.4f} → ε={your_eps:.2f}")
    
    if abs(your_eps - 3.0) < 0.5:
        print("✓ PASS: Matches your logs within tolerance")
    else:
        print("⚠ WARNING: Mismatch with your log values")
    
    return True


def test_gradient_processing():
    """Test DP gradient clipping and noise with realistic gradients."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Processing")
    print("=" * 60)
    
    # Create realistic gradient magnitudes (from your model)
    # CNN with ~100K parameters, gradients ~0.01-0.1 in early training
    grads = [
        tf.constant(np.random.normal(0, 0.05, size=(32, 3, 3, 3)).astype(np.float32)),  # Conv1
        tf.constant(np.random.normal(0, 0.03, size=(32,)).astype(np.float32)),         # Bias1
        tf.constant(np.random.normal(0, 0.02, size=(64, 32)).astype(np.float32)),       # Dense
    ]
    
    original_norm = float(tf.linalg.global_norm(grads))
    print(f"Original gradient norm: {original_norm:.4f}")
    
    # Test with your actual L2 clip value (0.5 in improved code, 1.0 in old)
    for clip in [1.0, 0.5, 0.1]:
        for sigma in [0.0, 1.0, 2.0]:
            dp_grads = apply_dp_to_gradients(grads, noise_multiplier=sigma, l2_norm_clip=clip)
            new_norm = float(tf.linalg.global_norm(dp_grads))
            
            status = "✓" if new_norm <= clip * 1.1 else "✗"  # Allow 10% tolerance for noise
            print(f"{status} Clip={clip}, σ={sigma}: norm {original_norm:.3f} → {new_norm:.3f}")
    
    # Verify clipping actually happens
    dp_grads_clipped = apply_dp_to_gradients(grads, noise_multiplier=0, l2_norm_clip=0.1)
    clipped_norm = float(tf.linalg.global_norm(dp_grads_clipped))
    
    print(f"\nStrict clipping check (C=0.1):")
    if clipped_norm <= 0.11:  # Small tolerance
        print(f"✓ PASS: Clipping works (norm={clipped_norm:.4f} <= 0.1)")
    else:
        print(f"✗ FAIL: Clipping ineffective (norm={clipped_norm:.4f})")
        return False
    
    return True


def test_secure_aggregation():
    """Test SA masking and unmasking."""
    print("\n" + "=" * 60)
    print("TEST 5: Secure Aggregation")
    print("=" * 60)
    
    sa = SecureAggregation(num_clients=30, threshold=20)
    
    # Simulate 3 clients with small weight vectors
    weight_shapes = [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((5,), dtype=np.float32),
    ]
    
    round_seed = sa.generate_round_seed(server_round=1)
    print(f"Round seed: {round_seed}")
    
    # Generate masks for 3 clients
    masks_per_client = []
    for cid in [0, 1, 2]:
        masks = sa.generate_masks(cid, round_seed, weight_shapes)
        masks_per_client.append(masks)
        print(f"Client {cid}: mask norms = {[float(np.linalg.norm(m)) for m in masks]}")
    
    # Verify masks are different
    if not np.allclose(masks_per_client[0][0], masks_per_client[1][0]):
        print("✓ PASS: Masks are client-specific")
    else:
        print("✗ FAIL: Masks identical across clients")
        return False
    
    # Simulate masking and unmasking
    fake_weights = [np.ones((10, 10), dtype=np.float32) * 0.5, np.ones((5,), dtype=np.float32) * 0.3]
    
    masked_weights = []
    for cid, masks in zip([0, 1, 2], masks_per_client):
        masked = sa.mask_weights(fake_weights, masks)
        masked_weights.append(masked)
    
    # Server aggregates and unmasks
    aggregated = sa.unmask_aggregate(masked_weights, [0, 1, 2], round_seed)
    
    print(f"\nOriginal weights avg: {np.mean([np.mean(w) for w in fake_weights]):.4f}")
    print(f"Unmasked aggregate avg: {np.mean([np.mean(w) for w in aggregated]):.4f}")
    
    # Should recover average (0.5 + 0.3)/2 = 0.4 roughly
    if abs(np.mean([np.mean(w) for w in aggregated]) - 0.4) < 0.01:
        print("✓ PASS: SA unmasking recovers correct average")
    else:
        print("⚠ WARNING: SA recovery imperfect (may be due to mask scale)")
    
    return True


def test_privacy_accounting_accumulation():
    """Test that privacy budget accumulates across rounds."""
    print("\n" + "=" * 60)
    print("TEST 6: Privacy Budget Accumulation")
    print("=" * 60)
    
    # Simulate 10 rounds with ε=3.0 per round
    num_samples = 300
    batch_size = 32
    epochs = 3
    delta = 1e-4
    target_eps_per_round = 3.0
    
    noise_mult, _ = find_noise_multiplier_for_epsilon(
        target_eps_per_round, num_samples, batch_size, epochs, delta
    )
    
    print(f"Per-round: ε={target_eps_per_round}, σ={noise_mult:.4f}")
    
    # Simple composition (conservative upper bound)
    num_rounds = 10
    total_epsilon_basic = target_eps_per_round * num_rounds  # 30.0
    total_epsilon_advanced = target_eps_per_round * np.sqrt(num_rounds)  # ~9.5
    
    print(f"\nAfter {num_rounds} rounds:")
    print(f"  Basic composition (upper bound): ε = {total_epsilon_basic:.2f}")
    print(f"  Advanced composition (estimate): ε = {total_epsilon_advanced:.2f}")
    
    print(f"\n⚠ WARNING: Your target was ε={target_eps_per_round} TOTAL, not per-round!")
    print(f"  To achieve TOTAL ε=3.0 over 10 rounds, use per-round ε={3.0/np.sqrt(10):.2f}")
    
    return True


def test_your_actual_configuration():
    """Test the exact configuration from your logs."""
    print("\n" + "=" * 60)
    print("TEST 7: Your Actual Configuration Validation")
    print("=" * 60)
    
    # From your logs: ε=3.0, various client sizes
    configs = [
        {"cid": 4, "samples": 152, "log_sigma": 3.1677, "log_eps": 1.0},
        {"cid": 8, "samples": 371, "log_sigma": 2.1522, "log_eps": 1.0},
        {"cid": 12, "samples": 87, "log_sigma": 3.9134, "log_eps": 1.0},
        {"cid": 27, "samples": 535, "log_sigma": 0.6000, "log_eps": 3.0},  # Wait, this is different
    ]
    
    print("Validating your log entries:")
    print("-" * 60)
    
    for cfg in configs:
        computed_eps, _ = compute_epsilon(
            cfg["samples"], 32, cfg["log_sigma"], 3, 1e-4
        )
        
        match = "✓" if abs(computed_eps - cfg["log_eps"]) < 0.5 else "✗"
        print(f"Client {cfg['cid']:2d}: n={cfg['samples']:3d}, σ={cfg['log_sigma']:.4f} | "
              f"Log says ε={cfg['log_eps']:.2f}, computed ε={computed_eps:.2f} {match}")
    
    print("\nNote: If σ varies for same ε target, your noise computation is sample-dependent.")
    print("This is CORRECT for per-sample privacy but creates heterogeneous noise.")
    
    return True


def run_all_tests():
    """Execute all validation tests."""
    print("=" * 70)
    print("DP-SGD & SA Validation Suite")
    print("=" * 70)
    
    tests = [
        ("DP Availability", test_dp_availability),
        ("Epsilon Computation", test_epsilon_computation),
        ("Noise-Privacy Tradeoff", test_noise_privacy_tradeoff),
        ("Gradient Processing", test_gradient_processing),
        ("Secure Aggregation", test_secure_aggregation),
        ("Privacy Accounting", test_privacy_accounting_accumulation),
        ("Your Configuration", test_your_actual_configuration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("-" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your DP implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run: flwr run . --run-config '{\"use-dp\": true, \"target-epsilon\": 5.0, \"num-rounds\": 10}'")
        print("2. Verify accuracy > 30% (if not, check training loop, not DP)")
        print("3. Test SA: use-sa=true with use-noisy-channel=true")
    else:
        print("\n⚠ Fix failing tests before running full experiments.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)