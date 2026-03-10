#!/usr/bin/env python3
"""Test Secure Aggregation mask cancellation."""

import numpy as np
from try_project.secure_agg import SecureAggregation, ProperSecureAggregation

def test_simple_sa():
    """Test simplified SA (masks approximately cancel)."""
    print("=" * 60)
    print("Testing Simplified Secure Aggregation")
    print("=" * 60)
    
    # Use threshold = 3 for testing
    sa = SecureAggregation(num_clients=30, threshold=3)
    round_seed = sa.generate_round_seed(1)
    
    # Simulate 3 clients with identical weights
    weight_shapes = [(10,)]
    weights = [np.ones(10, dtype=np.float32)]
    original_value = weights[0][0]
    
    print(f"Original weight value: {original_value:.6f}")
    
    # Generate masks for 3 clients
    client_ids = [0, 1, 2]
    masks = [sa.generate_masks(cid, round_seed, weight_shapes) for cid in client_ids]
    
    # Apply masks
    masked_weights = [sa.mask_weights(weights, m) for m in masks]
    
    print(f"Masked values: {[w[0][0] for w in masked_weights]}")
    
    # Aggregate (masks should approximately cancel)
    aggregated = sa.aggregate_masked(masked_weights, client_ids, round_seed)
    aggregated_value = aggregated[0][0]
    
    print(f"Aggregated value: {aggregated_value:.6f}")
    print(f"Difference from original: {abs(aggregated_value - original_value):.10f}")
    
    # Check if close (approximate cancellation due to random masks)
    if abs(aggregated_value - original_value) < 0.01:  # 1% tolerance
        print("✓ SUCCESS: Masks approximately cancel out")
        return True
    else:
        print("✗ FAIL: Masks did not cancel properly")
        return False

def test_pairwise_sa():
    """Test proper SA with pairwise masking (exact cancellation)."""
    print("\n" + "=" * 60)
    print("Testing Proper Secure Aggregation (Pairwise)")
    print("=" * 60)
    
    # Use threshold = 3 for testing
    sa = ProperSecureAggregation(num_clients=30, threshold=3)
    round_seed = sa.generate_round_seed(1)
    
    weight_shapes = [(10,)]
    weights = [np.ones(10, dtype=np.float32)]
    original_value = weights[0][0]
    
    print(f"Original weight value: {original_value:.6f}")
    
    client_ids = [0, 1, 2]
    
    # Generate pairwise masks (exact cancellation guaranteed)
    masks = [
        sa.generate_pairwise_masks(cid, round_seed, weight_shapes, client_ids)
        for cid in client_ids
    ]
    
    # Verify masks sum to zero
    mask_sum = [np.sum([m[i] for m in masks], axis=0) for i in range(len(weight_shapes))]
    total_mask_norm = sum(np.linalg.norm(m) for m in mask_sum)
    print(f"Sum of all masks norm: {total_mask_norm:.10f}")
    
    # Apply masks
    masked_weights = [sa.mask_weights(weights, m) for m in masks]
    print(f"Masked values: {[w[0][0] for w in masked_weights]}")
    
    # Aggregate (masks exactly cancel)
    aggregated = sa.aggregate_masked(masked_weights, client_ids, round_seed)
    aggregated_value = aggregated[0][0]
    
    print(f"Aggregated value: {aggregated_value:.6f}")
    print(f"Difference from original: {abs(aggregated_value - original_value):.10f}")
    
    if abs(aggregated_value - original_value) < 0.0001:  # Tighter tolerance
        print("✓ SUCCESS: Pairwise masks exactly cancel out")
        return True
    else:
        print("✗ FAIL: Pairwise masks did not cancel properly")
        return False

if __name__ == "__main__":
    success1 = test_simple_sa()
    success2 = test_pairwise_sa()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("All SA tests passed!")
    else:
        print("Some SA tests failed")
    print("=" * 60)