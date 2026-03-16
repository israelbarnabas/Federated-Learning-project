"""
Comprehensive test suite for enhanced AIoT FL system.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from try_project.enhanced_network_sim import AIoTChannel, ChannelState
from try_project.enhanced_secure_agg import SecureAggregation
from try_project.adaptive_scheduler import LinkAwareScheduler, SchedulerConfig
from try_project.dp_utils import find_noise_multiplier_for_epsilon


def test_markov_steady_state():
    print("\n" + "="*60)
    print("TEST 1: Markov Chain Steady-State Distribution")
    print("="*60)
    
    channel = AIoTChannel(num_clients=1000, max_concurrent_transmissions=100, seed=42)
    
    for _ in range(50):
        for cid in range(1000):
            channel._markov_step(cid)
    
    from collections import Counter
    state_counts = Counter(channel.client_states.values())
    total = sum(state_counts.values())
    
    good_pct = state_counts[ChannelState.GOOD] / total
    medium_pct = state_counts[ChannelState.MEDIUM] / total
    bad_pct = state_counts[ChannelState.BAD] / total
    
    print(f"Empirical: GOOD={good_pct:.3f}, MEDIUM={medium_pct:.3f}, BAD={bad_pct:.3f}")
    print("Expected:  GOOD~0.55, MEDIUM~0.30, BAD~0.15")
    
    assert 0.50 < good_pct < 0.60 and 0.25 < medium_pct < 0.35 and 0.10 < bad_pct < 0.20
    print("✓ PASS")
    return True


def test_pareto_latency():
    print("\n" + "="*60)
    print("TEST 2: Pareto Latency Distribution (Heavy Tails)")
    print("="*60)
    
    channel = AIoTChannel(num_clients=100, max_concurrent_transmissions=50, seed=42)
    
    for state in [ChannelState.GOOD, ChannelState.MEDIUM, ChannelState.BAD]:
        latencies = []
        for _ in range(5000):
            channel.client_states[0] = state
            latencies.append(channel._sample_latency(state))
        
        latencies = np.array(latencies)
        p99 = np.percentile(latencies, 99)
        mean = np.mean(latencies)
        
        print(f"{state.value.upper()}: mean={mean:.1f}ms, P99={p99:.1f}ms, ratio={p99/mean:.2f}x")
        
        if state == ChannelState.BAD:
            assert p99/mean > 2.0, "Heavy tail expected for BAD state"
    
    print("✓ PASS")
    return True


def test_secure_aggregation():
    print("\n" + "="*60)
    print("TEST 3: Secure Aggregation Mask Cancellation")
    print("="*60)
    
    sa = SecureAggregation(num_clients=10, threshold=3, mask_scale=1e-3)
    round_seed = sa.generate_round_seed(1)
    
    weight_shapes = [(100, 50), (50,)]
    client_ids = list(range(5))
    
    all_masks = [sa.generate_masks(cid, round_seed, weight_shapes, client_ids)[0] for cid in client_ids]
    mask_sum = [np.sum([client_masks[i] for client_masks in all_masks], axis=0) for i in range(len(weight_shapes))]
    total_norm = sum(np.linalg.norm(m) for m in mask_sum)
    
    print(f"Mask sum norm: {total_norm:.2e} (should be ~0)")
    assert total_norm < 1e-10, "Masks must cancel exactly"
    
    fake_weights = [np.ones((100, 50), dtype=np.float32) * 0.5, np.ones((50,), dtype=np.float32) * 0.3]
    masked_weights = []
    for cid in client_ids:
        masks, _ = sa.generate_masks(cid, round_seed, weight_shapes, client_ids)
        masked_weights.append(sa.mask_weights(fake_weights, masks))
    
    aggregated = sa.aggregate_masked(masked_weights, client_ids, round_seed)
    
    assert abs(aggregated[0][0, 0] - 0.5) < 1e-6
    assert abs(aggregated[1][0] - 0.3) < 1e-6
    
    print("✓ PASS: Masks cancel, aggregation correct")
    return True


def test_adaptive_scheduler():
    print("\n" + "="*60)
    print("TEST 4: Link-Aware Adaptive Scheduler")
    print("="*60)
    
    num_clients = 30
    channel = AIoTChannel(num_clients=num_clients, max_concurrent_transmissions=15, seed=42)
    
    for cid in range(num_clients):
        if cid < 10:
            channel.client_states[cid] = ChannelState.GOOD
        elif cid < 20:
            channel.client_states[cid] = ChannelState.MEDIUM
        else:
            channel.client_states[cid] = ChannelState.BAD
    
    config = SchedulerConfig(
        target_epsilon_total=5.0,
        epsilon_per_round_max=1.0,
        epsilon_per_round_min=0.1,
        max_clients_per_round=15,
        min_clients_per_round=5,
        prefer_good_channel=True
    )
    
    scheduler = LinkAwareScheduler(
        num_clients=num_clients,
        num_samples_per_client={i: 100 + i * 10 for i in range(num_clients)},
        config=config,
        channel=channel,
        seed=42
    )
    
    available = list(range(num_clients))
    selected, epsilons, noise_multipliers, use_sa, metadata = scheduler.select_clients_and_configure(
        available, current_round=1, total_rounds=10, current_accuracy=0.3
    )
    
    print(f"Selected {len(selected)} clients: {selected[:10]}...")
    print(f"State distribution: {metadata['state_distribution']}")
    print(f"Avg epsilon: {metadata['avg_epsilon']:.3f}")
    
    assert len(selected) <= config.max_clients_per_round
    assert len(selected) >= config.min_clients_per_round
    
    print("✓ PASS")
    return True


def test_congestion():
    print("\n" + "="*60)
    print("TEST 5: Shared Medium Congestion (Hard Cap K)")
    print("="*60)
    
    channel = AIoTChannel(num_clients=30, max_concurrent_transmissions=10, seed=42)
    all_clients = list(range(30))
    
    results, successful = channel.simulate_round(all_clients, update_size_bytes=1000000)
    
    admitted = len([r for r in results if not r.dropped_by_congestion])
    congestion_drops = len([r for r in results if r.dropped_by_congestion])
    
    print(f"Attempted: 30, Admitted: {admitted}, Congestion drops: {congestion_drops}")
    assert admitted <= 10, "Hard cap K must be enforced"
    assert congestion_drops == 20, "Exactly 20 should be dropped"
    
    print("✓ PASS")
    return True


def test_privacy_noise():
    print("\n" + "="*60)
    print("TEST 6: Privacy Noise Scaling")
    print("="*60)
    
    epsilons = [0.5, 1.0, 2.0, 5.0]
    noise_values = []
    
    for eps in epsilons:
        noise_mult, _ = find_noise_multiplier_for_epsilon(
            target_epsilon=eps, num_samples=300, batch_size=32, epochs=3, delta=1e-4
        )
        noise_values.append(noise_mult)
        print(f"ε={eps:.1f} -> σ={noise_mult:.4f}")
    
    for i in range(len(noise_values) - 1):
        if epsilons[i] < epsilons[i+1]:
            assert noise_values[i] > noise_values[i+1], "Noise should decrease with increasing epsilon"
    
    print("✓ PASS")
    return True


def run_all_tests():
    print("="*70)
    print("ENHANCED AIoT FL SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Markov Steady State", test_markov_steady_state),
        ("Pareto Latency", test_pareto_latency),
        ("SA Mask Cancellation", test_secure_aggregation),
        ("Adaptive Scheduler", test_adaptive_scheduler),
        ("Congestion Handling", test_congestion),
        ("Privacy Noise Scaling", test_privacy_noise),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("-"*70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready for experiments.")
    else:
        print("\n⚠ Some tests failed.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)