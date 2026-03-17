#!/usr/bin/env python3
"""
Diagnostic script for privacy budget and noise analysis.
Run this to understand the epsilon-noise relationship in your setup.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from try_project.dp_utils import find_noise_multiplier_for_epsilon_rdp

def analyze_epsilon_noise_tradeoff():
    """Analyze how epsilon affects noise multiplier."""
    
    # Your setup parameters
    num_samples = 723  # Average from your logs
    batch_size = 32
    epochs = 3
    delta = 1e-4
    
    # Test epsilon range
    epsilons = np.linspace(0.1, 3.0, 50)
    noise_multipliers = []
    
    print("Epsilon vs Noise Multiplier Analysis")
    print("=" * 60)
    print(f"Setup: n={num_samples}, batch={batch_size}, epochs={epochs}, δ={delta}")
    print("-" * 60)
    
    for eps in epsilons:
        noise_mult, achieved = find_noise_multiplier_for_epsilon_rdp(
            target_epsilon=eps,
            num_samples=num_samples,
            batch_size=batch_size,
            epochs=epochs,
            delta=delta
        )
        noise_multipliers.append(noise_mult)
        
        # Mark critical thresholds
        if abs(eps - 0.3) < 0.02:
            print(f"ε={eps:.3f} → σ={noise_mult:.3f}  <-- MIN_USEFUL_EPSILON threshold (BROKEN)")
        elif abs(eps - 0.5) < 0.02:
            print(f"ε={eps:.3f} → σ={noise_mult:.3f}  <-- RECOMMENDED minimum (FIXED)")
        elif abs(eps - 1.0) < 0.02:
            print(f"ε={eps:.3f} → σ={noise_mult:.3f}  <-- Good learning zone")
        elif abs(eps - 2.0) < 0.02:
            print(f"ε={eps:.3f} → σ={noise_mult:.3f}  <-- Fast convergence")
    
    print("-" * 60)
    print("\nKey Insights:")
    print("• σ > 10: Learning is essentially impossible (noise dominates signal)")
    print("• σ = 5-10: Very slow learning, need many rounds")
    print("• σ = 2-5: Moderate learning, acceptable with FedProx")
    print("• σ = 1-2: Good learning rate")
    print("• σ < 1: Fast convergence but higher privacy cost")
    
    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epsilons, noise_multipliers, 'b-', linewidth=2, label='Noise Multiplier σ')
        ax.axhline(y=10, color='r', linestyle='--', label='σ=10 (learning limit)')
        ax.axhline(y=5, color='orange', linestyle='--', label='σ=5 (slow learning)')
        ax.axhline(y=2, color='green', linestyle='--', label='σ=2 (good learning)')
        ax.axvline(x=0.3, color='purple', linestyle=':', label='ε=0.3 (old minimum)')
        ax.axvline(x=0.5, color='darkgreen', linestyle=':', label='ε=0.5 (new minimum)')
        
        ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax.set_ylabel('Noise Multiplier (σ)', fontsize=12)
        ax.set_title('Privacy-Noise Tradeoff for Your FL Setup', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 15)
        
        plt.tight_layout()
        plt.savefig('privacy_noise_analysis.png', dpi=150)
        print(f"\nPlot saved to privacy_noise_analysis.png")
    except ImportError:
        print("\n(matplotlib not installed, skipping plot)")
    
    return epsilons, noise_multipliers

def calculate_budget_pacing_scenarios():
    """Compare different budget pacing strategies."""
    
    print("\n\nBudget Pacing Scenarios (100 rounds, ε_total=5.0)")
    print("=" * 70)
    
    scenarios = [
        ("OLD (broken)", 0.4, 0.3, "40% over 30% rounds"),
        ("NEW (fixed)", 0.6, 0.4, "60% over 40% rounds"),
        ("Aggressive", 0.7, 0.3, "70% over 30% rounds"),
        ("Conservative", 0.5, 0.5, "50% over 50% rounds"),
    ]
    
    for name, budget_frac, round_frac, desc in scenarios:
        phase_budget = 5.0 * budget_frac
        phase_rounds = int(100 * round_frac)
        eps_per_round = phase_budget / phase_rounds
        
        # Calculate resulting noise
        noise, _ = find_noise_multiplier_for_epsilon_rdp(
            target_epsilon=max(eps_per_round, 0.5),
            num_samples=723,
            batch_size=32,
            epochs=3,
            delta=1e-4
        )
        
        status = "✓ GOOD" if eps_per_round >= 0.5 else "✗ TOO LOW"
        print(f"{name:15} | {desc:25} | ε/round={eps_per_round:.3f} | σ={noise:.2f} {status}")
    
    print("-" * 70)

if __name__ == "__main__":
    analyze_epsilon_noise_tradeoff()
    calculate_budget_pacing_scenarios()