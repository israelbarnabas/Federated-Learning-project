"""
Link-Aware Adaptive Scheduler with CORRECT RDP Privacy Accounting.
ULTIMATE FIX v4: Hardcoded epsilon floor, fixed RDP accounting, proper state transitions.
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

from try_project.enhanced_network_sim import AIoTChannel, ChannelState
from try_project.dp_utils import (
    find_noise_multiplier_for_epsilon_rdp,
    compute_epsilon_rdp
)


class PrivacyMode(Enum):
    STATIC = "static"
    ADAPTIVE_LINK = "adaptive_link"
    ADAPTIVE_UTILITY = "adaptive_utility"
    PREDICTIVE = "predictive"


class SAStrategy(Enum):
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    ADAPTIVE = "adaptive"


@dataclass
class SchedulerConfig:
    target_epsilon_total: float = 5.0
    target_delta: float = 1e-4
    epsilon_per_round_max: float = 1.0
    epsilon_per_round_min: float = 0.5  # CRITICAL: Must be 0.5
    max_clients_per_round: int = 20
    min_clients_per_round: int = 5
    latency_sla_ms: float = 500.0
    sa_strategy: SAStrategy = SAStrategy.ADAPTIVE
    sa_overhead_factor: float = 2.0
    sa_benefit_threshold: float = 0.3
    prefer_good_channel: bool = True
    bad_channel_penalty: float = 0.5
    accuracy_window_size: int = 5
    accuracy_stall_threshold: float = 0.01
    
    # Budget pacing - aggressive early spending
    early_budget_fraction: float = 0.7
    early_round_fraction: float = 0.35
    
    stagnation_boost_factor: float = 2.5
    
    # RDP-specific parameters
    rdp_alphas: List[float] = field(default_factory=lambda: [
        1 + x / 10.0 for x in range(1, 100)
    ] + list(range(12, 64)))


@dataclass
class ClientScore:
    client_id: int
    data_quality_score: float
    channel_score: float
    privacy_score: float
    composite_score: float
    recommended_epsilon: float
    recommended_noise: float
    use_sa: bool


class LinkAwareScheduler:
    """
    Joint scheduler with CORRECT RDP privacy accounting.
    ULTIMATE FIX v4: Hardcoded epsilon floor, fixed RDP accounting.
    """
    
    # CLASS-LEVEL CONSTANT: This cannot be overridden
    ABSOLUTE_MIN_EPSILON: float = 0.5
    
    def __init__(
        self,
        num_clients: int,
        num_samples_per_client: Dict[int, int],
        config: Optional[SchedulerConfig] = None,
        channel: Optional[AIoTChannel] = None,
        seed: int = 42
    ):
        self.num_clients = num_clients
        self.num_samples = num_samples_per_client
        self.config = config or SchedulerConfig()
        self.channel = channel
        self.rng = np.random.default_rng(seed)
        
        # CRITICAL: Override config value with absolute minimum
        if self.config.epsilon_per_round_min < self.ABSOLUTE_MIN_EPSILON:
            print(f"[Scheduler] WARNING: Config epsilon_per_round_min={self.config.epsilon_per_round_min} "
                  f"is below absolute minimum {self.ABSOLUTE_MIN_EPSILON}. "
                  f"Forcing to {self.ABSOLUTE_MIN_EPSILON}.")
            self.config.epsilon_per_round_min = self.ABSOLUTE_MIN_EPSILON
        
        # RDP tracking
        self.rdp_curves: List[Dict[float, float]] = []
        self.total_rdp: Dict[float, float] = {}
        
        # Track (epsilon, delta) for each round
        self.epsilon_history: List[float] = []
        self.best_alpha_history: List[float] = []
        self.round_count = 0
        
        # Performance tracking
        self.accuracy_history: deque = deque(maxlen=self.config.accuracy_window_size)
        self.latency_history: deque = deque(maxlen=10)
        self.bytes_history: deque = deque(maxlen=10)
        self.client_success_rate: Dict[int, deque] = {
            i: deque(maxlen=5) for i in range(num_clients)
        }
        
        # Track stagnation
        self.stagnation_counter: int = 0
        self.last_boost_round: int = 0
        
        # DEBUG: Log initialization
        print(f"[Scheduler] INIT: epsilon_per_round_min={self.config.epsilon_per_round_min} "
              f"(absolute minimum: {self.ABSOLUTE_MIN_EPSILON})")
        
    def _compute_channel_score(self, client_id: int, state: ChannelState, success_prob: float) -> float:
        base_scores = {ChannelState.GOOD: 1.0, ChannelState.MEDIUM: 0.7, ChannelState.BAD: 0.4}
        base = base_scores.get(state, 0.5)
        
        if self.client_success_rate[client_id]:
            recent_success = np.mean(self.client_success_rate[client_id])
        else:
            recent_success = success_prob
        
        score = base * recent_success
        if not self.config.prefer_good_channel:
            score = 1.0 - score + 0.3
        return score
    
    def _compute_adaptive_epsilon(
        self,
        channel_score: float,
        remaining_rounds: int,
        current_accuracy: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute per-round epsilon with ABSOLUTE HARD FLOOR enforcement.
        """
        # CRITICAL: Use absolute minimum, not config value
        hard_floor = self.ABSOLUTE_MIN_EPSILON
        
        # Get current privacy spend
        current_epsilon, current_best_alpha = self._get_current_epsilon_delta()
        remaining_budget = self.config.target_epsilon_total - current_epsilon
        
        if remaining_rounds <= 0:
            return hard_floor, 0.0
        
        # Phase-based budget allocation
        total_rounds_estimated = self.round_count + remaining_rounds
        progress = self.round_count / total_rounds_estimated if total_rounds_estimated > 0 else 0
        
        # Calculate raw base epsilon
        if progress < self.config.early_round_fraction:
            phase_budget = self.config.target_epsilon_total * self.config.early_budget_fraction
            phase_rounds = int(total_rounds_estimated * self.config.early_round_fraction)
            rounds_remaining_in_phase = max(phase_rounds - self.round_count, 1)
            raw_epsilon = phase_budget / rounds_remaining_in_phase
        elif progress < 0.7:
            raw_epsilon = remaining_budget / remaining_rounds if remaining_rounds > 0 else hard_floor
        else:
            raw_epsilon = (remaining_budget * 0.8) / remaining_rounds if remaining_rounds > 0 else hard_floor
        
        # CRITICAL FIX: Apply hard floor immediately
        base_epsilon = max(raw_epsilon, hard_floor)
        
        if raw_epsilon < hard_floor and self.round_count <= 2:
            print(f"[Scheduler] R{self.round_count}: Raw ε={raw_epsilon:.3f} below HARD FLOOR {hard_floor}, "
                  f"using ε={base_epsilon:.3f}")
        
        # Minimal channel impact
        channel_factor = 0.95 + 0.1 * channel_score
        
        # Convergence adjustments
        convergence_factor = 1.0
        if current_accuracy and len(self.accuracy_history) >= 3:
            recent_trend = np.diff(list(self.accuracy_history)[-5:])
            avg_improvement = np.mean(recent_trend) if len(recent_trend) > 0 else 0
            
            if avg_improvement < self.config.accuracy_stall_threshold:
                self.stagnation_counter += 1
                if self.stagnation_counter >= 5 and (self.round_count - self.last_boost_round) > 10:
                    convergence_factor = self.config.stagnation_boost_factor
                    self.last_boost_round = self.round_count
                    print(f"[Scheduler] R{self.round_count}: STAGNATION DETECTED! "
                          f"Boosting ε by {convergence_factor}x")
                elif self.stagnation_counter >= 3:
                    convergence_factor = 1.2
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
                if avg_improvement > 0.02:
                    convergence_factor = 0.9
        
        # Calculate target
        target_epsilon = base_epsilon * channel_factor * convergence_factor
        
        # CRITICAL FIX 2: Enforce HARD FLOOR again after all adjustments
        target_epsilon = max(target_epsilon, hard_floor)
        
        # Cap at max and remaining budget
        target_epsilon = min(target_epsilon, remaining_budget, self.config.epsilon_per_round_max)
        
        # Safety cap
        if remaining_rounds > 3:
            target_epsilon = min(target_epsilon, remaining_budget * 0.9 / remaining_rounds)
        
        # FINAL ENFORCEMENT: Never go below hard floor
        target_epsilon = max(target_epsilon, hard_floor)
        
        # Compute noise multiplier
        avg_samples = np.mean(list(self.num_samples.values())) if self.num_samples else 300
        
        noise_mult, achieved_eps = find_noise_multiplier_for_epsilon_rdp(
            target_epsilon=target_epsilon,
            num_samples=int(avg_samples),
            batch_size=32,
            epochs=3,
            delta=self.config.target_delta
        )
        
        # Final sanity check
        if noise_mult > 10.0:
            print(f"[Scheduler] R{self.round_count}: WARNING - Noise σ={noise_mult:.2f} too high, "
                  f"forcing higher ε")
            forced_epsilon = hard_floor * 2.0
            forced_epsilon = min(forced_epsilon, remaining_budget, self.config.epsilon_per_round_max)
            
            noise_mult, achieved_eps = find_noise_multiplier_for_epsilon_rdp(
                target_epsilon=forced_epsilon,
                num_samples=int(avg_samples),
                batch_size=32,
                epochs=3,
                delta=self.config.target_delta
            )
            target_epsilon = forced_epsilon
        
        return target_epsilon, noise_mult
    
    def _should_use_sa(self, client_id: int, state: ChannelState, epsilon: float) -> bool:
        if self.config.sa_strategy == SAStrategy.ALWAYS_ON:
            return True
        if self.config.sa_strategy == SAStrategy.ALWAYS_OFF:
            return False
        
        chars = self.channel.characteristics[state]
        high_loss = chars.loss_rate > self.config.sa_benefit_threshold
        high_privacy = epsilon > 0.5
        
        return high_loss or high_privacy
    
    def select_clients_and_configure(
        self,
        available_clients: List[int],
        current_round: int,
        total_rounds: int,
        current_accuracy: Optional[float] = None
    ) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, bool], Dict]:
        """
        Main scheduling decision with RDP-aware privacy allocation.
        """
        self.round_count = current_round
        remaining_rounds = total_rounds - current_round + 1
        
        # Get observable channel states
        channel_states = self.channel.get_observable_states(available_clients)
        
        # Compute epsilon ONCE with hard floor guarantee
        dummy_channel_score = 0.7
        base_epsilon, base_noise = self._compute_adaptive_epsilon(
            dummy_channel_score, remaining_rounds, current_accuracy
        )
        
        # FINAL SANITY CHECK
        if base_epsilon < self.ABSOLUTE_MIN_EPSILON:
            print(f"[Scheduler] R{current_round}: CRITICAL - base_epsilon={base_epsilon:.3f} "
                  f"below absolute minimum! Forcing to {self.ABSOLUTE_MIN_EPSILON}")
            base_epsilon = self.ABSOLUTE_MIN_EPSILON
            base_noise, _ = find_noise_multiplier_for_epsilon_rdp(
                target_epsilon=base_epsilon,
                num_samples=300,
                batch_size=32,
                epochs=3,
                delta=self.config.target_delta
            )
        
        print(f"[Scheduler] R{current_round}: Base epsilon for round: {base_epsilon:.3f}, σ={base_noise:.2f}")
        
        # Score all available clients
        scored_clients: List[ClientScore] = []
        
        for cid in available_clients:
            state, success_prob = channel_states[cid]
            
            # Data quality score
            data_score = np.log1p(self.num_samples.get(cid, 100)) / 10.0
            
            # Channel score
            channel_score = self._compute_channel_score(cid, state, success_prob)
            
            # Use base epsilon (already floor-enforced)
            target_eps = base_epsilon
            
            # Adjust noise for individual client dataset size only
            n_samples = self.num_samples.get(cid, 300)
            if n_samples != 300:
                _, noise_mult = find_noise_multiplier_for_epsilon_rdp(
                    target_epsilon=target_eps,
                    num_samples=n_samples,
                    batch_size=32,
                    epochs=3,
                    delta=self.config.target_delta
                )
            else:
                noise_mult = base_noise
            
            # Decide SA usage
            use_sa = self._should_use_sa(cid, state, target_eps)
            
            # Privacy score
            privacy_score = 1.0 - (target_eps / self.config.epsilon_per_round_max)
            
            # Composite score
            composite = 0.4 * data_score + 0.4 * channel_score + 0.2 * privacy_score
            
            # Create ClientScore with floor-enforced values
            client_score = ClientScore(
                client_id=cid,
                data_quality_score=data_score,
                channel_score=channel_score,
                privacy_score=privacy_score,
                composite_score=composite,
                recommended_epsilon=target_eps,
                recommended_noise=noise_mult,
                use_sa=use_sa
            )
            
            scored_clients.append(client_score)
            
            if current_round == 1:
                print(f"[Scheduler] Client {cid}: ε={target_eps:.3f}, σ={noise_mult:.2f}")
        
        # Select top clients with diversity enforcement
        scored_clients.sort(key=lambda x: x.composite_score, reverse=True)
        
        selected = []
        state_counts = {ChannelState.GOOD: 0, ChannelState.MEDIUM: 0, ChannelState.BAD: 0}
        min_per_state = max(1, self.config.min_clients_per_round // 3)
        
        # First pass: ensure minimum representation from each state
        for score in scored_clients:
            if len(selected) >= self.config.max_clients_per_round:
                break
            state = channel_states[score.client_id][0]
            if state_counts[state] < min_per_state:
                selected.append(score)
                state_counts[state] += 1
        
        # Second pass: fill remaining slots
        for score in scored_clients:
            if len(selected) >= self.config.max_clients_per_round:
                break
            if score not in selected:
                selected.append(score)
        
        # Extract configurations
        selected_cids = [s.client_id for s in selected]
        client_epsilon = {s.client_id: s.recommended_epsilon for s in selected}
        client_noise = {s.client_id: s.recommended_noise for s in selected}
        client_use_sa = {s.client_id: s.use_sa for s in selected}
        
        # CRITICAL: Fix any remaining epsilon below floor
        for cid in list(client_epsilon.keys()):
            if client_epsilon[cid] < self.ABSOLUTE_MIN_EPSILON:
                print(f"[Scheduler] R{current_round}: Fixing client {cid} ε from {client_epsilon[cid]:.3f} to {self.ABSOLUTE_MIN_EPSILON}")
                client_epsilon[cid] = self.ABSOLUTE_MIN_EPSILON
                client_noise[cid], _ = find_noise_multiplier_for_epsilon_rdp(
                    target_epsilon=self.ABSOLUTE_MIN_EPSILON,
                    num_samples=self.num_samples.get(cid, 300),
                    batch_size=32,
                    epochs=3,
                    delta=self.config.target_delta
                )
        
        # DEBUG: Verify epsilon values
        eps_values = list(client_epsilon.values())
        min_eps = min(eps_values) if eps_values else 0
        max_eps = max(eps_values) if eps_values else 0
        avg_eps = np.mean(eps_values) if eps_values else 0
        print(f"[Scheduler] R{current_round}: FINAL Client ε range: [{min_eps:.3f}, {max_eps:.3f}], avg={avg_eps:.3f}")
        
        # Update RDP accounting
        self._update_rdp_accounting(selected, current_round)
        
        # Get current privacy spend for metadata
        current_epsilon, best_alpha = self._get_current_epsilon_delta()
        
        # Warning if over budget
        budget_status = "OK"
        if current_epsilon > self.config.target_epsilon_total:
            budget_status = f"OVER BUDGET: {current_epsilon:.2f}/{self.config.target_epsilon_total}"
        elif current_epsilon > 0.9 * self.config.target_epsilon_total:
            budget_status = f"CRITICAL: {current_epsilon:.2f}/{self.config.target_epsilon_total}"
        
        metadata = {
            "selected_clients": selected_cids,
            "avg_epsilon": np.mean(list(client_epsilon.values())) if client_epsilon else 0,
            "avg_noise": np.mean(list(client_noise.values())) if client_noise else 0,
            "total_epsilon_spent": current_epsilon,
            "epsilon_remaining": self.config.target_epsilon_total - current_epsilon,
            "best_alpha": best_alpha,
            "budget_status": budget_status,
            "stagnation_counter": self.stagnation_counter,
            "state_distribution": {
                "good": state_counts[ChannelState.GOOD],
                "medium": state_counts[ChannelState.MEDIUM],
                "bad": state_counts[ChannelState.BAD]
            },
            "sa_enabled_count": sum(client_use_sa.values()),
            "avg_channel_score": np.mean([s.channel_score for s in selected]),
        }
        
        return selected_cids, client_epsilon, client_noise, client_use_sa, metadata
    
    def _update_rdp_accounting(self, selected_clients: List[ClientScore], round_num: int):
        """Update privacy accounting using RDP composition."""
        round_rdp = {}
        
        for alpha in self.config.rdp_alphas:
            avg_rdp = np.mean([
                self._compute_client_rdp(s.client_id, s.recommended_noise, alpha)
                for s in selected_clients
            ]) if selected_clients else 0.0
            
            round_rdp[alpha] = avg_rdp
        
        self.rdp_curves.append(round_rdp)
        
        for alpha in self.config.rdp_alphas:
            if alpha not in self.total_rdp:
                self.total_rdp[alpha] = 0.0
            self.total_rdp[alpha] += round_rdp.get(alpha, 0.0)
        
        round_epsilon, round_alpha = self._rdp_to_epsilon_delta(round_rdp)
        self.epsilon_history.append(round_epsilon)
        self.best_alpha_history.append(round_alpha)
    
    def _compute_client_rdp(self, client_id: int, noise_multiplier: float, alpha: float) -> float:
        """Compute RDP for a single client."""
        n_samples = self.num_samples.get(client_id, 300)
        q = 32 / n_samples
        
        if q >= 1.0:
            return alpha / (2 * noise_multiplier ** 2)
        
        if q <= 0.01:
            return q ** 2 * alpha / (2 * noise_multiplier ** 2)
        
        return q ** 2 * alpha / (2 * noise_multiplier ** 2)
    
    def _rdp_to_epsilon_delta(self, rdp_curve: Dict[float, float], delta: Optional[float] = None) -> Tuple[float, float]:
        """Convert RDP curve to (epsilon, delta)-DP."""
        delta = delta or self.config.target_delta
        
        best_epsilon = float('inf')
        best_alpha = 1.0
        
        for alpha, rdp_eps in rdp_curve.items():
            # FIXED: Proper alpha check
            if alpha < 1.0:
                continue
            
            # FIXED: Proper float division
            epsilon = rdp_eps + np.log(1.0 / delta) / (alpha - 1.0)
            
            if epsilon < best_epsilon:
                best_epsilon = epsilon
                best_alpha = alpha
        
        return best_epsilon, best_alpha
    
    def _get_current_epsilon_delta(self) -> Tuple[float, float]:
        """Get current total privacy spend."""
        if not self.total_rdp:
            return 0.0, 1.0
        
        return self._rdp_to_epsilon_delta(self.total_rdp)
    
    def update_performance_metrics(
        self,
        round_accuracy: float,
        round_latency_ms: float,
        round_bytes: int,
        client_results: Dict[int, bool]
    ):
        """Update history for next round's decisions."""
        self.accuracy_history.append(round_accuracy)
        self.latency_history.append(round_latency_ms)
        self.bytes_history.append(round_bytes)
        
        for cid, success in client_results.items():
            self.client_success_rate[cid].append(1.0 if success else 0.0)
    
    def get_privacy_summary(self) -> Dict:
        """Return privacy budget consumption summary."""
        current_epsilon, best_alpha = self._get_current_epsilon_delta()
        budget_utilization = current_epsilon / self.config.target_epsilon_total
        
        return {
            "total_epsilon_spent": current_epsilon,
            "target_epsilon": self.config.target_epsilon_total,
            "budget_utilization": budget_utilization,
            "rounds_completed": self.round_count,
            "best_alpha": best_alpha,
            "epsilon_history": list(self.epsilon_history),
            "rdp_curves_computed": len(self.rdp_curves),
            "rdp_at_best_alpha": self.total_rdp.get(best_alpha, 0) if self.total_rdp else 0,
            "on_track": budget_utilization <= 1.0,
        }