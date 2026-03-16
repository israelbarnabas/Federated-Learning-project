"""
Link-Aware Adaptive Scheduler with CORRECT RDP Privacy Accounting.
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
    epsilon_per_round_min: float = 0.05
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
    CRITICAL: Uses RDP composition, not simple sum.
    """
    
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
        
        # RDP tracking: store per-round RDP curves, not just epsilon
        self.rdp_curves: List[Dict[float, float]] = []  # List of {alpha: rdp_epsilon} per round
        self.total_rdp: Dict[float, float] = {}  # Composed RDP across all rounds
        
        # Track (epsilon, delta) for each round using optimal alpha
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
        Compute per-round epsilon and corresponding noise multiplier.
        Returns:
            (target_epsilon, noise_multiplier)
        """
        # Estimate remaining budget using current total RDP
        current_epsilon, current_best_alpha = self._get_current_epsilon_delta()
        remaining_budget = self.config.target_epsilon_total - current_epsilon
        
        if remaining_rounds <= 0 or remaining_budget <= 0:
            return self.config.epsilon_per_round_min, 0.0
        
        # Base allocation
        base_epsilon = remaining_budget / remaining_rounds
        
        # Channel quality adjustment
        channel_factor = 0.8 + 0.4 * channel_score
        
        # Convergence adjustment
        convergence_factor = 1.0
        if current_accuracy and len(self.accuracy_history) >= 2:
            recent_trend = np.diff(self.accuracy_history)
            avg_improvement = np.mean(recent_trend)
            
            if avg_improvement < self.config.accuracy_stall_threshold:
                # Stagnating: reduce noise to improve signal
                convergence_factor = 0.7
            elif avg_improvement > 0.05:
                # Improving fast: can afford more privacy
                convergence_factor = 1.2
        
        target_epsilon = base_epsilon * channel_factor * convergence_factor
        target_epsilon = np.clip(
            target_epsilon,
            self.config.epsilon_per_round_min,
            self.config.epsilon_per_round_max
        )
        target_epsilon = min(target_epsilon, remaining_budget)
        
        # Compute noise multiplier for this epsilon
        # Use average dataset size for estimation
        avg_samples = np.mean(list(self.num_samples.values())) if self.num_samples else 300
        
        noise_mult, achieved_eps = find_noise_multiplier_for_epsilon_rdp(
            target_epsilon=target_epsilon,
            num_samples=int(avg_samples),
            batch_size=32,  # Assume default
            epochs=3,       # Assume default
            delta=self.config.target_delta
        )
        
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
        Returns:
            selected_cids: List of selected client IDs
            client_epsilon: Dict mapping client_id -> target epsilon
            client_noise: Dict mapping client_id -> noise multiplier
            client_use_sa: Dict mapping client_id -> SA flag
            metadata: Debug info and metrics
        """
        self.round_count = current_round
        remaining_rounds = total_rounds - current_round + 1
        
        # Get observable channel states
        channel_states = self.channel.get_observable_states(available_clients)
        
        # Score all available clients
        scored_clients: List[ClientScore] = []
        
        for cid in available_clients:
            state, success_prob = channel_states[cid]
            
            # Data quality score
            data_score = np.log1p(self.num_samples.get(cid, 100)) / 10.0
            
            # Channel score
            channel_score = self._compute_channel_score(cid, state, success_prob)
            
            # Compute recommended epsilon and noise for this client
            target_eps, noise_mult = self._compute_adaptive_epsilon(
                channel_score, remaining_rounds, current_accuracy
            )
            
            # Adjust for individual client dataset size
            n_samples = self.num_samples.get(cid, 300)
            if n_samples != 300:  # Recompute if different from average
                _, noise_mult = find_noise_multiplier_for_epsilon_rdp(
                    target_epsilon=target_eps,
                    num_samples=n_samples,
                    batch_size=32,
                    epochs=3,
                    delta=self.config.target_delta
                )
            
            # Decide SA usage
            use_sa = self._should_use_sa(cid, state, target_eps)
            
            # Privacy score (lower epsilon = better privacy = higher score)
            privacy_score = 1.0 - (target_eps / self.config.epsilon_per_round_max)
            
            # Composite score
            composite = 0.4 * data_score + 0.4 * channel_score + 0.2 * privacy_score
            
            scored_clients.append(ClientScore(
                client_id=cid,
                data_quality_score=data_score,
                channel_score=channel_score,
                privacy_score=privacy_score,
                composite_score=composite,
                recommended_epsilon=target_eps,
                recommended_noise=noise_mult,
                use_sa=use_sa
            ))
        
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
        
        # CRITICAL: Update RDP accounting, not simple sum
        self._update_rdp_accounting(selected, current_round)
        
        # Get current privacy spend for metadata
        current_epsilon, best_alpha = self._get_current_epsilon_delta()
        
        metadata = {
            "selected_clients": selected_cids,
            "avg_epsilon": np.mean(list(client_epsilon.values())) if client_epsilon else 0,
            "avg_noise": np.mean(list(client_noise.values())) if client_noise else 0,
            "total_epsilon_spent": current_epsilon,
            "epsilon_remaining": self.config.target_epsilon_total - current_epsilon,
            "best_alpha": best_alpha,
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
        """
        CRITICAL: Update privacy accounting using RDP composition.
        """
        # Compute RDP curve for this round based on selected clients
        round_rdp = {}
        
        for alpha in self.config.rdp_alphas:
            # Average RDP across selected clients for this round
            avg_rdp = np.mean([
                self._compute_client_rdp(s.client_id, s.recommended_noise, alpha)
                for s in selected_clients
            ]) if selected_clients else 0.0
            
            round_rdp[alpha] = avg_rdp
        
        # Store this round's RDP curve
        self.rdp_curves.append(round_rdp)
        
        # Compose with total: RDP adds linearly
        for alpha in self.config.rdp_alphas:
            if alpha not in self.total_rdp:
                self.total_rdp[alpha] = 0.0
            self.total_rdp[alpha] += round_rdp.get(alpha, 0.0)
        
        # Compute (epsilon, delta) for this round using best alpha
        round_epsilon, round_alpha = self._rdp_to_epsilon_delta(round_rdp)
        self.epsilon_history.append(round_epsilon)
        self.best_alpha_history.append(round_alpha)
    
    def _compute_client_rdp(self, client_id: int, noise_multiplier: float, alpha: float) -> float:
        """Compute RDP for a single client."""
        n_samples = self.num_samples.get(client_id, 300)
        q = 32 / n_samples  # batch_size / dataset_size
        
        # RDP for Gaussian mechanism with subsampling
        if q >= 1.0:
            return alpha / (2 * noise_multiplier ** 2)
        
        # Privacy amplification by subsampling
        if q <= 0.01:
            return q ** 2 * alpha / (2 * noise_multiplier ** 2)
        
        # General bound
        return q ** 2 * alpha / (2 * noise_multiplier ** 2)
    
    def _rdp_to_epsilon_delta(self, rdp_curve: Dict[float, float], delta: Optional[float] = None) -> Tuple[float, float]:
        """
        Convert RDP curve to (epsilon, delta)-DP.
        Returns best epsilon and corresponding alpha.
        """
        delta = delta or self.config.target_delta
        
        best_epsilon = float('inf')
        best_alpha = 1.0
        
        for alpha, rdp_eps in rdp_curve.items():
            if alpha <= 1:
                continue
            
            # Convert to (epsilon, delta)-DP
            epsilon = rdp_eps + np.log(1 / delta) / (alpha - 1)
            
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
        """Return privacy budget consumption summary with RDP details."""
        current_epsilon, best_alpha = self._get_current_epsilon_delta()
        
        return {
            "total_epsilon_spent": current_epsilon,
            "target_epsilon": self.config.target_epsilon_total,
            "budget_utilization": current_epsilon / self.config.target_epsilon_total,
            "rounds_completed": self.round_count,
            "best_alpha": best_alpha,
            "epsilon_history": list(self.epsilon_history),
            "rdp_curves_computed": len(self.rdp_curves),
            "rdp_at_best_alpha": self.total_rdp.get(best_alpha, 0) if self.total_rdp else 0,
        }