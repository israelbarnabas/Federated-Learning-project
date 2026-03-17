"""
Enhanced AIoT Network Simulator with 3-State Markov Channel, Pareto Latency,
Shared Medium Congestion, and Physical-Layer Inspired Dynamics.
FIXED: Only transition states for selected clients, not all clients.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import numpy as np
from collections import deque


class ChannelState(Enum):
    """Three-state AIoT channel model."""
    GOOD = "good"
    MEDIUM = "medium"
    BAD = "bad"


@dataclass
class LinkCharacteristics:
    loss_rate: float
    latency_xm: float
    latency_alpha: float
    bandwidth_mbps: float
    jitter_factor: float


@dataclass
class TransmissionResult:
    success: bool
    client_id: int
    state: ChannelState
    latency_ms: float
    transmit_time_s: float
    bytes_sent: int
    bytes_lost: int
    queue_delay_ms: float
    net_loss_rate: float
    bandwidth_mbps: float
    retransmissions: int
    dropped_by_congestion: bool


class AIoTChannel:
    """3-State Markov AIoT Channel with Pareto latency and shared medium."""
    
    DEFAULT_TRANSITIONS = np.array([
        [0.94, 0.05, 0.01],
        [0.10, 0.75, 0.15],
        [0.30, 0.50, 0.20],
    ])
    
    DEFAULT_CHARACTERISTICS = {
        ChannelState.GOOD: LinkCharacteristics(
            loss_rate=0.01, latency_xm=45.0, latency_alpha=3.0,
            bandwidth_mbps=10.0, jitter_factor=0.1
        ),
        ChannelState.MEDIUM: LinkCharacteristics(
            loss_rate=0.10, latency_xm=60.0, latency_alpha=2.5,
            bandwidth_mbps=5.0, jitter_factor=0.3
        ),
        ChannelState.BAD: LinkCharacteristics(
            loss_rate=0.35, latency_xm=100.0, latency_alpha=2.0,
            bandwidth_mbps=2.0, jitter_factor=0.6
        )
    }
    
    def __init__(
        self,
        num_clients: int,
        max_concurrent_transmissions: int = 15,
        transition_matrix: Optional[np.ndarray] = None,
        state_characteristics: Optional[Dict[ChannelState, LinkCharacteristics]] = None,
        queue_size: int = 2,
        enable_retransmission: bool = True,
        max_retransmissions: int = 2,
        seed: int = 42
    ):
        self.num_clients = num_clients
        self.K = max_concurrent_transmissions
        self.queue_size = queue_size
        self.enable_retransmission = enable_retransmission
        self.max_retransmissions = max_retransmissions
        
        self.rng = np.random.default_rng(seed)
        self.transition_matrix = transition_matrix or self.DEFAULT_TRANSITIONS
        self.characteristics = state_characteristics or self.DEFAULT_CHARACTERISTICS
        
        self.client_states: Dict[int, ChannelState] = {}
        self.state_history: Dict[int, List[ChannelState]] = {i: [] for i in range(num_clients)}
        self._initialize_states()
        
        self.client_queues: Dict[int, deque] = {i: deque(maxlen=queue_size) for i in range(num_clients)}
        self.queue_occupancy: Dict[int, int] = {i: 0 for i in range(num_clients)}
        
        self.round_metrics: List[Dict] = []
        self.total_bytes_sent = 0
        self.total_bytes_lost = 0
        self.total_transmissions = 0
        self.successful_transmissions = 0
        
    def _initialize_states(self):
        """Initialize channel states from steady-state distribution."""
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        steady = np.real(eigenvecs[:, np.isclose(eigenvals, 1)])
        steady = steady / steady.sum()
        steady = steady.flatten()
        
        states = [ChannelState.GOOD, ChannelState.MEDIUM, ChannelState.BAD]
        for cid in range(self.num_clients):
            self.client_states[cid] = self.rng.choice(states, p=steady)
    
    def _markov_step(self, client_id: int) -> ChannelState:
        """Perform one Markov transition for a single client."""
        current = self.client_states[client_id]
        state_idx = [ChannelState.GOOD, ChannelState.MEDIUM, ChannelState.BAD].index(current)
        probs = self.transition_matrix[state_idx]
        new_state = self.rng.choice([ChannelState.GOOD, ChannelState.MEDIUM, ChannelState.BAD], p=probs)
        self.client_states[client_id] = new_state
        self.state_history[client_id].append(new_state)
        return new_state
    
    def _sample_latency(self, state: ChannelState) -> float:
        """Sample latency from Pareto distribution."""
        chars = self.characteristics[state]
        pareto_sample = self.rng.pareto(chars.latency_alpha)
        latency = chars.latency_xm * (1 + pareto_sample)
        return float(latency)
    
    def simulate_round(
        self,
        selected_clients: List[int],
        update_size_bytes: int = 1_200_000,
        priority_func: Optional[Callable[[int], int]] = None
    ) -> Tuple[List[TransmissionResult], List[int]]:
        """
        Simulate one round of transmissions.
        FIXED: Only transition states for SELECTED clients.
        """
        # FIXED: Only transition states for selected clients, not all clients
        for cid in selected_clients:
            self._markov_step(cid)
        
        # Priority-based admission
        if priority_func:
            sorted_clients = sorted(selected_clients, key=priority_func, reverse=True)
        else:
            sorted_clients = list(selected_clients)
            self.rng.shuffle(sorted_clients)
        
        # Admit top K clients
        admitted_clients = sorted_clients[:self.K]
        dropped_by_congestion = set(sorted_clients[self.K:])
        
        # Simulate transmissions
        results = []
        for cid in admitted_clients:
            result = self._attempt_transmission(cid, update_size_bytes)
            results.append(result)
        
        # Record congestion drops
        for cid in dropped_by_congestion:
            results.append(TransmissionResult(
                success=False, client_id=cid, state=self.client_states[cid],
                latency_ms=0.0, transmit_time_s=0.0, bytes_sent=0,
                bytes_lost=update_size_bytes, queue_delay_ms=0.0,
                net_loss_rate=self.characteristics[self.client_states[cid]].loss_rate,
                bandwidth_mbps=self.characteristics[self.client_states[cid]].bandwidth_mbps,
                retransmissions=0, dropped_by_congestion=True
            ))
            self.total_bytes_lost += update_size_bytes
        
        successful_cids = [r.client_id for r in results if r.success]
        
        # Record metrics
        self.round_metrics.append({
            "attempted": len(selected_clients),
            "admitted": len(admitted_clients),
            "successful": len(successful_cids),
            "congestion_drops": len(dropped_by_congestion),
            "avg_latency_ms": np.mean([r.latency_ms for r in results]) if results else 0,
            "p95_latency_ms": np.percentile([r.latency_ms for r in results], 95) if results else 0,
        })
        
        return results, successful_cids
    
    def _attempt_transmission(self, client_id: int, update_size_bytes: int) -> TransmissionResult:
        """Attempt transmission for a single client."""
        state = self.client_states[client_id]
        chars = self.characteristics[state]
        
        # Calculate latency
        base_latency = self._sample_latency(state)
        queue_delay = self.queue_occupancy[client_id] * 10.0
        total_latency = base_latency + queue_delay
        
        # Calculate transmission time
        bandwidth_bps = chars.bandwidth_mbps * 1_000_000.0
        transmit_time = (update_size_bytes * 8.0) / bandwidth_bps
        
        # Attempt transmission with retransmissions
        success = True
        bytes_sent = 0
        bytes_lost = 0
        retrans_count = 0
        
        if self.enable_retransmission:
            for attempt in range(self.max_retransmissions + 1):
                self.total_transmissions += 1
                if self.rng.random() > chars.loss_rate:
                    bytes_sent = update_size_bytes * (attempt + 1)
                    self.successful_transmissions += 1
                    retrans_count = attempt
                    break
                else:
                    bytes_lost += update_size_bytes
                    retrans_count = attempt
            else:
                success = False
                bytes_lost = update_size_bytes * (self.max_retransmissions + 1)
        else:
            self.total_transmissions += 1
            if self.rng.random() > chars.loss_rate:
                bytes_sent = update_size_bytes
                self.successful_transmissions += 1
            else:
                success = False
                bytes_lost = update_size_bytes
        
        self.total_bytes_sent += bytes_sent
        self.total_bytes_lost += bytes_lost
        
        return TransmissionResult(
            success=success, client_id=client_id, state=state,
            latency_ms=total_latency, transmit_time_s=transmit_time + (total_latency / 1000.0),
            bytes_sent=bytes_sent, bytes_lost=bytes_lost, queue_delay_ms=queue_delay,
            net_loss_rate=chars.loss_rate, bandwidth_mbps=chars.bandwidth_mbps,
            retransmissions=retrans_count, dropped_by_congestion=False
        )
    
    def get_observable_states(self, client_ids: Optional[List[int]] = None) -> Dict[int, Tuple[ChannelState, float]]:
        """Get observable channel states for clients."""
        if client_ids is None:
            client_ids = list(range(self.num_clients))
        observable = {}
        for cid in client_ids:
            state = self.client_states[cid]
            chars = self.characteristics[state]
            est_success = 1 - (chars.loss_rate ** (self.max_retransmissions + 1)) if self.enable_retransmission else 1 - chars.loss_rate
            observable[cid] = (state, est_success)
        return observable
    
    def get_channel_statistics(self) -> Dict:
        """Get channel statistics."""
        if not self.round_metrics:
            return {}
        recent = self.round_metrics[-10:]
        return {
            "avg_success_rate": self.successful_transmissions / max(self.total_transmissions, 1),
            "state_distribution": {
                "good": sum(1 for s in self.client_states.values() if s == ChannelState.GOOD) / self.num_clients,
                "medium": sum(1 for s in self.client_states.values() if s == ChannelState.MEDIUM) / self.num_clients,
                "bad": sum(1 for s in self.client_states.values() if s == ChannelState.BAD) / self.num_clients,
            },
            "recent_avg_latency_ms": np.mean([r["avg_latency_ms"] for r in recent]),
            "recent_p95_latency_ms": np.mean([r["p95_latency_ms"] for r in recent]),
        }


class SimpleGilbertElliott:
    """Backward compatibility wrapper."""
    def __init__(self, **kwargs):
        self.channel = AIoTChannel(
            num_clients=kwargs.get("num_clients", 30),
            max_concurrent_transmissions=kwargs.get("max_concurrent", 30),
            seed=kwargs.get("seed", 42)
        )
        loss_bad = kwargs.get("loss_bad", 0.25)
        self.channel.characteristics[ChannelState.BAD].loss_rate = loss_bad
        
    def simulate_update_transmission(self, client_id: int, update_size_bytes: int):
        results, _ = self.channel.simulate_round([client_id], update_size_bytes)
        r = results[0]
        return r.success, {
            "loss_rate": r.net_loss_rate,
            "latency_ms": r.latency_ms,
            "state": r.state.value,
            "bandwidth_mbps": r.bandwidth_mbps,
            "transmit_time_s": r.transmit_time_s,
            "retrans_factor": 1 + r.retransmissions,
        }