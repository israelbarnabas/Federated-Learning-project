"""
Minimal Gilbert-Elliott channel simulator - NO adaptations.
Just injects losses to measure baseline KPI impact.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class NetworkStats:
    client_id: int
    loss_rate: float
    latency_ms: float
    state: str
    bandwidth_mbps: float


class SimpleGilbertElliott:
    def __init__(
        self,
        p_gb: float = 0.05,
        p_bg: float = 0.95,
        loss_good: float = 0.01,
        loss_bad: float = 0.25,
        base_latency_ms: float = 50.0,
        bandwidth_good_mbps: float = 10.0,
        bandwidth_bad_factor: float = 0.5,
        seed: int = 42,
    ):
        self.p_gb = p_gb
        self.p_bg = p_bg
        self.loss_good = loss_good
        self.loss_bad = loss_bad
        self.base_latency_ms = base_latency_ms
        self.bandwidth_good = bandwidth_good_mbps
        self.bandwidth_bad_factor = bandwidth_bad_factor
        
        self.client_states: Dict[int, str] = {}
        self.rng = np.random.default_rng(seed)
        
    def get_state(self, client_id: int) -> str:
        if client_id not in self.client_states:
            self.client_states[client_id] = "good"
            return "good"
        return self.client_states[client_id]
    
    def _transition(self, client_id: int) -> str:
        current = self.get_state(client_id)
        r = self.rng.random()
        
        if current == "good":
            if r < self.p_gb:
                self.client_states[client_id] = "bad"
                return "bad"
        else:
            if r < self.p_bg:
                self.client_states[client_id] = "good"
                return "good"
        
        return current
    
    def sample_channel(self, client_id: int) -> NetworkStats:
        state = self._transition(client_id)
        loss_rate = self.loss_good if state == "good" else self.loss_bad
        
        if state == "bad":
            latency = self.base_latency_ms * 2.0 + self.rng.uniform(-10, 10)
        else:
            latency = self.base_latency_ms + self.rng.uniform(-10, 10)
        latency = max(10.0, latency)
        
        bw = self.bandwidth_good * (self.bandwidth_bad_factor if state == "bad" else 1.0)
        
        return NetworkStats(
            client_id=client_id,
            loss_rate=loss_rate,
            latency_ms=latency,
            state=state,
            bandwidth_mbps=bw,
        )
    
    def simulate_update_transmission(
        self,
        client_id: int,
        update_size_bytes: int = 1_200_000,
    ):
        stats = self.sample_channel(client_id)
        
        bw_bps = stats.bandwidth_mbps * 1_000_000.0
        base_time = (update_size_bytes * 8.0) / bw_bps
        latency_s = stats.latency_ms / 1000.0
        
        retrans_factor = 1.0 / (1.0 - stats.loss_rate) if stats.loss_rate < 1.0 else 10.0
        total_time = base_time * retrans_factor + latency_s
        
        success = self.rng.random() > stats.loss_rate
        
        metrics = {
            "success": success,
            "state": stats.state,
            "loss_rate": stats.loss_rate,
            "latency_ms": stats.latency_ms,
            "bandwidth_mbps": stats.bandwidth_mbps,
            "transmit_time_s": total_time,
            "retrans_factor": retrans_factor,
        }
        
        return success, metrics