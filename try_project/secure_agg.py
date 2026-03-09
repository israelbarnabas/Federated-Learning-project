"""
Secure Aggregation controller for FL.
ON: Adds 2x communication overhead for cryptographic masks.
OFF: Plaintext aggregation (faster, no overhead).
"""

class SecureAggregationController:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.bytes_overhead = 2.0 if enabled else 1.0
    
    def should_enable(self, loss_rate: float, latency_ms: float) -> bool:
        """Dynamic decision: only use SA when link quality is good."""
        if not self.enabled:
            return False
        # Don't waste bytes on lossy links
        if loss_rate > 0.15:
            return False
        if latency_ms > 100:
            return False
        return True
    
    def get_bytes_multiplier(self) -> float:
        return self.bytes_overhead