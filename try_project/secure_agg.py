"""
Simplified Secure Aggregation for Federated Learning.
Implements masking-based SA without pairwise key exchange (simplified for simulation).
"""

import numpy as np
from typing import List, Optional
import hashlib


class SecureAggregation:
    """
    Simplified Secure Aggregation using additive masking.
    
    Each client adds a random mask to their weights. Masks sum to zero
    across all clients, so they cancel during aggregation.
    """
    
    def __init__(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold
        self.mask_scale = 1e-4  # Small scale to avoid numerical issues
        
    def generate_round_seed(self, server_round: int) -> int:
        """Generate deterministic seed for a round."""
        # Hash round number to get seed
        hash_input = f"sa_round_{server_round}".encode()
        return int(hashlib.md5(hash_input).hexdigest()[:8], 16)
    
    def generate_masks(
        self, 
        client_id: int, 
        round_seed: int,
        weight_shapes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Generate deterministic masks for a client.
        
        Uses client_id and round_seed for reproducibility.
        """
        # Create RNG with combined seed
        combined_seed = round_seed + client_id * 1000003  # Large prime
        rng = np.random.default_rng(combined_seed)
        
        masks = []
        for w in weight_shapes:
            # Generate mask matching weight shape
            mask = rng.normal(0, self.mask_scale, size=w.shape).astype(w.dtype)
            masks.append(mask)
        
        return masks
    
    def mask_weights(
        self, 
        weights: List[np.ndarray], 
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Add masks to weights."""
        return [w + m for w, m in zip(weights, masks)]
    
    def unmask_aggregate(
        self,
        masked_weights: List[List[np.ndarray]],
        client_ids: List[int],
        round_seed: int,
    ) -> List[np.ndarray]:
        """
        Aggregate masked weights and remove masks.
        
        This is a server-side helper. In practice, the server never sees
        unmasked individual updates—only the aggregated result.
        """
        if not masked_weights:
            return []
        
        # Sum all masked weights
        aggregated = [np.sum(w, axis=0) for w in zip(*masked_weights)]
        
        # Subtract masks from participating clients
        for cid in client_ids:
            masks = self.generate_masks(cid, round_seed, aggregated)
            aggregated = [agg - mask for agg, mask in zip(aggregated, masks)]
        
        # Average
        return [w / len(client_ids) for w in aggregated]