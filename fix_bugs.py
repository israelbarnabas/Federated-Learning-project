#!/usr/bin/env python3
"""Apply critical bug fixes to dp_utils.py and secure_agg.py"""

import re

# Fix 1: dp_utils.py - add missing keys to return dict
with open('try_project/dp_utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already fixed
if '"log_term": log_term,' not in content:
    # Add the missing lines before the closing brace
    old_return = '''    return epsilon, {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "sampling_rate": sampling_rate,
        "noise_multiplier": noise_multiplier,
        "epsilon": epsilon,
    }'''
    
    new_return = '''    return epsilon, {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "sampling_rate": sampling_rate,
        "noise_multiplier": noise_multiplier,
        "epsilon": epsilon,
        "log_term": log_term,
        "composition_factor": composition_factor,
    }'''
    
    content = content.replace(old_return, new_return)
    
    with open('try_project/dp_utils.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed dp_utils.py")
else:
    print("dp_utils.py already fixed")

# Fix 2: secure_agg.py - handle numpy array shapes
with open('try_project/secure_agg.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix generate_masks
old_generate = '''    def generate_masks(
        self, 
        client_id: int, 
        round_seed: int,
        weight_shapes: List[Tuple],
    ) -> List[np.ndarray]:
        """
        Generate masks for a client using deterministic PRNG.
        
        Each client's masks are derived from:
        - Global round seed (known to all)
        - Client ID (unique per client)
        
        The key: sum of all client masks ≈ 0 (by design of shared randomness)
        """
        client_seed = self.generate_client_seed(client_id, round_seed)
        rng = np.random.default_rng(client_seed)
        
        masks = []
        for shape in weight_shapes:
            # Generate mask with small scale
            mask = rng.normal(0, self.mask_scale, size=shape).astype(np.float32)
            masks.append(mask)
        
        return masks'''

new_generate = '''    def generate_masks(
        self, 
        client_id: int, 
        round_seed: int,
        weight_shapes: List[Tuple],
    ) -> List[np.ndarray]:
        """
        Generate masks for a client using deterministic PRNG.
        
        Each client's masks are derived from:
        - Global round seed (known to all)
        - Client ID (unique per client)
        
        The key: sum of all client masks ≈ 0 (by design of shared randomness)
        """
        client_seed = self.generate_client_seed(client_id, round_seed)
        rng = np.random.default_rng(client_seed)
        
        masks = []
        for shape in weight_shapes:
            # FIX: Ensure shape is a tuple, not numpy array
            if isinstance(shape, np.ndarray):
                shape = tuple(shape.astype(int))
            elif not isinstance(shape, tuple):
                shape = tuple(shape)
                
            # Generate mask with small scale
            mask = rng.normal(0, self.mask_scale, size=shape).astype(np.float32)
            masks.append(mask)
        
        return masks'''

if new_generate not in content:
    content = content.replace(old_generate, new_generate)
    with open('try_project/secure_agg.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed secure_agg.py generate_masks")
else:
    print("secure_agg.py already fixed")

print("\nAll fixes applied! Run tests again:")
print("   python test_dp_setup.py")
print("   python test_sa.py")