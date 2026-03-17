"""
Enhanced Network Wrapper with Link-Aware Adaptive Scheduling.
FIXED VERSION - Properly propagates corrected epsilon values to clients.
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import numpy as np

from try_project.enhanced_network_sim import AIoTChannel, ChannelState, TransmissionResult
from try_project.adaptive_scheduler import LinkAwareScheduler, SchedulerConfig, SAStrategy
from try_project.enhanced_secure_agg import SecureAggregation, create_sa_for_federation

class LinkAwareNetworkWrapper:
    def __init__(
        self,
        base_strategy: Strategy,
        channel: AIoTChannel,
        scheduler_config: Optional[SchedulerConfig] = None,
        model_bytes: int = 1_200_000,
        use_adaptive_dp: bool = True,
        use_adaptive_sa: bool = True,
        total_rounds: int = 10,
        seed: int = 42
    ):
        self.base = base_strategy
        self.channel = channel
        self.model_bytes = model_bytes
        self.use_adaptive_dp = use_adaptive_dp
        self.use_adaptive_sa = use_adaptive_sa
        self.total_rounds = total_rounds
        self.rng = np.random.default_rng(seed)
        
        # CRITICAL: Create deterministic mapping from Flower CID -> sequential ID
        self._client_id_map: Dict[str, int] = {}
        self._reverse_map: Dict[int, str] = {}  # For debugging
        self._next_available_id = 0
        
        # Initialize scheduler with default num_samples to avoid empty dict issues
        initial_num_samples = {i: 300 for i in range(channel.num_clients)}  # Default 300 samples
        
        self.scheduler = None
        if use_adaptive_dp or use_adaptive_sa:
            self.scheduler = LinkAwareScheduler(
                num_clients=channel.num_clients,
                num_samples_per_client=initial_num_samples,
                config=scheduler_config or SchedulerConfig(),
                channel=channel,
                seed=seed
            )
            # DEBUG: Verify scheduler config
            if self.scheduler.config:
                print(f"[Wrapper] Scheduler initialized with epsilon_per_round_min="
                      f"{self.scheduler.config.epsilon_per_round_min}")
        
        # Only create SA if explicitly enabled
        self.sa: Optional[SecureAggregation] = None
        if use_adaptive_sa:
            self.sa = create_sa_for_federation(
                num_clients=channel.num_clients,
                threshold=max(2, channel.K // 2),
                use_secret_sharing=True
            )
            print(f"[Wrapper] Secure Aggregation ENABLED with threshold {max(2, channel.K // 2)}")
        else:
            print(f"[Wrapper] Secure Aggregation DISABLED")
        
        self.round_metrics: List[Dict] = []
        self.current_round = 0
        self.global_accuracy = 0.0
        
    def _get_sequential_id(self, flower_cid: str) -> int:
        """
        Map Flower's random client ID to sequential ID (0 to num_clients-1).
        This ensures consistent mapping across rounds.
        """
        cid_str = str(flower_cid)
        if cid_str not in self._client_id_map:
            # Assign new sequential ID modulo num_clients to wrap around if needed
            assigned_id = self._next_available_id % self.channel.num_clients
            self._client_id_map[cid_str] = assigned_id
            self._reverse_map[assigned_id] = cid_str
            self._next_available_id += 1
            
            # Update scheduler's num_samples if needed
            if self.scheduler and assigned_id not in self.scheduler.num_samples:
                self.scheduler.num_samples[assigned_id] = 300  # Default
        
        return self._client_id_map[cid_str]
    
    def __getattr__(self, name):
        return getattr(self.base, name)
    
    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.current_round = server_round
        
        # Get Flower client IDs and map to sequential IDs
        flower_clients = list(client_manager.all().keys())
        print(f"[Wrapper] Round {server_round}: {len(flower_clients)} Flower clients available")
        
        # Map to sequential IDs with deduplication
        available_cids = []
        seen = set()
        for cid in flower_clients:
            seq_id = self._get_sequential_id(str(cid))
            if seq_id not in seen:
                available_cids.append(seq_id)
                seen.add(seq_id)
        
        if not available_cids:
            available_cids = list(range(self.channel.num_clients))
        
        print(f"[Wrapper] Mapped to {len(available_cids)} unique sequential IDs: {available_cids[:10]}...")
        
        if self.scheduler:
            selected_cids, client_epsilon, client_noise, client_use_sa, metadata = \
                self.scheduler.select_clients_and_configure(
                    available_cids, server_round, self.total_rounds, self.global_accuracy
                )
            
            # CRITICAL DEBUG: Verify what the scheduler returned
            print(f"[Wrapper] DEBUG: Scheduler returned client_epsilon = {client_epsilon}")
            print(f"[Scheduler] R{server_round}: Selected {len(selected_cids)} clients, "
                  f"avg ε={metadata['avg_epsilon']:.3f}")
        else:
            # No scheduler - simple selection
            selected_cids = available_cids[:self.channel.K]
            client_epsilon = {cid: 1.0 for cid in selected_cids}
            client_noise = {cid: 0.0 for cid in selected_cids}
            client_use_sa = {cid: False for cid in selected_cids}
            metadata = {}
        
        # Get base strategy's client instructions
        config_pairs = self.base.configure_fit(server_round, parameters, client_manager)
        
        # Filter to selected clients only, with deduplication
        spawned_cids = []
        valid_pairs = []
        
        for client, fit_ins in config_pairs:
            seq_cid = self._get_sequential_id(str(client.cid))
            
            # Only include if selected and not already spawned (prevent duplicates)
            if seq_cid in selected_cids and seq_cid not in spawned_cids:
                spawned_cids.append(seq_cid)
                valid_pairs.append((client, fit_ins, seq_cid))
                
                # Respect hardware cap K
                if len(spawned_cids) >= self.channel.K:
                    break
        
        # Store EXACT list of clients that will be spawned
        self._current_round_config = {
            'selected_cids': spawned_cids,  # These are the ACTUAL clients being used
            'client_epsilon': client_epsilon,
            'client_use_sa': client_use_sa,
            'metadata': metadata
        }
        
        print(f"[Wrapper] Will spawn {len(spawned_cids)} clients: {spawned_cids}")
        
        # Build final config pairs with adaptive settings
        filtered_pairs = []
        for client, fit_ins, cid in valid_pairs:
            new_config = dict(fit_ins.config)
            
            # CRITICAL FIX: Add DP config if enabled - USE CORRECTED VALUES FROM SCHEDULER
            if self.use_adaptive_dp and cid in client_epsilon:
                eps_value = client_epsilon[cid]
                noise_value = client_noise.get(cid, 0.0)
                
                # FINAL SANITY CHECK: Ensure epsilon is not too low
                if eps_value < 0.5:
                    print(f"[Wrapper] WARNING: Client {cid} has ε={eps_value:.3f}, "
                          f"forcing to 0.5")
                    eps_value = 0.5
                    # Recalculate noise for forced epsilon
                    from try_project.dp_utils import find_noise_multiplier_for_epsilon_rdp
                    noise_value, _ = find_noise_multiplier_for_epsilon_rdp(
                        target_epsilon=eps_value,
                        num_samples=300,
                        batch_size=32,
                        epochs=3,
                        delta=1e-4
                    )
                
                new_config['target_epsilon'] = eps_value
                new_config['noise_multiplier'] = noise_value
                new_config['adaptive_epsilon'] = True
                
                print(f"[Wrapper] Client {cid}: ε={eps_value:.3f}, σ={noise_value:.2f}")
            
            # Add SA config if enabled AND client should use SA
            if self.use_adaptive_sa and cid in client_use_sa and client_use_sa[cid]:
                new_config['use_sa'] = True
                if self.sa:
                    round_seed = self.sa.generate_round_seed(server_round)
                    new_config['sa_round_seed'] = round_seed.hex()
                    new_config['sa_all_clients'] = json.dumps(spawned_cids)
            else:
                new_config['use_sa'] = False
            
            new_fit_ins = FitIns(parameters=fit_ins.parameters, config=new_config)
            filtered_pairs.append((client, new_fit_ins))
        
        return filtered_pairs
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        if not results:
            print(f"[Wrapper] R{server_round}: No results received")
            return self.base.aggregate_fit(server_round, results, failures)
        
        # Map results to sequential IDs
        client_ids = []
        for client, fit_res in results:
            seq_cid = self._get_sequential_id(str(client.cid))
            client_ids.append(seq_cid)
        
        print(f"[Wrapper] R{server_round}: Received {len(results)} results from clients {client_ids}")
        
        # Apply SA masking if enabled (clients will unmask themselves in this simulation)
        # Note: In real deployment, server would handle unmasking with threshold
        if self.sa and self.use_adaptive_sa:
            results = self._apply_sa_masking(results, server_round)
        
        # Simulate network transmission
        transmission_results, successful_cids = self._simulate_transmission(results, client_ids, server_round)
        
        # Filter to successful transmissions only
        successful_results = [
            (client, fit_res) for (client, fit_res), cid in zip(results, client_ids)
            if cid in successful_cids
        ]
        
        print(f"[Wrapper] R{server_round}: {len(successful_results)} successful transmissions")
        
        # Update scheduler with performance metrics
        if self.scheduler:
            accuracies = [fit_res.metrics.get('accuracy', 0.0) for _, fit_res in successful_results if fit_res.metrics]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            self.global_accuracy = avg_accuracy
            
            avg_latency = np.mean([r.latency_ms for r in transmission_results]) if transmission_results else 0.0
            p95_latency = np.percentile([r.latency_ms for r in transmission_results], 95) if len(transmission_results) > 0 else 0.0
            total_bytes = sum(r.bytes_sent for r in transmission_results)
            
            # Track which selected clients succeeded
            selected_set = set(self._current_round_config.get('selected_cids', []))
            client_success = {cid: (cid in successful_cids) for cid in selected_set}
            
            self.scheduler.update_performance_metrics(avg_accuracy, avg_latency, total_bytes, client_success)
            
            # Save metrics
            os.makedirs("results", exist_ok=True)
            drop_rate = 1.0 - (len(successful_cids) / max(len(client_ids), 1))
            current_epsilon = self.scheduler.get_privacy_summary()["total_epsilon_spent"]
            
            round_data = {
                "round": server_round,
                "accuracy": float(avg_accuracy),
                "bytes_sent_mb": float(total_bytes / 1e6),
                "epsilon_spent": float(current_epsilon),
                "latency_p95_ms": float(p95_latency),
                "drop_rate": float(drop_rate),
                "clients_attempted": len(client_ids),
                "clients_successful": len(successful_cids)
            }
            
            with open(f"results/metrics_round_{server_round:03d}.json", "w") as f:
                json.dump(round_data, f, indent=2)
        
        # Aggregate with base strategy
        return self.base.aggregate_fit(server_round, successful_results, failures)
    
    def _apply_sa_masking(self, results: List[Tuple[ClientProxy, FitRes]], server_round: int):
        """Apply SA masking to results if SA is enabled."""
        if not self.sa:
            return results
            
        round_seed = self.sa.generate_round_seed(server_round)
        spawned_cids = self._current_round_config.get('selected_cids', [])
        
        masked_results = []
        for client, fit_res in results:
            seq_cid = self._get_sequential_id(str(client.cid))
            
            # Only mask if SA is enabled for this client
            if not self._current_round_config.get('client_use_sa', {}).get(seq_cid, False):
                masked_results.append((client, fit_res))
                continue
            
            try:
                weights = parameters_to_ndarrays(fit_res.parameters)
                weight_shapes = [w.shape for w in weights]
                masks, _ = self.sa.generate_masks(seq_cid, round_seed, weight_shapes, spawned_cids)
                masked_weights = self.sa.mask_weights(weights, masks)
                fit_res.parameters = ndarrays_to_parameters(masked_weights)
                fit_res.metrics = fit_res.metrics or {}
                fit_res.metrics['_sa_masked'] = True
                print(f"[SA] Client {seq_cid} masked")
            except Exception as e:
                print(f"[SA] Masking failed for client {seq_cid}: {e}")
            
            masked_results.append((client, fit_res))
        
        return masked_results
    
    def _simulate_transmission(self, results: List[Tuple[ClientProxy, FitRes]], client_ids: List[int], server_round: int):
        """Simulate network transmission with channel model."""
        def priority_func(cid):
            if self.scheduler and cid in self.scheduler.num_samples:
                return self.scheduler.num_samples[cid]
            return 100  # Default priority
        
        transmission_results, successful_cids = self.channel.simulate_round(
            selected_clients=client_ids,
            update_size_bytes=self.model_bytes,
            priority_func=priority_func
        )
        
        # Log results
        success_rate = len(successful_cids) / max(len(client_ids), 1)
        latencies = [r.latency_ms for r in transmission_results] if transmission_results else []
        p95 = np.percentile(latencies, 95) if latencies else 0.0
        
        print(f"[Network] R{server_round}: {len(successful_cids)}/{len(client_ids)} succeeded "
              f"({success_rate:.1%}), p95 latency={p95:.1f}ms")
        
        return transmission_results, successful_cids
    
    def get_final_summary(self) -> Dict[str, Any]:
        """Get final experiment summary."""
        channel_stats = self.channel.get_channel_statistics()
        privacy_summary = self.scheduler.get_privacy_summary() if self.scheduler else {}
        
        return {
            "channel_statistics": channel_stats,
            "privacy_summary": privacy_summary,
            "round_metrics": self.round_metrics,
            "adaptive_dp_enabled": self.use_adaptive_dp,
            "adaptive_sa_enabled": self.use_adaptive_sa,
            "total_rounds": self.total_rounds,
        }