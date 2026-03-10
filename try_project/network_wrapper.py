"""
Network wrapper with Secure Aggregation toggle and comprehensive metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import numpy as np
import time

from try_project.simple_network_sim import SimpleGilbertElliott
from try_project.secure_agg import SecureAggregation


class PassiveNetworkWrapper:
    """
    Passive network wrapper that injects channel effects and manages Secure Aggregation.
    """
    
    def __init__(
        self,
        base_strategy,
        channel: Optional[SimpleGilbertElliott],
        model_bytes: int = 1_200_000,
        use_sa: bool = False,
        sa_threshold: int = 20,
    ):
        self.base = base_strategy
        self.channel = channel
        self.model_bytes = model_bytes
        self.use_sa = use_sa
        self.sa_threshold = sa_threshold
        
        # Initialize SA if enabled
        self.sa = None
        if use_sa:
            # Estimate max clients from strategy
            self.sa = SecureAggregation(num_clients=30, threshold=sa_threshold)
            print(f"[NET] Secure Aggregation ENABLED (threshold={sa_threshold})")
        
        # Metrics tracking
        self.round_metrics: List[Dict] = []
        self.total_bytes_sent = 0
        self.total_bytes_lost = 0
        self.total_rounds = 0
        
    def __getattr__(self, name):
        """Delegate unknown attributes to base strategy."""
        return getattr(self.base, name)
    
    def initialize_parameters(self, client_manager):
        """Delegate to base strategy."""
        return self.base.initialize_parameters(client_manager)
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure fit with optional SA setup."""
        config_pairs = self.base.configure_fit(server_round, parameters, client_manager)
        
        # Add SA configuration to client configs if enabled
        if self.use_sa and self.sa:
            # Generate round seed for SA
            round_seed = self.sa.generate_round_seed(server_round)
            
            # Add SA config to each client's instructions
            updated_pairs = []
            for client, config in config_pairs:
                config["sa_enabled"] = True
                config["sa_round_seed"] = round_seed
                config["sa_threshold"] = self.sa_threshold
                updated_pairs.append((client, config))
            return updated_pairs
        
        return config_pairs
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        """Delegate to base strategy."""
        return self.base.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[BaseException]
    ):
        """
        Aggregate fit results with network effects and optional SA.
        """
        pre_success = len(results)
        self.total_rounds = server_round
        
        # Phase 1: Apply SA masking if enabled
        if self.use_sa and self.sa:
            results = self._apply_sa_masking(results, server_round)
        
        # Phase 2: Apply network effects (loss, latency)
        network_results, network_failures, round_bytes = self._apply_network_effects(
            results, failures, server_round
        )
        
        # Update byte counters
        self.total_bytes_sent += round_bytes["sent"]
        self.total_bytes_lost += round_bytes["lost_equivalent"]
        
        post_success = len(network_results)
        
        # Phase 3: SA unmasking if enabled
        if self.use_sa and self.sa:
            if post_success < self.sa_threshold:
                print(f"[NET] R{server_round}: INSUFFICIENT clients for SA "
                      f"({post_success}/{self.sa_threshold}), aborting aggregation")
                # Return previous parameters (no update)
                return None, {}
            
            # Unmask the aggregated results
            network_results = self._apply_sa_unmasking(network_results, server_round)
            print(f"[NET] R{server_round}: SA unmasking with {post_success} clients")
        
        # Log metrics
        self._log_round_metrics(server_round, pre_success, post_success, round_bytes)
        
        # Delegate to base strategy for actual aggregation
        return self.base.aggregate_fit(server_round, network_results, network_failures)
    
    def _apply_sa_masking(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        server_round: int
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Apply SA masking to client weights before transmission."""
        masked_results = []
        round_seed = self.sa.generate_round_seed(server_round)
        
        for client, fit_res in results:
            try:
                cid = int(client.cid)
                # Get weights from FitRes
                weights = parameters_to_ndarrays(fit_res.parameters)
                
                # Generate and apply masks
                masks = self.sa.generate_masks(cid, round_seed, weights)
                masked_weights = self.sa.mask_weights(weights, masks)
                
                # Create new FitRes with masked weights
                fit_res.parameters = ndarrays_to_parameters(masked_weights)
                
                # Store masks for later unmasking (server-side tracking)
                fit_res.metrics = fit_res.metrics or {}
                fit_res.metrics["_sa_masks"] = masks  # Internal use only
                
                masked_results.append((client, fit_res))
                
            except Exception as e:
                print(f"[SA] Masking failed for client {client.cid}: {e}")
                # Include unmasked (will fail at unmasking if too many)
                masked_results.append((client, fit_res))
        
        return masked_results
    
    def _apply_network_effects(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
        server_round: int,
    ) -> Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException], Dict[str, float]]:
        """Apply Gilbert-Elliott channel effects."""
        network_results = []
        network_failures = list(failures)
        round_bytes_sent = 0.0
        round_bytes_lost = 0.0
        
        for client, fit_res in results:
            try:
                cid = int(client.cid)
            except (ValueError, AttributeError):
                cid = 0
            
            # Simulate transmission
            if self.channel:
                success, net_metrics = self.channel.simulate_update_transmission(
                    cid, self.model_bytes
                )
            else:
                # Reliable channel
                success = True
                net_metrics = {
                    "loss_rate": 0.0,
                    "latency_ms": 50.0,
                    "state": "good",
                    "bandwidth_mbps": 10.0,
                    "transmit_time_s": 1.0,
                    "retrans_factor": 1.0,
                }
            
            if success:
                # Calculate actual bytes transmitted (including retransmissions)
                bytes_transmitted = self.model_bytes * net_metrics.get("retrans_factor", 1.0)
                
                # SA doubles the upload (masks + masked weights)
                if self.use_sa:
                    bytes_transmitted *= 2.0
                
                round_bytes_sent += bytes_transmitted
                
                # Add network metrics to fit_res
                fit_res.metrics = fit_res.metrics or {}
                fit_res.metrics.update({
                    "net_loss_rate": float(net_metrics["loss_rate"]),
                    "net_latency_ms": float(net_metrics["latency_ms"]),
                    "net_transmit_time_s": float(net_metrics["transmit_time_s"]),
                    "net_retrans_factor": float(net_metrics["retrans_factor"]),
                    "net_bytes_sent": float(bytes_transmitted),
                    "sa_enabled": self.use_sa,
                })
                
                network_results.append((client, fit_res))
            else:
                # Lost update
                lost_bytes = self.model_bytes * 1.5  # Assumed retrans attempt
                if self.use_sa:
                    lost_bytes *= 2.0
                round_bytes_lost += lost_bytes
                
                network_failures.append((client, fit_res))
                print(f"[NET] R{server_round} C{cid}: LOST "
                      f"(state={net_metrics['state']}, loss={net_metrics['loss_rate']:.1%})")
        
        return network_results, network_failures, {
            "sent": round_bytes_sent,
            "lost_equivalent": round_bytes_lost,
        }
    
    def _apply_sa_unmasking(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Remove SA masks from aggregated results."""
        round_seed = self.sa.generate_round_seed(server_round)
        client_ids = []
        all_masked_weights = []
        
        # Collect weights and client IDs
        for client, fit_res in results:
            try:
                cid = int(client.cid)
                weights = parameters_to_ndarrays(fit_res.parameters)
                all_masked_weights.append(weights)
                client_ids.append(cid)
            except Exception as e:
                print(f"[SA] Unmasking prep failed for {client.cid}: {e}")
        
        if not all_masked_weights:
            return results
        
        # Sum all masked weights (server-side aggregation)
        aggregated_masked = [np.sum(w, axis=0) for w in zip(*all_masked_weights)]
        
        # Subtract masks from surviving clients
        for cid in client_ids:
            masks = self.sa.generate_masks(cid, round_seed, aggregated_masked)
            aggregated_masked = [agg - mask for agg, mask in zip(aggregated_masked, masks)]
        
        # Average
        aggregated = [w / len(client_ids) for w in aggregated_masked]
        
        # Create dummy FitRes with unmasked weights (attach to first result)
        if results:
            first_client, first_fit_res = results[0]
            first_fit_res.parameters = ndarrays_to_parameters(aggregated)
            # Clear internal metrics
            if first_fit_res.metrics and "_sa_masks" in first_fit_res.metrics:
                del first_fit_res.metrics["_sa_masks"]
            
            # Return only the aggregated result (others are redundant)
            return [(first_client, first_fit_res)]
        
        return results
    
    def _log_round_metrics(
        self, 
        server_round: int, 
        pre_success: int, 
        post_success: int,
        round_bytes: Dict[str, float],
    ):
        """Log round metrics."""
        avg_transmit_time = 0.0
        if network_results := getattr(self, '_last_network_results', None):
            times = [r.metrics.get("net_transmit_time_s", 0) 
                    for _, r in network_results if r.metrics]
            if times:
                avg_transmit_time = sum(times) / len(times)
        
        self.round_metrics.append({
            "round": server_round,
            "pre_network_success": pre_success,
            "post_network_success": post_success,
            "network_failures": pre_success - post_success,
            "bytes_sent_mb": round_bytes["sent"] / 1e6,
            "bytes_lost_mb": round_bytes["lost_equivalent"] / 1e6,
            "avg_transmit_time_s": avg_transmit_time,
            "sa_enabled": self.use_sa,
        })
        
        print(f"[NET] R{server_round}: {post_success}/{pre_success} successful, "
              f"{round_bytes['sent']/1e6:.2f}MB sent, "
              f"SA={'ON' if self.use_sa else 'OFF'}")
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Delegate to base strategy."""
        return self.base.aggregate_evaluate(server_round, results, failures)
    
    def evaluate(self, server_round: int, parameters: Parameters):
        """Delegate to base strategy."""
        return self.base.evaluate(server_round, parameters)
    
    def get_final_summary(self) -> Dict[str, Any]:
        """Get final experiment summary."""
        total_mb = self.total_bytes_sent / 1e6
        lost_mb = self.total_bytes_lost / 1e6
        
        return {
            "total_rounds": self.total_rounds,
            "total_bytes_sent_mb": total_mb,
            "total_bytes_lost_mb": lost_mb,
            "average_bytes_per_round_mb": total_mb / max(self.total_rounds, 1),
            "sa_enabled": self.use_sa,
            "sa_threshold": self.sa_threshold if self.use_sa else None,
            "round_metrics": self.round_metrics,
        }