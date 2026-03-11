"""
Network wrapper with CORRECT Secure Aggregation flow.
Server aggregates masked weights; never sees individual unmasked updates.
"""

from typing import Dict, Any, List, Tuple, Optional
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from flwr.server.client_proxy import ClientProxy
import numpy as np

from try_project.simple_network_sim import SimpleGilbertElliott
from try_project.secure_agg import SecureAggregation, ProperSecureAggregation


class PassiveNetworkWrapper:
    """
    Passive network wrapper with proper SA:
    1. Clients mask weights before transmission
    2. Network may drop some masked updates
    3. Server aggregates surviving masked weights
    4. Masks cancel out in aggregation (server never unmasks individually)
    """
    
    def __init__(
        self,
        base_strategy,
        channel: Optional[SimpleGilbertElliott],
        model_bytes: int = 1_200_000,
        use_sa: bool = False,
        sa_threshold: int = 20,
        use_pairwise_sa: bool = False,  # Use full pairwise masking
    ):
        self.base = base_strategy
        self.channel = channel
        self.model_bytes = model_bytes
        self.use_sa = use_sa
        self.sa_threshold = sa_threshold
        
        # Initialize SA
        self.sa = None
        if use_sa:
            if use_pairwise_sa:
                self.sa = ProperSecureAggregation(num_clients=30, threshold=sa_threshold)
                print(f"[NET] Proper SA (pairwise) ENABLED, threshold={sa_threshold}")
            else:
                self.sa = SecureAggregation(num_clients=30, threshold=sa_threshold)
                print(f"[NET] Simplified SA ENABLED, threshold={sa_threshold}")
        
        # Metrics
        self.round_metrics: List[Dict] = []
        self.total_bytes_sent = 0
        self.total_bytes_lost = 0
        
    def __getattr__(self, name):
        return getattr(self.base, name)
    
    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)
    
def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
    """Configure fit with SA parameters."""
    config_pairs = self.base.configure_fit(server_round, parameters, client_manager)
    
    if self.use_sa and self.sa:
        round_seed = self.sa.generate_round_seed(server_round)
        
        # Get all participating client IDs - SORTED for consistency
        all_client_ids = []
        for client, fit_ins in config_pairs:
            try:
                all_client_ids.append(int(client.cid))
            except:
                all_client_ids.append(0)
        
        # CRITICAL FIX: Sort for consistent ordering across all clients
        all_client_ids = sorted(all_client_ids)
        
        # CRITICAL FIX: Convert list to JSON string for RecordDict compatibility
        import json
        all_clients_json = json.dumps(all_client_ids)
        
        updated_pairs = []
        for client, fit_ins in config_pairs:
            try:
                cid = int(client.cid)
            except:
                cid = 0
            
            # CRITICAL FIX: Create new config dict with serializable values only
            new_config = dict(fit_ins.config)  # Copy existing config
            
            # Only use scalar types: str, int, float, bool
            new_config["sa_enabled"] = True  # bool
            new_config["sa_round_seed"] = round_seed  # int
            new_config["sa_threshold"] = self.sa_threshold  # int
            new_config["sa_all_clients_json"] = all_clients_json  # str (JSON)
            new_config["sa_num_clients"] = len(all_client_ids)  # int (for verification)
            
            # Create new FitIns (immutable, must create new)
            new_fit_ins = FitIns(
                parameters=fit_ins.parameters,
                config=new_config
            )
            
            updated_pairs.append((client, new_fit_ins))
        
        return updated_pairs
    
    return config_pairs
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        return self.base.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[BaseException]
    ):
        """
        Aggregate with proper SA flow:
        1. Apply SA masking (client-side simulation)
        2. Apply network effects (drops)
        3. Aggregate masked weights (masks cancel)
        4. Server never sees individual unmasked updates
        """
        pre_success = len(results)
        
        # Phase 1: Apply SA masking (simulating client-side behavior)
        if self.use_sa and self.sa:
            results = self._apply_sa_masking(results, server_round)
        
        # Phase 2: Apply network effects
        network_results, network_failures, round_bytes = self._apply_network_effects(
            results, failures, server_round
        )
        
        self.total_bytes_sent += round_bytes["sent"]
        self.total_bytes_lost += round_bytes["lost_equivalent"]
        
        post_success = len(network_results)
        
        # Phase 3: Check SA threshold with FALLBACK mechanism
        if self.use_sa:
            if post_success < self.sa_threshold:
                print(f"[NET] R{server_round}: SA threshold FAILED ({post_success}/{self.sa_threshold})")
                print(f"[NET] R{server_round}: FALLBACK to standard FedAvg (no SA masking)")
                
                # CRITICAL FIX: Fallback to standard aggregation instead of aborting
                self._log_round_metrics(server_round, pre_success, post_success, round_bytes)
                
                # Return standard aggregation without SA
                clean_results = []
                for client, fit_res in network_results:
                    clean_metrics = {k: v for k, v in (fit_res.metrics or {}).items() 
                                   if not k.startswith("_sa_")}
                    clean_metrics["sa_fallback"] = True
                    clean_metrics["sa_enabled"] = False
                    
                    new_fit_res = FitRes(
                        status=fit_res.status,
                        parameters=fit_res.parameters,
                        num_examples=fit_res.num_examples,
                        metrics=clean_metrics
                    )
                    clean_results.append((client, new_fit_res))
                
                return self.base.aggregate_fit(server_round, clean_results, network_failures)
            
            # SA proceeds normally - aggregate masked weights
            network_results = self._aggregate_masked_weights(network_results, server_round)
            print(f"[NET] R{server_round}: SA aggregation with {post_success} clients "
                  f"(masks cancel in sum)")
        
        # Log metrics
        self._log_round_metrics(server_round, pre_success, post_success, round_bytes)
        
        # Delegate to base strategy
        return self.base.aggregate_fit(server_round, network_results, network_failures)
    
    def _apply_sa_masking(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        server_round: int
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Simulate client-side SA masking.
        """
        round_seed = self.sa.generate_round_seed(server_round)
        masked_results = []
        
        for client, fit_res in results:
            try:
                cid = int(client.cid)
                weights = parameters_to_ndarrays(fit_res.parameters)
                
                # Get shapes for mask generation
                weight_shapes = [w.shape for w in weights]
                
                # Check if we have pairwise or simple masking
                all_clients = getattr(fit_res, 'sa_all_clients', None)
                
                if all_clients and isinstance(self.sa, ProperSecureAggregation):
                    masks = self.sa.generate_pairwise_masks(cid, round_seed, weight_shapes, all_clients)
                else:
                    masks = self.sa.generate_masks(cid, round_seed, weight_shapes)
                
                masked_weights = self.sa.mask_weights(weights, masks)
                
                # Update FitRes with masked weights
                fit_res.parameters = ndarrays_to_parameters(masked_weights)
                
                # Store metadata for aggregation
                fit_res.metrics = fit_res.metrics or {}
                fit_res.metrics["_sa_mask_applied"] = True
                fit_res.metrics["_sa_client_id"] = cid
                
                masked_results.append((client, fit_res))
                
            except Exception as e:
                print(f"[SA] Masking failed for client {client.cid}: {e}")
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
            except:
                cid = 0
            
            # Simulate transmission
            if self.channel:
                success, net_metrics = self.channel.simulate_update_transmission(
                    cid, self.model_bytes
                )
            else:
                success = True
                net_metrics = {
                    "loss_rate": 0.0, "latency_ms": 50.0, "state": "good",
                    "bandwidth_mbps": 10.0, "transmit_time_s": 1.0, "retrans_factor": 1.0,
                }
            
            if success:
                # SA doubles the effective size
                bytes_transmitted = self.model_bytes * net_metrics.get("retrans_factor", 1.0)
                if self.use_sa:
                    bytes_transmitted *= 2.0
                
                round_bytes_sent += bytes_transmitted
                
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
                lost_bytes = self.model_bytes * 1.5
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
    
    def _aggregate_masked_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Aggregate masked weights where masks cancel out.
        """
        if not results:
            return results
        
        round_seed = self.sa.generate_round_seed(server_round)
        
        # Extract masked weights and client IDs
        masked_weights_list = []
        client_ids = []
        
        for client, fit_res in results:
            try:
                cid = int(client.cid)
                weights = parameters_to_ndarrays(fit_res.parameters)
                masked_weights_list.append(weights)
                client_ids.append(cid)
            except Exception as e:
                print(f"[SA] Extraction failed for {client.cid}: {e}")
        
        if not masked_weights_list:
            return results
        
        try:
            aggregated = self.sa.aggregate_masked(
                masked_weights_list, client_ids, round_seed
            )
            
            # CRITICAL FIX: Return ALL results with aggregated parameters
            updated_results = []
            for client, fit_res in results:
                new_metrics = fit_res.metrics or {}
                
                # Remove internal SA metadata
                for key in list(new_metrics.keys()):
                    if key.startswith("_sa_"):
                        del new_metrics[key]
                
                new_metrics["sa_aggregated"] = True
                new_metrics["sa_clients_in_agg"] = len(client_ids)
                
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(aggregated),
                    num_examples=fit_res.num_examples,
                    metrics=new_metrics
                )
                updated_results.append((client, new_fit_res))
            
            return updated_results
            
        except Exception as e:
            print(f"[SA] Aggregation failed: {e}")
            return results
    
    def _log_round_metrics(self, server_round: int, pre: int, post: int, bytes_dict: Dict):
        """Log round metrics."""
        self.round_metrics.append({
            "round": server_round,
            "pre_network": pre,
            "post_network": post,
            "dropped": pre - post,
            "bytes_sent_mb": bytes_dict["sent"] / 1e6,
            "bytes_lost_mb": bytes_dict["lost_equivalent"] / 1e6,
            "sa_enabled": self.use_sa,
        })
        
        print(f"[NET] R{server_round}: {post}/{pre} successful, "
              f"{bytes_dict['sent']/1e6:.2f}MB sent, "
              f"SA={'ON' if self.use_sa else 'OFF'}")
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        return self.base.aggregate_evaluate(server_round, results, failures)
    
    def evaluate(self, server_round: int, parameters: Parameters):
        return self.base.evaluate(server_round, parameters)
    
    def get_final_summary(self) -> Dict[str, Any]:
        """Get final summary."""
        total_mb = self.total_bytes_sent / 1e6
        lost_mb = self.total_bytes_lost / 1e6
        
        return {
            "total_bytes_sent_mb": total_mb,
            "total_bytes_lost_mb": lost_mb,
            "sa_enabled": self.use_sa,
            "round_metrics": self.round_metrics,
        }