"""
Passive network wrapper - injects channel effects WITHOUT modifying strategy.
"""

from typing import Dict, Any, List, Tuple, Optional
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

from try_project.simple_network_sim import SimpleGilbertElliott


class PassiveNetworkWrapper:
    def __init__(
        self,
        base_strategy,
        channel: SimpleGilbertElliott,
        model_bytes: int = 1_200_000,
    ):
        self.base = base_strategy
        self.channel = channel
        self.model_bytes = model_bytes
        self.round_metrics: List[Dict] = []
        
    def __getattr__(self, name):
        return getattr(self.base, name)
    
    def initialize_parameters(self, client_manager):
        """Delegate to base strategy."""
        return self.base.initialize_parameters(client_manager)
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Delegate to base strategy."""
        return self.base.configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        """Delegate to base strategy."""
        return self.base.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        """Inject network losses before aggregating."""
        pre_success = len(results)
        
        network_results = []
        network_failures = list(failures)
        
        for client, fit_res in results:
            try:
                cid = int(client.cid)
            except (ValueError, AttributeError):
                cid = 0
            
            success, net_metrics = self.channel.simulate_update_transmission(
                cid, self.model_bytes
            )
            
            if success:
                fit_res.metrics = fit_res.metrics or {}
                # Only add NUMERIC metrics (no strings like state)
                fit_res.metrics.update({
                    "net_loss_rate": float(net_metrics["loss_rate"]),
                    "net_latency_ms": float(net_metrics["latency_ms"]),
                    "net_transmit_time_s": float(net_metrics["transmit_time_s"]),
                    "net_retrans_factor": float(net_metrics["retrans_factor"]),
                })
                network_results.append((client, fit_res))
            else:
                network_failures.append((client, fit_res))
                print(f"[NET] R{server_round} C{cid}: LOST (state={net_metrics['state']}, loss={net_metrics['loss_rate']:.1%})")
        
        post_success = len(network_results)
        self.round_metrics.append({
            "round": server_round,
            "pre_network_success": pre_success,
            "post_network_success": post_success,
            "network_failures": pre_success - post_success,
            "avg_transmit_time": sum(
                r.metrics.get("net_transmit_time_s", 0) 
                for _, r in network_results
            ) / max(len(network_results), 1),
        })
        
        print(f"[NET] R{server_round}: {post_success}/{pre_success} successful, {pre_success - post_success} lost")
        
        return self.base.aggregate_fit(server_round, network_results, network_failures)
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Delegate to base strategy."""
        return self.base.aggregate_evaluate(server_round, results, failures)
    
    def evaluate(self, server_round: int, parameters: Parameters):
        """Delegate to base strategy."""
        return self.base.evaluate(server_round, parameters)