"""
Enhanced Network Wrapper with Link-Aware Adaptive Scheduling.
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
        
        self.scheduler = None
        if use_adaptive_dp or use_adaptive_sa:
            self.scheduler = LinkAwareScheduler(
                num_clients=channel.num_clients,
                num_samples_per_client={},
                config=scheduler_config or SchedulerConfig(),
                channel=channel,
                seed=seed
            )
        
        self.sa: Optional[SecureAggregation] = None
        if use_adaptive_sa:
            self.sa = create_sa_for_federation(
                num_clients=channel.num_clients,
                threshold=max(2, channel.K // 2),
                use_secret_sharing=True
            )
        
        self.round_metrics: List[Dict] = []
        self.current_round = 0
        self.global_accuracy = 0.0
        
    def __getattr__(self, name):
        return getattr(self.base, name)
    
    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.current_round = server_round
        
        available_clients = list(client_manager.all().keys())
        if not available_clients:
            available_clients = list(range(self.channel.num_clients))
        
        available_cids = []
        for cid in available_clients:
            try:
                # FIXED: Force modulo immediately upon parsing
                available_cids.append(int(cid) % self.channel.num_clients)
            except:
                available_cids.append(hash(cid) % self.channel.num_clients)
        
        if self.scheduler:
            selected_cids, client_epsilon, client_noise, client_use_sa, metadata = \
                self.scheduler.select_clients_and_configure(
                    available_cids, server_round, self.total_rounds, self.global_accuracy
                )
            print(f"[Scheduler] R{server_round}: Selected {len(selected_cids)} clients, "
                  f"avg ε={metadata['avg_epsilon']:.3f}, SA={metadata['sa_enabled_count']}/{len(selected_cids)}")
        else:
            selected_cids = available_cids[:self.channel.K]
            client_epsilon = {cid: 1.0 for cid in selected_cids}
            client_noise = {cid: 0.0 for cid in selected_cids}
            client_use_sa = {cid: False for cid in selected_cids}
            metadata = {}
        
        self._current_round_config = {
            'selected_cids': selected_cids,
            'client_epsilon': client_epsilon,
            'client_use_sa': client_use_sa,
            'metadata': metadata
        }
        
        config_pairs = self.base.configure_fit(server_round, parameters, client_manager)
        
        filtered_pairs = []
        for client, fit_ins in config_pairs:
            try:
                # FIXED: Force modulo immediately
                cid = int(client.cid) % self.channel.num_clients
            except:
                cid = hash(client.cid) % self.channel.num_clients
            
            if cid in selected_cids:
                new_config = dict(fit_ins.config)
                
                if self.use_adaptive_dp and cid in client_epsilon:
                    new_config['target_epsilon'] = client_epsilon[cid]
                    new_config['noise_multiplier'] = client_noise[cid] 
                    new_config['adaptive_epsilon'] = True
                
                if self.use_adaptive_sa and cid in client_use_sa:
                    new_config['use_sa'] = client_use_sa[cid]
                    if client_use_sa[cid] and self.sa:
                        round_seed = self.sa.generate_round_seed(server_round)
                        new_config['sa_round_seed'] = round_seed.hex()
                        new_config['sa_all_clients'] = json.dumps(selected_cids)
                
                new_fit_ins = FitIns(parameters=fit_ins.parameters, config=new_config)
                filtered_pairs.append((client, new_fit_ins))
        
        return filtered_pairs
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        if not results:
            return self.base.aggregate_fit(server_round, results, failures)
        
        client_ids = []
        for client, fit_res in results:
            try:
                # FIXED: Force modulo immediately
                cid = int(client.cid) % self.channel.num_clients
            except:
                cid = hash(client.cid) % self.channel.num_clients
            client_ids.append(cid)
        
        if self.sa and self._current_round_config.get('client_use_sa'):
            results = self._apply_sa_masking(results, server_round)
        
        transmission_results, successful_cids = self._simulate_transmission(results, client_ids, server_round)
        
        successful_results = [
            (client, fit_res) for (client, fit_res), cid in zip(results, client_ids)
            if cid in successful_cids
        ]
        
        if self.sa and self._current_round_config.get('client_use_sa'):
            successful_results = self._aggregate_with_unmasking(successful_results, successful_cids, server_round)
        
        if self.scheduler:
            accuracies = [fit_res.metrics.get('accuracy', 0.0) for _, fit_res in successful_results if fit_res.metrics]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            self.global_accuracy = avg_accuracy
            
            avg_latency = np.mean([r.latency_ms for r in transmission_results]) if transmission_results else 0.0
            p95_latency = np.percentile([r.latency_ms for r in transmission_results], 95) if transmission_results else 0.0
            total_bytes = sum(r.bytes_sent for r in transmission_results)
            
            client_success = {cid: (cid in successful_cids) for cid in self._current_round_config.get('selected_cids', [])}
            self.scheduler.update_performance_metrics(avg_accuracy, avg_latency, total_bytes, client_success)
            
            os.makedirs("results", exist_ok=True)
            drop_rate = 1.0 - (len(successful_cids) / max(len(client_ids), 1))
            current_epsilon = self.scheduler.get_privacy_summary()["total_epsilon_spent"]
            
            round_data = {
                "round": server_round,
                "accuracy": float(avg_accuracy),
                "bytes_sent_mb": float(total_bytes / 1e6),
                "epsilon_spent": float(current_epsilon),
                "latency_p95_ms": float(p95_latency),
                "drop_rate": float(drop_rate)
            }
            
            with open(f"results/metrics_round_{server_round:02d}.json", "w") as f:
                json.dump(round_data, f, indent=4)
            
        return self.base.aggregate_fit(server_round, successful_results, failures)
    
    def _apply_sa_masking(self, results: List[Tuple[ClientProxy, FitRes]], server_round: int):
        round_seed = self.sa.generate_round_seed(server_round)
        selected_cids = self._current_round_config.get('selected_cids', [])
        
        masked_results = []
        for client, fit_res in results:
            try:
                # FIXED: Force modulo immediately
                cid = int(client.cid) % self.channel.num_clients
            except:
                cid = hash(client.cid) % self.channel.num_clients
            
            if not self._current_round_config.get('client_use_sa', {}).get(cid, False):
                masked_results.append((client, fit_res))
                continue
            
            try:
                weights = parameters_to_ndarrays(fit_res.parameters)
                weight_shapes = [w.shape for w in weights]
                masks, _ = self.sa.generate_masks(cid, round_seed, weight_shapes, selected_cids)
                masked_weights = self.sa.mask_weights(weights, masks)
                fit_res.parameters = ndarrays_to_parameters(masked_weights)
                fit_res.metrics = fit_res.metrics or {}
                fit_res.metrics['_sa_masked'] = True
            except Exception as e:
                print(f"[SA] Masking failed for client {cid}: {e}")
            
            masked_results.append((client, fit_res))
        
        return masked_results
    
    def _simulate_transmission(self, results: List[Tuple[ClientProxy, FitRes]], client_ids: List[int], server_round: int):
        def priority_func(cid):
            return self.scheduler.num_samples.get(cid, 100) if self.scheduler else 100
        
        transmission_results, successful_cids = self.channel.simulate_round(
            selected_clients=client_ids,
            update_size_bytes=self.model_bytes,
            priority_func=priority_func
        )
        
        success_rate = len(successful_cids) / max(len(client_ids), 1)
        print(f"[Network] R{server_round}: {len(successful_cids)}/{len(client_ids)} succeeded "
              f"({success_rate:.1%}), p95 latency={np.percentile([r.latency_ms for r in transmission_results], 95):.1f}ms")
        
        return transmission_results, successful_cids
    
    def _aggregate_with_unmasking(self, results: List[Tuple[ClientProxy, FitRes]], surviving_cids: List[int], server_round: int):
        if not results or not self.sa:
            return results
        
        round_seed = self.sa.generate_round_seed(server_round)
        
        masked_weights_list = []
        for client, fit_res in results:
            if fit_res.metrics and fit_res.metrics.get('_sa_masked'):
                weights = parameters_to_ndarrays(fit_res.parameters)
                masked_weights_list.append(weights)
        
        if not masked_weights_list:
            return results
        
        try:
            aggregated = self.sa.aggregate_masked(masked_weights_list, surviving_cids, round_seed)
            
            updated_results = []
            for client, fit_res in results:
                new_metrics = {k: v for k, v in (fit_res.metrics or {}).items() if not k.startswith('_')}
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(aggregated),
                    num_examples=fit_res.num_examples,
                    metrics=new_metrics
                )
                new_fit_res.metrics['sa_aggregated'] = True
                new_fit_res.metrics['sa_survivors'] = len(surviving_cids)
                updated_results.append((client, new_fit_res))
            
            return updated_results
        except Exception as e:
            print(f"[SA] Aggregation failed: {e}")
            return results
    
    def get_final_summary(self) -> Dict[str, Any]:
        channel_stats = self.channel.get_channel_statistics()
        privacy_summary = self.scheduler.get_privacy_summary() if self.scheduler else {}
        
        return {
            "channel_statistics": channel_stats,
            "privacy_summary": privacy_summary,
            "round_metrics": self.round_metrics,
            "adaptive_dp_enabled": self.use_adaptive_dp,
            "adaptive_sa_enabled": self.use_adaptive_sa,
        }