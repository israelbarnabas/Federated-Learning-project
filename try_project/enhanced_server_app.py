"""
Enhanced Server with Link-Aware Adaptive Scheduling and Full Integration.
"""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx, FedAvg
from flwr.common import Context, ndarrays_to_parameters
import numpy as np

from try_project.enhanced_network_sim import AIoTChannel
from try_project.enhanced_network_wrapper import LinkAwareNetworkWrapper
from try_project.adaptive_scheduler import SchedulerConfig, SAStrategy
from try_project.task import load_model

def weighted_average(metrics):
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    first_metrics = metrics[0][1]
    numeric_metric_names = [
        k for k in first_metrics.keys()
        if isinstance(first_metrics.get(k), (int, float)) and not k.startswith("_")
    ]
    
    aggregated = {}
    for metric_name in numeric_metric_names:
        weighted_sum = sum(num_examples * m.get(metric_name, 0) for num_examples, m in metrics)
        aggregated[metric_name] = weighted_sum / total_examples
    
    return aggregated

def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    
    print("=" * 70)
    print("ENHANCED LINK-AWARE FL SERVER")
    print("=" * 70)
    
    num_rounds = int(cfg.get("num-rounds", cfg.get("num_rounds", 10)))
    num_clients = int(cfg.get("num-clients", cfg.get("num_clients", 30)))
    
    use_adaptive_dp = cfg.get("use-adaptive-dp", cfg.get("use_adaptive_dp", True))
    use_adaptive_sa = cfg.get("use-adaptive-sa", cfg.get("use_adaptive_sa", True))
    use_fedprox = cfg.get("use-fedprox", cfg.get("use_fedprox", False))
    use_noisy_channel = cfg.get("use-noisy-channel", cfg.get("use_noisy_channel", True))
    
    target_epsilon_total = float(cfg.get("target-epsilon-total", cfg.get("target_epsilon_total", 5.0)))
    epsilon_per_round_max = float(cfg.get("epsilon-per-round-max", cfg.get("epsilon_per_round_max", 1.0)))
    
    max_concurrent = int(cfg.get("max-concurrent", cfg.get("max_concurrent", 15)))
    
    print(f"[Server] Rounds: {num_rounds}, Clients: {num_clients}")
    print(f"[Server] Adaptive DP: {use_adaptive_dp}, Adaptive SA: {use_adaptive_sa}")
    print(f"[Server] Total ε budget: {target_epsilon_total}, max per round: {epsilon_per_round_max}")
    
    channel = AIoTChannel(
        num_clients=num_clients,
        max_concurrent_transmissions=max_concurrent,
        seed=cfg.get("simulation-seed", 42)
    )
    
    scheduler_config = SchedulerConfig(
        target_epsilon_total=target_epsilon_total,
        target_delta=1e-4,
        epsilon_per_round_max=epsilon_per_round_max,
        epsilon_per_round_min=0.05,
        max_clients_per_round=max_concurrent,
        min_clients_per_round=5,
        latency_sla_ms=500.0,
        sa_strategy=SAStrategy.ADAPTIVE if use_adaptive_sa else SAStrategy.ALWAYS_OFF,
        prefer_good_channel=True,
    )
    
    dummy_model = load_model(input_shape=(100, 3), num_classes=6, for_dp=use_adaptive_dp)
    initial_params = ndarrays_to_parameters(dummy_model.get_weights())
    
    fedprox_mu = float(cfg.get("fedprox-mu", cfg.get("fedprox_mu", 0.05)))
    
    if use_fedprox:
        print(f"[Server] Using FedProx with μ={fedprox_mu}")
        base_strategy = FedProx(
            fraction_fit=cfg.get("fraction-train", 1.0),
            fraction_evaluate=cfg.get("fraction-evaluate", 0.5),
            min_fit_clients=3, min_evaluate_clients=1, min_available_clients=3,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=fedprox_mu,
        )
    else:
        print(f"[Server] Using FedAvg")
        base_strategy = FedAvg(
            fraction_fit=cfg.get("fraction-train", 1.0),
            fraction_evaluate=cfg.get("fraction-evaluate", 0.5),
            min_fit_clients=3, min_evaluate_clients=1, min_available_clients=3,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    strategy = LinkAwareNetworkWrapper(
        base_strategy=base_strategy,
        channel=channel,
        scheduler_config=scheduler_config,
        model_bytes=cfg.get("model-size-bytes", 1_200_000),
        use_adaptive_dp=use_adaptive_dp,
        use_adaptive_sa=use_adaptive_sa,
        total_rounds=num_rounds,
        seed=cfg.get("simulation-seed", 42)
    )
    
    server_config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)