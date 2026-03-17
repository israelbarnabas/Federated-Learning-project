"""
Enhanced Server with Link-Aware Adaptive Scheduling and Full Integration.
FIXED: Properly read epsilon_per_round_min from config to ensure meaningful privacy budget.
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
    """
    Aggregate metrics weighted by number of examples.
    Handles edge cases: empty metrics, None values, zero total examples.
    """
    if not metrics:
        return {}
    
    # Filter out entries with None metrics
    valid_metrics = [(n, m) for n, m in metrics if m is not None]
    if not valid_metrics:
        return {}
    
    # Calculate total examples
    total_examples = sum(num_examples for num_examples, _ in valid_metrics)
    if total_examples == 0:
        return {}
    
    # Get numeric metric names from first valid entry
    first_metrics = valid_metrics[0][1]
    numeric_metric_names = [
        k for k in first_metrics.keys()
        if isinstance(first_metrics.get(k), (int, float)) and not k.startswith("_")
    ]
    
    # Compute weighted average for each metric
    aggregated = {}
    for metric_name in numeric_metric_names:
        weighted_sum = sum(
            num_examples * m.get(metric_name, 0.0) 
            for num_examples, m in valid_metrics
        )
        aggregated[metric_name] = weighted_sum / total_examples
    
    return aggregated


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    
    print("=" * 70)
    print("ENHANCED LINK-AWARE FL SERVER")
    print("=" * 70)
    
    # Parse configuration with fallbacks
    num_rounds = int(cfg.get("num-rounds", cfg.get("num_rounds", 10)))
    num_clients = int(cfg.get("num-clients", cfg.get("num_clients", 30)))
    
    use_adaptive_dp = cfg.get("use-adaptive-dp", cfg.get("use_adaptive_dp", True))
    use_adaptive_sa = cfg.get("use-adaptive-sa", cfg.get("use_adaptive_sa", False))
    use_fedprox = cfg.get("use-fedprox", cfg.get("use_fedprox", False))
    use_noisy_channel = cfg.get("use-noisy-channel", cfg.get("use_noisy_channel", True))
    
    target_epsilon_total = float(cfg.get("target-epsilon-total", cfg.get("target_epsilon_total", 5.0)))
    epsilon_per_round_max = float(cfg.get("epsilon-per-round-max", cfg.get("epsilon_per_round_max", 1.0)))
    # CRITICAL FIX: Read epsilon_per_round_min from config with proper default
    epsilon_per_round_min = float(cfg.get("epsilon-per-round-min", cfg.get("epsilon_per_round_min", 0.5)))
    
    max_concurrent = int(cfg.get("max-concurrent", cfg.get("max_concurrent", 15)))
    
    print(f"[Server] Rounds: {num_rounds}, Clients: {num_clients}")
    print(f"[Server] Adaptive DP: {use_adaptive_dp}, Adaptive SA: {use_adaptive_sa}")
    print(f"[Server] Total ε budget: {target_epsilon_total}, max per round: {epsilon_per_round_max}")
    print(f"[Server] Min ε per round: {epsilon_per_round_min}")  # NEW: Log this
    print(f"[Server] Max concurrent transmissions: {max_concurrent}")
    
    # Initialize channel simulator
    channel = AIoTChannel(
        num_clients=num_clients,
        max_concurrent_transmissions=max_concurrent,
        seed=cfg.get("simulation-seed", 42)
    )
    
    # Configure scheduler with PROPER epsilon_per_round_min from config
    scheduler_config = SchedulerConfig(
        target_epsilon_total=target_epsilon_total,
        target_delta=1e-4,
        epsilon_per_round_max=epsilon_per_round_max,
        epsilon_per_round_min=epsilon_per_round_min,  # CRITICAL: Use config value, not default!
        max_clients_per_round=max_concurrent,
        min_clients_per_round=2,
        latency_sla_ms=500.0,
        sa_strategy=SAStrategy.ADAPTIVE if use_adaptive_sa else SAStrategy.ALWAYS_OFF,
        prefer_good_channel=True,
    )
    
    print(f"[Server] Scheduler config: ε_per_round_min={scheduler_config.epsilon_per_round_min}")
    
    # Create initial model parameters
    dummy_model = load_model(input_shape=(100, 3), num_classes=6, for_dp=use_adaptive_dp)
    initial_params = ndarrays_to_parameters(dummy_model.get_weights())
    print(f"[Server] Model: ~{dummy_model.count_params()/1000:.1f}K parameters")
    
    fedprox_mu = float(cfg.get("fedprox-mu", cfg.get("fedprox_mu", 0.05)))
    
    # Create base strategy
    if use_fedprox:
        print(f"[Server] Using FedProx with μ={fedprox_mu}")
        base_strategy = FedProx(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=fedprox_mu,
        )
    else:
        print(f"[Server] Using FedAvg")
        base_strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    # Wrap with link-aware network simulation
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
    
    print("=" * 70)
    
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)