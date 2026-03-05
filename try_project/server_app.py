from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters
import numpy as np

from try_project.simple_network_sim import SimpleGilbertElliott
from try_project.network_wrapper import PassiveNetworkWrapper


def weighted_average(metrics):
    """Aggregate metrics using weighted average by num_examples."""
    if not metrics:
        return {}
    
    # Calculate weighted averages
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Get all numeric metric names (exclude string metrics like _net_state)
    first_metrics = metrics[0][1]
    numeric_metric_names = [
        k for k in first_metrics.keys() 
        if isinstance(first_metrics.get(k), (int, float))
    ]
    
    # Aggregate only numeric metrics
    aggregated = {}
    for metric_name in numeric_metric_names:
        weighted_sum = sum(
            num_examples * m.get(metric_name, 0) 
            for num_examples, m in metrics
        )
        aggregated[metric_name] = weighted_sum / total_examples
    
    return aggregated


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    
    # Read num_rounds from config
    num_rounds = cfg.get("num-rounds") or cfg.get("num_rounds", 10)
    if not isinstance(num_rounds, int) or num_rounds < 1:
        num_rounds = 10
    
    print(f"[Server] Starting with {num_rounds} rounds")
    
    # Create initial parameters
    from try_project.task import load_model
    dummy_model = load_model(input_shape=(100, 3), num_classes=6)
    initial_params = ndarrays_to_parameters(dummy_model.get_weights())
    
    # Standard FedAvg WITH metrics aggregation
    base_strategy = FedAvg(
        fraction_fit=cfg.get("fraction-train") or cfg.get("fraction_train", 1.0),
        fraction_evaluate=cfg.get("fraction-evaluate") or cfg.get("fraction_evaluate", 0.5),
        min_fit_clients=3,
        min_evaluate_clients=1,
        min_available_clients=3,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Check if noisy channel is enabled
    use_noisy = cfg.get("use-noisy-channel") or cfg.get("use_noisy_channel", False)
    if use_noisy:
        print("[Server] NOISY CHANNEL enabled")
        
        channel = SimpleGilbertElliott(
            p_gb=cfg.get("p-gb") or cfg.get("p_gb", 0.05),
            p_bg=cfg.get("p-bg") or cfg.get("p_bg", 0.95),
            loss_good=cfg.get("loss-good") or cfg.get("loss_good", 0.01),
            loss_bad=cfg.get("loss-bad") or cfg.get("loss_bad", 0.25),
            base_latency_ms=cfg.get("base-latency-ms") or cfg.get("base_latency_ms", 50.0),
            seed=cfg.get("simulation-seed") or cfg.get("simulation_seed", 42),
        )
        
        strategy = PassiveNetworkWrapper(
            base_strategy=base_strategy,
            channel=channel,
            model_bytes=cfg.get("model-size-bytes") or cfg.get("model_size_bytes", 1_200_000),
        )
    else:
        print("[Server] RELIABLE channel (baseline)")
        strategy = base_strategy
    
    server_config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )


app = ServerApp(server_fn=server_fn)