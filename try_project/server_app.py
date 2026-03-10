from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from flwr.common import Context, ndarrays_to_parameters
import numpy as np

from try_project.simple_network_sim import SimpleGilbertElliott
from try_project.network_wrapper import PassiveNetworkWrapper


def weighted_average(metrics):
    """Aggregate metrics using weighted average by num_examples."""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Get numeric metric names
    first_metrics = metrics[0][1]
    numeric_metric_names = [
        k for k in first_metrics.keys() 
        if isinstance(first_metrics.get(k), (int, float)) and not k.startswith("_")
    ]
    
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
    
    # Debug config
    print("=" * 70)
    print("SERVER CONFIG")
    print("=" * 70)
    
    # Parse num_rounds
    num_rounds = cfg.get("num-rounds", cfg.get("num_rounds", 10))
    try:
        num_rounds = int(num_rounds)
    except:
        num_rounds = 10
    
    # Parse features
    use_dp = cfg.get("use-dp", cfg.get("use_dp", False))
    use_fedprox = cfg.get("use-fedprox", cfg.get("use_fedprox", False))
    use_sa = cfg.get("use-sa", cfg.get("use_sa", False))
    use_noisy = cfg.get("use-noisy-channel", cfg.get("use_noisy_channel", False))
    
    target_epsilon = cfg.get("target-epsilon", cfg.get("target_epsilon", 5.0))
    fedprox_mu = cfg.get("fedprox-mu", cfg.get("fedprox_mu", 0.1))
    
    print(f"[Server] Rounds: {num_rounds}")
    print(f"[Server] DP: {use_dp}, FedProx: {use_fedprox}, SA: {use_sa}, Noisy: {use_noisy}")
    
    # Create initial parameters
    from try_project.task import load_model
    dummy_model = load_model(input_shape=(100, 3), num_classes=6)
    initial_params = ndarrays_to_parameters(dummy_model.get_weights())
    
    # Create strategy: FedProx if enabled, else FedAvg
    if use_fedprox:
        print(f"[Server] Using FedProx with μ={fedprox_mu}")
        base_strategy = FedProx(
            fraction_fit=cfg.get("fraction-train", 1.0),
            fraction_evaluate=cfg.get("fraction-evaluate", 0.5),
            min_fit_clients=3,
            min_evaluate_clients=1,
            min_available_clients=3,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=fedprox_mu,
        )
    else:
        from flwr.server.strategy import FedAvg
        base_strategy = FedAvg(
            fraction_fit=cfg.get("fraction-train", 1.0),
            fraction_evaluate=cfg.get("fraction-evaluate", 0.5),
            min_fit_clients=3,
            min_evaluate_clients=1,
            min_available_clients=3,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    # Wrap with network effects and SA
    if use_noisy or use_sa:
        channel = None
        if use_noisy:
            print(f"[Server] Noisy channel: loss_bad={cfg.get('loss-bad', 0.25)}")
            channel = SimpleGilbertElliott(
                p_gb=cfg.get("p-gb", 0.05),
                p_bg=cfg.get("p-bg", 0.95),
                loss_good=cfg.get("loss-good", 0.01),
                loss_bad=cfg.get("loss-bad", 0.25),
                base_latency_ms=cfg.get("base-latency-ms", 50.0),
                seed=cfg.get("simulation-seed", 42),
            )
        
        sa_threshold = cfg.get("sa-threshold", cfg.get("sa_threshold", 20))
        
        strategy = PassiveNetworkWrapper(
            base_strategy=base_strategy,
            channel=channel,
            model_bytes=cfg.get("model-size-bytes", 1_200_000),
            use_sa=use_sa,
            sa_threshold=sa_threshold,
        )
    else:
        strategy = base_strategy
    
    server_config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )


app = ServerApp(server_fn=server_fn)