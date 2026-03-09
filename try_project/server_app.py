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
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    first_metrics = metrics[0][1]
    numeric_metric_names = [
        k for k in first_metrics.keys() 
        if isinstance(first_metrics.get(k), (int, float))
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
    
    # CRITICAL DEBUG: Print the entire config
    print("=" * 70)
    print("SERVER CONFIG DEBUG")
    print("=" * 70)
    print(f"Config type: {type(cfg)}")
    print(f"Config keys: {sorted(cfg.keys()) if isinstance(cfg, dict) else 'N/A'}")
    print(f"Full config dict:")
    for k, v in sorted(cfg.items()) if isinstance(cfg, dict) else []:
        print(f"  {k}: {v} (type: {type(v).__name__})")
    print("=" * 70)
    
    # Try to read num_rounds with multiple key variants
    num_rounds = None
    for key in ["num-rounds", "num_rounds", "numRounds", "numrounds"]:
        if key in cfg:
            num_rounds = cfg[key]
            print(f"[Config] Found num_rounds via key '{key}': {num_rounds}")
            break
    
    if num_rounds is None:
        print("[Config] WARNING: num_rounds not found, using default 3")
        num_rounds = 3
    else:
        try:
            num_rounds = int(num_rounds)
        except (ValueError, TypeError):
            print(f"[Config] WARNING: Invalid num_rounds value: {num_rounds}, using 3")
            num_rounds = 3
    
    # Try to read use_dp with multiple key variants
    use_dp = False
    for key in ["use-dp", "use_dp", "useDp", "useDP"]:
        if key in cfg:
            val = cfg[key]
            print(f"[Config] Found use_dp candidate via key '{key}': {val} (type: {type(val).__name__})")
            # Handle boolean or string
            if isinstance(val, bool):
                use_dp = val
            elif isinstance(val, str):
                use_dp = val.lower() in ("true", "1", "yes", "on")
            elif isinstance(val, (int, float)):
                use_dp = bool(val)
            if use_dp:
                print(f"[Config] use_dp ENABLED via key '{key}'")
                break
    
    # Try to read target_epsilon
    target_epsilon = 3.0  # default
    for key in ["target-epsilon", "target_epsilon", "targetEpsilon", "epsilon"]:
        if key in cfg:
            try:
                target_epsilon = float(cfg[key])
                print(f"[Config] Found target_epsilon via key '{key}': {target_epsilon}")
                break
            except (ValueError, TypeError):
                pass
    
    print(f"[Server] Starting with {num_rounds} rounds")
    if use_dp:
        print(f"[Server] DP-SGD ENABLED, target ε = {target_epsilon}")
    else:
        print(f"[Server] DP-SGD disabled (use_dp={use_dp})")
    
    # TEMPORARY: Force enable DP for testing if requested in pyproject.toml
    # Check if we should force it based on federation name or other hints
    if not use_dp:
        print("[Server] NOTE: DP not enabled via config. To force DP, set use-dp=true in pyproject.toml")
    
    # Create initial parameters
    from try_project.task import load_model
    dummy_model = load_model(input_shape=(100, 3), num_classes=6)
    initial_params = ndarrays_to_parameters(dummy_model.get_weights())
    
    # Standard FedAvg with metrics aggregation
    base_strategy = FedAvg(
        fraction_fit=cfg.get("fraction-train", cfg.get("fraction_train", 1.0)),
        fraction_evaluate=cfg.get("fraction-evaluate", cfg.get("fraction_evaluate", 0.5)),
        min_fit_clients=3,
        min_evaluate_clients=1,
        min_available_clients=3,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Check if noisy channel is enabled
    use_noisy = False
    for key in ["use-noisy-channel", "use_noisy_channel", "useNoisyChannel"]:
        if key in cfg:
            val = cfg[key]
            if isinstance(val, bool):
                use_noisy = val
            elif isinstance(val, str):
                use_noisy = val.lower() == "true"
            break
    
    if use_noisy:
        print("[Server] NOISY CHANNEL enabled")
        
        channel = SimpleGilbertElliott(
            p_gb=cfg.get("p-gb", cfg.get("p_gb", 0.05)),
            p_bg=cfg.get("p-bg", cfg.get("p_bg", 0.95)),
            loss_good=cfg.get("loss-good", cfg.get("loss_good", 0.01)),
            loss_bad=cfg.get("loss-bad", cfg.get("loss_bad", 0.25)),
            base_latency_ms=cfg.get("base-latency-ms", cfg.get("base_latency_ms", 50.0)),
            seed=cfg.get("simulation-seed", cfg.get("simulation_seed", 42)),
        )
        
        strategy = PassiveNetworkWrapper(
            base_strategy=base_strategy,
            channel=channel,
            model_bytes=cfg.get("model-size-bytes", cfg.get("model_size_bytes", 1_200_000)),
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