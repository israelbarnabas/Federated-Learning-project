from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, Any, List, Tuple
import numpy as np

from try_project.task import get_client_data, load_model
from try_project.dp_utils import (
    check_dp_available,
    create_dp_optimizer,
    compute_epsilon,
    find_noise_multiplier_for_epsilon,
)


class BaselineClient(NumPyClient):
    def __init__(
        self, 
        cid: int, 
        num_clients: int, 
        model, 
        x_train, 
        y_train, 
        x_test, 
        y_test,
        use_dp: bool = False,
        dp_config: Dict[str, Any] = None,
    ):
        self.cid = cid
        self.num_clients = num_clients
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.use_dp = use_dp
        self.dp_config = dp_config or {}
        
        # Track DP metrics
        self.dp_metrics = {}
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train and return updated weights as TUPLE."""
        # Handle empty parameters
        if not parameters or len(parameters) == 0:
            print(f"[Client {self.cid}] WARNING: Empty parameters, using current weights")
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        epochs = config.get("local-epochs") or config.get("local_epochs", 1)
        batch_size = config.get("batch-size") or config.get("batch_size", 32)
        
        # Train
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        
        # Prepare metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "client_id": self.cid,
        }
        
        # Add DP metrics if enabled
        if self.use_dp and self.dp_metrics:
            metrics.update({
                "dp_epsilon": self.dp_metrics.get("epsilon", 0),
                "dp_noise_multiplier": self.dp_metrics.get("noise_multiplier", 0),
                "dp_l2_clip": self.dp_metrics.get("l2_norm_clip", 0),
            })
        
        # Return TUPLE: (parameters, num_examples, metrics)
        return self.model.get_weights(), len(self.x_train), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """Evaluate and return metrics as TUPLE."""
        # Handle empty parameters
        if not parameters or len(parameters) == 0:
            print(f"[Client {self.cid}] WARNING: Empty parameters in evaluate")
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Return TUPLE: (loss, num_examples, metrics)
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def create_model_with_dp(
    input_shape: Tuple[int, int],
    num_classes: int,
    use_dp: bool,
    dp_config: Dict[str, Any],
    num_samples: int,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create model, optionally with DP-SGD optimizer.
    
    Returns:
        Tuple of (model, dp_info_dict)
    """
    from try_project.task import load_model
    
    dp_info = {}
    
    if not use_dp:
        # Standard non-DP model
        model = load_model(input_shape=input_shape, num_classes=num_classes)
        return model, dp_info
    
    # DP-enabled model
    if not check_dp_available():
        raise RuntimeError("DP requested but tensorflow-privacy not installed")
    
    # Extract DP parameters
    target_epsilon = dp_config.get("target_epsilon", 5.0)
    target_delta = dp_config.get("target_delta", 1e-4)
    l2_norm_clip = dp_config.get("l2_norm_clip", 1.0)
    local_epochs = dp_config.get("local_epochs", 3)
    batch_size = dp_config.get("batch_size", 32)
    noise_multiplier = dp_config.get("noise_multiplier", 0.0)
    
    # If noise_multiplier not specified, compute from target epsilon
    if noise_multiplier <= 0:
        noise_multiplier, achieved_eps = find_noise_multiplier_for_epsilon(
            target_epsilon=target_epsilon,
            num_samples=num_samples,
            batch_size=batch_size,
            epochs=local_epochs,
            delta=target_delta,
        )
        print(f"[DP] Computed noise_multiplier={noise_multiplier:.4f} for ε={achieved_eps:.2f}")
    else:
        # Verify achieved epsilon
        achieved_eps, _ = compute_epsilon(
            num_samples=num_samples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=local_epochs,
            delta=target_delta,
        )
        print(f"[DP] Using noise_multiplier={noise_multiplier:.4f}, achieves ε={achieved_eps:.2f}")
    
    # Determine number of microbatches
    num_microbatches = dp_config.get("num_microbatches", 1)
    if num_microbatches > batch_size:
        num_microbatches = batch_size
    
    # Create DP optimizer
    optimizer = create_dp_optimizer(
        noise_multiplier=noise_multiplier,
        l2_norm_clip=l2_norm_clip,
        num_microbatches=num_microbatches,
        learning_rate=0.001,
        optimizer_type="sgd",
    )
    
    # Build model with DP optimizer
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers, Input
    
    model = Sequential([
        Input(shape=input_shape),
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    dp_info = {
        "epsilon": achieved_eps,
        "delta": target_delta,
        "noise_multiplier": noise_multiplier,
        "l2_norm_clip": l2_norm_clip,
        "num_microbatches": num_microbatches,
    }
    
    return model, dp_info


def client_fn(context: Context):
    """Create client instance."""
    cfg = context.run_config
    
    # Parse node_id to get partition_id
    raw_node_id = context.node_id
    if isinstance(raw_node_id, str):
        if raw_node_id.isdigit():
            node_id_int = int(raw_node_id)
        else:
            digits = ''.join(filter(str.isdigit, raw_node_id))
            node_id_int = int(digits) if digits else 0
    else:
        node_id_int = int(raw_node_id)
    
    num_clients = cfg.get("num-clients") or cfg.get("num_clients", 30)
    partition_id = node_id_int % num_clients
    
    print(f"[Client] node_id={raw_node_id} -> partition_id={partition_id}")
    
    # Load data
    non_iid = cfg.get("non-iid") or cfg.get("non_iid", True)
    alpha = cfg.get("alpha", 0.5)
    
    try:
        x_train, y_train, x_test, y_test = get_client_data(
            client_id=partition_id,
            num_clients=num_clients,
            non_iid=non_iid,
            alpha=alpha,
        )
        print(f"[Client {partition_id}] Loaded: {len(x_train)} train, {len(x_test)} test")
    except Exception as e:
        print(f"[Client] ERROR loading data: {e}")
        # Fallback data
        x_train = np.random.randn(50, 100, 3).astype(np.float32)
        y_train = np.random.randint(0, 6, 50)
        x_test = np.random.randn(10, 100, 3).astype(np.float32)
        y_test = np.random.randint(0, 6, 10)
    
    # Check if DP is enabled
    use_dp = cfg.get("use-dp") or cfg.get("use_dp", False)
    dp_config = None
    dp_info = {}
    
    if use_dp:
        print(f"[Client {partition_id}] DP-SGD ENABLED")
        dp_config = {
            "target_epsilon": cfg.get("target-epsilon") or cfg.get("target_epsilon", 5.0),
            "target_delta": cfg.get("target-delta") or cfg.get("target_delta", 1e-4),
            "l2_norm_clip": cfg.get("l2-norm-clip") or cfg.get("l2_norm_clip", 1.0),
            "noise_multiplier": cfg.get("noise-multiplier") or cfg.get("noise_multiplier", 0.0),
            "num_microbatches": cfg.get("num-microbatches") or cfg.get("num_microbatches", 1),
            "local_epochs": cfg.get("local-epochs") or cfg.get("local_epochs", 3),
            "batch_size": cfg.get("batch-size") or cfg.get("batch_size", 32),
        }
    
    # Create model (with DP if enabled)
    try:
        model, dp_info = create_model_with_dp(
            input_shape=(100, 3),
            num_classes=6,
            use_dp=use_dp,
            dp_config=dp_config,
            num_samples=len(x_train),
        )
    except Exception as e:
        print(f"[Client {partition_id}] ERROR creating model: {e}")
        # Fallback to non-DP
        model = load_model(input_shape=(100, 3), num_classes=6)
        use_dp = False
    
    # Create client
    numpy_client = BaselineClient(
        cid=partition_id,
        num_clients=num_clients,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        use_dp=use_dp,
        dp_config=dp_info,
    )
    
    # Store DP info for metrics
    if dp_info:
        numpy_client.dp_metrics = dp_info
    
    # CRITICAL: Convert to Client using .to_client()
    return numpy_client.to_client()


app = ClientApp(client_fn=client_fn)