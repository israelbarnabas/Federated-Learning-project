from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, Any, List, Tuple
import numpy as np
import tensorflow as tf

from try_project.task import get_client_data, load_model
from try_project.dp_utils import (
    check_dp_available,
    compute_epsilon,
    find_noise_multiplier_for_epsilon,
    apply_dp_to_gradients,
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
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train and return updated weights with optional DP-SGD."""
        # Handle empty parameters
        if not parameters or len(parameters) == 0:
            print(f"[Client {self.cid}] WARNING: Empty parameters, using current weights")
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        epochs = config.get("local-epochs") or config.get("local_epochs", 1)
        batch_size = config.get("batch-size") or config.get("batch_size", 32)
        
        # Check if DP is enabled
        use_dp = self.use_dp and self.dp_config
        noise_multiplier = self.dp_config.get("noise_multiplier", 0) if use_dp else 0
        l2_norm_clip = self.dp_config.get("l2_norm_clip", 1.0) if use_dp else 0
        
        if use_dp and noise_multiplier > 0:
            # DP-SGD training with manual gradient processing
            print(f"[Client {self.cid}] Training with DP-SGD (σ={noise_multiplier:.4f}, C={l2_norm_clip})")
            history = self._train_with_dp(
                epochs=epochs,
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=l2_norm_clip,
            )
        else:
            # Standard training
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
        if use_dp and self.dp_config:
            metrics.update({
                "dp_epsilon": self.dp_config.get("epsilon", 0),
                "dp_noise_multiplier": noise_multiplier,
                "dp_l2_clip": l2_norm_clip,
            })
        
        # Return TUPLE: (parameters, num_examples, metrics)
        return self.model.get_weights(), len(self.x_train), metrics
    
    def _train_with_dp(self, epochs: int, batch_size: int, noise_multiplier: float, l2_norm_clip: float):
        """Custom training loop with DP-SGD (gradient clipping + noise)."""
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        # Get optimizer and loss from model
        optimizer = self.model.optimizer
        loss_fn = self.model.compiled_loss
        
        # For compiled model, we need to handle metrics manually
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            epoch_losses = []
            correct_predictions = 0
            total_samples = 0
            
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self.model(x_batch, training=True)
                    # Compute loss
                    loss = loss_fn(y_batch, predictions)
                
                # Compute gradients
                trainable_vars = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                
                # Apply DP: clip gradients and add noise
                dp_gradients = apply_dp_to_gradients(
                    gradients, 
                    noise_multiplier=noise_multiplier, 
                    l2_norm_clip=l2_norm_clip
                )
                
                # Apply processed gradients
                optimizer.apply_gradients(zip(dp_gradients, trainable_vars))
                
                # Track metrics
                epoch_losses.append(float(loss))
                
                # Compute accuracy for this batch
                pred_classes = tf.argmax(predictions, axis=1)
                true_classes = tf.cast(y_batch, pred_classes.dtype)
                correct_predictions += tf.reduce_sum(
                    tf.cast(pred_classes == true_classes, tf.int32)
                ).numpy()
                total_samples += len(y_batch)
            
            # Record epoch metrics
            history["loss"].append(np.mean(epoch_losses))
            history["accuracy"].append(correct_predictions / total_samples if total_samples > 0 else 0)
            
            print(f"[Client {self.cid}] Epoch {epoch+1}/{epochs}: loss={history['loss'][-1]:.4f}, acc={history['accuracy'][-1]:.4f}")
        
        # Return history-like object
        class SimpleHistory:
            pass
        h = SimpleHistory()
        h.history = history
        return h
    
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
    Create model and compute DP parameters.
    
    Returns:
        Tuple of (model, dp_info_dict)
    """
    from try_project.task import load_model
    
    dp_info = {}
    
    # Always create standard model first
    model = load_model(input_shape=input_shape, num_classes=num_classes)
    
    if not use_dp:
        return model, dp_info
    
    # DP enabled: compute noise multiplier for target epsilon
    print(f"[DP] Configuring DP-SGD for {num_samples} samples")
    
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
    
    dp_info = {
        "epsilon": achieved_eps,
        "delta": target_delta,
        "noise_multiplier": noise_multiplier,
        "l2_norm_clip": l2_norm_clip,
        "target_epsilon": target_epsilon,
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
    
    # Create model (with DP config if enabled)
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
        dp_config=dp_info if use_dp else None,
    )
    
    # CRITICAL: Convert to Client using .to_client()
    return numpy_client.to_client()


app = ClientApp(client_fn=client_fn)