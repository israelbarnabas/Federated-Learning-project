from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf

from try_project.task import get_client_data, load_model, get_model_size
from try_project.dp_utils import (
    check_dp_available,
    compute_epsilon,
    find_noise_multiplier_for_epsilon,
    apply_dp_to_gradients,
)
from try_project.secure_agg import SecureAggregation, ProperSecureAggregation


class BaselineClient(NumPyClient):
    """
    Enhanced client with proper DP-SGD, FedProx support, and client-side SA masking.
    """
    
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
        use_fedprox: bool = False,
        fedprox_mu: float = 0.0,
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
        self.use_fedprox = use_fedprox
        self.fedprox_mu = fedprox_mu
        
        # Store global weights for FedProx
        self.global_weights: Optional[List[np.ndarray]] = None
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train with optional DP-SGD, FedProx, and client-side SA masking."""
        # Handle empty parameters
        if not parameters or len(parameters) == 0:
            print(f"[Client {self.cid}] WARNING: Empty parameters, using current weights")
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        # Store global weights for FedProx
        self.global_weights = [p.copy() for p in parameters]
        
        # Get training config
        epochs = config.get("local-epochs") or config.get("local_epochs", 3)
        batch_size = config.get("batch-size") or config.get("batch_size", 32)
        
        # Check for SA configuration from server
        sa_enabled = config.get("sa_enabled", False)
        sa_round_seed = config.get("sa_round_seed")
        sa_all_clients = config.get("sa_all_clients", [])
        sa_threshold = config.get("sa_threshold", 20)
        
        # Determine training mode
        use_dp = self.use_dp and self.dp_config and self.dp_config.get("noise_multiplier", 0) > 0
        
        if use_dp:
            noise_multiplier = self.dp_config.get("noise_multiplier", 0)
            l2_norm_clip = self.dp_config.get("l2_norm_clip", 0.5)
            print(f"[Client {self.cid}] DP-SGD: σ={noise_multiplier:.4f}, C={l2_norm_clip}, epochs={epochs}")
            
            history = self._train_with_dp(
                epochs=epochs,
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
                l2_norm_clip=l2_norm_clip,
            )
        else:
            if self.use_fedprox and self.fedprox_mu > 0:
                print(f"[Client {self.cid}] FedProx: μ={self.fedprox_mu}, epochs={epochs}")
                history = self._train_with_fedprox(
                    epochs=epochs,
                    batch_size=batch_size,
                    mu=self.fedprox_mu,
                )
            else:
                print(f"[Client {self.cid}] Standard training: epochs={epochs}")
                history = self.model.fit(
                    self.x_train, self.y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        # Apply SA masking on CLIENT SIDE (correct flow)
        if sa_enabled and sa_round_seed is not None:
            use_pairwise = len(sa_all_clients) > 0 and len(sa_all_clients) == self.num_clients
            
            if use_pairwise:
                sa = ProperSecureAggregation(num_clients=self.num_clients, threshold=sa_threshold)
                weight_shapes = [w.shape for w in updated_weights]
                masks = sa.generate_pairwise_masks(
                    self.cid, sa_round_seed, weight_shapes, sa_all_clients
                )
                print(f"[Client {self.cid}] SA pairwise masking applied")
            else:
                sa = SecureAggregation(num_clients=self.num_clients, threshold=sa_threshold)
                weight_shapes = [w.shape for w in updated_weights]
                masks = sa.generate_masks(self.cid, sa_round_seed, weight_shapes)
                print(f"[Client {self.cid}] SA standard masking applied")
            
            masked_weights = sa.mask_weights(updated_weights, masks)
            updated_weights = masked_weights
        
        # Prepare metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "client_id": self.cid,
            "num_samples": len(self.x_train),
        }
        
        # Add DP metrics
        if use_dp:
            metrics.update({
                "dp_epsilon": self.dp_config.get("epsilon", 0),
                "dp_delta": self.dp_config.get("delta", 1e-4),
                "dp_noise_multiplier": self.dp_config.get("noise_multiplier", 0),
                "dp_l2_clip": self.dp_config.get("l2_norm_clip", 0.5),
            })
        
        # Add FedProx metrics
        if self.use_fedprox:
            metrics["fedprox_mu"] = self.fedprox_mu
        
        # Add SA metrics
        if sa_enabled:
            metrics["sa_applied"] = True
            metrics["sa_seed"] = sa_round_seed
        
        return updated_weights, len(self.x_train), metrics
    
    def _train_with_dp(
        self, 
        epochs: int, 
        batch_size: int, 
        noise_multiplier: float, 
        l2_norm_clip: float,
    ):
        """Custom training loop with proper DP-SGD."""
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.shuffle(buffer_size=min(1000, len(self.x_train)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        optimizer = self.model.optimizer
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            epoch_losses = []
            correct_predictions = 0
            total_samples = 0
            
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch, training=True)
                    loss = loss_fn(y_batch, predictions)
                
                trainable_vars = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                
                dp_gradients = apply_dp_to_gradients(
                    gradients, 
                    noise_multiplier=noise_multiplier, 
                    l2_norm_clip=l2_norm_clip
                )
                
                optimizer.apply_gradients(zip(dp_gradients, trainable_vars))
                
                epoch_losses.append(float(loss))
                pred_classes = tf.argmax(predictions, axis=1)
                true_classes = tf.cast(y_batch, pred_classes.dtype)
                correct_predictions += int(tf.reduce_sum(
                    tf.cast(pred_classes == true_classes, tf.int32)
                ))
                total_samples += len(y_batch)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if epoch == 0 or epoch == epochs - 1:
                print(f"[Client {self.cid}] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")
        
        class History:
            pass
        h = History()
        h.history = history
        return h
    
    def _train_with_fedprox(
        self,
        epochs: int,
        batch_size: int,
        mu: float,
    ):
        """FedProx training with proximal term."""
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.shuffle(buffer_size=min(1000, len(self.x_train)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        optimizer = self.model.optimizer
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            epoch_losses = []
            correct_predictions = 0
            total_samples = 0
            
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch, training=True)
                    base_loss = loss_fn(y_batch, predictions)
                    
                    if self.global_weights is not None:
                        prox_term = 0.0
                        for w, w_global in zip(self.model.trainable_variables, self.global_weights):
                            prox_term += tf.reduce_sum(tf.square(w - w_global))
                        total_loss = base_loss + (mu / 2.0) * prox_term
                    else:
                        total_loss = base_loss
                    
                    epoch_losses.append(float(base_loss))
                
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                pred_classes = tf.argmax(predictions, axis=1)
                true_classes = tf.cast(y_batch, pred_classes.dtype)
                correct_predictions += int(tf.reduce_sum(
                    tf.cast(pred_classes == true_classes, tf.int32)
                ))
                total_samples += len(y_batch)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if epoch == 0 or epoch == epochs - 1:
                print(f"[Client {self.cid}] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f} (FedProx μ={mu})")
        
        class History:
            pass
        h = History()
        h.history = history
        return h
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """Evaluate model."""
        if not parameters or len(parameters) == 0:
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def create_model_with_dp(
    input_shape: Tuple[int, int],
    num_classes: int,
    use_dp: bool,
    dp_config: Dict[str, Any],
    num_samples: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Create model and compute DP parameters."""
    from try_project.task import load_model
    
    dp_info = {}
    model = load_model(input_shape=input_shape, num_classes=num_classes, for_dp=use_dp)
    
    if not use_dp:
        return model, dp_info
    
    print(f"[DP] Configuring DP-SGD for {num_samples} samples")
    
    target_epsilon = dp_config.get("target_epsilon", 5.0)
    target_delta = dp_config.get("target_delta", 1e-4)
    l2_norm_clip = dp_config.get("l2_norm_clip", 0.5)
    local_epochs = dp_config.get("local_epochs", 3)
    batch_size = dp_config.get("batch_size", 32)
    noise_multiplier = dp_config.get("noise_multiplier", 0.0)
    
    if noise_multiplier <= 0:
        noise_multiplier, achieved_eps = find_noise_multiplier_for_epsilon(
            target_epsilon=target_epsilon,
            num_samples=num_samples,
            batch_size=batch_size,
            epochs=local_epochs,
            delta=target_delta,
        )
        print(f"[DP] Computed σ={noise_multiplier:.4f} for ε={achieved_eps:.2f}")
    else:
        achieved_eps, _ = compute_epsilon(
            num_samples=num_samples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=local_epochs,
            delta=target_delta,
        )
        print(f"[DP] Using σ={noise_multiplier:.4f}, achieves ε={achieved_eps:.2f}")
    
    dp_info = {
        "epsilon": achieved_eps,
        "delta": target_delta,
        "noise_multiplier": noise_multiplier,
        "l2_norm_clip": l2_norm_clip,
        "target_epsilon": target_epsilon,
    }
    
    return model, dp_info


def client_fn(context: Context):
    """Create client instance with all features."""
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
    
    num_clients = cfg.get("num-clients", cfg.get("num_clients", 30))
    partition_id = node_id_int % num_clients
    
    print(f"[Client] node_id={raw_node_id} -> partition_id={partition_id}")
    
    # Load data with improved task.py
    non_iid = cfg.get("non-iid", cfg.get("non_iid", True))
    alpha = cfg.get("alpha", 0.5)
    data_seed = cfg.get("data-seed", cfg.get("data_seed", 42))
    
    try:
        x_train, y_train, x_test, y_test = get_client_data(
            client_id=partition_id,
            num_clients=num_clients,
            non_iid=non_iid,
            alpha=alpha,
            seed=data_seed,
        )
        print(f"[Client {partition_id}] Loaded: {len(x_train)} train, {len(x_test)} test")
        print(f"[Client {partition_id}] Data range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    except Exception as e:
        print(f"[Client] ERROR loading data: {e}")
        x_train = np.random.randn(100, 100, 3).astype(np.float32)
        y_train = np.random.randint(0, 6, 100)
        x_test = np.random.randn(20, 100, 3).astype(np.float32)
        y_test = np.random.randint(0, 6, 20)
    
    # Check features
    use_dp = cfg.get("use-dp", cfg.get("use_dp", False))
    use_fedprox = cfg.get("use-fedprox", cfg.get("use_fedprox", False))
    fedprox_mu = cfg.get("fedprox-mu", cfg.get("fedprox_mu", 0.1))
    
    dp_info = {}
    
    if use_dp:
        print(f"[Client {partition_id}] DP-SGD ENABLED")
        dp_config = {
            "target_epsilon": cfg.get("target-epsilon", cfg.get("target_epsilon", 5.0)),
            "target_delta": cfg.get("target-delta", cfg.get("target_delta", 1e-4)),
            "l2_norm_clip": cfg.get("l2-norm-clip", cfg.get("l2_norm_clip", 0.5)),
            "local_epochs": cfg.get("local-epochs", cfg.get("local_epochs", 3)),
            "batch_size": cfg.get("batch-size", cfg.get("batch_size", 32)),
        }
    else:
        dp_config = {}
    
    # Create model
    try:
        model, dp_info = create_model_with_dp(
            input_shape=(100, 3),
            num_classes=6,
            use_dp=use_dp,
            dp_config=dp_config if use_dp else {},
            num_samples=len(x_train),
        )
    except Exception as e:
        print(f"[Client {partition_id}] ERROR creating model: {e}")
        model = load_model(input_shape=(100, 3), num_classes=6, for_dp=False)
        use_dp = False
    
    # Log model size
    model_size = get_model_size(model)
    print(f"[Client {partition_id}] Model size: {model_size/1e6:.2f} MB")
    
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
        use_fedprox=use_fedprox,
        fedprox_mu=fedprox_mu,
    )
    
    return numpy_client.to_client()


app = ClientApp(client_fn=client_fn)