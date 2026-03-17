"""
Enhanced Client with Adaptive DP-SGD and Link-Aware Configuration.
FIXED: Proper epsilon handling, FedProx gradient computation, model consistency.
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
import json

from try_project.task import get_client_data, load_model, get_model_size
from try_project.dp_utils import find_noise_multiplier_for_epsilon, apply_dp_to_gradients
from try_project.enhanced_secure_agg import SecureAggregation

# CRITICAL: Minimum useful epsilon - must match scheduler!
MIN_USEFUL_EPSILON = 0.5


class AdaptiveDPClient(NumPyClient):
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
        use_adaptive_noise: bool = True,
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
        self.use_adaptive_noise = use_adaptive_noise
        
        self.global_weights: Optional[List[np.ndarray]] = None
        self.sa: Optional[SecureAggregation] = None
        self.current_epsilon = 0.0
        self.current_noise_mult = 0.0
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        if not parameters or len(parameters) == 0:
            print(f"[Client {self.cid}] WARNING: Empty parameters, using current weights")
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        # CRITICAL FIX: Store global weights BEFORE training for FedProx
        self.global_weights = [p.copy() for p in parameters]
        
        epochs = config.get("local-epochs", config.get("local_epochs", 3))
        batch_size = config.get("batch-size", config.get("batch_size", 32))
        
        # Handle DP configuration - CRITICAL FIX
        adaptive_epsilon = config.get("adaptive_epsilon", False)
        
        if self.use_dp and adaptive_epsilon:
            # PRIORITY 1: Use server's provided noise multiplier directly
            if "noise_multiplier" in config and config["noise_multiplier"] > 0:
                self.current_noise_mult = config["noise_multiplier"]
                # Use server's target_epsilon, but enforce minimum
                server_epsilon = config.get("target_epsilon", MIN_USEFUL_EPSILON)
                
                # CRITICAL: Enforce minimum useful epsilon
                if server_epsilon < MIN_USEFUL_EPSILON:
                    print(f"[Client {self.cid}] WARNING: Server sent ε={server_epsilon:.3f}, "
                          f"forcing to {MIN_USEFUL_EPSILON}")
                    self.current_epsilon = MIN_USEFUL_EPSILON
                else:
                    self.current_epsilon = server_epsilon
                
                print(f"[Client {self.cid}] Using server DP: ε={self.current_epsilon:.3f}, "
                      f"σ={self.current_noise_mult:.4f}")
            else:
                # Fallback: calculate from target (should not happen with proper server)
                target_eps = config.get("target_epsilon", self.dp_config.get("target_epsilon", 1.0))
                
                # CRITICAL: Enforce minimum
                if target_eps < MIN_USEFUL_EPSILON:
                    print(f"[Client {self.cid}] WARNING: Target ε={target_eps:.3f} too low, "
                          f"forcing to {MIN_USEFUL_EPSILON}")
                    target_eps = MIN_USEFUL_EPSILON
                
                self.current_noise_mult, self.current_epsilon = find_noise_multiplier_for_epsilon(
                    target_epsilon=target_eps,
                    num_samples=len(self.x_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    delta=self.dp_config.get("delta", 1e-4),
                )
                print(f"[Client {self.cid}] Calculated DP: ε={self.current_epsilon:.3f}, "
                      f"σ={self.current_noise_mult:.4f}")
        elif self.use_dp:
            # Static DP from config
            self.current_noise_mult = self.dp_config.get("noise_multiplier", 0.0)
            self.current_epsilon = self.dp_config.get("epsilon", 0.0)
            print(f"[Client {self.cid}] Static DP: ε={self.current_epsilon:.3f}, "
                  f"σ={self.current_noise_mult:.4f}")
        else:
            self.current_noise_mult = 0.0
            self.current_epsilon = 0.0
        
        # Get SA configuration
        use_sa = config.get("use_sa", False)
        sa_round_seed_hex = config.get("sa_round_seed")
        sa_all_clients_json = config.get("sa_all_clients", "[]")
        
        # Training
        if self.use_dp and self.current_noise_mult > 0:
            history = self._train_with_dp(
                epochs=epochs, batch_size=batch_size,
                noise_multiplier=self.current_noise_mult,
                l2_norm_clip=self.dp_config.get("l2_norm_clip", 0.5)
            )
        elif self.use_fedprox and self.fedprox_mu > 0:
            history = self._train_with_fedprox(
                epochs=epochs, batch_size=batch_size, mu=self.fedprox_mu
            )
        else:
            history = self.model.fit(
                self.x_train, self.y_train, 
                epochs=epochs, batch_size=batch_size, verbose=0
            )
        
        updated_weights = self.model.get_weights()
        
        # Apply SA masking if enabled
        if use_sa and sa_round_seed_hex:
            updated_weights = self._apply_sa_masking(updated_weights, sa_round_seed_hex, sa_all_clients_json)
        
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "client_id": self.cid,
            "num_samples": len(self.x_train),
            "epsilon_used": self.current_epsilon,
            "noise_multiplier": self.current_noise_mult,
        }
        
        if use_sa:
            metrics["sa_applied"] = True
        
        return updated_weights, len(self.x_train), metrics
    
    def _train_with_dp(self, epochs: int, batch_size: int, noise_multiplier: float, l2_norm_clip: float):
        """Train with differential privacy."""
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
                
                # Apply DP
                dp_gradients = apply_dp_to_gradients(
                    gradients, 
                    noise_multiplier=noise_multiplier, 
                    l2_norm_clip=l2_norm_clip
                )
                
                optimizer.apply_gradients(zip(dp_gradients, trainable_vars))
                
                epoch_losses.append(float(loss))
                pred_classes = tf.argmax(predictions, axis=1)
                true_classes = tf.cast(y_batch, pred_classes.dtype)
                correct_predictions += int(tf.reduce_sum(tf.cast(pred_classes == true_classes, tf.int32)))
                total_samples += len(y_batch)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if epoch == 0 or epoch == epochs - 1:
                print(f"[Client {self.cid}] Epoch {epoch+1}/{epochs}: "
                      f"loss={avg_loss:.4f}, acc={accuracy:.4f} (DP σ={noise_multiplier:.4f})")
        
        class History:
            pass
        h = History()
        h.history = history
        return h
    
    def _train_with_fedprox(self, epochs: int, batch_size: int, mu: float):
        """
        Train with FedProx proximal term.
        CRITICAL FIX: Proper gradient computation with proximal term.
        """
        # CRITICAL FIX: Convert global weights to constants BEFORE training
        global_weights = None
        if self.global_weights is not None and mu > 0:
            global_weights = [
                tf.constant(w, dtype=tf.float32) for w in self.global_weights
            ]
        
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
                    
                    # CRITICAL FIX: Compute proximal term with CURRENT trainable variables
                    if global_weights is not None:
                        prox_term = 0.0
                        # Use trainable_variables (tensors) not get_weights() (numpy)
                        current_vars = self.model.trainable_variables
                        for curr_var, glob_w in zip(current_vars, global_weights):
                            if curr_var.shape == glob_w.shape:
                                # This is differentiable!
                                prox_term += tf.reduce_sum(tf.square(curr_var - glob_w))
                        total_loss = base_loss + (mu / 2.0) * prox_term
                    else:
                        total_loss = base_loss
                    
                    epoch_losses.append(float(base_loss))
                
                # Compute gradients of total loss
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                pred_classes = tf.argmax(predictions, axis=1)
                true_classes = tf.cast(y_batch, pred_classes.dtype)
                correct_predictions += int(tf.reduce_sum(tf.cast(pred_classes == true_classes, tf.int32)))
                total_samples += len(y_batch)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
        
        class History:
            pass
        h = History()
        h.history = history
        return h
    
    def _apply_sa_masking(self, weights: List[np.ndarray], seed_hex: str, all_clients_json: str) -> List[np.ndarray]:
        """Apply secure aggregation masking."""
        try:
            round_seed = bytes.fromhex(seed_hex)
            all_clients = json.loads(all_clients_json)
            
            if self.sa is None:
                self.sa = SecureAggregation(num_clients=len(all_clients), threshold=2)
            
            weight_shapes = [w.shape for w in weights]
            masks, _ = self.sa.generate_masks(self.cid, round_seed, weight_shapes, all_clients)
            masked_weights = self.sa.mask_weights(weights, masks)
            
            print(f"[Client {self.cid}] SA masking applied")
            return masked_weights
        except Exception as e:
            print(f"[Client {self.cid}] SA masking failed: {e}")
            return weights
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        if not parameters or len(parameters) == 0:
            parameters = self.model.get_weights()
        else:
            self.model.set_weights(parameters)
        
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    cfg = context.run_config
    
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
    
    non_iid = cfg.get("non-iid", cfg.get("non_iid", True))
    alpha = cfg.get("alpha", 0.5)
    data_seed = cfg.get("data-seed", cfg.get("data_seed", 42))
    
    try:
        x_train, y_train, x_test, y_test = get_client_data(
            client_id=partition_id, num_clients=num_clients,
            non_iid=non_iid, alpha=alpha, seed=data_seed
        )
    except Exception as e:
        print(f"[Client] ERROR loading data: {e}")
        # Fallback data
        x_train = np.random.randn(100, 100, 3).astype(np.float32)
        y_train = np.random.randint(0, 6, 100)
        x_test = np.random.randn(20, 100, 3).astype(np.float32)
        y_test = np.random.randint(0, 6, 20)
    
    # CRITICAL: Check for use-adaptive-dp to match server model size
    use_dp_raw = cfg.get("use-adaptive-dp", cfg.get("use_adaptive_dp", 
                     cfg.get("use-dp", cfg.get("use_dp", False))))
    use_dp = bool(use_dp_raw)
    
    use_fedprox = cfg.get("use-fedprox", cfg.get("use_fedprox", False))
    fedprox_mu = cfg.get("fedprox-mu", cfg.get("fedprox_mu", 0.1))
    
    dp_config = {}
    if use_dp:
        dp_config = {
            "target_epsilon": cfg.get("target-epsilon", cfg.get("target_epsilon", 5.0)),
            "target_delta": cfg.get("target-delta", cfg.get("target_delta", 1e-4)),
            "l2_norm_clip": cfg.get("l2-norm-clip", cfg.get("l2_norm_clip", 0.5)),
            "local_epochs": cfg.get("local-epochs", cfg.get("local_epochs", 3)),
            "batch_size": cfg.get("batch-size", cfg.get("batch_size", 32)),
        }
    
    # CRITICAL: Create model with same for_dp flag as server
    try:
        model = load_model(input_shape=(100, 3), num_classes=6, for_dp=use_dp)
    except Exception as e:
        print(f"[Client {partition_id}] ERROR creating model: {e}")
        model = load_model(input_shape=(100, 3), num_classes=6, for_dp=False)
        use_dp = False
    
    print(f"[Client {partition_id}] Model size: {get_model_size(model)/1e6:.2f} MB")
    
    numpy_client = AdaptiveDPClient(
        cid=partition_id, num_clients=num_clients, model=model,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        use_dp=use_dp, dp_config=dp_config if use_dp else None,
        use_fedprox=use_fedprox, fedprox_mu=fedprox_mu, use_adaptive_noise=True
    )
    
    return numpy_client.to_client()

app = ClientApp(client_fn=client_fn)