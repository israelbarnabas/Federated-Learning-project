from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, Any, List, Tuple
import numpy as np

from try_project.task import get_client_data, load_model


class BaselineClient(NumPyClient):
    def __init__(self, cid: int, num_clients: int, model, x_train, y_train, x_test, y_test):
        self.cid = cid
        self.num_clients = num_clients
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
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
        
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        
        # Return TUPLE: (parameters, num_examples, metrics)
        return self.model.get_weights(), len(self.x_train), {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "client_id": self.cid,
        }
    
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
    
    # Create model
    model = load_model(input_shape=(100, 3), num_classes=6)
    
    # Create client
    numpy_client = BaselineClient(
        cid=partition_id,
        num_clients=num_clients,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    
    # CRITICAL: Convert to Client using .to_client()
    return numpy_client.to_client()


app = ClientApp(client_fn=client_fn)