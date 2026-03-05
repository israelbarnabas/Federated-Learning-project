"""
Centralized configuration management for reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class NetworkConfig:
    """Network simulation parameters."""
    enabled: bool = False
    p_gb: float = 0.05          # Good -> Bad transition
    p_bg: float = 0.95          # Bad -> Good transition  
    loss_good: float = 0.01     # Loss rate in good state
    loss_bad: float = 0.25      # Loss rate in bad state
    base_latency_ms: float = 50.0
    bandwidth_mbps: float = 10.0
    bad_bandwidth_factor: float = 0.5
    jitter_ms: float = 10.0
    round_budget_s: float = 2.0  # Deadline for client participation
    
    # Mobility patterns
    mobility_type: str = "static"  # static, pedestrian, vehicular
    mobility_speed_m_s: float = 0.0
    
    # Random seed for network conditions
    seed: int = 42


@dataclass
class FLConfig:
    """Federated Learning parameters."""
    num_rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    fraction_fit: float = 1.0
    fraction_evaluate: float = 0.5
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    
    # Data configuration
    non_iid: bool = True
    alpha: float = 0.5  # Dirichlet concentration
    
    # Adaptive training
    adaptive_training: bool = False  # Adjust epochs/batch based on link quality


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_id: str = "exp_001"
    random_seed: int = 42
    network_seed: int = 42
    
    fl: FLConfig = field(default_factory=FLConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Logging
    log_dir: str = "results"
    save_models: bool = False
    
    # Reproducibility
    deterministic: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "random_seed": self.random_seed,
            "network_seed": self.network_seed,
            "fl": self.fl.__dict__,
            "network": self.network.__dict__,
            "log_dir": self.log_dir,
            "save_models": self.save_models,
            "deterministic": self.deterministic,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        fl_config = FLConfig(**d.get("fl", {}))
        net_config = NetworkConfig(**d.get("network", {}))
        return cls(
            experiment_id=d.get("experiment_id", "exp_001"),
            random_seed=d.get("random_seed", 42),
            network_seed=d.get("network_seed", 42),
            fl=fl_config,
            network=net_config,
            log_dir=d.get("log_dir", "results"),
            save_models=d.get("save_models", False),
            deterministic=d.get("deterministic", True),
        )
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON."""
        if path is None:
            path = Path(self.log_dir) / f"{self.experiment_id}_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Default configurations for common experiments
BASELINE_CONFIG = ExperimentConfig(
    experiment_id="baseline",
    network=NetworkConfig(enabled=False),
)

UNRELIABLE_CONFIG = ExperimentConfig(
    experiment_id="unreliable",
    network=NetworkConfig(enabled=True, loss_bad=0.25),
)

ADAPTIVE_CONFIG = ExperimentConfig(
    experiment_id="adaptive",
    fl=FLConfig(adaptive_training=True),
    network=NetworkConfig(enabled=True, loss_bad=0.25),
)