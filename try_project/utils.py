"""
Utility functions for reproducibility, logging, and data handling.
"""

import os
import random
import numpy as np
import tensorflow as tf
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from contextlib import contextmanager


def set_global_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # TensorFlow deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


class ExperimentLogger:
    """
    Structured logging for experiment metrics.
    """
    
    def __init__(self, log_dir: str, experiment_id: str):
        self.log_dir = Path(log_dir) / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.summary_file = self.log_dir / "summary.json"
        self.start_time = time.time()
        self.round_metrics: List[Dict[str, Any]] = []
        
    def log_round(self, round_num: int, metrics: Dict[str, Any]):
        """Log metrics for a single round."""
        entry = {
            "round": round_num,
            "timestamp": time.time() - self.start_time,
            **metrics
        }
        self.round_metrics.append(entry)
        
        # Append to JSONL file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary."""
        summary["total_time"] = time.time() - self.start_time
        summary["num_rounds"] = len(self.round_metrics)
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all logged metrics."""
        return self.round_metrics


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.elapsed = None
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def moving_average(data: List[float], window: int = 5) -> List[float]:
    """Calculate moving average."""
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result