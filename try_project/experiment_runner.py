"""
Automated experiment runner for comparative analysis.
Executes sweeps over network conditions and collects results.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from itertools import product
import pandas as pd
from dataclasses import asdict

from try_project.config import (
    ExperimentConfig, 
    FLConfig, 
    NetworkConfig,
    BASELINE_CONFIG,
    UNRELIABLE_CONFIG,
    ADAPTIVE_CONFIG,
)


class ExperimentRunner:
    """
    Orchestrates multiple FL experiments with different configurations.
    """
    
    def __init__(
        self,
        output_dir: str = "experiments",
        num_supernodes: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.num_supernodes = num_supernodes
        self.results: List[Dict] = []
        
    def run_single(
        self,
        config: ExperimentConfig,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single experiment.
        """
        print(f"\n{'='*60}")
        print(f"Running: {config.experiment_id}")
        print(f"{'='*60}")
        
        # Save configuration
        config_path = self.output_dir / f"{config.experiment_id}_config.json"
        config.save(config_path)
        
        if dry_run:
            print(f"[DRY RUN] Would execute: flwr run . --run-config <config>")
            return {"status": "dry_run", "config": config.to_dict()}
        
        # Build run config
        run_config = {
            "experiment-id": config.experiment_id,
            "random-seed": config.random_seed,
            "network-seed": config.network_seed,
            "num-server-rounds": config.fl.num_rounds,
            "local-epochs": config.fl.local_epochs,
            "batch-size": config.fl.batch_size,
            "fraction-train": config.fl.fraction_fit,
            "fraction-evaluate": config.fl.fraction_evaluate,
            "min-fit-clients": config.fl.min_fit_clients,
            "min-evaluate-clients": config.fl.min_evaluate_clients,
            "non-iid": config.fl.non_iid,
            "alpha": config.fl.alpha,
            "adaptive-training": config.fl.adaptive_training,
            "use-ns3": config.network.enabled,
            "p-gb": config.network.p_gb,
            "p-bg": config.network.p_bg,
            "loss-good": config.network.loss_good,
            "loss-bad": config.network.loss_bad,
            "latency-ms": config.network.base_latency_ms,
            "bandwidth-mbps": config.network.bandwidth_mbps,
            "bad-bandwidth-factor": config.network.bad_bandwidth_factor,
            "mobility-type": config.network.mobility_type,
            "round-budget-s": config.network.round_budget_s,
        }
        
        # Execute Flower run
        start_time = time.time()
        try:
            result = subprocess.run(
                ["flwr", "run", ".", "--run-config", json.dumps(run_config)],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            duration = time.time() - start_time
            
            # Parse results
            exp_result = {
                "experiment_id": config.experiment_id,
                "config": config.to_dict(),
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
                "status": "success" if result.returncode == 0 else "failed",
            }
            
            # Load metrics if available
            metrics_path = Path("results") / config.experiment_id / "metrics.jsonl"
            if metrics_path.exists():
                exp_result["metrics_file"] = str(metrics_path)
                exp_result["has_metrics"] = True
            
            self.results.append(exp_result)
            return exp_result
            
        except subprocess.TimeoutExpired:
            return {
                "experiment_id": config.experiment_id,
                "status": "timeout",
                "duration": 3600,
            }
        except Exception as e:
            return {
                "experiment_id": config.experiment_id,
                "status": "error",
                "error": str(e),
            }
    
    def run_baseline_comparison(
        self,
        loss_rates: List[float] = [0.15, 0.25, 0.35],
        latencies: List[float] = [50, 100, 200],
        seeds: List[int] = [42, 43, 44, 45, 46],
    ):
        """
        Run full 3x3x5 experimental sweep.
        """
        print(f"Starting baseline comparison sweep...")
        print(f"Conditions: {len(loss_rates)} loss × {len(latencies)} latency × {len(seeds)} seeds = {len(loss_rates)*len(latencies)*len(seeds)} experiments")
        
        for seed in seeds:
            # Baseline (perfect network)
            baseline = BASELINE_CONFIG
            baseline.random_seed = seed
            baseline.network_seed = seed
            baseline.experiment_id = f"baseline_seed{seed}"
            self.run_single(baseline)
            
            # Network conditions
            for loss in loss_rates:
                for latency in latencies:
                    # Unreliable (no adaptation)
                    unreliable = UNRELIABLE_CONFIG
                    unreliable.random_seed = seed
                    unreliable.network_seed = seed
                    unreliable.network.loss_bad = loss
                    unreliable.network.base_latency_ms = latency
                    unreliable.experiment_id = f"unreliable_loss{loss}_lat{latency}_seed{seed}"
                    self.run_single(unreliable)
                    
                    # Adaptive
                    adaptive = ADAPTIVE_CONFIG
                    adaptive.random_seed = seed
                    adaptive.network_seed = seed
                    adaptive.network.loss_bad = loss
                    adaptive.network.base_latency_ms = latency
                    adaptive.experiment_id = f"adaptive_loss{loss}_lat{latency}_seed{seed}"
                    self.run_single(adaptive)
        
        # Save summary
        self._save_summary()
    
    def run_quick_test(self):
        """Run quick validation experiment."""
        print("Running quick validation test...")
        
        # Baseline
        baseline = BASELINE_CONFIG
        baseline.experiment_id = "test_baseline"
        baseline.fl.num_rounds = 5
        self.run_single(baseline)
        
        # Network
        network = UNRELIABLE_CONFIG
        network.experiment_id = "test_network"
        network.fl.num_rounds = 5
        self.run_single(network)
        
        print("Quick test complete. Check results/ directory.")
    
    def _save_summary(self):
        """Save experiment summary to CSV."""
        summary_path = self.output_dir / "experiment_summary.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FL experiments")
    parser.add_argument("--mode", choices=["test", "full", "single"], default="test")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--num-supernodes", type=int, default=30)
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        num_supernodes=args.num_supernodes,
    )
    
    if args.mode == "test":
        runner.run_quick_test()
    elif args.mode == "full":
        runner.run_baseline_comparison()
    else:
        # Single experiment
        config = BASELINE_CONFIG
        config.experiment_id = "single_run"
        runner.run_single(config)


if __name__ == "__main__":
    main()