"""
Statistical analysis and visualization of experiment results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ExperimentAnalyzer:
    """
    Analyze and visualize comparative FL experiments.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.experiments: Dict[str, pd.DataFrame] = {}
        
    def load_experiment(self, experiment_id: str) -> pd.DataFrame:
        """Load metrics from experiment."""
        metrics_file = self.results_dir / experiment_id / "metrics.jsonl"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"No metrics found for {experiment_id}")
        
        records = []
        with open(metrics_file) as f:
            for line in f:
                records.append(json.loads(line))
        
        df = pd.DataFrame(records)
        self.experiments[experiment_id] = df
        return df
    
    def load_all(self, pattern: str = "*"):
        """Load all experiments matching pattern."""
        for exp_dir in self.results_dir.glob(pattern):
            if exp_dir.is_dir():
                try:
                    self.load_experiment(exp_dir.name)
                    print(f"Loaded: {exp_dir.name}")
                except Exception as e:
                    print(f"Failed to load {exp_dir.name}: {e}")
    
    def compare_convergence(
        self,
        experiment_ids: List[str],
        metric: str = "train_accuracy",
        save_path: Optional[str] = None,
    ):
        """
        Plot convergence curves for multiple experiments.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                self.load_experiment(exp_id)
            
            df = self.experiments[exp_id]
            ax.plot(df["round"], df[metric], label=exp_id, marker='o', markersize=3)
        
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Convergence Comparison: {metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_frontier(
        self,
        metric_x: str = "total_bytes",
        metric_y: str = "train_accuracy",
        save_path: Optional[str] = None,
    ):
        """
        Plot Pareto frontier for communication vs. accuracy tradeoff.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Aggregate final metrics across experiments
        summary_data = []
        for exp_id, df in self.experiments.items():
            final = df.iloc[-1]
            summary_data.append({
                "experiment_id": exp_id,
                "final_accuracy": final.get("train_accuracy", 0),
                "total_bytes": df["total_bytes"].sum(),
                "avg_participation": df["participation_rate"].mean(),
                "config": self._extract_config_name(exp_id),
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Color by configuration type
        colors = {"baseline": "green", "unreliable": "red", "adaptive": "blue"}
        
        for config_type in summary_df["config"].unique():
            subset = summary_df[summary_df["config"] == config_type]
            ax.scatter(
                subset["total_bytes"],
                subset["final_accuracy"],
                label=config_type,
                c=colors.get(config_type, "gray"),
                s=100,
                alpha=0.6,
            )
        
        ax.set_xlabel("Total Bytes Transmitted")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("Communication-Accuracy Tradeoff")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return summary_df
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary statistics table for all experiments.
        """
        rows = []
        for exp_id, df in self.experiments.items():
            row = {
                "experiment_id": exp_id,
                "final_accuracy": df["train_accuracy"].iloc[-1],
                "final_loss": df["train_loss"].iloc[-1],
                "avg_participation": df["participation_rate"].mean(),
                "avg_dropout": df["dropout_rate"].mean(),
                "total_bytes": df["total_bytes"].sum(),
                "avg_latency_ms": df.get("avg_latency_ms", pd.Series([0])).mean(),
                "rounds_to_target": self._rounds_to_target(df, 0.8),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def statistical_comparison(
        self,
        baseline_id: str,
        treatment_ids: List[str],
        metric: str = "final_accuracy",
    ) -> pd.DataFrame:
        """
        Perform statistical comparison between baseline and treatments.
        """
        from scipy import stats
        
        baseline_df = self.experiments[baseline_id]
        baseline_values = baseline_df[metric].values if metric in baseline_df.columns else [baseline_df.iloc[-1]["train_accuracy"]]
        
        results = []
        for tid in treatment_ids:
            if tid not in self.experiments:
                continue
            
            treat_df = self.experiments[tid]
            treat_values = treat_df[metric].values if metric in treat_df.columns else [treat_df.iloc[-1]["train_accuracy"]]
            
            # Paired t-test if same length, otherwise independent
            if len(baseline_values) == len(treat_values):
                t_stat, p_value = stats.ttest_rel(baseline_values, treat_values)
            else:
                t_stat, p_value = stats.ttest_ind(baseline_values, treat_values)
            
            results.append({
                "comparison": f"{baseline_id} vs {tid}",
                "baseline_mean": np.mean(baseline_values),
                "treatment_mean": np.mean(treat_values),
                "difference": np.mean(treat_values) - np.mean(baseline_values),
                "percent_change": (np.mean(treat_values) - np.mean(baseline_values)) / np.mean(baseline_values) * 100,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            })
        
        return pd.DataFrame(results)
    
    def _extract_config_name(self, experiment_id: str) -> str:
        """Extract configuration type from experiment ID."""
        if "baseline" in experiment_id:
            return "baseline"
        elif "adaptive" in experiment_id:
            return "adaptive"
        elif "unreliable" in experiment_id:
            return "unreliable"
        return "unknown"
    
    def _rounds_to_target(self, df: pd.DataFrame, target: float) -> int:
        """Find first round where target accuracy was reached."""
        reached = df[df["train_accuracy"] >= target]
        return reached["round"].iloc[0] if not reached.empty else len(df)


def main():
    """CLI for analysis."""
    analyzer = ExperimentAnalyzer()
    analyzer.load_all()
    
    # Generate plots
    analyzer.compare_convergence(
        ["baseline_seed42", "unreliable_loss0.25_lat50_seed42", "adaptive_loss0.25_lat50_seed42"],
        save_path="convergence_comparison.png"
    )
    
    analyzer.plot_pareto_frontier(save_path="pareto_frontier.png")
    
    # Print summary
    summary = analyzer.generate_summary_table()
    print("\nExperiment Summary:")
    print(summary.to_string())
    
    # Statistical test
    if len(analyzer.experiments) > 1:
        baseline = [k for k in analyzer.experiments.keys() if "baseline" in k][0]
        others = [k for k in analyzer.experiments.keys() if k != baseline]
        comparison = analyzer.statistical_comparison(baseline, others)
        print("\nStatistical Comparison:")
        print(comparison.to_string())


if __name__ == "__main__":
    main()