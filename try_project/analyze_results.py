"""
Pareto Analysis and Visualization for Privacy-Communication-Utility Tradeoffs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_experiment_results(result_dir: str) -> Dict:
    result_path = Path(result_dir)
    data = {
        'rounds': [], 'accuracy': [], 'bytes_sent': [],
        'epsilon_spent': [], 'latency_p95': [], 'drop_rate': []
    }
    
    for metrics_file in sorted(result_path.glob("metrics_round_*.json")):
        with open(metrics_file) as f:
            metrics = json.load(f)
            data['rounds'].append(metrics.get('round', 0))
            data['accuracy'].append(metrics.get('accuracy', 0))
            data['bytes_sent'].append(metrics.get('bytes_sent_mb', 0))
            data['epsilon_spent'].append(metrics.get('epsilon_spent', 0))
            data['latency_p95'].append(metrics.get('latency_p95_ms', 0))
            data['drop_rate'].append(metrics.get('drop_rate', 0))
    
    return data


def plot_pareto_frontier(results_list: List[Dict], labels: List[str], output_path: str = "pareto_frontier.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, (data, label) in enumerate(zip(results_list, labels)):
        final_acc = data['accuracy'][-1] if data['accuracy'] else 0
        total_bytes = sum(data['bytes_sent'])
        total_eps = data['epsilon_spent'][-1] if data['epsilon_spent'] else 0
        avg_latency = np.mean(data['latency_p95']) if data['latency_p95'] else 0
        
        axes[0].scatter(total_eps, total_bytes, s=200, c=[colors[i]], label=label, alpha=0.7, edgecolors='black')
        axes[0].annotate(f"{final_acc:.2f}", (total_eps, total_bytes), fontsize=9, ha='center')
        
        axes[1].scatter(total_eps, avg_latency, s=200, c=[colors[i]], label=label, alpha=0.7, edgecolors='black')
        axes[2].scatter(total_bytes, final_acc, s=200, c=[colors[i]], label=label, alpha=0.7, edgecolors='black')
    
    axes[0].set_xlabel("Privacy Cost (ε, lower=better)")
    axes[0].set_ylabel("Communication Cost (MB)")
    axes[0].set_title("Privacy-Communication Tradeoff")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Privacy Cost (ε)")
    axes[1].set_ylabel("95th Percentile Latency (ms)")
    axes[1].set_title("Privacy-Latency Tradeoff")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel("Communication Cost (MB)")
    axes[2].set_ylabel("Final Accuracy")
    axes[2].set_title("Communication-Utility Tradeoff")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto frontier to {output_path}")


def generate_summary_table(results_list: List[Dict], labels: List[str]) -> pd.DataFrame:
    rows = []
    for data, label in zip(results_list, labels):
        row = {
            'Configuration': label,
            'Final Accuracy': data['accuracy'][-1] if data['accuracy'] else 0,
            'Total MB': sum(data['bytes_sent']),
            'Total ε': data['epsilon_spent'][-1] if data['epsilon_spent'] else 0,
            'Avg P95 Latency (ms)': np.mean(data['latency_p95']) if data['latency_p95'] else 0,
            'Avg Drop Rate': np.mean(data['drop_rate']) if data['drop_rate'] else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze FL experiment results")
    parser.add_argument("--results-dir", nargs='+', required=True)
    parser.add_argument("--labels", nargs='+', required=True)
    parser.add_argument("--output-dir", default="analysis_output")
    
    args = parser.parse_args()
    assert len(args.results_dir) == len(args.labels)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = [load_experiment_results(d) for d in args.results_dir]
    
    plot_pareto_frontier(all_results, args.labels, output_dir / "pareto_frontier.png")
    
    summary_df = generate_summary_table(all_results, args.labels)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()