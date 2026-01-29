"""DPO Results Comparison Tool

Compare training metrics across multiple DPO experiments and generate
comparison reports with tables and plots.

Usage:
    python compare_dpo_results.py --experiments logs/multi_dpo/diffusion_dpo logs/multi_dpo/sdpo logs/multi_dpo/kto
    python compare_dpo_results.py --experiments logs/multi_dpo/* --output comparison_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys


def load_metrics(experiment_dir: Path) -> list[dict[str, Any]]:
    """Load metrics.jsonl from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment output directory
        
    Returns:
        List of metric dictionaries
    """
    metrics_file = experiment_dir / 'metrics.jsonl'
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return []
    
    metrics = []
    with open(metrics_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    
    return metrics


def extract_key_metrics(metrics: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Extract key training metrics from metrics log.
    
    Args:
        metrics: List of metric dictionaries
        
    Returns:
        Dictionary mapping metric names to value lists
    """
    key_metrics = {
        'epoch': [],
        'loss': [],
        'implicit_acc': [],
        'learning_rate': [],
        'grad_norm': []
    }
    
    for entry in metrics:
        # Epoch is always present
        if 'epoch' in entry:
            key_metrics['epoch'].append(entry['epoch'])
        
        # Loss
        if 'loss' in entry:
            key_metrics['loss'].append(entry['loss'])
        elif 'train/loss' in entry:
            key_metrics['loss'].append(entry['train/loss'])
        
        # Implicit accuracy
        if 'implicit_acc' in entry:
            key_metrics['implicit_acc'].append(entry['implicit_acc'])
        elif 'train/implicit_acc' in entry:
            key_metrics['implicit_acc'].append(entry['train/implicit_acc'])
        
        # Learning rate
        if 'learning_rate' in entry:
            key_metrics['learning_rate'].append(entry['learning_rate'])
        elif 'train/learning_rate' in entry:
            key_metrics['learning_rate'].append(entry['train/learning_rate'])
        
        # Gradient norm
        if 'grad_norm' in entry:
            key_metrics['grad_norm'].append(entry['grad_norm'])
        elif 'train/grad_norm' in entry:
            key_metrics['grad_norm'].append(entry['train/grad_norm'])
    
    # Remove empty metrics
    return {k: v for k, v in key_metrics.items() if v}


def get_summary_stats(values: list[float]) -> dict[str, float]:
    """Calculate summary statistics for a metric.
    
    Args:
        values: List of metric values
        
    Returns:
        Dictionary with min, max, mean, final values
    """
    if not values:
        return {'min': 0, 'max': 0, 'mean': 0, 'final': 0}
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'final': values[-1]
    }


def generate_comparison_table(experiments: dict[str, dict[str, list[float]]]) -> str:
    """Generate markdown comparison table.
    
    Args:
        experiments: Dictionary mapping experiment names to their metrics
        
    Returns:
        Markdown table string
    """
    table = ["# DPO Methods Comparison Report\n"]
    table.append("## Summary Statistics\n")
    
    # Loss comparison
    table.append("### Loss\n")
    table.append("| Method | Min | Max | Mean | Final |")
    table.append("|--------|-----|-----|------|-------|")
    
    for exp_name, metrics in experiments.items():
        if 'loss' in metrics:
            stats = get_summary_stats(metrics['loss'])
            table.append(
                f"| {exp_name} | {stats['min']:.4f} | {stats['max']:.4f} | "
                f"{stats['mean']:.4f} | {stats['final']:.4f} |"
            )
    
    table.append("\n")
    
    # Implicit Accuracy comparison
    table.append("### Implicit Accuracy\n")
    table.append("| Method | Min | Max | Mean | Final |")
    table.append("|--------|-----|-----|------|-------|")
    
    for exp_name, metrics in experiments.items():
        if 'implicit_acc' in metrics:
            stats = get_summary_stats(metrics['implicit_acc'])
            table.append(
                f"| {exp_name} | {stats['min']:.4f} | {stats['max']:.4f} | "
                f"{stats['mean']:.4f} | {stats['final']:.4f} |"
            )
    
    table.append("\n")
    
    # Training Progress
    table.append("### Training Progress\n")
    table.append("| Method | Total Epochs | Final LR | Final Grad Norm |")
    table.append("|--------|--------------|----------|-----------------|")
    
    for exp_name, metrics in experiments.items():
        total_epochs = len(metrics.get('epoch', []))
        final_lr = metrics.get('learning_rate', [0])[-1] if metrics.get('learning_rate') else 0
        final_grad = metrics.get('grad_norm', [0])[-1] if metrics.get('grad_norm') else 0
        
        table.append(
            f"| {exp_name} | {total_epochs} | {final_lr:.2e} | {final_grad:.4f} |"
        )
    
    table.append("\n")
    
    return "\n".join(table)


def generate_text_plots(experiments: dict[str, dict[str, list[float]]]) -> str:
    """Generate simple text-based plots showing trends.
    
    Args:
        experiments: Dictionary mapping experiment names to their metrics
        
    Returns:
        Text plot string
    """
    plots = ["## Metric Trends\n"]
    
    # Show last 10 epochs for each metric
    for metric_name in ['loss', 'implicit_acc']:
        plots.append(f"### {metric_name.replace('_', ' ').title()} (Last 10 Epochs)\n")
        plots.append("```")
        
        for exp_name, metrics in experiments.items():
            if metric_name in metrics:
                values = metrics[metric_name][-10:]
                epochs = metrics['epoch'][-10:] if 'epoch' in metrics else list(range(len(values)))
                
                plots.append(f"{exp_name}:")
                for epoch, value in zip(epochs, values):
                    plots.append(f"  Epoch {epoch:4d}: {value:.4f}")
                plots.append("")
        
        plots.append("```\n")
    
    return "\n".join(plots)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DPO training results across multiple experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        required=True,
        help='Paths to experiment output directories'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output markdown file path (default: print to stdout)'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Include text-based metric plots in report'
    )
    
    args = parser.parse_args()
    
    # Load all experiments
    experiments = {}
    for exp_path in args.experiments:
        exp_dir = Path(exp_path)
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} does not exist, skipping")
            continue
        
        exp_name = exp_dir.name
        metrics = load_metrics(exp_dir)
        
        if metrics:
            experiments[exp_name] = extract_key_metrics(metrics)
            print(f"✓ Loaded {len(metrics)} records from {exp_name}")
        else:
            print(f"✗ No metrics found for {exp_name}")
    
    if not experiments:
        print("Error: No valid experiments found")
        sys.exit(1)
    
    print(f"\nComparing {len(experiments)} experiments\n")
    
    # Generate report
    report = []
    report.append(generate_comparison_table(experiments))
    
    if args.show_plots:
        report.append(generate_text_plots(experiments))
    
    # Add experiment details
    report.append("## Experiment Details\n")
    for exp_name, metrics in experiments.items():
        report.append(f"### {exp_name}\n")
        report.append(f"- Total epochs: {len(metrics.get('epoch', []))}")
        report.append(f"- Metrics tracked: {', '.join(metrics.keys())}")
        report.append("")
    
    report_content = "\n".join(report)
    
    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding='utf-8')
        print(f"✓ Report saved to {output_path}")
    else:
        print(report_content)


if __name__ == '__main__':
    main()
