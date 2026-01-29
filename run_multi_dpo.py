"""Multi-Method DPO Comparison Launcher

This script enables parallel comparison of multiple DPO methods (diffusion-dpo, sdpo, kto, etc.)
by launching separate training processes, each assigned to specific GPUs with independent
checkpoint and evaluation directories.

Usage:
    python run_multi_dpo.py --config configs/multi_dpo_config.json
    python run_multi_dpo.py --config configs/multi_dpo_config.json --methods diffusion_dpo sdpo
    python run_multi_dpo.py --config configs/multi_dpo_config.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_multi_config(config_path: str) -> dict[str, Any]:
    """Load multi-method DPO configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_experiment_config(base_config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    """Merge base config with experiment-specific settings.
    
    Args:
        base_config: Base DPO configuration
        experiment: Experiment-specific overrides
        
    Returns:
        Merged configuration dictionary
    """
    # Deep copy base config
    merged = json.loads(json.dumps(base_config))
    
    # Apply experiment-specific overrides
    if 'output_dir' in experiment:
        merged.setdefault('run', {})['output_dir'] = experiment['output_dir']
    
    if 'resume_from' in experiment:
        merged.setdefault('run', {})['resume_from'] = experiment['resume_from']
    
    if 'eval_output_dir' in experiment:
        merged.setdefault('run', {})['eval_output_dir'] = experiment['eval_output_dir']
    
    if 'dpo_method' in experiment:
        merged.setdefault('dpo', {})['train_method'] = experiment['dpo_method']
    
    if 'beta_dpo' in experiment:
        merged.setdefault('dpo', {})['beta_dpo'] = experiment['beta_dpo']
    
    # KTO-specific parameters
    if 'kto_lambda_d' in experiment:
        merged.setdefault('dpo', {})['kto_lambda_d'] = experiment['kto_lambda_d']
    
    if 'kto_lambda_u' in experiment:
        merged.setdefault('dpo', {})['kto_lambda_u'] = experiment['kto_lambda_u']
    
    # SDPO-specific parameters
    if 'sdpo_mu' in experiment:
        merged.setdefault('dpo', {})['sdpo_mu'] = experiment['sdpo_mu']
    
    if 'sdpo_alpha' in experiment:
        merged.setdefault('dpo', {})['sdpo_alpha'] = experiment['sdpo_alpha']
    
    # Wandb naming
    merged.setdefault('logging', {})['wandb_run_name'] = experiment.get('name', 'dpo_experiment')
    
    return merged


def create_temp_config(experiment_name: str, config: dict[str, Any], output_dir: Path) -> Path:
    """Create temporary config file for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        output_dir: Directory to save temp config
        
    Returns:
        Path to the created config file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{experiment_name}.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def launch_experiment(
    experiment_name: str,
    gpus: str,
    config_path: Path,
    global_overrides: list[str],
    dry_run: bool = False
) -> subprocess.Popen | None:
    """Launch a single DPO training experiment.
    
    Args:
        experiment_name: Name of the experiment
        gpus: Comma-separated GPU IDs
        config_path: Path to experiment config
        global_overrides: Additional config overrides from command line
        dry_run: If True, only print command without executing
        
    Returns:
        Process object if launched, None if dry_run
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    
    cmd = [
        sys.executable,
        'run.py',
        '--config', str(config_path),
        '--mode', 'dpo'
    ]
    
    # Add global overrides
    for override in global_overrides:
        cmd.extend(['--set', override])
    
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"GPUs: {gpus} (CUDA_VISIBLE_DEVICES={gpus})")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        return None
    
    # Launch process
    log_dir = Path(config_path).parent
    log_file = log_dir / f"launch_{experiment_name}.log"
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent
        )
    
    print(f"✓ Launched {experiment_name} (PID: {process.pid}, log: {log_file})")
    return process


def monitor_processes(processes: dict[str, subprocess.Popen]) -> None:
    """Monitor running processes and report status.
    
    Args:
        processes: Dictionary mapping experiment names to process objects
    """
    print(f"\n{'='*80}")
    print(f"Monitoring {len(processes)} experiments...")
    print(f"Press Ctrl+C to stop all processes")
    print(f"{'='*80}\n")
    
    try:
        while processes:
            time.sleep(5)  # Check every 5 seconds
            
            completed = []
            for name, proc in processes.items():
                retcode = proc.poll()
                if retcode is not None:
                    completed.append(name)
                    if retcode == 0:
                        print(f"✓ {name} completed successfully")
                    else:
                        print(f"✗ {name} failed with exit code {retcode}")
            
            # Remove completed processes
            for name in completed:
                del processes[name]
            
            if processes:
                print(f"Still running: {', '.join(processes.keys())}")
        
        print("\n✓ All experiments completed!")
        
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal. Terminating all processes...")
        for name, proc in processes.items():
            print(f"  Terminating {name} (PID: {proc.pid})")
            proc.terminate()
        
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if still running
        for name, proc in processes.items():
            if proc.poll() is None:
                print(f"  Force killing {name}")
                proc.kill()
        
        print("✓ All processes terminated")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch multiple DPO experiments for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to multi-method DPO config JSON file'
    )
    
    parser.add_argument(
        '--methods',
        nargs='*',
        help='Specific experiment names to run (default: all in config)'
    )
    
    parser.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        help='Global config overrides applied to all experiments (e.g., run.num_epochs=100)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print experiment configs and commands without launching'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run experiments sequentially instead of in parallel'
    )
    
    args = parser.parse_args()
    
    # Load multi-method config
    multi_config = load_multi_config(args.config)
    
    # Load base config
    base_config_path = multi_config.get('base_config', 'configs/dpo_pickapic.json')
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    # Filter experiments if specific methods requested
    experiments = multi_config['experiments']
    if args.methods:
        experiments = [exp for exp in experiments if exp['name'] in args.methods]
        if not experiments:
            print(f"Error: No experiments found with names: {args.methods}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Multi-Method DPO Comparison")
    print(f"Base Config: {base_config_path}")
    print(f"Experiments: {len(experiments)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTION'}")
    print(f"{'='*80}\n")
    
    # Create temp directory for experiment configs
    temp_dir = Path('.multi_dpo_temp')
    processes = {}
    
    for exp in experiments:
        exp_name = exp['name']
        exp_gpus = exp.get('gpus', '0')
        
        # Merge configs
        merged_config = merge_experiment_config(base_config, exp)
        
        # Create temp config file
        config_path = create_temp_config(exp_name, merged_config, temp_dir)
        
        # Launch experiment
        proc = launch_experiment(
            exp_name,
            exp_gpus,
            config_path,
            args.overrides,
            dry_run=args.dry_run
        )
        
        if proc:
            processes[exp_name] = proc
            
            # If sequential mode, wait for completion
            if args.sequential:
                print(f"Waiting for {exp_name} to complete (sequential mode)...")
                retcode = proc.wait()
                if retcode != 0:
                    print(f"✗ {exp_name} failed with exit code {retcode}")
                    print("Stopping sequential execution due to failure")
                    sys.exit(retcode)
                print(f"✓ {exp_name} completed successfully\n")
    
    # Monitor all processes (if parallel mode and not dry run)
    if processes and not args.sequential:
        monitor_processes(processes)
    elif args.dry_run:
        print("\n✓ Dry run completed. Use without --dry-run to launch experiments.")


if __name__ == '__main__':
    main()
