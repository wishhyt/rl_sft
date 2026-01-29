from __future__ import annotations

import sys
import os
import argparse
import subprocess
from pathlib import Path


def main() -> None:
    # 1. Check if we are already running in a distributed worker process
    #    (accelerate sets LOCAL_RANK, RANK, etc.)
    if "LOCAL_RANK" in os.environ:
        _run_training_entrypoint()
        return

    # 2. Parse launcher arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU IDs to use (e.g. '0,1')")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs to use")
    
    # Use parse_known_args so unknown args (like --config, --mode) are passed through
    launcher_args, script_args = parser.parse_known_args()

    # 3. If no multi-GPU intent is detected, assume single-GPU / standard run
    if notLauncherArgsArePresent(launcher_args):
        # Pass full original args to the training entrypoint (argparse there will handle them)
        _run_training_entrypoint()
        return

    # 4. Construct 'accelerate launch' command
    cmd = ["accelerate", "launch"]
    
    # Determine number of processes and visible devices
    num_processes = 1
    env = os.environ.copy()

    if launcher_args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = launcher_args.gpus
        gpu_ids = launcher_args.gpus.split(",")
        num_processes = len(gpu_ids)
        # If user also specified num_gpus, ensure it matches
        if launcher_args.num_gpus and launcher_args.num_gpus != num_processes:
             print(f"Warning: --num-gpus ({launcher_args.num_gpus}) ignored in favor of --gpus count ({num_processes})")
    elif launcher_args.num_gpus:
        num_processes = launcher_args.num_gpus
        
    cmd.extend(["--num_processes", str(num_processes)])
    
    # Append the script itself
    cmd.append(str(Path(__file__).resolve()))
    
    # Append the rest of the arguments (forwarded to the script)
    cmd.extend(script_args)
    
    # 5. Execute
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def _run_training_entrypoint() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from rl_sft.train import main as _main
    _main()


def notLauncherArgsArePresent(args):
    return not (args.gpus or args.num_gpus)


if __name__ == "__main__":
    main()

