from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config
from .trainer import train_grpo, train_sft, train_dpo
from .bootstrap import bootstrap


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="SD1.4 SFT/GRPO training entrypoint")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: key=value (dot path supported)",
    )
    parser.add_argument("--mode", choices=["grpo", "sft", "dpo"], help="Training mode override")
    parser.add_argument("--dataset", choices=["geneval", "ocr"], help="Dataset override")
    parser.add_argument("--reward", choices=["geneval", "ocr"], help="Reward override")
    parser.add_argument("--no-cfg", action="store_true", help="Disable classifier-free guidance")
    parser.add_argument("--grpo-guard", action="store_true", help="Enable GRPO-Guard ratio correction")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.mode:
        cfg.training.mode = args.mode
    if args.dataset:
        cfg.dataset.name = args.dataset
    if args.reward:
        cfg.reward.weights = {args.reward: 1.0}
    if args.no_cfg:
        cfg.grpo.no_cfg = True
    if args.grpo_guard:
        cfg.grpo.guard = True

    if args.dry_run:
        repo_paths = bootstrap()
        cfg.resolve(repo_paths)
        cfg.validate()
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    if cfg.training.mode == "grpo":
        train_grpo(cfg)
    elif cfg.training.mode == "dpo":
        train_dpo(cfg)
    else:
        train_sft(cfg)


if __name__ == "__main__":
    main()

