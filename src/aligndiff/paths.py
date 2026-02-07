from __future__ import annotations

from pathlib import Path


REQUIRED_REPOS = ("flow_grpo",)


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if all((candidate / name).exists() for name in REQUIRED_REPOS):
            return candidate
    raise RuntimeError(
        "Could not locate repo root containing DanceGRPO and flow_grpo."
    )


def resolve_repo_paths(anchor: Path) -> dict[str, Path]:
    root = find_repo_root(anchor)
    return {
        "root": root,
        # "dancegrpo": root / "DanceGRPO",
        "flow_grpo": root / "flow_grpo",
        "datasets": root / "flow_grpo" / "dataset",
    }


def default_dataset_root(repo_paths: dict[str, Path], dataset_name: str) -> Path:
    return repo_paths["datasets"] / dataset_name

