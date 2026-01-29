from __future__ import annotations

import sys
from pathlib import Path

from .paths import resolve_repo_paths


def bootstrap() -> dict[str, Path]:
    repo_paths = resolve_repo_paths(Path(__file__).resolve())
    # Ensure local sources are importable before any third-party resolution.
    for key in ("flow_grpo",):
        path = str(repo_paths[key])
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_paths

