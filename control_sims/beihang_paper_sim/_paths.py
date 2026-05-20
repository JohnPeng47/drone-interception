"""Import-path bootstrap for running Beihang scripts from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    workspace_root = repo_root.parent
    simulations_root = workspace_root.parent
    drake_sims_root = workspace_root / "drake_sims"

    for path in (
        repo_root,
        repo_root / "shared",
        drake_sims_root / "src",
        workspace_root / "intercept_sim" / "src",
        simulations_root / "rotorpy",
        drake_sims_root / "sims",
    ):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
