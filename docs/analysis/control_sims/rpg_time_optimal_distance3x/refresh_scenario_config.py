from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.generator import get_config
from backends.csim.generator.instance_store import read_sim_instances, write_sim_instances


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh generated scenario configs from backends/csim/configs/base.py."
    )
    parser.add_argument("scenario_table", type=Path, nargs="+")
    args = parser.parse_args()

    base_config = get_config("base")
    for path in args.scenario_table:
        instances = read_sim_instances(path)
        refreshed = [replace(instance, config=base_config) for instance in instances]
        write_sim_instances(path, refreshed)
        print(f"refreshed {len(refreshed)} instances: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
