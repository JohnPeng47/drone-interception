from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.baseline_harness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENARIO_TABLE,
    BaselineHarnessConfig,
    run_baseline_harness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the current RPG IPOPT portfolio planner.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--seeds", default="1", help="Comma-separated SimInstance seeds.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="ipopt_portfolio_baseline")
    args = parser.parse_args()

    seeds = tuple(int(seed) for seed in args.seeds.split(",") if seed.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    summary = run_baseline_harness(
        BaselineHarnessConfig(
            scenario_table=args.scenario_table,
            output_dir=args.out_dir,
            seeds=seeds,
            workers=int(args.workers),
            label=str(args.label),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not bool(summary["passed_acceptance"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
