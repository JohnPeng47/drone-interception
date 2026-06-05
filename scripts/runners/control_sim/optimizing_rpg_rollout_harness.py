from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.rollout_harness import (  # noqa: E402
    DEFAULT_BASELINE_SUMMARY,
    DEFAULT_OUTPUT_DIR,
    RolloutHarnessConfig,
    run_rollout_harness,
)
from control_sims.optimizing_rpg.baseline_harness import DEFAULT_SCENARIO_TABLE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the numeric RPG rollout core against the IPOPT baseline.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="numeric_rollout_core")
    args = parser.parse_args()

    summary = run_rollout_harness(
        RolloutHarnessConfig(
            scenario_table=args.scenario_table,
            baseline_summary=args.baseline_summary,
            output_dir=args.out_dir,
            seed=int(args.seed),
            repeats=int(args.repeats),
            label=str(args.label),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not bool(summary["passed_acceptance"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
