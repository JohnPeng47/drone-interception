from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.fixed_time_harness import (  # noqa: E402
    DEFAULT_BASELINE_SUMMARY,
    DEFAULT_CATCH_TABLE,
    DEFAULT_OUTPUT_DIR,
    FixedTimeHarnessConfig,
    run_fixed_time_harness,
)
from control_sims.optimizing_rpg.baseline_harness import DEFAULT_SCENARIO_TABLE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fixed-time RPG feasibility evaluation.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--catch-table", type=Path, default=DEFAULT_CATCH_TABLE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--catch-seeds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="fixed_time_feasibility")
    args = parser.parse_args()

    catch_seeds = tuple(int(seed) for seed in str(args.catch_seeds).split(",") if seed.strip())
    if not catch_seeds:
        raise ValueError("--catch-seeds must contain at least one integer seed")
    summary = run_fixed_time_harness(
        FixedTimeHarnessConfig(
            scenario_table=args.scenario_table,
            catch_table=args.catch_table,
            baseline_summary=args.baseline_summary,
            output_dir=args.out_dir,
            seed=int(args.seed),
            catch_seeds=catch_seeds,
            label=str(args.label),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not bool(summary["passed_acceptance"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
