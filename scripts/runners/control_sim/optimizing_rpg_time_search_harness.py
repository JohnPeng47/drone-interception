from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.baseline_harness import DEFAULT_SCENARIO_TABLE  # noqa: E402
from control_sims.optimizing_rpg.fixed_time_harness import DEFAULT_CATCH_TABLE  # noqa: E402
from control_sims.optimizing_rpg.rollout_harness import DEFAULT_BASELINE_SUMMARY  # noqa: E402
from control_sims.optimizing_rpg.time_search_harness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIME_MULTIPLIERS,
    TimeSearchHarnessConfig,
    run_time_search_harness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RPG parallel fixed-time probe search.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--catch-table", type=Path, default=DEFAULT_CATCH_TABLE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--catch-seeds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-multipliers", default=",".join(str(value) for value in DEFAULT_TIME_MULTIPLIERS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="parallel_time_search")
    args = parser.parse_args()

    catch_seeds = tuple(int(seed) for seed in str(args.catch_seeds).split(",") if seed.strip())
    if not catch_seeds:
        raise ValueError("--catch-seeds must contain at least one integer seed")
    time_multipliers = tuple(float(value) for value in str(args.time_multipliers).split(",") if value.strip())
    summary = run_time_search_harness(
        TimeSearchHarnessConfig(
            scenario_table=args.scenario_table,
            catch_table=args.catch_table,
            baseline_summary=args.baseline_summary,
            output_dir=args.out_dir,
            seed=int(args.seed),
            catch_seeds=catch_seeds,
            workers=int(args.workers),
            label=str(args.label),
            time_multipliers=time_multipliers,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary["passed_acceptance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
