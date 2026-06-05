from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.baseline_harness import DEFAULT_SCENARIO_TABLE  # noqa: E402
from control_sims.optimizing_rpg.fixed_time_harness import DEFAULT_CATCH_TABLE  # noqa: E402
from control_sims.optimizing_rpg.final_harness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    FinalHarnessConfig,
    run_final_harness,
)
from control_sims.optimizing_rpg.rollout_harness import DEFAULT_BASELINE_SUMMARY  # noqa: E402
from control_sims.optimizing_rpg.time_search_harness import DEFAULT_TIME_MULTIPLIERS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final RPG optimization parallelization harness.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--multi-table", type=Path, default=DEFAULT_CATCH_TABLE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--multi-seeds", default="1,2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-multipliers", default=",".join(str(value) for value in DEFAULT_TIME_MULTIPLIERS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="performance_hardening")
    args = parser.parse_args()

    multi_seeds = tuple(int(seed) for seed in str(args.multi_seeds).split(",") if seed.strip())
    if len(set(multi_seeds)) != len(multi_seeds) or len(multi_seeds) <= 1:
        raise ValueError("--multi-seeds must contain more than one distinct seed")
    if int(args.workers) < 2:
        raise ValueError("--workers must be at least 2")
    time_multipliers = tuple(float(value) for value in str(args.time_multipliers).split(",") if value.strip())
    summary = run_final_harness(
        FinalHarnessConfig(
            scenario_table=args.scenario_table,
            multi_table=args.multi_table,
            baseline_summary=args.baseline_summary,
            output_dir=args.out_dir,
            seed=int(args.seed),
            multi_seeds=multi_seeds,
            workers=int(args.workers),
            label=str(args.label),
            time_multipliers=time_multipliers,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary["passed_acceptance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
