from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.baseline_harness import DEFAULT_SCENARIO_TABLE  # noqa: E402
from control_sims.optimizing_rpg.rollout_harness import DEFAULT_BASELINE_SUMMARY  # noqa: E402
from control_sims.optimizing_rpg.structured_update_harness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    StructuredUpdateHarnessConfig,
    run_structured_update_harness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark structured RPG trajectory updates.")
    parser.add_argument("--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--active-window-nodes", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="structured_trajectory_update")
    args = parser.parse_args()

    summary = run_structured_update_harness(
        StructuredUpdateHarnessConfig(
            scenario_table=args.scenario_table,
            baseline_summary=args.baseline_summary,
            output_dir=args.out_dir,
            seed=int(args.seed),
            active_window_nodes=int(args.active_window_nodes),
            label=str(args.label),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary["passed_acceptance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
