from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from control_sims.optimizing_rpg.switching_template import SwitchingTemplateConfig  # noqa: E402
from control_sims.optimizing_rpg.switching_template_harness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    SwitchingTemplateHarnessConfig,
    run_switching_template_harness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark no-IPOPT RPG switching-template solver.")
    parser.add_argument("--scenario-table", type=Path, default=Path("scripts/generators/sim_instances/sobol_samples_512.csimin"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--portfolio-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="switching_template")
    parser.add_argument("--min-time", type=float, default=0.6)
    parser.add_argument("--max-time", type=float, default=1.5)
    parser.add_argument("--time-step", type=float, default=0.15)
    parser.add_argument("--replay-sample-dt", type=float, default=None)
    parser.add_argument("--replay-top-k", type=int, default=2)
    parser.add_argument("--screen-replay-margin", type=float, default=0.5)
    args = parser.parse_args()

    seeds = tuple(int(seed) for seed in str(args.seeds).split(",") if seed.strip())
    summary = run_switching_template_harness(
        SwitchingTemplateHarnessConfig(
            scenario_table=args.scenario_table,
            output_dir=args.out_dir,
            seeds=seeds,
            label=str(args.label),
            portfolio_csv=args.portfolio_csv,
            solver=SwitchingTemplateConfig(
                min_time_s=float(args.min_time),
                max_time_s=float(args.max_time),
                time_step_s=float(args.time_step),
                replay_sample_dt_s=None if args.replay_sample_dt is None else float(args.replay_sample_dt),
                replay_top_k=int(args.replay_top_k),
                screen_replay_margin_m=float(args.screen_replay_margin),
            ),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
