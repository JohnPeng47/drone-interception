from __future__ import annotations

import csv
from pathlib import Path

from backends.csim.generator.instance_store import read_sim_instances


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_TABLE = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin"
GOLDEN_ROOT = REPO_ROOT / "docs/analysis/control_sims/beihang_runner_regression"


def read_six_robust_intercept_samples():
    instances = read_sim_instances(SCENARIO_TABLE)
    assert [instance.seed for instance in instances] == [1, 2, 3, 4, 5, 6]
    return instances


def read_golden_trial_rows(sim_name: str) -> dict[int, dict[str, str]]:
    path = GOLDEN_ROOT / sim_name / "trials.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["seed"]): row for row in rows}
