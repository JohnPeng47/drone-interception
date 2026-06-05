from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backends.csim.generator.instance_store import read_sim_instances

from .switching_template import SwitchingTemplateConfig, find_switching_template_intercept


DEFAULT_SCENARIO_TABLE = Path("scripts/generators/sim_instances/sobol_samples_512.csimin")
DEFAULT_OUTPUT_DIR = Path("docs/analysis/control_sims/rpg_improve/switching_template_6_caught")
DEFAULT_SEEDS = (1, 2, 4, 6, 8, 10)


@dataclass(frozen=True)
class SwitchingTemplateHarnessConfig:
    scenario_table: Path = DEFAULT_SCENARIO_TABLE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    label: str = "switching_template"
    portfolio_csv: Path | None = None
    solver: SwitchingTemplateConfig = SwitchingTemplateConfig()


def run_switching_template_harness(config: SwitchingTemplateHarnessConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_rows = _portfolio_rows(config.portfolio_csv)
    instances = _load_seed_instances(Path(config.scenario_table), config.seeds)
    rows = []
    for instance in instances:
        result = find_switching_template_intercept(instance, config.solver)
        rows.append(_row(config, result, portfolio_rows.get(int(instance.seed))))
    _write_csv(output_dir / "switching_template_rows.csv", rows)
    summary = _summary(config, rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _row(config: SwitchingTemplateHarnessConfig, result, portfolio: dict[str, str] | None) -> dict[str, Any]:
    candidate = result.best_candidate
    portfolio_wall_s = _float_or_nan(portfolio.get("wall_s")) if portfolio else math.nan
    wall_delta_s = portfolio_wall_s - float(result.wall_s) if math.isfinite(portfolio_wall_s) else math.nan
    speedup = portfolio_wall_s / float(result.wall_s) if math.isfinite(portfolio_wall_s) and result.wall_s > 0.0 else math.nan
    return {
        "label": str(config.label),
        "seed": int(result.seed),
        "caught": bool(result.caught),
        "catch_time_s": float(result.catch_time_s),
        "fastest_caught_time_s": float(result.fastest_caught_time_s),
        "min_distance_m": float(result.min_distance_m),
        "final_distance_m": float(result.final_distance_m),
        "wall_s": float(result.wall_s),
        "templates_evaluated": int(result.templates_evaluated),
        "time_groups_evaluated": int(result.time_groups_evaluated),
        "distance_source": str(result.distance_source),
        "candidate_total_time_s": float(candidate.total_time_s) if candidate else math.nan,
        "candidate_thrust_fraction": float(candidate.thrust_fraction) if candidate else math.nan,
        "candidate_rate_fraction": float(candidate.rate_fraction) if candidate else math.nan,
        "candidate_first_switch_fraction": float(candidate.first_switch_fraction) if candidate else math.nan,
        "candidate_second_switch_fraction": float(candidate.second_switch_fraction) if candidate else math.nan,
        "candidate_counter_rate_fraction": float(candidate.counter_rate_fraction) if candidate else math.nan,
        "candidate_vertical_bias_gain": float(candidate.vertical_bias_gain) if candidate else math.nan,
        "candidate_direction_sign": float(candidate.direction_sign) if candidate else math.nan,
        "portfolio_wall_s": float(portfolio_wall_s),
        "portfolio_caught": str(portfolio.get("caught")) if portfolio else "",
        "portfolio_catch_time_s": _float_or_nan(portfolio.get("catch_time_s")) if portfolio else math.nan,
        "portfolio_min_distance_m": _float_or_nan(portfolio.get("min_distance_m")) if portfolio else math.nan,
        "portfolio_vs_switching_delta_s": float(wall_delta_s),
        "portfolio_vs_switching_speedup": float(speedup),
        "error": str(result.error),
    }


def _summary(config: SwitchingTemplateHarnessConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught = [bool(row["caught"]) for row in rows]
    valid = [row for row in rows if not row["error"]]
    portfolio_comparable = [row for row in rows if math.isfinite(float(row["portfolio_wall_s"]))]
    catch_equivalent = [
        row
        for row in portfolio_comparable
        if bool(row["caught"]) and str(row["portfolio_caught"]) == "True"
    ]
    switching_total = sum(float(row["wall_s"]) for row in rows)
    portfolio_total = sum(float(row["portfolio_wall_s"]) for row in portfolio_comparable)
    portfolio_covers_all_rows = len(portfolio_comparable) == len(rows)
    catch_equivalent_switching_total = sum(float(row["wall_s"]) for row in catch_equivalent)
    catch_equivalent_portfolio_total = sum(float(row["portfolio_wall_s"]) for row in catch_equivalent)
    return {
        "config": {
            "scenario_table": str(config.scenario_table),
            "output_dir": str(config.output_dir),
            "seeds": list(config.seeds),
            "label": str(config.label),
            "portfolio_csv": str(config.portfolio_csv) if config.portfolio_csv else None,
            "solver": asdict(config.solver),
        },
        "num_scenarios": int(len(rows)),
        "valid": int(len(valid)),
        "errors": int(len(rows) - len(valid)),
        "caught": int(sum(caught)),
        "catch_fraction": float(sum(caught) / len(caught)) if caught else math.nan,
        "switching_total_wall_s": float(switching_total),
        "switching_mean_wall_s": float(switching_total / len(rows)) if rows else math.nan,
        "portfolio_total_wall_s": float(portfolio_total) if portfolio_comparable else math.nan,
        "portfolio_mean_wall_s": float(portfolio_total / len(portfolio_comparable)) if portfolio_comparable else math.nan,
        "portfolio_comparable_rows": int(len(portfolio_comparable)),
        "portfolio_covers_all_rows": bool(portfolio_covers_all_rows),
        "portfolio_vs_switching_attempted_total_speedup": (
            float(portfolio_total / switching_total)
            if portfolio_covers_all_rows and switching_total > 0.0 else math.nan
        ),
        "catch_equivalent_num_scenarios": int(len(catch_equivalent)),
        "catch_equivalent_portfolio_total_wall_s": float(catch_equivalent_portfolio_total) if catch_equivalent else math.nan,
        "catch_equivalent_switching_total_wall_s": float(catch_equivalent_switching_total) if catch_equivalent else math.nan,
        "catch_equivalent_portfolio_vs_switching_speedup": (
            float(catch_equivalent_portfolio_total / catch_equivalent_switching_total)
            if catch_equivalent and catch_equivalent_switching_total > 0.0 else math.nan
        ),
        "artifacts": {
            "rows_csv": "switching_template_rows.csv",
            "summary_json": "summary.json",
        },
    }


def _load_seed_instances(path: Path, seeds: tuple[int, ...]):
    wanted = {int(seed) for seed in seeds}
    instances = [instance for instance in read_sim_instances(path) if int(instance.seed) in wanted]
    found = {int(instance.seed) for instance in instances}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"scenario table missing requested seeds: {missing}")
    return sorted(instances, key=lambda instance: int(instance.seed))


def _portfolio_rows(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def _float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
