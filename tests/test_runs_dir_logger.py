from __future__ import annotations

import csv
import datetime as dt
import json

import pytest

from control_sims.sim_runner import BeihangMinimalControlSimRunner, BeihangPaperControlSimRunner
from utils.logging import RunsDirLogger


def _fixed_clock() -> dt.datetime:
    return dt.datetime(2026, 5, 28, 12, 0, 0)


def test_runs_dir_logger_creates_date_partitioned_prefix_suffix_dir(tmp_path):
    logger = RunsDirLogger("beihang_minimal", root=tmp_path / ".runs", clock=_fixed_clock)

    run_dir = logger.create_run_dir("smoke")

    assert run_dir == tmp_path / ".runs" / "2026-05-28" / "beihang_minimal_smoke"
    assert run_dir.is_dir()


def test_runs_dir_logger_uses_prefix_without_suffix(tmp_path):
    logger = RunsDirLogger("beihang_paper", root=tmp_path / ".runs", clock=_fixed_clock)

    run_dir = logger.create_run_dir()

    assert run_dir == tmp_path / ".runs" / "2026-05-28" / "beihang_paper"


def test_runs_dir_logger_rejects_duplicate_run_dir(tmp_path):
    logger = RunsDirLogger("beihang_minimal", root=tmp_path / ".runs", clock=_fixed_clock)
    logger.create_run_dir("smoke")

    with pytest.raises(FileExistsError):
        logger.create_run_dir("smoke")


def test_runs_dir_logger_writes_json_and_csv_relative_to_run_dir(tmp_path):
    logger = RunsDirLogger("unit", root=tmp_path / ".runs", clock=_fixed_clock)
    run_dir = logger.create_run_dir("artifacts")

    json_path = logger.write_json(run_dir, "summary.json", {"b": 2, "a": 1})
    csv_path = logger.write_csv(
        run_dir,
        "tables/trials.csv",
        [{"seed": 1, "caught": True}],
        ["seed", "caught"],
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"seed": "1", "caught": "True"}]


def test_runs_dir_logger_rejects_nested_names_and_escaping_paths(tmp_path):
    with pytest.raises(ValueError):
        RunsDirLogger("nested/name", root=tmp_path / ".runs", clock=_fixed_clock)

    logger = RunsDirLogger("unit", root=tmp_path / ".runs", clock=_fixed_clock)
    run_dir = logger.create_run_dir()
    with pytest.raises(ValueError):
        logger.write_json(run_dir, "../summary.json", {})


def test_control_sim_runners_use_distinct_runs_dir_prefixes(tmp_path):
    minimal = BeihangMinimalControlSimRunner(
        RunsDirLogger("beihang_minimal", root=tmp_path / ".runs", clock=_fixed_clock)
    )
    paper = BeihangPaperControlSimRunner(
        RunsDirLogger("beihang_paper", root=tmp_path / ".runs", clock=_fixed_clock)
    )

    assert minimal.create_run_dir(suffix="smoke").name == "beihang_minimal_smoke"
    assert paper.create_run_dir(suffix="smoke").name == "beihang_paper_smoke"
