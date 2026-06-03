from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from control_sims.logging import LoggingConfig, SnapshotLogger
from backends.csim.bindings.types import SimSnapshots
from backends.csim.runner import CtbrCommandBatch, SimRunnerState, SimRunnerStep


def test_snapshot_logger_writes_batch_snapshot_rows(tmp_path):
    snapshot = SimSnapshots.from_arrays(
        pursuer=np.array([[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            0.1, 0.2, 0.3, 0.9,
            0.4, 0.5, 0.6,
            100.0, 101.0, 102.0, 103.0,
        ]], dtype=np.float32),
        target=np.array([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
        metrics=np.array([[13.0, 12.5, 1.0, 0.75, 0.0]], dtype=np.float32),
        camera=np.array([[1.0, 0.25, -0.5]], dtype=np.float32),
        max_rate_rps=np.array([1.0], dtype=np.float32),
        max_rpm=np.array([1000.0], dtype=np.float32),
    )
    state = SimRunnerState(
        snapshot=snapshot,
        active=np.array([True]),
        workload_indices=np.array([4]),
        instances=(SimpleNamespace(seed=99),),
        elapsed_s=np.array([0.5], dtype=np.float32),
        steps=np.array([10], dtype=np.int32),
    )
    step = SimRunnerStep(
        state=state,
        completed=(),
        commands=CtbrCommandBatch(
            thrust_n=np.array([9.81], dtype=np.float32),
            body_rates_b=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        ),
    )

    logger = SnapshotLogger("unit", LoggingConfig(output_dir=tmp_path / "snapshots", every_n_ticks=5))
    logger.log_snapshots(step)
    logger.close()

    with (tmp_path / "snapshots" / "unit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["sim"] == "unit"
    assert rows[0]["seed"] == "99"
    assert rows[0]["tick"] == "10"
    assert rows[0]["pursuer_qw"] == "0.8999999761581421"
    assert rows[0]["motor_3_rpm"] == "103.0"
    assert rows[0]["intercepted"] == "True"
    assert rows[0]["command_thrust_n"] == "9.8100004196167"

    config = json.loads((tmp_path / "snapshots" / "logging_config.json").read_text(encoding="utf-8"))
    assert config["every_n_ticks"] == 5
    assert config["sim"] == "unit"
    assert config["output_dir"] == str(tmp_path / "snapshots")


def test_snapshot_logger_respects_tick_rate(tmp_path):
    snapshot = SimSnapshots.from_arrays(
        pursuer=np.zeros((1, 17), dtype=np.float32),
        target=np.zeros((1, 6), dtype=np.float32),
        metrics=np.zeros((1, 5), dtype=np.float32),
        camera=np.zeros((1, 3), dtype=np.float32),
        max_rate_rps=np.array([1.0], dtype=np.float32),
        max_rpm=np.array([1000.0], dtype=np.float32),
    )
    state = SimRunnerState(
        snapshot=snapshot,
        active=np.array([True]),
        workload_indices=np.array([0]),
        instances=(SimpleNamespace(seed=1),),
        elapsed_s=np.array([0.05], dtype=np.float32),
        steps=np.array([9], dtype=np.int32),
    )
    logger = SnapshotLogger("unit", LoggingConfig(output_dir=tmp_path / "snapshots", every_n_ticks=10))
    logger.log_snapshots(SimRunnerStep(state=state, completed=()))
    logger.close()

    with (tmp_path / "snapshots" / "unit.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
