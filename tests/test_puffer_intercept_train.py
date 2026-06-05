from __future__ import annotations

from pathlib import Path

import torch

from ai.rl.puffer_intercept.native_backend import reward_source_sha256
from ai.rl.puffer_intercept.train import train
from scripts.runners.rl.puffer_intercept_runner import make_parser


def test_puffer_intercept_trainer_smoke(tmp_path: Path) -> None:
    scenario_table = Path("scripts/generators/sim_instances/sobol_samples_512.csimin")
    args = make_parser().parse_args([
        "--scenario-table",
        str(scenario_table),
        "--num-envs",
        "4",
        "--horizon",
        "2",
        "--total-timesteps",
        "8",
        "--minibatch-size",
        "2",
        "--hidden-size",
        "8",
        "--num-layers",
        "1",
        "--device",
        "cpu",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--checkpoint-interval-steps",
        "8",
        "--diagnostic-sample-size",
        "0",
    ])

    train(args)

    checkpoint_path = tmp_path / "checkpoints" / "000000000008.pt"
    latest_path = tmp_path / "checkpoints" / "latest.pt"
    assert checkpoint_path.exists()
    assert latest_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 8
    assert checkpoint["checkpoint_type"] == "puffer_intercept_training"
    assert checkpoint["generator"]["backend"] == "puffer_intercept"
    assert checkpoint["generator"]["scenario_count"] == 512
    assert checkpoint["generator"]["reward_source"].endswith("ai/rl/puffer_intercept/rewards/default.c")
    assert checkpoint["generator"]["reward_source_sha256"] == reward_source_sha256("ai/rl/puffer_intercept/rewards/default.c")
    assert checkpoint["args"]["reward_source"].endswith("ai/rl/puffer_intercept/rewards/default.c")
    assert checkpoint["args"]["reward_source_sha256"] == checkpoint["generator"]["reward_source_sha256"]
