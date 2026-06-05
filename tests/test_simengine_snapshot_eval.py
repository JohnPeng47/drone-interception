from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import torch

from ai.rl.puffer_intercept.checkpointing import save_training_checkpoint
from ai.rl.puffer_intercept.puffer_ppo import PufferMLPPolicy, PufferPPOConfig
from ai.rl.puffer_intercept.snapshot_eval import SnapshotEvalConfig, run_snapshot_eval, select_stratified_indices
from ai.rl.puffer_intercept.scenario_table import ScenarioTable
from backends import (
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetConfig,
    TargetInitialState,
    write_sim_instances,
)


def test_select_stratified_indices_round_robins_manifest_cells(tmp_path) -> None:
    table_path = tmp_path / "samples.csimin"
    manifest_path = tmp_path / "manifest.json"
    write_sim_instances(table_path, _instances(6))
    manifest_path.write_text(
        json.dumps({
            "grid": {
                "samples_per_cell": 2,
                "cells": [
                    {"cell_index": 0, "range_m": 10.0, "closing_speed_mps": 1.0},
                    {"cell_index": 1, "range_m": 20.0, "closing_speed_mps": 2.0},
                    {"cell_index": 2, "range_m": 30.0, "closing_speed_mps": 3.0},
                ],
            }
        }),
        encoding="utf-8",
    )
    table = ScenarioTable(table_path, manifest_path=manifest_path)

    indices = select_stratified_indices(table, max_episodes=3, seed=1)

    assert len(indices) == 3
    assert [table.label(index).cell_index for index in indices] == [0, 1, 2]


def test_run_snapshot_eval_writes_summary_and_npz(tmp_path) -> None:
    table_path = tmp_path / "samples.csimin"
    checkpoint_dir = tmp_path / "checkpoints"
    out_dir = tmp_path / "snapshots"
    write_sim_instances(table_path, _instances(2))
    checkpoint = _write_checkpoint(checkpoint_dir)

    summary_path = run_snapshot_eval(
        SnapshotEvalConfig(
            scenario_table=table_path,
            checkpoint=checkpoint,
            out_dir=out_dir,
            max_episodes=2,
            num_envs=2,
            device="cpu",
            snapshot_stride=1,
        )
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["episodes"] == 2
    assert len(summary["episodes"]) == 2
    for episode in summary["episodes"]:
        artifact = np.load(episode["path"])
        assert artifact["pursuer"].ndim == 2
        assert artifact["pursuer"].shape[1] == 17
        assert artifact["target"].shape[1] == 6
        assert artifact["metrics"].shape[1] == 5
        assert artifact["actions"].shape[1] == 4
        assert artifact["dones"][-1]


def test_run_snapshot_eval_accepts_schema_compatible_simengine_batch_checkpoint(tmp_path) -> None:
    table_path = tmp_path / "samples.csimin"
    checkpoint_dir = tmp_path / "checkpoints"
    out_dir = tmp_path / "snapshots"
    write_sim_instances(table_path, _instances(1))
    checkpoint = _write_checkpoint(checkpoint_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["checkpoint_type"] = "simengine_batch_training"
    torch.save(payload, checkpoint)

    summary_path = run_snapshot_eval(
        SnapshotEvalConfig(
            scenario_table=table_path,
            checkpoint=checkpoint,
            out_dir=out_dir,
            max_episodes=1,
            num_envs=1,
            device="cpu",
            snapshot_stride=1,
        )
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["checkpoint_info"]["checkpoint_type"] == "simengine_batch_training"
    assert summary["summary"]["episodes"] == 1


def _write_checkpoint(checkpoint_dir):
    model = PufferMLPPolicy(25, 4, hidden_size=8, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters())
    args = Namespace(
        hidden_size=8,
        num_layers=1,
        total_timesteps=8,
        horizon=4,
        num_envs=2,
        minibatch_size=4,
        learning_rate=3e-4,
        anneal_lr=True,
        min_lr_ratio=0.0,
        gamma=0.995,
        gae_lambda=0.9,
        replay_ratio=1.0,
        clip_coef=0.2,
        vf_coef=2.0,
        vf_clip_coef=0.2,
        max_grad_norm=1.5,
        ent_coef=0.001,
        beta1=0.95,
        beta2=0.999,
        eps=1e-12,
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        prio_alpha=0.8,
        prio_beta0=0.2,
        optimizer="adam",
    )
    saved = save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        ppo_epoch=1,
        generator_state={"strategy": "grid_balanced", "num_envs": 2, "rng_state": {}, "cursor": 0, "cell_cursor": 0},
        checkpoint_dir=checkpoint_dir,
        global_step=8,
        args=args,
        save_latest=True,
    )
    return saved.checkpoint_path


def _instances(count: int) -> tuple[SimInstance, ...]:
    return tuple(_instance(seed=100 + index, target_x=1.0 + index) for index in range(count))


def _instance(*, seed: int, target_x: float) -> SimInstance:
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    config = SimConfig(
        pursuer=params,
        options=SimOptions(duration_s=0.02),
        targets=(TargetConfig(id="target", kind="target", radius_m=0.2),),
        intercept_radius_m=0.1,
        max_thrust_n=0.5,
        max_rate_rps=8.0,
        bounds_w=(30.0, 30.0, 20.0),
    )
    return SimInstance(
        seed=seed,
        pursuer_initial=PursuerInitialState(
            position_w=np.zeros(3),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        target_initials=(
            TargetInitialState(
                position_w=np.array([target_x, 0.0, 0.0]),
                velocity_w=np.zeros(3),
            ),
        ),
        config=config,
    )
