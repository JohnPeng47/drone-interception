from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_ROOT = Path("/workspace/drone-interception")
REMOTE_SCENARIO = REMOTE_ROOT / "data" / "scenarios" / "sobol_samples.csimin"
REMOTE_SCENARIO_MANIFEST = REMOTE_ROOT / "data" / "scenarios" / "sobol_samples.json"
RUNTIME_UPLOAD_SOURCES = [
    "./ai/rl/",
    "./backends/",
    "./puffer/src/",
    "./puffer/vendor/",
    "./scripts/runners/rl/",
]
RUNTIME_UPLOAD_EXCLUDES = [
    "__pycache__/",
    ".pytest_cache/",
    "_build/",
    "ai/rl/runs/",
    "wandb/",
]


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._file.close()

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def command(self, message: str) -> None:
        self._write("CMD", message)

    def output(self, message: str) -> None:
        self._write("OUT", message.rstrip("\n"))

    def _write(self, level: str, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] [{level}] {message}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()


def main() -> int:
    args = parse_args()
    apply_smoke_defaults(args)
    run_date = datetime.now().strftime("%Y-%m-%d")
    run_name = args.run_name or default_run_name(args)
    local_run_dir = REPO_ROOT / "ai" / "rl" / "runs" / run_date / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(local_run_dir / "orchestrator.log")
    pod: dict[str, Any] | None = None
    try:
        logger.info(f"run_name={run_name}")
        logger.info(f"local_run_dir={local_run_dir}")
        write_run_config(local_run_dir, args, run_name)

        if args.local:
            run_local_flow(args, local_run_dir, logger)
            logger.info("run completed successfully")
            return 0

        pod = load_or_create_pod(args, local_run_dir, logger)
        upload_inputs(args, pod, logger)
        remote_run_dir = REMOTE_ROOT / "remote_runs" / run_name
        bootstrap_remote(args, pod, logger)
        train_remote(args, pod, remote_run_dir, logger)
        if args.post_run_snapshots:
            post_run_snapshots_remote(args, pod, remote_run_dir, logger)
        sync_remote_artifacts(args, pod, remote_run_dir, local_run_dir, logger)
        if args.teardown:
            terminate_pod(pod, logger)
        logger.info("run completed successfully")
        return 0
    except Exception as exc:
        logger.info(f"run failed: {exc}")
        if pod is not None and args.teardown_on_failure:
            terminate_pod(pod, logger)
        raise
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision RunPod, train puffer_intercept PPO, and sync artifacts.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end training/sync job.")
    parser.add_argument("--local", action="store_true", help="Run the same training/artifact flow on this machine.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scenario-table", type=Path, default=Path("scripts/generators/sim_instances/sobol_samples_512.csimin"))
    parser.add_argument("--reward-source", type=Path, default=Path("ai/rl/puffer_intercept/rewards/default.c"))

    parser.add_argument("--runpod-api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--pod-json", type=Path, default=None, help="Reuse an existing pod metadata JSON instead of provisioning.")
    parser.add_argument("--pod-name", default="drone-puffer-intercept-rl")
    parser.add_argument("--pod-image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    parser.add_argument("--pod-disk-gb", type=int, default=80)
    parser.add_argument("--pod-min-vcpu", type=int, default=16)
    parser.add_argument("--pod-min-mem-gb", type=int, default=32)
    parser.add_argument("--pod-cloud-type", default="COMMUNITY")
    parser.add_argument("--gpu-type-id", default="NVIDIA GeForce RTX 4090")
    parser.add_argument(
        "--gpu-fallback-ids",
        default="NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA L4,NVIDIA RTX A4000",
        help="Comma-separated GPU IDs to try if --gpu-type-id has no capacity.",
    )
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--teardown", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--teardown-on-failure", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=500_000_000)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "muon"], default="adam")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=1_000_000)
    parser.add_argument("--log-interval-steps", type=int, default=8192)
    parser.add_argument("--diagnostic-sample-size", type=int, default=4096)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-s3-uri", default=None)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--s3-checkpoint-prefix", default=None)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-project", default="drone-interception")
    parser.add_argument("--wandb-group", default="puffer_intercept_4090")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-diagnostic-visuals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-aws-credentials", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aws-credentials-dir", type=Path, default=Path.home() / ".aws")

    parser.add_argument(
        "--post-run-snapshots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run ai/rl/scripts/post_run.py over checkpoints after training.",
    )
    parser.add_argument("--post-run-manifest", type=Path, default=None)
    parser.add_argument("--post-run-snapshot-dir-name", default=None)
    parser.add_argument("--post-run-num-envs", type=int, default=32)
    parser.add_argument("--post-run-seed", type=int, default=1)
    parser.add_argument("--post-run-device", default=None, help="Snapshot eval device; defaults to --device.")
    parser.add_argument("--post-run-stochastic", action="store_true")
    parser.add_argument("--post-run-snapshot-stride", type=int, default=10)
    parser.add_argument("--post-run-max-episode-steps", type=int, default=None)
    parser.add_argument("--post-run-max-episodes", type=int, default=36, help="Scenarios to evaluate per checkpoint; 0 means all scenarios.")

    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if args.teardown is None:
        args.teardown = bool(args.smoke)
    if args.teardown_on_failure is None:
        args.teardown_on_failure = bool(args.smoke)
    if args.wandb is None:
        args.wandb = (
            not args.smoke
            and ((REPO_ROOT / ".wandb_key").exists() or bool(os.environ.get("WANDB_API_KEY")))
        )
    if args.post_run_snapshots is None:
        args.post_run_snapshots = not args.smoke
    if not args.smoke:
        return
    if args.local:
        args.device = "cpu"
    args.num_envs = min(args.num_envs, 4)
    args.horizon = min(args.horizon, 2)
    args.total_timesteps = min(args.total_timesteps, 16)
    args.minibatch_size = min(args.minibatch_size, 2)
    args.checkpoint_interval_steps = min(args.checkpoint_interval_steps, 8)
    args.log_interval_steps = min(args.log_interval_steps, 8)
    args.diagnostic_sample_size = 0
    if args.wandb_name is None:
        args.wandb_name = "smoke"


def default_run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%H%M%S")
    prefix = "smoke_puffer_intercept" if args.smoke else "puffer_intercept"
    return f"{prefix}_{stamp}"


def write_run_config(local_run_dir: Path, args: argparse.Namespace, run_name: str) -> None:
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    redact_sensitive_config(payload)
    payload["run_name"] = run_name
    (local_run_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def redact_sensitive_config(payload: dict[str, Any]) -> None:
    sensitive_keys = {
        "runpod_api_key",
    }
    for key in sensitive_keys:
        if payload.get(key):
            payload[key] = "<redacted>"


def run_local_flow(args: argparse.Namespace, local_run_dir: Path, logger: RunLogger) -> None:
    logger.info("local mode: running train/artifact flow without RunPod")
    scenario = require_local_file(args.scenario_table)
    log_dir = local_run_dir / "logs"
    checkpoints_dir = local_run_dir / "checkpoints" / "puffer_intercept"
    wandb_dir = local_run_dir / "wandb"
    for path in (log_dir, checkpoints_dir, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_DIR"] = str(wandb_dir)
    train_cmd = train_command(args, scenario, checkpoints_dir, local=True)
    run_streaming(train_cmd, logger, cwd=REPO_ROOT, env=env, log_path=log_dir / "train.log")
    if args.post_run_snapshots:
        run_post_run_snapshots_local(args, local_run_dir, logger)


def load_or_create_pod(args: argparse.Namespace, local_run_dir: Path, logger: RunLogger) -> dict[str, Any]:
    if args.pod_json is not None:
        pod = json.loads(args.pod_json.read_text(encoding="utf-8"))
        logger.info(f"reusing pod metadata from {args.pod_json}")
    else:
        if not args.runpod_api_key:
            raise RuntimeError("RUNPOD_API_KEY is required unless --pod-json is provided")
        pod = provision_pod(args, logger)
    (local_run_dir / "pod.json").write_text(json.dumps(pod, indent=2, sort_keys=True), encoding="utf-8")
    endpoint = pod_endpoint(pod)
    logger.info(f"pod ssh endpoint: {endpoint['ip']}:{endpoint['port']}")
    return pod


def provision_pod(args: argparse.Namespace, logger: RunLogger) -> dict[str, Any]:
    errors = []
    for gpu_type_id in gpu_type_ids(args):
        try:
            return provision_pod_for_gpu(args, gpu_type_id, logger)
        except RuntimeError as exc:
            message = str(exc)
            errors.append({"gpu_type_id": gpu_type_id, "error": message})
            logger.info(f"provision failed for {gpu_type_id}: {message}")
            if "does not have the resources" not in message and "no longer available" not in message:
                break
    raise RuntimeError(f"pod provisioning failed for all GPU choices: {json.dumps(errors, sort_keys=True)}")


def provision_pod_for_gpu(args: argparse.Namespace, gpu_type_id: str, logger: RunLogger) -> dict[str, Any]:
    query = f"""
mutation {{
  podFindAndDeployOnDemand(input: {{
    cloudType: {args.pod_cloud_type},
    gpuCount: 1,
    gpuTypeId: "{gpu_type_id}",
    minVcpuCount: {args.pod_min_vcpu},
    minMemoryInGb: {args.pod_min_mem_gb},
    containerDiskInGb: {args.pod_disk_gb},
    volumeInGb: 0,
    name: "{args.pod_name}",
    imageName: "{args.pod_image}",
    ports: "22/tcp",
    startSsh: true
  }}) {{
    id
    machineId
    costPerHr
    desiredStatus
  }}
}}
"""
    logger.info(f"provisioning {gpu_type_id} {args.pod_cloud_type} pod")
    response = runpod_graphql(args.runpod_api_key, query)
    pod_id = response["data"]["podFindAndDeployOnDemand"]["id"]
    logger.info(f"pod {pod_id} created; waiting for SSH")
    status: dict[str, Any] | None = None
    for attempt in range(1, 91):
        status = runpod_graphql(
            args.runpod_api_key,
            f'query {{ pod(input: {{podId: "{pod_id}"}}) {{ id desiredStatus runtime {{ ports {{ privatePort publicPort ip isIpPublic type }} }} }} }}',
        )
        pod = status["data"]["pod"]
        ports = (pod.get("runtime") or {}).get("ports") or []
        if any(int(port.get("privatePort", 0)) == 22 for port in ports):
            logger.info(f"pod {pod_id} SSH is ready after {attempt} checks")
            return pod
        if attempt == 1 or attempt % 6 == 0:
            logger.info(f"waiting for SSH: attempt={attempt} desiredStatus={pod.get('desiredStatus')}")
        time.sleep(5)
    raise RuntimeError(f"pod {pod_id} did not become SSH-ready: {json.dumps(status, sort_keys=True)}")


def gpu_type_ids(args: argparse.Namespace) -> list[str]:
    values = [args.gpu_type_id]
    values.extend(part.strip() for part in str(args.gpu_fallback_ids).split(",") if part.strip())
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def upload_inputs(args: argparse.Namespace, pod: dict[str, Any], logger: RunLogger) -> None:
    scenario = require_local_file(args.scenario_table)
    reward_source = require_uploaded_runtime_file(args.reward_source)
    scenario_sha = sha256_file(scenario)
    ssh_opts = ssh_options(pod)
    ssh = ssh_base(args, pod)
    logger.info("ensuring remote rsync is installed")
    run_streaming([
        *ssh,
        "command -v rsync >/dev/null || (export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq rsync)",
    ], logger)
    logger.info(f"reward source: {reward_source}")
    logger.info(f"reward source sha256: {sha256_file(reward_source)}")

    logger.info("uploading runtime code")
    run_streaming([*ssh, f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}"], logger)
    rsync_cmd = [
        "rsync",
        "-az",
        "--delete",
        "--relative",
        "-e",
        " ".join(shlex.quote(part) for part in ["ssh", *ssh_opts]),
    ]
    for pattern in RUNTIME_UPLOAD_EXCLUDES:
        rsync_cmd.append(f"--exclude={pattern}")
    rsync_cmd += [
        *RUNTIME_UPLOAD_SOURCES,
        f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:{REMOTE_ROOT}/",
    ]
    run_streaming(rsync_cmd, logger, cwd=REPO_ROOT)

    logger.info(f"scenario table: {scenario}")
    logger.info(f"scenario table sha256: {scenario_sha}")
    run_streaming([*ssh, f"mkdir -p {shlex.quote(str(REMOTE_SCENARIO.parent))}"], logger)
    remote_sha = remote_sha256(args, pod, REMOTE_SCENARIO)
    if remote_sha == scenario_sha:
        logger.info("remote scenario table already matches local sha256; skipping scenario upload")
    else:
        if remote_sha:
            logger.info(f"remote scenario table sha256 differs: {remote_sha}")
        else:
            logger.info("remote scenario table missing; uploading scenario table")
        run_streaming([
            "rsync",
            "-az",
            "--partial",
            "-e",
            " ".join(shlex.quote(part) for part in ["ssh", *ssh_opts]),
            str(scenario),
            f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:{REMOTE_SCENARIO}",
        ], logger)
    upload_post_run_manifest(args, pod, logger)
    upload_credentials(args, pod, logger)
    upload_resume_checkpoint(args, pod, logger)


def upload_post_run_manifest(args: argparse.Namespace, pod: dict[str, Any], logger: RunLogger) -> None:
    if not args.post_run_snapshots:
        return
    manifest = local_post_run_manifest(args)
    if manifest is None:
        logger.info("post-run snapshots: no scenario manifest found")
        return
    logger.info(f"uploading post-run scenario manifest: {manifest}")
    run_streaming([*ssh_base(args, pod), f"mkdir -p {shlex.quote(str(REMOTE_SCENARIO_MANIFEST.parent))}"], logger)
    run_streaming([
        "rsync",
        "-az",
        "--partial",
        "-e",
        " ".join(shlex.quote(part) for part in ["ssh", *ssh_options(pod)]),
        str(manifest),
        f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:{REMOTE_SCENARIO_MANIFEST}",
    ], logger)


def upload_credentials(args: argparse.Namespace, pod: dict[str, Any], logger: RunLogger) -> None:
    ssh = ssh_base(args, pod)
    ssh_opts = ssh_options(pod)
    wandb_key_file = REPO_ROOT / ".wandb_key"
    if not args.wandb:
        logger.info("W&B disabled; skipping W&B credential upload")
    elif wandb_key_file.exists():
        logger.info("uploading W&B key from .wandb_key")
        run_streaming(["scp", "-q", *scp_options(pod), str(wandb_key_file), f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:/root/.wandb_key"], logger)
        run_streaming([*ssh, "chmod 600 /root/.wandb_key"], logger)
    elif os.environ.get("WANDB_API_KEY"):
        logger.info("uploading W&B key from WANDB_API_KEY")
        run_streaming([
            *ssh,
            f"cat > /root/.wandb_key && chmod 600 /root/.wandb_key",
        ], logger, input_text=os.environ["WANDB_API_KEY"])
    else:
        logger.info("no W&B key found; remote training will run without W&B unless credentials already exist")

    if args.upload_aws_credentials and args.aws_credentials_dir.is_dir():
        logger.info(f"uploading AWS credentials from {args.aws_credentials_dir}")
        run_streaming([*ssh, "mkdir -p /root/.aws && chmod 700 /root/.aws"], logger)
        run_streaming([
            "rsync",
            "-az",
            "-e",
            " ".join(shlex.quote(part) for part in ["ssh", *ssh_opts]),
            f"{args.aws_credentials_dir}/",
            f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:/root/.aws/",
        ], logger)
        run_streaming([*ssh, "chmod -R go-rwx /root/.aws"], logger)


def upload_resume_checkpoint(args: argparse.Namespace, pod: dict[str, Any], logger: RunLogger) -> None:
    if args.resume_from is None or not args.resume_from.exists():
        return
    remote_resume = REMOTE_ROOT / "data" / "resume.pt"
    logger.info(f"uploading local resume checkpoint: {args.resume_from}")
    run_streaming([*ssh_base(args, pod), f"mkdir -p {shlex.quote(str(remote_resume.parent))}"], logger)
    run_streaming([
        "rsync",
        "-az",
        "-e",
        " ".join(shlex.quote(part) for part in ["ssh", *ssh_options(pod)]),
        str(args.resume_from),
        f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:{remote_resume}",
    ], logger)
    args.resume_from = remote_resume


def bootstrap_remote(args: argparse.Namespace, pod: dict[str, Any], logger: RunLogger) -> None:
    logger.info("bootstrapping remote environment")
    remote_reward_source = remote_runtime_path(args.reward_source)
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/root/.local/bin:$PATH
cd {shlex.quote(str(REMOTE_ROOT))}
echo "=== nvidia-smi ==="
nvidia-smi
echo "=== system deps ==="
apt-get update -qq
apt-get install -y -qq jq rsync ccache clang libomp-dev 2>&1 | tail -5
echo "=== python deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet numpy scipy torch wandb rich boto3
if [ -f /root/.wandb_key ]; then
    wandb login --relogin "$(cat /root/.wandb_key)"
fi
echo "=== puffer_intercept native smoke ==="
python3 - <<'PY'
from ai.rl.puffer_intercept import NativeInterceptBackend
p = "{REMOTE_SCENARIO}"
reward_source = "{remote_reward_source}"
env = NativeInterceptBackend(p, num_envs=2, reward_source=reward_source)
try:
    obs = env.reset()
    print("scenario_count", env.scenario_count)
    print("obs_shape", obs.shape)
    print("reward_source", reward_source)
finally:
    env.close()
PY
"""
    run_remote_script(args, pod, script, logger, "remote bootstrap")


def train_remote(args: argparse.Namespace, pod: dict[str, Any], remote_run_dir: Path, logger: RunLogger) -> None:
    logger.info("starting remote puffer_intercept training")
    train_args = train_command(args, REMOTE_SCENARIO, remote_run_dir / "checkpoints" / "puffer_intercept", local=False)
    train_line = " ".join(shlex.quote(str(part)) for part in train_args)
    script = f"""
set -euo pipefail
cd {shlex.quote(str(REMOTE_ROOT))}
mkdir -p {shlex.quote(str(remote_run_dir / "logs"))} {shlex.quote(str(remote_run_dir / "checkpoints" / "puffer_intercept"))} {shlex.quote(str(remote_run_dir / "wandb"))}
export WANDB_DIR={shlex.quote(str(remote_run_dir / "wandb"))}
echo "=== run metadata ===" | tee {shlex.quote(str(remote_run_dir / "logs" / "train.log"))}
git rev-parse HEAD 2>/dev/null | tee -a {shlex.quote(str(remote_run_dir / "logs" / "train.log"))} || true
sha256sum {shlex.quote(str(REMOTE_SCENARIO))} | tee -a {shlex.quote(str(remote_run_dir / "logs" / "train.log"))}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a {shlex.quote(str(remote_run_dir / "logs" / "train.log"))}
{train_line} 2>&1 | tee -a {shlex.quote(str(remote_run_dir / "logs" / "train.log"))}
"""
    run_remote_script(args, pod, script, logger, "remote training")


def post_run_snapshots_remote(args: argparse.Namespace, pod: dict[str, Any], remote_run_dir: Path, logger: RunLogger) -> None:
    logger.info("starting remote post-run checkpoint snapshots")
    command = post_run_command(
        args,
        run_dir=remote_run_dir,
        scenario=REMOTE_SCENARIO,
        manifest=REMOTE_SCENARIO_MANIFEST if local_post_run_manifest(args) is not None else None,
        local=False,
    )
    line = " ".join(shlex.quote(str(part)) for part in command)
    script = f"""
set -euo pipefail
cd {shlex.quote(str(REMOTE_ROOT))}
{line} 2>&1 | tee -a {shlex.quote(str(remote_run_dir / "logs" / "post_run.log"))}
"""
    run_remote_script(args, pod, script, logger, "remote post-run snapshots")


def run_post_run_snapshots_local(args: argparse.Namespace, local_run_dir: Path, logger: RunLogger) -> None:
    logger.info("starting local post-run checkpoint snapshots")
    command = post_run_command(
        args,
        run_dir=local_run_dir,
        scenario=require_local_file(args.scenario_table),
        manifest=local_post_run_manifest(args),
        local=True,
    )
    run_streaming(command, logger, cwd=REPO_ROOT, log_path=local_run_dir / "logs" / "post_run.log")


def sync_remote_artifacts(args: argparse.Namespace, pod: dict[str, Any], remote_run_dir: Path, local_run_dir: Path, logger: RunLogger) -> None:
    logger.info(f"syncing remote artifacts to {local_run_dir}")
    run_streaming([
        "rsync",
        "-az",
        "-e",
        " ".join(shlex.quote(part) for part in ["ssh", *ssh_options(pod)]),
        f"{args.ssh_user}@{pod_endpoint(pod)['ip']}:{remote_run_dir}/",
        f"{local_run_dir}/",
    ], logger)
    logger.info("artifact sync complete")


def terminate_pod(pod: dict[str, Any], logger: RunLogger) -> None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        logger.info("RUNPOD_API_KEY is not set; skipping teardown")
        return
    pod_id = str(pod["id"])
    logger.info(f"terminating pod {pod_id}")
    response = runpod_graphql(api_key, f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}')
    logger.info(f"terminate response: {json.dumps(response, sort_keys=True)}")


def train_command(args: argparse.Namespace, scenario: Path, checkpoint_dir: Path, *, local: bool) -> list[str]:
    reward_source = require_local_file(args.reward_source) if local else remote_runtime_path(args.reward_source)
    cmd = [
        sys.executable if local else "python3",
        "scripts/runners/rl/puffer_intercept_runner.py",
        "--scenario-table",
        str(scenario),
        "--reward-source",
        str(reward_source),
        "--num-envs",
        str(args.num_envs),
        "--horizon",
        str(args.horizon),
        "--minibatch-size",
        str(args.minibatch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--replay-ratio",
        str(args.replay_ratio),
        "--optimizer",
        str(args.optimizer),
        "--total-timesteps",
        str(args.total_timesteps),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-interval-steps",
        str(args.checkpoint_interval_steps),
        "--log-interval-steps",
        str(args.log_interval_steps),
        "--diagnostic-sample-size",
        str(args.diagnostic_sample_size),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--wandb-project",
        str(args.wandb_project),
        "--wandb-group",
        str(args.wandb_group),
    ]
    if args.max_episode_steps is not None:
        cmd += ["--max-episode-steps", str(args.max_episode_steps)]
    if args.resume_from is not None:
        cmd += ["--resume-from", str(args.resume_from)]
    if args.resume_s3_uri:
        cmd += ["--resume-s3-uri", str(args.resume_s3_uri)]
    s3_checkpoint_prefix = args.s3_checkpoint_prefix or (f"{args.s3_prefix.rstrip('/')}/checkpoints" if args.s3_prefix else None)
    if s3_checkpoint_prefix:
        cmd += ["--s3-checkpoint-prefix", s3_checkpoint_prefix]
    if args.wandb_name:
        cmd += ["--wandb-name", str(args.wandb_name)]
    if args.wandb:
        cmd.append("--wandb")
    cmd.append("--wandb-diagnostic-visuals" if args.wandb_diagnostic_visuals else "--no-wandb-diagnostic-visuals")
    return cmd


def post_run_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    scenario: Path,
    manifest: Path | None,
    local: bool,
) -> list[str]:
    if args.post_run_max_episodes < 0:
        raise ValueError("--post-run-max-episodes must be non-negative")
    command = [
        sys.executable if local else "python3",
        "ai/rl/scripts/post_run.py",
        str(run_dir),
        "--scenario-table",
        str(scenario),
        "--snapshot-dir-name",
        post_run_snapshot_dir_name(args),
        "--num-envs",
        str(args.post_run_num_envs),
        "--seed",
        str(args.post_run_seed),
        "--device",
        str(post_run_device(args)),
        "--snapshot-stride",
        str(args.post_run_snapshot_stride),
    ]
    if manifest is not None:
        command += ["--manifest", str(manifest)]
    if args.post_run_max_episodes > 0:
        command += ["--max-episodes", str(args.post_run_max_episodes)]
    max_episode_steps = args.post_run_max_episode_steps if args.post_run_max_episode_steps is not None else args.max_episode_steps
    if max_episode_steps is not None:
        command += ["--max-episode-steps", str(max_episode_steps)]
    if args.post_run_stochastic:
        command.append("--stochastic")
    return command


def post_run_device(args: argparse.Namespace) -> str:
    return str(args.post_run_device if args.post_run_device is not None else args.device)


def post_run_snapshot_dir_name(args: argparse.Namespace) -> str:
    if args.post_run_snapshot_dir_name:
        return str(args.post_run_snapshot_dir_name)
    scenario = Path(args.scenario_table)
    if scenario.parent.name and scenario.parent.name not in {"sim_instances", "scenarios", "data"}:
        return scenario.parent.name
    return scenario.stem


def run_remote_script(args: argparse.Namespace, pod: dict[str, Any], script: str, logger: RunLogger, label: str) -> None:
    logger.info(label)
    run_streaming([*ssh_base(args, pod), f"bash -lc {shlex.quote(script)}"], logger)


def run_streaming(
    cmd: list[str],
    logger: RunLogger,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    input_text: str | None = None,
) -> None:
    logger.command(" ".join(shlex.quote(str(part)) for part in cmd))
    log_file = None if log_path is None else log_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=None if cwd is None else str(cwd),
            env=env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if input_text is not None and proc.stdin is not None:
            proc.stdin.write(input_text)
            proc.stdin.close()
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.output(line)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)
    finally:
        if log_file is not None:
            log_file.close()


def runpod_graphql(api_key: str, query: str) -> dict[str, Any]:
    payload = json.dumps({"query": query}).encode("utf-8")
    if shutil_which("curl") is not None:
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-X",
                "POST",
                "https://api.runpod.io/graphql",
                "-H",
                "Content-Type: application/json",
                "-H",
                f"Authorization: Bearer {api_key}",
                "-d",
                payload.decode("utf-8"),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"RunPod curl failed: {proc.stderr.strip()}")
        out = json.loads(proc.stdout)
        if out.get("errors"):
            raise RuntimeError(f"RunPod GraphQL errors: {json.dumps(out['errors'], sort_keys=True)}")
        return out
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "curl/8.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            out = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod GraphQL HTTP {exc.code}: {body}") from exc
    if out.get("errors"):
        raise RuntimeError(f"RunPod GraphQL errors: {json.dumps(out['errors'], sort_keys=True)}")
    return out


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory) / command
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def require_local_file(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def require_uploaded_runtime_file(path: Path) -> Path:
    resolved = require_local_file(path)
    relative = repo_relative_path(resolved)
    if not any(is_relative_to(relative, Path(source)) for source in runtime_upload_paths()):
        raise ValueError(
            f"reward source must be under one of the uploaded runtime paths: {resolved}"
        )
    return resolved


def remote_runtime_path(path: Path) -> Path:
    return REMOTE_ROOT / repo_relative_path(require_uploaded_runtime_file(path))


def repo_relative_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    try:
        return resolved.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {resolved}") from exc


def runtime_upload_paths() -> tuple[Path, ...]:
    return tuple(Path(str(source).removeprefix("./")) for source in RUNTIME_UPLOAD_SOURCES)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def local_post_run_manifest(args: argparse.Namespace) -> Path | None:
    if args.post_run_manifest is not None:
        return require_local_file(args.post_run_manifest)
    scenario = require_local_file(args.scenario_table)
    manifest = scenario.with_suffix(".json")
    return manifest if manifest.exists() else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha256(args: argparse.Namespace, pod: dict[str, Any], path: Path) -> str | None:
    quoted_path = shlex.quote(str(path))
    command = f"if [ -f {quoted_path} ]; then sha256sum {quoted_path} | awk '{{print $1}}'; fi"
    proc = subprocess.run(
        [*ssh_base(args, pod), command],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def pod_endpoint(pod: dict[str, Any]) -> dict[str, str]:
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if int(port.get("privatePort", 0)) == 22:
            return {"ip": str(port["ip"]), "port": str(port["publicPort"])}
    raise RuntimeError(f"pod metadata does not include SSH port: {json.dumps(pod, sort_keys=True)}")


def ssh_options(pod: dict[str, Any]) -> list[str]:
    endpoint = pod_endpoint(pod)
    return [
        "-p",
        endpoint["port"],
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


def scp_options(pod: dict[str, Any]) -> list[str]:
    endpoint = pod_endpoint(pod)
    return [
        "-P",
        endpoint["port"],
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


def ssh_base(args: argparse.Namespace, pod: dict[str, Any]) -> list[str]:
    endpoint = pod_endpoint(pod)
    return ["ssh", *ssh_options(pod), f"{args.ssh_user}@{endpoint['ip']}"]


if __name__ == "__main__":
    raise SystemExit(main())
