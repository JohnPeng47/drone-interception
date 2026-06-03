from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.csim.generator.instance_store import read_sim_instances, write_sim_instances

SOURCE_TABLE = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin"
OUT_DIR = REPO_ROOT / "scripts/generators/sim_instances/controller_regression_6_distance3x"
OUT_TABLE = OUT_DIR / "sobol_samples.csimin"
OUT_RECORDS = OUT_DIR / "distance3x_records.json"


def main() -> None:
    instances = read_sim_instances(SOURCE_TABLE)
    scaled = []
    records = []
    for instance in instances:
        target_position_w = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3)
        pursuer_position_w = np.asarray(instance.pursuer_initial.position_w, dtype=float).reshape(3)
        new_pursuer_position_w = target_position_w + 3.0 * (pursuer_position_w - target_position_w)
        new_pursuer_initial = replace(
            instance.pursuer_initial,
            position_w=new_pursuer_position_w.astype(float),
        )
        scaled_instance = replace(instance, pursuer_initial=new_pursuer_initial)
        scaled.append(scaled_instance)
        records.append(
            {
                "seed": int(instance.seed),
                "original_pursuer_position_w": pursuer_position_w.tolist(),
                "scaled_pursuer_position_w": new_pursuer_position_w.tolist(),
                "target_position_w": target_position_w.tolist(),
                "original_range_m": float(np.linalg.norm(target_position_w - pursuer_position_w)),
                "scaled_range_m": float(np.linalg.norm(target_position_w - new_pursuer_position_w)),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_sim_instances(OUT_TABLE, scaled)
    OUT_RECORDS.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_TABLE)


if __name__ == "__main__":
    main()
