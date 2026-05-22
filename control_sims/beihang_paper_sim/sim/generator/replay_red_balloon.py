from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from backends import SimGenerator, SimInstance

from ...noise_config import NoiseConfig
from .base import run_drake_config, write_json
from .red_balloon import RedBalloonConfigGenerator


EXPECTED_MISSES_M = {
    1: 1.34581805001,
    2: 1.34398564408,
    3: 1.31358854071,
}


class ReplayRedBalloonGenerator(SimGenerator):
    def __init__(
        self,
        *,
        seeds: tuple[int, ...] = (1, 2, 3),
        miss_atol: float = 1e-5,
        out: Path | None = None,
    ):
        self.seeds = tuple(int(seed) for seed in seeds)
        self.miss_atol = float(miss_atol)
        self.out = out
        self.config_generator = RedBalloonConfigGenerator()

    def sample(self, *, seed: int, **kwargs: Any) -> SimInstance:
        overrides = {
            "sim": {"backend": "puffer_c"},
            "vehicle": {"initial_pitch_offset_deg": 0.0},
            "controller": {"gains": {"k_1": 1.0}},
            "perception": {"pixel_noise_std_px": [1.0, 1.0]},
        }
        if "overrides" in kwargs:
            _deep_update(overrides, dict(kwargs.pop("overrides")))
        return self.config_generator.sample(seed=seed, overrides=overrides, **kwargs)

    def run(self) -> dict[str, Any]:
        rows = []
        ok = True
        print("seed | expected miss | replay miss | diff | final | status")
        print("-----+---------------+-------------+------+-------+--------")
        for seed in self.seeds:
            instance = self.sample(seed=seed)
            result = run_drake_config(
                instance.raw_config,
                seed=seed,
                controller_gains=instance.raw_config["controller"].get("gains"),
                noise_config=NoiseConfig(rng_seed=seed),
            )
            row = result.row()
            expected = EXPECTED_MISSES_M.get(seed)
            diff = None if expected is None else abs(float(row["miss_distance_m"]) - expected)
            status = "no expected" if expected is None else ("PASS" if diff <= self.miss_atol else "FAIL")
            ok = ok and status != "FAIL"
            row.update({
                "expected_miss_distance_m": expected,
                "miss_diff_m": diff,
                "status": status,
            })
            rows.append(row)
            expected_s = "n/a" if expected is None else f"{expected:.12f}"
            diff_s = "n/a" if diff is None else f"{diff:.3g}"
            print(
                f"{seed:>4} | {expected_s:>13} | "
                f"{row['miss_distance_m']:>11.12f} | {diff_s:>4} | "
                f"{row['final_distance_m']:>5.3f} | {status}"
            )

        payload = {"miss_atol": self.miss_atol, "rows": rows, "pass": ok}
        if self.out is not None:
            write_json(self.out, payload)
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--miss-atol", type=float, default=1e-5)
    args = parser.parse_args()
    result = ReplayRedBalloonGenerator(miss_atol=args.miss_atol, out=args.out).run()
    return 0 if result["pass"] else 1


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
