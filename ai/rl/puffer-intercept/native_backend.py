from __future__ import annotations

import ctypes as C
import subprocess
from pathlib import Path

import numpy as np


OBS_SIZE = 26
ACTION_SIZE = 4


class NativeInterceptBackend:
    def __init__(self, scenario_path: str | Path, *, num_envs: int, max_episode_steps: int | None = None):
        self.scenario_path = Path(scenario_path)
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self._lib = _load_lib()
        self._env = C.c_void_p()
        ok = self._lib.puffer_intercept_create(
            str(self.scenario_path).encode("utf-8"),
            C.c_int(self.num_envs),
            C.c_int(1),
            C.c_int(0 if max_episode_steps is None else int(max_episode_steps)),
            C.byref(self._env),
        )
        if not ok:
            raise RuntimeError(f"failed to create Puffer intercept env from {self.scenario_path}")
        self.observations = _array_from_ptr(
            self._lib.puffer_intercept_observations(self._env),
            self.num_envs * OBS_SIZE,
        ).reshape(self.num_envs, OBS_SIZE)
        self.actions = _array_from_ptr(
            self._lib.puffer_intercept_actions(self._env),
            self.num_envs * ACTION_SIZE,
        ).reshape(self.num_envs, ACTION_SIZE)
        self.rewards = _array_from_ptr(
            self._lib.puffer_intercept_rewards(self._env),
            self.num_envs,
        )
        self.terminals = _array_from_ptr(
            self._lib.puffer_intercept_terminals(self._env),
            self.num_envs,
        )

    @property
    def scenario_count(self) -> int:
        return int(self._lib.puffer_intercept_scenario_count(self._env))

    def reset(self) -> np.ndarray:
        self._lib.puffer_intercept_reset(self._env)
        return self.observations

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action_arr = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, ACTION_SIZE)
        self.actions[:] = action_arr
        self._lib.puffer_intercept_step(self._env)
        return self.observations, self.rewards, self.terminals > 0.5

    def close(self) -> None:
        if getattr(self, "_env", None):
            self._lib.puffer_intercept_destroy(self._env)
            self._env = C.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _load_lib() -> C.CDLL:
    lib_path = build_native()
    lib = C.CDLL(str(lib_path))
    lib.puffer_intercept_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p)]
    lib.puffer_intercept_create.restype = C.c_int
    lib.puffer_intercept_destroy.argtypes = [C.c_void_p]
    lib.puffer_intercept_destroy.restype = None
    lib.puffer_intercept_scenario_count.argtypes = [C.c_void_p]
    lib.puffer_intercept_scenario_count.restype = C.c_int
    lib.puffer_intercept_reset.argtypes = [C.c_void_p]
    lib.puffer_intercept_reset.restype = None
    lib.puffer_intercept_step.argtypes = [C.c_void_p]
    lib.puffer_intercept_step.restype = None
    lib.puffer_intercept_observations.argtypes = [C.c_void_p]
    lib.puffer_intercept_observations.restype = C.POINTER(C.c_float)
    lib.puffer_intercept_actions.argtypes = [C.c_void_p]
    lib.puffer_intercept_actions.restype = C.POINTER(C.c_float)
    lib.puffer_intercept_rewards.argtypes = [C.c_void_p]
    lib.puffer_intercept_rewards.restype = C.POINTER(C.c_float)
    lib.puffer_intercept_terminals.argtypes = [C.c_void_p]
    lib.puffer_intercept_terminals.restype = C.POINTER(C.c_float)
    return lib


def build_native() -> Path:
    root = Path(__file__).resolve().parents[3]
    src_dir = Path(__file__).resolve().parent / "c"
    csim_dir = root / "backends" / "csim"
    rendering_dir = csim_dir / "rendering"

    from backends.csim.rendering.python.build_native import build_native as build_render_native

    render_lib_path = build_render_native()
    sources = [
        src_dir / "puffer_intercept_binding.c",
        csim_dir / "pursuer_sim.c",
        csim_dir / "target_sim.c",
        csim_dir / "sim_engine.c",
        csim_dir / "camera_sim.c",
    ]
    headers = [
        csim_dir / "sim_engine.h",
        csim_dir / "sim_types.h",
        csim_dir / "sim_math.h",
        csim_dir / "target_sim.h",
        csim_dir / "camera_sim.h",
        rendering_dir / "include" / "liftoff_render_api.h",
        root / "puffer" / "src" / "vecenv.h",
        root / "puffer" / "src" / "tensor.h",
    ]
    build_dir = Path(__file__).resolve().parent / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    out = build_dir / "libpuffer_intercept_vec.so"
    newest = max(path.stat().st_mtime for path in [*sources, *headers])
    if out.exists() and out.stat().st_mtime >= newest:
        return out
    subprocess.run(
        [
            "cc",
            "-std=gnu99",
            "-O3",
            "-fPIC",
            "-shared",
            "-fopenmp",
            f"-I{csim_dir}",
            f"-I{rendering_dir / 'include'}",
            f"-I{root / 'puffer' / 'src'}",
            f"-I{root / 'puffer' / 'vendor'}",
            *(str(src) for src in sources),
            str(render_lib_path),
            "-Wl,-rpath," + str(render_lib_path.parent),
            "-lm",
            "-o",
            str(out),
        ],
        check=True,
    )
    return out


def _array_from_ptr(ptr: C.POINTER(C.c_float), count: int) -> np.ndarray:
    if not ptr:
        raise RuntimeError("native backend returned a null buffer pointer")
    return np.ctypeslib.as_array(ptr, shape=(int(count),))
