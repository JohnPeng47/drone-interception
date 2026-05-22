from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backends.input import InitialState


@dataclass(frozen=True)
class TargetInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: InitialState
    target_initial: TargetInitialState
    raw_config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class SimGenerator(ABC):
    """Shared scenario-generation contract for control sim and RL.

    `sample()` is the required boundary: it resolves a distribution/scenario
    seed into concrete initial conditions and backend-consumable config. Invalid
    generated samples are discarded and regenerated, so callers only receive
    validated samples.
    `run()` is optional execution glue for deterministic scripts or RL jobs.
    """

    max_sample_attempts: int = 1024

    def sample(self, *, seed: int, **kwargs: Any) -> SimInstance:
        first_error: ValueError | None = None
        for attempt in range(self.max_sample_attempts):
            attempted_seed = int(seed) + attempt
            instance = self._sample_once(seed=attempted_seed, **kwargs)
            try:
                self._validate_config(instance.raw_config)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                continue
            if attempt:
                metadata = dict(instance.metadata)
                metadata["requested_seed"] = int(seed)
                metadata["sample_attempts"] = attempt + 1
                instance = SimInstance(
                    seed=instance.seed,
                    pursuer_initial=instance.pursuer_initial,
                    target_initial=instance.target_initial,
                    raw_config=instance.raw_config,
                    metadata=metadata,
                )
            return instance
        message = (
            f"{type(self).__name__} failed to generate a valid sample after "
            f"{self.max_sample_attempts} attempts starting at seed {seed}"
        )
        if first_error is not None:
            message = f"{message}; first validation error: {first_error}"
        raise RuntimeError(message)

    def sample_many(self, *, count: int, seed_start: int = 1, **kwargs: Any) -> list[SimInstance]:
        instances: list[SimInstance] = []
        cursor = int(seed_start)
        attempts = 0
        max_attempts = max(int(count), 1) * self.max_sample_attempts
        first_error: ValueError | None = None
        while len(instances) < int(count) and attempts < max_attempts:
            instance = self._sample_once(seed=cursor, **kwargs)
            attempts += 1
            cursor += 1
            try:
                self._validate_config(instance.raw_config)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                continue
            if attempts != len(instances) + 1:
                metadata = dict(instance.metadata)
                metadata["sample_attempts"] = attempts
                instance = SimInstance(
                    seed=instance.seed,
                    pursuer_initial=instance.pursuer_initial,
                    target_initial=instance.target_initial,
                    raw_config=instance.raw_config,
                    metadata=metadata,
                )
            instances.append(instance)
        if len(instances) != int(count):
            message = (
                f"{type(self).__name__} generated {len(instances)} valid samples, "
                f"expected {count}, after {attempts} attempts"
            )
            if first_error is not None:
                message = f"{message}; first validation error: {first_error}"
            raise RuntimeError(message)
        return instances

    @abstractmethod
    def _sample_once(self, *, seed: int, **kwargs: Any) -> SimInstance:
        raise NotImplementedError

    def _validate_config(self, raw_config: dict[str, Any]) -> None:
        """Validate a resolved config before it is returned to callers."""
        from .validations import validate_kinematic_intercept, validate_target_in_fov

        validate_target_in_fov(raw_config)
        validate_kinematic_intercept(raw_config)

    def run(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")
