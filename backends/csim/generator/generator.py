from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from backends.csim.bindings.types import SimInstance, TargetInitialState


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
                self._validate_instance(instance)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                continue
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
                self._validate_instance(instance)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                continue
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

    def _validate_instance(self, instance: SimInstance) -> None:
        """Validate a resolved typed instance before it is returned to callers."""
        from .validations import validate_kinematic_intercept, validate_target_in_fov

        validate_target_in_fov(instance)
        validate_kinematic_intercept(instance)

    def run(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")


class PregeneratedSimGenerator(SimGenerator):
    """SimGenerator backed by already-resolved SimInstance records."""

    def __init__(self, instances: list[SimInstance] | tuple[SimInstance, ...]):
        self.instances = tuple(instances)
        self._by_seed: dict[int, SimInstance] = {}
        for instance in self.instances:
            if instance.seed in self._by_seed:
                raise ValueError(f"Duplicate SimInstance seed {instance.seed}")
            self._by_seed[instance.seed] = instance

    @classmethod
    def from_disk(cls, path: str | Path) -> "PregeneratedSimGenerator":
        return cls(cls.sample_many_from_disk(path))

    @staticmethod
    def sample_many_from_disk(
        path: str | Path,
        *,
        count: int | None = None,
        offset: int = 0,
    ) -> list[SimInstance]:
        from .instance_store import read_sim_instances

        instances = read_sim_instances(path)
        offset = int(offset)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if count is None:
            return instances[offset:]
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        return instances[offset:offset + count]

    def sample(self, *, seed: int, **kwargs: Any) -> SimInstance:
        if kwargs:
            raise TypeError(f"{type(self).__name__}.sample does not accept kwargs")
        try:
            return self._by_seed[int(seed)]
        except KeyError as exc:
            raise KeyError(f"No pregenerated SimInstance for seed {seed}") from exc

    def sample_many(self, *, count: int, seed_start: int = 1, **kwargs: Any) -> list[SimInstance]:
        if kwargs:
            raise TypeError(f"{type(self).__name__}.sample_many does not accept kwargs")
        return [self.sample(seed=seed) for seed in range(int(seed_start), int(seed_start) + int(count))]

    def _sample_once(self, *, seed: int, **kwargs: Any) -> SimInstance:
        return self.sample(seed=seed, **kwargs)
