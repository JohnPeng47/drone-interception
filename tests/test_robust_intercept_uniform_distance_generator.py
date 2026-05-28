from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from scripts.generators.robust_intercept_uniform_distance import (
    RobustInterceptUniformDistanceConfigGenerator,
)


def test_uniform_distance_generator_default_distribution() -> None:
    generator = RobustInterceptUniformDistanceConfigGenerator()

    points = generator._sample_points
    assert len(points) == 1500
    assert Counter(point.values["closing_speed_mps"] for point in points) == {
        0.0: 375,
        5.0: 375,
        10.0: 375,
        20.0: 375,
    }

    ranges = np.array([point.values["range_m"] for point in points])
    assert np.all(ranges >= 5.0)
    assert np.all(ranges < 20.0)

    for point in points[::137]:
        instance = generator._sample_once(seed=point.seed)
        target_position = instance.target_initials[0].position_w
        pursuer_position = instance.pursuer_initial.position_w
        assert np.linalg.norm(target_position - pursuer_position) == pytest.approx(point.values["range_m"])
