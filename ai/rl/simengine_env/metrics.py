from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetricAccumulator:
    recent_episodes: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5000))
    scenario_indices: set[int] = field(default_factory=set)

    def observe_infos(self, infos: list[dict[str, Any]]) -> None:
        for info in infos:
            self.scenario_indices.add(int(info.get("scenario_index", -1)))
            episode = info.get("episode")
            if episode is not None:
                self.recent_episodes.append(episode)

    def summary(self, *, scenario_count: int | None = None) -> dict[str, float]:
        episodes = list(self.recent_episodes)
        out: dict[str, float] = {
            "episodes": float(len(episodes)),
            "scenario_unique_count": float(len(self.scenario_indices)),
        }
        if scenario_count:
            out["scenario_coverage_fraction"] = len(self.scenario_indices) / float(scenario_count)
        if not episodes:
            return out
        intercepted = np.asarray([bool(ep.get("intercepted", False)) for ep in episodes], dtype=float)
        lengths = np.asarray([float(ep.get("length", 0.0)) for ep in episodes], dtype=float)
        returns = np.asarray([float(ep.get("return", 0.0)) for ep in episodes], dtype=float)
        min_dist = np.asarray([float(ep.get("min_distance_m", np.nan)) for ep in episodes], dtype=float)
        out.update({
            "catch_rate": float(np.mean(intercepted)),
            "episode_length": float(np.mean(lengths)),
            "episode_return": float(np.mean(returns)),
            "min_distance_m": float(np.nanmean(min_dist)),
            "timeout_rate": _reason_rate(episodes, "timeout"),
            "oob_rate": _reason_rate(episodes, "oob"),
            "nonfinite_rate": _reason_rate(episodes, "nonfinite"),
        })
        cell_counts = Counter(int(ep.get("cell_index", -1)) for ep in episodes)
        cell_hits = defaultdict(list)
        for ep in episodes:
            cell_hits[int(ep.get("cell_index", -1))].append(bool(ep.get("intercepted", False)))
        for cell, count in sorted(cell_counts.items()):
            if cell < 0:
                continue
            out[f"cell_{cell}_count"] = float(count)
            out[f"cell_{cell}_catch_rate"] = float(np.mean(cell_hits[cell]))
        return out


def _reason_rate(episodes: list[dict[str, Any]], reason: str) -> float:
    return float(np.mean([ep.get("terminal_reason") == reason for ep in episodes]))
