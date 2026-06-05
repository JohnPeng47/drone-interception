from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_TRACE = Path(__file__).resolve().parent / "iter_8_warm_portfolio_validation" / "portfolio_candidates.csv"


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.trace_csv.parent / "pruning_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.trace_csv.open(newline="", encoding="utf-8")))
    gates = _gates(args)
    summary_rows = []
    selection_rows = []
    for gate in gates:
        selections = _simulate_gate(rows, gate)
        summary_rows.append(_summarize_gate(gate, selections))
        selection_rows.extend(_selection_rows(gate, selections))
    summary_rows.sort(
        key=lambda row: (
            -int(row["catch_count"]),
            int(row["solved_candidates"]),
            -int(row["accepted_early_count"]),
            _finite_or_inf(row["worst_selected_min_distance_m"]),
        )
    )
    _write_csv(output_dir / "pruning_summary.csv", summary_rows)
    _write_csv(output_dir / "pruning_selections.csv", selection_rows)
    payload = {
        "trace_csv": str(args.trace_csv),
        "output_dir": str(output_dir),
        "best_gate": summary_rows[0] if summary_rows else None,
        "artifacts": {
            "pruning_summary_csv": "pruning_summary.csv",
            "pruning_selections_csv": "pruning_selections.csv",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay portfolio early-accept pruning gates from candidate traces.")
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-consecutive-capture-steps", default="10,15,18")
    parser.add_argument("--max-min-distance-m", default=",0.45,0.4")
    parser.add_argument("--max-tracking-error-m", default=",0.5,0.3")
    return parser.parse_args()


def _gates(args: argparse.Namespace) -> list[dict[str, float | int | None]]:
    gates = []
    for min_consecutive in _int_list(args.min_consecutive_capture_steps):
        for max_distance in _optional_float_list(args.max_min_distance_m):
            for max_tracking in _optional_float_list(args.max_tracking_error_m):
                gates.append(
                    {
                        "min_consecutive_capture_steps": int(min_consecutive),
                        "max_min_distance_m": max_distance,
                        "max_tracking_error_m": max_tracking,
                    }
                )
    return gates


def _simulate_gate(rows: list[dict[str, Any]], gate: dict[str, float | int | None]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    selections = []
    for seed, seed_rows in sorted(by_seed.items()):
        ordered = sorted(seed_rows, key=lambda row: int(row["candidate_index"]))
        accepted = None
        for row in ordered:
            if _passes_gate(row, gate):
                accepted = row
                break
        if accepted is None:
            clean = [row for row in ordered if _as_bool(row.get("clean", False)) and not _as_bool(row.get("skipped", False))]
            accepted = min(clean, key=_production_score_key) if clean else ordered[-1]
            stop_reason = "full_portfolio"
            solved = len(ordered)
        else:
            stop_reason = "early_accept"
            solved = int(accepted["candidate_index"]) + 1
        selection = dict(accepted)
        selection["stop_reason_replay"] = stop_reason
        selection["solved_candidates_replay"] = solved
        selections.append(selection)
    return selections


def _passes_gate(row: dict[str, Any], gate: dict[str, float | int | None]) -> bool:
    if not _as_bool(row.get("clean", False)) or not _as_bool(row.get("rollout_caught_radius", False)):
        return False
    consecutive = row.get("rollout_max_consecutive_capture_steps", "")
    if consecutive in {"", None}:
        consecutive = row.get("rollout_capture_steps", 0)
    if int(float(consecutive or 0)) < int(gate["min_consecutive_capture_steps"]):
        return False
    max_distance = gate["max_min_distance_m"]
    if max_distance is not None and float(row["rollout_min_distance_m"]) > float(max_distance):
        return False
    max_tracking = gate["max_tracking_error_m"]
    if max_tracking is not None and float(row["rollout_position_tracking_error_mean_m"]) > float(max_tracking):
        return False
    return True


def _summarize_gate(gate: dict[str, float | int | None], selections: list[dict[str, Any]]) -> dict[str, Any]:
    catches = [row for row in selections if _as_bool(row.get("rollout_caught_radius", False))]
    early = [row for row in selections if row.get("stop_reason_replay") == "early_accept"]
    solved_count = sum(int(row["solved_candidates_replay"]) for row in selections)
    return {
        "gate": _gate_name(gate),
        "min_consecutive_capture_steps": gate["min_consecutive_capture_steps"],
        "max_min_distance_m": "" if gate["max_min_distance_m"] is None else gate["max_min_distance_m"],
        "max_tracking_error_m": "" if gate["max_tracking_error_m"] is None else gate["max_tracking_error_m"],
        "seeds": len(selections),
        "catch_count": len(catches),
        "catch_fraction": len(catches) / max(len(selections), 1),
        "accepted_early_count": len(early),
        "solved_candidates": solved_count,
        "skipped_candidates": len(selections) * 3 - solved_count,
        "worst_selected_min_distance_m": max((float(row["rollout_min_distance_m"]) for row in catches), default=math.inf),
        "mean_selected_min_distance_m": (
            sum(float(row["rollout_min_distance_m"]) for row in catches) / max(len(catches), 1)
        ),
        "total_selected_dwell_steps": sum(int(float(row.get("rollout_capture_steps", 0) or 0)) for row in selections),
        "total_selected_consecutive_dwell_steps": sum(_consecutive_steps(row) for row in selections),
    }


def _selection_rows(gate: dict[str, float | int | None], selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for selection in selections:
        rows.append(
            {
                "gate": _gate_name(gate),
                "seed": int(selection["seed"]),
                "candidate_index": int(selection["candidate_index"]),
                "candidate": selection["candidate"],
                "stop_reason": selection["stop_reason_replay"],
                "solved_candidates": int(selection["solved_candidates_replay"]),
                "caught": _as_bool(selection["rollout_caught_radius"]),
                "rollout_min_distance_m": float(selection["rollout_min_distance_m"]),
                "rollout_capture_steps": int(float(selection.get("rollout_capture_steps", 0) or 0)),
                "rollout_max_consecutive_capture_steps": _consecutive_steps(selection),
                "rollout_position_tracking_error_mean_m": float(selection["rollout_position_tracking_error_mean_m"]),
            }
        )
    return rows


def _production_score_key(row: dict[str, Any]) -> tuple[bool, int, float, float, float]:
    return (
        not _as_bool(row.get("rollout_caught_radius", False)),
        -int(float(row.get("rollout_capture_steps", 0) or 0)),
        _finite_or_inf(row.get("rollout_min_distance_m", math.inf)),
        _finite_or_inf(row.get("rollout_position_tracking_error_mean_m", math.inf)),
        _finite_or_inf(row.get("plan_total_time_s", math.inf)),
    )


def _consecutive_steps(row: dict[str, Any]) -> int:
    value = row.get("rollout_max_consecutive_capture_steps", "")
    if value in {"", None}:
        value = row.get("rollout_capture_steps", 0)
    return int(float(value or 0))


def _gate_name(gate: dict[str, float | int | None]) -> str:
    parts = [f"consec{gate['min_consecutive_capture_steps']}"]
    if gate["max_min_distance_m"] is not None:
        parts.append(f"dist{str(gate['max_min_distance_m']).replace('.', 'p')}")
    if gate["max_tracking_error_m"] is not None:
        parts.append(f"track{str(gate['max_tracking_error_m']).replace('.', 'p')}")
    return "_".join(parts)


def _optional_float_list(value: str) -> list[float | None]:
    items: list[float | None] = []
    for item in str(value).split(","):
        stripped = item.strip()
        items.append(None if stripped == "" else float(stripped))
    return items


def _int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _finite_or_inf(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    return number if math.isfinite(number) else math.inf


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
