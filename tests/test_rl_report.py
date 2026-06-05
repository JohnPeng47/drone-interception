from __future__ import annotations

import json
from pathlib import Path

from ai.rl.report.report import generate_report, load_snapshot_evals, load_train_log, write_reports_index


def test_report_parses_train_and_snapshot_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "snapshots" / "puffer_intercept" / "000000000128"
    logs_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (logs_dir / "train.log").write_text(
        "\n".join(
            [
                "not json",
                json.dumps({"event": "start", "args": {"vf_coef": 2.0, "ent_coef": 0.1}}),
                json.dumps(
                    {
                        "global_step": 128,
                        "loss/policy_loss": 1.0,
                        "loss/value_loss": 0.25,
                        "loss/entropy": 0.5,
                        "min_distance_m": 3.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (eval_dir / "snapshot_eval.json").write_text(
        json.dumps(
            {
                "checkpoint_info": {"global_step": 128},
                "summary": {"catch_rate": 0.75, "min_distance_p50_m": 1.25, "episodes": 4, "terminal_counts": {"oob": 2}},
            }
        ),
        encoding="utf-8",
    )

    train_rows, train_meta = load_train_log(run_dir)
    eval_rows = load_snapshot_evals(run_dir)

    assert train_meta["args"]["vf_coef"] == 2.0
    assert train_rows[0]["global_step"] == 128.0
    assert eval_rows == [
        {
            "global_step": 128.0,
            "catch_rate": 0.75,
            "min_distance_p50_m": 1.25,
            "episodes": 4.0,
            "out_of_bounds": 2.0,
        }
    ]

    report_path = generate_report(run_dir, run_dir / "report.html")
    html = report_path.read_text(encoding="utf-8")
    assert "Loss Over Training Steps" in html
    assert "Snapshot Trends" in html
    assert "Capture %" in html
    assert "Median Min Distance" in html
    assert "Out-of-Bounds v. Snapshots" in html
    assert "75.0%" in html
    assert "No loss series available" not in html


def test_report_explains_missing_series(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "000000000128.pt").write_bytes(b"checkpoint")

    report_path = generate_report(run_dir, run_dir / "report.html")
    html = report_path.read_text(encoding="utf-8")

    assert "Missing Data" in html
    assert "No JSON training metrics found" in html
    assert "No snapshots/**/snapshot_eval.json files found" in html


def test_reports_index_links_report_folders(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    (reports_root / "run_a").mkdir(parents=True)
    (reports_root / "run_b").mkdir(parents=True)
    (reports_root / "run_a" / "index.html").write_text("a", encoding="utf-8")
    (reports_root / "run_b" / "index.html").write_text("b", encoding="utf-8")

    index_path = write_reports_index(reports_root, tmp_path / "index.html")
    html = index_path.read_text(encoding="utf-8")

    assert 'href="reports/run_a/index.html"' in html
    assert 'href="reports/run_b/index.html"' in html
