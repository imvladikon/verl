from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from verify_surgery_grpo_run import (  # noqa: E402
    parse_gpu_csv,
    parse_step_metrics,
    parse_time_file,
    validate_resume_progress,
    validate_step_metrics,
)


def metric_line(step: int, *, grad_norm: str = "9.5", reward_max: str = "0.15") -> str:
    return (
        f"\x1b[36m(TaskRunnerV1 pid=123)\x1b[0m step:{step} - "
        "actor/pg_loss:-0.02 - actor/loss:-0.02 - "
        f"actor/grad_norm:{grad_norm} - training/rollout_probs_diff_valid:1.0 - "
        "critic/rewards/mean:0.10 - critic/rewards/min:0.05 - "
        f"critic/rewards/max:{reward_max} - critic/advantages/mean:0.0 - "
        "critic/advantages/min:-0.7071 - critic/advantages/max:0.7071\n"
    )


def test_grpo_metric_parser_accepts_two_finite_steps(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("startup\n" + metric_line(1) + metric_line(2))

    result = validate_step_metrics(parse_step_metrics(log), (1, 2))

    assert [entry["step"] for entry in result] == [1, 2]
    assert result[1]["actor/grad_norm"] == 9.5


def test_grpo_metric_parser_rejects_nonfinite_gradient(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(metric_line(1) + metric_line(2, grad_norm="nan"))

    with pytest.raises(ValueError, match="non-finite actor/grad_norm"):
        validate_step_metrics(parse_step_metrics(log), (1, 2))


def test_grpo_metric_parser_rejects_missing_step(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(metric_line(1))

    with pytest.raises(ValueError, match="step drift"):
        validate_step_metrics(parse_step_metrics(log), (1, 2))


def test_grpo_metric_parser_rejects_zero_reward_spread(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(metric_line(1, reward_max="0.05") + metric_line(2))

    with pytest.raises(ValueError, match="reward group has no spread"):
        validate_step_metrics(parse_step_metrics(log), (1, 2))


def test_gpu_csv_parser_reports_sampled_peaks(tmp_path: Path) -> None:
    metrics = tmp_path / "gpu.csv"
    metrics.write_text(
        "2026/09/04 01:00:00.000, 4, 18, 81920, 0, 55.0\n"
        "2026/09/04 01:00:01.000, 4, 24576, 81920, 91, 310.0\n"
    )

    assert parse_gpu_csv(metrics) == {
        "samples": 2,
        "peak_memory_used_mib": 24576,
        "peak_utilization_percent": 91.0,
    }


def test_time_parser_handles_colons_in_field_name_and_value(tmp_path: Path) -> None:
    metrics = tmp_path / "time.txt"
    metrics.write_text(
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 6:46.34\n"
        "\tMaximum resident set size (kbytes): 1580280\n"
        "\tExit status: 0\n"
    )

    assert parse_time_file(metrics) == {
        "elapsed_wall": "6:46.34",
        "maximum_rss_kib": 1580280,
        "exit_status": 0,
    }


def test_resume_validator_requires_initial_progress_at_checkpoint_step(tmp_path: Path) -> None:
    log = tmp_path / "resume.log"
    log.write_text("Training Progress:  67%|######6   | 2/3 [00:00<?, ?it/s]\n")

    validate_resume_progress(log, 2, 3)

    with pytest.raises(ValueError, match="does not show resumed progress 1/3"):
        validate_resume_progress(log, 1, 3)
