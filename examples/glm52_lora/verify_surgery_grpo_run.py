#!/usr/bin/env python3
"""Verify retained metrics, telemetry, and actor checkpoint from the GRPO gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"(?:^|\s)step:(?P<step>[1-9][0-9]*)\s+-\s+")
REQUIRED_METRICS = (
    "actor/grad_norm",
    "actor/pg_loss",
    "actor/loss",
    "training/rollout_probs_diff_valid",
    "critic/rewards/mean",
    "critic/rewards/min",
    "critic/rewards/max",
    "critic/advantages/mean",
    "critic/advantages/min",
    "critic/advantages/max",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def parse_step_metrics(log_path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if match is None:
            continue
        fields: dict[str, float] = {}
        metric_text = line[match.end() :]
        for piece in metric_text.split(" - "):
            key, separator, raw_value = piece.partition(":")
            if not separator or key not in REQUIRED_METRICS:
                continue
            try:
                fields[key] = float(raw_value.strip().split()[0])
            except (IndexError, ValueError) as error:
                raise ValueError(f"invalid {key} value on step {match.group('step')}") from error
        if fields:
            steps[int(match.group("step"))] = fields
    return steps


def validate_resume_progress(log_path: Path, resumed_from_step: int, final_step: int) -> None:
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    pattern = re.compile(
        rf"Training Progress:.*?\|\s*{resumed_from_step}/{final_step}\s*\["
    )
    require(
        pattern.search(text) is not None,
        f"log does not show resumed progress {resumed_from_step}/{final_step}",
    )


def validate_step_metrics(
    steps: dict[int, dict[str, float]], expected_steps: tuple[int, ...]
) -> list[dict[str, float | int]]:
    require(tuple(sorted(steps)) == expected_steps, f"step drift: {sorted(steps)} != {list(expected_steps)}")
    result: list[dict[str, float | int]] = []
    for step in expected_steps:
        metrics = steps[step]
        missing = sorted(set(REQUIRED_METRICS) - set(metrics))
        require(not missing, f"step {step} missing metrics: {missing}")
        for key in REQUIRED_METRICS:
            require(math.isfinite(metrics[key]), f"step {step} has non-finite {key}")
        require(metrics["actor/grad_norm"] > 0, f"step {step} gradient norm is not positive")
        require(
            metrics["training/rollout_probs_diff_valid"] == 1.0,
            f"step {step} rollout probability comparison is invalid",
        )
        require(
            metrics["critic/rewards/max"] > metrics["critic/rewards/min"],
            f"step {step} reward group has no spread",
        )
        require(
            metrics["critic/advantages/min"] < 0 < metrics["critic/advantages/max"],
            f"step {step} advantages do not contain both signs",
        )
        result.append({"step": step, **metrics})
    return result


def parse_gpu_csv(path: Path) -> dict[str, float | int]:
    samples = 0
    peak_used_mib = 0
    peak_utilization_percent = 0.0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) >= 5, f"gpu.csv line {line_number} has only {len(fields)} fields")
        try:
            used_mib = int(fields[2])
            utilization = float(fields[4])
        except ValueError as error:
            raise ValueError(f"invalid gpu.csv line {line_number}") from error
        samples += 1
        peak_used_mib = max(peak_used_mib, used_mib)
        peak_utilization_percent = max(peak_utilization_percent, utilization)
    require(samples > 0, "gpu.csv contains no samples")
    return {
        "samples": samples,
        "peak_memory_used_mib": peak_used_mib,
        "peak_utilization_percent": peak_utilization_percent,
    }


def parse_time_file(path: Path) -> dict[str, str | int]:
    fields: dict[str, str | int] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Elapsed (wall clock) time "):
            _, separator, value = stripped.partition("):")
            require(separator, "malformed elapsed wall-clock field")
            fields["elapsed_wall"] = value.strip()
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        if key == "Maximum resident set size (kbytes)":
            fields["maximum_rss_kib"] = int(value.strip())
        elif key == "Exit status":
            fields["exit_status"] = int(value.strip())
    require(fields.get("exit_status") == 0, f"time file has nonzero or missing exit status: {fields}")
    require("maximum_rss_kib" in fields, "time file is missing maximum RSS")
    require("elapsed_wall" in fields, "time file is missing elapsed wall time")
    return fields


def verify_run(
    run_dir: Path,
    log_path: Path,
    *,
    expected_steps: int,
    first_step: int = 1,
    resumed_from_step: int | None = None,
    layers: int = 10,
    rank: int = 16,
    alpha: int = 32,
) -> dict[str, Any]:
    require(expected_steps >= 2, "expected_steps must be at least 2")
    require(1 <= first_step <= expected_steps, "first_step is outside the expected range")
    if resumed_from_step is not None:
        require(
            first_step == resumed_from_step + 1,
            "first_step must immediately follow resumed_from_step",
        )
        validate_resume_progress(log_path, resumed_from_step, expected_steps)
    metrics = validate_step_metrics(
        parse_step_metrics(log_path), tuple(range(first_step, expected_steps + 1))
    )
    gpu_path = run_dir / "gpu.csv"
    time_path = run_dir / "time.txt"
    require(gpu_path.is_file(), f"missing GPU telemetry: {gpu_path}")
    require(time_path.is_file(), f"missing time telemetry: {time_path}")
    actor_root = run_dir / f"global_step_{expected_steps}" / "actor"
    require(actor_root.is_dir(), f"missing final actor checkpoint: {actor_root}")

    from verify_surgery_adapter import verify as verify_adapter

    adapter = verify_adapter(actor_root, layers=layers, rank=rank, alpha=alpha)
    require(adapter["checkpoint_kind"] == "ppo_actor", "final checkpoint is not a PPO actor")
    return {
        "schema_version": 1,
        "status": "SURGERY-GRPO-RUNTIME-PASS",
        "first_step": first_step,
        "expected_steps": expected_steps,
        "resumed_from_step": resumed_from_step,
        "metrics": metrics,
        "gpu": parse_gpu_csv(gpu_path),
        "process": parse_time_file(time_path),
        "actor_adapter": adapter,
        "evidence": {
            "log_sha256": sha256(log_path),
            "gpu_csv_sha256": sha256(gpu_path),
            "time_sha256": sha256(time_path),
        },
        "scope": "Surgery-model engineering proof only; no full-model quality claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--resumed-from-step", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_run(
                args.run_dir,
                args.log,
                expected_steps=args.expected_steps,
                first_step=args.first_step,
                resumed_from_step=args.resumed_from_step,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
