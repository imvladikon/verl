#!/usr/bin/env python3
"""Fail closed on the runtime evidence from the GLM-5.2 TP2+EP2 gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from verify_tp2_resume_run import parse_step_metrics, require


def verify_run(
    run_dir: str | Path,
    expected_tokens: tuple[int, int, int] = (186, 288, 214),
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    initial_log = run_dir / "run.log"
    resume_log = run_dir / "resume.log"
    require(initial_log.is_file(), "initial log is missing")
    require(resume_log.is_file(), "resume log is missing")

    initial = parse_step_metrics(initial_log)
    resumed = parse_step_metrics(resume_log)
    require([entry["step"] for entry in initial] == [1, 2], "initial run must contain exactly steps 1 and 2")
    require([entry["step"] for entry in resumed] == [3], "resume must contain exactly step 3")
    observed_tokens = tuple(int(entry["global_tokens"]) for entry in [*initial, *resumed])
    require(observed_tokens == expected_tokens, "locked dataloader token sequence drift")
    require(observed_tokens[2] != observed_tokens[0], "resume appears to have restarted the dataloader")
    for entry in [*initial, *resumed]:
        require(math.isfinite(float(entry["loss"])), f"step {entry['step']} loss is not finite")
        require(math.isfinite(float(entry["grad_norm"])), f"step {entry['step']} gradient norm is not finite")

    checkpoint_step2 = run_dir / "global_step_2"
    checkpoint_step3 = run_dir / "resumed" / "global_step_3"
    for checkpoint in (checkpoint_step2, checkpoint_step3):
        for dp_rank in (0, 1):
            require((checkpoint / f"data_{dp_rank}.pt").is_file(), f"DP{dp_rank} dataloader state missing")
        require((checkpoint / "ckpt_contents.json").is_file(), f"checkpoint manifest missing from {checkpoint}")

    resume_text = resume_log.read_text(encoding="utf-8", errors="replace")
    for rank in range(4):
        for phrase, suffix in (
            ("Loaded PEFT adapter checkpoint from", f"{checkpoint_step2}/model/dist_ckpt"),
            ("Loaded optimizer checkpoint from", f"{checkpoint_step2}/optimizer/dist_ckpt"),
            ("Loaded RNG states from", f"{checkpoint_step2}/extra/dist_ckpt"),
        ):
            require(f"[Rank {rank}] {phrase} {suffix}" in resume_text, f"rank {rank} is missing {phrase}")

    return {
        "status": "RUNTIME-PASS",
        "topology": {
            "nodes": 1,
            "gpus": 4,
            "tp": 2,
            "ep": 2,
            "etp": 1,
            "dense_dp": 2,
            "expert_dp": 2,
        },
        "initial_steps": initial,
        "resumed_steps": resumed,
        "dataloader_token_sequence": list(observed_tokens),
        "loaded_contents": ["adapter", "optimizer", "rng", "dataloader"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-tokens", default="186,288,214")
    args = parser.parse_args()
    expected_tokens = tuple(int(value) for value in args.expected_tokens.split(","))
    require(len(expected_tokens) == 3, "expected token sequence must contain three integers")
    print(json.dumps(verify_run(args.run_dir, expected_tokens), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
