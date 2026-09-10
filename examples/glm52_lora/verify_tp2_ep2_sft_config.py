#!/usr/bin/env python3
"""Validate the four-GPU TP2+EP2 GLM-5.2 surgery LoRA SFT gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from verify_tp2_sft_config import EXPECTED_TARGETS, canonical, require


def verify_config(
    config: dict[str, Any],
    *,
    expected_model_path: str | Path,
    expected_train_file: str | Path,
    expected_run_dir: str | Path,
    expected_steps: int = 2,
    expected_resume_from_path: str | Path | None = None,
) -> dict[str, Any]:
    model = config["model"]
    require(canonical(model["path"]) == canonical(expected_model_path), "model path drift")
    require(model["lora"]["target_modules"] == EXPECTED_TARGETS, "LoRA targets drift")
    require(model["lora"]["rank"] == 16, "rank drift")
    require(model["lora"]["alpha"] == 32, "alpha drift")
    require(model["lora"]["merge"] is False, "adapter merge enabled")
    require(model["lora"]["dtype"] == "bfloat16", "adapter dtype drift")
    require(model["mtp"]["enable"] is False, "MTP must remain disabled")
    require(model["use_remove_padding"] is False, "expected BSHD path")

    engine = config["engine"]
    require(engine["use_mbridge"] is True, "Megatron Bridge disabled")
    require(engine["vanilla_mbridge"] is False, "vanilla Bridge path selected")
    require(engine["tensor_model_parallel_size"] == 2, "TP drift")
    require(engine["expert_model_parallel_size"] == 2, "EP drift")
    require(engine["expert_tensor_parallel_size"] == 1, "ETP drift")
    require(engine["pipeline_model_parallel_size"] == 1, "PP drift")
    require(engine["context_parallel_size"] == 1, "CP drift")
    require(engine["sequence_parallel"] is True, "TP+EP requires sequence parallel")
    require(engine["use_distributed_optimizer"] is True, "distributed optimizer disabled")
    require(engine["param_offload"] is False, "parameter offload enabled")
    require(engine["optimizer_offload"] is False, "optimizer offload enabled")
    override = engine["override_transformer_config"]
    require(override["dsa_kernel_backend"] == "none", "unqualified DSA backend")
    require(override["moe_router_dtype"] == "fp32", "router dtype drift")
    require(override["recompute_granularity"] is None, "recompute must remain disabled")
    require(override["recompute_method"] is None, "recompute method must be unset")
    require(override["recompute_num_layers"] is None, "recompute layer count must be unset")

    data = config["data"]
    train_files = data["train_files"]
    if isinstance(train_files, str):
        train_files = [train_files]
    require(isinstance(train_files, list), "train files must resolve to a path or list")
    require(
        [canonical(path) for path in train_files] == [canonical(expected_train_file)],
        "train file drift",
    )
    require(data["val_files"] is None, "validation data unexpectedly enabled")
    require(data["train_batch_size"] == 2, "global batch drift")
    require(data["micro_batch_size_per_gpu"] == 1, "micro batch drift")
    require(data["max_length"] == 256, "sequence length drift")
    require(data["max_token_len_per_gpu"] == 256, "per-GPU sequence length drift")
    require(data["truncation"] == "error", "truncation must fail closed")
    require(data["tokenize_full_conversation"] is True, "full-chat tokenization disabled")

    trainer = config["trainer"]
    require(trainer["nnodes"] == 1, "node-count drift")
    require(trainer["n_gpus_per_node"] == 4, "GPU-count drift")
    require(trainer["total_training_steps"] == expected_steps, "step-count drift")
    require(trainer["save_freq"] == 1, "checkpoint frequency drift")
    require(trainer["test_freq"] == -1, "unexpected validation schedule")
    require(canonical(trainer["default_local_dir"]) == canonical(expected_run_dir), "run directory drift")
    if expected_resume_from_path is None:
        require(trainer["resume_mode"] == "disable", "resume mode drift")
        require(trainer.get("resume_from_path") is None, "unexpected resume checkpoint")
        phase = "initial"
    else:
        require(trainer["resume_mode"] == "resume_path", "resume mode drift")
        require(
            canonical(trainer["resume_from_path"]) == canonical(expected_resume_from_path),
            "resume checkpoint drift",
        )
        phase = "resume"

    checkpoint = config["checkpoint"]
    require(checkpoint["save_lora_only"] is True, "full-model checkpoint export enabled")
    require(set(checkpoint["save_contents"]) == {"model", "optimizer", "extra"}, "checkpoint contents drift")

    trainable = 10 * 16 * 85_056
    return {
        "status": "CONFIG-PASS/RUNTIME-PENDING",
        "phase": phase,
        "total_training_steps": expected_steps,
        "topology": {
            "nodes": 1,
            "gpus": 4,
            "tp": 2,
            "ep": 2,
            "etp": 1,
            "dense_dp": 2,
            "expert_dp": 2,
        },
        "trainable_parameters": trainable,
        "bf16_adapter_mib": round(trainable * 2 / 2**20, 3),
        "required_runtime_evidence": [
            "two finite optimizer steps with simultaneous TP2 and EP2",
            "two-rank dataloader state and four-rank distributed checkpoint",
            "adapter/optimizer/RNG resume to the next global batch",
            "HF adapter export with finite nonzero LoRA-B tensors",
            "fresh single-GPU reload with nonzero finite logit delta",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resolved_config", type=Path)
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--expected-train-file", required=True)
    parser.add_argument("--expected-run-dir", required=True)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--expected-resume-from-path")
    args = parser.parse_args()
    config = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            verify_config(
                config,
                expected_model_path=args.expected_model_path,
                expected_train_file=args.expected_train_file,
                expected_run_dir=args.expected_run_dir,
                expected_steps=args.expected_steps,
                expected_resume_from_path=args.expected_resume_from_path,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
