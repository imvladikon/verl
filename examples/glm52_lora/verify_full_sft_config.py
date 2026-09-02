#!/usr/bin/env python3
"""Validate the resolved Hydra config for the 64-H200 GLM-5.2 SFT gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXPECTED_TARGETS = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CONFIG-FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("resolved_config", type=Path)
    parser.add_argument("--expected-max-length", type=int, default=256)
    args = parser.parse_args()
    config = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))

    require(
        config["model"]["lora"]["target_modules"] == EXPECTED_TARGETS,
        "LoRA targets drift",
    )
    require(config["model"]["lora"]["rank"] == 16, "rank drift")
    require(config["model"]["lora"]["alpha"] == 32, "alpha drift")
    require(config["model"]["lora"]["merge"] is False, "adapter merge enabled")
    require(
        config["model"]["use_remove_padding"] is False,
        "expected source-qualified BSHD path",
    )

    engine = config["engine"]
    require(engine["tensor_model_parallel_size"] == 8, "TP drift")
    require(engine["expert_model_parallel_size"] == 32, "EP drift")
    require(engine["expert_tensor_parallel_size"] == 1, "ETP drift")
    require(engine["pipeline_model_parallel_size"] == 1, "PP drift")
    require(engine["context_parallel_size"] == 1, "CP drift")
    require(engine["sequence_parallel"] is True, "sequence parallel disabled")
    require(
        engine["override_transformer_config"]["dsa_kernel_backend"] == "none",
        "unqualified DSA backend",
    )
    require(
        engine["override_transformer_config"]["moe_router_dtype"] == "fp32",
        "router dtype drift",
    )
    require(
        engine["override_transformer_config"]["recompute_granularity"] is None,
        "layer recompute breaks cross-layer DSA order",
    )
    require(
        engine["override_transformer_config"]["recompute_method"] is None,
        "recompute method must be unset",
    )
    require(
        engine["override_transformer_config"]["recompute_num_layers"] is None,
        "recompute layer count must be unset",
    )

    data = config["data"]
    require(data["train_batch_size"] == 64, "global batch drift")
    require(data["micro_batch_size_per_gpu"] == 1, "micro batch drift")
    require(
        data["max_length"] == args.expected_max_length
        and data["max_token_len_per_gpu"] == args.expected_max_length,
        "sequence length drift",
    )
    require(data["truncation"] == "error", "truncation must fail closed")
    require(
        data["tokenize_full_conversation"] is True,
        "GLM full-chat tokenization disabled",
    )

    trainer = config["trainer"]
    require(
        trainer["nnodes"] == 8 and trainer["n_gpus_per_node"] == 8,
        "64-GPU topology drift",
    )
    require(2 <= trainer["total_training_steps"] <= 8, "step bound drift")
    require(
        config["checkpoint"]["save_lora_only"] is True,
        "full-model checkpoint export enabled",
    )

    per_layer_rank_one = 85056
    trainable = 78 * 16 * per_layer_rank_one
    result = {
        "status": "CONFIG-PASS/RUNTIME-PENDING",
        "topology": {
            "nodes": 8,
            "gpus_per_node": 8,
            "tp": 8,
            "ep": 32,
            "dp": 8,
            "pp": 1,
            "cp": 1,
        },
        "trainable_parameters": trainable,
        "max_length": args.expected_max_length,
        "bf16_adapter_mib": round(trainable * 2 / 2**20, 3),
        "unsharded_16_byte_bundle_gib": round(trainable * 16 / 2**30, 3),
        "full_model_runtime": None,
        "required_prior_gate": "TP2 adapter-only save/reload SHA",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
