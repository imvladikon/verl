#!/usr/bin/env python3
"""Validate the resolved Hydra config for the 64-H200 GLM-5.2 SFT candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from estimate_full_sft_memory import EXPECTED_FULL_POLICY_PARAMETERS, estimate

EXPECTED_TARGETS = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
]
EXPECTED_NUM_EXPERTS = 256
EXPECTED_TP_GATE_SHA256 = (
    "80ce91da59c5615618b03c14fb74163374c7bb8e529c699ab0a661cfcd0ee958"
)
EXPECTED_EP_GATE_SHA256 = (
    "a6a739c9e8a8031e89506da1f582b0255b5513823d5ace17b4fe5f723aa0ee13"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CONFIG-FAIL: {message}")


def file_count(value: str | list[str] | None, *, label: str) -> int:
    """Count Hydra path-or-list fields without treating a path as characters."""
    if isinstance(value, str):
        require(bool(value), f"{label} path must not be empty")
        return 1
    require(isinstance(value, list), f"{label} files must resolve to a path or list")
    require(
        all(isinstance(path, str) and path for path in value), f"invalid {label} path"
    )
    return len(value)


def compute_parallel_topology(config: dict) -> dict[str, int]:
    """Mirror MCore's independent dense and expert process-grid arithmetic."""
    engine = config["engine"]
    trainer = config["trainer"]
    nodes = trainer["nnodes"]
    gpus_per_node = trainer["n_gpus_per_node"]
    world_size = nodes * gpus_per_node
    tp = engine["tensor_model_parallel_size"]
    ep = engine["expert_model_parallel_size"]
    etp = engine["expert_tensor_parallel_size"]
    pp = engine["pipeline_model_parallel_size"]
    cp = engine["context_parallel_size"]

    for name, value in {
        "world size": world_size,
        "TP": tp,
        "EP": ep,
        "ETP": etp,
        "PP": pp,
        "CP": cp,
    }.items():
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{name} must be positive",
        )

    dense_grid = tp * pp * cp
    expert_grid = etp * ep * pp
    require(
        world_size % dense_grid == 0,
        f"world size {world_size} is not divisible by dense TP*PP*CP grid {dense_grid}",
    )
    require(
        world_size % expert_grid == 0,
        f"world size {world_size} is not divisible by expert ETP*EP*PP grid {expert_grid}",
    )
    require(
        EXPECTED_NUM_EXPERTS % ep == 0,
        f"{EXPECTED_NUM_EXPERTS} routed experts are not divisible by EP={ep}",
    )
    if tp > 1 and ep > 1:
        require(engine["sequence_parallel"] is True, "TP+EP requires sequence parallel")

    return {
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "tp": tp,
        "ep": ep,
        "etp": etp,
        "pp": pp,
        "cp": cp,
        "dense_dp": world_size // dense_grid,
        "expert_dp": world_size // expert_grid,
        "experts_per_ep_rank": EXPECTED_NUM_EXPERTS // ep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("resolved_config", type=Path)
    parser.add_argument("--expected-max-length", type=int, default=256)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-train-file-count", type=int, default=1)
    parser.add_argument("--expected-val-file-count", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))

    require(
        config["model"]["lora"]["target_modules"] == EXPECTED_TARGETS,
        "LoRA targets drift",
    )
    require(config["model"]["lora"]["rank"] == 16, "rank drift")
    require(config["model"]["lora"]["alpha"] == 32, "alpha drift")
    require(config["model"]["lora"]["merge"] is False, "adapter merge enabled")
    require(config["model"]["mtp"]["enable"] is False, "MTP must remain disabled")
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
    require(
        file_count(data["train_files"], label="train")
        == args.expected_train_file_count,
        "train-file count drift",
    )
    require(
        file_count(data["val_files"], label="validation")
        == args.expected_val_file_count,
        "validation-file count drift",
    )

    trainer = config["trainer"]
    require(
        trainer["nnodes"] == 8 and trainer["n_gpus_per_node"] == 8,
        "64-GPU topology drift",
    )
    topology = compute_parallel_topology(config)
    require(topology["dense_dp"] == 8, "dense DP drift")
    require(topology["expert_dp"] == 2, "expert DP drift")
    require(topology["experts_per_ep_rank"] == 8, "expert ownership drift")
    if args.expected_steps is None:
        require(2 <= trainer["total_training_steps"] <= 8, "step bound drift")
    else:
        require(
            trainer["total_training_steps"] == args.expected_steps,
            "step count drift",
        )
    require(
        config["checkpoint"]["save_lora_only"] is True,
        "full-model checkpoint export enabled",
    )

    model_config_path = Path(config["model"]["path"]) / "config.json"
    require(
        model_config_path.is_file(), f"model config is missing: {model_config_path}"
    )
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    memory = estimate(
        model_config,
        tp=topology["tp"],
        ep=topology["ep"],
        etp=topology["etp"],
        lora_rank=config["model"]["lora"]["rank"],
        sequence_length=args.expected_max_length,
    )
    require(
        memory["parameter_breakdown"]["policy_parameters"]
        == EXPECTED_FULL_POLICY_PARAMETERS,
        "full-model policy parameter count drift",
    )
    trainable = memory["lora"]["global_parameters"]
    result = {
        "status": "CONFIG-PASS/RUNTIME-PENDING",
        "topology": topology,
        "trainable_parameters": trainable,
        "training_steps": trainer["total_training_steps"],
        "max_length": args.expected_max_length,
        "bf16_adapter_mib": round(trainable * 2 / 2**20, 3),
        "unsharded_16_byte_bundle_gib": round(trainable * 16 / 2**30, 3),
        "memory": memory,
        "full_model_runtime": None,
        "required_prior_gates": {
            "tp2_adapter_resume_evidence_root": EXPECTED_TP_GATE_SHA256,
            "ep2_routing_resume_evidence_root": EXPECTED_EP_GATE_SHA256,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
