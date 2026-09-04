#!/usr/bin/env python3
"""Validate a resolved full GLM-5.2 SFT candidate topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from estimate_full_sft_memory import EXPECTED_FULL_POLICY_PARAMETERS, estimate

MLA_TARGETS = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
]
LORA_PROFILES = {
    "mla-only": MLA_TARGETS,
    "mla-lm-head": [*MLA_TARGETS, "output_layer"],
}
# Kept for callers that imported the original five-target constant.
EXPECTED_TARGETS = MLA_TARGETS
EXPECTED_NUM_EXPERTS = 256
EXPECTED_TP_GATE_SHA256 = "80ce91da59c5615618b03c14fb74163374c7bb8e529c699ab0a661cfcd0ee958"
EXPECTED_EP_GATE_SHA256 = "a6a739c9e8a8031e89506da1f582b0255b5513823d5ace17b4fe5f723aa0ee13"
EXPECTED_TP_EP_GATE_SHA256 = "dbf6d87a6ffdb2065a5a6bb066558d92a07aff8f63e7a0192ff257da2ebca711"
CLEAN_V4_VIEW_REVISION = "mixture_targeted_wikipedia_v4_train_1792"
CLEAN_V4_VIEW_MANIFEST_SHA256 = "389e9574d42b234419f1fb9f4b9ed8c2771aaba40800d84e12e22ae019bca69c"
CLEAN_V4_SOURCE_MIXTURE_SHA256 = "34f0d92ad9b46f0289f26c7aec8cee1b4bdae76310bceda3a8bb36a71d211442"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CONFIG-FAIL: {message}")


def file_count(value: str | list[str] | None, *, label: str) -> int:
    """Count Hydra path-or-list fields without treating a path as characters."""
    if isinstance(value, str):
        require(bool(value), f"{label} path must not be empty")
        return 1
    require(isinstance(value, list), f"{label} files must resolve to a path or list")
    require(all(isinstance(path, str) and path for path in value), f"invalid {label} path")
    return len(value)


def file_list(value: str | list[str] | None, *, label: str) -> list[str]:
    file_count(value, label=label)
    return [value] if isinstance(value, str) else value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_clean_v4_padding_contract(config: dict) -> dict:
    """Keep max_length as a fail-closed cap, not a per-row BSHD allocation."""
    data = config["data"]
    engine = config["engine"]
    require(data["pad_mode"] == "no_padding", "clean-v4 requires no_padding")
    require(
        data["micro_batch_size_per_gpu"] == 1,
        "clean-v4 padding contract requires one example per micro-batch",
    )
    require(
        data["use_dynamic_bsz"] is False,
        "clean-v4 padding contract forbids multi-example dynamic micro-batches",
    )
    require(
        engine["pad_bshd_to_minibatch_max"] is False,
        "clean-v4 must keep one-example BSHD micro-batches at their own length",
    )
    return {
        "dataset_padding": "none",
        "bshd_padding_scope": "one-example-microbatch",
        "max_length_role": "fail-closed-upper-bound",
    }


def validate_clean_v4_view_binding(config: dict, manifest_path: Path) -> dict:
    """Bind a resolved local config to the checked-in exact 28x64 view."""
    require(
        manifest_path.is_file(),
        f"training-view manifest is missing: {manifest_path}",
    )
    require(
        sha256(manifest_path) == CLEAN_V4_VIEW_MANIFEST_SHA256,
        "clean-v4 training-view manifest SHA-256 drift",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest["dataset_revision"] == CLEAN_V4_VIEW_REVISION,
        "clean-v4 training-view revision drift",
    )
    require(
        manifest["scope"] == "formatting-and-script-repair-only",
        "clean-v4 training-view scope drift",
    )
    require(
        manifest["broad_russian_quality_proof"] is False,
        "clean-v4 view must not claim broad Russian quality",
    )
    require(
        manifest["source"]["mixture_rows_sha256"] == CLEAN_V4_SOURCE_MIXTURE_SHA256,
        "clean-v4 source mixture drift",
    )
    selection = manifest["selection"]
    require(selection["source_train_rows"] == 1812, "clean-v4 source count drift")
    require(selection["selected_train_rows"] == 1792, "clean-v4 selected count drift")
    require(selection["omitted_train_rows"] == 20, "clean-v4 omission count drift")
    require(selection["global_batch_size"] == 64, "clean-v4 batch size drift")
    require(selection["optimizer_steps"] == 28, "clean-v4 step count drift")
    require(selection["consumed_train_rows"] == 1792, "clean-v4 consumption drift")
    require(selection["algorithm"]["sampling"] == "none", "clean-v4 sampling enabled")
    require(
        selection["algorithm"]["replacement"] is False,
        "clean-v4 replacement enabled",
    )
    require(
        selection["per_source"]["project-authored/glm52-targeted-quality"]
        == {"source": 204, "selected": 204, "omitted": 0},
        "clean-v4 targeted-row retention drift",
    )

    view_root = manifest_path.parent.resolve()
    data = config["data"]
    train_files = file_list(data["train_files"], label="train")
    val_files = file_list(data["val_files"], label="validation")
    require(len(train_files) == len(val_files) == 1, "clean-v4 requires single tables")
    train_artifact = manifest["artifacts"]["train"]
    validation_artifact = manifest["artifacts"]["validation"]
    test_artifact = manifest["artifacts"]["test"]
    expected_train = (view_root / train_artifact["sft_parquet"]).resolve()
    expected_validation = (view_root / validation_artifact["sft_parquet"]).resolve()
    expected_test = (view_root / test_artifact["sft_parquet"]).resolve()
    require(Path(train_files[0]).resolve() == expected_train, "clean-v4 train path drift")
    require(
        Path(val_files[0]).resolve() == expected_validation,
        "clean-v4 validation path drift",
    )
    configured_files = {Path(path).resolve() for path in train_files + val_files}
    require(expected_test not in configured_files, "test split entered training config")
    require(
        sha256(expected_train) == train_artifact["sft_parquet_sha256"],
        "clean-v4 train SHA-256 drift",
    )
    require(
        sha256(expected_validation) == validation_artifact["sft_parquet_sha256"],
        "clean-v4 validation SHA-256 drift",
    )
    require(
        sha256(expected_test) == test_artifact["sft_parquet_sha256"],
        "clean-v4 test SHA-256 drift",
    )
    require(data["train_max_samples"] == 1792, "clean-v4 train_max_samples drift")
    require(data["val_max_samples"] == 244, "clean-v4 val_max_samples drift")
    trainer = config["trainer"]
    require(trainer["total_training_steps"] == 28, "clean-v4 step count drift")
    require(data["train_batch_size"] == 64, "clean-v4 global batch drift")
    require(trainer["total_epochs"] == 1, "clean-v4 must remain one epoch")
    require(
        trainer["total_training_steps"] * data["train_batch_size"] == 1792,
        "clean-v4 optimizer budget does not consume the view exactly once",
    )
    return {
        "dataset_revision": manifest["dataset_revision"],
        "scope": manifest["scope"],
        "train_rows": 1792,
        "validation_rows": 244,
        "untouched_test_rows": 184,
        "sampling": "none",
        "replacement": False,
    }


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
    parser.add_argument("--expected-nnodes", type=int, default=8)
    parser.add_argument("--expected-gpus-per-node", type=int, default=8)
    parser.add_argument("--expected-tp", type=int, default=8)
    parser.add_argument("--expected-ep", type=int, default=32)
    parser.add_argument("--expected-etp", type=int, default=1)
    parser.add_argument("--expected-pp", type=int, default=1)
    parser.add_argument("--expected-cp", type=int, default=1)
    parser.add_argument("--expected-global-batch-size", type=int, default=64)
    parser.add_argument("--clean-v4-training-view-manifest", type=Path)
    parser.add_argument(
        "--expected-lora-profile",
        choices=sorted(LORA_PROFILES),
        default="mla-only",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))

    require(
        config["model"]["lora"]["target_modules"] == LORA_PROFILES[args.expected_lora_profile],
        f"LoRA targets drift for profile {args.expected_lora_profile}",
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
    require(engine["tensor_model_parallel_size"] == args.expected_tp, "TP drift")
    require(engine["expert_model_parallel_size"] == args.expected_ep, "EP drift")
    require(engine["expert_tensor_parallel_size"] == args.expected_etp, "ETP drift")
    require(engine["pipeline_model_parallel_size"] == args.expected_pp, "PP drift")
    require(engine["context_parallel_size"] == args.expected_cp, "CP drift")
    require(
        engine["sequence_parallel"] is (args.expected_tp > 1),
        "sequence parallel drift",
    )
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
    require(
        data["train_batch_size"] == args.expected_global_batch_size,
        "global batch drift",
    )
    require(data["micro_batch_size_per_gpu"] == 1, "micro batch drift")
    require(
        data["max_length"] == args.expected_max_length and data["max_token_len_per_gpu"] == args.expected_max_length,
        "sequence length drift",
    )
    require(data["truncation"] == "error", "truncation must fail closed")
    require(
        data["tokenize_full_conversation"] is True,
        "GLM full-chat tokenization disabled",
    )
    require(
        file_count(data["train_files"], label="train") == args.expected_train_file_count,
        "train-file count drift",
    )
    require(
        file_count(data["val_files"], label="validation") == args.expected_val_file_count,
        "validation-file count drift",
    )

    trainer = config["trainer"]
    require(trainer["nnodes"] == args.expected_nnodes, "node-count drift")
    require(
        trainer["n_gpus_per_node"] == args.expected_gpus_per_node,
        "per-node GPU-count drift",
    )
    topology = compute_parallel_topology(config)
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
    training_view = None
    if args.clean_v4_training_view_manifest is not None:
        padding_contract = validate_clean_v4_padding_contract(config)
        training_view = validate_clean_v4_view_binding(config, args.clean_v4_training_view_manifest)
        training_view["padding_contract"] = padding_contract

    model_config_path = Path(config["model"]["path"]) / "config.json"
    require(model_config_path.is_file(), f"model config is missing: {model_config_path}")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    memory = estimate(
        model_config,
        tp=topology["tp"],
        ep=topology["ep"],
        etp=topology["etp"],
        lora_rank=config["model"]["lora"]["rank"],
        include_output_layer=args.expected_lora_profile == "mla-lm-head",
        sequence_length=args.expected_max_length,
    )
    require(
        memory["parameter_breakdown"]["policy_parameters"] == EXPECTED_FULL_POLICY_PARAMETERS,
        "full-model policy parameter count drift",
    )
    trainable = memory["lora"]["global_parameters"]
    result = {
        "status": "CONFIG-PASS/RUNTIME-PENDING",
        "lora_profile": args.expected_lora_profile,
        "topology": topology,
        "trainable_parameters": trainable,
        "training_steps": trainer["total_training_steps"],
        "max_length": args.expected_max_length,
        "bf16_adapter_mib": round(trainable * 2 / 2**20, 3),
        "unsharded_18_byte_bundle_gib": round(trainable * 18 / 2**30, 3),
        "memory": memory,
        "full_model_runtime": None,
        "training_view": training_view,
        "required_prior_gates": {
            "tp2_adapter_resume_evidence_root": EXPECTED_TP_GATE_SHA256,
            "ep2_routing_resume_evidence_root": EXPECTED_EP_GATE_SHA256,
            "tp2_ep2_combined_resume_evidence_root": EXPECTED_TP_EP_GATE_SHA256,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
