#!/usr/bin/env python3
"""Build a hashed SGLang runtime contract for GLM-5.2 quality generation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from build_blind_quality_review import canonical_sha256, file_sha256, write_json
from generate_full_quality_outputs_sglang import (
    MLA_TARGETS,
    OFFICIAL_INFERENCE_BASES,
    OFFICIAL_STATUS,
    OFFICIAL_TRAINER,
    TEST_ACK,
    TEST_STATUS,
    validate_runtime_manifest,
)
from launch_quality_sglang_server import build_server_kwargs

FULL_TRAINABLE_PARAMETERS = {
    "mla-only": 106_149_888,
    "mla-lm-head": 108_726_272,
}


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def snapshot_has_revision(path: Path, expected_revision: str) -> bool:
    resolved = path.resolve()
    if resolved.name == expected_revision:
        return True
    for sentinel_name in (".glm52_snapshot_revision", ".snapshot_revision"):
        sentinel = resolved / sentinel_name
        if sentinel.is_file() and sentinel.read_text(encoding="utf-8").strip() == expected_revision:
            return True
    return False


def require_snapshot(path: Path, revision: str, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    if not snapshot_has_revision(resolved, revision):
        raise ValueError(f"{label} is not pinned to revision {revision}")
    return resolved


def git_output(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_sglang_checkout(path: Path) -> dict[str, str]:
    checkout = path.resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("SGLang HEAD is not a full Git revision")
    tracked_status = git_output(checkout, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise ValueError("SGLang checkout has tracked modifications")
    remotes = git_output(checkout, "remote").splitlines()
    repository = "local-checkout"
    for remote in ("fork", "origin", *remotes):
        if remote not in remotes:
            continue
        candidate = git_output(checkout, "remote", "get-url", remote)
        if repository == "local-checkout" or candidate.startswith(("https://", "ssh://", "git@")):
            repository = candidate
        if candidate.startswith(("https://", "ssh://", "git@")):
            break
    return {"checkout": str(checkout), "repository": repository, "revision": revision}


def validate_model_snapshot(
    path: Path,
    *,
    model_id: str,
    revision: str,
    official: bool,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    snapshot = require_snapshot(path, revision, label)
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("model config or safetensors index is missing")
    config = read_json(config_path, "model config")
    if config.get("model_type") != "glm_moe_dsa":
        raise ValueError(f"{label} is not glm_moe_dsa")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if official and text_config.get("num_hidden_layers") != 78:
        raise ValueError("official full GLM-5.2 must retain all 78 decoder layers")
    if official and model_id.endswith("-FP8"):
        quantization = config.get("quantization_config") or text_config.get(
            "quantization_config"
        )
        if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
            raise ValueError("official FP8 inference snapshot lacks its fp8 contract")
    return snapshot, {
        "model_id": model_id,
        "revision": revision,
        "revision_verified": True,
        "config_sha256": file_sha256(config_path),
        "weights_index_sha256": file_sha256(index_path),
    }


def validate_adapter(
    path: Path,
    verification_path: Path,
    *,
    name: str,
    profile: str,
    trainer_revision: str,
    official: bool,
) -> tuple[Path, dict[str, Any]]:
    adapter = path.resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter}")
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("adapter_config.json or adapter_model.safetensors is missing")
    config = read_json(config_path, "adapter config")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("adapter is not a causal-language-model LoRA")
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise ValueError("adapter is not rank 16 / alpha 32")
    expected_targets = set(MLA_TARGETS)
    if profile == "mla-lm-head":
        expected_targets.add("lm_head")
    elif profile != "mla-only":
        raise ValueError("profile must be mla-only or mla-lm-head")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or set(targets) != expected_targets:
        raise ValueError(f"adapter target modules differ from the locked {profile} profile")
    base_path = config.get("base_model_name_or_path")
    if not isinstance(base_path, str) or not snapshot_has_revision(Path(base_path), trainer_revision):
        raise ValueError("adapter base_model_name_or_path is not pinned to the trainer revision")

    verification = read_json(verification_path, "adapter verification")
    if Path(str(verification.get("adapter_dir", ""))).resolve() != adapter:
        raise ValueError("adapter verification points at a different adapter directory")
    weights_sha256 = file_sha256(weights_path)
    if verification.get("serialized_sha256") != weights_sha256:
        raise ValueError("adapter weights differ from the verified serialization")
    if verification.get("all_lora_b_nonzero") is not True:
        raise ValueError("adapter verification did not prove every LoRA-B tensor nonzero")
    parameter_count = verification.get("parameter_count")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise ValueError("adapter verification has an invalid parameter count")
    if official and parameter_count != FULL_TRAINABLE_PARAMETERS[profile]:
        raise ValueError("full-model adapter parameter count differs from the locked profile")
    return adapter, {
        "name": name,
        "artifact_sha256": weights_sha256,
        "config_sha256": file_sha256(config_path),
        "verification_sha256": file_sha256(verification_path),
        "trainer_base_revision": trainer_revision,
        "profile": profile,
        "rank": 16,
        "alpha": 32,
        "parameter_count": parameter_count,
        "target_modules": sorted(expected_targets),
    }


def validate_server_args(
    value: dict[str, Any],
    *,
    model_path: Path,
    adapter_path: Path,
    adapter_name: str,
    target_modules: list[str],
    official: bool,
    inference_model_id: str,
) -> tuple[dict[str, Any], str]:
    if Path(str(value.get("model_path", ""))).resolve() != model_path:
        raise ValueError("server_args.model_path differs from the verified model snapshot")
    if value.get("enable_lora") is not True or value.get("lora_strict_loading") is not True:
        raise ValueError("server must enable strict LoRA loading")
    lora_paths = value.get("lora_paths")
    if not isinstance(lora_paths, dict) or set(lora_paths) != {adapter_name}:
        raise ValueError("server_args.lora_paths must contain exactly the selected adapter")
    if Path(str(lora_paths[adapter_name])).resolve() != adapter_path:
        raise ValueError("server_args LoRA path differs from the verified adapter")
    if value.get("max_lora_rank") != 16:
        raise ValueError("server_args.max_lora_rank must be 16")
    if set(value.get("lora_target_modules") or []) != set(target_modules):
        raise ValueError("server LoRA targets differ from the verified adapter")
    tp_size = value.get("tp_size")
    gpu_ids = value.get("gpu_ids")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
        raise ValueError("server_args.tp_size must be positive")
    if (
        not isinstance(gpu_ids, list)
        or any(isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids)
        or len(gpu_ids) != tp_size
        or len(set(gpu_ids)) != tp_size
    ):
        raise ValueError("server_args.gpu_ids must contain one unique GPU per TP rank")
    library_paths = value.get("ld_library_paths", [])
    if not isinstance(library_paths, list) or any(
        not isinstance(library_path, str)
        or not library_path
        or not Path(library_path).is_absolute()
        or not Path(library_path).is_dir()
        for library_path in library_paths
    ):
        raise ValueError("server_args.ld_library_paths must contain existing absolute directories")
    if official:
        minimum_tp = 8 if inference_model_id.endswith("-FP8") else 16
        if tp_size < minimum_tp:
            raise ValueError(f"official {inference_model_id} requires tp_size >= {minimum_tp}")
    max_model_len = value.get("max_model_len")
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 2048:
        raise ValueError("server_args.max_model_len must be at least 2048")
    endpoint = value.get("endpoint")
    if not isinstance(endpoint, str):
        raise ValueError("server_args.endpoint must be an absolute HTTP(S) URL")
    parsed_endpoint = urlparse(endpoint)
    if (
        parsed_endpoint.scheme not in {"http", "https"}
        or not parsed_endpoint.netloc
        or not parsed_endpoint.path.endswith("/v1/chat/completions")
    ):
        raise ValueError("server_args.endpoint must be an absolute HTTP(S) chat-completions URL")
    return value, endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-verification", type=Path, required=True)
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--profile", choices=("mla-only", "mla-lm-head"), required=True)
    parser.add_argument("--sglang-checkout", type=Path, required=True)
    parser.add_argument("--server-args", type=Path, required=True)
    parser.add_argument("--server-instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-model-id", default=OFFICIAL_TRAINER[0])
    parser.add_argument("--trainer-revision", default=OFFICIAL_TRAINER[1])
    parser.add_argument("--inference-model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument(
        "--inference-revision",
        default="f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    )
    parser.add_argument("--test-checkpoint-ack")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("runtime manifest exists; pass --overwrite explicitly")
    trainer = (args.trainer_model_id, args.trainer_revision)
    inference = (args.inference_model_id, args.inference_revision)
    official = trainer == OFFICIAL_TRAINER and inference in OFFICIAL_INFERENCE_BASES
    if not official and args.test_checkpoint_ack != TEST_ACK:
        raise ValueError(f"nonofficial runtime requires --test-checkpoint-ack {TEST_ACK!r}")
    _, trainer_block = validate_model_snapshot(
        args.trainer_model_path,
        model_id=trainer[0],
        revision=trainer[1],
        official=official,
        label="trainer model",
    )
    model_path, inference_block = validate_model_snapshot(
        args.model_path,
        model_id=inference[0],
        revision=inference[1],
        official=official,
        label="inference model",
    )
    adapter_path, adapter = validate_adapter(
        args.adapter_path,
        args.adapter_verification,
        name=args.adapter_name,
        profile=args.profile,
        trainer_revision=trainer[1],
        official=official,
    )
    server_args, endpoint = validate_server_args(
        read_json(args.server_args, "server args"),
        model_path=model_path,
        adapter_path=adapter_path,
        adapter_name=args.adapter_name,
        target_modules=adapter["target_modules"],
        official=official,
        inference_model_id=inference[0],
    )
    manifest = {
        "schema_version": 1,
        "status": OFFICIAL_STATUS if official else TEST_STATUS,
        "server_instance_id": args.server_instance_id,
        "endpoint": endpoint,
        "served_base_model": server_args.get("served_base_model", inference[0]),
        "trainer_base": trainer_block,
        "inference_base": inference_block,
        "adapter": adapter,
        "sglang": validate_sglang_checkout(args.sglang_checkout),
        "server_args": server_args,
        "server_args_sha256": canonical_sha256(server_args),
    }
    build_server_kwargs(manifest)
    validate_runtime_manifest(manifest, test_checkpoint_ack=args.test_checkpoint_ack)
    write_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
