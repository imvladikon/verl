#!/usr/bin/env python3
"""Build a hashed SGLang runtime contract for GLM-5.2 quality generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from build_blind_quality_review import canonical_sha256, file_sha256, write_json
from generate_full_quality_outputs_sglang import (
    ADAPTER_SERVER_FIELDS,
    APPROVED_SGLANG_RELEASES,
    MLA_TARGETS,
    OFFICIAL_INFERENCE_BASES,
    OFFICIAL_MODEL_ARTIFACTS,
    OFFICIAL_PENDING_STATUS,
    OFFICIAL_STATUS,
    OFFICIAL_TRAINER,
    SCHEMA_VERSION,
    TEST_ACK,
    TEST_STATUS,
    TRUSTED_WEIGHT_SHARD_MANIFESTS,
    _validate_loopback_endpoint,
    build_pair_runtime_contract,
    read_api_secret,
    secret_sha256,
    validate_runtime_manifest,
)
from launch_quality_sglang_server import build_server_kwargs
from safetensors import safe_open

FULL_TRAINABLE_PARAMETERS = {
    "mla-only": 106_149_888,
    "mla-lm-head": 108_726_272,
}
TARGET_SHAPES = {
    "q_a_proj": (("rank", 6144), (2048, "rank")),
    "q_b_proj": (("rank", 2048), (16384, "rank")),
    "kv_a_proj_with_mqa": (("rank", 6144), (576, "rank")),
    "kv_b_proj": (("rank", 512), (28672, "rank")),
    "o_proj": (("rank", 16384), (6144, "rank")),
}
LM_HEAD_SHAPES = (("rank", 6144), (154880, "rank"))
ADAPTER_KEY_RE = re.compile(
    r"(?:^|\.)model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<target>q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\."
    r"lora_(?P<side>[AB])(?:\.default)?\.weight$"
)
LM_HEAD_KEY_RE = re.compile(r"(?:^|\.)lm_head\.lora_(?P<side>[AB])(?:\.default)?\.weight$")
SHARD_MANIFEST_FIELDS = {
    "schema_version",
    "model_id",
    "revision",
    "weights_index_sha256",
    "hash_algorithm",
    "shards",
}
SHARD_ENTRY_FIELDS = {"filename", "size", "sha256"}
ADAPTER_TRAINING_PROVENANCE_FIELDS = {
    "model_id",
    "revision",
    "config_sha256",
    "weights_index_sha256",
    "weight_shard_manifest_sha256",
}
SHARD_VERIFICATION_METHOD = "trusted-sha256-manifest+full-read-once+stat-cache-v1"
SGLANG_LIVE_CODE_PATHS = (
    "python/sglang/launch_server.py",
    "python/sglang/srt/server_args.py",
    "python/sglang/srt/models/glm4_moe.py",
    "python/sglang/srt/models/deepseek_v2.py",
    "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py",
    "python/sglang/srt/lora/lora_manager.py",
    "python/sglang/srt/lora/lora_registry.py",
)


def _reject_absolute_paths(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"{label} must not bind a machine-local absolute path")


def portable_adapter_config_sha256(config: dict[str, Any]) -> str:
    portable = dict(config)
    portable["base_model_name_or_path"] = "<bound-trainer-base>"
    _reject_absolute_paths(portable, "adapter config")
    return canonical_sha256(portable)


def portable_adapter_verification_sha256(verification: dict[str, Any]) -> str:
    portable = dict(verification)
    portable.pop("adapter_dir", None)
    _reject_absolute_paths(portable, "adapter verification")
    return canonical_sha256(portable)


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


def _canonical_git_repository(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(r"git@github\.com:(.+?)(?:\.git)?", value)
    if match:
        value = f"https://github.com/{match.group(1)}"
    elif value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix("ssh://git@github.com/")
    value = value.removesuffix(".git")
    return value.rstrip("/")


def validate_sglang_checkout(path: Path, *, official: bool = False) -> dict[str, Any]:
    checkout = path.resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("SGLang HEAD is not a full Git revision")
    status = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError("SGLang checkout is not clean, including untracked files")
    tree = git_output(checkout, "rev-parse", "HEAD^{tree}")
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
    repository = _canonical_git_repository(repository)
    live_code_sha256: dict[str, str] = {}
    for relative_name in SGLANG_LIVE_CODE_PATHS:
        source = checkout / relative_name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"bound SGLang source is missing: {relative_name}")
        live_blob = git_output(checkout, "hash-object", relative_name)
        committed_blob = git_output(checkout, "rev-parse", f"HEAD:{relative_name}")
        if live_blob != committed_blob:
            raise ValueError(f"live SGLang source differs from HEAD: {relative_name}")
        live_code_sha256[relative_name] = file_sha256(source)
    identity = repository, revision, tree
    if official and identity not in APPROVED_SGLANG_RELEASES:
        raise ValueError("official SGLang repository/revision/tree is not approved")
    return {
        "checkout": str(checkout),
        "repository": repository,
        "revision": revision,
        "tree": tree,
        "live_code_sha256": live_code_sha256,
    }


def installed_distributions_sha256() -> str:
    distributions: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        direct_url = distribution.read_text("direct_url.json")
        distributions.append(
            {
                "name": name.casefold().replace("_", "-"),
                "version": distribution.version,
                "direct_url": json.loads(direct_url) if direct_url else None,
            }
        )
    distributions.sort(key=lambda item: (item["name"], item["version"], repr(item["direct_url"])))
    return canonical_sha256(distributions)


def runtime_code_artifacts() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = (
        "build_quality_sglang_runtime.py",
        "generate_full_quality_outputs_sglang.py",
        "launch_quality_sglang_server.py",
        "build_blind_quality_review.py",
    )
    return {
        "runtime_scripts": {
            name: {
                "path": str((root / name).resolve()),
                "sha256": file_sha256(root / name),
            }
            for name in names
        }
    }


def runtime_environment_artifacts() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "python_executable": str(executable),
        "python_executable_sha256": file_sha256(executable),
        "python_version": platform.python_version(),
        "installed_distributions_sha256": installed_distributions_sha256(),
    }


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
    small_files = {
        "config_sha256": config_path,
        "weights_index_sha256": index_path,
        "tokenizer_json_sha256": snapshot / "tokenizer.json",
        "tokenizer_config_sha256": snapshot / "tokenizer_config.json",
        "chat_template_sha256": snapshot / "chat_template.jinja",
    }
    missing_small = [str(path) for path in small_files.values() if not path.is_file()]
    if missing_small:
        raise FileNotFoundError(f"{label} small files are missing: {missing_small}")
    config = read_json(config_path, "model config")
    if config.get("model_type") != "glm_moe_dsa":
        raise ValueError(f"{label} is not glm_moe_dsa")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if official and text_config.get("num_hidden_layers") != 78:
        raise ValueError("official full GLM-5.2 must retain all 78 decoder layers")
    if official and model_id.endswith("-FP8"):
        quantization = config.get("quantization_config") or text_config.get("quantization_config")
        if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
            raise ValueError("official FP8 inference snapshot lacks its fp8 contract")
    index = read_json(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{label} safetensors index has an empty or invalid weight_map")
    if any(
        not isinstance(name, str) or not name or not isinstance(shard, str) or not shard or Path(shard).name != shard
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"{label} safetensors index contains invalid tensor/shard names")
    shard_names = sorted(set(weight_map.values()))
    shard_paths = [snapshot / shard for shard in shard_names]
    missing_shards = [shard.name for shard in shard_paths if not shard.is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"{label} is missing {len(missing_shards)} index-referenced shard(s): {missing_shards[:5]}"
        )
    empty_shards = [shard.name for shard in shard_paths if shard.stat().st_size <= 0]
    if empty_shards:
        raise ValueError(f"{label} contains empty shard(s): {empty_shards[:5]}")
    metadata = index.get("metadata")
    index_total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if isinstance(index_total_size, bool) or not isinstance(index_total_size, int) or index_total_size <= 0:
        raise ValueError(f"{label} index metadata.total_size must be positive")
    block = {
        "model_id": model_id,
        "revision": revision,
        "revision_verified": True,
        **{field: file_sha256(path) for field, path in small_files.items()},
        "weight_count": len(weight_map),
        "shard_count": len(shard_names),
        "index_total_size": index_total_size,
        "shard_bytes_on_disk": sum(shard.stat().st_size for shard in shard_paths),
    }
    expected = OFFICIAL_MODEL_ARTIFACTS.get((model_id, revision))
    if expected is not None:
        actual = {field: block[field] for field in expected}
        if actual != expected:
            raise ValueError(f"{label} differs from the locked official artifact contract")
    return snapshot, block


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_shard_manifest(
    path: Path,
    *,
    snapshot: Path,
    model: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], str]:
    manifest = read_json(path, f"{label} shard manifest")
    if set(manifest) != SHARD_MANIFEST_FIELDS:
        raise ValueError(f"{label} shard manifest fields are invalid")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{label} shard manifest schema is unsupported")
    if manifest.get("model_id") != model["model_id"] or manifest.get("revision") != model["revision"]:
        raise ValueError(f"{label} shard manifest model identity is invalid")
    if manifest.get("weights_index_sha256") != model["weights_index_sha256"]:
        raise ValueError(f"{label} shard manifest index digest is invalid")
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError(f"{label} shard manifest hash algorithm is invalid")
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{label} shard manifest inventory is empty")
    expected_names = sorted(
        set(read_json(snapshot / "model.safetensors.index.json", "weights index")["weight_map"].values())
    )
    observed_names: list[str] = []
    for index, entry_value in enumerate(entries):
        if not isinstance(entry_value, dict) or set(entry_value) != SHARD_ENTRY_FIELDS:
            raise ValueError(f"{label} shard manifest entry {index} is invalid")
        filename = entry_value.get("filename")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"{label} shard manifest filename is invalid")
        _strict_positive_int(entry_value.get("size"), f"{label} shard size")
        digest = entry_value.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest == "0" * 64:
            raise ValueError(f"{label} shard digest is invalid")
        observed_names.append(filename)
        if (snapshot / filename).stat().st_size != entry_value["size"]:
            raise ValueError(f"{label} shard size differs: {filename}")
    if observed_names != sorted(observed_names) or observed_names != expected_names:
        raise ValueError(f"{label} shard manifest inventory differs from the index")
    if len(entries) != model["shard_count"] or sum(entry["size"] for entry in entries) != model["shard_bytes_on_disk"]:
        raise ValueError(f"{label} shard manifest aggregate differs from the snapshot")
    manifest_digest = canonical_sha256(manifest)
    trusted = TRUSTED_WEIGHT_SHARD_MANIFESTS.get((model["model_id"], model["revision"]))
    if trusted != manifest_digest:
        raise ValueError(f"{label} shard manifest is not in the reviewed trust allowlist")
    return manifest, manifest_digest


def _secure_json(path: Path, label: str) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError(f"{label} must be owner-only and owned by the current user")
    return read_json(path, label)


def validate_weight_shard_identity(
    snapshot: Path,
    model: dict[str, Any],
    *,
    manifest_path: Path | None,
    cache_dir: Path | None,
    verify_full: bool,
    label: str,
) -> dict[str, Any]:
    """Verify a reviewed shard list once, then reuse an inode/ctime-bound receipt.

    The expensive full SHA-256 pass is only performed with ``verify_full``. Later
    runs validate the signed-off manifest plus immutable file identity metadata,
    making the normal check O(number of shards), not O(total model bytes).
    """
    base = {
        "model_id": model["model_id"],
        "revision": model["revision"],
        "verification_method": SHARD_VERIFICATION_METHOD,
        "shard_count": model["shard_count"],
        "shard_bytes_on_disk": model["shard_bytes_on_disk"],
    }
    if manifest_path is None:
        return {
            **base,
            "status": "PENDING-TRUSTED-MANIFEST",
            "manifest_sha256": None,
            "local_verification_receipt_sha256": None,
        }
    manifest_path = manifest_path.resolve()
    manifest, manifest_digest = _validate_shard_manifest(manifest_path, snapshot=snapshot, model=model, label=label)
    if cache_dir is None:
        if verify_full:
            raise ValueError("--verify-weight-shards requires a verification cache dir")
        return {
            **base,
            "status": "PENDING-LOCAL-FULL-READ",
            "manifest_sha256": manifest_digest,
            "local_verification_receipt_sha256": None,
        }
    cache_dir = cache_dir.resolve()
    if cache_dir.exists():
        cache_info = cache_dir.lstat()
        if (
            not stat.S_ISDIR(cache_info.st_mode)
            or cache_dir.is_symlink()
            or cache_info.st_uid != os.geteuid()
            or cache_info.st_mode & 0o077
        ):
            raise ValueError("weight verification cache must be owner-only")
    else:
        cache_dir.mkdir(parents=True, mode=0o700)
    snapshot_key = hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
    receipt_path = cache_dir / f"{manifest_digest}.{snapshot_key}.json"

    def current_entries(*, hash_contents: bool) -> list[dict[str, Any]]:
        results = []
        for expected in manifest["shards"]:
            shard = snapshot / expected["filename"]
            link_before = shard.lstat()
            symlink_target_before = os.readlink(shard) if stat.S_ISLNK(link_before.st_mode) else None
            target_before_path = shard.resolve(strict=True)
            target_before = target_before_path.stat()
            if not stat.S_ISREG(target_before.st_mode):
                raise ValueError(f"{label} shard target must be a regular file: {shard.name}")
            digest = _sha256_stream(shard) if hash_contents else expected["sha256"]
            link_after = shard.lstat()
            symlink_target_after = os.readlink(shard) if stat.S_ISLNK(link_after.st_mode) else None
            target_after_path = shard.resolve(strict=True)
            target_after = target_after_path.stat()
            link_identity = (
                link_after.st_dev,
                link_after.st_ino,
                link_after.st_size,
                link_after.st_mtime_ns,
                link_after.st_ctime_ns,
            )
            target_identity = (
                target_after.st_dev,
                target_after.st_ino,
                target_after.st_size,
                target_after.st_mtime_ns,
                target_after.st_ctime_ns,
            )
            if (
                link_identity
                != (
                    link_before.st_dev,
                    link_before.st_ino,
                    link_before.st_size,
                    link_before.st_mtime_ns,
                    link_before.st_ctime_ns,
                )
                or symlink_target_after != symlink_target_before
                or target_after_path != target_before_path
                or target_identity
                != (
                    target_before.st_dev,
                    target_before.st_ino,
                    target_before.st_size,
                    target_before.st_mtime_ns,
                    target_before.st_ctime_ns,
                )
            ):
                raise ValueError(f"{label} shard changed while being verified")
            if digest != expected["sha256"]:
                raise ValueError(f"{label} shard digest differs: {shard.name}")
            results.append(
                {
                    "filename": shard.name,
                    "size": target_after.st_size,
                    "sha256": digest,
                    "symlink_target": symlink_target_after,
                    "link_st_dev": link_after.st_dev,
                    "link_st_ino": link_after.st_ino,
                    "link_st_mtime_ns": link_after.st_mtime_ns,
                    "link_st_ctime_ns": link_after.st_ctime_ns,
                    "target_st_dev": target_after.st_dev,
                    "target_st_ino": target_after.st_ino,
                    "target_st_mtime_ns": target_after.st_mtime_ns,
                    "target_st_ctime_ns": target_after.st_ctime_ns,
                }
            )
        return results

    if receipt_path.exists() and not verify_full:
        receipt = _secure_json(receipt_path, f"{label} verification receipt")
        expected_receipt = {
            "schema_version": 1,
            "model_id": model["model_id"],
            "revision": model["revision"],
            "snapshot_path": str(snapshot),
            "manifest_sha256": manifest_digest,
            "verification_method": SHARD_VERIFICATION_METHOD,
            "shards": current_entries(hash_contents=False),
        }
        if receipt != expected_receipt:
            raise ValueError(f"{label} verification receipt is stale or invalid")
    elif verify_full:
        receipt = {
            "schema_version": 1,
            "model_id": model["model_id"],
            "revision": model["revision"],
            "snapshot_path": str(snapshot),
            "manifest_sha256": manifest_digest,
            "verification_method": SHARD_VERIFICATION_METHOD,
            "shards": current_entries(hash_contents=True),
        }
        write_json(receipt_path, receipt)
        os.chmod(receipt_path, 0o600)
    else:
        return {
            **base,
            "status": "PENDING-LOCAL-FULL-READ",
            "manifest_sha256": manifest_digest,
            "local_verification_receipt_sha256": None,
        }
    return {
        **base,
        "status": "VERIFIED",
        "manifest_sha256": manifest_digest,
        "local_verification_receipt_sha256": file_sha256(receipt_path),
    }


def validate_adapter(
    path: Path,
    verification_path: Path,
    *,
    name: str,
    profile: str,
    trainer_revision: str,
    official: bool,
    trainer_model: dict[str, Any] | None = None,
    trainer_weight_shard_manifest_sha256: str | None = None,
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
    base_reference = config.get("base_model_name_or_path")
    if not isinstance(base_reference, str) or not base_reference.strip():
        raise ValueError("adapter base_model_name_or_path must be nonempty")

    verification = read_json(verification_path, "adapter verification")
    if official:
        if trainer_model is None:
            raise ValueError("official adapter requires trainer model provenance")
        expected_training_provenance = {
            field: trainer_model[field]
            for field in (
                "model_id",
                "revision",
                "config_sha256",
                "weights_index_sha256",
            )
        }
        expected_training_provenance["weight_shard_manifest_sha256"] = trainer_weight_shard_manifest_sha256
        if (
            set(expected_training_provenance) != ADAPTER_TRAINING_PROVENANCE_FIELDS
            or not isinstance(trainer_weight_shard_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", trainer_weight_shard_manifest_sha256)
            or trainer_weight_shard_manifest_sha256 == "0" * 64
            or verification.get("training_provenance") != expected_training_provenance
        ):
            raise ValueError("official adapter verification does not bind its trainer base shards")
    weights_sha256 = file_sha256(weights_path)
    if verification.get("serialized_sha256") != weights_sha256:
        raise ValueError("adapter weights differ from the verified serialization")
    rank = config["r"]
    tensors: dict[tuple[int | str, str, str], dict[str, Any]] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as archive:
        keys = list(archive.keys())
        if not keys:
            raise ValueError("adapter safetensors archive is empty")
        for key in keys:
            target_match = ADAPTER_KEY_RE.search(key)
            head_match = LM_HEAD_KEY_RE.search(key)
            if target_match:
                identity = (
                    int(target_match.group("layer")),
                    target_match.group("target"),
                    target_match.group("side"),
                )
            elif head_match and profile == "mla-lm-head":
                identity = ("lm_head", "lm_head", head_match.group("side"))
            else:
                raise ValueError(f"unexpected adapter tensor: {key}")
            if identity in tensors:
                raise ValueError(f"duplicate adapter tensor identity: {identity}")
            tensor = archive.get_tensor(key)
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"adapter tensor is not BF16: {key}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"adapter tensor contains non-finite values: {key}")
            if identity[2] == "B" and not bool(torch.count_nonzero(tensor)):
                raise ValueError(f"adapter LoRA-B tensor is all zero: {key}")
            tensors[identity] = {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }

    layers = sorted({identity[0] for identity in tensors if isinstance(identity[0], int)})
    expected_layers = list(range(78)) if official else list(range(len(layers)))
    if layers != expected_layers:
        raise ValueError("adapter layer topology is not contiguous and complete")
    for layer in layers:
        for target in MLA_TARGETS:
            a = tensors.get((layer, target, "A"))
            b = tensors.get((layer, target, "B"))
            if a is None or b is None:
                raise ValueError(f"adapter is missing {target} A/B at layer {layer}")
            if len(a["shape"]) != 2 or len(b["shape"]) != 2:
                raise ValueError(f"adapter {target} tensors must be matrices")
            if a["shape"][0] != rank or b["shape"][1] != rank or a["shape"][1] <= 0 or b["shape"][0] <= 0:
                raise ValueError(f"adapter {target} rank dimension is invalid")
            if official:
                expected_a, expected_b = TARGET_SHAPES[target]
                resolved_a = tuple(rank if item == "rank" else item for item in expected_a)
                resolved_b = tuple(rank if item == "rank" else item for item in expected_b)
                if tuple(a["shape"]) != resolved_a or tuple(b["shape"]) != resolved_b:
                    raise ValueError(f"official adapter {target} shape is invalid")
    head_identities = {identity[2]: value for identity, value in tensors.items() if identity[0] == "lm_head"}
    if profile == "mla-lm-head":
        if set(head_identities) != {"A", "B"}:
            raise ValueError("adapter is missing lm_head A/B tensors")
        if official:
            expected_a, expected_b = LM_HEAD_SHAPES
            for side, expected_shape in (("A", expected_a), ("B", expected_b)):
                resolved = tuple(rank if item == "rank" else item for item in expected_shape)
                if tuple(head_identities[side]["shape"]) != resolved:
                    raise ValueError(f"official adapter lm_head LoRA-{side} shape is invalid")
    elif head_identities:
        raise ValueError("mla-only adapter contains lm_head tensors")

    parameter_count = sum(value["numel"] for value in tensors.values())
    claimed_parameter_count = verification.get("parameter_count")
    if (
        isinstance(claimed_parameter_count, bool)
        or not isinstance(claimed_parameter_count, int)
        or claimed_parameter_count <= 0
    ):
        raise ValueError("adapter verification has an invalid parameter count")
    if claimed_parameter_count != parameter_count:
        raise ValueError("adapter verification parameter count differs from tensors")
    if official and parameter_count != FULL_TRAINABLE_PARAMETERS[profile]:
        raise ValueError("full-model adapter parameter count differs from the locked profile")
    topology = [
        {
            "identity": list(identity),
            "key": value["key"],
            "shape": value["shape"],
            "dtype": value["dtype"],
        }
        for identity, value in sorted(tensors.items(), key=lambda item: str(item[0]))
    ]
    return adapter, {
        "name": name,
        "artifact_sha256": weights_sha256,
        # The PEFT config and verification proof contain relocatable path fields.
        # Bind every semantic field while normalizing those machine-local paths;
        # their actual locations remain in ``local_artifacts``.
        "config_sha256": portable_adapter_config_sha256(config),
        "verification_sha256": portable_adapter_verification_sha256(verification),
        "trainer_base_revision": trainer_revision,
        "profile": profile,
        "rank": 16,
        "alpha": 32,
        "parameter_count": parameter_count,
        "target_modules": sorted(expected_targets),
        "tensor_count": len(tensors),
        "lora_b_tensor_count": sum(identity[2] == "B" for identity in tensors),
        "tensor_dtype": "torch.bfloat16",
        "topology_sha256": canonical_sha256(topology),
        "tensor_validation_status": "FINITE-NONZERO-B-TOPOLOGY-VERIFIED",
    }


def validate_server_args(
    value: dict[str, Any],
    *,
    runtime_mode: str,
    model_path: Path,
    adapter_path: Path | None,
    adapter_name: str | None,
    target_modules: list[str] | None,
    official: bool,
    inference_model_id: str,
) -> tuple[dict[str, Any], str]:
    if Path(str(value.get("model_path", ""))).resolve() != model_path:
        raise ValueError("server_args.model_path differs from the verified model snapshot")
    if runtime_mode == "base":
        if value.get("enable_lora") is not False:
            raise ValueError("base server must explicitly disable LoRA")
        unexpected_adapter_fields = ADAPTER_SERVER_FIELDS.intersection(value)
        if unexpected_adapter_fields:
            raise ValueError(f"base server contains adapter-specific arguments: {sorted(unexpected_adapter_fields)}")
        if any(item is not None for item in (adapter_path, adapter_name, target_modules)):
            raise ValueError("base server validation must not receive adapter metadata")
    elif runtime_mode == "adapter":
        if adapter_path is None or adapter_name is None or target_modules is None:
            raise ValueError("adapter server validation requires verified adapter metadata")
        if value.get("enable_lora") is not True or value.get("lora_strict_loading") is not True:
            raise ValueError("adapter server must enable strict LoRA loading")
        lora_paths = value.get("lora_paths")
        if not isinstance(lora_paths, dict) or set(lora_paths) != {adapter_name}:
            raise ValueError("server_args.lora_paths must contain exactly the selected adapter")
        if Path(str(lora_paths[adapter_name])).resolve() != adapter_path:
            raise ValueError("server_args LoRA path differs from the verified adapter")
        if value.get("max_lora_rank") != 16:
            raise ValueError("server_args.max_lora_rank must be 16")
        if set(value.get("lora_target_modules") or []) != set(target_modules):
            raise ValueError("server LoRA targets differ from the verified adapter")
    else:
        raise ValueError("runtime_mode must be base or adapter")
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
        load_format = value.get("load_format", "auto")
        if load_format not in {"auto", "safetensors"}:
            raise ValueError(
                "official quality runtime requires real safetensors weights; "
                "load_format must be 'auto' or 'safetensors'"
            )
    max_model_len = value.get("max_model_len")
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 2048:
        raise ValueError("server_args.max_model_len must be at least 2048")
    endpoint = value.get("endpoint")
    if not isinstance(endpoint, str):
        raise TypeError("server_args.endpoint must be an absolute HTTP(S) URL")
    _validate_loopback_endpoint(endpoint, label="server_args.endpoint")
    return value, endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-mode", choices=("base", "adapter"), required=True)
    parser.add_argument("--trainer-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-verification", type=Path)
    parser.add_argument("--adapter-name")
    parser.add_argument("--profile", choices=("mla-only", "mla-lm-head"))
    parser.add_argument("--sglang-checkout", type=Path, required=True)
    parser.add_argument("--server-args", type=Path, required=True)
    parser.add_argument("--server-instance-id", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-model-id", default=OFFICIAL_TRAINER[0])
    parser.add_argument("--trainer-revision", default=OFFICIAL_TRAINER[1])
    parser.add_argument("--inference-model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument(
        "--inference-revision",
        default="f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    )
    parser.add_argument("--trainer-weight-shard-manifest", type=Path)
    parser.add_argument("--inference-weight-shard-manifest", type=Path)
    parser.add_argument("--weight-verification-cache-dir", type=Path)
    parser.add_argument(
        "--verify-weight-shards",
        action="store_true",
        help="Perform the one-time full SHA-256 pass and write reusable receipts.",
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
    api_secret = read_api_secret(
        args.api_key_file,
        expected_sha256=None,
        required=official,
    )
    trainer_path, trainer_block = validate_model_snapshot(
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
    trainer_shard_identity = validate_weight_shard_identity(
        trainer_path,
        trainer_block,
        manifest_path=args.trainer_weight_shard_manifest,
        cache_dir=args.weight_verification_cache_dir,
        verify_full=args.verify_weight_shards,
        label="trainer model",
    )
    inference_shard_identity = validate_weight_shard_identity(
        model_path,
        inference_block,
        manifest_path=args.inference_weight_shard_manifest,
        cache_dir=args.weight_verification_cache_dir,
        verify_full=args.verify_weight_shards,
        label="inference model",
    )
    adapter_values = (
        args.adapter_path,
        args.adapter_verification,
        args.adapter_name,
        args.profile,
    )
    if args.runtime_mode == "base":
        if any(value is not None for value in adapter_values):
            raise ValueError("base runtime must not receive adapter arguments")
        adapter_path = None
        adapter = None
    else:
        if any(value is None for value in adapter_values):
            raise ValueError("adapter runtime requires all adapter arguments")
        adapter_path, adapter = validate_adapter(
            args.adapter_path,
            args.adapter_verification,
            name=args.adapter_name,
            profile=args.profile,
            trainer_revision=trainer[1],
            official=official,
            trainer_model=trainer_block,
            trainer_weight_shard_manifest_sha256=trainer_shard_identity["manifest_sha256"],
        )
    server_args, endpoint = validate_server_args(
        read_json(args.server_args, "server args"),
        runtime_mode=args.runtime_mode,
        model_path=model_path,
        adapter_path=adapter_path,
        adapter_name=None if adapter is None else adapter["name"],
        target_modules=None if adapter is None else adapter["target_modules"],
        official=official,
        inference_model_id=inference[0],
    )
    sglang = validate_sglang_checkout(args.sglang_checkout, official=official)
    local_artifacts = {
        "trainer_model_path": str(trainer_path),
        "inference_model_path": str(model_path),
        "trainer_weight_shard_manifest": (
            None if args.trainer_weight_shard_manifest is None else str(args.trainer_weight_shard_manifest.resolve())
        ),
        "inference_weight_shard_manifest": (
            None
            if args.inference_weight_shard_manifest is None
            else str(args.inference_weight_shard_manifest.resolve())
        ),
        "weight_verification_cache_dir": (
            None if args.weight_verification_cache_dir is None else str(args.weight_verification_cache_dir.resolve())
        ),
    }
    if adapter_path is not None:
        local_artifacts.update(
            {
                "adapter_path": str(adapter_path),
                "adapter_verification_path": str(args.adapter_verification.resolve()),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            OFFICIAL_STATUS
            if official
            and trainer_shard_identity["status"] == "VERIFIED"
            and inference_shard_identity["status"] == "VERIFIED"
            else OFFICIAL_PENDING_STATUS
            if official
            else TEST_STATUS
        ),
        "runtime_mode": args.runtime_mode,
        "server_instance_id": args.server_instance_id,
        "endpoint": endpoint,
        "served_base_model": server_args.get("served_base_model", inference[0]),
        "trainer_base": trainer_block,
        "inference_base": inference_block,
        "artifact_contract": {
            "trainer_base": trainer_block,
            "inference_base": inference_block,
        },
        "weight_shard_identity": {
            "trainer": trainer_shard_identity,
            "inference": inference_shard_identity,
        },
        "local_artifacts": local_artifacts,
        "adapter": adapter,
        "sglang": sglang,
        "code_artifacts": runtime_code_artifacts(),
        "environment_artifacts": runtime_environment_artifacts(),
        "api_secret_sha256": None if api_secret is None else secret_sha256(api_secret),
        "server_args": server_args,
        "server_args_sha256": canonical_sha256(server_args),
    }
    manifest["pair_runtime_contract"] = build_pair_runtime_contract(manifest)
    manifest["pair_runtime_contract_sha256"] = canonical_sha256(manifest["pair_runtime_contract"])
    build_server_kwargs(manifest, api_key=api_secret)
    validation_ack = args.test_checkpoint_ack
    if official and manifest["status"] == OFFICIAL_PENDING_STATUS:
        # Building a PENDING provenance record is safe; generation still rejects
        # it unless the operator separately supplies the explicit test ack.
        validation_ack = TEST_ACK
    validate_runtime_manifest(manifest, test_checkpoint_ack=validation_ack)
    write_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
