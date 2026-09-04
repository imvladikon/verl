#!/usr/bin/env python3
"""Launch the SGLang endpoint bound by a GLM-5.2 runtime manifest."""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from build_blind_quality_review import file_sha256
from generate_full_quality_outputs_sglang import (
    read_api_secret,
    validate_runtime_manifest,
)

PASSTHROUGH_FIELDS = {
    "attention_backend",
    "chunked_prefill_size",
    "disable_cuda_graph",
    "dsa_decode_backend",
    "dsa_prefill_backend",
    "dsa_topk_backend",
    "dtype",
    "load_format",
    "mem_fraction_static",
    "moe_runner_backend",
    "quantization",
    "watchdog_timeout",
}
CONTRACT_FIELDS = {
    "enable_lora",
    "endpoint",
    "gpu_ids",
    "lora_paths",
    "lora_strict_loading",
    "lora_target_modules",
    "ld_library_paths",
    "max_lora_rank",
    "max_model_len",
    "model_path",
    "served_base_model",
    "tp_size",
}
SGLANG_IMPORT_PATHS = {
    "sglang": "python/sglang/__init__.py",
    "sglang.launch_server": "python/sglang/launch_server.py",
    "sglang.srt.server_args": "python/sglang/srt/server_args.py",
}


def read_manifest(path: Path, test_checkpoint_ack: str | None) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_runtime_manifest(manifest, test_checkpoint_ack=test_checkpoint_ack)
    return manifest


def validate_live_environment(manifest: dict[str, Any], *, official: bool) -> None:
    from build_quality_sglang_runtime import (
        runtime_environment_artifacts,
        validate_adapter,
        validate_model_snapshot,
        validate_sglang_checkout,
        validate_weight_shard_identity,
    )

    checkout = Path(manifest["sglang"]["checkout"]).resolve()
    live_sglang = validate_sglang_checkout(checkout, official=official)
    if live_sglang != manifest["sglang"]:
        raise ValueError("live SGLang checkout differs from the runtime manifest")
    for name, block in manifest["code_artifacts"]["runtime_scripts"].items():
        path = Path(block["path"])
        if not path.is_file() or file_sha256(path) != block["sha256"]:
            raise ValueError(f"live runtime code artifact differs: {name}")
    if runtime_environment_artifacts() != manifest["environment_artifacts"]:
        raise ValueError("live Python environment differs from the runtime manifest")

    local = manifest["local_artifacts"]
    for label, block_name, path_name, manifest_name in (
        (
            "trainer model",
            "trainer_base",
            "trainer_model_path",
            "trainer_weight_shard_manifest",
        ),
        (
            "inference model",
            "inference_base",
            "inference_model_path",
            "inference_weight_shard_manifest",
        ),
    ):
        block = manifest[block_name]
        _, live_block = validate_model_snapshot(
            Path(local[path_name]),
            model_id=block["model_id"],
            revision=block["revision"],
            official=official,
            label=label,
        )
        if live_block != block:
            raise ValueError(f"live {label} differs from the runtime manifest")
        live_shards = validate_weight_shard_identity(
            Path(local[path_name]).resolve(),
            live_block,
            manifest_path=(None if local[manifest_name] is None else Path(local[manifest_name])),
            cache_dir=(
                None if local["weight_verification_cache_dir"] is None else Path(local["weight_verification_cache_dir"])
            ),
            verify_full=False,
            label=label,
        )
        expected_key = "trainer" if block_name == "trainer_base" else "inference"
        if live_shards != manifest["weight_shard_identity"][expected_key]:
            raise ValueError(f"live {label} shard identity differs from the manifest")
    if manifest["runtime_mode"] == "adapter":
        adapter = manifest["adapter"]
        _, live_adapter = validate_adapter(
            Path(local["adapter_path"]),
            Path(local["adapter_verification_path"]),
            name=adapter["name"],
            profile=adapter["profile"],
            trainer_revision=manifest["trainer_base"]["revision"],
            official=official,
            trainer_model=manifest["trainer_base"],
            trainer_weight_shard_manifest_sha256=manifest["weight_shard_identity"]["trainer"]["manifest_sha256"],
        )
        if live_adapter != adapter:
            raise ValueError("live adapter differs from the runtime manifest")
    expected_python_path = (checkout / "python").resolve()
    python_path = [Path(value).resolve() for value in os.environ.get("PYTHONPATH", "").split(":") if value]
    if not python_path or python_path[0] != expected_python_path:
        raise ValueError(f"PYTHONPATH must begin with the bound SGLang source: {expected_python_path}")

    expected_gpus = ",".join(str(gpu_id) for gpu_id in manifest["server_args"]["gpu_ids"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpus:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must equal {expected_gpus!r}")
    expected_library_path = ":".join(manifest["server_args"].get("ld_library_paths", []))
    if os.environ.get("LD_LIBRARY_PATH", "") != expected_library_path:
        raise ValueError("LD_LIBRARY_PATH differs from the hashed server arguments")


def validate_sglang_import_origins(
    checkout: Path,
    modules: dict[str, Any],
) -> None:
    """Prove that the process imported SGLang from the bound clean checkout."""
    if set(modules) != set(SGLANG_IMPORT_PATHS):
        raise ValueError("SGLang import-origin module set is incomplete")
    for name, relative_path in SGLANG_IMPORT_PATHS.items():
        module_file = getattr(modules[name], "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ValueError(f"imported {name} has no source file")
        expected = (checkout / relative_path).resolve()
        if Path(module_file).resolve() != expected:
            raise ValueError(f"imported {name} is outside the bound SGLang checkout")


def gpu_inventory() -> dict[int, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    inventory: dict[int, str] = {}
    for line in output.splitlines():
        index_text, separator, uuid = line.partition(",")
        if not separator or not index_text.strip().isdigit() or not uuid.strip():
            raise ValueError("nvidia-smi returned an invalid GPU inventory")
        inventory[int(index_text.strip())] = uuid.strip()
    return inventory


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must use UTC")
    return parsed


def validate_gpu_lease_attestation(
    path: Path,
    *,
    runtime_manifest_sha256: str,
    gpu_ids: list[int],
    now: datetime | None = None,
    live_inventory: dict[int, str] | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the caller owns a live lease for exactly these GPUs."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError("GPU lease attestation must be a regular non-symlink file")
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("GPU lease attestation must be owner-only (mode 0600)")
    attestation = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "status",
        "lease_id",
        "owner_uid",
        "host",
        "runtime_manifest_sha256",
        "gpu_ids",
        "gpu_uuids",
        "issued_at_utc",
        "expires_at_utc",
        "shared_host_acknowledged",
        "foreign_processes_checked",
    }
    if not isinstance(attestation, dict) or set(attestation) != fields:
        raise ValueError("GPU lease attestation fields are invalid")
    owner_uid = attestation.get("owner_uid")
    if (
        attestation.get("schema_version") != 1
        or attestation.get("status") != "ACTIVE"
        or not isinstance(attestation.get("lease_id"), str)
        or not attestation["lease_id"].strip()
        or isinstance(owner_uid, bool)
        or not isinstance(owner_uid, int)
        or owner_uid != os.geteuid()
        or attestation.get("shared_host_acknowledged") is not True
        or attestation.get("foreign_processes_checked") is not True
    ):
        raise ValueError("GPU lease attestation ownership contract is invalid")
    if attestation.get("host") != (hostname or socket.getfqdn()):
        raise ValueError("GPU lease attestation host differs from this host")
    if attestation.get("runtime_manifest_sha256") != runtime_manifest_sha256:
        raise ValueError("GPU lease attestation is for a different runtime manifest")
    if attestation.get("gpu_ids") != gpu_ids:
        raise ValueError("GPU lease attestation covers different GPU IDs")
    issued = _parse_utc(attestation.get("issued_at_utc"), "issued_at_utc")
    expires = _parse_utc(attestation.get("expires_at_utc"), "expires_at_utc")
    current = now or datetime.now(timezone.utc)
    if not issued <= current < expires or (expires - issued).total_seconds() > 43_200:
        raise ValueError("GPU lease attestation is expired, future, or over 12 hours")
    inventory = live_inventory if live_inventory is not None else gpu_inventory()
    expected_uuids = []
    for gpu_id in gpu_ids:
        if gpu_id not in inventory:
            raise ValueError("GPU lease attestation references an unavailable GPU")
        expected_uuids.append(inventory[gpu_id])
    if attestation.get("gpu_uuids") != expected_uuids:
        raise ValueError("GPU lease attestation UUIDs differ from live hardware")
    return attestation


def build_server_kwargs(
    manifest: dict[str, Any],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    value = manifest["server_args"]
    unexpected = set(value) - CONTRACT_FIELDS - PASSTHROUGH_FIELDS
    if unexpected:
        raise ValueError(f"unsupported server argument fields: {sorted(unexpected)}")
    endpoint = urlparse(value["endpoint"])
    if endpoint.port is None:
        raise ValueError("server endpoint must include an explicit port")
    kwargs = {
        "model_path": value["model_path"],
        "served_model_name": value.get("served_base_model", manifest["served_base_model"]),
        "host": endpoint.hostname,
        "port": endpoint.port,
        "tp_size": value["tp_size"],
        "enable_lora": value["enable_lora"],
        # ``--max-model-len`` is a CLI alias; the ServerArgs dataclass field is
        # named ``context_length``.
        "context_length": value["max_model_len"],
    }
    expected_secret = manifest.get("api_secret_sha256")
    if expected_secret is not None:
        if api_key is None:
            raise ValueError("runtime server requires its bound API key")
        from generate_full_quality_outputs_sglang import secret_sha256

        if secret_sha256(api_key) != expected_secret:
            raise ValueError("runtime server API key differs from its commitment")
        kwargs["api_key"] = api_key
    if manifest.get("runtime_mode") == "adapter":
        kwargs.update(
            {
                "lora_strict_loading": value["lora_strict_loading"],
                "lora_paths": value["lora_paths"],
                "max_lora_rank": value["max_lora_rank"],
                "lora_target_modules": value["lora_target_modules"],
            }
        )
    elif manifest.get("runtime_mode") != "base":
        raise ValueError("runtime_mode must be base or adapter")
    kwargs.update({field: value[field] for field in PASSTHROUGH_FIELDS if field in value})
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--test-checkpoint-ack")
    parser.add_argument("--gpu-lease-attestation", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.runtime_manifest, args.test_checkpoint_ack)
    official = validate_runtime_manifest(
        manifest,
        test_checkpoint_ack=args.test_checkpoint_ack,
    )
    api_key = read_api_secret(
        args.api_key_file,
        expected_sha256=manifest.get("api_secret_sha256"),
        required=official,
    )
    validate_live_environment(manifest, official=official)
    kwargs = build_server_kwargs(manifest, api_key=api_key)

    validate_gpu_lease_attestation(
        args.gpu_lease_attestation,
        runtime_manifest_sha256=file_sha256(args.runtime_manifest),
        gpu_ids=manifest["server_args"]["gpu_ids"],
    )

    import sglang
    import sglang.launch_server as launch_server_module
    import sglang.srt.server_args as server_args_module

    validate_sglang_import_origins(
        Path(manifest["sglang"]["checkout"]),
        {
            "sglang": sglang,
            "sglang.launch_server": launch_server_module,
            "sglang.srt.server_args": server_args_module,
        },
    )
    run_server = launch_server_module.run_server
    ServerArgs = server_args_module.ServerArgs
    run_server(ServerArgs(**kwargs))


if __name__ == "__main__":
    main()
