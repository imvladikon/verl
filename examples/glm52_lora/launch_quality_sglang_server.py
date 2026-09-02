#!/usr/bin/env python3
"""Launch the SGLang endpoint bound by a GLM-5.2 runtime manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from generate_full_quality_outputs_sglang import validate_runtime_manifest

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


def read_manifest(path: Path, test_checkpoint_ack: str | None) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_runtime_manifest(manifest, test_checkpoint_ack=test_checkpoint_ack)
    return manifest


def git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_live_environment(manifest: dict[str, Any]) -> None:
    checkout = Path(manifest["sglang"]["checkout"]).resolve()
    if git_head(checkout) != manifest["sglang"]["revision"]:
        raise ValueError("live SGLang checkout revision differs from the runtime manifest")
    tracked = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tracked:
        raise ValueError("live SGLang checkout has tracked modifications")
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


def build_server_kwargs(manifest: dict[str, Any]) -> dict[str, Any]:
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
        "lora_strict_loading": value["lora_strict_loading"],
        "lora_paths": value["lora_paths"],
        "max_lora_rank": value["max_lora_rank"],
        "lora_target_modules": value["lora_target_modules"],
        # ``--max-model-len`` is a CLI alias; the ServerArgs dataclass field is
        # named ``context_length``.
        "context_length": value["max_model_len"],
    }
    kwargs.update({field: value[field] for field in PASSTHROUGH_FIELDS if field in value})
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--test-checkpoint-ack")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.runtime_manifest, args.test_checkpoint_ack)
    validate_live_environment(manifest)
    kwargs = build_server_kwargs(manifest)

    from sglang.launch_server import run_server
    from sglang.srt.server_args import ServerArgs

    run_server(ServerArgs(**kwargs))


if __name__ == "__main__":
    main()
