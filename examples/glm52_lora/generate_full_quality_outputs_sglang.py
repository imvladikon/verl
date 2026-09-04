#!/usr/bin/env python3
"""Generate exact-runtime paired GLM-5.2 quality outputs through SGLang.

The script is intentionally a client, not a server launcher. Separate base and
adapter runtime manifests bind the live endpoint to their exact artifacts. A
shared pair contract excludes adapter-specific and machine-local fields, but
binds the immutable model files, SGLang revision, server/decode semantics, and
held-out request contract. Official full-model manifests can produce quality
evidence; surgery manifests require an explicit test acknowledgement and
remain engineering evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from build_blind_quality_review import (
    canonical_sha256,
    file_sha256,
    is_sha256,
    read_contracts,
    read_jsonl,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = 3
OFFICIAL_TRAINER = (
    "zai-org/GLM-5.2",
    "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
)
OFFICIAL_INFERENCE_BASES = {
    OFFICIAL_TRAINER,
    (
        "zai-org/GLM-5.2-FP8",
        "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    ),
}
# A shard manifest is trusted only after its canonical SHA-256 is reviewed and
# pinned here.  The production map deliberately starts empty: aggregate byte
# counts and an index digest do not identify 755 GB / 1.5 TB of shard content.
# Runtime construction therefore remains quality=PENDING until independently
# generated per-shard manifests are pinned.  Tests inject tiny reviewed
# manifests rather than weakening this trust boundary.
TRUSTED_WEIGHT_SHARD_MANIFESTS: dict[tuple[str, str], str] = {}
APPROVED_SGLANG_RELEASES = {
    (
        "https://github.com/imvladikon/sglang",
        "0dbdb73509fbf6b3381359df87cde267d453c8d3",
        "5678fc2ab88fd65411b833c065f510b6d4f5d59c",
    )
}
TOKENIZER_ARTIFACTS = {
    "tokenizer_json_sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config_sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    "chat_template_sha256": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
}
OFFICIAL_MODEL_ARTIFACTS = {
    OFFICIAL_TRAINER: {
        "config_sha256": "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
        "weights_index_sha256": "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
        **TOKENIZER_ARTIFACTS,
        "weight_count": 59_585,
        "shard_count": 282,
        "index_total_size": 1_506_659_919_872,
        "shard_bytes_on_disk": 1_506_667_387_408,
    },
    ("zai-org/GLM-5.2-FP8", "f33c6dc501ee5a2c7e35155653b1b1abbc320951"): {
        "config_sha256": "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff",
        "weights_index_sha256": "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf",
        **TOKENIZER_ARTIFACTS,
        "weight_count": 118_629,
        "shard_count": 141,
        "index_total_size": 755_617_140_416,
        "shard_bytes_on_disk": 755_632_050_320,
    },
}
TEST_ACK = "nonofficial-checkpoint-output-is-not-quality-evidence"
OFFICIAL_STATUS = "EXACT-REVISION-SERVER-READY"
OFFICIAL_PENDING_STATUS = "OFFICIAL-QUALITY-PENDING-SHARD-IDENTITY"
TEST_STATUS = "TEST-CHECKPOINT-SERVER-READY"
MLA_TARGETS = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
}
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
RUNTIME_MODES = {"base", "adapter"}
ADAPTER_SERVER_FIELDS = {
    "lora_paths",
    "lora_strict_loading",
    "lora_target_modules",
    "max_lora_rank",
}
SERVER_ARGUMENT_FIELDS = {
    "attention_backend",
    "chunked_prefill_size",
    "disable_cuda_graph",
    "dsa_decode_backend",
    "dsa_prefill_backend",
    "dsa_topk_backend",
    "dtype",
    "enable_lora",
    "endpoint",
    "gpu_ids",
    "ld_library_paths",
    "load_format",
    "lora_paths",
    "lora_strict_loading",
    "lora_target_modules",
    "max_lora_rank",
    "max_model_len",
    "mem_fraction_static",
    "model_path",
    "moe_runner_backend",
    "quantization",
    "served_base_model",
    "tp_size",
    "watchdog_timeout",
}
RUNTIME_MANIFEST_FIELDS = {
    "adapter",
    "api_secret_sha256",
    "artifact_contract",
    "code_artifacts",
    "endpoint",
    "environment_artifacts",
    "inference_base",
    "local_artifacts",
    "pair_runtime_contract",
    "pair_runtime_contract_sha256",
    "runtime_mode",
    "schema_version",
    "served_base_model",
    "server_args",
    "server_args_sha256",
    "server_instance_id",
    "sglang",
    "status",
    "trainer_base",
    "weight_shard_identity",
}
PAIR_SERVER_SEMANTIC_FIELDS = {
    "attention_backend",
    "chunked_prefill_size",
    "disable_cuda_graph",
    "dsa_decode_backend",
    "dsa_prefill_backend",
    "dsa_topk_backend",
    "dtype",
    "load_format",
    "max_model_len",
    "mem_fraction_static",
    "moe_runner_backend",
    "quantization",
    "served_base_model",
    "tp_size",
    "watchdog_timeout",
}
MODEL_ARTIFACT_FIELDS = {
    "model_id",
    "revision",
    "revision_verified",
    "config_sha256",
    "weights_index_sha256",
    "tokenizer_json_sha256",
    "tokenizer_config_sha256",
    "chat_template_sha256",
    "weight_count",
    "shard_count",
    "index_total_size",
    "shard_bytes_on_disk",
}
HAN_EVALUATION_MODES = {
    "spontaneous",
    "input_conditioned_cleanup",
    "input_conditioned_scope_control",
    "excluded_han_allowed",
}
SGLANG_LIVE_CODE_PATHS = {
    "python/sglang/launch_server.py",
    "python/sglang/srt/server_args.py",
    "python/sglang/srt/models/glm4_moe.py",
    "python/sglang/srt/models/deepseek_v2.py",
    "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py",
    "python/sglang/srt/lora/lora_manager.py",
    "python/sglang/srt/lora/lora_registry.py",
}


class RequestFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Never forward prompts or authorization headers to a redirected URL."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _revision_block(value: Any, label: str) -> tuple[str, str]:
    block = _mapping(value, label)
    if set(block) != MODEL_ARTIFACT_FIELDS:
        raise ValueError(f"{label} artifact fields are invalid")
    model_id = _nonempty_string(block.get("model_id"), f"{label}.model_id")
    revision = block.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(f"{label}.revision must be a 40-character Git revision")
    if block.get("revision_verified") is not True:
        raise ValueError(f"{label}.revision_verified must be true")
    for field in (
        "config_sha256",
        "weights_index_sha256",
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
    ):
        if not is_sha256(block.get(field)) or block[field] == "0" * 64:
            raise ValueError(f"{label}.{field} must be a SHA-256 digest")
    for field in (
        "weight_count",
        "shard_count",
        "index_total_size",
        "shard_bytes_on_disk",
    ):
        field_value = block.get(field)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{label}.{field} must be a positive integer")
    return model_id, revision


def _validate_weight_shard_identity_block(
    value: Any,
    *,
    model: dict[str, Any],
    label: str,
) -> bool:
    block = _mapping(value, label)
    if set(block) != {
        "status",
        "model_id",
        "revision",
        "manifest_sha256",
        "local_verification_receipt_sha256",
        "verification_method",
        "shard_count",
        "shard_bytes_on_disk",
    }:
        raise ValueError(f"{label} fields are invalid")
    if block.get("model_id") != model["model_id"] or block.get("revision") != model["revision"]:
        raise ValueError(f"{label} model identity differs from the snapshot")
    if (
        block.get("shard_count") != model["shard_count"]
        or block.get("shard_bytes_on_disk") != model["shard_bytes_on_disk"]
    ):
        raise ValueError(f"{label} shard inventory differs from the snapshot")
    if block.get("verification_method") != ("trusted-sha256-manifest+full-read-once+stat-cache-v1"):
        raise ValueError(f"{label} verification method is invalid")
    status = block.get("status")
    if status not in {
        "PENDING-TRUSTED-MANIFEST",
        "PENDING-LOCAL-FULL-READ",
        "VERIFIED",
    }:
        raise ValueError(f"{label} status is invalid")
    manifest_sha256 = block.get("manifest_sha256")
    receipt_sha256 = block.get("local_verification_receipt_sha256")
    if status == "PENDING-TRUSTED-MANIFEST":
        if manifest_sha256 is not None or receipt_sha256 is not None:
            raise ValueError(f"{label} pending-manifest state contains proof digests")
        return False
    if not is_sha256(manifest_sha256) or manifest_sha256 == "0" * 64:
        raise ValueError(f"{label} manifest_sha256 is invalid")
    identity = model["model_id"], model["revision"]
    if TRUSTED_WEIGHT_SHARD_MANIFESTS.get(identity) != manifest_sha256:
        raise ValueError(f"{label} manifest is not in the reviewed trust allowlist")
    if status == "PENDING-LOCAL-FULL-READ":
        if receipt_sha256 is not None:
            raise ValueError(f"{label} pending local verification has a receipt")
        return False
    if not is_sha256(receipt_sha256) or receipt_sha256 == "0" * 64:
        raise ValueError(f"{label} local verification receipt is invalid")
    return True


def _validate_loopback_endpoint(endpoint: str, *, label: str = "endpoint") -> None:
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/chat/completions"
        or parsed.port is None
    ):
        raise ValueError(f"{label} must be a literal loopback HTTP chat-completions URL with a port")
    hostname = parsed.hostname
    try:
        loopback = ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(f"{label} must use a literal loopback IP address")


def secret_sha256(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def read_api_secret(
    path: Path | None,
    *,
    expected_sha256: str | None,
    required: bool,
) -> str | None:
    if path is None:
        if required or expected_sha256 is not None:
            raise ValueError("this runtime requires --api-key-file")
        return None
    if not path.is_file():
        raise FileNotFoundError(f"API key file does not exist: {path}")
    try:
        secret = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("API key file must be UTF-8 text") from error
    if len(secret.encode("utf-8")) < 32 or "\x00" in secret or "\n" in secret or "\r" in secret:
        raise ValueError("API key must be one nonempty line of at least 32 UTF-8 bytes")
    actual = secret_sha256(secret)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("API key differs from the runtime secret commitment")
    return secret


def _validate_adapter_block(
    value: Any,
    *,
    trainer_revision: str,
) -> dict[str, Any]:
    adapter = _mapping(value, "adapter")
    if set(adapter) != {
        "alpha",
        "artifact_sha256",
        "config_sha256",
        "name",
        "parameter_count",
        "profile",
        "rank",
        "target_modules",
        "trainer_base_revision",
        "verification_sha256",
        "tensor_count",
        "lora_b_tensor_count",
        "tensor_dtype",
        "topology_sha256",
        "tensor_validation_status",
    }:
        raise ValueError("adapter artifact fields are invalid")
    _nonempty_string(adapter.get("name"), "adapter.name")
    for field in ("artifact_sha256", "config_sha256", "verification_sha256"):
        if not is_sha256(adapter.get(field)) or adapter[field] == "0" * 64:
            raise ValueError(f"adapter.{field} must be a SHA-256 digest")
    if not is_sha256(adapter.get("topology_sha256")) or adapter["topology_sha256"] == "0" * 64:
        raise ValueError("adapter.topology_sha256 must be a SHA-256 digest")
    if adapter.get("trainer_base_revision") != trainer_revision:
        raise ValueError("adapter trainer revision differs from trainer_base")
    if adapter.get("rank") != 16 or adapter.get("alpha") != 32:
        raise ValueError("adapter must use the locked rank-16/alpha-32 contract")
    parameter_count = adapter.get("parameter_count")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise ValueError("adapter.parameter_count must be positive")
    for field in ("tensor_count", "lora_b_tensor_count"):
        count = adapter.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"adapter.{field} must be positive")
    if adapter["lora_b_tensor_count"] * 2 != adapter["tensor_count"]:
        raise ValueError("adapter tensor A/B cardinality is inconsistent")
    if (
        adapter.get("tensor_dtype") != "torch.bfloat16"
        or adapter.get("tensor_validation_status") != "FINITE-NONZERO-B-TOPOLOGY-VERIFIED"
    ):
        raise ValueError("adapter tensor validation contract is invalid")
    profile = adapter.get("profile")
    expected_targets = set(MLA_TARGETS)
    if profile == "mla-lm-head":
        expected_targets.add("lm_head")
    elif profile != "mla-only":
        raise ValueError("adapter.profile must be mla-only or mla-lm-head")
    targets = adapter.get("target_modules")
    if not isinstance(targets, list) or set(targets) != expected_targets:
        raise ValueError(f"adapter target modules differ from the locked {profile} profile")
    return adapter


def _validate_local_path(value: Any, label: str) -> str:
    path = Path(_nonempty_string(value, label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return str(path)


def _validate_code_and_environment_blocks(manifest: dict[str, Any]) -> None:
    code = _mapping(manifest.get("code_artifacts"), "code_artifacts")
    if set(code) != {"runtime_scripts"}:
        raise ValueError("code_artifacts fields are invalid")
    scripts = _mapping(code.get("runtime_scripts"), "code_artifacts.runtime_scripts")
    if set(scripts) != {
        "build_quality_sglang_runtime.py",
        "generate_full_quality_outputs_sglang.py",
        "launch_quality_sglang_server.py",
        "build_blind_quality_review.py",
    }:
        raise ValueError("runtime script artifact set is invalid")
    for name, value in scripts.items():
        block = _mapping(value, f"runtime script {name}")
        if set(block) != {"path", "sha256"}:
            raise ValueError(f"runtime script {name} fields are invalid")
        _validate_local_path(block.get("path"), f"runtime script {name}.path")
        if not is_sha256(block.get("sha256")) or block["sha256"] == "0" * 64:
            raise ValueError(f"runtime script {name}.sha256 is invalid")

    environment = _mapping(manifest.get("environment_artifacts"), "environment_artifacts")
    if set(environment) != {
        "python_executable",
        "python_executable_sha256",
        "python_version",
        "installed_distributions_sha256",
    }:
        raise ValueError("environment_artifacts fields are invalid")
    _validate_local_path(
        environment.get("python_executable"),
        "environment_artifacts.python_executable",
    )
    _nonempty_string(environment.get("python_version"), "environment_artifacts.python_version")
    for field in ("python_executable_sha256", "installed_distributions_sha256"):
        if not is_sha256(environment.get(field)) or environment[field] == "0" * 64:
            raise ValueError(f"environment_artifacts.{field} is invalid")


def build_pair_runtime_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the adapter-independent runtime semantics used by both variants."""
    server_args = _mapping(manifest.get("server_args"), "server_args")
    sglang = _mapping(manifest.get("sglang"), "sglang")
    code_artifacts = _mapping(manifest.get("code_artifacts"), "code_artifacts")
    runtime_scripts = _mapping(
        code_artifacts.get("runtime_scripts"),
        "code_artifacts.runtime_scripts",
    )
    environment = _mapping(
        manifest.get("environment_artifacts"),
        "environment_artifacts",
    )
    weight_shards = _mapping(
        manifest.get("weight_shard_identity"),
        "weight_shard_identity",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_contract": manifest.get("artifact_contract"),
        "weight_shard_manifest_sha256": {
            key: _mapping(weight_shards.get(key), f"weight_shard_identity.{key}").get("manifest_sha256")
            for key in ("trainer", "inference")
        },
        "sglang": {
            "repository": sglang.get("repository"),
            "revision": sglang.get("revision"),
            "tree": sglang.get("tree"),
        },
        "runtime_script_sha256": {name: block.get("sha256") for name, block in sorted(runtime_scripts.items())},
        "environment_semantics": {
            "python_version": environment.get("python_version"),
            "python_executable_sha256": environment.get("python_executable_sha256"),
            "installed_distributions_sha256": environment.get("installed_distributions_sha256"),
        },
        "server_semantics": {
            field: server_args[field] for field in sorted(PAIR_SERVER_SEMANTIC_FIELDS) if field in server_args
        },
    }


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_han_evaluation_mode(source: dict[str, Any], *, input_han_count: int) -> str:
    contract = _mapping(source.get("contract"), "held-out contract")
    allow_han = contract.get("allow_han")
    if not isinstance(allow_han, bool):
        raise TypeError("held-out contract.allow_han must be boolean")
    requested_language = _nonempty_string(
        contract.get("requested_language"),
        "held-out contract.requested_language",
    ).lower()
    language_root = requested_language.split("-", 1)[0].split("_", 1)[0]
    tags = source.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError("held-out tags must be a list of nonempty strings")
    tag_set = set(tags)
    cleanup = "han-cleanup" in tag_set
    scope_control = bool(tag_set.intersection({"accidental-han-control", "han-scope-control"}))
    allowed = allow_han or language_root in {"zh", "ja"}
    if cleanup and scope_control:
        raise ValueError("held-out row cannot be both Han cleanup and scope control")
    if allowed:
        if cleanup or scope_control:
            raise ValueError("Han-allowed row cannot also be cleanup or scope control")
        return "excluded_han_allowed"
    if cleanup:
        if input_han_count <= 0:
            raise ValueError("Han cleanup row must contain Han in its request")
        return "input_conditioned_cleanup"
    if scope_control:
        if input_han_count <= 0:
            raise ValueError("Han scope-control row must contain Han in its request")
        return "input_conditioned_scope_control"
    if input_han_count != 0:
        raise ValueError("Han-bearing request must be tagged as cleanup/scope-control or explicitly allow Han")
    return "spontaneous"


def evaluation_cluster_id(source: dict[str, Any]) -> str:
    provenance = _mapping(source.get("provenance"), "held-out provenance")
    source_text_sha256 = provenance.get("source_text_sha256")
    if provenance.get("dataset") == "wikimedia/wikipedia":
        if not is_sha256(source_text_sha256):
            raise ValueError("Wikipedia held-out row lacks source_text_sha256")
        return f"wikipedia-source:{source_text_sha256}"
    stable_provenance = {
        key: provenance[key] for key in ("dataset", "license", "revision", "source_split") if key in provenance
    }
    if not stable_provenance:
        raise ValueError("non-Wikipedia held-out row lacks stable provenance")
    cluster = {
        "reference_response_sha256": text_sha256(_nonempty_string(source.get("response"), "held-out response")),
        "provenance": stable_provenance,
    }
    return f"reference-provenance:{canonical_sha256(cluster)}"


def build_pair_contract(
    source: dict[str, Any],
    *,
    runtime: dict[str, Any],
    decoding: dict[str, Any],
) -> dict[str, Any]:
    """Bind shared runtime semantics to one immutable held-out request."""
    messages = request_messages(source)
    input_han_count = request_input_han_count(messages)
    han_mode = derive_han_evaluation_mode(source, input_han_count=input_han_count)
    if han_mode not in HAN_EVALUATION_MODES:  # defensive against future drift
        raise ValueError("unsupported Han evaluation mode")
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime["pair_runtime_contract"],
        "decoding": decoding,
        "held_out": {
            "id": source["id"],
            "split": source["split"],
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "source_row_sha256": canonical_sha256(source),
            "reference_response_sha256": text_sha256(source["response"]),
            "request_messages_sha256": canonical_sha256(messages),
            "input_han_count": input_han_count,
            "input_contains_han": input_han_count > 0,
            "han_evaluation_mode": han_mode,
            "evaluation_cluster_id": evaluation_cluster_id(source),
        },
    }


def require_runtime_mode(runtime: dict[str, Any], variant: str) -> None:
    if variant not in RUNTIME_MODES:
        raise ValueError(f"unsupported generation variant: {variant!r}")
    if runtime.get("runtime_mode") != variant:
        raise ValueError(f"{variant} generation requires a {variant} runtime manifest")


def request_messages(source: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": source["system"]},
        {"role": "user", "content": source["prompt"]},
    ]


def request_input_han_count(messages: list[dict[str, str]]) -> int:
    count = 0
    for message in messages:
        content = message["content"]
        for character in content:
            name = unicodedata.name(character, "")
            count += int("CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name)
    return count


def validate_runtime_manifest(
    manifest: Any,
    *,
    test_checkpoint_ack: str | None,
) -> bool:
    manifest = _mapping(manifest, "runtime manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported runtime manifest schema")
    if set(manifest) != RUNTIME_MANIFEST_FIELDS:
        raise ValueError("runtime manifest fields are invalid")
    trainer = _revision_block(manifest.get("trainer_base"), "trainer_base")
    inference = _revision_block(manifest.get("inference_base"), "inference_base")
    official_identity = trainer == OFFICIAL_TRAINER and inference in OFFICIAL_INFERENCE_BASES
    artifact_contract = _mapping(manifest.get("artifact_contract"), "artifact_contract")
    if artifact_contract != {
        "trainer_base": manifest["trainer_base"],
        "inference_base": manifest["inference_base"],
    }:
        raise ValueError("artifact_contract differs from the bound model artifacts")
    shard_identity = _mapping(manifest.get("weight_shard_identity"), "weight_shard_identity")
    if set(shard_identity) != {"trainer", "inference"}:
        raise ValueError("weight_shard_identity fields are invalid")
    trainer_shards_verified = _validate_weight_shard_identity_block(
        shard_identity.get("trainer"),
        model=manifest["trainer_base"],
        label="weight_shard_identity.trainer",
    )
    inference_shards_verified = _validate_weight_shard_identity_block(
        shard_identity.get("inference"),
        model=manifest["inference_base"],
        label="weight_shard_identity.inference",
    )
    shard_identity_verified = trainer_shards_verified and inference_shards_verified

    sglang = _mapping(manifest.get("sglang"), "sglang")
    if set(sglang) != {
        "checkout",
        "repository",
        "revision",
        "tree",
        "live_code_sha256",
    }:
        raise ValueError("sglang artifact fields are invalid")
    _validate_local_path(sglang.get("checkout"), "sglang.checkout")
    _nonempty_string(sglang.get("repository"), "sglang.repository")
    for field in ("revision", "tree"):
        digest = sglang.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 40
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"sglang.{field} must be a 40-character Git object ID")
    live_code = _mapping(sglang.get("live_code_sha256"), "sglang.live_code_sha256")
    if set(live_code) != SGLANG_LIVE_CODE_PATHS or any(
        not is_sha256(digest) or digest == "0" * 64 for digest in live_code.values()
    ):
        raise ValueError("sglang live source hash set is invalid")
    if official_identity and (sglang["repository"], sglang["revision"], sglang["tree"]) not in APPROVED_SGLANG_RELEASES:
        raise ValueError("official runtime SGLang repository/revision/tree is not approved")
    _validate_code_and_environment_blocks(manifest)
    local_artifacts = _mapping(manifest.get("local_artifacts"), "local_artifacts")
    required_local = {
        "trainer_model_path",
        "inference_model_path",
        "trainer_weight_shard_manifest",
        "inference_weight_shard_manifest",
        "weight_verification_cache_dir",
    }
    runtime_mode = manifest.get("runtime_mode")
    if runtime_mode == "adapter":
        required_local.update({"adapter_path", "adapter_verification_path"})
    if set(local_artifacts) != required_local:
        raise ValueError("local_artifacts fields differ from the runtime mode")
    for field in required_local:
        value = local_artifacts.get(field)
        if (
            field
            in {
                "trainer_weight_shard_manifest",
                "inference_weight_shard_manifest",
                "weight_verification_cache_dir",
            }
            and value is None
        ):
            continue
        _validate_local_path(value, f"local_artifacts.{field}")
    for key, local_field in (
        ("trainer", "trainer_weight_shard_manifest"),
        ("inference", "inference_weight_shard_manifest"),
    ):
        shard_status = shard_identity[key]["status"]
        manifest_path = local_artifacts[local_field]
        if shard_status == "PENDING-TRUSTED-MANIFEST" and manifest_path is not None:
            raise ValueError(f"{local_field} must be null while its manifest is pending")
        if shard_status != "PENDING-TRUSTED-MANIFEST" and manifest_path is None:
            raise ValueError(f"{local_field} is required for a bound shard manifest")
        if shard_status == "VERIFIED" and local_artifacts["weight_verification_cache_dir"] is None:
            raise ValueError("verified shard identity requires its local receipt cache")
    server_args = _mapping(manifest.get("server_args"), "server_args")
    if not set(server_args).issubset(SERVER_ARGUMENT_FIELDS):
        raise ValueError("server_args contains unsupported fields")
    if manifest.get("server_args_sha256") != canonical_sha256(server_args):
        raise ValueError("server_args_sha256 differs from the embedded server arguments")
    endpoint = _nonempty_string(manifest.get("endpoint"), "endpoint")
    _validate_loopback_endpoint(endpoint)
    _nonempty_string(manifest.get("server_instance_id"), "server_instance_id")
    served_base_model = _nonempty_string(
        manifest.get("served_base_model"),
        "served_base_model",
    )
    if server_args.get("endpoint") != endpoint:
        raise ValueError("server_args.endpoint differs from the runtime endpoint")
    if server_args.get("served_base_model") != served_base_model:
        raise ValueError("server_args.served_base_model differs from the runtime model")
    if (
        Path(str(server_args.get("model_path", ""))).resolve()
        != Path(local_artifacts["inference_model_path"]).resolve()
    ):
        raise ValueError("server_args.model_path differs from local inference artifact")
    tp_size = server_args.get("tp_size")
    max_model_len = server_args.get("max_model_len")
    gpu_ids = server_args.get("gpu_ids")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
        raise ValueError("server_args.tp_size must be positive")
    if (
        not isinstance(gpu_ids, list)
        or len(gpu_ids) != tp_size
        or len(set(gpu_ids)) != tp_size
        or any(isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids)
    ):
        raise ValueError("server_args.gpu_ids must contain one unique GPU per TP rank")
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 2048:
        raise ValueError("server_args.max_model_len must be at least 2048")

    if runtime_mode not in RUNTIME_MODES:
        raise ValueError("runtime_mode must be base or adapter")
    if runtime_mode == "base":
        if manifest.get("adapter") is not None:
            raise ValueError("base runtime must not bind an adapter")
        if server_args.get("enable_lora") is not False:
            raise ValueError("base runtime must disable LoRA")
        unexpected_adapter_fields = ADAPTER_SERVER_FIELDS.intersection(server_args)
        if unexpected_adapter_fields:
            raise ValueError(
                f"base runtime contains adapter-specific server arguments: {sorted(unexpected_adapter_fields)}"
            )
    else:
        adapter = _validate_adapter_block(
            manifest.get("adapter"),
            trainer_revision=trainer[1],
        )
        if server_args.get("enable_lora") is not True:
            raise ValueError("adapter runtime must enable LoRA")
        if server_args.get("lora_strict_loading") is not True:
            raise ValueError("adapter runtime must enable strict LoRA loading")
        lora_paths = server_args.get("lora_paths")
        if not isinstance(lora_paths, dict) or set(lora_paths) != {adapter["name"]}:
            raise ValueError("adapter runtime must load exactly its verified adapter")
        if Path(str(lora_paths[adapter["name"]])).resolve() != Path(local_artifacts["adapter_path"]).resolve():
            raise ValueError("adapter runtime LoRA path differs from the verified adapter")
        if server_args.get("max_lora_rank") != adapter["rank"]:
            raise ValueError("adapter runtime max LoRA rank differs from the adapter")
        if set(server_args.get("lora_target_modules") or []) != set(adapter["target_modules"]):
            raise ValueError("adapter runtime target modules differ from the adapter")

    pair_runtime_contract = _mapping(
        manifest.get("pair_runtime_contract"),
        "pair_runtime_contract",
    )
    if pair_runtime_contract != build_pair_runtime_contract(manifest):
        raise ValueError("pair_runtime_contract differs from the shared runtime semantics")
    if manifest.get("pair_runtime_contract_sha256") != canonical_sha256(pair_runtime_contract):
        raise ValueError("pair_runtime_contract_sha256 is invalid")

    quality_claim_allowed = official_identity and shard_identity_verified
    if official_identity and server_args.get("load_format", "auto") not in {
        "auto",
        "safetensors",
    }:
        raise ValueError("official quality runtime requires real safetensors weights")
    for identity, block in (
        (trainer, manifest["trainer_base"]),
        (inference, manifest["inference_base"]),
    ):
        expected_artifacts = OFFICIAL_MODEL_ARTIFACTS.get(identity)
        if expected_artifacts is not None:
            actual_artifacts = {field: block[field] for field in expected_artifacts}
            if actual_artifacts != expected_artifacts:
                raise ValueError(f"official model artifact contract mismatch for {identity[0]}")
    api_secret_digest = manifest.get("api_secret_sha256")
    if api_secret_digest is not None and (not is_sha256(api_secret_digest) or api_secret_digest == "0" * 64):
        raise ValueError("api_secret_sha256 must be a nonzero SHA-256 digest")
    if official_identity and api_secret_digest is None:
        raise ValueError("official runtime requires an API secret commitment")
    if quality_claim_allowed:
        expected_status = OFFICIAL_STATUS
    elif official_identity:
        expected_status = OFFICIAL_PENDING_STATUS
    else:
        expected_status = TEST_STATUS
    if manifest.get("status") != expected_status:
        raise ValueError(f"runtime status must be {expected_status}")
    if not quality_claim_allowed and test_checkpoint_ack != TEST_ACK:
        raise ValueError(f"quality-pending/nonofficial runtime requires --test-checkpoint-ack {TEST_ACK!r}")
    return quality_claim_allowed


def load_runtime_manifest(
    path: Path,
    *,
    test_checkpoint_ack: str | None,
) -> tuple[dict[str, Any], str, bool]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    official = validate_runtime_manifest(
        manifest,
        test_checkpoint_ack=test_checkpoint_ack,
    )
    return manifest, file_sha256(path), official


def decoding_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_completion_tokens,
        "seed": args.seed,
        "n": 1,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def request_payload(
    source: dict[str, Any],
    *,
    variant: str,
    runtime: dict[str, Any],
    decoding: dict[str, Any],
) -> dict[str, Any]:
    require_runtime_mode(runtime, variant)
    model = runtime["served_base_model"]
    payload = {
        "model": model,
        "messages": request_messages(source),
        **{key: value for key, value in decoding.items() if key != "schema_version"},
    }
    if variant == "adapter":
        adapter_name = runtime["adapter"]["name"]
        payload["model"] = adapter_name
        payload["lora_path"] = adapter_name
    return payload


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    _validate_loopback_endpoint(endpoint, label="request endpoint")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    # An empty ProxyHandler prevents HTTP(S)_PROXY from routing held-out data
    # off-host. The redirect handler prevents a loopback process from
    # forwarding the request body or bearer secret to any second endpoint.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read(2048).decode("utf-8", errors="replace")
        raise RequestFailure(
            f"HTTP {error.code}: {body}",
            retryable=error.code in RETRYABLE_HTTP_STATUS,
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RequestFailure(str(error), retryable=True) from error
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise RequestFailure("server returned non-JSON response", retryable=False) from error
    if not isinstance(decoded, dict):
        raise RequestFailure("server response must be an object", retryable=False)
    return decoded


def response_row(
    source: dict[str, Any],
    response: dict[str, Any],
    *,
    variant: str,
    runtime: dict[str, Any],
    runtime_sha256: str,
    official: bool,
    decoding: dict[str, Any],
) -> dict[str, Any]:
    require_runtime_mode(runtime, variant)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RequestFailure("response must contain exactly one choice", retryable=False)
    choice = _mapping(choices[0], "response choice")
    message = _mapping(choice.get("message"), "response message")
    completion = message.get("content")
    if not isinstance(completion, str) or not completion.strip():
        raise RequestFailure("server returned an empty visible completion", retryable=False)
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise RequestFailure(f"completion did not finish normally: {finish_reason!r}", retryable=False)
    expected_model = runtime["served_base_model"] if variant == "base" else runtime["adapter"]["name"]
    if response.get("model") != expected_model:
        raise RequestFailure("response model differs from the requested runtime variant", retryable=False)
    usage = _mapping(response.get("usage"), "response usage")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) or completion_tokens < 1:
        raise RequestFailure("response must report a positive completion token count", retryable=False)
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
        or isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise RequestFailure("response token accounting is invalid", retryable=False)
    messages = request_messages(source)
    input_han_count = request_input_han_count(messages)
    decoding_sha256 = canonical_sha256(decoding)
    pair_contract = build_pair_contract(source, runtime=runtime, decoding=decoding)
    held_out = pair_contract["held_out"]
    return {
        "id": source["id"],
        "split": source["split"],
        "completion": completion,
        "completion_token_count": completion_tokens,
        "contract": source["contract"],
        "prompt_sha256": source["prompt_sha256"],
        "source_row_sha256": held_out["source_row_sha256"],
        "reference_response_sha256": held_out["reference_response_sha256"],
        "request_messages_sha256": canonical_sha256(messages),
        "input_han_count": input_han_count,
        "input_contains_han": input_han_count > 0,
        "han_evaluation_mode": held_out["han_evaluation_mode"],
        "evaluation_cluster_id": held_out["evaluation_cluster_id"],
        "decoding_contract_sha256": decoding_sha256,
        "pair_contract": pair_contract,
        "pair_contract_sha256": canonical_sha256(pair_contract),
        "generation": {
            "variant": variant,
            "runtime_mode": runtime["runtime_mode"],
            "runtime_manifest_sha256": runtime_sha256,
            "pair_runtime_contract_sha256": runtime["pair_runtime_contract_sha256"],
            "quality_claim_allowed": official,
            "api_secret_sha256": runtime.get("api_secret_sha256"),
            "server_instance_id": runtime["server_instance_id"],
            "trainer_base": runtime["trainer_base"],
            "inference_base": runtime["inference_base"],
            "adapter": runtime["adapter"] if variant == "adapter" else None,
            "sglang": runtime["sglang"],
            "response_id": response.get("id"),
            "response_model": response.get("model"),
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def generate_one(
    source: dict[str, Any],
    *,
    variant: str,
    runtime: dict[str, Any],
    runtime_sha256: str,
    official: bool,
    decoding: dict[str, Any],
    endpoint: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    request_fn: Callable[..., dict[str, Any]] = post_json,
) -> dict[str, Any]:
    payload = request_payload(source, variant=variant, runtime=runtime, decoding=decoding)
    expected_secret = runtime.get("api_secret_sha256")
    if official and expected_secret is None:
        raise ValueError("official generation runtime lacks an API secret commitment")
    if expected_secret is not None and (api_key is None or secret_sha256(api_key) != expected_secret):
        raise ValueError("generation API key differs from the runtime secret commitment")
    last_error: RequestFailure | None = None
    for attempt in range(retries + 1):
        try:
            response = request_fn(endpoint, payload, timeout=timeout, api_key=api_key)
            return response_row(
                source,
                response,
                variant=variant,
                runtime=runtime,
                runtime_sha256=runtime_sha256,
                official=official,
                decoding=decoding,
            )
        except RequestFailure as error:
            last_error = error
            if not error.retryable or attempt == retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("request retry loop exhausted") from last_error


def validate_existing_rows(
    rows: list[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    *,
    variant: str,
    runtime: dict[str, Any],
    runtime_sha256: str,
    official: bool,
    decoding: dict[str, Any],
) -> set[str]:
    require_runtime_mode(runtime, variant)
    decoding_sha256 = canonical_sha256(decoding)
    expected_row_fields = {
        "id",
        "split",
        "completion",
        "completion_token_count",
        "contract",
        "prompt_sha256",
        "source_row_sha256",
        "reference_response_sha256",
        "request_messages_sha256",
        "input_han_count",
        "input_contains_han",
        "han_evaluation_mode",
        "evaluation_cluster_id",
        "decoding_contract_sha256",
        "pair_contract",
        "pair_contract_sha256",
        "generation",
    }
    expected_generation_fields = {
        "variant",
        "runtime_mode",
        "runtime_manifest_sha256",
        "pair_runtime_contract_sha256",
        "quality_claim_allowed",
        "api_secret_sha256",
        "server_instance_id",
        "trainer_base",
        "inference_base",
        "adapter",
        "sglang",
        "response_id",
        "response_model",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    seen: set[str] = set()
    for row in rows:
        if set(row) != expected_row_fields:
            raise ValueError("existing output row fields are invalid")
        example_id = str(row.get("id", "")).strip()
        if not example_id or example_id in seen:
            raise ValueError(f"existing output has missing or duplicate ID {example_id!r}")
        source = contracts.get(example_id)
        if source is None:
            raise ValueError(f"existing output ID {example_id!r} is outside the selected split")
        pair_contract = build_pair_contract(
            source,
            runtime=runtime,
            decoding=decoding,
        )
        held_out = pair_contract["held_out"]
        expected = {
            "split": source["split"],
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "source_row_sha256": held_out["source_row_sha256"],
            "reference_response_sha256": held_out["reference_response_sha256"],
            "request_messages_sha256": held_out["request_messages_sha256"],
            "input_han_count": held_out["input_han_count"],
            "input_contains_han": held_out["input_contains_han"],
            "han_evaluation_mode": held_out["han_evaluation_mode"],
            "evaluation_cluster_id": held_out["evaluation_cluster_id"],
            "decoding_contract_sha256": decoding_sha256,
            "pair_contract": pair_contract,
            "pair_contract_sha256": canonical_sha256(pair_contract),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{example_id}: existing {field} differs from the current run")
        generation = _mapping(row.get("generation"), f"{example_id}.generation")
        if set(generation) != expected_generation_fields:
            raise ValueError(f"{example_id}: existing generation fields are invalid")
        expected_generation = {
            "variant": variant,
            "runtime_mode": variant,
            "runtime_manifest_sha256": runtime_sha256,
            "pair_runtime_contract_sha256": runtime["pair_runtime_contract_sha256"],
            "quality_claim_allowed": official,
            "api_secret_sha256": runtime.get("api_secret_sha256"),
            "server_instance_id": runtime["server_instance_id"],
            "trainer_base": runtime["trainer_base"],
            "inference_base": runtime["inference_base"],
            "adapter": runtime["adapter"] if variant == "adapter" else None,
            "sglang": runtime["sglang"],
            "response_model": (runtime["served_base_model"] if variant == "base" else runtime["adapter"]["name"]),
            "finish_reason": "stop",
        }
        for field, value in expected_generation.items():
            if generation.get(field) != value:
                raise ValueError(f"{example_id}: existing generation {field} differs")
        if not isinstance(generation.get("response_id"), str) or not generation["response_id"].strip():
            raise ValueError(f"{example_id}: existing response ID is invalid")
        completion = row.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{example_id}: existing completion is empty")
        token_count = row.get("completion_token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
            raise ValueError(f"{example_id}: existing completion token count is invalid")
        prompt_tokens = generation.get("prompt_tokens")
        completion_tokens = generation.get("completion_tokens")
        total_tokens = generation.get("total_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 1
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens != token_count
            or isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens != prompt_tokens + completion_tokens
        ):
            raise ValueError(f"{example_id}: existing token accounting is invalid")
        seen.add(example_id)
    return seen


def _validate_endpoint(endpoint: str) -> None:
    _validate_loopback_endpoint(endpoint)


def contract_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"ordinal": ordinal, "filename": path.name, "sha256": file_sha256(path)} for ordinal, path in enumerate(paths)
    ]


def output_manifest_contract(
    *,
    status: str,
    count: int,
    output_sha256: str,
    variant: str,
    split: str,
    official: bool,
    runtime_sha256: str,
    runtime: dict[str, Any],
    decoding: dict[str, Any],
    contracts_paths: list[Path],
) -> dict[str, Any]:
    artifacts = contract_artifacts(contracts_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "variant": variant,
        "split": split,
        "count": count,
        "quality_claim_allowed": official,
        "runtime_manifest_sha256": runtime_sha256,
        "pair_runtime_contract_sha256": runtime["pair_runtime_contract_sha256"],
        "api_secret_sha256": runtime.get("api_secret_sha256"),
        "decoding_contract": decoding,
        "decoding_contract_sha256": canonical_sha256(decoding),
        "contract_artifacts": artifacts,
        "contract_artifacts_sha256": canonical_sha256(artifacts),
        "output_sha256": output_sha256,
    }


def validate_existing_output_manifest(
    manifest: Any,
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    variant: str,
    split: str,
    official: bool,
    runtime_sha256: str,
    runtime: dict[str, Any],
    decoding: dict[str, Any],
    contracts_paths: list[Path],
) -> None:
    manifest = _mapping(manifest, "existing output manifest")
    status = manifest.get("status")
    allowed_statuses = {
        "QUALITY-OUTPUTS-IN-PROGRESS" if official else "TEST-OUTPUTS-IN-PROGRESS",
        "QUALITY-OUTPUTS-COMPLETE" if official else "TEST-OUTPUTS-COMPLETE",
    }
    if status not in allowed_statuses:
        raise ValueError("existing output manifest status is invalid")
    expected = output_manifest_contract(
        status=status,
        count=len(rows),
        output_sha256=file_sha256(output_path),
        variant=variant,
        split=split,
        official=official,
        runtime_sha256=runtime_sha256,
        runtime=runtime,
        decoding=decoding,
        contracts_paths=contracts_paths,
    )
    if manifest != expected:
        raise ValueError("existing output manifest differs from current artifacts or run contract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contracts", nargs="+", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-checkpoint-ack")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_endpoint(args.endpoint)
    if not 1 <= args.concurrency <= 64:
        raise ValueError("concurrency must be in [1, 64]")
    if args.max_completion_tokens < 1 or args.request_timeout <= 0 or args.retries < 0:
        raise ValueError("token limit and timeout must be positive; retries must be nonnegative")
    output_exists = args.output.exists()
    manifest_exists = args.manifest.exists()
    if not args.resume and (output_exists or manifest_exists):
        raise FileExistsError("output/manifest exists; pass --resume to validate and continue")
    if args.resume and (not output_exists or not manifest_exists):
        raise FileNotFoundError("resume requires both the existing output and its manifest")

    contracts = read_contracts(args.contracts, split=args.split)
    runtime, runtime_sha256, official = load_runtime_manifest(
        args.runtime_manifest,
        test_checkpoint_ack=args.test_checkpoint_ack,
    )
    require_runtime_mode(runtime, args.variant)
    if args.endpoint.rstrip("/") != runtime["endpoint"].rstrip("/"):
        raise ValueError("endpoint differs from the hashed runtime manifest")
    decoding = decoding_contract(args)
    existing_rows = read_jsonl(args.output) if output_exists else []
    completed = validate_existing_rows(
        existing_rows,
        contracts,
        variant=args.variant,
        runtime=runtime,
        runtime_sha256=runtime_sha256,
        official=official,
        decoding=decoding,
    )
    if args.resume:
        existing_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate_existing_output_manifest(
            existing_manifest,
            output_path=args.output,
            rows=existing_rows,
            variant=args.variant,
            split=args.split,
            official=official,
            runtime_sha256=runtime_sha256,
            runtime=runtime,
            decoding=decoding,
            contracts_paths=args.contracts,
        )
    pending = [contracts[example_id] for example_id in sorted(contracts) if example_id not in completed]
    api_key = read_api_secret(
        args.api_key_file,
        expected_sha256=runtime.get("api_secret_sha256"),
        required=official,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    generate_one,
                    source,
                    variant=args.variant,
                    runtime=runtime,
                    runtime_sha256=runtime_sha256,
                    official=official,
                    decoding=decoding,
                    endpoint=args.endpoint,
                    api_key=api_key,
                    timeout=args.request_timeout,
                    retries=args.retries,
                ): source["id"]
                for source in pending
            }
            for index, future in enumerate(as_completed(futures), len(completed) + 1):
                example_id = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    for queued in futures:
                        queued.cancel()
                    raise RuntimeError(f"generation failed for {example_id}") from error
                with args.output.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    output.flush()
                current_rows = read_jsonl(args.output)
                progress_manifest = output_manifest_contract(
                    status=("QUALITY-OUTPUTS-IN-PROGRESS" if official else "TEST-OUTPUTS-IN-PROGRESS"),
                    count=len(current_rows),
                    output_sha256=file_sha256(args.output),
                    variant=args.variant,
                    split=args.split,
                    official=official,
                    runtime_sha256=runtime_sha256,
                    runtime=runtime,
                    decoding=decoding,
                    contracts_paths=args.contracts,
                )
                write_json(args.manifest, progress_manifest)
                print(
                    f"[{index}/{len(contracts)}] {example_id}",
                    file=sys.stderr,
                    flush=True,
                )

    rows = read_jsonl(args.output)
    completed = validate_existing_rows(
        rows,
        contracts,
        variant=args.variant,
        runtime=runtime,
        runtime_sha256=runtime_sha256,
        official=official,
        decoding=decoding,
    )
    if completed != set(contracts):
        raise RuntimeError("output does not cover the complete selected split")
    rows.sort(key=lambda row: row["id"])
    write_jsonl(args.output, rows)
    output_manifest = output_manifest_contract(
        status="QUALITY-OUTPUTS-COMPLETE" if official else "TEST-OUTPUTS-COMPLETE",
        count=len(rows),
        output_sha256=file_sha256(args.output),
        variant=args.variant,
        split=args.split,
        official=official,
        runtime_sha256=runtime_sha256,
        runtime=runtime,
        decoding=decoding,
        contracts_paths=args.contracts,
    )
    write_json(args.manifest, output_manifest)
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
