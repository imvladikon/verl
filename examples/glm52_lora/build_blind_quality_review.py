#!/usr/bin/env python3
"""Prepare and adjudicate blinded paired GLM-5.2 quality reviews.

Reviewers score meaning preservation, Russian naturalness, factual accuracy,
and instruction fulfillment without seeing which completion is the base or
adapter.  The blinding key is never written to an artifact; only its SHA-256
commitment is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import unicodedata
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

SCHEMA_VERSION = 3
METHOD = "blinded-human-rubric-v1"
RATING_FIELDS = (
    "meaning_preservation",
    "russian_naturalness",
    "factual_accuracy",
    "instruction_fulfillment",
)
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
OFFICIAL_MODEL_ARTIFACTS = {
    OFFICIAL_TRAINER: {
        "config_sha256": "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
        "weights_index_sha256": "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
        "tokenizer_json_sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer_config_sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
        "chat_template_sha256": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
        "weight_count": 59_585,
        "shard_count": 282,
        "index_total_size": 1_506_659_919_872,
        "shard_bytes_on_disk": 1_506_667_387_408,
    },
    ("zai-org/GLM-5.2-FP8", "f33c6dc501ee5a2c7e35155653b1b1abbc320951"): {
        "config_sha256": "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff",
        "weights_index_sha256": "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf",
        "tokenizer_json_sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer_config_sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
        "chat_template_sha256": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
        "weight_count": 118_629,
        "shard_count": 141,
        "index_total_size": 755_617_140_416,
        "shard_bytes_on_disk": 755_632_050_320,
    },
}
PREDICTION_FIELDS = {
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
GENERATION_FIELDS = {
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
ADAPTER_FIELDS = {
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
}
MLA_TARGETS = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
}
FULL_TRAINABLE_PARAMETERS = {
    "mla-only": 106_149_888,
    "mla-lm-head": 108_726_272,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_sha256(prompt: str) -> str:
    normalized = " ".join(prompt.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"ordinal": ordinal, "filename": path.name, "sha256": file_sha256(path)} for ordinal, path in enumerate(paths)
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def require_absent(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing}; pass --overwrite explicitly")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def jsonl_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return digest.hexdigest()


def index_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        example_id = str(row.get("id", "")).strip()
        if not example_id or example_id in indexed:
            raise ValueError(f"{label} row {row_number}: missing or duplicate ID {example_id!r}")
        indexed[example_id] = row
    return indexed


def read_contracts(paths: Iterable[Path], *, split: str) -> dict[str, dict[str, Any]]:
    rows = [row for path in paths for row in read_jsonl(path) if row.get("split") == split]
    if not rows:
        raise ValueError(f"no {split} contracts found")
    contracts = index_rows(rows, f"{split} contracts")
    for example_id, row in contracts.items():
        for field in ("system", "prompt", "response"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{example_id}: held-out {field} must be a nonempty string")
        if not isinstance(row.get("contract"), dict):
            raise TypeError(f"{example_id}: held-out contract must be an object")
        if row.get("prompt_sha256") != prompt_sha256(row["prompt"]):
            raise ValueError(f"{example_id}: held-out prompt_sha256 is invalid")
    return contracts


def read_blinding_key(path: Path) -> bytes:
    key = path.read_bytes().strip()
    if len(key) < 16:
        raise ValueError("blinding key must contain at least 16 bytes")
    return key


def _request_messages(source: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": source["system"]},
        {"role": "user", "content": source["prompt"]},
    ]


def _request_input_han_count(messages: list[dict[str, str]]) -> int:
    count = 0
    for message in messages:
        for character in message["content"]:
            name = unicodedata.name(character, "")
            count += int("CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name)
    return count


def _pair_is_flipped(key: bytes, example_id: str) -> bool:
    return bool(hmac.new(key, example_id.encode("utf-8"), hashlib.sha256).digest()[0] & 1)


def _full_model_identity(value: Any, label: str) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != MODEL_ARTIFACT_FIELDS:
        raise ValueError(f"{label} fields are invalid")
    if value.get("revision_verified") is not True:
        raise ValueError(f"{label} revision is not verified")
    revision = value.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(f"{label} revision is invalid")
    for field in (
        "config_sha256",
        "weights_index_sha256",
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
    ):
        if not is_sha256(value.get(field)) or value[field] == "0" * 64:
            raise ValueError(f"{label} {field} is invalid")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"{label} model_id is invalid")
    identity = (model_id, revision)
    expected = OFFICIAL_MODEL_ARTIFACTS.get(identity)
    if expected is None or any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise ValueError(f"{label} is not the locked official artifact contract")
    return identity


def _validate_pair(
    example_id: str,
    source: dict[str, Any],
    base: dict[str, Any],
    adapter: dict[str, Any],
) -> None:
    # Local import avoids a module cycle: the generation module reuses the
    # canonical hashing helpers defined in this file.
    from generate_full_quality_outputs_sglang import APPROVED_SGLANG_RELEASES

    for label, prediction in (("base", base), ("adapter", adapter)):
        if set(prediction) != PREDICTION_FIELDS:
            raise ValueError(f"{example_id}: {label} prediction fields are invalid")
    pair_contract = base.get("pair_contract")
    if not isinstance(pair_contract, dict):
        raise TypeError(f"{example_id}: base pair contract must be an object")
    if pair_contract != adapter.get("pair_contract"):
        raise ValueError(f"{example_id}: base and adapter pair contracts differ")
    pair_contract_sha256 = base.get("pair_contract_sha256")
    if not is_sha256(pair_contract_sha256):
        raise ValueError(f"{example_id}: base pair contract must have a SHA-256 digest")
    if pair_contract_sha256 != adapter.get("pair_contract_sha256"):
        raise ValueError(f"{example_id}: base and adapter pair contract hashes differ")
    if pair_contract_sha256 != canonical_sha256(pair_contract):
        raise ValueError(f"{example_id}: pair_contract_sha256 is invalid")
    if pair_contract.get("schema_version") != 3:
        raise ValueError(f"{example_id}: unsupported pair contract schema")
    pair_runtime = pair_contract.get("runtime")
    pair_decoding = pair_contract.get("decoding")
    pair_held_out = pair_contract.get("held_out")
    if not isinstance(pair_runtime, dict):
        raise TypeError(f"{example_id}: pair runtime contract must be an object")
    if set(pair_runtime) != {
        "schema_version",
        "artifact_contract",
        "weight_shard_manifest_sha256",
        "sglang",
        "runtime_script_sha256",
        "environment_semantics",
        "server_semantics",
    }:
        raise ValueError(f"{example_id}: pair runtime contract fields are invalid")
    if pair_runtime.get("schema_version") != 3:
        raise ValueError(f"{example_id}: unsupported pair runtime contract schema")
    shard_manifests = pair_runtime.get("weight_shard_manifest_sha256")
    if (
        not isinstance(shard_manifests, dict)
        or set(shard_manifests) != {"trainer", "inference"}
        or any(not is_sha256(value) for value in shard_manifests.values())
    ):
        raise ValueError(f"{example_id}: pair shard manifest hashes are invalid")
    artifact_contract = pair_runtime.get("artifact_contract")
    if not isinstance(artifact_contract, dict) or set(artifact_contract) != {
        "trainer_base",
        "inference_base",
    }:
        raise ValueError(f"{example_id}: pair artifact contract fields are invalid")
    trainer_identity = _full_model_identity(
        artifact_contract.get("trainer_base"),
        f"{example_id}: pair trainer base",
    )
    inference_identity = _full_model_identity(
        artifact_contract.get("inference_base"),
        f"{example_id}: pair inference base",
    )
    if trainer_identity != OFFICIAL_TRAINER or inference_identity not in OFFICIAL_INFERENCE_BASES:
        raise ValueError(f"{example_id}: pair runtime is not an official full-model oracle")
    sglang = pair_runtime.get("sglang")
    if not isinstance(sglang, dict) or set(sglang) != {
        "repository",
        "revision",
        "tree",
    }:
        raise ValueError(f"{example_id}: pair SGLang contract fields are invalid")
    if not isinstance(sglang.get("repository"), str) or not sglang["repository"].strip():
        raise ValueError(f"{example_id}: pair SGLang repository is invalid")
    for field in ("revision", "tree"):
        object_id = sglang.get(field)
        if (
            not isinstance(object_id, str)
            or len(object_id) != 40
            or any(character not in "0123456789abcdef" for character in object_id)
        ):
            raise ValueError(f"{example_id}: pair SGLang {field} is invalid")
    if (sglang["repository"], sglang["revision"], sglang["tree"]) not in (APPROVED_SGLANG_RELEASES):
        raise ValueError(f"{example_id}: pair SGLang release is not approved")
    runtime_scripts = pair_runtime.get("runtime_script_sha256")
    if not isinstance(runtime_scripts, dict) or set(runtime_scripts) != {
        "build_quality_sglang_runtime.py",
        "generate_full_quality_outputs_sglang.py",
        "launch_quality_sglang_server.py",
        "build_blind_quality_review.py",
    }:
        raise ValueError(f"{example_id}: pair runtime script fields are invalid")
    if any(not is_sha256(value) or value == "0" * 64 for value in runtime_scripts.values()):
        raise ValueError(f"{example_id}: pair runtime script digest is invalid")
    environment = pair_runtime.get("environment_semantics")
    if not isinstance(environment, dict) or set(environment) != {
        "python_version",
        "python_executable_sha256",
        "installed_distributions_sha256",
    }:
        raise ValueError(f"{example_id}: pair environment fields are invalid")
    if not isinstance(environment["python_version"], str) or not environment["python_version"]:
        raise ValueError(f"{example_id}: pair Python version is invalid")
    if any(
        not is_sha256(environment[field]) or environment[field] == "0" * 64
        for field in ("python_executable_sha256", "installed_distributions_sha256")
    ):
        raise ValueError(f"{example_id}: pair environment digest is invalid")
    server_semantics = pair_runtime.get("server_semantics")
    if not isinstance(server_semantics, dict):
        raise TypeError(f"{example_id}: pair server semantics must be an object")
    if not set(server_semantics).issubset(PAIR_SERVER_SEMANTIC_FIELDS):
        raise ValueError(f"{example_id}: pair server semantic fields are invalid")
    if not {"served_base_model", "tp_size", "max_model_len"}.issubset(server_semantics):
        raise ValueError(f"{example_id}: pair server semantics are incomplete")
    if not isinstance(pair_decoding, dict):
        raise TypeError(f"{example_id}: pair decoding contract must be an object")
    if pair_decoding.get("schema_version") != 3:
        raise ValueError(f"{example_id}: unsupported pair decoding contract schema")
    if not isinstance(pair_held_out, dict):
        raise TypeError(f"{example_id}: pair held-out contract must be an object")
    pair_runtime_sha256 = canonical_sha256(pair_runtime)
    expected_messages = _request_messages(source)
    expected_messages_hash = canonical_sha256(expected_messages)
    expected_input_han_count = _request_input_han_count(expected_messages)
    for label, prediction in (("base", base), ("adapter", adapter)):
        completion = prediction.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{example_id}: {label} completion must be a nonempty string")
        if prediction.get("contract") != source.get("contract"):
            raise ValueError(f"{example_id}: {label} quality contract differs from held-out source")
        if prediction.get("prompt_sha256") != source.get("prompt_sha256"):
            raise ValueError(f"{example_id}: {label} prompt hash differs from held-out source")
        if prediction.get("request_messages_sha256") != expected_messages_hash:
            raise ValueError(f"{example_id}: {label} request messages differ from held-out source")
        if prediction.get("input_han_count") != expected_input_han_count:
            raise ValueError(f"{example_id}: {label} input Han count differs from held-out source")
        if prediction.get("input_contains_han") is not (expected_input_han_count > 0):
            raise ValueError(f"{example_id}: {label} input Han flag differs from held-out source")
        generation = prediction.get("generation")
        if not isinstance(generation, dict):
            raise TypeError(f"{example_id}: {label} generation provenance must be an object")
        if set(generation) != GENERATION_FIELDS:
            raise ValueError(f"{example_id}: {label} generation fields are invalid")
        if generation.get("variant") != label:
            raise ValueError(f"{example_id}: {label} generation variant is invalid")
        if generation.get("runtime_mode") != label:
            raise ValueError(f"{example_id}: {label} runtime mode is invalid")
        if not is_sha256(generation.get("runtime_manifest_sha256")):
            raise ValueError(f"{example_id}: {label} runtime manifest hash is invalid")
        if generation.get("pair_runtime_contract_sha256") != pair_runtime_sha256:
            raise ValueError(f"{example_id}: {label} pair runtime contract is invalid")
        if generation.get("trainer_base") != artifact_contract["trainer_base"]:
            raise ValueError(f"{example_id}: {label} trainer base differs from the pair contract")
        if generation.get("inference_base") != artifact_contract["inference_base"]:
            raise ValueError(f"{example_id}: {label} inference base differs from the pair contract")
        if not is_sha256(generation.get("api_secret_sha256")):
            raise ValueError(f"{example_id}: {label} API secret commitment is invalid")
        generated_sglang = generation.get("sglang")
        if (
            not isinstance(generated_sglang, dict)
            or set(generated_sglang) != {"checkout", "repository", "revision", "tree"}
            or {key: generated_sglang[key] for key in sglang} != sglang
        ):
            raise ValueError(f"{example_id}: {label} SGLang differs from the pair contract")
        generated_adapter = generation.get("adapter")
        if label == "base" and generated_adapter is not None:
            raise ValueError(f"{example_id}: base generation unexpectedly binds an adapter")
        if label == "adapter":
            if not isinstance(generated_adapter, dict) or set(generated_adapter) != ADAPTER_FIELDS:
                raise ValueError(f"{example_id}: adapter generation lacks exact adapter provenance")
            expected_targets = set(MLA_TARGETS)
            if generated_adapter.get("profile") == "mla-lm-head":
                expected_targets.add("lm_head")
            elif generated_adapter.get("profile") != "mla-only":
                raise ValueError(f"{example_id}: adapter profile is invalid")
            if (
                generated_adapter.get("rank") != 16
                or generated_adapter.get("alpha") != 32
                or generated_adapter.get("trainer_base_revision") != trainer_identity[1]
                or set(generated_adapter.get("target_modules") or []) != expected_targets
                or generated_adapter.get("parameter_count") != FULL_TRAINABLE_PARAMETERS[generated_adapter["profile"]]
            ):
                raise ValueError(f"{example_id}: adapter contract is invalid")
            for field in (
                "artifact_sha256",
                "config_sha256",
                "verification_sha256",
            ):
                if not is_sha256(generated_adapter.get(field)):
                    raise ValueError(f"{example_id}: adapter {field} is invalid")
            parameter_count = generated_adapter.get("parameter_count")
            if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
                raise ValueError(f"{example_id}: adapter parameter count is invalid")
        if (
            not isinstance(generation.get("server_instance_id"), str)
            or not generation["server_instance_id"].strip()
            or not isinstance(generation.get("response_id"), str)
            or not generation["response_id"].strip()
            or not isinstance(generation.get("response_model"), str)
            or not generation["response_model"].strip()
            or generation.get("finish_reason") != "stop"
        ):
            raise ValueError(f"{example_id}: {label} response provenance is invalid")
        expected_response_model = (
            server_semantics["served_base_model"] if label == "base" else generated_adapter["name"]
        )
        if generation["response_model"] != expected_response_model:
            raise ValueError(f"{example_id}: {label} response model is invalid")
        prompt_tokens = generation.get("prompt_tokens")
        completion_tokens = generation.get("completion_tokens")
        total_tokens = generation.get("total_tokens")
        completion_token_count = prediction.get("completion_token_count")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 1
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens < 1
            or completion_tokens != completion_token_count
            or isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens != prompt_tokens + completion_tokens
        ):
            raise ValueError(f"{example_id}: {label} token accounting is invalid")
        if generation.get("quality_claim_allowed") is not True:
            raise ValueError(f"{example_id}: {label} runtime is not a full-model quality oracle")
    decoding_hash = base.get("decoding_contract_sha256")
    if not is_sha256(decoding_hash):
        raise ValueError(f"{example_id}: base decoding contract must be a SHA-256 digest")
    if decoding_hash != adapter.get("decoding_contract_sha256"):
        raise ValueError(f"{example_id}: base and adapter decoding contracts differ")
    if decoding_hash != canonical_sha256(pair_decoding):
        raise ValueError(f"{example_id}: pair decoding contract hash is invalid")
    from generate_full_quality_outputs_sglang import (
        derive_han_evaluation_mode,
        evaluation_cluster_id,
        text_sha256,
    )

    expected_han_mode = derive_han_evaluation_mode(
        source,
        input_han_count=expected_input_han_count,
    )
    expected_held_out = {
        "id": example_id,
        "split": source["split"],
        "contract": source["contract"],
        "prompt_sha256": source["prompt_sha256"],
        "source_row_sha256": canonical_sha256(source),
        "reference_response_sha256": text_sha256(source["response"]),
        "request_messages_sha256": expected_messages_hash,
        "input_han_count": expected_input_han_count,
        "input_contains_han": expected_input_han_count > 0,
        "han_evaluation_mode": expected_han_mode,
        "evaluation_cluster_id": evaluation_cluster_id(source),
    }
    if pair_held_out != expected_held_out:
        raise ValueError(f"{example_id}: pair held-out contract differs from source")
    for label, prediction in (("base", base), ("adapter", adapter)):
        for field in (
            "split",
            "source_row_sha256",
            "reference_response_sha256",
            "han_evaluation_mode",
            "evaluation_cluster_id",
        ):
            if prediction.get(field) != expected_held_out[field]:
                raise ValueError(f"{example_id}: {label} {field} differs from held-out source")


def _blank_review() -> dict[str, Any]:
    blank_ratings = {field: None for field in RATING_FIELDS}
    return {
        "reviewer": None,
        "candidate_a": {**blank_ratings, "severe_error": None, "notes": ""},
        "candidate_b": {**blank_ratings, "severe_error": None, "notes": ""},
    }


def build_packet(
    contracts: dict[str, dict[str, Any]],
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    *,
    blinding_key: bytes,
) -> list[dict[str, Any]]:
    base = index_rows(base_rows, "base predictions")
    adapter = index_rows(adapter_rows, "adapter predictions")
    if set(base) != set(adapter):
        raise ValueError("base and adapter prediction IDs differ")
    if set(base) != set(contracts):
        missing = sorted(set(contracts) - set(base))
        unexpected = sorted(set(base) - set(contracts))
        raise ValueError(
            f"predictions do not cover the complete held-out split: missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    packet: list[dict[str, Any]] = []
    for example_id in sorted(contracts):
        source = contracts[example_id]
        _validate_pair(example_id, source, base[example_id], adapter[example_id])
        completions = (
            base[example_id]["completion"],
            adapter[example_id]["completion"],
        )
        if _pair_is_flipped(blinding_key, example_id):
            completions = (completions[1], completions[0])
        item = {
            "schema_version": SCHEMA_VERSION,
            "id": example_id,
            "split": source["split"],
            "system": source["system"],
            "prompt": source["prompt"],
            "reference_response": source["response"],
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "request_messages_sha256": canonical_sha256(_request_messages(source)),
            "input_han_count": base[example_id]["input_han_count"],
            "input_contains_han": base[example_id]["input_contains_han"],
            "han_evaluation_mode": base[example_id]["han_evaluation_mode"],
            "evaluation_cluster_id": base[example_id]["evaluation_cluster_id"],
            "source_row_sha256": base[example_id]["source_row_sha256"],
            "reference_response_sha256": base[example_id]["reference_response_sha256"],
            "decoding_contract_sha256": base[example_id].get("decoding_contract_sha256"),
            "pair_contract_sha256": base[example_id]["pair_contract_sha256"],
            "candidate_a": completions[0],
            "candidate_b": completions[1],
            "rubric": {
                "scale": "integer 1 (unacceptable) through 5 (fully correct/natural)",
                "fields": list(RATING_FIELDS),
                "severe_error": "true for a harmful factual or meaning-changing failure",
            },
        }
        item["review_item_sha256"] = canonical_sha256(item)
        item["review"] = _blank_review()
        packet.append(item)
    return packet


def packet_manifest(
    packet: list[dict[str, Any]],
    *,
    key: bytes,
    contracts_paths: Iterable[Path],
    base_path: Path,
    adapter_path: Path,
    base_bundle: dict[str, Any],
    adapter_bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLINDED-REVIEW-PENDING",
        "method": METHOD,
        "count": len(packet),
        "blinding_key_sha256": hashlib.sha256(key).hexdigest(),
        "contract_artifacts": artifact_records(contracts_paths),
        "base_predictions_sha256": file_sha256(base_path),
        "adapter_predictions_sha256": file_sha256(adapter_path),
        "base_generation_bundle": base_bundle,
        "adapter_generation_bundle": adapter_bundle,
        "packet_sha256": jsonl_sha256(packet),
        "rubric": {
            "fields": list(RATING_FIELDS),
            "range": [1, 5],
            "score": "mean((rating - 1) / 4); capped at 0.25 for severe_error",
        },
    }


def validate_generation_bundle(
    *,
    predictions_path: Path,
    output_manifest_path: Path,
    runtime_manifest_path: Path,
    contracts_paths: list[Path],
    contracts: dict[str, dict[str, Any]],
    split: str,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate an immutable generation output through both source manifests."""

    from generate_full_quality_outputs_sglang import (
        load_runtime_manifest,
        validate_existing_output_manifest,
        validate_existing_rows,
    )

    runtime, runtime_sha256, official = load_runtime_manifest(
        runtime_manifest_path,
        test_checkpoint_ack=None,
    )
    if not official:
        raise ValueError("blind quality review requires an official runtime")
    if runtime.get("runtime_mode") != variant:
        raise ValueError(f"{variant} runtime manifest has the wrong mode")
    rows = read_jsonl(predictions_path)
    output_manifest = read_json(output_manifest_path, f"{variant} output manifest")
    if output_manifest.get("status") != "QUALITY-OUTPUTS-COMPLETE":
        raise ValueError(f"{variant} output manifest is not complete")
    decoding = output_manifest.get("decoding_contract")
    if not isinstance(decoding, dict):
        raise TypeError(f"{variant} decoding contract must be an object")
    seen = validate_existing_rows(
        rows,
        contracts,
        variant=variant,
        runtime=runtime,
        runtime_sha256=runtime_sha256,
        official=True,
        decoding=decoding,
    )
    if seen != set(contracts):
        raise ValueError(f"{variant} predictions do not cover the selected split")
    validate_existing_output_manifest(
        output_manifest,
        output_path=predictions_path,
        rows=rows,
        variant=variant,
        split=split,
        official=True,
        runtime_sha256=runtime_sha256,
        runtime=runtime,
        decoding=decoding,
        contracts_paths=contracts_paths,
    )
    return rows, {
        "predictions_sha256": file_sha256(predictions_path),
        "output_manifest_sha256": file_sha256(output_manifest_path),
        "runtime_manifest_sha256": runtime_sha256,
    }


def _static_review_item(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "review"}


def _validate_candidate_review(example_id: str, label: str, review: Any) -> tuple[float, dict[str, int]]:
    if not isinstance(review, dict):
        raise TypeError(f"{example_id}: {label} review must be an object")
    ratings: dict[str, int] = {}
    for field in RATING_FIELDS:
        value = review.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{example_id}: {label}.{field} must be an integer in [1, 5]")
        ratings[field] = value
    severe_error = review.get("severe_error")
    if not isinstance(severe_error, bool):
        raise ValueError(f"{example_id}: {label}.severe_error must be boolean")
    if not isinstance(review.get("notes", ""), str):
        raise TypeError(f"{example_id}: {label}.notes must be a string")
    score = fmean((value - 1) / 4 for value in ratings.values())
    if severe_error:
        score = min(score, 0.25)
    return score, ratings


def read_completed_reviews(
    paths: Iterable[Path],
    expected_packet: list[dict[str, Any]],
    *,
    minimum_reviewers: int,
) -> tuple[list[tuple[str, dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
    if minimum_reviewers < 1:
        raise ValueError("minimum_reviewers must be positive")
    expected = index_rows(expected_packet, "expected packet")
    completed: list[tuple[str, dict[str, dict[str, Any]]]] = []
    review_artifacts: list[dict[str, Any]] = []
    seen_reviewers: set[str] = set()
    for path in paths:
        rows = index_rows(read_jsonl(path), str(path))
        if set(rows) != set(expected):
            raise ValueError(f"{path}: review IDs differ from the prepared packet")
        reviewer_names: set[str] = set()
        for example_id, row in rows.items():
            expected_static = _static_review_item(expected[example_id])
            if _static_review_item(row) != expected_static:
                raise ValueError(f"{path}:{example_id}: blinded review item was modified")
            if row.get("review_item_sha256") != canonical_sha256(
                {key: value for key, value in expected_static.items() if key != "review_item_sha256"}
            ):
                raise ValueError(f"{path}:{example_id}: review item hash is invalid")
            review = row.get("review")
            if not isinstance(review, dict):
                raise TypeError(f"{path}:{example_id}: review must be an object")
            reviewer_value = review.get("reviewer")
            if not isinstance(reviewer_value, str) or not reviewer_value.strip():
                raise ValueError(f"{path}:{example_id}: nonempty reviewer identity is required")
            reviewer = reviewer_value.strip()
            reviewer_names.add(reviewer)
            _validate_candidate_review(example_id, "candidate_a", review.get("candidate_a"))
            _validate_candidate_review(example_id, "candidate_b", review.get("candidate_b"))
        if len(reviewer_names) != 1:
            raise ValueError(f"{path}: exactly one consistent reviewer identity is required")
        reviewer = reviewer_names.pop()
        if reviewer in seen_reviewers:
            raise ValueError(f"duplicate reviewer identity: {reviewer}")
        seen_reviewers.add(reviewer)
        completed.append((reviewer, rows))
        review_artifacts.append(
            {
                "ordinal": len(review_artifacts),
                "filename": path.name,
                "sha256": file_sha256(path),
            }
        )
    if len(completed) < minimum_reviewers:
        raise ValueError(f"at least {minimum_reviewers} distinct completed reviews are required; got {len(completed)}")
    return completed, review_artifacts


def adjudicate(
    contracts: dict[str, dict[str, Any]],
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    completed_reviews: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    blinding_key: bytes,
    review_hashes: list[dict[str, Any]],
    prepared_manifest_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    packet = build_packet(contracts, base_rows, adapter_rows, blinding_key=blinding_key)
    base = index_rows(deepcopy(base_rows), "base predictions")
    adapter = index_rows(deepcopy(adapter_rows), "adapter predictions")
    reviewers = sorted(reviewer for reviewer, _ in completed_reviews)
    packet_items_sha256 = canonical_sha256([_static_review_item(item) for item in packet])
    adjudication_contract = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "count": len(packet),
        "reviewers": reviewers,
        "review_artifacts": review_hashes,
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "packet_items_sha256": packet_items_sha256,
        "prepared_manifest_sha256": prepared_manifest_sha256,
    }
    adjudication_contract_sha256 = canonical_sha256(adjudication_contract)

    for item in packet:
        example_id = item["id"]
        flipped = _pair_is_flipped(blinding_key, example_id)
        base_label, adapter_label = ("candidate_b", "candidate_a") if flipped else ("candidate_a", "candidate_b")
        model_scores: dict[str, list[float]] = {"base": [], "adapter": []}
        component_scores: dict[str, dict[str, list[int]]] = {
            "base": {field: [] for field in RATING_FIELDS},
            "adapter": {field: [] for field in RATING_FIELDS},
        }
        for _, review_rows in completed_reviews:
            review = review_rows[example_id]["review"]
            for model_name, candidate_label in (
                ("base", base_label),
                ("adapter", adapter_label),
            ):
                score, ratings = _validate_candidate_review(
                    example_id,
                    candidate_label,
                    review[candidate_label],
                )
                model_scores[model_name].append(score)
                for field, value in ratings.items():
                    component_scores[model_name][field].append(value)

        for model_name, target in (
            ("base", base[example_id]),
            ("adapter", adapter[example_id]),
        ):
            target["semantic_score"] = fmean(model_scores[model_name])
            target["semantic_score_provenance"] = {
                "method": METHOD,
                "reviewers": reviewers,
                "review_artifacts": review_hashes,
                "review_scores": sorted(model_scores[model_name]),
                "mean_ratings": {field: fmean(values) for field, values in component_scores[model_name].items()},
                "completion_sha256": hashlib.sha256(target["completion"].encode("utf-8")).hexdigest(),
                "review_item_sha256": item["review_item_sha256"],
                "pair_contract_sha256": target["pair_contract_sha256"],
                "adjudication_contract_sha256": adjudication_contract_sha256,
            }

    base_scored = [base[row["id"]] for row in base_rows]
    adapter_scored = [adapter[row["id"]] for row in adapter_rows]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "BLINDED-REVIEW-COMPLETE",
        "method": METHOD,
        "count": len(base_scored),
        "reviewers": reviewers,
        "review_artifacts": review_hashes,
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "packet_items_sha256": packet_items_sha256,
        "adjudication_contract": adjudication_contract,
        "adjudication_contract_sha256": adjudication_contract_sha256,
        "base_semantic_score_mean": fmean(row["semantic_score"] for row in base_scored),
        "adapter_semantic_score_mean": fmean(row["semantic_score"] for row in adapter_scored),
    }
    return base_scored, adapter_scored, summary


def _prepare(args: argparse.Namespace) -> None:
    require_absent((args.packet, args.manifest), overwrite=args.overwrite)
    key = read_blinding_key(args.blinding_key_file)
    contracts = read_contracts(args.contracts, split=args.split)
    base_rows, base_bundle = validate_generation_bundle(
        predictions_path=args.base,
        output_manifest_path=args.base_output_manifest,
        runtime_manifest_path=args.base_runtime_manifest,
        contracts_paths=args.contracts,
        contracts=contracts,
        split=args.split,
        variant="base",
    )
    adapter_rows, adapter_bundle = validate_generation_bundle(
        predictions_path=args.adapter,
        output_manifest_path=args.adapter_output_manifest,
        runtime_manifest_path=args.adapter_runtime_manifest,
        contracts_paths=args.contracts,
        contracts=contracts,
        split=args.split,
        variant="adapter",
    )
    packet = build_packet(
        contracts,
        base_rows,
        adapter_rows,
        blinding_key=key,
    )
    write_jsonl(args.packet, packet)
    manifest = packet_manifest(
        packet,
        key=key,
        contracts_paths=args.contracts,
        base_path=args.base,
        adapter_path=args.adapter,
        base_bundle=base_bundle,
        adapter_bundle=adapter_bundle,
    )
    write_json(args.manifest, manifest)


def _adjudicate(args: argparse.Namespace) -> None:
    require_absent(
        (args.base_output, args.adapter_output, args.manifest),
        overwrite=args.overwrite,
    )
    key = read_blinding_key(args.blinding_key_file)
    contracts = read_contracts(args.contracts, split=args.split)
    base_rows, base_bundle = validate_generation_bundle(
        predictions_path=args.base,
        output_manifest_path=args.base_output_manifest,
        runtime_manifest_path=args.base_runtime_manifest,
        contracts_paths=args.contracts,
        contracts=contracts,
        split=args.split,
        variant="base",
    )
    adapter_rows, adapter_bundle = validate_generation_bundle(
        predictions_path=args.adapter,
        output_manifest_path=args.adapter_output_manifest,
        runtime_manifest_path=args.adapter_runtime_manifest,
        contracts_paths=args.contracts,
        contracts=contracts,
        split=args.split,
        variant="adapter",
    )
    packet = build_packet(contracts, base_rows, adapter_rows, blinding_key=key)
    prepared_manifest = read_json(args.prepared_manifest, "prepared review manifest")
    expected_prepared_manifest = packet_manifest(
        packet,
        key=key,
        contracts_paths=args.contracts,
        base_path=args.base,
        adapter_path=args.adapter,
        base_bundle=base_bundle,
        adapter_bundle=adapter_bundle,
    )
    if prepared_manifest != expected_prepared_manifest:
        raise ValueError("prepared review manifest differs from current artifacts")
    reviews, review_hashes = read_completed_reviews(
        args.review,
        packet,
        minimum_reviewers=args.minimum_reviewers,
    )
    base_scored, adapter_scored, manifest = adjudicate(
        contracts,
        base_rows,
        adapter_rows,
        reviews,
        blinding_key=key,
        review_hashes=review_hashes,
        prepared_manifest_sha256=file_sha256(args.prepared_manifest),
    )
    write_jsonl(args.base_output, base_scored)
    write_jsonl(args.adapter_output, adapter_scored)
    manifest.update(
        {
            "contract_artifacts": artifact_records(args.contracts),
            "base_input_sha256": file_sha256(args.base),
            "adapter_input_sha256": file_sha256(args.adapter),
            "base_output_sha256": file_sha256(args.base_output),
            "adapter_output_sha256": file_sha256(args.adapter_output),
        }
    )
    write_json(args.manifest, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build an unlabeled A/B review packet")
    prepare.add_argument("contracts", nargs="+", type=Path)
    prepare.add_argument("--split", choices=("validation", "test"), default="validation")
    prepare.add_argument("--base", type=Path, required=True)
    prepare.add_argument("--adapter", type=Path, required=True)
    prepare.add_argument("--base-output-manifest", type=Path, required=True)
    prepare.add_argument("--adapter-output-manifest", type=Path, required=True)
    prepare.add_argument("--base-runtime-manifest", type=Path, required=True)
    prepare.add_argument("--adapter-runtime-manifest", type=Path, required=True)
    prepare.add_argument("--blinding-key-file", type=Path, required=True)
    prepare.add_argument("--packet", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=_prepare)

    adjudicate_parser = subparsers.add_parser(
        "adjudicate",
        help="validate completed reviews and add paired semantic scores",
    )
    adjudicate_parser.add_argument("contracts", nargs="+", type=Path)
    adjudicate_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    adjudicate_parser.add_argument("--base", type=Path, required=True)
    adjudicate_parser.add_argument("--adapter", type=Path, required=True)
    adjudicate_parser.add_argument("--base-output-manifest", type=Path, required=True)
    adjudicate_parser.add_argument("--adapter-output-manifest", type=Path, required=True)
    adjudicate_parser.add_argument("--base-runtime-manifest", type=Path, required=True)
    adjudicate_parser.add_argument("--adapter-runtime-manifest", type=Path, required=True)
    adjudicate_parser.add_argument("--prepared-manifest", type=Path, required=True)
    adjudicate_parser.add_argument("--blinding-key-file", type=Path, required=True)
    adjudicate_parser.add_argument("--review", action="append", type=Path, required=True)
    adjudicate_parser.add_argument("--minimum-reviewers", type=int, default=2)
    adjudicate_parser.add_argument("--base-output", type=Path, required=True)
    adjudicate_parser.add_argument("--adapter-output", type=Path, required=True)
    adjudicate_parser.add_argument("--manifest", type=Path, required=True)
    adjudicate_parser.add_argument("--overwrite", action="store_true")
    adjudicate_parser.set_defaults(handler=_adjudicate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
