#!/usr/bin/env python3
"""Compare paired full-model base and adapter quality outputs fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any

from build_blind_quality_review import (
    METHOD as BLIND_REVIEW_METHOD,
)
from build_blind_quality_review import (
    PREDICTION_FIELDS,
    RATING_FIELDS,
    jsonl_sha256,
)
from build_blind_quality_review import (
    canonical_sha256 as review_canonical_sha256,
)
from evaluate_quality_outputs import (
    evaluate_rows,
    read_jsonl,
    semantic_provenance_pair_context,
    validate_semantic_score,
)
from generate_full_quality_outputs_sglang import (
    APPROVED_SGLANG_RELEASES,
    MODEL_ARTIFACT_FIELDS,
    OFFICIAL_MODEL_ARTIFACTS,
)
from quality_reward import contract_from_mapping

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
PAIR_CONTRACT_SCHEMA_VERSION = 3
BOOTSTRAP_METHOD = "paired-cluster-bootstrap-row-macro-mean"
DEFAULT_MINIMUM_SLICE_ROWS = 10
DEFAULT_MINIMUM_SLICE_CLUSTERS = 5
MARKDOWN_FAMILIES = ("list", "table", "code", "mixed")
KNOWN_RUSSIAN_DEFECT_ID_MARKERS = (
    "markdown-list",
    "markdown-table",
    "markdown-code",
    "markdown-mixed",
    "russian-style",
    "russian-latin-confusable-cleanup",
    "russian-case-period-restoration",
    "han-cleanup",
)
GENERAL_RUSSIAN_ID_MARKERS = ("general-russian",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        if not _is_sha256(value.get(field)) or value[field] == "0" * 64:
            raise ValueError(f"{label} {field} is invalid")
    for field in (
        "weight_count",
        "shard_count",
        "index_total_size",
        "shard_bytes_on_disk",
    ):
        field_value = value.get(field)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{label} {field} is invalid")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"{label} model_id is invalid")
    identity = model_id, revision
    expected_artifacts = OFFICIAL_MODEL_ARTIFACTS.get(identity)
    if expected_artifacts is None or any(
        value.get(field) != expected for field, expected in expected_artifacts.items()
    ):
        raise ValueError(f"{label} artifacts do not match the official revision")
    return identity


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    observations: list[tuple[str, float]], *, samples: int, seed: int
) -> dict[str, float | int | str] | None:
    """Bootstrap paired row deltas by resampling dependency clusters.

    Each draw samples the original number of clusters with replacement, then
    takes the row-level macro mean over all rows in the selected clusters.
    This keeps related variants together while preserving the estimand used by
    the comparator: an equally weighted mean over evaluation rows.
    """
    if not observations:
        return None
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    clusters: dict[str, list[float]] = {}
    for cluster_id, value in observations:
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            raise ValueError("evaluation_cluster_id must be a nonempty string")
        clusters.setdefault(cluster_id, []).append(value)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        sampled_values: list[float] = []
        for _ in cluster_ids:
            sampled_values.extend(clusters[cluster_ids[rng.randrange(len(cluster_ids))]])
        means.append(fmean(sampled_values))
    values = [value for _, value in observations]
    return {
        "method": BOOTSTRAP_METHOD,
        "row_count": len(observations),
        "cluster_count": len(cluster_ids),
        "macro_per_row_mean": fmean(values),
        "ci95_low": _quantile(means, 0.025),
        "ci95_high": _quantile(means, 0.975),
    }


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"{label} row {row_number}: missing id")
        if example_id in indexed:
            raise ValueError(f"{label}: duplicate id {example_id!r}")
        indexed[example_id] = row
    return indexed


def _require_equal_pair_contract(example_id: str, base: dict[str, Any], adapter: dict[str, Any]) -> tuple[str, str]:
    for label, prediction in (("base", base), ("adapter", adapter)):
        extra_fields = set(prediction) - PREDICTION_FIELDS
        if extra_fields in ({"semantic_score"}, {"semantic_score_provenance"}):
            raise ValueError(f"{example_id}: {label} semantic_score and semantic_score_provenance must appear together")
        if extra_fields not in (set(), {"semantic_score", "semantic_score_provenance"}):
            raise ValueError(f"{example_id}: {label} prediction fields are invalid: unexpected={sorted(extra_fields)}")
        missing_fields = PREDICTION_FIELDS - set(prediction)
        if missing_fields:
            raise ValueError(f"{example_id}: {label} prediction fields are invalid: missing={sorted(missing_fields)}")
    if base.get("contract") != adapter.get("contract"):
        raise ValueError(f"{example_id}: base and adapter contracts differ")
    for field in (
        "prompt_sha256",
        "source_row_sha256",
        "reference_response_sha256",
        "request_messages_sha256",
        "decoding_contract_sha256",
    ):
        base_value = base.get(field)
        adapter_value = adapter.get(field)
        if not _is_sha256(base_value) or not _is_sha256(adapter_value):
            raise ValueError(f"{example_id}: base and adapter {field} must be SHA-256 digests")
        if base_value != adapter_value:
            raise ValueError(f"{example_id}: base and adapter {field} differ")

    split = base.get("split")
    if split not in {"validation", "test"} or adapter.get("split") != split:
        raise ValueError(f"{example_id}: base and adapter split is invalid or differs")
    input_han_count = base.get("input_han_count")
    input_contains_han = base.get("input_contains_han")
    if isinstance(input_han_count, bool) or not isinstance(input_han_count, int) or input_han_count < 0:
        raise ValueError(f"{example_id}: input_han_count must be a nonnegative integer")
    if not isinstance(input_contains_han, bool):
        raise TypeError(f"{example_id}: input_contains_han must be boolean")
    if input_contains_han != (input_han_count > 0):
        raise ValueError(f"{example_id}: input Han count and flag disagree")
    han_mode = base.get("han_evaluation_mode")
    cluster_id = base.get("evaluation_cluster_id")
    for field in (
        "input_han_count",
        "input_contains_han",
        "han_evaluation_mode",
        "evaluation_cluster_id",
    ):
        if adapter.get(field) != base.get(field):
            raise ValueError(f"{example_id}: base and adapter {field} differ")
    if not isinstance(han_mode, str) or not han_mode:
        raise ValueError(f"{example_id}: han_evaluation_mode is invalid")
    if not isinstance(cluster_id, str) or not cluster_id.strip():
        raise ValueError(f"{example_id}: evaluation_cluster_id is invalid")

    pair_contract = base.get("pair_contract")
    if not isinstance(pair_contract, dict):
        raise TypeError(f"{example_id}: base pair contract must be an object")
    if pair_contract != adapter.get("pair_contract"):
        raise ValueError(f"{example_id}: base and adapter pair contracts differ")
    pair_contract_sha256 = base.get("pair_contract_sha256")
    if not _is_sha256(pair_contract_sha256) or not _is_sha256(adapter.get("pair_contract_sha256")):
        raise ValueError(f"{example_id}: pair_contract_sha256 must be a SHA-256 digest")
    if pair_contract_sha256 != adapter.get("pair_contract_sha256"):
        raise ValueError(f"{example_id}: base and adapter pair contract hashes differ")
    if pair_contract_sha256 != _canonical_sha256(pair_contract):
        raise ValueError(f"{example_id}: pair_contract_sha256 is invalid")
    if pair_contract.get("schema_version") != PAIR_CONTRACT_SCHEMA_VERSION:
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
    if pair_runtime.get("schema_version") != PAIR_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"{example_id}: unsupported pair runtime contract schema")
    shard_manifests = pair_runtime.get("weight_shard_manifest_sha256")
    if (
        not isinstance(shard_manifests, dict)
        or set(shard_manifests) != {"trainer", "inference"}
        or any(not _is_sha256(value) for value in shard_manifests.values())
    ):
        raise ValueError(f"{example_id}: pair shard manifest hashes are invalid")
    artifact_contract = pair_runtime.get("artifact_contract")
    if not isinstance(artifact_contract, dict) or set(artifact_contract) != {
        "trainer_base",
        "inference_base",
    }:
        raise ValueError(f"{example_id}: pair artifact contract fields are invalid")
    trainer_identity = _full_model_identity(artifact_contract.get("trainer_base"), f"{example_id}: pair trainer base")
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
    if any(not _is_sha256(value) or value == "0" * 64 for value in runtime_scripts.values()):
        raise ValueError(f"{example_id}: pair runtime script digest is invalid")
    environment = pair_runtime.get("environment_semantics")
    if not isinstance(environment, dict) or set(environment) != {
        "python_version",
        "python_executable_sha256",
        "installed_distributions_sha256",
    }:
        raise ValueError(f"{example_id}: pair environment fields are invalid")
    if not isinstance(environment["python_version"], str) or not environment["python_version"].strip():
        raise ValueError(f"{example_id}: pair Python version is invalid")
    if any(
        not _is_sha256(environment[field]) or environment[field] == "0" * 64
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
    if pair_decoding.get("schema_version") != PAIR_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"{example_id}: unsupported pair decoding contract schema")
    if base["decoding_contract_sha256"] != _canonical_sha256(pair_decoding):
        raise ValueError(f"{example_id}: pair decoding contract hash is invalid")
    if not isinstance(pair_held_out, dict):
        raise TypeError(f"{example_id}: pair held-out contract must be an object")
    expected_held_out = {
        "id": example_id,
        "split": split,
        "contract": base["contract"],
        "prompt_sha256": base["prompt_sha256"],
        "source_row_sha256": base["source_row_sha256"],
        "reference_response_sha256": base["reference_response_sha256"],
        "request_messages_sha256": base["request_messages_sha256"],
        "input_han_count": input_han_count,
        "input_contains_han": input_contains_han,
        "han_evaluation_mode": han_mode,
        "evaluation_cluster_id": cluster_id,
    }
    if pair_held_out != expected_held_out:
        raise ValueError(f"{example_id}: pair held-out contract differs from predictions")

    pair_runtime_sha256 = _canonical_sha256(pair_runtime)
    for label, prediction in (("base", base), ("adapter", adapter)):
        generation = prediction.get("generation")
        if not isinstance(generation, dict):
            raise TypeError(f"{example_id}: {label} generation provenance is missing")
        if generation.get("variant") != label:
            raise ValueError(f"{example_id}: {label} generation variant is invalid")
        if generation.get("runtime_mode") != label:
            raise ValueError(f"{example_id}: {label} runtime mode is invalid")
        if not _is_sha256(generation.get("runtime_manifest_sha256")):
            raise ValueError(f"{example_id}: {label} runtime manifest hash is invalid")
        if generation.get("pair_runtime_contract_sha256") != pair_runtime_sha256:
            raise ValueError(f"{example_id}: {label} pair runtime contract is invalid")
        if generation.get("trainer_base") != artifact_contract["trainer_base"]:
            raise ValueError(f"{example_id}: {label} trainer base differs from the pair contract")
        if generation.get("inference_base") != artifact_contract["inference_base"]:
            raise ValueError(f"{example_id}: {label} inference base differs from the pair contract")
        if not _is_sha256(generation.get("api_secret_sha256")):
            raise ValueError(f"{example_id}: {label} API secret commitment is invalid")
        generated_sglang = generation.get("sglang")
        if (
            not isinstance(generated_sglang, dict)
            or {key: generated_sglang.get(key) for key in ("repository", "revision", "tree")} != sglang
        ):
            raise ValueError(f"{example_id}: {label} SGLang differs from pair contract")
        generated_adapter = generation.get("adapter")
        if label == "base" and generated_adapter is not None:
            raise ValueError(f"{example_id}: base generation unexpectedly binds an adapter")
        if label == "adapter" and not isinstance(generated_adapter, dict):
            raise ValueError(f"{example_id}: adapter generation lacks adapter provenance")
        if generation.get("quality_claim_allowed") is not True:
            raise ValueError(f"{example_id}: {label} runtime is not a full-model quality oracle")

    base_score, base_semantic_provenance = validate_semantic_score(base, example_id)
    adapter_score, adapter_semantic_provenance = validate_semantic_score(adapter, example_id)
    if (base_score is None) != (adapter_score is None):
        raise ValueError(f"{example_id}: semantic score coverage is not paired")
    if base_semantic_provenance is not None:
        assert adapter_semantic_provenance is not None
        if semantic_provenance_pair_context(base_semantic_provenance) != semantic_provenance_pair_context(
            adapter_semantic_provenance
        ):
            raise ValueError(f"{example_id}: semantic score provenance differs across the pair")
    return cluster_id, han_mode


def validate_adjudication_artifacts(
    manifest_path: Path,
    prepared_manifest_path: Path,
    base_path: Path,
    adapter_path: Path,
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind scored rows back to generation outputs and the blinded packet."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(prepared, dict):
        raise TypeError("review manifests must be objects")
    expected_prepared_fields = {
        "schema_version",
        "status",
        "method",
        "count",
        "blinding_key_sha256",
        "contract_artifacts",
        "base_predictions_sha256",
        "adapter_predictions_sha256",
        "base_generation_bundle",
        "adapter_generation_bundle",
        "packet_sha256",
        "rubric",
    }
    if set(prepared) != expected_prepared_fields:
        raise ValueError("prepared review manifest fields are invalid")
    if (
        prepared.get("schema_version") != 3
        or prepared.get("status") != "BLINDED-REVIEW-PENDING"
        or prepared.get("method") != BLIND_REVIEW_METHOD
    ):
        raise ValueError("prepared review manifest status is invalid")
    if (
        isinstance(prepared.get("count"), bool)
        or not isinstance(prepared.get("count"), int)
        or prepared.get("count") != len(base_rows)
    ):
        raise ValueError("prepared review count differs from scored outputs")
    if not _is_sha256(prepared.get("blinding_key_sha256")) or not _is_sha256(prepared.get("packet_sha256")):
        raise ValueError("prepared review manifest digests are invalid")
    expected_rubric = {
        "fields": list(RATING_FIELDS),
        "range": [1, 5],
        "score": "mean((rating - 1) / 4); capped at 0.25 for severe_error",
    }
    if prepared.get("rubric") != expected_rubric:
        raise ValueError("prepared review rubric is invalid")

    def validate_records(value: Any, label: str, *, allow_empty: bool) -> None:
        if (
            not isinstance(value, list)
            or (not allow_empty and not value)
            or any(
                not isinstance(item, dict)
                or set(item) != {"ordinal", "filename", "sha256"}
                or item.get("ordinal") != ordinal
                or not isinstance(item.get("filename"), str)
                or not item["filename"]
                or not _is_sha256(item.get("sha256"))
                for ordinal, item in enumerate(value)
            )
        ):
            raise ValueError(f"{label} records are invalid")

    validate_records(
        prepared.get("contract_artifacts"),
        "prepared contract artifact",
        allow_empty=False,
    )
    for variant in ("base", "adapter"):
        bundle = prepared.get(f"{variant}_generation_bundle")
        if not isinstance(bundle, dict) or set(bundle) != {
            "predictions_sha256",
            "output_manifest_sha256",
            "runtime_manifest_sha256",
        }:
            raise ValueError(f"{variant} generation bundle fields are invalid")
        if any(not _is_sha256(value) for value in bundle.values()):
            raise ValueError(f"{variant} generation bundle digests are invalid")
        if bundle["predictions_sha256"] != prepared[f"{variant}_predictions_sha256"]:
            raise ValueError(f"{variant} generation bundle does not bind its predictions")
    expected_manifest_fields = {
        "schema_version",
        "status",
        "method",
        "count",
        "reviewers",
        "review_artifacts",
        "blinding_key_sha256",
        "packet_items_sha256",
        "adjudication_contract",
        "adjudication_contract_sha256",
        "base_semantic_score_mean",
        "adapter_semantic_score_mean",
        "contract_artifacts",
        "base_input_sha256",
        "adapter_input_sha256",
        "base_output_sha256",
        "adapter_output_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("adjudication manifest fields are invalid")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("status") != "BLINDED-REVIEW-COMPLETE"
        or manifest.get("method") != BLIND_REVIEW_METHOD
    ):
        raise ValueError("adjudication manifest status is invalid")
    validate_records(
        manifest.get("review_artifacts"),
        "adjudication review artifact",
        allow_empty=False,
    )
    validate_records(
        manifest.get("contract_artifacts"),
        "adjudication contract artifact",
        allow_empty=False,
    )
    if manifest["contract_artifacts"] != prepared["contract_artifacts"]:
        raise ValueError("prepared and adjudicated contract artifacts differ")
    if manifest.get("blinding_key_sha256") != prepared["blinding_key_sha256"]:
        raise ValueError("prepared and adjudicated blinding commitments differ")
    if manifest.get("base_output_sha256") != _sha256(base_path) or manifest.get("adapter_output_sha256") != _sha256(
        adapter_path
    ):
        raise ValueError("scored outputs differ from the adjudication manifest")
    if (
        isinstance(manifest.get("count"), bool)
        or not isinstance(manifest.get("count"), int)
        or manifest.get("count") != len(base_rows)
        or len(base_rows) != len(adapter_rows)
    ):
        raise ValueError("adjudication count differs from scored outputs")
    reviewers = manifest.get("reviewers")
    if (
        not isinstance(reviewers, list)
        or len(reviewers) < 2
        or reviewers != sorted(reviewers)
        or len(set(reviewers)) != len(reviewers)
        or any(not isinstance(reviewer, str) or not reviewer for reviewer in reviewers)
        or len(manifest["review_artifacts"]) != len(reviewers)
    ):
        raise ValueError("adjudication reviewers are invalid")
    adjudication_contract = manifest.get("adjudication_contract")
    if (
        not isinstance(adjudication_contract, dict)
        or set(adjudication_contract)
        != {
            "schema_version",
            "method",
            "count",
            "reviewers",
            "review_artifacts",
            "blinding_key_sha256",
            "packet_items_sha256",
            "prepared_manifest_sha256",
        }
        or manifest.get("adjudication_contract_sha256") != review_canonical_sha256(adjudication_contract)
    ):
        raise ValueError("adjudication contract hash is invalid")
    if adjudication_contract.get("prepared_manifest_sha256") != _sha256(prepared_manifest_path):
        raise ValueError("prepared review manifest is not the adjudicated packet")
    if (
        adjudication_contract.get("method") != manifest["method"]
        or adjudication_contract.get("count") != manifest["count"]
        or adjudication_contract.get("reviewers") != manifest["reviewers"]
        or adjudication_contract.get("review_artifacts") != manifest["review_artifacts"]
        or adjudication_contract.get("blinding_key_sha256") != manifest["blinding_key_sha256"]
        or adjudication_contract.get("packet_items_sha256") != manifest["packet_items_sha256"]
    ):
        raise ValueError("adjudication contract differs from its manifest")
    if prepared.get("base_predictions_sha256") != manifest.get("base_input_sha256") or prepared.get(
        "adapter_predictions_sha256"
    ) != manifest.get("adapter_input_sha256"):
        raise ValueError("prepared packet and adjudicated generation inputs differ")
    raw_base = [{key: value for key, value in row.items() if key in PREDICTION_FIELDS} for row in base_rows]
    raw_adapter = [{key: value for key, value in row.items() if key in PREDICTION_FIELDS} for row in adapter_rows]
    if (
        jsonl_sha256(raw_base) != manifest["base_input_sha256"]
        or jsonl_sha256(raw_adapter) != manifest["adapter_input_sha256"]
    ):
        raise ValueError("scored rows differ from the reviewed generation inputs")
    expected_contract_sha256 = manifest["adjudication_contract_sha256"]
    for label, rows in (("base", base_rows), ("adapter", adapter_rows)):
        for row in rows:
            provenance = row.get("semantic_score_provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("adjudication_contract_sha256") != expected_contract_sha256
            ):
                raise ValueError(f"{label} semantic score is not bound to this adjudication")
    if not math.isclose(
        fmean(float(row["semantic_score"]) for row in base_rows),
        float(manifest["base_semantic_score_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        fmean(float(row["semantic_score"]) for row in adapter_rows),
        float(manifest["adapter_semantic_score_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("semantic score means differ from adjudication")
    return manifest


def _metric_status(metric: dict[str, float | int | str] | None, *, regression_margin: float = 0.0) -> str:
    if metric is None:
        return "PENDING"
    mean = float(metric["macro_per_row_mean"])
    ci95_low = float(metric["ci95_low"])
    if mean < -regression_margin:
        return "FAIL"
    if mean > 0.0 and ci95_low >= -regression_margin:
        return "PASS"
    return "PENDING"


def _noninferiority_status(metric: dict[str, float | int | str] | None, *, margin: float) -> str:
    if metric is None:
        return "PENDING"
    ci95_low = float(metric["ci95_low"])
    ci95_high = float(metric["ci95_high"])
    if ci95_high < -margin:
        return "FAIL"
    if ci95_low >= -margin:
        return "PASS"
    return "PENDING"


def _markdown_family(example_id: str, contract: Any) -> str:
    """Return a stable family from fields already bound by the pair contract."""
    for family in MARKDOWN_FAMILIES:
        if f"markdown-{family}" in example_id:
            return family
    required_blocks = sorted(set(contract.required_blocks))
    if required_blocks:
        return "contract:" + "+".join(required_blocks)
    return "contract:markdown"


def _is_known_russian_defect_row(
    example_id: str,
    *,
    markdown_required: bool,
    base_markdown_valid: bool,
    base_russian_script_score: float,
    base_han_count: int,
    han_mode: str,
) -> bool:
    """Classify only from pair-bound identity/contract and pre-adapter output."""
    if any(marker in example_id for marker in KNOWN_RUSSIAN_DEFECT_ID_MARKERS):
        return True
    if any(marker in example_id for marker in GENERAL_RUSSIAN_ID_MARKERS):
        return False
    return (
        (markdown_required and not base_markdown_valid)
        or base_russian_script_score < 1.0
        or (han_mode in {"spontaneous", "input_conditioned_cleanup"} and base_han_count > 0)
    )


def _sample_size_status(row_clusters: list[str], *, minimum_rows: int, minimum_clusters: int) -> tuple[str, int]:
    cluster_count = len(set(row_clusters))
    status = "PASS" if len(row_clusters) >= minimum_rows and cluster_count >= minimum_clusters else "PENDING"
    return status, cluster_count


def _gate_slice_status(status: str, sample_size_status: str, row_count: int) -> str:
    # A directly observed regression remains a failure even when the slice is
    # small.  Conversely, a small slice can never establish PASS (or establish
    # that an expected defect was not reproduced).
    if status in {"FAIL", "NOT_APPLICABLE"} or row_count == 0:
        return status
    if sample_size_status != "PASS":
        return "PENDING"
    return status


def compare_rows(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 52,
    required_semantic_coverage: float = 1.0,
    semantic_noninferiority_margin: float = 0.02,
    required_retention_semantic_coverage: float = 1.0,
    retention_noninferiority_margin: float = 0.02,
    minimum_evaluation_rows: int = 20,
    minimum_evaluation_clusters: int = 10,
    minimum_slice_rows: int = DEFAULT_MINIMUM_SLICE_ROWS,
    minimum_slice_clusters: int = DEFAULT_MINIMUM_SLICE_CLUSTERS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for label, value in (
        ("minimum evaluation rows", minimum_evaluation_rows),
        ("minimum evaluation clusters", minimum_evaluation_clusters),
        ("minimum slice rows", minimum_slice_rows),
        ("minimum slice clusters", minimum_slice_clusters),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not 0.0 <= required_semantic_coverage <= 1.0:
        raise ValueError("required semantic coverage must be in [0, 1]")
    if not math.isfinite(semantic_noninferiority_margin) or semantic_noninferiority_margin < 0.0:
        raise ValueError("semantic noninferiority margin must be nonnegative")
    if not 0.0 <= required_retention_semantic_coverage <= 1.0:
        raise ValueError("required retention semantic coverage must be in [0, 1]")
    if not math.isfinite(retention_noninferiority_margin) or retention_noninferiority_margin < 0.0:
        raise ValueError("retention noninferiority margin must be nonnegative")

    base_index = _index_rows(base_rows, "base")
    adapter_index = _index_rows(adapter_rows, "adapter")
    base_ids = set(base_index)
    adapter_ids = set(adapter_index)
    if base_ids != adapter_ids:
        missing = sorted(base_ids - adapter_ids)
        unexpected = sorted(adapter_ids - base_ids)
        raise ValueError(f"base and adapter ids differ: missing={missing[:5]}, unexpected={unexpected[:5]}")

    pair_assignments = {
        example_id: _require_equal_pair_contract(example_id, base_index[example_id], adapter_index[example_id])
        for example_id in sorted(base_ids)
    }
    evaluation_cluster_count = len({cluster_id for cluster_id, _ in pair_assignments.values()})
    base_summary, base_details = evaluate_rows(base_rows)
    adapter_summary, adapter_details = evaluate_rows(adapter_rows)
    base_detail_index = {row["id"]: row for row in base_details}
    adapter_detail_index = {row["id"]: row for row in adapter_details}

    observations: dict[str, list[tuple[str, float]]] = {
        "constraint_score_macro_per_row_adapter_minus_base": [],
        "russian_script_score_macro_per_row_adapter_minus_base": [],
        "russian_semantic_score_macro_per_row_adapter_minus_base": [],
        "non_russian_semantic_score_macro_per_row_adapter_minus_base": [],
        "required_markdown_valid_macro_per_row_adapter_minus_base": [],
        "spontaneous_han_output_row_macro_per_row_base_minus_adapter": [],
        "spontaneous_han_character_count_macro_per_row_base_minus_adapter": [],
        "spontaneous_han_characters_per_1000_tokens_macro_per_row_base_minus_adapter": [],
        "input_conditioned_cleanup_success_macro_per_row_adapter_minus_base": [],
        "input_conditioned_cleanup_han_character_count_macro_per_row_base_minus_adapter": [],
        "input_conditioned_scope_control_han_output_row_macro_per_row_base_minus_adapter": [],
        "input_conditioned_scope_control_han_character_count_macro_per_row_base_minus_adapter": [],
    }
    paired_details: list[dict[str, Any]] = []
    russian_count = 0
    retention_count = 0
    base_markdown_defects = 0
    adapter_markdown_defects = 0
    russian_slice_rows: dict[str, list[str]] = {
        "known_defect_correction": [],
        "general_noninferiority": [],
    }
    russian_slice_observations: dict[str, list[tuple[str, float]]] = {key: [] for key in russian_slice_rows}
    retention_slice_rows: dict[str, list[str]] = {
        "all_non_russian": [],
        "legitimate_chinese": [],
    }
    retention_slice_observations: dict[str, list[tuple[str, float]]] = {key: [] for key in retention_slice_rows}
    markdown_family_rows: dict[str, list[str]] = {}
    markdown_family_observations: dict[str, list[tuple[str, float]]] = {}
    markdown_family_defects: dict[str, dict[str, int]] = {}
    cohort_rows = {
        "spontaneous": 0,
        "input_conditioned_cleanup": 0,
        "input_conditioned_scope_control": 0,
        "excluded_han_allowed": 0,
    }
    base_han_rows = {key: 0 for key in cohort_rows}
    adapter_han_rows = {key: 0 for key in cohort_rows}
    base_han_characters = {key: 0 for key in cohort_rows}
    adapter_han_characters = {key: 0 for key in cohort_rows}

    def add(metric: str, cluster_id: str, value: float) -> None:
        observations[metric].append((cluster_id, value))

    for example_id in sorted(base_ids):
        base = base_index[example_id]
        adapter = adapter_index[example_id]
        cluster_id, han_mode = pair_assignments[example_id]
        contract = contract_from_mapping(base.get("contract") or {})
        base_detail = base_detail_index[example_id]
        adapter_detail = adapter_detail_index[example_id]
        cohort_rows[han_mode] += 1

        markdown_required = bool(contract.require_markdown or contract.required_blocks)
        base_markdown_valid = bool(base_detail["markdown_valid"])
        adapter_markdown_valid = bool(adapter_detail["markdown_valid"])
        base_han = int(base_detail["output_han_count"])
        adapter_han = int(adapter_detail["output_han_count"])

        constraint_delta = float(adapter_detail["constraint"]["score"]) - float(base_detail["constraint"]["score"])
        add(
            "constraint_score_macro_per_row_adapter_minus_base",
            cluster_id,
            constraint_delta,
        )
        detail: dict[str, Any] = {
            "id": example_id,
            "evaluation_cluster_id": cluster_id,
            "han_evaluation_mode": han_mode,
            "constraint_score_adapter_minus_base": constraint_delta,
        }

        language_root = contract.requested_language.split("-", 1)[0].split("_", 1)[0]
        if language_root == "ru":
            russian_count += 1
            russian_slice = (
                "known_defect_correction"
                if _is_known_russian_defect_row(
                    example_id,
                    markdown_required=markdown_required,
                    base_markdown_valid=base_markdown_valid,
                    base_russian_script_score=float(base_detail["constraint"]["russian_script_score"]),
                    base_han_count=base_han,
                    han_mode=han_mode,
                )
                else "general_noninferiority"
            )
            russian_slice_rows[russian_slice].append(cluster_id)
            detail["russian_semantic_slice"] = russian_slice
            script_delta = float(adapter_detail["constraint"]["russian_script_score"]) - float(
                base_detail["constraint"]["russian_script_score"]
            )
            add(
                "russian_script_score_macro_per_row_adapter_minus_base",
                cluster_id,
                script_delta,
            )
            detail["russian_script_score_adapter_minus_base"] = script_delta
            if base.get("semantic_score") is not None:
                semantic_delta = float(adapter["semantic_score"]) - float(base["semantic_score"])
                add(
                    "russian_semantic_score_macro_per_row_adapter_minus_base",
                    cluster_id,
                    semantic_delta,
                )
                russian_slice_observations[russian_slice].append((cluster_id, semantic_delta))
                detail["semantic_score_adapter_minus_base"] = semantic_delta
        else:
            retention_count += 1
            retention_slice_rows["all_non_russian"].append(cluster_id)
            retention_slices = ["all_non_russian"]
            if language_root == "zh" and contract.allow_han:
                retention_slice_rows["legitimate_chinese"].append(cluster_id)
                retention_slices.append("legitimate_chinese")
            detail["retention_slices"] = retention_slices
            if base.get("semantic_score") is not None:
                retention_delta = float(adapter["semantic_score"]) - float(base["semantic_score"])
                add(
                    "non_russian_semantic_score_macro_per_row_adapter_minus_base",
                    cluster_id,
                    retention_delta,
                )
                for retention_slice in retention_slices:
                    retention_slice_observations[retention_slice].append((cluster_id, retention_delta))
                detail["retention_semantic_score_adapter_minus_base"] = retention_delta

        if markdown_required:
            base_markdown_defects += int(not base_markdown_valid)
            adapter_markdown_defects += int(not adapter_markdown_valid)
            markdown_delta = float(adapter_markdown_valid) - float(base_markdown_valid)
            add(
                "required_markdown_valid_macro_per_row_adapter_minus_base",
                cluster_id,
                markdown_delta,
            )
            markdown_family = _markdown_family(example_id, contract)
            markdown_family_rows.setdefault(markdown_family, []).append(cluster_id)
            markdown_family_observations.setdefault(markdown_family, []).append((cluster_id, markdown_delta))
            family_defects = markdown_family_defects.setdefault(markdown_family, {"base": 0, "adapter": 0})
            family_defects["base"] += int(not base_markdown_valid)
            family_defects["adapter"] += int(not adapter_markdown_valid)
            detail["markdown_family"] = markdown_family
            detail["markdown_valid_adapter_minus_base"] = markdown_delta

        base_han_rows[han_mode] += int(base_han > 0)
        adapter_han_rows[han_mode] += int(adapter_han > 0)
        base_han_characters[han_mode] += base_han
        adapter_han_characters[han_mode] += adapter_han
        if han_mode == "spontaneous":
            event_delta = float(base_han > 0) - float(adapter_han > 0)
            count_delta = float(base_han - adapter_han)
            add(
                "spontaneous_han_output_row_macro_per_row_base_minus_adapter",
                cluster_id,
                event_delta,
            )
            add(
                "spontaneous_han_character_count_macro_per_row_base_minus_adapter",
                cluster_id,
                count_delta,
            )
            detail["spontaneous_han_output_row_base_minus_adapter"] = event_delta
            detail["spontaneous_han_character_count_base_minus_adapter"] = count_delta
            base_tokens = base_detail["completion_token_count"]
            adapter_tokens = adapter_detail["completion_token_count"]
            if base_tokens is not None and adapter_tokens is not None:
                token_delta = 1000.0 * (base_han / base_tokens - adapter_han / adapter_tokens)
                add(
                    "spontaneous_han_characters_per_1000_tokens_macro_per_row_base_minus_adapter",
                    cluster_id,
                    token_delta,
                )
                detail["spontaneous_han_characters_per_1000_tokens_base_minus_adapter"] = token_delta
        elif han_mode == "input_conditioned_cleanup":
            cleanup_delta = float(adapter_han == 0) - float(base_han == 0)
            count_delta = float(base_han - adapter_han)
            add(
                "input_conditioned_cleanup_success_macro_per_row_adapter_minus_base",
                cluster_id,
                cleanup_delta,
            )
            add(
                "input_conditioned_cleanup_han_character_count_macro_per_row_base_minus_adapter",
                cluster_id,
                count_delta,
            )
            detail["cleanup_success_adapter_minus_base"] = cleanup_delta
            detail["cleanup_han_character_count_base_minus_adapter"] = count_delta
        elif han_mode == "input_conditioned_scope_control":
            event_delta = float(base_han > 0) - float(adapter_han > 0)
            count_delta = float(base_han - adapter_han)
            add(
                "input_conditioned_scope_control_han_output_row_macro_per_row_base_minus_adapter",
                cluster_id,
                event_delta,
            )
            add(
                ("input_conditioned_scope_control_han_character_count_macro_per_row_base_minus_adapter"),
                cluster_id,
                count_delta,
            )
            detail["scope_control_han_output_row_base_minus_adapter"] = event_delta
            detail["scope_control_han_character_count_base_minus_adapter"] = count_delta
        paired_details.append(detail)

    metric_seed = bootstrap_seed

    def bootstrap(
        values: list[tuple[str, float]],
    ) -> dict[str, float | int | str] | None:
        nonlocal metric_seed
        result = paired_cluster_bootstrap(values, samples=bootstrap_samples, seed=metric_seed)
        metric_seed += 1
        return result

    metrics = {name: bootstrap(values) for name, values in observations.items()}
    russian_slice_metrics = {name: bootstrap(russian_slice_observations[name]) for name in russian_slice_rows}
    retention_slice_metrics = {name: bootstrap(retention_slice_observations[name]) for name in retention_slice_rows}
    markdown_family_metrics = {
        name: bootstrap(markdown_family_observations[name]) for name in sorted(markdown_family_rows)
    }
    spontaneous_token_metric = "spontaneous_han_characters_per_1000_tokens_macro_per_row_base_minus_adapter"
    spontaneous_token_paired_rows = len(observations[spontaneous_token_metric])
    if spontaneous_token_paired_rows != cohort_rows["spontaneous"]:
        # Never publish a per-row token-normalized comparison from a partial
        # spontaneous cohort; a missing token denominator can select results.
        metrics[spontaneous_token_metric] = None
    semantic_observations = observations["russian_semantic_score_macro_per_row_adapter_minus_base"]
    semantic_coverage = len(semantic_observations) / russian_count if russian_count else 0.0
    russian_slice_status: dict[str, dict[str, Any]] = {}
    for name, row_clusters in russian_slice_rows.items():
        observations_for_slice = russian_slice_observations[name]
        coverage = len(observations_for_slice) / len(row_clusters) if row_clusters else 0.0
        if name == "known_defect_correction":
            provisional_status = (
                _metric_status(
                    russian_slice_metrics[name],
                    regression_margin=semantic_noninferiority_margin,
                )
                if row_clusters
                else "NOT_REPRODUCED"
            )
        else:
            provisional_status = (
                _noninferiority_status(
                    russian_slice_metrics[name],
                    margin=semantic_noninferiority_margin,
                )
                if row_clusters
                else "PENDING"
            )
        if row_clusters and coverage < required_semantic_coverage:
            provisional_status = "PENDING"
        slice_size_status, cluster_count = _sample_size_status(
            row_clusters,
            minimum_rows=minimum_slice_rows,
            minimum_clusters=minimum_slice_clusters,
        )
        russian_slice_status[name] = {
            "status": _gate_slice_status(provisional_status, slice_size_status, len(row_clusters)),
            "row_count": len(row_clusters),
            "cluster_count": cluster_count,
            "minimum_rows": minimum_slice_rows,
            "minimum_clusters": minimum_slice_clusters,
            "sample_size_status": slice_size_status,
            "paired_semantic_count": len(observations_for_slice),
            "paired_semantic_coverage": coverage,
            "metric": russian_slice_metrics[name],
        }
    russian_component_values = {item["status"] for item in russian_slice_status.values()}
    if "FAIL" in russian_component_values:
        semantic_status = "FAIL"
    elif russian_component_values == {"PASS"}:
        semantic_status = "PASS"
    else:
        semantic_status = "PENDING"

    retention_observations = observations["non_russian_semantic_score_macro_per_row_adapter_minus_base"]
    retention_semantic_coverage = len(retention_observations) / retention_count if retention_count else 0.0
    retention_slice_status: dict[str, dict[str, Any]] = {}
    for name, row_clusters in retention_slice_rows.items():
        observations_for_slice = retention_slice_observations[name]
        coverage = len(observations_for_slice) / len(row_clusters) if row_clusters else 0.0
        provisional_status = (
            _noninferiority_status(
                retention_slice_metrics[name],
                margin=retention_noninferiority_margin,
            )
            if row_clusters
            else "NOT_APPLICABLE"
        )
        if row_clusters and coverage < required_retention_semantic_coverage:
            provisional_status = "PENDING"
        slice_size_status, cluster_count = _sample_size_status(
            row_clusters,
            minimum_rows=minimum_slice_rows,
            minimum_clusters=minimum_slice_clusters,
        )
        retention_slice_status[name] = {
            "status": _gate_slice_status(provisional_status, slice_size_status, len(row_clusters)),
            "row_count": len(row_clusters),
            "cluster_count": cluster_count,
            "minimum_rows": minimum_slice_rows,
            "minimum_clusters": minimum_slice_clusters,
            "sample_size_status": slice_size_status,
            "paired_semantic_count": len(observations_for_slice),
            "paired_semantic_coverage": coverage,
            "metric": retention_slice_metrics[name],
        }
    applicable_retention_statuses = {
        item["status"] for item in retention_slice_status.values() if item["status"] != "NOT_APPLICABLE"
    }
    if not applicable_retention_statuses:
        retention_status = "NOT_APPLICABLE"
    elif "FAIL" in applicable_retention_statuses:
        retention_status = "FAIL"
    elif "PENDING" in applicable_retention_statuses:
        retention_status = "PENDING"
    else:
        retention_status = "PASS"

    markdown_metric = metrics["required_markdown_valid_macro_per_row_adapter_minus_base"]
    if adapter_markdown_defects > base_markdown_defects:
        markdown_status = "FAIL"
    elif base_markdown_defects == 0:
        markdown_status = "NOT_REPRODUCED"
    else:
        markdown_status = _metric_status(markdown_metric)
    markdown_family_status: dict[str, dict[str, Any]] = {}
    for name in sorted(markdown_family_rows):
        row_clusters = markdown_family_rows[name]
        defects = markdown_family_defects[name]
        if defects["adapter"] > defects["base"]:
            family_status = "FAIL"
        elif defects["base"] == 0:
            family_status = "NOT_REPRODUCED"
        else:
            family_status = _metric_status(markdown_family_metrics[name])
        slice_size_status, cluster_count = _sample_size_status(
            row_clusters,
            minimum_rows=minimum_slice_rows,
            minimum_clusters=minimum_slice_clusters,
        )
        family_status = _gate_slice_status(family_status, slice_size_status, len(row_clusters))
        markdown_family_status[name] = {
            "status": family_status,
            "row_count": len(row_clusters),
            "cluster_count": cluster_count,
            "minimum_rows": minimum_slice_rows,
            "minimum_clusters": minimum_slice_clusters,
            "sample_size_status": slice_size_status,
            "base_defects": defects["base"],
            "adapter_defects": defects["adapter"],
            "metric": markdown_family_metrics[name],
        }
    family_statuses = {item["status"] for item in markdown_family_status.values()}
    if "FAIL" in family_statuses:
        markdown_status = "FAIL"
    elif "PENDING" in family_statuses:
        markdown_status = "PENDING"

    spontaneous_metric = metrics["spontaneous_han_output_row_macro_per_row_base_minus_adapter"]
    spontaneous_rows = cohort_rows["spontaneous"]
    base_spontaneous_han = base_han_rows["spontaneous"]
    adapter_spontaneous_han = adapter_han_rows["spontaneous"]
    spontaneous_character_metric = metrics["spontaneous_han_character_count_macro_per_row_base_minus_adapter"]
    if spontaneous_rows == 0:
        han_status = "NOT_APPLICABLE"
    elif (
        adapter_spontaneous_han > base_spontaneous_han
        or adapter_han_characters["spontaneous"] > base_han_characters["spontaneous"]
    ):
        han_status = "FAIL"
    elif base_spontaneous_han == 0:
        han_status = "NOT_REPRODUCED"
    else:
        han_status = _metric_status(spontaneous_metric)
        character_status = _noninferiority_status(
            spontaneous_character_metric,
            margin=0.0,
        )
        if character_status == "FAIL":
            han_status = "FAIL"
        elif han_status == "PASS" and character_status != "PASS":
            han_status = "PENDING"

    cleanup_rows = cohort_rows["input_conditioned_cleanup"]
    if cleanup_rows == 0:
        cleanup_status = "NOT_APPLICABLE"
    elif (
        adapter_han_rows["input_conditioned_cleanup"] > base_han_rows["input_conditioned_cleanup"]
        or adapter_han_characters["input_conditioned_cleanup"] > base_han_characters["input_conditioned_cleanup"]
    ):
        cleanup_status = "FAIL"
    else:
        cleanup_status = _noninferiority_status(
            metrics["input_conditioned_cleanup_success_macro_per_row_adapter_minus_base"],
            margin=0.0,
        )
        cleanup_character_status = _noninferiority_status(
            metrics["input_conditioned_cleanup_han_character_count_macro_per_row_base_minus_adapter"],
            margin=0.0,
        )
        if cleanup_character_status == "FAIL":
            cleanup_status = "FAIL"
        elif cleanup_status == "PASS" and cleanup_character_status != "PASS":
            cleanup_status = "PENDING"

    scope_rows = cohort_rows["input_conditioned_scope_control"]
    if scope_rows == 0:
        scope_status = "NOT_APPLICABLE"
    elif (
        adapter_han_rows["input_conditioned_scope_control"] > base_han_rows["input_conditioned_scope_control"]
        or adapter_han_characters["input_conditioned_scope_control"]
        > base_han_characters["input_conditioned_scope_control"]
    ):
        scope_status = "FAIL"
    else:
        scope_status = _noninferiority_status(
            metrics["input_conditioned_scope_control_han_character_count_macro_per_row_base_minus_adapter"],
            margin=0.0,
        )

    han_component_status = {
        "spontaneous": han_status,
        "input_conditioned_cleanup": cleanup_status,
        "input_conditioned_scope_control": scope_status,
    }
    if "FAIL" in han_component_status.values():
        han_status = "FAIL"
    elif han_status == "PASS" and any(status == "PENDING" for status in han_component_status.values()):
        han_status = "PENDING"

    target_status = {
        "russian_semantic_quality": semantic_status,
        "required_markdown_validity": markdown_status,
        "accidental_han": han_status,
        "non_russian_semantic_retention": retention_status,
    }
    if "FAIL" in target_status.values():
        overall_status = "FAIL"
    elif all(status in {"PASS", "NOT_APPLICABLE"} for status in target_status.values()):
        overall_status = "PASS"
    else:
        overall_status = "PENDING"
    sample_size_gate = (
        len(base_ids) >= minimum_evaluation_rows and evaluation_cluster_count >= minimum_evaluation_clusters
    )
    if overall_status == "PASS" and not sample_size_gate:
        overall_status = "PENDING"

    result = {
        "status": overall_status,
        "target_status": target_status,
        "count": len(base_ids),
        "evaluation_cluster_count": evaluation_cluster_count,
        "minimum_evaluation_rows": minimum_evaluation_rows,
        "minimum_evaluation_clusters": minimum_evaluation_clusters,
        "sample_size_gate": "PASS" if sample_size_gate else "PENDING",
        "minimum_slice_rows": minimum_slice_rows,
        "minimum_slice_clusters": minimum_slice_clusters,
        "russian_example_count": russian_count,
        "paired_russian_semantic_count": len(semantic_observations),
        "paired_russian_semantic_coverage": semantic_coverage,
        "required_semantic_coverage": required_semantic_coverage,
        "semantic_noninferiority_margin": semantic_noninferiority_margin,
        "russian_semantic_slice_status": russian_slice_status,
        "retention_example_count": retention_count,
        "paired_retention_semantic_count": len(retention_observations),
        "paired_retention_semantic_coverage": retention_semantic_coverage,
        "required_retention_semantic_coverage": required_retention_semantic_coverage,
        "retention_noninferiority_margin": retention_noninferiority_margin,
        "retention_slice_status": retention_slice_status,
        "base_required_markdown_defects": base_markdown_defects,
        "adapter_required_markdown_defects": adapter_markdown_defects,
        "required_markdown_family_status": markdown_family_status,
        "han_cohort_row_counts": cohort_rows,
        "base_han_output_rows_by_cohort": base_han_rows,
        "adapter_han_output_rows_by_cohort": adapter_han_rows,
        "base_han_character_counts_by_cohort": base_han_characters,
        "adapter_han_character_counts_by_cohort": adapter_han_characters,
        "han_component_status": han_component_status,
        "base_spontaneous_han_output_rows": base_spontaneous_han,
        "adapter_spontaneous_han_output_rows": adapter_spontaneous_han,
        "paired_spontaneous_completion_token_count": (spontaneous_token_paired_rows),
        "paired_spontaneous_completion_token_coverage": (
            spontaneous_token_paired_rows / spontaneous_rows if spontaneous_rows else None
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_method": BOOTSTRAP_METHOD,
        "metric_definitions": {
            "*_macro_per_row_*": (
                "equally weighted paired row deltas; confidence intervals "
                "resample evaluation_cluster_id clusters with replacement"
            ),
            "evaluator_token_micro_rates": (
                "separate cohort-level ratios of summed Han characters to "
                "summed completion tokens; never used as these row-macro metrics"
            ),
        },
        "metrics": metrics,
        "base_summary": base_summary,
        "adapter_summary": adapter_summary,
    }
    return result, paired_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_predictions", type=Path)
    parser.add_argument("adapter_predictions", type=Path)
    parser.add_argument("--adjudication-manifest", type=Path, required=True)
    parser.add_argument("--prepared-review-manifest", type=Path, required=True)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=52)
    parser.add_argument("--required-semantic-coverage", type=float, default=1.0)
    parser.add_argument("--semantic-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--required-retention-semantic-coverage", type=float, default=1.0)
    parser.add_argument("--retention-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--minimum-slice-rows", type=int, default=DEFAULT_MINIMUM_SLICE_ROWS)
    parser.add_argument("--minimum-slice-clusters", type=int, default=DEFAULT_MINIMUM_SLICE_CLUSTERS)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_predictions)
    adapter_rows = read_jsonl(args.adapter_predictions)
    adjudication = validate_adjudication_artifacts(
        args.adjudication_manifest,
        args.prepared_review_manifest,
        args.base_predictions,
        args.adapter_predictions,
        base_rows,
        adapter_rows,
    )
    result, details = compare_rows(
        base_rows,
        adapter_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        required_semantic_coverage=args.required_semantic_coverage,
        semantic_noninferiority_margin=args.semantic_noninferiority_margin,
        required_retention_semantic_coverage=(args.required_retention_semantic_coverage),
        retention_noninferiority_margin=args.retention_noninferiority_margin,
        minimum_evaluation_rows=20,
        minimum_evaluation_clusters=10,
        minimum_slice_rows=args.minimum_slice_rows,
        minimum_slice_clusters=args.minimum_slice_clusters,
    )
    result["base_predictions_sha256"] = _sha256(args.base_predictions)
    result["adapter_predictions_sha256"] = _sha256(args.adapter_predictions)
    result["adjudication_manifest_sha256"] = _sha256(args.adjudication_manifest)
    result["adjudication_contract_sha256"] = adjudication["adjudication_contract_sha256"]
    if args.details is not None:
        args.details.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details))
        result["details_sha256"] = _sha256(args.details)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
