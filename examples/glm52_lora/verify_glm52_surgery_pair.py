#!/usr/bin/env python3
"""Verify qualified GLM-5.2 surgery plans and optional built checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from glm52_surgery_io import DTYPE_BYTES, read_header, sha256_file

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|[*])$")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def plan_entries(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {entry["target_name"]: entry for entry in plan["tensors"]}
    if len(entries) != len(plan["tensors"]):
        raise ValueError("duplicate target tensor in plan")
    return entries


def logical_entries(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: entry for name, entry in plan_entries(plan).items() if not name.endswith("_scale_inv")}


def verify_pair_contract(bf16_plan: dict[str, Any], fp8_plan: dict[str, Any]) -> None:
    """Verify pair equivalence without importing a source-discovery planner."""
    if bf16_plan.get("pair_id") != fp8_plan.get("pair_id"):
        raise ValueError("plans have different pair ids")
    if bf16_plan.get("expert_selection") != fp8_plan.get("expert_selection"):
        raise ValueError("plans have different expert selections")
    bf16 = logical_entries(bf16_plan)
    fp8 = logical_entries(fp8_plan)
    if set(bf16) != set(fp8):
        raise ValueError(
            "BF16/FP8 logical tensor sets differ: "
            f"bf16_only={len(set(bf16) - set(fp8))}, "
            f"fp8_only={len(set(fp8) - set(bf16))}"
        )
    for name, bf16_entry in bf16.items():
        if bf16_entry["shape"] != fp8[name]["shape"]:
            raise ValueError(f"BF16/FP8 shape mismatch for {name}")

    fp8_all = plan_entries(fp8_plan)
    scale_count = 0
    for name, entry in fp8.items():
        scale_name = f"{name}_scale_inv"
        if entry["dtype"] == "F8_E4M3":
            if scale_name not in fp8_all:
                raise ValueError(f"missing FP8 scale for {name}")
            if len(entry["shape"]) != 2:
                raise ValueError(f"block-FP8 tensor is not a matrix: {name}")
            expected = [math.ceil(size / 128) for size in entry["shape"]]
            scale = fp8_all[scale_name]
            if scale["dtype"] != "F32" or scale["shape"] != expected:
                raise ValueError(f"bad FP8 scale grid for {name}: {scale['shape']} != {expected}")
            scale_count += 1
        elif scale_name in fp8_all:
            raise ValueError(f"non-FP8 tensor unexpectedly has a scale: {name}")
    if scale_count == 0:
        raise ValueError("FP8 plan contains no quantized matrices")


def verify_source_range_receipts(manifest: dict[str, Any]) -> dict[str, int]:
    """Verify v2 builder receipts; retain compatibility with published v1."""
    if manifest.get("builder") != "bounded-memory-python-http-range-v2-receipts":
        return {"receipt_count": 0, "receipted_bytes": 0}
    receipts = manifest.get("source_range_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("receipt-producing builder has no source range receipts")
    source = manifest.get("source") or {}
    receipted_bytes = 0
    totals_by_shard: dict[str, int] = {}
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise TypeError(f"source range receipt {index} is not an object")
        if receipt.get("repository") != source.get("repository"):
            raise ValueError(f"source range receipt {index} repository mismatch")
        if receipt.get("revision") != source.get("revision"):
            raise ValueError(f"source range receipt {index} revision mismatch")
        start = int(receipt.get("start", -1))
        end = int(receipt.get("end_exclusive", -1))
        requested_end = int(receipt.get("requested_end_exclusive", -1))
        size = int(receipt.get("bytes", -1))
        if start < 0 or end <= start or requested_end < end or size != end - start:
            raise ValueError(f"source range receipt {index} has invalid bounds")
        if not _SHA256_RE.fullmatch(str(receipt.get("sha256", ""))):
            raise ValueError(f"source range receipt {index} has invalid digest")
        match = _CONTENT_RANGE_RE.fullmatch(str(receipt.get("content_range", "")))
        if not match:
            raise ValueError(f"source range receipt {index} has invalid Content-Range")
        claimed_start, claimed_end = map(int, match.group(1, 2))
        if match.group(3) == "*":
            raise ValueError(f"source range receipt {index} has unknown total size")
        claimed_total = int(match.group(3))
        if claimed_start != start or end > claimed_end + 1 or claimed_end >= requested_end:
            raise ValueError(f"source range receipt {index} contradicts Content-Range")
        if requested_end > claimed_total or claimed_end >= claimed_total:
            raise ValueError(f"source range receipt {index} exceeds source size")
        if receipt.get("reported_total_bytes") != claimed_total:
            raise ValueError(f"source range receipt {index} total-size mismatch")
        shard = str(receipt.get("shard", ""))
        if not shard:
            raise ValueError(f"source range receipt {index} has no shard")
        prior_total = totals_by_shard.setdefault(shard, claimed_total)
        if prior_total != claimed_total:
            raise ValueError(f"source range receipt {index} total-size drift")
        complete_response = receipt.get("complete_response")
        if complete_response not in (True, False):
            raise ValueError(f"source range receipt {index} has no completion state")
        if complete_response != (end == claimed_end + 1):
            raise ValueError(f"source range receipt {index} has wrong completion state")
        receipted_bytes += size
    if receipted_bytes != int(manifest.get("remote_source_bytes", -1)):
        raise ValueError("source range receipt bytes differ from manifest remote_source_bytes")
    return {"receipt_count": len(receipts), "receipted_bytes": receipted_bytes}


def verify_fp8_scale_contract(entries: dict[str, dict[str, Any]], plan: dict[str, Any]) -> int:
    """Prove that every block-FP8 matrix has one exact 128x128 scale grid."""
    block_size = plan.get("config", {}).get("quantization_config", {}).get("weight_block_size")
    if block_size != [128, 128]:
        raise ValueError(f"unexpected FP8 block size: {block_size!r}")

    fp8_weights = {name: entry for name, entry in entries.items() if entry["dtype"] == "F8_E4M3"}
    scale_suffix = "_scale_inv"
    scale_names = {name for name in entries if name.endswith(scale_suffix)}
    for name, entry in fp8_weights.items():
        if len(entry["shape"]) != 2:
            raise ValueError(f"block-FP8 tensor is not a matrix: {name}")
        scale_name = f"{name}{scale_suffix}"
        if scale_name not in entries:
            raise ValueError(f"missing FP8 scale for {name}")
        expected_shape = [math.ceil(entry["shape"][axis] / block_size[axis]) for axis in range(2)]
        scale = entries[scale_name]
        if scale["dtype"] != "F32" or scale["shape"] != expected_shape:
            raise ValueError(
                f"bad FP8 scale grid for {name}: dtype={scale['dtype']}, shape={scale['shape']} != {expected_shape}"
            )

    expected_scales = {f"{name}{scale_suffix}" for name in fp8_weights}
    if scale_names != expected_scales:
        extras = sorted(scale_names - expected_scales)
        missing = sorted(expected_scales - scale_names)
        raise ValueError(f"FP8 scale set mismatch: missing={missing[:3]}, extra={extras[:3]}")
    expected_count = int(plan["target"].get("fp8_weight_count", -1))
    if len(fp8_weights) != expected_count:
        raise ValueError(f"FP8 weight count mismatch: {len(fp8_weights)} != {expected_count}")
    if not fp8_weights:
        raise ValueError("FP8 plan contains no quantized matrices")
    return len(fp8_weights)


def verify_plan(plan: dict[str, Any], expected_precision: str) -> dict[str, Any]:
    if not str(plan.get("pair_id", "")).startswith(("glm52-5l32e-", "glm52-9b-")):
        raise ValueError("invalid pair id")
    target = plan["target"]
    config = plan["config"]
    profiles = {
        "5l-h6144-e32-top8": (5, 32, "glm52-5l-h6144-e32-top8"),
        "10l-h6144-e16-top8": (10, 16, "glm52-10l-h6144-e16-top8"),
    }
    if target["profile"] not in profiles:
        raise ValueError("invalid target profile")
    expected_layers, expected_experts, expected_config_profile = profiles[target["profile"]]
    if target["precision_role"] != expected_precision:
        raise ValueError("unexpected precision role")
    required_config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": expected_layers,
        "hidden_size": 6144,
        "n_routed_experts": expected_experts,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "surgery_dummy": True,
        "surgery_pair_id": plan["pair_id"],
        "surgery_precision_role": expected_precision,
        "surgery_profile": expected_config_profile,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key}: {config.get(key)!r} != {expected!r}")
    entries = plan_entries(plan)
    model_parameters = sum(
        math.prod(entry["shape"])
        for name, entry in entries.items()
        if not name.endswith(("_scale_inv", ".mlp.gate.e_score_correction_bias"))
    )
    serialized_bytes = sum(int(entry["nbytes"]) for entry in entries.values())
    if model_parameters != int(target["model_parameter_count"]):
        raise ValueError("model parameter count mismatch")
    if serialized_bytes != int(target["serialized_bytes"]):
        raise ValueError("serialized byte count mismatch")
    result = {
        "precision_role": expected_precision,
        "tensor_count": len(entries),
        "model_parameter_count": model_parameters,
        "serialized_bytes": serialized_bytes,
    }
    if expected_precision == "fp8-rollout":
        result["fp8_weight_count"] = verify_fp8_scale_contract(entries, plan)
    return result


def verify_built(model_dir: Path, plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    required = (
        "config.json",
        "model.safetensors.index.json",
        "surgery_manifest.json",
        "surgery_plan.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing files in {model_dir}: {missing}")
    if load_json(model_dir / "config.json") != plan["config"]:
        raise ValueError(f"built config differs from plan: {model_dir}")
    copied_plan = model_dir / "surgery_plan.json"
    if sha256_file(copied_plan) != sha256_file(plan_path):
        raise ValueError(f"built surgery plan hash mismatch: {model_dir}")
    index = load_json(model_dir / "model.safetensors.index.json")
    manifest = load_json(model_dir / "surgery_manifest.json")
    if manifest.get("pair_id") != plan["pair_id"]:
        raise ValueError("manifest pair id mismatch")
    if manifest.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("manifest plan hash mismatch")
    if manifest.get("source") != plan.get("source"):
        raise ValueError("manifest source differs from plan")
    if manifest.get("target") != plan.get("target"):
        raise ValueError("manifest target differs from plan")
    receipt_summary = verify_source_range_receipts(manifest)

    entries = plan_entries(plan)
    weight_map = index.get("weight_map", {})
    if set(weight_map) != set(entries):
        raise ValueError("built index tensor set differs from plan")
    referenced_shards = set(weight_map.values())
    if referenced_shards != set(manifest.get("shards", {})):
        raise ValueError("index and manifest shard sets differ")
    actual_shards = {path.name for path in model_dir.glob("model-*.safetensors")}
    if actual_shards != referenced_shards:
        raise ValueError("model directory contains missing or unindexed shard files")

    found: set[str] = set()
    data_bytes = 0
    scale_tensor_count = 0
    scale_value_count = 0
    scale_min = math.inf
    scale_max = -math.inf
    for shard_name in sorted(referenced_shards):
        shard_path = model_dir / shard_name
        record = manifest["shards"][shard_name]
        if shard_path.stat().st_size != int(record["size"]):
            raise ValueError(f"shard size mismatch: {shard_name}")
        if sha256_file(shard_path) != record["sha256"]:
            raise ValueError(f"shard hash mismatch: {shard_name}")
        header, data_start = read_header(shard_path)
        cursor = 0
        with shard_path.open("rb") as shard_handle:
            for name, metadata in sorted(
                ((name, metadata) for name, metadata in header.items() if name != "__metadata__"),
                key=lambda item: int(item[1]["data_offsets"][0]),
            ):
                if name in found or weight_map.get(name) != shard_name:
                    raise ValueError(f"duplicate or misindexed tensor: {name}")
                planned = entries[name]
                if metadata["dtype"] != planned["dtype"]:
                    raise ValueError(f"dtype mismatch for {name}")
                if metadata["shape"] != planned["shape"]:
                    raise ValueError(f"shape mismatch for {name}")
                start, end = map(int, metadata["data_offsets"])
                if start != cursor:
                    raise ValueError(f"gap or overlap before {name}")
                elements = math.prod(metadata["shape"])
                if end - start != elements * DTYPE_BYTES[metadata["dtype"]]:
                    raise ValueError(f"payload size mismatch for {name}")
                if name.endswith("_scale_inv"):
                    shard_handle.seek(data_start + start)
                    payload = shard_handle.read(end - start)
                    if len(payload) != end - start:
                        raise ValueError(f"truncated scale payload for {name}")
                    scales = np.frombuffer(payload, dtype="<f4")
                    if not np.isfinite(scales).all():
                        raise ValueError(f"non-finite FP8 scale in {name}")
                    if not (scales > 0).all():
                        raise ValueError(f"non-positive FP8 scale in {name}")
                    scale_tensor_count += 1
                    scale_value_count += scales.size
                    scale_min = min(scale_min, float(scales.min()))
                    scale_max = max(scale_max, float(scales.max()))
                cursor = end
                found.add(name)
        if shard_path.stat().st_size - data_start != cursor:
            raise ValueError(f"trailing or missing payload bytes: {shard_name}")
        data_bytes += cursor

    if found != set(entries):
        raise ValueError("not every planned tensor was found")
    if data_bytes != int(index["metadata"]["total_size"]):
        raise ValueError("index total size mismatch")
    if data_bytes != int(plan["target"]["serialized_bytes"]):
        raise ValueError("plan total size mismatch")
    result = {
        "model_dir": str(model_dir.resolve()),
        "shard_count": len(referenced_shards),
        "tensor_count": len(found),
        "serialized_bytes": data_bytes,
        "source_ranges": receipt_summary,
        "status": "verified",
    }
    if plan["target"]["precision_role"] == "fp8-rollout":
        expected_scale_count = int(plan["target"]["fp8_weight_count"])
        if scale_tensor_count != expected_scale_count:
            raise ValueError(f"verified scale tensor count mismatch: {scale_tensor_count} != {expected_scale_count}")
        result["fp8_scales"] = {
            "tensor_count": scale_tensor_count,
            "value_count": scale_value_count,
            "minimum": scale_min,
            "maximum": scale_max,
            "all_finite_and_positive": True,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-plan",
        type=Path,
        default=Path("plans/glm52_5l32e_surgery_bf16.json"),
    )
    parser.add_argument(
        "--fp8-plan",
        type=Path,
        default=Path("plans/glm52_5l32e_surgery_fp8.json"),
    )
    parser.add_argument("--bf16-model", type=Path)
    parser.add_argument("--fp8-model", type=Path)
    parser.add_argument(
        "--single-plan",
        type=Path,
        help="verify one qualified plan, for example the FP8 twin of an older anchor",
    )
    parser.add_argument("--single-model", type=Path)
    parser.add_argument(
        "--single-precision",
        choices=("bf16", "fp8-rollout"),
        default="fp8-rollout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_model and not args.single_plan:
        raise ValueError("--single-model requires --single-plan")
    if args.single_plan:
        plan_path = args.single_plan.resolve()
        plan = load_json(plan_path)
        result: dict[str, Any] = {
            "pair_id": plan["pair_id"],
            "plan": verify_plan(plan, args.single_precision),
            "status": "plan_verified",
        }
        if args.single_model:
            result["model"] = verify_built(args.single_model.resolve(), plan_path, plan)
            result["status"] = "model_verified"
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    bf16_plan = load_json(args.bf16_plan)
    fp8_plan = load_json(args.fp8_plan)
    bf16_summary = verify_plan(bf16_plan, "bf16")
    fp8_summary = verify_plan(fp8_plan, "fp8-rollout")
    verify_pair_contract(bf16_plan, fp8_plan)
    result: dict[str, Any] = {
        "pair_id": bf16_plan["pair_id"],
        "bf16_plan": bf16_summary,
        "fp8_plan": fp8_summary,
        "status": "plans_verified",
    }
    if args.bf16_model:
        result["bf16_model"] = verify_built(args.bf16_model.resolve(), args.bf16_plan.resolve(), bf16_plan)
    if args.fp8_model:
        result["fp8_model"] = verify_built(args.fp8_model.resolve(), args.fp8_plan.resolve(), fp8_plan)
    if args.bf16_model and args.fp8_model:
        result["status"] = "pair_verified"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
