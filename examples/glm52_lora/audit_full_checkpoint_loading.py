#!/usr/bin/env python3
"""Audit GLM-5.2 safetensors granularity and Bridge load amplification.

Only the index and the JSON header at the start of every shard are read. Tensor
payloads are never materialized. The reported read volume is logical traffic
implied by Megatron Bridge's per-rank HF import; filesystem page caches can
reduce physical backing-store traffic but must not be assumed by a launch gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

EXPECTED_REVISION = "cf457fa734ab149ffef225f80893eb38c6ff5cdc"
EXPECTED_CONFIG_SHA256 = (
    "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
)
EXPECTED_INDEX_SHA256 = (
    "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
)
EXPECTED_TOTAL_BYTES = 1_506_659_919_872
EXPECTED_TENSOR_COUNT = 59_585
EXPECTED_SHARD_COUNT = 282
EXPECTED_EXPERT_TENSOR_COUNT = 58_368
EXPECTED_EXPERT_BYTES = 1_468_878_815_232
EXPECTED_MAX_TENSOR_BYTES = 1_903_165_440
EXPECTED_DTYPE_COUNTS = {"BF16": 59_509, "F32": 76}
EXPECTED_DTYPE_BYTES = {"BF16": 1_506_659_842_048, "F32": 77_824}
EXPECTED_AUXILIARY_TENSOR_COUNT = 791
EXPECTED_AUXILIARY_BYTES = 19_905_841_664
EXPECTED_POLICY_TENSOR_COUNT = 58_794
EXPECTED_POLICY_BYTES = 1_486_754_078_208
EXPECTED_POLICY_EXPERT_TENSOR_COUNT = 57_600
EXPECTED_POLICY_EXPERT_BYTES = 1_449_551_462_400
EXPECTED_BRIDGE_REVISION = "d0c6228a2a832f566dd44a3a179b3136613c11b7"

EXPECTED_FP8_CONFIG_SHA256 = (
    "d1539d36be7546a1d827fe9cf74c55874695652efb6a5aaa3e60cde1c76ba819"
)
EXPECTED_FP8_INDEX_SHA256 = (
    "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
)
EXPECTED_FP8_TOTAL_BYTES = 755_617_140_416
EXPECTED_FP8_TENSOR_COUNT = 118_629
EXPECTED_FP8_SHARD_COUNT = 141
EXPECTED_FP8_DTYPE_COUNTS = {"BF16": 465, "F32": 59_120, "F8_E4M3": 59_044}
EXPECTED_FP8_DTYPE_BYTES = {
    "BF16": 4_207_458_304,
    "F32": 183_490_240,
    "F8_E4M3": 751_226_191_872,
}
EXPECTED_FP8_WEIGHT_COUNT = 59_044
EXPECTED_FP8_SCALE_COUNT = 59_044
EXPECTED_FP8_EXCLUDED_MODULE_COUNT = 541
EXPECTED_FP8_EXPERT_TENSOR_COUNT = 116_736
EXPECTED_FP8_EXPERT_BYTES = 734_618_714_112
EXPECTED_FP8_AUXILIARY_TENSOR_COUNT = 1_569
EXPECTED_FP8_AUXILIARY_BYTES = 10_032_632_960
EXPECTED_FP8_POLICY_TENSOR_COUNT = 117_060
EXPECTED_FP8_POLICY_BYTES = 745_584_507_456
EXPECTED_FP8_POLICY_SCALE_COUNT = 58_266
EXPECTED_FP8_POLICY_EXPERT_TENSOR_COUNT = 115_200
EXPECTED_FP8_POLICY_EXPERT_BYTES = 724_952_678_400
EXPECTED_FP8_POLICY_DESTINATION_BF16_BYTES = 1_486_754_078_208
EXPECTED_FP8_LARGEST_TENSOR_BYTES = 1_903_165_440
EXPECTED_FP8_BRIDGE_BASE_REVISION = EXPECTED_BRIDGE_REVISION
EXPECTED_FP8_BRIDGE_PATCH_SHA256 = (
    "d5764f406994684392cb78bc2977b6ca90a30680c448022742023d9c1298c590"
)

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
EXPERT_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight(?:_scale_inv)?$"
)
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
MAX_HEADER_BYTES = 64 * 2**20


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CHECKPOINT-LOAD-AUDIT-FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_is_excluded(name: str, exclusions: list[str]) -> bool:
    return any(name == item or name.startswith(item + ".") for item in exclusions)


def read_safetensors_header(path: Path) -> tuple[dict, int, int]:
    """Return parsed header, payload offset, and file size without reading data."""
    require(path.is_file(), f"missing shard: {path}")
    file_size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        require(len(prefix) == 8, f"truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        require(
            2 <= header_size <= MAX_HEADER_BYTES,
            f"unsafe safetensors header size {header_size} in {path}",
        )
        raw_header = source.read(header_size)
    require(len(raw_header) == header_size, f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"CHECKPOINT-LOAD-AUDIT-FAIL: invalid safetensors header in {path}: {error}"
        ) from error
    require(isinstance(header, dict), f"non-object safetensors header: {path}")
    return header, 8 + header_size, file_size


def inspect_shards(checkpoint_dir: Path, weight_map: dict[str, str]) -> dict:
    expected_by_file: dict[str, set[str]] = {}
    for tensor_name, filename in weight_map.items():
        require(
            isinstance(tensor_name, str) and isinstance(filename, str),
            "index weight_map must contain string names and filenames",
        )
        expected_by_file.setdefault(filename, set()).add(tensor_name)

    tensor_bytes: dict[str, int] = {}
    tensor_dtypes: dict[str, str] = {}
    tensor_shapes: dict[str, list[int]] = {}
    largest_shard_bytes = 0
    largest_shard = None
    header_bytes_total = 0
    for filename in sorted(expected_by_file):
        require(Path(filename).name == filename, f"unsafe shard filename: {filename}")
        shard_path = checkpoint_dir / filename
        header, payload_offset, file_size = read_safetensors_header(shard_path)
        header_bytes_total += payload_offset
        largest_shard_bytes = max(largest_shard_bytes, file_size)
        if file_size == largest_shard_bytes:
            largest_shard = filename

        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        require(
            set(entries) == expected_by_file[filename],
            f"header/index key mismatch in {filename}",
        )
        max_payload_end = 0
        for tensor_name, spec in entries.items():
            require(isinstance(spec, dict), f"invalid tensor spec for {tensor_name}")
            dtype = spec.get("dtype")
            shape = spec.get("shape")
            offsets = spec.get("data_offsets")
            require(dtype in DTYPE_BYTES, f"unsupported dtype {dtype!r} for {tensor_name}")
            require(
                isinstance(shape, list)
                and all(isinstance(dim, int) and dim >= 0 for dim in shape),
                f"invalid shape for {tensor_name}",
            )
            require(
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(offset, int) and offset >= 0 for offset in offsets)
                and offsets[1] >= offsets[0],
                f"invalid offsets for {tensor_name}",
            )
            size = math.prod(shape) * DTYPE_BYTES[dtype]
            require(
                offsets[1] - offsets[0] == size,
                f"shape/dtype byte count disagrees with offsets for {tensor_name}",
            )
            tensor_bytes[tensor_name] = size
            tensor_dtypes[tensor_name] = dtype
            tensor_shapes[tensor_name] = shape
            max_payload_end = max(max_payload_end, offsets[1])
        require(
            payload_offset + max_payload_end == file_size,
            f"header payload extent disagrees with file size for {filename}",
        )

    return {
        "tensor_bytes": tensor_bytes,
        "tensor_dtypes": tensor_dtypes,
        "tensor_shapes": tensor_shapes,
        "header_bytes_read": header_bytes_total,
        "largest_shard": largest_shard,
        "largest_shard_bytes": largest_shard_bytes,
    }


def audit_checkpoint(
    checkpoint_dir: Path,
    *,
    world_size: int,
    tp: int,
    ep: int,
    etp: int,
    pp: int,
    cp: int,
    official_glm52: bool,
    bridge_revision: str,
    official_profile: str | None = None,
    bridge_base_revision: str | None = None,
    bridge_patch_sha256: str | None = None,
) -> dict:
    if official_glm52:
        require(
            official_profile in (None, "bf16"),
            "--official-glm52 is only compatible with the BF16 profile",
        )
        official_profile = "bf16"
    require(
        official_profile in (None, "bf16", "fp8-dequant"),
        f"unknown official profile: {official_profile}",
    )
    for name, value in {
        "world_size": world_size,
        "tp": tp,
        "ep": ep,
        "etp": etp,
        "pp": pp,
        "cp": cp,
    }.items():
        require(value > 0, f"{name} must be positive")
    require(world_size % (tp * pp * cp) == 0, "invalid dense process grid")
    require(world_size % (etp * ep * pp) == 0, "invalid expert process grid")

    config_path = checkpoint_dir / "config.json"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    require(config_path.is_file(), f"missing config: {config_path}")
    require(index_path.is_file(), f"missing index: {index_path}")
    config_sha = sha256(config_path)
    index_sha = sha256(index_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_hidden_layers = config.get("num_hidden_layers")
    require(
        isinstance(num_hidden_layers, int) and num_hidden_layers > 0,
        "config num_hidden_layers must be positive",
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "missing index weight_map")
    metadata_total = index.get("metadata", {}).get("total_size")
    require(isinstance(metadata_total, int), "missing integer metadata.total_size")

    inspected = inspect_shards(checkpoint_dir, weight_map)
    tensor_bytes = inspected.pop("tensor_bytes")
    tensor_dtypes = inspected.pop("tensor_dtypes")
    tensor_shapes = inspected.pop("tensor_shapes")
    dtype_counts = {
        dtype: sum(value == dtype for value in tensor_dtypes.values())
        for dtype in sorted(set(tensor_dtypes.values()))
    }
    dtype_bytes = {
        dtype: sum(
            tensor_bytes[key] for key, value in tensor_dtypes.items() if value == dtype
        )
        for dtype in dtype_counts
    }
    calculated_total = sum(tensor_bytes.values())
    require(calculated_total == metadata_total, "index total_size disagrees with headers")

    expert_keys = [key for key in tensor_bytes if EXPERT_RE.fullmatch(key)]
    expert_bytes = sum(tensor_bytes[key] for key in expert_keys)
    nonexpert_bytes = calculated_total - expert_bytes
    auxiliary_keys = []
    for key in tensor_bytes:
        layer_match = LAYER_RE.match(key)
        if layer_match is not None and int(layer_match.group(1)) >= num_hidden_layers:
            auxiliary_keys.append(key)
    auxiliary_key_set = set(auxiliary_keys)
    policy_keys = [key for key in tensor_bytes if key not in auxiliary_key_set]
    policy_expert_keys = [key for key in expert_keys if key not in auxiliary_key_set]
    auxiliary_bytes = sum(tensor_bytes[key] for key in auxiliary_keys)
    policy_bytes = sum(tensor_bytes[key] for key in policy_keys)
    policy_expert_bytes = sum(tensor_bytes[key] for key in policy_expert_keys)
    policy_nonexpert_bytes = policy_bytes - policy_expert_bytes
    largest_name, largest_bytes = max(tensor_bytes.items(), key=lambda item: item[1])

    scale_keys = {key for key in tensor_bytes if key.endswith(".weight_scale_inv")}
    fp8_keys = {
        key
        for key, dtype in tensor_dtypes.items()
        if dtype in {"F8_E4M3", "F8_E5M2"}
    }
    require(
        all(key.endswith(".weight") for key in fp8_keys),
        "FP8 checkpoint contains a non-weight tensor",
    )
    missing_scale_keys = sorted(
        key for key in fp8_keys if key + "_scale_inv" not in scale_keys
    )
    orphan_scale_keys = sorted(
        key for key in scale_keys if key.removesuffix("_scale_inv") not in fp8_keys
    )
    require(not missing_scale_keys, f"FP8 weights without scales: {missing_scale_keys[:8]}")
    require(not orphan_scale_keys, f"orphan FP8 scales: {orphan_scale_keys[:8]}")

    quantization_config = config.get("quantization_config", {})
    block_size = quantization_config.get("weight_block_size")
    if fp8_keys:
        require(block_size == [128, 128], f"unexpected FP8 block size: {block_size}")
    for weight_key in fp8_keys:
        scale_key = weight_key + "_scale_inv"
        require(tensor_dtypes[scale_key] == "F32", f"non-F32 scale: {scale_key}")
        require(
            len(tensor_shapes[weight_key]) == 2,
            f"non-matrix FP8 weight: {weight_key}",
        )
        expected_scale_shape = [
            math.ceil(size / block)
            for size, block in zip(tensor_shapes[weight_key], block_size, strict=True)
        ]
        require(
            tensor_shapes[scale_key] == expected_scale_shape,
            f"FP8 scale geometry mismatch: {weight_key}",
        )

    exclusions = quantization_config.get("modules_to_not_convert", [])
    require(isinstance(exclusions, list), "modules_to_not_convert must be a list")
    excluded_fp8 = sorted(
        key for key in fp8_keys if module_is_excluded(key, exclusions)
    )
    require(not excluded_fp8, f"excluded modules stored as FP8: {excluded_fp8[:8]}")
    policy_scale_keys = scale_keys.intersection(policy_keys)
    policy_destination_bf16_bytes = sum(
        math.prod(tensor_shapes[key])
        * (2 if tensor_dtypes[key] in {"F8_E4M3", "F8_E5M2"} else DTYPE_BYTES[tensor_dtypes[key]])
        for key in policy_keys
        if key not in policy_scale_keys
    )

    gated_bundles = []
    for key, size in tensor_bytes.items():
        if key.endswith(".gate_proj.weight"):
            up_key = key.removesuffix(".gate_proj.weight") + ".up_proj.weight"
            if up_key in tensor_bytes:
                gated_bundles.append((size + tensor_bytes[up_key], key, up_key))
    largest_gated_bundle = max(gated_bundles, default=(0, None, None))

    # Bridge loads HF inputs before TP/ETP scatter. Non-expert tensors are read
    # by every rank in their owning PP stage. EP partitions expert identities;
    # ETP ranks and expert-DP replicas each read the full source expert tensor.
    nonexpert_read_factor = world_size // pp
    expert_read_factor = world_size // (ep * pp)
    logical_nonexpert_bytes = policy_nonexpert_bytes * nonexpert_read_factor
    logical_expert_bytes = policy_expert_bytes * expert_read_factor
    logical_total_bytes = logical_nonexpert_bytes + logical_expert_bytes
    checkpoint_upper_bound_bytes = (
        nonexpert_bytes * nonexpert_read_factor + expert_bytes * expert_read_factor
    )

    if official_profile == "bf16":
        require(config_sha == EXPECTED_CONFIG_SHA256, "official config SHA-256 drift")
        require(index_sha == EXPECTED_INDEX_SHA256, "official index SHA-256 drift")
        require(bridge_revision == EXPECTED_BRIDGE_REVISION, "Bridge revision drift")
        require(len(weight_map) == EXPECTED_TENSOR_COUNT, "official tensor-count drift")
        require(len(set(weight_map.values())) == EXPECTED_SHARD_COUNT, "official shard-count drift")
        require(calculated_total == EXPECTED_TOTAL_BYTES, "official payload-size drift")
        require(len(expert_keys) == EXPECTED_EXPERT_TENSOR_COUNT, "expert granularity drift")
        require(expert_bytes == EXPECTED_EXPERT_BYTES, "expert byte-count drift")
        require(largest_bytes == EXPECTED_MAX_TENSOR_BYTES, "largest source tensor drift")
        require(dtype_counts == EXPECTED_DTYPE_COUNTS, "official dtype-count drift")
        require(dtype_bytes == EXPECTED_DTYPE_BYTES, "official dtype byte-count drift")
        f32_keys = [key for key, dtype in tensor_dtypes.items() if dtype == "F32"]
        require(
            all(key.endswith(".mlp.gate.e_score_correction_bias") for key in f32_keys),
            "unexpected official FP32 parameter",
        )
        require(
            len(auxiliary_keys) == EXPECTED_AUXILIARY_TENSOR_COUNT,
            "MTP/auxiliary tensor-count drift",
        )
        require(
            auxiliary_bytes == EXPECTED_AUXILIARY_BYTES,
            "MTP/auxiliary byte-count drift",
        )
        require(len(policy_keys) == EXPECTED_POLICY_TENSOR_COUNT, "policy tensor-count drift")
        require(policy_bytes == EXPECTED_POLICY_BYTES, "policy byte-count drift")
        require(
            len(policy_expert_keys) == EXPECTED_POLICY_EXPERT_TENSOR_COUNT,
            "policy expert tensor-count drift",
        )
        require(
            policy_expert_bytes == EXPECTED_POLICY_EXPERT_BYTES,
            "policy expert byte-count drift",
        )
    elif official_profile == "fp8-dequant":
        require(config_sha == EXPECTED_FP8_CONFIG_SHA256, "FP8 config SHA-256 drift")
        require(index_sha == EXPECTED_FP8_INDEX_SHA256, "FP8 index SHA-256 drift")
        require(
            bridge_base_revision == EXPECTED_FP8_BRIDGE_BASE_REVISION,
            "FP8 Bridge base revision drift",
        )
        require(
            bridge_patch_sha256 == EXPECTED_FP8_BRIDGE_PATCH_SHA256,
            "FP8 Bridge dequantization patch drift",
        )
        require(len(weight_map) == EXPECTED_FP8_TENSOR_COUNT, "FP8 tensor-count drift")
        require(
            len(set(weight_map.values())) == EXPECTED_FP8_SHARD_COUNT,
            "FP8 shard-count drift",
        )
        require(calculated_total == EXPECTED_FP8_TOTAL_BYTES, "FP8 payload-size drift")
        require(largest_bytes == EXPECTED_FP8_LARGEST_TENSOR_BYTES, "FP8 largest tensor drift")
        require(dtype_counts == EXPECTED_FP8_DTYPE_COUNTS, "FP8 dtype-count drift")
        require(dtype_bytes == EXPECTED_FP8_DTYPE_BYTES, "FP8 dtype byte-count drift")
        require(len(fp8_keys) == EXPECTED_FP8_WEIGHT_COUNT, "FP8 weight-count drift")
        require(len(scale_keys) == EXPECTED_FP8_SCALE_COUNT, "FP8 scale-count drift")
        require(
            len(expert_keys) == EXPECTED_FP8_EXPERT_TENSOR_COUNT,
            "FP8 expert tensor-count drift",
        )
        require(expert_bytes == EXPECTED_FP8_EXPERT_BYTES, "FP8 expert byte-count drift")
        require(
            len(exclusions) == EXPECTED_FP8_EXCLUDED_MODULE_COUNT,
            "FP8 exclusion-count drift",
        )
        require(
            len(auxiliary_keys) == EXPECTED_FP8_AUXILIARY_TENSOR_COUNT,
            "FP8 MTP/auxiliary tensor-count drift",
        )
        require(
            auxiliary_bytes == EXPECTED_FP8_AUXILIARY_BYTES,
            "FP8 MTP/auxiliary byte-count drift",
        )
        require(
            len(policy_keys) == EXPECTED_FP8_POLICY_TENSOR_COUNT,
            "FP8 policy tensor-count drift",
        )
        require(policy_bytes == EXPECTED_FP8_POLICY_BYTES, "FP8 policy byte-count drift")
        require(
            len(policy_scale_keys) == EXPECTED_FP8_POLICY_SCALE_COUNT,
            "FP8 policy scale-count drift",
        )
        require(
            len(policy_expert_keys) == EXPECTED_FP8_POLICY_EXPERT_TENSOR_COUNT,
            "FP8 policy expert tensor-count drift",
        )
        require(
            policy_expert_bytes == EXPECTED_FP8_POLICY_EXPERT_BYTES,
            "FP8 policy expert byte-count drift",
        )
        require(
            policy_destination_bf16_bytes
            == EXPECTED_FP8_POLICY_DESTINATION_BF16_BYTES,
            "FP8 policy BF16 destination-size drift",
        )
        non_scale_f32 = [
            key
            for key, dtype in tensor_dtypes.items()
            if dtype == "F32" and key not in scale_keys
        ]
        require(
            all(
                key.endswith(".mlp.gate.e_score_correction_bias")
                for key in non_scale_f32
            ),
            "unexpected non-scale FP32 parameter",
        )

    return {
        "status": "CHECKPOINT-LOAD-AUDIT-PASS",
        "official_profile": official_profile,
        "checkpoint": {
            "revision": EXPECTED_REVISION if official_profile == "bf16" else None,
            "config_sha256": config_sha,
            "index_sha256": index_sha,
            "tensor_count": len(weight_map),
            "shard_count": len(set(weight_map.values())),
            "payload_bytes": calculated_total,
            "payload_gib": calculated_total / 2**30,
            "expert_tensor_count": len(expert_keys),
            "expert_bytes": expert_bytes,
            "nonexpert_bytes": nonexpert_bytes,
            "dtype_counts": dtype_counts,
            "dtype_bytes": dtype_bytes,
            "auxiliary_tensor_count": len(auxiliary_keys),
            "auxiliary_bytes": auxiliary_bytes,
            "fp8_weight_count": len(fp8_keys),
            "scale_count": len(scale_keys),
            "excluded_module_count": len(exclusions),
        },
        "policy_import": {
            "mtp_enabled": False,
            "num_hidden_layers": num_hidden_layers,
            "tensor_count": len(policy_keys),
            "payload_bytes": policy_bytes,
            "expert_tensor_count": len(policy_expert_keys),
            "expert_bytes": policy_expert_bytes,
            "nonexpert_bytes": policy_nonexpert_bytes,
            "scale_count": len(policy_scale_keys),
            "destination_bf16_bytes": policy_destination_bf16_bytes,
        },
        "source_working_set": {
            "largest_tensor": largest_name,
            "largest_tensor_bytes": largest_bytes,
            "largest_tensor_gib": largest_bytes / 2**30,
            "largest_gate_up_bundle": list(largest_gated_bundle[1:]),
            "largest_gate_up_bundle_bytes": largest_gated_bundle[0],
            "largest_shard": inspected["largest_shard"],
            "largest_shard_bytes": inspected["largest_shard_bytes"],
            "headers_read_bytes": inspected["header_bytes_read"],
            "whole_shard_materialization_required": False,
        },
        "bridge_import": {
            "revision": bridge_revision,
            "base_revision": bridge_base_revision,
            "dequantization_patch_sha256": bridge_patch_sha256,
            "contract": "lazy-tensor-read-before-rank-local-tp-or-etp-scatter",
            "nonexpert_read_factor": nonexpert_read_factor,
            "expert_read_factor": expert_read_factor,
            "logical_nonexpert_read_bytes": logical_nonexpert_bytes,
            "logical_expert_read_bytes": logical_expert_bytes,
            "logical_total_read_bytes": logical_total_bytes,
            "logical_total_read_tib": logical_total_bytes / 2**40,
            "average_logical_read_gib_per_rank": logical_total_bytes / world_size / 2**30,
            "whole_checkpoint_upper_bound_bytes": checkpoint_upper_bound_bytes,
            "whole_checkpoint_upper_bound_tib": checkpoint_upper_bound_bytes / 2**40,
            "page_cache_reduction_assumed": False,
        },
        "topology": {
            "world_size": world_size,
            "tp": tp,
            "ep": ep,
            "etp": etp,
            "pp": pp,
            "cp": cp,
            "dense_dp": world_size // (tp * pp * cp),
            "expert_dp": world_size // (etp * ep * pp),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--etp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--bridge-revision", required=True)
    parser.add_argument("--bridge-base-revision")
    parser.add_argument("--bridge-patch-sha256")
    parser.add_argument("--official-glm52", action="store_true")
    parser.add_argument(
        "--official-profile",
        choices=("bf16", "fp8-dequant"),
    )
    args = parser.parse_args()
    result = audit_checkpoint(
        args.checkpoint_dir,
        world_size=args.world_size,
        tp=args.tp,
        ep=args.ep,
        etp=args.etp,
        pp=args.pp,
        cp=args.cp,
        official_glm52=args.official_glm52,
        bridge_revision=args.bridge_revision,
        official_profile=args.official_profile,
        bridge_base_revision=args.bridge_base_revision,
        bridge_patch_sha256=args.bridge_patch_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
