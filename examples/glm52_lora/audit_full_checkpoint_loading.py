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
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
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
            max_payload_end = max(max_payload_end, offsets[1])
        require(
            payload_offset + max_payload_end == file_size,
            f"header payload extent disagrees with file size for {filename}",
        )

    return {
        "tensor_bytes": tensor_bytes,
        "tensor_dtypes": tensor_dtypes,
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
) -> dict:
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

    if official_glm52:
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

    return {
        "status": "CHECKPOINT-LOAD-AUDIT-PASS",
        "checkpoint": {
            "revision": EXPECTED_REVISION if official_glm52 else None,
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
        },
        "policy_import": {
            "mtp_enabled": False,
            "num_hidden_layers": num_hidden_layers,
            "tensor_count": len(policy_keys),
            "payload_bytes": policy_bytes,
            "expert_tensor_count": len(policy_expert_keys),
            "expert_bytes": policy_expert_bytes,
            "nonexpert_bytes": policy_nonexpert_bytes,
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
    parser.add_argument("--official-glm52", action="store_true")
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
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
