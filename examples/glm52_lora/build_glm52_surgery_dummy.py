#!/usr/bin/env python3
"""Materialize a planned GLM-5.2 surgery checkpoint with bounded memory.

The plan is authoritative: the builder verifies the pinned source metadata,
every source tensor location, the exact target config, and every emitted shard.
It never downloads a complete donor shard and records receipts for all HTTP
source ranges used by an executed build.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from glm52_surgery_io import (
    DTYPE_BYTES,
    DigestWriter,
    HubRangeReader,
    sha256_file,
)

MIB = 1024**2
GIB = 1024**3
SUPPORTED_KINDS = {
    "direct",
    "router_gather_rows",
    "router_gather_bias",
    "router_cluster_centroid",
    "router_cluster_bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size-gib", type=float, default=1.5)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine MemAvailable")


def plan_entries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    entries = plan.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise ValueError("surgery plan has no tensors")
    names = [str(entry.get("target_name")) for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("surgery plan has duplicate target tensors")
    unknown = sorted({str(entry.get("kind")) for entry in entries} - SUPPORTED_KINDS)
    if unknown:
        raise ValueError(f"unsupported surgery operation(s): {unknown}")
    return sorted(entries, key=lambda entry: str(entry["target_name"]))


def validate_metadata(
    plan: dict[str, Any], plan_path: Path, metadata_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("unsupported surgery plan schema")
    source = plan.get("source") or {}
    target = plan.get("target") or {}
    if not source.get("repository") or not source.get("revision"):
        raise ValueError("plan source is not pinned")
    if not target.get("repository") or not target.get("profile"):
        raise ValueError("plan target contract is incomplete")
    config_path = metadata_dir / "config.json"
    index_path = metadata_dir / "model.safetensors.index.json"
    for path in (plan_path, config_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(config_path) != source.get("config_sha256"):
        raise ValueError("source config hash differs from the pinned plan")
    if sha256_file(index_path) != source.get("index_sha256"):
        raise ValueError("source index hash differs from the pinned plan")
    source_config = load_json(config_path)
    source_index = load_json(index_path)
    target_config = plan.get("config")
    if not isinstance(target_config, dict):
        raise TypeError("plan has no exact target config")
    if target_config.get("surgery_pair_id") != plan.get("pair_id"):
        raise ValueError("target config surgery pair id differs from the plan")
    return source_config, source_index, sha256_file(plan_path)


def validate_plan_totals(plan: dict[str, Any], entries: list[dict[str, Any]]) -> int:
    target = plan["target"]
    serialized_bytes = sum(int(entry["nbytes"]) for entry in entries)
    serialized_elements = sum(math.prod(entry["shape"]) for entry in entries)
    model_parameters = sum(
        math.prod(entry["shape"])
        for entry in entries
        if not str(entry["target_name"]).endswith(("_scale_inv", ".mlp.gate.e_score_correction_bias"))
    )
    if serialized_bytes != int(target["serialized_bytes"]):
        raise ValueError("plan serialized byte total is internally inconsistent")
    expected_elements = target.get("serialized_elements_including_fp8_scales_and_router_buffers")
    if expected_elements is not None and serialized_elements != int(expected_elements):
        raise ValueError("plan serialized element total is internally inconsistent")
    if model_parameters != int(target["model_parameter_count"]):
        raise ValueError("plan logical parameter total is internally inconsistent")
    if target.get("precision_role") == "fp8-rollout":
        fp8_weights = sum(entry["dtype"] == "F8_E4M3" for entry in entries)
        if fp8_weights != int(target.get("fp8_weight_count", -1)):
            raise ValueError("plan FP8 weight count is internally inconsistent")
    return serialized_bytes


def validate_source_locations(entries: list[dict[str, Any]], reader: HubRangeReader) -> None:
    """Bind every planned byte range to the pinned index and live header."""
    for entry in entries:
        source = entry.get("source")
        if not isinstance(source, dict):
            raise TypeError(f"tensor {entry['target_name']} has no source location")
        source_name = str(source.get("name"))
        location = reader.location(source_name)
        expected = {
            "name": location.name,
            "shard": location.shard,
            "dtype": location.dtype,
            "shape": list(location.shape),
            "file_start": location.file_start,
            "file_end": location.file_end,
        }
        actual = {
            "name": source_name,
            "shard": source.get("shard"),
            "dtype": source.get("dtype"),
            "shape": source.get("shape"),
            "file_start": source.get("file_start"),
            "file_end": source.get("file_end"),
        }
        if actual != expected:
            raise ValueError(f"planned source location drift for {entry['target_name']}: {actual!r} != {expected!r}")
        shape = list(map(int, entry["shape"]))
        dtype = str(entry["dtype"])
        expected_nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if int(entry["nbytes"]) != expected_nbytes:
            raise ValueError(f"target byte count mismatch for {entry['target_name']}")


def encode_array(array: np.ndarray, dtype: str) -> bytes:
    contiguous = np.ascontiguousarray(array)
    if dtype == "BF16":
        bits = contiguous.astype("<f4", copy=False).view("<u4") >> 16
        return bits.astype("<u2").tobytes()
    dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
    if dtype not in dtypes:
        raise ValueError(f"cannot encode generated {dtype} tensor")
    return contiguous.astype(dtypes[dtype], copy=False).tobytes()


def _selection(plan: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    target_layer = str(entry.get("target_layer"))
    selection = (plan.get("expert_selection") or {}).get(target_layer)
    if not isinstance(selection, dict):
        raise TypeError(f"missing expert selection for target layer {target_layer}")
    return selection


def generated_payload(plan: dict[str, Any], entry: dict[str, Any], reader: HubRangeReader) -> bytes | None:
    kind = str(entry["kind"])
    if kind == "direct":
        return None
    selection = _selection(plan, entry)
    selected = list(map(int, selection.get("selected_source_experts", [])))
    source_name = str(entry["source"]["name"])
    source = reader.tensor(source_name)
    if kind in {"router_gather_rows", "router_gather_bias"}:
        generated = source[selected]
    else:
        clusters = selection.get("source_clusters")
        if not isinstance(clusters, list) or len(clusters) != len(selected):
            raise ValueError(f"missing source clusters for {entry['target_name']}")
        generated = np.stack([source[np.asarray(cluster, dtype=np.int64)].mean(axis=0) for cluster in clusters])
    payload = encode_array(generated, str(entry["dtype"]))
    if len(payload) != int(entry["nbytes"]):
        raise ValueError(
            f"generated payload size mismatch for {entry['target_name']}: {len(payload)} != {entry['nbytes']}"
        )
    return payload


def safetensors_prefix(entries: list[dict[str, Any]]) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for entry in entries:
        nbytes = int(entry["nbytes"])
        header[str(entry["target_name"])] = {
            "dtype": str(entry["dtype"]),
            "shape": list(map(int, entry["shape"])),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    return struct.pack("<Q", len(payload)) + payload


def group_shards(entries: list[dict[str, Any]], max_data_bytes: int) -> list[list[dict[str, Any]]]:
    if max_data_bytes <= 0:
        raise ValueError("maximum shard size must be positive")
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for entry in entries:
        nbytes = int(entry["nbytes"])
        if current and current_bytes + nbytes > max_data_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += nbytes
    if current:
        groups.append(current)
    return groups


def write_readme(output: Path, plan: dict[str, Any]) -> None:
    target = plan["target"]
    text = f"""# {str(target["repository"]).split("/")[-1]}

Test-only checkpoint for GLM-5.2 post-training, LoRA, weight-sync, and
sharding integration. It is not a usable chat or benchmark model.

The exact source revisions, layer/expert selection, tensor byte ranges, and
mixed-precision contract are recorded in `surgery_plan.json`. Pair ID:
`{plan["pair_id"]}`. Logical parameters: {int(target["model_parameter_count"]):,}.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_shard_size_gib <= 0 or args.chunk_mib <= 0:
        raise ValueError("shard size and chunk size must be positive")
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    _, source_index, plan_sha256 = validate_metadata(plan, plan_path, args.metadata_dir)
    entries = plan_entries(plan)
    source = plan["source"]
    reader = HubRangeReader(
        str(source["repository"]),
        str(source["revision"]),
        source_index,
        chunk_bytes=args.chunk_mib * MIB,
    )
    validate_source_locations(entries, reader)
    serialized_bytes = validate_plan_totals(plan, entries)
    shards = group_shards(entries, int(args.max_shard_size_gib * GIB))
    summary = {
        "pair_id": plan["pair_id"],
        "plan_sha256": plan_sha256,
        "tensor_count": len(entries),
        "serialized_bytes": serialized_bytes,
        "output_shards": len(shards),
        "status": "plan_verified",
    }
    if not args.execute:
        return summary

    args.output.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(args.output).free
    if free_disk < serialized_bytes + 2 * GIB:
        raise OSError(f"insufficient disk: {free_disk / GIB:.1f} GiB free for {serialized_bytes / GIB:.1f} GiB output")
    if available_memory_bytes() < 512 * MIB:
        raise MemoryError("less than 512 MiB MemAvailable before surgery")

    state_path = args.output / ".surgery_state.json"
    identity = {
        "schema_version": 1,
        "pair_id": plan["pair_id"],
        "plan_sha256": plan_sha256,
        "source": plan["source"],
        "max_shard_data_bytes": int(args.max_shard_size_gib * GIB),
    }
    if state_path.exists():
        state = load_json(state_path)
        if {key: state.get(key) for key in identity} != identity:
            raise ValueError("existing surgery state belongs to a different plan")
    else:
        state = {
            **identity,
            "completed": {},
            "remote_source_bytes": 0,
            "source_range_receipts": [],
        }

    prior_remote_source_bytes = int(state.get("remote_source_bytes", 0))
    persisted_receipts = list(state.get("source_range_receipts", []))
    receipt_cursor = 0

    def persist_state() -> None:
        nonlocal receipt_cursor
        persisted_receipts.extend(reader.range_receipts[receipt_cursor:])
        receipt_cursor = len(reader.range_receipts)
        state["source_range_receipts"] = persisted_receipts
        state["remote_source_bytes"] = prior_remote_source_bytes + reader.remote_bytes
        atomic_json(state_path, state)

    output_weight_map: dict[str, str] = {}
    started = time.monotonic()
    completed_data = 0
    for shard_index, shard_entries in enumerate(shards, start=1):
        filename = f"model-{shard_index:05d}-of-{len(shards):05d}.safetensors"
        destination = args.output / filename
        prefix = safetensors_prefix(shard_entries)
        data_bytes = sum(int(entry["nbytes"]) for entry in shard_entries)
        expected_size = len(prefix) + data_bytes
        recorded = state["completed"].get(filename)
        if (
            recorded
            and destination.is_file()
            and destination.stat().st_size == expected_size
            and sha256_file(destination) == recorded.get("sha256")
        ):
            completed_data += data_bytes
        else:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with temporary.open("wb") as handle:
                writer = DigestWriter(handle)
                writer.write(prefix)
                for entry in shard_entries:
                    payload = generated_payload(plan, entry, reader)
                    if payload is not None:
                        writer.write(payload)
                    else:
                        source_location = entry["source"]
                        reader.copy_range(
                            str(source_location["shard"]),
                            int(source_location["file_start"]),
                            int(source_location["file_end"]),
                            writer,
                        )
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != expected_size:
                raise OSError(f"wrong output size for {filename}")
            temporary.replace(destination)
            state["completed"][filename] = {
                "sha256": writer.digest.hexdigest(),
                "size": expected_size,
            }
            persist_state()
            completed_data += data_bytes
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = completed_data / elapsed
            remaining = serialized_bytes - completed_data
            print(
                f"wrote {filename}: {completed_data / GIB:.2f}/"
                f"{serialized_bytes / GIB:.2f} GiB, {rate / MIB:.1f} MiB/s, "
                f"ETA {remaining / max(rate, 1):.0f}s"
            )
        for entry in shard_entries:
            output_weight_map[str(entry["target_name"])] = filename

    # Persist header receipts even when every data shard was resumed.
    persist_state()

    atomic_json(
        args.output / "model.safetensors.index.json",
        {
            "metadata": {"total_size": serialized_bytes},
            "weight_map": output_weight_map,
        },
    )
    atomic_json(args.output / "config.json", plan["config"])
    shutil.copy2(plan_path, args.output / "surgery_plan.json")
    for filename in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source_path = args.metadata_dir / filename
        if source_path.is_file():
            shutil.copy2(source_path, args.output / filename)
    manifest = {
        "schema_version": 1,
        "builder": "bounded-memory-python-http-range-v2-receipts",
        "pair_id": plan["pair_id"],
        "plan_sha256": plan_sha256,
        "source": plan["source"],
        "target": plan["target"],
        "shards": state["completed"],
        "remote_source_bytes": state["remote_source_bytes"],
        "source_range_receipts": state["source_range_receipts"],
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(args.output / "surgery_manifest.json", manifest)
    write_readme(args.output, plan)
    summary["status"] = "built"
    summary["remote_source_bytes"] = state["remote_source_bytes"]
    summary["source_range_receipt_count"] = len(state["source_range_receipts"])
    return summary


def main() -> None:
    result = build(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
