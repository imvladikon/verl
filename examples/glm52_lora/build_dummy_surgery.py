#!/usr/bin/env python3
"""Build a full-width GLM-5.2 LoRA/sharding dummy with bounded memory.

The builder never instantiates or downloads the 1.5 TB donor.  It reads small
safetensors headers, selects router-diverse experts, and copies only the exact
HTTP byte ranges needed by the 10-layer/16-expert target.  Output shards are
written atomically and the operation is resumable at shard boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

DEFAULT_REPOSITORY = "zai-org/GLM-5.2"
DEFAULT_REVISION = "cf457fa734ab149ffef225f80893eb38c6ff5cdc"
DEFAULT_LAYER_MAP = (0, 1, 2, 3, 15, 27, 38, 51, 63, 77)
TARGET_EXPERTS = 16
EXPECTED_PARAMETERS = 8_763_269_120
EXPECTED_CHECKPOINT_ELEMENTS = 8_763_269_232
MIB = 1024**2
GIB = 1024**3
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-shard-gib", type=float, default=2.0)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine MemAvailable")


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    file_start: int
    file_end: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.file_end - self.file_start


class DigestWriter:
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, payload: bytes | bytearray | memoryview) -> None:
        written = self.handle.write(payload)
        if written != len(payload):
            raise OSError(f"short output write: {written} != {len(payload)}")
        self.digest.update(payload)
        self.bytes_written += written


class HubRangeReader:
    def __init__(
        self,
        repository: str,
        revision: str,
        weight_map: dict[str, str],
        *,
        chunk_bytes: int,
        retries: int = 5,
        timeout_seconds: int = 120,
    ) -> None:
        self.repository = repository
        self.revision = revision
        self.weight_map = weight_map
        self.chunk_bytes = chunk_bytes
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self._headers: dict[str, dict[str, Any]] = {}
        self._header_lengths: dict[str, int] = {}
        self.remote_bytes = 0

    def _url(self, shard: str) -> str:
        quoted = urllib.parse.quote(shard, safe="/")
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{quoted}"

    def read_range(self, shard: str, start: int, end: int) -> bytes:
        if start < 0 or end <= start:
            raise ValueError(f"invalid range {start}:{end} for {shard}")
        from io import BytesIO

        output = BytesIO()
        writer = DigestWriter(output)
        self.copy_range(shard, start, end, writer)
        return output.getvalue()

    def copy_range(
        self, shard: str, start: int, end: int, writer: DigestWriter
    ) -> None:
        position = start
        failures = 0
        while position < end:
            request = urllib.request.Request(
                self._url(shard),
                headers={
                    "Range": f"bytes={position}-{end - 1}",
                    "User-Agent": "glm52-lora-surgery/1.0",
                },
            )
            before = position
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"range request returned HTTP {response.status} for {shard}"
                        )
                    while position < end:
                        chunk = response.read(min(self.chunk_bytes, end - position))
                        if not chunk:
                            break
                        writer.write(chunk)
                        position += len(chunk)
                if position == before:
                    raise OSError(f"zero-byte range response for {shard}")
                failures = 0
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                failures += 1
                if failures >= self.retries:
                    raise RuntimeError(
                        f"failed at {shard} byte {position} of {end}"
                    ) from error
                time.sleep(min(2**failures, 12))
        self.remote_bytes += end - start

    def header(self, shard: str) -> dict[str, Any]:
        if shard in self._headers:
            return self._headers[shard]
        prefix = self.read_range(shard, 0, 8)
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > 64 * MIB:
            raise ValueError(f"implausible header length {header_length} in {shard}")
        payload = self.read_range(shard, 8, 8 + header_length)
        header = json.loads(payload.decode("utf-8"))
        self._headers[shard] = header
        self._header_lengths[shard] = header_length
        return header

    def location(self, name: str) -> TensorLocation:
        shard = self.weight_map[name]
        metadata = self.header(shard)[name]
        data_start = 8 + self._header_lengths[shard]
        relative_start, relative_end = metadata["data_offsets"]
        location = TensorLocation(
            name=name,
            shard=shard,
            dtype=metadata["dtype"],
            shape=tuple(metadata["shape"]),
            file_start=data_start + relative_start,
            file_end=data_start + relative_end,
        )
        expected = location.numel * DTYPE_BYTES[location.dtype]
        if location.nbytes != expected:
            raise ValueError(f"shape/byte mismatch for {name}: {location}")
        return location

    def tensor(self, name: str) -> np.ndarray:
        location = self.location(name)
        payload = self.read_range(
            location.shard, location.file_start, location.file_end
        )
        if location.dtype == "BF16":
            bits = np.frombuffer(payload, dtype="<u2").astype("<u4") << 16
            return bits.view("<f4").reshape(location.shape)
        dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
        if location.dtype not in dtypes:
            raise ValueError(f"cannot decode {location.dtype} tensor {name}")
        return np.frombuffer(payload, dtype=dtypes[location.dtype]).reshape(
            location.shape
        )


def router_maximin(rows: np.ndarray, count: int) -> list[int]:
    rows = rows.astype(np.float32, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit = rows / np.maximum(norms, 1e-12)
    centroid = unit.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    selected = [int(np.argmax(unit @ centroid))]
    min_distance = 1.0 - unit @ unit[selected[0]]
    min_distance[selected[0]] = -1.0
    while len(selected) < count:
        candidate = int(np.argmax(min_distance))
        selected.append(candidate)
        min_distance = np.minimum(min_distance, 1.0 - unit @ unit[candidate])
        min_distance[selected] = -1.0
    return sorted(selected)


def encode_rows(array: np.ndarray, dtype: str, indices: list[int]) -> bytes:
    selected = np.ascontiguousarray(array[indices])
    if dtype == "BF16":
        bits = selected.astype("<f4", copy=False).view("<u4") >> 16
        return bits.astype("<u2").tobytes()
    dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
    if dtype not in dtypes:
        raise ValueError(f"cannot encode gathered {dtype} rows")
    return selected.astype(dtypes[dtype], copy=False).tobytes()


def safetensors_prefix(entries: list[dict[str, Any]]) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for entry in entries:
        nbytes = int(entry["nbytes"])
        header[entry["target_name"]] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    return struct.pack("<Q", len(payload)) + payload


def group_shards(
    entries: list[dict[str, Any]], max_data_bytes: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for entry in sorted(entries, key=lambda item: item["target_name"]):
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


def target_config(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config["num_hidden_layers"] = len(DEFAULT_LAYER_MAP)
    config["n_routed_experts"] = TARGET_EXPERTS
    config["num_nextn_predict_layers"] = 0
    config["indexer_types"] = [
        source["indexer_types"][layer] for layer in DEFAULT_LAYER_MAP
    ]
    config["mlp_layer_types"] = [
        source["mlp_layer_types"][layer] for layer in DEFAULT_LAYER_MAP
    ]
    if source.get("layer_types") is not None:
        config["layer_types"] = [
            source["layer_types"][layer] for layer in DEFAULT_LAYER_MAP
        ]
    return config


def build_plan(
    source_config: dict[str, Any],
    weight_map: dict[str, str],
    reader: HubRangeReader,
) -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[str, bytes]]:
    expert_selection: dict[int, list[int]] = {}
    generated: dict[str, bytes] = {}
    for source_layer in DEFAULT_LAYER_MAP:
        if source_config["mlp_layer_types"][source_layer] != "sparse":
            continue
        gate_name = f"model.layers.{source_layer}.mlp.gate.weight"
        gate = reader.tensor(gate_name)
        expert_selection[source_layer] = router_maximin(gate, TARGET_EXPERTS)

    source_to_target_layer = {
        source: target for target, source in enumerate(DEFAULT_LAYER_MAP)
    }
    entries: list[dict[str, Any]] = []
    for source_name in sorted(weight_map):
        parts = source_name.split(".")
        if len(parts) >= 3 and parts[:2] == ["model", "layers"]:
            try:
                source_layer = int(parts[2])
            except ValueError:
                continue
            if source_layer not in source_to_target_layer:
                continue
            target_layer = source_to_target_layer[source_layer]
            target_name = ".".join(
                ["model", "layers", str(target_layer), *parts[3:]]
            )
            if ".mlp.experts." in source_name:
                expert_position = parts.index("experts") + 1
                source_expert = int(parts[expert_position])
                selected = expert_selection[source_layer]
                if source_expert not in selected:
                    continue
                target_expert = selected.index(source_expert)
                target_parts = target_name.split(".")
                target_parts[expert_position] = str(target_expert)
                target_name = ".".join(target_parts)
        else:
            target_name = source_name

        location = reader.location(source_name)
        shape = list(location.shape)
        operation = "copy"
        if source_name.endswith(".mlp.gate.weight") or source_name.endswith(
            ".mlp.gate.e_score_correction_bias"
        ):
            source_layer = int(parts[2])
            selected = expert_selection[source_layer]
            payload = encode_rows(reader.tensor(source_name), location.dtype, selected)
            shape[0] = TARGET_EXPERTS
            generated[target_name] = payload
            operation = "gather_router_rows"
            nbytes = len(payload)
        else:
            nbytes = location.nbytes
        entries.append(
            {
                "target_name": target_name,
                "source_name": source_name,
                "source_shard": location.shard,
                "file_start": location.file_start,
                "file_end": location.file_end,
                "dtype": location.dtype,
                "shape": shape,
                "nbytes": nbytes,
                "operation": operation,
            }
        )
    if len({entry["target_name"] for entry in entries}) != len(entries):
        raise ValueError("target tensor names are not unique")
    return entries, expert_selection, generated


def write_readme(output: Path, parameters: int) -> None:
    text = f"""# GLM-5.2-9B-LoRA-Surgery-Dummy

Test-only checkpoint for GLM-5.2 post-training, LoRA, weight-sync, and
sharding integration. It is not a usable chat or benchmark model.

The checkpoint preserves the donor width and MLA/DSA geometry while reducing
78 decoder layers to 10 and 256 routed experts to 16. It contains
{parameters:,} parameters. Layer and expert provenance is recorded in
`surgery_manifest.json`; experts are selected independently per MoE layer by a
deterministic cosine-maximin traversal of the donor router rows.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_shard_gib <= 0 or args.chunk_mib <= 0:
        raise ValueError("max-shard-gib and chunk-mib must be positive")
    source_config_path = args.metadata_dir / "config.json"
    source_index_path = args.metadata_dir / "model.safetensors.index.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    weight_map = source_index["weight_map"]
    reader = HubRangeReader(
        args.repository,
        args.revision,
        weight_map,
        chunk_bytes=args.chunk_mib * MIB,
    )
    entries, expert_selection, generated = build_plan(
        source_config, weight_map, reader
    )
    checkpoint_elements = sum(math.prod(entry["shape"]) for entry in entries)
    correction_bias_elements = sum(
        math.prod(entry["shape"])
        for entry in entries
        if entry["target_name"].endswith(".mlp.gate.e_score_correction_bias")
    )
    parameters = checkpoint_elements - correction_bias_elements
    if parameters != EXPECTED_PARAMETERS:
        raise AssertionError(f"parameter count {parameters} != {EXPECTED_PARAMETERS}")
    if checkpoint_elements != EXPECTED_CHECKPOINT_ELEMENTS:
        raise AssertionError(
            f"checkpoint elements {checkpoint_elements} != {EXPECTED_CHECKPOINT_ELEMENTS}"
        )
    total_data_bytes = sum(int(entry["nbytes"]) for entry in entries)
    shards = group_shards(entries, int(args.max_shard_gib * GIB))
    plan = {
        "repository": args.repository,
        "revision": args.revision,
        "layer_map": list(DEFAULT_LAYER_MAP),
        "expert_selection": {str(k): v for k, v in expert_selection.items()},
        "target_experts": TARGET_EXPERTS,
        "parameters": parameters,
        "checkpoint_elements": checkpoint_elements,
        "buffer_elements": correction_bias_elements,
        "data_bytes": total_data_bytes,
        "tensor_count": len(entries),
        "output_shards": len(shards),
        "source_config_sha256": sha256_file(source_config_path),
        "source_index_sha256": sha256_file(source_index_path),
        "entries": entries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "surgery_plan.json", plan)
    print(
        f"planned {parameters:,} parameters, {total_data_bytes / GIB:.2f} GiB, "
        f"{len(entries)} tensors, {len(shards)} shards"
    )
    if not args.execute:
        print("dry run only; pass --execute to transfer weight ranges")
        return

    free_disk = shutil.disk_usage(args.output).free
    if free_disk < total_data_bytes + 2 * GIB:
        raise OSError(
            f"insufficient disk: {free_disk / GIB:.1f} GiB free for "
            f"{total_data_bytes / GIB:.1f} GiB output"
        )
    if available_memory_bytes() < 512 * MIB:
        raise MemoryError("less than 512 MiB MemAvailable before surgery")

    state_path = args.output / ".surgery_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"completed": {}}
    )
    output_weight_map: dict[str, str] = {}
    started = time.monotonic()
    completed_data = 0
    shard_count = len(shards)
    for shard_index, shard_entries in enumerate(shards, start=1):
        filename = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        destination = args.output / filename
        prefix = safetensors_prefix(shard_entries)
        expected_size = len(prefix) + sum(int(entry["nbytes"]) for entry in shard_entries)
        recorded = state["completed"].get(filename)
        if (
            recorded
            and destination.is_file()
            and destination.stat().st_size == expected_size
            and sha256_file(destination) == recorded["sha256"]
        ):
            completed_data += sum(int(entry["nbytes"]) for entry in shard_entries)
            print(f"resume {filename}: verified")
        else:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with temporary.open("wb") as handle:
                writer = DigestWriter(handle)
                writer.write(prefix)
                for entry in shard_entries:
                    payload = generated.get(entry["target_name"])
                    if payload is not None:
                        writer.write(payload)
                    else:
                        reader.copy_range(
                            entry["source_shard"],
                            int(entry["file_start"]),
                            int(entry["file_end"]),
                            writer,
                        )
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != expected_size:
                raise OSError(
                    f"wrong output size for {filename}: "
                    f"{temporary.stat().st_size} != {expected_size}"
                )
            temporary.replace(destination)
            state["completed"][filename] = {
                "sha256": writer.digest.hexdigest(),
                "bytes": expected_size,
            }
            atomic_json(state_path, state)
            completed_data += sum(int(entry["nbytes"]) for entry in shard_entries)
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = completed_data / elapsed
            remaining = total_data_bytes - completed_data
            print(
                f"wrote {filename}: {completed_data / GIB:.2f}/{total_data_bytes / GIB:.2f} GiB, "
                f"{rate / MIB:.1f} MiB/s, ETA {remaining / max(rate, 1):.0f}s"
            )
        for entry in shard_entries:
            output_weight_map[entry["target_name"]] = filename

    atomic_json(
        args.output / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_data_bytes},
            "weight_map": output_weight_map,
        },
    )
    atomic_json(args.output / "config.json", target_config(source_config))
    for filename in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source = args.metadata_dir / filename
        if source.is_file():
            shutil.copy2(source, args.output / filename)
    manifest = {
        key: value for key, value in plan.items() if key != "entries"
    }
    manifest["shards"] = state["completed"]
    manifest["remote_bytes_read"] = reader.remote_bytes
    atomic_json(args.output / "surgery_manifest.json", manifest)
    write_readme(args.output, parameters)
    print(f"completed {args.output}")


if __name__ == "__main__":
    main()
