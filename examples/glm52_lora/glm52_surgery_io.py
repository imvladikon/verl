#!/usr/bin/env python3
"""Bounded, receipt-producing I/O primitives for GLM-5.2 surgery tools."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

MIB = 1024**2
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|[*])$")
_COMMIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path, chunk_bytes: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """Read a safetensors header without mapping its tensor payload."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > 64 * MIB:
            raise ValueError(f"implausible safetensors header length: {path}")
        payload = handle.read(header_length)
    if len(payload) != header_length:
        raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(payload.decode("utf-8"))
    if not isinstance(header, dict):
        raise TypeError(f"safetensors header is not an object: {path}")
    return header, 8 + header_length


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
    """Write-through sink that also records an exact payload digest."""

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
    """Read pinned Hub blobs through strict, auditable HTTP range requests."""

    def __init__(
        self,
        repository: str,
        revision: str,
        source_index: dict[str, Any],
        *,
        chunk_bytes: int = 8 * MIB,
        retries: int = 5,
        timeout_seconds: int = 120,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if chunk_bytes <= 0 or retries <= 0 or timeout_seconds <= 0:
            raise ValueError("chunk size, retries, and timeout must be positive")
        if not _COMMIT_REVISION_RE.fullmatch(revision):
            raise ValueError("source revision must be an immutable 40-character commit SHA")
        weight_map = source_index.get("weight_map", source_index)
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("source index has no non-empty weight_map")
        self.repository = repository
        self.revision = revision
        self.weight_map = {str(key): str(value) for key, value in weight_map.items()}
        self.chunk_bytes = chunk_bytes
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen
        self._sleeper = sleeper
        self._headers: dict[str, dict[str, Any]] = {}
        self._header_lengths: dict[str, int] = {}
        self._remote_sizes: dict[str, int] = {}
        self.remote_bytes = 0
        self.range_receipts: list[dict[str, Any]] = []

    def _url(self, shard: str) -> str:
        quoted = urllib.parse.quote(shard, safe="/")
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{quoted}"

    @staticmethod
    def _response_header(response: Any, name: str) -> str | None:
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        value = headers.get(name)
        return None if value is None else str(value)

    def _validated_response_range(
        self, response: Any, *, shard: str, requested_start: int, requested_end: int
    ) -> tuple[int, int, int, str]:
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "getcode"):
            status = response.getcode()
        if status != 206:
            raise RuntimeError(f"range request returned HTTP {status} for {shard}; expected 206")
        raw = self._response_header(response, "Content-Range")
        match = _CONTENT_RANGE_RE.fullmatch(raw or "")
        if not match:
            raise RuntimeError(f"missing or invalid Content-Range for {shard}: {raw!r}")
        response_start, response_end = map(int, match.group(1, 2))
        if match.group(3) == "*":
            raise RuntimeError(f"Content-Range has unknown total size for {shard}")
        total = int(match.group(3))
        if response_start != requested_start:
            raise RuntimeError(f"Content-Range starts at {response_start}, expected {requested_start} for {shard}")
        if response_end < response_start or response_end >= requested_end:
            raise RuntimeError(
                f"Content-Range end {response_end} escapes requested half-open "
                f"range [{requested_start}, {requested_end}) for {shard}"
            )
        if total <= response_end or requested_end > total:
            raise RuntimeError(f"invalid Content-Range total {total} for {shard}")
        content_encoding = self._response_header(response, "Content-Encoding")
        if content_encoding not in (None, "", "identity"):
            raise RuntimeError(f"range response uses unsupported Content-Encoding {content_encoding!r} for {shard}")
        declared = response_end - response_start + 1
        content_length = self._response_header(response, "Content-Length")
        if content_length is not None and int(content_length) != declared:
            raise RuntimeError(
                f"Content-Length {content_length} disagrees with Content-Range length {declared} for {shard}"
            )
        prior_total = self._remote_sizes.setdefault(shard, total)
        if prior_total != total:
            raise RuntimeError(f"Content-Range total changed for {shard}: {prior_total} != {total}")
        return response_start, response_end, total, raw or ""

    def read_range(self, shard: str, start: int, end: int) -> bytes:
        output = BytesIO()
        self.copy_range(shard, start, end, DigestWriter(output))
        return output.getvalue()

    def copy_range(self, shard: str, start: int, end: int, writer: DigestWriter) -> None:
        if start < 0 or end <= start:
            raise ValueError(f"invalid half-open range {start}:{end} for {shard}")
        if shard not in set(self.weight_map.values()):
            raise ValueError(f"shard is not bound by the source index: {shard}")
        position = start
        failures = 0
        while position < end:
            request_start = position
            request = urllib.request.Request(
                self._url(shard),
                headers={
                    "Range": f"bytes={request_start}-{end - 1}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "glm52-lora-surgery/2.0",
                },
            )
            response_digest = hashlib.sha256()
            response_bytes = 0
            claimed_end: int | None = None
            content_range: str | None = None
            total: int | None = None
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    _, claimed_end, total, content_range = self._validated_response_range(
                        response,
                        shard=shard,
                        requested_start=request_start,
                        requested_end=end,
                    )
                    declared = claimed_end - request_start + 1
                    try:
                        while response_bytes < declared:
                            remaining = declared - response_bytes
                            chunk = response.read(min(self.chunk_bytes, remaining))
                            if not chunk:
                                break
                            if len(chunk) > remaining:
                                raise RuntimeError(f"range response exceeded Content-Range for {shard}")
                            writer.write(chunk)
                            response_digest.update(chunk)
                            response_bytes += len(chunk)
                            position += len(chunk)
                    finally:
                        complete = response_bytes == declared
                        if response_bytes:
                            self.range_receipts.append(
                                {
                                    "repository": self.repository,
                                    "revision": self.revision,
                                    "shard": shard,
                                    "start": request_start,
                                    "end_exclusive": request_start + response_bytes,
                                    "requested_end_exclusive": end,
                                    "content_range": content_range,
                                    "reported_total_bytes": total,
                                    "bytes": response_bytes,
                                    "sha256": response_digest.hexdigest(),
                                    "complete_response": complete,
                                }
                            )
                    if not complete:
                        raise OSError(f"truncated range response for {shard}: {response_bytes}/{declared} bytes")
                failures = 0
            except (OSError, RuntimeError, urllib.error.URLError, ValueError) as error:
                failures += 1
                if failures >= self.retries:
                    raise RuntimeError(f"failed at {shard} byte {position} of {end}") from error
                self._sleeper(min(2**failures, 12))
        transferred = position - start
        if transferred != end - start:
            raise RuntimeError(f"range transfer length mismatch for {shard}: {transferred} != {end - start}")
        self.remote_bytes += transferred

    def header(self, shard: str) -> dict[str, Any]:
        if shard in self._headers:
            return self._headers[shard]
        prefix = self.read_range(shard, 0, 8)
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > 64 * MIB:
            raise ValueError(f"implausible header length {header_length} in {shard}")
        payload = self.read_range(shard, 8, 8 + header_length)
        header = json.loads(payload.decode("utf-8"))
        if not isinstance(header, dict):
            raise TypeError(f"safetensors header is not an object: {shard}")
        self._headers[shard] = header
        self._header_lengths[shard] = header_length
        return header

    def location(self, name: str) -> TensorLocation:
        shard = self.weight_map[name]
        header = self.header(shard)
        if name not in header:
            raise KeyError(f"{name} is indexed into {shard} but absent from its header")
        metadata = header[name]
        data_start = 8 + self._header_lengths[shard]
        relative_start, relative_end = map(int, metadata["data_offsets"])
        location = TensorLocation(
            name=name,
            shard=shard,
            dtype=str(metadata["dtype"]),
            shape=tuple(map(int, metadata["shape"])),
            file_start=data_start + relative_start,
            file_end=data_start + relative_end,
        )
        try:
            expected = location.numel * DTYPE_BYTES[location.dtype]
        except KeyError as error:
            raise ValueError(f"unsupported dtype {location.dtype} for {name}") from error
        if location.nbytes != expected:
            raise ValueError(f"shape/byte mismatch for {name}: {location}")
        return location

    def tensor(self, name: str) -> np.ndarray:
        location = self.location(name)
        payload = self.read_range(location.shard, location.file_start, location.file_end)
        if location.dtype == "BF16":
            bits = np.frombuffer(payload, dtype="<u2").astype("<u4") << 16
            return bits.view("<f4").reshape(location.shape)
        dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
        if location.dtype not in dtypes:
            raise ValueError(f"cannot decode {location.dtype} tensor {name}")
        return np.frombuffer(payload, dtype=dtypes[location.dtype]).reshape(location.shape)


def select_router_medoids(rows: np.ndarray, correction_bias: np.ndarray, count: int) -> dict[str, Any]:
    """Select deterministic cosine medoids and retain their source clusters."""
    rows = np.asarray(rows, dtype=np.float32)
    correction_bias = np.asarray(correction_bias, dtype=np.float32).reshape(-1)
    if rows.ndim != 2 or rows.shape[0] != correction_bias.size:
        raise ValueError("router weights and correction bias have incompatible shapes")
    if not np.isfinite(rows).all() or not np.isfinite(correction_bias).all():
        raise ValueError("router geometry contains non-finite values")
    if count <= 0 or count > rows.shape[0]:
        raise ValueError("invalid target medoid count")

    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit = rows / np.maximum(norms, 1e-12)
    pair_distance = np.maximum(0.0, 1.0 - unit @ unit.T)
    centroid = unit.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    medoids = [int(np.argmax(unit @ centroid))]
    nearest = pair_distance[:, medoids[0]].copy()
    nearest[medoids[0]] = -1.0
    while len(medoids) < count:
        candidate = int(np.argmax(nearest))
        medoids.append(candidate)
        nearest = np.minimum(nearest, pair_distance[:, candidate])
        nearest[medoids] = -1.0

    # A small deterministic PAM refinement makes the selected rows true medoids,
    # while selection itself remains driven by the router vectors. Bias is kept
    # as associated routing state and is averaged over the same source clusters.
    for _ in range(16):
        assignments = np.argmin(pair_distance[:, medoids], axis=1)
        assignments[np.asarray(medoids)] = np.arange(count)
        updated: list[int] = []
        for cluster_index, fallback in enumerate(medoids):
            members = np.flatnonzero(assignments == cluster_index)
            if members.size == 0:
                updated.append(fallback)
                continue
            costs = pair_distance[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(costs))]))
        if updated == medoids:
            break
        if len(set(updated)) != count:
            raise RuntimeError("medoid refinement produced duplicate experts")
        medoids = updated

    order = np.argsort(medoids)
    medoids = [medoids[int(index)] for index in order]
    assignments = np.argmin(pair_distance[:, medoids], axis=1)
    assignments[np.asarray(medoids)] = np.arange(count)
    clusters = [np.flatnonzero(assignments == index).astype(int).tolist() for index in range(count)]
    if any(not cluster for cluster in clusters):
        raise RuntimeError("router medoid selection produced an empty cluster")
    objective = float(
        sum(pair_distance[member, medoids[index]] for index, cluster in enumerate(clusters) for member in cluster)
    )
    return {
        "method": "deterministic-cosine-pam-v1",
        "selected_source_experts": medoids,
        "source_clusters": clusters,
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "cosine_distance_objective": objective,
        "selected_correction_bias": correction_bias[medoids].astype(float).tolist(),
    }
