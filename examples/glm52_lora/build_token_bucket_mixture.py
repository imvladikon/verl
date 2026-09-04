#!/usr/bin/env python3
"""Build an exact-hash, no-truncation quality mixture in token-length buckets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_quality_tokens import summarize, token_count
from audit_split_isolation import (
    PRODUCTION_NEAR_DUPLICATE_THRESHOLD,
    audit_rows,
    load_source_samples,
)
from build_quality_dataset import read_jsonl, validate_rows, write_artifacts


@dataclass(frozen=True)
class InputSpec:
    label: str
    path: Path
    expected_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_buckets(values: Iterable[int]) -> tuple[int, ...]:
    buckets = tuple(sorted(set(values)))
    if not buckets or buckets[0] <= 0:
        raise ValueError("token buckets must be positive integers")
    return buckets


def full_chat_tokens(row: dict[str, Any], tokenizer: Any) -> int:
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return token_count(encoded)


def partition_rows(
    rows: list[dict[str, Any]], tokenizer: Any, buckets: Iterable[int]
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[int]]]:
    normalized = normalize_buckets(buckets)
    partition = {bucket: [] for bucket in normalized}
    lengths = {bucket: [] for bucket in normalized}
    for row in rows:
        length = full_chat_tokens(row, tokenizer)
        bucket = next((candidate for candidate in normalized if length <= candidate), None)
        if bucket is None:
            raise ValueError(f"{row['id']}: {length} tokens exceed largest bucket {normalized[-1]}")
        partition[bucket].append(row)
        lengths[bucket].append(length)
    return partition, lengths


def load_locked_inputs(
    specs: list[InputSpec],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("input labels must be unique")
    raw_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}
    for spec in specs:
        actual = sha256(spec.path)
        if actual != spec.expected_sha256:
            raise ValueError(f"{spec.label}: SHA-256 mismatch: expected={spec.expected_sha256} actual={actual}")
        source_rows = read_jsonl(spec.path)
        raw_rows.extend(source_rows)
        source_manifest[spec.label] = {
            "file": spec.path.name,
            "sha256": actual,
            "rows": len(source_rows),
        }
    return validate_rows(raw_rows), source_manifest


def enforce_split_isolation(
    rows: list[dict[str, Any]],
    *,
    source_index: dict[str, Any] | None = None,
    required_source_datasets: Iterable[str] = (),
    inline_source_datasets: Iterable[str] = (),
) -> dict[str, Any]:
    audit = audit_rows(
        rows,
        near_threshold=PRODUCTION_NEAR_DUPLICATE_THRESHOLD,
        source_index=source_index,
        required_source_datasets=required_source_datasets,
        inline_source_datasets=inline_source_datasets,
    )
    if audit["status"] != "PASS":
        raise ValueError("cross-split content leakage detected: " + json.dumps(audit["counts"], sort_keys=True))
    return audit


def split_isolation_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": audit["algorithm"],
        "auditor_code_sha256": audit["auditor_code_sha256"],
        "canonical_rows_sha256": audit["canonical_rows_sha256"],
        "counts": audit["counts"],
        "input_sha256": audit["input_sha256"],
        "near_duplicate_threshold": audit["near_duplicate_threshold"],
        "row_count": audit["row_count"],
        "schema_version": audit["schema_version"],
        "shingle_width": audit["shingle_width"],
        "source_coverage": audit["source_coverage"],
        "split_counts": audit["split_counts"],
        "status": audit["status"],
        "stored_violation_limit_per_category": audit["stored_violation_limit_per_category"],
    }


def write_bucket(
    bucket: int,
    rows: list[dict[str, Any]],
    lengths: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    bucket_dir = output_dir / f"seq{bucket}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    rows_path = bucket_dir / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    artifact_manifest = write_artifacts(rows, bucket_dir)
    result = {
        **artifact_manifest,
        "bucket_max_tokens": bucket,
        "observed_full_chat_tokens": summarize(lengths),
        "rows_sha256": sha256(rows_path),
        "truncation": "error",
    }
    (bucket_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "rows": len(rows),
        "counts": result["counts"],
        "observed_full_chat_tokens": result["observed_full_chat_tokens"],
        "rows_sha256": result["rows_sha256"],
    }


def build(
    specs: list[InputSpec],
    output_dir: Path,
    tokenizer_path: Path,
    buckets: Iterable[int],
    *,
    expected_tokenizer_json_sha256: str,
    expected_tokenizer_config_sha256: str,
    source_index: dict[str, Any] | None = None,
    required_source_datasets: Iterable[str] = (),
    inline_source_datasets: Iterable[str] = (),
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer_json = tokenizer_path / "tokenizer.json"
    tokenizer_config = tokenizer_path / "tokenizer_config.json"
    for path, expected in (
        (tokenizer_json, expected_tokenizer_json_sha256),
        (tokenizer_config, expected_tokenizer_config_sha256),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"tokenizer lock mismatch for {path.name}: expected={expected} actual={actual}")

    rows, source_manifest = load_locked_inputs(specs)
    split_isolation_audit = enforce_split_isolation(
        rows,
        source_index=source_index,
        required_source_datasets=required_source_datasets,
        inline_source_datasets=inline_source_datasets,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    partition, lengths = partition_rows(rows, tokenizer, buckets)
    if any(not bucket_rows for bucket_rows in partition.values()):
        empty = [bucket for bucket, bucket_rows in partition.items() if not bucket_rows]
        raise ValueError(f"empty token buckets: {empty}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mixture_path = output_dir / "mixture_rows.jsonl"
    mixture_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    mixture_rows_sha256 = sha256(mixture_path)
    if split_isolation_audit["canonical_rows_sha256"] != mixture_rows_sha256:
        raise AssertionError("split-isolation audit is not bound to mixture_rows.jsonl")
    audit_path = output_dir / "split_isolation_audit.json"
    audit_path.write_text(
        json.dumps(split_isolation_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_isolation_audit_sha256 = sha256(audit_path)
    split_isolation = split_isolation_summary(split_isolation_audit)
    bucket_manifest = {
        str(bucket): write_bucket(bucket, partition[bucket], lengths[bucket], output_dir) for bucket in partition
    }
    manifest = {
        "schema_version": 1,
        "status": "DATA-ENGINEERING-PASS/CONTENT-AND-FULL-MODEL-RUNTIME-PENDING",
        "sources": source_manifest,
        "total_rows": len(rows),
        "mixture_rows_sha256": mixture_rows_sha256,
        "source_counts": dict(sorted(Counter(row["provenance"]["dataset"] for row in rows).items())),
        "split_isolation": split_isolation,
        "split_isolation_audit_sha256": split_isolation_audit_sha256,
        "tokenizer_source": tokenizer_path.name,
        "tokenizer_json_sha256": expected_tokenizer_json_sha256,
        "tokenizer_config_sha256": expected_tokenizer_config_sha256,
        "buckets": bucket_manifest,
        "production_gates": [
            "license/distribution review",
            "sampled-content review",
            "real-checkpoint held-out base-vs-adapter evaluation",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("tokenizer_path", type=Path)
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("LABEL", "JSONL", "SHA256"),
        required=True,
    )
    parser.add_argument("--bucket", action="append", type=int, required=True)
    parser.add_argument("--tokenizer-json-sha256", required=True)
    parser.add_argument("--tokenizer-config-sha256", required=True)
    parser.add_argument(
        "--source-sample",
        action="append",
        nargs=3,
        metavar=("DATASET", "JSONL", "SHA256"),
        default=[],
    )
    parser.add_argument(
        "--require-source-coverage",
        action="append",
        metavar="DATASET",
        default=[],
    )
    parser.add_argument(
        "--allow-inline-source-dataset",
        action="append",
        metavar="DATASET",
        default=[],
    )
    args = parser.parse_args()
    specs = [InputSpec(label, Path(path), digest) for label, path, digest in args.input]
    source_specs = [(dataset, Path(path), digest) for dataset, path, digest in args.source_sample]
    source_index = load_source_samples(source_specs) if source_specs else None
    manifest = build(
        specs,
        args.output_dir,
        args.tokenizer_path,
        args.bucket,
        expected_tokenizer_json_sha256=args.tokenizer_json_sha256,
        expected_tokenizer_config_sha256=args.tokenizer_config_sha256,
        source_index=source_index,
        required_source_datasets=args.require_source_coverage,
        inline_source_datasets=args.allow_inline_source_dataset,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
