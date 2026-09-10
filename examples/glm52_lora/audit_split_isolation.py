#!/usr/bin/env python3
"""Audit train/evaluation isolation for GLM-5.2 quality JSONL.

The audit is intentionally independent of example IDs. It exhaustively checks
every cross-split row pair at the configured lexical threshold, catches
prompt/response cross-field reuse, and can resolve source provenance against
one or more immutable external source samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
MAX_SAMPLE_IDS = 8
MAX_STORED_VIOLATIONS = 8
SCHEMA_VERSION = 3
ALGORITHM_VERSION = "exhaustive-cross-split-source-visible-v4"
DEFAULT_SHINGLE_WIDTH = 5
MIN_NEAR_TOKENS = 5
PRODUCTION_NEAR_DUPLICATE_THRESHOLD = 0.7
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class ViolationSummary:
    """Exact violation count with a bounded, deterministically sorted sample."""

    count: int
    samples: list[dict[str, Any]]


@dataclass(frozen=True)
class TextFingerprint:
    shingles: frozenset[str]
    tokens: tuple[str, ...]
    token_counts: Counter[str]


def _near_pair_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (-item["similarity"], item["ids"], item["fields"])


def _record_bounded_near_pair(summary: ViolationSummary, item: dict[str, Any]) -> None:
    summary.count += 1
    summary.samples.append(item)
    summary.samples.sort(key=_near_pair_sort_key)
    del summary.samples[MAX_STORED_VIOLATIONS:]


def _summarize(items: list[dict[str, Any]]) -> ViolationSummary:
    return ViolationSummary(count=len(items), samples=items[:MAX_STORED_VIOLATIONS])


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    # Format controls such as ZERO WIDTH SPACE must not make an otherwise
    # identical value evade exact or shingle matching.
    normalized = "".join(character for character in normalized if unicodedata.category(character) != "Cf")
    return WHITESPACE_RE.sub(" ", normalized).strip().casefold()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return digest.hexdigest()


def auditor_code_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    for index, row in enumerate(rows):
        if row.get("split") not in SPLITS:
            raise ValueError(f"row {index}: invalid split")
        for field in ("id", "prompt", "response"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"row {index}: invalid {field}")
    return rows


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{label} {path}:{line_number}: expected an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty {label}: {path}")
    return rows


def split_pairs(items: list[tuple[str, str]]) -> dict[str, int]:
    counts = Counter(split for split, _ in items)
    return {
        f"{left}-{right}": counts[left] * counts[right]
        for left, right in combinations(SPLITS, 2)
        if counts[left] and counts[right]
    }


def exact_cross_split_groups(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str | None],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        value = key(row)
        if value:
            grouped[value].append((row["split"], row["id"]))
    result = []
    for value, items in grouped.items():
        if len({split for split, _ in items}) < 2:
            continue
        result.append(
            {
                "fingerprint": hashlib.sha256(value.encode()).hexdigest(),
                "id_count": len(items),
                "pair_counts": split_pairs(items),
                "sample_ids": sorted(example_id for _, example_id in items)[:MAX_SAMPLE_IDS],
                "splits": sorted({split for split, _ in items}),
            }
        )
    return sorted(result, key=lambda item: item["fingerprint"])


def exact_cross_field_groups(rows: list[dict[str, Any]], left_field: str, right_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: {left_field: [], right_field: []})
    for row in rows:
        for field in (left_field, right_field):
            grouped[normalize_text(row[field])][field].append((row["split"], row["id"]))

    result: list[dict[str, Any]] = []
    for value, fields in grouped.items():
        left_counts = Counter(split for split, _ in fields[left_field])
        right_counts = Counter(split for split, _ in fields[right_field])
        pair_count = sum(
            left_count * right_counts[right_split]
            for left_split, left_count in left_counts.items()
            for right_split in SPLITS
            if left_split != right_split
        )
        if not pair_count:
            continue
        occurrences = {(split, example_id, field) for field, items in fields.items() for split, example_id in items}
        result.append(
            {
                "fingerprint": hashlib.sha256(value.encode()).hexdigest(),
                "pair_count": pair_count,
                "sample_occurrences": [
                    {"field": field, "id": example_id, "split": split}
                    for split, example_id, field in sorted(occurrences)
                ][:MAX_SAMPLE_IDS],
                "splits": sorted({split for split in SPLITS if left_counts[split] or right_counts[split]}),
            }
        )
    return sorted(result, key=lambda item: item["fingerprint"])


def shingles(value: str, width: int = DEFAULT_SHINGLE_WIDTH) -> frozenset[str]:
    if width < 1:
        raise ValueError("shingle width must be positive")
    tokens = TOKEN_RE.findall(normalize_text(value))
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def _text_fingerprint(value: str, *, width: int) -> TextFingerprint:
    normalized = normalize_text(value)
    tokens = tuple(WORD_RE.findall(normalized))
    return TextFingerprint(
        shingles=shingles(normalized, width=width),
        tokens=tokens,
        token_counts=Counter(tokens),
    )


def _near_similarity(left: TextFingerprint, right: TextFingerprint, *, threshold: float) -> float | None:
    if not left.shingles or not right.shingles:
        return None
    shorter = min(len(left.shingles), len(right.shingles))
    shared = len(left.shingles & right.shingles)
    jaccard = shared / (len(left.shingles) + len(right.shingles) - shared)
    containment = shared / shorter
    token_containment = 0.0
    if min(len(left.tokens), len(right.tokens)) >= MIN_NEAR_TOKENS:
        shared_tokens = sum(min(count, right.token_counts.get(token, 0)) for token, count in left.token_counts.items())
        token_containment = shared_tokens / min(len(left.tokens), len(right.tokens))
    similarity = max(jaccard, containment, token_containment)
    # Keep 1.0: different normalized strings can have identical shingle sets,
    # so the exact-string checks are not a substitute for this comparison.
    if similarity < threshold:
        return None
    return similarity


def near_cross_split_pairs(
    rows: list[dict[str, Any]],
    left_field: str,
    right_field: str | None = None,
    *,
    threshold: float,
    shingle_width: int = DEFAULT_SHINGLE_WIDTH,
) -> ViolationSummary:
    """Exhaustively compare the selected fields for every cross-split row pair."""

    right_field = right_field or left_field
    fields = {left_field, right_field}
    documents = {
        (index, field): _text_fingerprint(row[field], width=shingle_width)
        for index, row in enumerate(rows)
        for field in fields
    }
    indexes_by_split = {split: [index for index, row in enumerate(rows) if row["split"] == split] for split in SPLITS}
    result = ViolationSummary(count=0, samples=[])
    for left_split, right_split in combinations(SPLITS, 2):
        orientations = [(left_field, right_field)]
        if left_field != right_field:
            orientations.append((right_field, left_field))
        for left_index in indexes_by_split[left_split]:
            for right_index in indexes_by_split[right_split]:
                for actual_left_field, actual_right_field in orientations:
                    similarity = _near_similarity(
                        documents[(left_index, actual_left_field)],
                        documents[(right_index, actual_right_field)],
                        threshold=threshold,
                    )
                    if similarity is None:
                        continue
                    _record_bounded_near_pair(
                        result,
                        {
                            "fields": [actual_left_field, actual_right_field],
                            "ids": [rows[left_index]["id"], rows[right_index]["id"]],
                            "similarity": round(similarity, 6),
                            "splits": [left_split, right_split],
                        },
                    )
    return result


def _source_content(record: dict[str, Any]) -> str | None:
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text
    body: list[str] = []
    sentences = record.get("sentences")
    if isinstance(sentences, list):
        body.extend(sentence for sentence in sentences if isinstance(sentence, str) and sentence.strip())
    if not body:
        return None
    title = record.get("title")
    if isinstance(title, str) and title.strip():
        body.insert(0, title)
    return "\n".join(body)


def _source_fragments(record: dict[str, Any], content: str) -> list[str]:
    """Return whole-source and sentence-level views without duplicating text."""

    candidates: list[str] = [content]
    title = record.get("title")
    if isinstance(title, str) and title.strip():
        candidates.append(title)
    sentences = record.get("sentences")
    if isinstance(sentences, list):
        candidates.extend(sentence for sentence in sentences if isinstance(sentence, str) and sentence.strip())
    seen: set[str] = set()
    fragments: list[str] = []
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        fragments.append(candidate)
    return fragments


def source_index_from_records(
    records: Iterable[dict[str, Any]],
    *,
    source_files: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for position, record in enumerate(records):
        dataset = record.get("dataset")
        record_id = record.get("id", record.get("source_record_id"))
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"source record {position}: missing dataset")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"source record {position}: missing id")
        key = (dataset.strip(), record_id.strip())
        if key in indexed:
            raise ValueError(f"duplicate external source identity: {key!r}")
        revision = record.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"source record {position}: missing revision")
        config = record.get("config", record.get("source_split"))
        if not isinstance(config, str) or not config.strip():
            raise ValueError(f"source record {position}: missing config")
        text_sha256 = record.get("text_sha256", record.get("source_text_sha256"))
        if not isinstance(text_sha256, str) or SHA256_RE.fullmatch(text_sha256) is None:
            raise ValueError(f"source record {position}: invalid text SHA-256")
        content = _source_content(record)
        if content is None:
            raise ValueError(f"source record {position}: missing content")
        indexed[key] = {
            "config": config.strip(),
            "content": content,
            "dataset": key[0],
            "id": key[1],
            "revision": revision.strip(),
            "fragments": _source_fragments(record, content),
            "text_sha256": text_sha256,
        }
    return {"files": list(source_files), "records": indexed}


def load_source_samples(
    specs: Sequence[tuple[str, Path, str]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for dataset, path, expected_sha256 in specs:
        if SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError(f"{path}: expected source SHA-256 is invalid")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{path}: source sample SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
            )
        source_rows = read_jsonl(path, "source sample")
        for row in source_rows:
            if row.get("dataset") != dataset:
                raise ValueError(f"{path}: source sample dataset differs from {dataset!r}")
        records.extend(source_rows)
        files.append(
            {
                "dataset": dataset,
                "file": path.name,
                "record_count": len(source_rows),
                "sha256": actual_sha256,
            }
        )
    return source_index_from_records(records, source_files=files)


def _source_record_candidates(source_record_id: str) -> Iterable[str]:
    candidate = source_record_id
    yield candidate
    while ":" in candidate:
        candidate = candidate.rsplit(":", 1)[0]
        yield candidate


def _resolve_source(row: dict[str, Any], source_index: dict[str, Any] | None) -> dict[str, Any] | None:
    if source_index is None:
        return None
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    dataset = provenance.get("dataset")
    source_record_id = provenance.get("source_record_id")
    if not isinstance(dataset, str) or not isinstance(source_record_id, str):
        return None
    records = source_index["records"]
    for candidate in _source_record_candidates(source_record_id):
        source = records.get((dataset, candidate))
        if source is not None:
            return source
    return None


def _provenance_value(row: dict[str, Any], field: str) -> Any:
    provenance = row.get("provenance")
    return provenance.get(field) if isinstance(provenance, dict) else None


def _source_record_identity(row: dict[str, Any]) -> str | None:
    dataset = _provenance_value(row, "dataset")
    source_record_id = _provenance_value(row, "source_record_id")
    if not isinstance(dataset, str) or not dataset.strip():
        return None
    if not isinstance(source_record_id, str) or not source_record_id.strip():
        return None
    # Revision or source-split drift must not hide reuse of the same record.
    return f"{dataset.strip()}\0{source_record_id.strip()}"


def _source_violations(
    rows: list[dict[str, Any]],
    *,
    source_index: dict[str, Any] | None,
    required_source_datasets: frozenset[str],
    inline_source_datasets: frozenset[str],
    near_threshold: float,
    shingle_width: int,
) -> tuple[dict[str, ViolationSummary], dict[str, Any]]:
    configuration_gaps: list[dict[str, Any]] = []
    invalid_hashes: list[dict[str, Any]] = []
    coverage_gaps: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, Any]] = []
    metadata_gaps: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    required_row_count = 0
    resolved_required_row_count = 0

    input_dataset_counts = Counter(
        _provenance_value(row, "dataset") for row in rows if isinstance(_provenance_value(row, "dataset"), str)
    )
    source_dataset_counts: Counter[str] = Counter()
    if source_index is not None:
        source_dataset_counts.update(dataset for dataset, _ in source_index["records"])
    source_contract_enabled = bool(source_index is not None or required_source_datasets or inline_source_datasets)
    for dataset in sorted(inline_source_datasets):
        if input_dataset_counts[dataset] == 0:
            configuration_gaps.append({"dataset": dataset, "reason": "inline dataset has zero input rows"})
        if source_dataset_counts[dataset]:
            configuration_gaps.append(
                {
                    "dataset": dataset,
                    "reason": "dataset cannot be both inline and externally resolved",
                }
            )
    for dataset in sorted(required_source_datasets):
        if input_dataset_counts[dataset] == 0:
            configuration_gaps.append({"dataset": dataset, "reason": "required dataset has zero input rows"})
        if source_index is None:
            configuration_gaps.append({"dataset": dataset, "reason": "source index is unavailable"})
        elif source_dataset_counts[dataset] == 0:
            configuration_gaps.append({"dataset": dataset, "reason": "source sample has zero records"})

    for row in rows:
        provenance = row.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        dataset = provenance.get("dataset")
        claimed_hash = provenance.get("source_text_sha256")
        if claimed_hash is not None and (
            not isinstance(claimed_hash, str) or SHA256_RE.fullmatch(claimed_hash) is None
        ):
            invalid_hashes.append({"id": row["id"], "value": claimed_hash})

        dataset_is_present = isinstance(dataset, str) and bool(dataset.strip())
        has_source_metadata = any(
            provenance.get(field) is not None
            for field in (
                "revision",
                "source_record_id",
                "source_split",
                "source_text_sha256",
            )
        )
        # Every provenance-bearing dataset is externally resolved unless it is
        # explicitly declared inline. This catches misspelled dataset names
        # even when source_text_sha256 was accidentally omitted.
        source_bound = claimed_hash is not None
        required = (
            source_bound
            or (dataset_is_present and dataset not in inline_source_datasets)
            or (not dataset_is_present and has_source_metadata)
            or (source_contract_enabled and not dataset_is_present)
        )
        required_row_count += int(required)
        if required:
            missing_fields = []
            if not dataset_is_present:
                missing_fields.append("dataset")
            for field in ("revision", "source_record_id", "source_split"):
                value = provenance.get(field)
                if not isinstance(value, str) or not value.strip():
                    missing_fields.append(field)
            if not isinstance(claimed_hash, str) or SHA256_RE.fullmatch(claimed_hash) is None:
                missing_fields.append("source_text_sha256")
            if missing_fields:
                metadata_gaps.append({"fields": sorted(set(missing_fields)), "id": row["id"]})
        source = _resolve_source(row, source_index)
        if source is None:
            if required:
                coverage_gaps.append(
                    {
                        "dataset": dataset,
                        "id": row["id"],
                        "source_record_id": provenance.get("source_record_id"),
                    }
                )
            continue
        resolved_required_row_count += int(required)

        mismatched_fields: list[str] = []
        if provenance.get("revision") != source["revision"]:
            mismatched_fields.append("revision")
        if provenance.get("source_split") != source["config"]:
            mismatched_fields.append("source_split")
        if mismatched_fields:
            identity_mismatches.append(
                {
                    "fields": mismatched_fields,
                    "id": row["id"],
                    "source_id": source["id"],
                }
            )

        source_hash = source.get("text_sha256")
        if source_hash is not None and claimed_hash != source_hash:
            hash_mismatches.append(
                {
                    "actual": claimed_hash,
                    "expected": source_hash,
                    "id": row["id"],
                    "source_id": source["id"],
                }
            )
        resolved_rows.append(
            {
                "id": row["id"],
                "source_content": source.get("content"),
                "source_fragments": source.get("fragments"),
                "source_identity": f"{source['dataset']}\0{source['id']}",
                "split": row["split"],
            }
        )

    source_identity_groups = exact_cross_split_groups(resolved_rows, lambda row: row["source_identity"])
    unique_documents: dict[tuple[str, str], dict[str, Any]] = {}
    for row in resolved_rows:
        if not row.get("source_content"):
            continue
        unique_documents[(row["source_identity"], row["split"])] = {
            "id": row["source_identity"].replace("\0", ":"),
            "source_content": row["source_content"],
            "source_fragments": row["source_fragments"],
            "split": row["split"],
        }
    source_documents = list(unique_documents.values())
    exact_source_content = exact_cross_split_groups(source_documents, lambda row: normalize_text(row["source_content"]))
    source_model_rows = [
        {
            "id": row["id"],
            "source_content": row["source_content"],
            "source_fragments": row["source_fragments"],
            "split": row["split"],
        }
        for row in unique_documents.values()
    ]

    source_fragment_shingles = [
        [_text_fingerprint(fragment, width=shingle_width) for fragment in row["source_fragments"]]
        for row in source_model_rows
    ]

    near_source = ViolationSummary(count=0, samples=[])
    indexes_by_split = {
        split: [index for index, row in enumerate(source_model_rows) if row["split"] == split] for split in SPLITS
    }
    for left_split, right_split in combinations(SPLITS, 2):
        for left_index in indexes_by_split[left_split]:
            for right_index in indexes_by_split[right_split]:
                best_similarity: float | None = None
                for left_shingles in source_fragment_shingles[left_index]:
                    for right_shingles in source_fragment_shingles[right_index]:
                        similarity = _near_similarity(
                            left_shingles,
                            right_shingles,
                            threshold=near_threshold,
                        )
                        if similarity is not None and (best_similarity is None or similarity > best_similarity):
                            best_similarity = similarity
                if best_similarity is None:
                    continue
                _record_bounded_near_pair(
                    near_source,
                    {
                        "fields": ["source_fragment", "source_fragment"],
                        "ids": [
                            source_model_rows[left_index]["id"],
                            source_model_rows[right_index]["id"],
                        ],
                        "similarity": round(best_similarity, 6),
                        "splits": [left_split, right_split],
                    },
                )

    visible_rows = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "split": row["split"],
        }
        for row in rows
    ]

    def source_visible_pairs(field: str) -> ViolationSummary:
        result = ViolationSummary(count=0, samples=[])
        visible_shingles = [_text_fingerprint(row[field], width=shingle_width) for row in visible_rows]
        for source_position, source_row in enumerate(source_model_rows):
            for visible_position, visible_row in enumerate(visible_rows):
                if source_row["split"] == visible_row["split"]:
                    continue
                best_similarity: float | None = None
                for source_shingles in source_fragment_shingles[source_position]:
                    similarity = _near_similarity(
                        source_shingles,
                        visible_shingles[visible_position],
                        threshold=near_threshold,
                    )
                    if similarity is not None and (best_similarity is None or similarity > best_similarity):
                        best_similarity = similarity
                if best_similarity is None:
                    continue
                _record_bounded_near_pair(
                    result,
                    {
                        "fields": ["source_fragment", field],
                        "ids": [source_row["id"], visible_row["id"]],
                        "similarity": round(best_similarity, 6),
                        "splits": [source_row["split"], visible_row["split"]],
                    },
                )
        return result

    near_source_prompt = source_visible_pairs("prompt")
    near_source_response = source_visible_pairs("response")
    return (
        {
            "exact_source_content_groups": _summarize(exact_source_content),
            "invalid_source_hash_rows": _summarize(invalid_hashes),
            "near_source_pairs": near_source,
            "near_source_prompt_pairs": near_source_prompt,
            "near_source_response_pairs": near_source_response,
            "source_configuration_gaps": _summarize(configuration_gaps),
            "source_coverage_gaps": _summarize(coverage_gaps),
            "source_hash_mismatches": _summarize(hash_mismatches),
            "source_identity_groups": _summarize(source_identity_groups),
            "source_identity_mismatches": _summarize(identity_mismatches),
            "source_metadata_gaps": _summarize(metadata_gaps),
        },
        {
            "available": source_index is not None,
            "files": [] if source_index is None else source_index["files"],
            "input_dataset_row_counts": {
                dataset: input_dataset_counts[dataset] for dataset in sorted(required_source_datasets)
            },
            "required_datasets": sorted(required_source_datasets),
            "inline_source_datasets": sorted(inline_source_datasets),
            "required_row_count": required_row_count,
            "resolved_required_row_count": resolved_required_row_count,
            "source_dataset_record_counts": {
                dataset: source_dataset_counts[dataset] for dataset in sorted(required_source_datasets)
            },
            "unique_resolved_document_count": len(source_documents),
        },
    )


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    near_threshold: float = PRODUCTION_NEAR_DUPLICATE_THRESHOLD,
    shingle_width: int = DEFAULT_SHINGLE_WIDTH,
    source_index: dict[str, Any] | None = None,
    required_source_datasets: Iterable[str] = (),
    inline_source_datasets: Iterable[str] = (),
    input_file_sha256: str | None = None,
) -> dict[str, Any]:
    if not 0.0 < near_threshold <= 1.0:
        raise ValueError("near threshold must be in (0, 1]")
    if shingle_width < 1:
        raise ValueError("shingle width must be positive")
    if input_file_sha256 is not None and SHA256_RE.fullmatch(input_file_sha256) is None:
        raise ValueError("input file SHA-256 is invalid")
    required_sources = frozenset(required_source_datasets)
    inline_sources = frozenset(inline_source_datasets)
    if required_sources & inline_sources:
        raise ValueError("a dataset cannot be both required external and inline")

    exact_prompt = exact_cross_split_groups(rows, lambda row: normalize_text(row["prompt"]))
    exact_response = exact_cross_split_groups(rows, lambda row: normalize_text(row["response"]))
    exact_prompt_response = exact_cross_field_groups(rows, "prompt", "response")
    source_text = exact_cross_split_groups(
        rows,
        lambda row: _provenance_value(row, "source_text_sha256"),
    )
    source_record = exact_cross_split_groups(
        rows,
        _source_record_identity,
    )
    near_prompt = near_cross_split_pairs(
        rows,
        "prompt",
        threshold=near_threshold,
        shingle_width=shingle_width,
    )
    near_response = near_cross_split_pairs(
        rows,
        "response",
        threshold=near_threshold,
        shingle_width=shingle_width,
    )
    near_prompt_response = near_cross_split_pairs(
        rows,
        "prompt",
        "response",
        threshold=near_threshold,
        shingle_width=shingle_width,
    )
    source_violations, source_coverage = _source_violations(
        rows,
        source_index=source_index,
        required_source_datasets=required_sources,
        inline_source_datasets=inline_sources,
        near_threshold=near_threshold,
        shingle_width=shingle_width,
    )
    complete_violations = {
        "exact_prompt_groups": _summarize(exact_prompt),
        "exact_prompt_response_groups": _summarize(exact_prompt_response),
        "exact_response_groups": _summarize(exact_response),
        "near_prompt_pairs": near_prompt,
        "near_prompt_response_pairs": near_prompt_response,
        "near_response_pairs": near_response,
        "source_record_groups": _summarize(source_record),
        "source_text_groups": _summarize(source_text),
        **source_violations,
    }
    counts = {key: value.count for key, value in complete_violations.items()}
    violations = {key: value.samples for key, value in complete_violations.items()}
    canonical_sha256 = canonical_rows_sha256(rows)
    return {
        "algorithm": {
            "matching": ("exact-and-exhaustive-cross-split-shingle-jaccard-containment-or-token-containment"),
            "minimum_sequence_tokens": MIN_NEAR_TOKENS,
            "near_duplicate_threshold": near_threshold,
            "shingle_width": shingle_width,
            "version": ALGORITHM_VERSION,
        },
        "auditor_code_sha256": auditor_code_sha256(),
        "canonical_rows_sha256": canonical_sha256,
        "counts": counts,
        "input_sha256": input_file_sha256 or canonical_sha256,
        "near_duplicate_threshold": near_threshold,
        "row_count": len(rows),
        "schema_version": SCHEMA_VERSION,
        "shingle_width": shingle_width,
        "stored_violation_limit_per_category": MAX_STORED_VIOLATIONS,
        "source_coverage": source_coverage,
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "status": "PASS" if not any(counts.values()) else "FAIL-SPLIT-ISOLATION",
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=PRODUCTION_NEAR_DUPLICATE_THRESHOLD,
    )
    parser.add_argument("--shingle-width", type=int, default=DEFAULT_SHINGLE_WIDTH)
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.near_threshold <= 1.0:
        raise SystemExit("--near-threshold must be in (0, 1]")
    if args.shingle_width < 1:
        raise SystemExit("--shingle-width must be positive")
    source_specs = [(dataset, Path(path), digest) for dataset, path, digest in args.source_sample]
    source_index = load_source_samples(source_specs) if source_specs else None
    rows = read_rows(args.input_jsonl)
    result = audit_rows(
        rows,
        near_threshold=args.near_threshold,
        shingle_width=args.shingle_width,
        source_index=source_index,
        required_source_datasets=args.require_source_coverage,
        inline_source_datasets=args.allow_inline_source_dataset,
        input_file_sha256=file_sha256(args.input_jsonl),
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.require_clean and result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
