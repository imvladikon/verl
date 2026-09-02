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
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

SCHEMA_VERSION = 1
METHOD = "blinded-human-rubric-v1"
RATING_FIELDS = (
    "meaning_preservation",
    "russian_naturalness",
    "factual_accuracy",
    "instruction_fulfillment",
)


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
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise FileExistsError(
            f"refusing to overwrite existing output(s): {existing}; pass --overwrite explicitly"
        )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _pair_is_flipped(key: bytes, example_id: str) -> bool:
    return bool(hmac.new(key, example_id.encode("utf-8"), hashlib.sha256).digest()[0] & 1)


def _validate_pair(
    example_id: str,
    source: dict[str, Any],
    base: dict[str, Any],
    adapter: dict[str, Any],
) -> None:
    runtime_pair_hash = base.get("generation_pair_contract_sha256")
    if not is_sha256(runtime_pair_hash):
        raise ValueError(f"{example_id}: base generation pair contract must be a SHA-256 digest")
    if runtime_pair_hash != adapter.get("generation_pair_contract_sha256"):
        raise ValueError(f"{example_id}: base and adapter generation pair contracts differ")
    for label, prediction in (("base", base), ("adapter", adapter)):
        completion = prediction.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{example_id}: {label} completion must be a nonempty string")
        if prediction.get("contract") != source.get("contract"):
            raise ValueError(f"{example_id}: {label} quality contract differs from held-out source")
        if prediction.get("prompt_sha256") != source.get("prompt_sha256"):
            raise ValueError(f"{example_id}: {label} prompt hash differs from held-out source")
        expected_messages_hash = canonical_sha256(_request_messages(source))
        if prediction.get("request_messages_sha256") != expected_messages_hash:
            raise ValueError(f"{example_id}: {label} request messages differ from held-out source")
        generation = prediction.get("generation")
        if not isinstance(generation, dict):
            raise TypeError(f"{example_id}: {label} generation provenance must be an object")
        if generation.get("variant") != label:
            raise ValueError(f"{example_id}: {label} generation variant is invalid")
        if generation.get("runtime_manifest_sha256") != runtime_pair_hash:
            raise ValueError(f"{example_id}: {label} runtime manifest hash differs from the pair contract")
        if generation.get("quality_claim_allowed") is not True:
            raise ValueError(f"{example_id}: {label} runtime is not a full-model quality oracle")
    decoding_hash = base.get("decoding_contract_sha256")
    if not is_sha256(decoding_hash):
        raise ValueError(f"{example_id}: base decoding contract must be a SHA-256 digest")
    if decoding_hash != adapter.get("decoding_contract_sha256"):
        raise ValueError(f"{example_id}: base and adapter decoding contracts differ")


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
            f"predictions do not cover the complete held-out split: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    packet: list[dict[str, Any]] = []
    for example_id in sorted(contracts):
        source = contracts[example_id]
        _validate_pair(example_id, source, base[example_id], adapter[example_id])
        completions = (base[example_id]["completion"], adapter[example_id]["completion"])
        if _pair_is_flipped(blinding_key, example_id):
            completions = (completions[1], completions[0])
        item = {
            "schema_version": SCHEMA_VERSION,
            "id": example_id,
            "system": source["system"],
            "prompt": source["prompt"],
            "reference_response": source["response"],
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "request_messages_sha256": canonical_sha256(_request_messages(source)),
            "decoding_contract_sha256": base[example_id].get("decoding_contract_sha256"),
            "generation_pair_contract_sha256": base[example_id]["generation_pair_contract_sha256"],
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
    packet_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLINDED-REVIEW-PENDING",
        "method": METHOD,
        "count": len(packet),
        "blinding_key_sha256": hashlib.sha256(key).hexdigest(),
        "contracts_sha256": {str(path): file_sha256(path) for path in contracts_paths},
        "base_predictions_sha256": file_sha256(base_path),
        "adapter_predictions_sha256": file_sha256(adapter_path),
        "packet_sha256": file_sha256(packet_path),
        "rubric": {
            "fields": list(RATING_FIELDS),
            "range": [1, 5],
            "score": "mean((rating - 1) / 4); capped at 0.25 for severe_error",
        },
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
) -> tuple[list[tuple[str, dict[str, dict[str, Any]]]], dict[str, str]]:
    if minimum_reviewers < 1:
        raise ValueError("minimum_reviewers must be positive")
    expected = index_rows(expected_packet, "expected packet")
    completed: list[tuple[str, dict[str, dict[str, Any]]]] = []
    review_hashes: dict[str, str] = {}
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
        review_hashes[str(path)] = file_sha256(path)
    if len(completed) < minimum_reviewers:
        raise ValueError(
            f"at least {minimum_reviewers} distinct completed reviews are required; got {len(completed)}"
        )
    return completed, review_hashes


def adjudicate(
    contracts: dict[str, dict[str, Any]],
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    completed_reviews: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    blinding_key: bytes,
    review_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    packet = build_packet(contracts, base_rows, adapter_rows, blinding_key=blinding_key)
    base = index_rows(deepcopy(base_rows), "base predictions")
    adapter = index_rows(deepcopy(adapter_rows), "adapter predictions")
    reviewers = sorted(reviewer for reviewer, _ in completed_reviews)

    for item in packet:
        example_id = item["id"]
        flipped = _pair_is_flipped(blinding_key, example_id)
        base_label, adapter_label = (("candidate_b", "candidate_a") if flipped else ("candidate_a", "candidate_b"))
        model_scores: dict[str, list[float]] = {"base": [], "adapter": []}
        component_scores: dict[str, dict[str, list[int]]] = {
            "base": {field: [] for field in RATING_FIELDS},
            "adapter": {field: [] for field in RATING_FIELDS},
        }
        for _, review_rows in completed_reviews:
            review = review_rows[example_id]["review"]
            for model_name, candidate_label in (("base", base_label), ("adapter", adapter_label)):
                score, ratings = _validate_candidate_review(
                    example_id,
                    candidate_label,
                    review[candidate_label],
                )
                model_scores[model_name].append(score)
                for field, value in ratings.items():
                    component_scores[model_name][field].append(value)

        for model_name, target in (("base", base[example_id]), ("adapter", adapter[example_id])):
            target["semantic_score"] = fmean(model_scores[model_name])
            target["semantic_score_provenance"] = {
                "method": METHOD,
                "reviewers": reviewers,
                "review_file_sha256": review_hashes,
                "mean_ratings": {
                    field: fmean(values) for field, values in component_scores[model_name].items()
                },
            }

    base_scored = [base[example_id] for example_id in sorted(base)]
    adapter_scored = [adapter[example_id] for example_id in sorted(adapter)]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "BLINDED-REVIEW-COMPLETE",
        "method": METHOD,
        "count": len(base_scored),
        "reviewers": reviewers,
        "review_file_sha256": review_hashes,
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "base_semantic_score_mean": fmean(row["semantic_score"] for row in base_scored),
        "adapter_semantic_score_mean": fmean(row["semantic_score"] for row in adapter_scored),
    }
    return base_scored, adapter_scored, summary


def _prepare(args: argparse.Namespace) -> None:
    require_absent((args.packet, args.manifest), overwrite=args.overwrite)
    key = read_blinding_key(args.blinding_key_file)
    contracts = read_contracts(args.contracts, split=args.split)
    packet = build_packet(
        contracts,
        read_jsonl(args.base),
        read_jsonl(args.adapter),
        blinding_key=key,
    )
    write_jsonl(args.packet, packet)
    manifest = packet_manifest(
        packet,
        key=key,
        contracts_paths=args.contracts,
        base_path=args.base,
        adapter_path=args.adapter,
        packet_path=args.packet,
    )
    write_json(args.manifest, manifest)


def _adjudicate(args: argparse.Namespace) -> None:
    require_absent(
        (args.base_output, args.adapter_output, args.manifest),
        overwrite=args.overwrite,
    )
    key = read_blinding_key(args.blinding_key_file)
    contracts = read_contracts(args.contracts, split=args.split)
    base_rows = read_jsonl(args.base)
    adapter_rows = read_jsonl(args.adapter)
    packet = build_packet(contracts, base_rows, adapter_rows, blinding_key=key)
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
    )
    write_jsonl(args.base_output, base_scored)
    write_jsonl(args.adapter_output, adapter_scored)
    manifest.update(
        {
            "contracts_sha256": {str(path): file_sha256(path) for path in args.contracts},
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
