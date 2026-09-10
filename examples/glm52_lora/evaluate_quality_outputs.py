#!/usr/bin/env python3
"""Aggregate held-out GLM-5.2 Russian, Markdown, and Han diagnostics.

Han metrics are cohort-specific. In particular, prompts that explicitly ask
the model to remove Han text cannot improve the spontaneous-Han metric merely
by being numerous. Token-normalized metrics are micro rates: their numerator
and denominator are summed over a whole cohort, and they are unavailable when
even one row lacks a native positive integer completion-token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from quality_reward import contract_from_mapping, score_constraints

HAN_EVALUATION_MODES = frozenset(
    {
        "spontaneous",
        "input_conditioned_cleanup",
        "input_conditioned_scope_control",
        "excluded_han_allowed",
    }
)
SEMANTIC_PROVENANCE_METHOD = "blinded-human-rubric-v1"
SEMANTIC_RATING_FIELDS = frozenset(
    {
        "meaning_preservation",
        "russian_naturalness",
        "factual_accuracy",
        "instruction_fulfillment",
    }
)
TOKEN_MICRO_RATE_DEFINITION = (
    "1000 * sum(output Han characters) / sum(completion tokens) within the "
    "cohort; null unless every cohort row has a native positive integer "
    "completion_token_count"
)


def _native_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a native nonnegative integer")
    if value < 0:
        raise ValueError(f"{label} must be a native nonnegative integer")
    return value


def _optional_native_positive_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a native positive integer")
    if value <= 0:
        raise ValueError(f"{label} must be a native positive integer")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_semantic_score(row: dict[str, Any], example_id: str) -> tuple[float | None, dict[str, Any] | None]:
    """Validate a score and the blinded-review evidence that produced it."""
    raw_score = row.get("semantic_score")
    provenance = row.get("semantic_score_provenance")
    if raw_score is None:
        if provenance is not None:
            raise ValueError(f"{example_id}: semantic_score_provenance exists without semantic_score")
        return None, None
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise TypeError(f"{example_id}: semantic_score must be a native number")
    score = float(raw_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{example_id}: semantic_score must be finite in [0, 1]")
    if not isinstance(provenance, dict) or set(provenance) != {
        "method",
        "reviewers",
        "review_artifacts",
        "review_scores",
        "mean_ratings",
        "completion_sha256",
        "review_item_sha256",
        "pair_contract_sha256",
        "adjudication_contract_sha256",
    }:
        raise ValueError(f"{example_id}: semantic_score_provenance fields are invalid")
    if provenance.get("method") != SEMANTIC_PROVENANCE_METHOD:
        raise ValueError(f"{example_id}: semantic score method is invalid")
    reviewers = provenance.get("reviewers")
    if (
        not isinstance(reviewers, list)
        or len(reviewers) < 2
        or any(not isinstance(item, str) or not item.strip() for item in reviewers)
        or len(set(reviewers)) != len(reviewers)
        or reviewers != sorted(reviewers)
    ):
        raise ValueError(f"{example_id}: semantic score reviewers are invalid")
    review_hashes = provenance.get("review_artifacts")
    if (
        not isinstance(review_hashes, list)
        or len(review_hashes) != len(reviewers)
        or any(
            not isinstance(item, dict)
            or set(item) != {"ordinal", "filename", "sha256"}
            or item.get("ordinal") != ordinal
            or not isinstance(item.get("filename"), str)
            or not item["filename"]
            or not _is_sha256(item.get("sha256"))
            for ordinal, item in enumerate(review_hashes)
        )
    ):
        raise ValueError(f"{example_id}: semantic review file hashes are invalid")
    review_scores = provenance.get("review_scores")
    if (
        not isinstance(review_scores, list)
        or len(review_scores) != len(reviewers)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in review_scores
        )
    ):
        raise ValueError(f"{example_id}: semantic reviewer scores are invalid")
    if not math.isclose(score, fmean(review_scores), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{example_id}: semantic score differs from reviewer scores")
    mean_ratings = provenance.get("mean_ratings")
    if not isinstance(mean_ratings, dict) or set(mean_ratings) != SEMANTIC_RATING_FIELDS:
        raise ValueError(f"{example_id}: semantic mean ratings are invalid")
    for value in mean_ratings.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{example_id}: semantic mean ratings must be numeric")
        if not math.isfinite(float(value)) or not 1.0 <= float(value) <= 5.0:
            raise ValueError(f"{example_id}: semantic mean ratings must be in [1, 5]")
    completion = row.get("completion")
    if not isinstance(completion, str) or hashlib.sha256(completion.encode("utf-8")).hexdigest() != provenance.get(
        "completion_sha256"
    ):
        raise ValueError(f"{example_id}: semantic score is bound to another completion")
    for field in (
        "review_item_sha256",
        "pair_contract_sha256",
        "adjudication_contract_sha256",
    ):
        if not _is_sha256(provenance.get(field)):
            raise ValueError(f"{example_id}: semantic {field} is invalid")
    if provenance["pair_contract_sha256"] != row.get("pair_contract_sha256"):
        raise ValueError(f"{example_id}: semantic score uses another pair contract")
    return score, provenance


def semantic_provenance_pair_context(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return review evidence that must be identical for both paired outputs."""
    return {
        "method": provenance["method"],
        "reviewers": provenance["reviewers"],
        "review_artifacts": provenance["review_artifacts"],
        "adjudication_contract_sha256": provenance["adjudication_contract_sha256"],
    }


def _validate_han_assignment(
    row: dict[str, Any], example_id: str, *, han_is_allowed: bool
) -> tuple[str, int, bool, str]:
    raw_mode = row.get("han_evaluation_mode")
    if not isinstance(raw_mode, str) or raw_mode not in HAN_EVALUATION_MODES:
        raise ValueError(f"{example_id}: han_evaluation_mode is invalid")
    input_han_count = _native_nonnegative_int(row.get("input_han_count"), label=f"{example_id}: input_han_count")
    input_contains_han = row.get("input_contains_han")
    if not isinstance(input_contains_han, bool):
        raise TypeError(f"{example_id}: input_contains_han must be a boolean")
    if input_contains_han != (input_han_count > 0):
        raise ValueError(f"{example_id}: input_contains_han disagrees with input_han_count")
    cluster_id = row.get("evaluation_cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id.strip():
        raise ValueError(f"{example_id}: evaluation_cluster_id is invalid")

    if raw_mode == "excluded_han_allowed":
        if not han_is_allowed:
            raise ValueError(f"{example_id}: excluded_han_allowed conflicts with the output contract")
    else:
        if han_is_allowed:
            raise ValueError(f"{example_id}: {raw_mode} conflicts with a Han-allowed output contract")
        if raw_mode == "spontaneous" and input_han_count != 0:
            raise ValueError(f"{example_id}: spontaneous mode requires Han-free input")
        if (
            raw_mode
            in {
                "input_conditioned_cleanup",
                "input_conditioned_scope_control",
            }
            and input_han_count == 0
        ):
            raise ValueError(f"{example_id}: {raw_mode} requires Han-bearing input")
    return raw_mode, input_han_count, input_contains_han, cluster_id


def _empty_han_cohort() -> dict[str, int]:
    return {
        "row_count": 0,
        "han_output_row_count": 0,
        "han_character_count": 0,
        "visible_character_count": 0,
        "completion_token_count": 0,
        "completion_token_count_covered_rows": 0,
    }


def _record_han_cohort(
    cohort: dict[str, int],
    *,
    han_count: int,
    visible_character_count: int,
    token_count: int | None,
) -> None:
    cohort["row_count"] += 1
    cohort["han_output_row_count"] += int(han_count > 0)
    cohort["han_character_count"] += han_count
    cohort["visible_character_count"] += visible_character_count
    if token_count is not None:
        cohort["completion_token_count"] += token_count
        cohort["completion_token_count_covered_rows"] += 1


def _han_cohort_summary(cohort: dict[str, int]) -> dict[str, int | float | None]:
    rows = cohort["row_count"]
    han_rows = cohort["han_output_row_count"]
    visible_characters = cohort["visible_character_count"]
    tokens = cohort["completion_token_count"]
    token_covered_rows = cohort["completion_token_count_covered_rows"]
    return {
        "row_count": rows,
        "han_output_row_count": han_rows,
        "han_output_row_rate": han_rows / rows if rows else None,
        "han_character_count": cohort["han_character_count"],
        "han_character_micro_rate_per_1000_visible_characters": (
            1000.0 * cohort["han_character_count"] / visible_characters if visible_characters else None
        ),
        "han_character_micro_rate_per_1000_completion_tokens": (
            1000.0 * cohort["han_character_count"] / tokens if rows and token_covered_rows == rows else None
        ),
        "completion_token_count": tokens if token_covered_rows == rows else None,
        "completion_token_count_covered_rows": token_covered_rows,
        "completion_token_count_coverage": token_covered_rows / rows if rows else None,
    }


def _cleanup_cohort_summary(cohort: dict[str, int]) -> dict[str, int | float | None]:
    generic = _han_cohort_summary(cohort)
    rows = cohort["row_count"]
    residual_rows = cohort["han_output_row_count"]
    return {
        **generic,
        "successful_cleanup_row_count": rows - residual_rows,
        # This metric exists only on explicitly labelled cleanup rows.
        "cleanup_success_rate": (rows - residual_rows) / rows if rows else None,
        "residual_han_output_row_count": residual_rows,
        "residual_han_output_row_rate": generic["han_output_row_rate"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    return rows


def evaluate_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    ids: set[str] = set()
    cohorts = {
        mode: _empty_han_cohort()
        for mode in (
            "spontaneous",
            "input_conditioned_cleanup",
            "input_conditioned_scope_control",
        )
    }
    excluded_han_allowed_count = 0
    russian_script_scores: list[float] = []
    semantic_scores: list[float] = []
    required_markdown_results: list[float] = []

    for index, row in enumerate(rows):
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"row {index}: missing id")
        if example_id in ids:
            raise ValueError(f"row {index}: duplicate id: {example_id!r}")
        ids.add(example_id)
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise TypeError(f"{example_id}: completion must be a string")
        contract_data = row.get("contract")
        if not isinstance(contract_data, dict):
            raise TypeError(f"{example_id}: contract must be an object")
        contract = contract_from_mapping(contract_data)
        result = score_constraints(completion, contract)
        language_root = contract.requested_language.split("-", 1)[0].split("_", 1)[0]
        han_is_allowed = contract.allow_han or language_root in {
            "zh",
            "ja",
        }
        mode, input_han_count, input_contains_han, cluster_id = _validate_han_assignment(
            row,
            example_id,
            han_is_allowed=han_is_allowed,
        )
        token_count = _optional_native_positive_int(
            row.get("completion_token_count"),
            label=f"{example_id}: completion_token_count",
        )
        if mode == "excluded_han_allowed":
            excluded_han_allowed_count += 1
        else:
            _record_han_cohort(
                cohorts[mode],
                han_count=result.han_count,
                visible_character_count=result.visible_character_count,
                token_count=token_count,
            )
        if language_root == "ru":
            russian_script_scores.append(result.russian_script_score)

        semantic_score, _ = validate_semantic_score(row, example_id)
        if semantic_score is not None:
            semantic_scores.append(semantic_score)

        markdown_valid = not result.markdown_defects
        if contract.require_markdown or contract.required_blocks:
            required_markdown_results.append(float(markdown_valid))
        details.append(
            {
                "id": example_id,
                "evaluation_cluster_id": cluster_id,
                "constraint": asdict(result),
                "input_han_count": input_han_count,
                "input_contains_han": input_contains_han,
                "han_evaluation_mode": mode,
                "output_han_count": result.han_count,
                "markdown_valid": markdown_valid,
                "completion_token_count": token_count,
                "semantic_score": semantic_score,
            }
        )

    count = len(details)
    summary = {
        "count": count,
        "constraint_score_mean": fmean(detail["constraint"]["score"] for detail in details),
        "russian_example_count": len(russian_script_scores),
        "russian_script_score_mean": (fmean(russian_script_scores) if russian_script_scores else None),
        "markdown_structural_valid_rate": fmean(float(detail["markdown_valid"]) for detail in details),
        "required_markdown_valid_rate": (fmean(required_markdown_results) if required_markdown_results else None),
        "spontaneous_han": _han_cohort_summary(cohorts["spontaneous"]),
        "input_conditioned_han_cleanup": _cleanup_cohort_summary(cohorts["input_conditioned_cleanup"]),
        "input_conditioned_han_scope_control": _han_cohort_summary(cohorts["input_conditioned_scope_control"]),
        "excluded_han_allowed_count": excluded_han_allowed_count,
        "semantic_score_mean": fmean(semantic_scores) if semantic_scores else None,
        "semantic_score_coverage": len(semantic_scores) / count,
        "metric_definitions": {
            "han_character_micro_rate_per_1000_completion_tokens": (TOKEN_MICRO_RATE_DEFINITION),
            "han_output_row_rate": ("rows with one or more output Han characters / cohort rows"),
        },
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("--details", type=Path)
    args = parser.parse_args()
    summary, details = evaluate_rows(read_jsonl(args.predictions_jsonl))
    if args.details is not None:
        args.details.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
