#!/usr/bin/env python3
"""Aggregate held-out GLM-5.2 Russian/Markdown/Han output diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from quality_reward import contract_from_mapping, score_constraints


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


def evaluate_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    ids: set[str] = set()
    applicable_han_examples = 0
    applicable_visible_characters = 0
    applicable_tokens = 0
    token_covered_examples = 0
    accidental_han_count = 0
    russian_script_scores: list[float] = []
    semantic_scores: list[float] = []
    required_markdown_results: list[float] = []

    for index, row in enumerate(rows):
        example_id = str(row.get("id", "")).strip()
        completion = str(row.get("completion", ""))
        if not example_id or example_id in ids:
            raise ValueError(f"row {index}: missing or duplicate id: {example_id!r}")
        ids.add(example_id)
        contract_data = row.get("contract") or {}
        if not isinstance(contract_data, dict):
            raise TypeError(f"{example_id}: contract must be an object")
        contract = contract_from_mapping(contract_data)
        result = score_constraints(completion, contract)
        han_is_allowed = contract.allow_han or contract.requested_language in {"zh", "ja"}
        accidental = 0 if han_is_allowed else result.han_count
        accidental_han_count += accidental
        if not han_is_allowed:
            applicable_han_examples += 1
            applicable_visible_characters += result.visible_character_count
        if contract.requested_language == "ru":
            russian_script_scores.append(result.russian_script_score)

        token_count = row.get("completion_token_count")
        if token_count is not None:
            token_count = int(token_count)
            if token_count <= 0:
                raise ValueError(f"{example_id}: completion_token_count must be positive")
            if not han_is_allowed:
                applicable_tokens += token_count
                token_covered_examples += 1

        semantic_score = row.get("semantic_score")
        if semantic_score is not None:
            semantic_score = float(semantic_score)
            if not math.isfinite(semantic_score) or not 0.0 <= semantic_score <= 1.0:
                raise ValueError(f"{example_id}: semantic_score must be finite in [0, 1]")
            semantic_scores.append(semantic_score)

        markdown_valid = float(not result.markdown_defects)
        if contract.require_markdown or contract.required_blocks:
            required_markdown_results.append(markdown_valid)
        details.append(
            {
                "id": example_id,
                "constraint": asdict(result),
                "accidental_han_count": accidental,
                "markdown_valid": bool(markdown_valid),
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
        "accidental_han_applicable_count": applicable_han_examples,
        "accidental_han_example_rate": (
            sum(detail["accidental_han_count"] > 0 for detail in details) / applicable_han_examples
            if applicable_han_examples
            else None
        ),
        "accidental_han_count": accidental_han_count,
        "accidental_han_per_1000_visible_characters": (
            1000.0 * accidental_han_count / applicable_visible_characters if applicable_visible_characters else None
        ),
        "accidental_han_per_1000_tokens": (
            1000.0 * accidental_han_count / applicable_tokens if applicable_tokens else None
        ),
        "token_count_coverage": (token_covered_examples / applicable_han_examples if applicable_han_examples else None),
        "semantic_score_mean": fmean(semantic_scores) if semantic_scores else None,
        "semantic_score_coverage": len(semantic_scores) / count,
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
