#!/usr/bin/env python3
"""Compare paired full-model base and adapter quality outputs fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluate_quality_outputs import evaluate_rows, read_jsonl
from quality_reward import contract_from_mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(differences: list[float], *, samples: int, seed: int) -> dict[str, float | int] | None:
    if not differences:
        return None
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    rng = random.Random(seed)
    count = len(differences)
    means = [fmean(differences[rng.randrange(count)] for _ in range(count)) for _ in range(samples)]
    return {
        "count": count,
        "mean": fmean(differences),
        "ci95_low": _quantile(means, 0.025),
        "ci95_high": _quantile(means, 0.975),
    }


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        example_id = str(row.get("id", "")).strip()
        if not example_id:
            raise ValueError(f"{label} row {row_number}: missing id")
        if example_id in indexed:
            raise ValueError(f"{label}: duplicate id {example_id!r}")
        indexed[example_id] = row
    return indexed


def _require_equal_pair_contract(example_id: str, base: dict[str, Any], adapter: dict[str, Any]) -> None:
    if base.get("contract") != adapter.get("contract"):
        raise ValueError(f"{example_id}: base and adapter contracts differ")
    for field in ("prompt_sha256", "decoding_contract_sha256"):
        base_value = base.get(field)
        adapter_value = adapter.get(field)
        if base_value is None and adapter_value is None:
            continue
        if not base_value or base_value != adapter_value:
            raise ValueError(f"{example_id}: base and adapter {field} differ")
    if (base.get("semantic_score") is None) != (adapter.get("semantic_score") is None):
        raise ValueError(f"{example_id}: semantic score coverage is not paired")


def _metric_status(
    metric: dict[str, float | int] | None,
    *,
    regression_margin: float = 0.0,
) -> str:
    if metric is None:
        return "PENDING"
    mean = float(metric["mean"])
    ci95_low = float(metric["ci95_low"])
    if mean < -regression_margin:
        return "FAIL"
    if mean > 0.0 and ci95_low >= -regression_margin:
        return "PASS"
    return "PENDING"


def compare_rows(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 52,
    required_semantic_coverage: float = 1.0,
    semantic_noninferiority_margin: float = 0.02,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0.0 <= required_semantic_coverage <= 1.0:
        raise ValueError("required semantic coverage must be in [0, 1]")
    if semantic_noninferiority_margin < 0.0:
        raise ValueError("semantic noninferiority margin must be nonnegative")

    base_index = _index_rows(base_rows, "base")
    adapter_index = _index_rows(adapter_rows, "adapter")
    base_ids = set(base_index)
    adapter_ids = set(adapter_index)
    if base_ids != adapter_ids:
        missing = sorted(base_ids - adapter_ids)
        unexpected = sorted(adapter_ids - base_ids)
        raise ValueError(f"base and adapter ids differ: missing={missing[:5]}, unexpected={unexpected[:5]}")

    base_summary, base_details = evaluate_rows(base_rows)
    adapter_summary, adapter_details = evaluate_rows(adapter_rows)
    base_detail_index = {row["id"]: row for row in base_details}
    adapter_detail_index = {row["id"]: row for row in adapter_details}

    constraint_differences: list[float] = []
    russian_script_differences: list[float] = []
    semantic_differences: list[float] = []
    markdown_improvements: list[float] = []
    han_example_improvements: list[float] = []
    han_count_improvements: list[float] = []
    han_per_1000_token_improvements: list[float] = []
    paired_details: list[dict[str, Any]] = []
    russian_count = 0
    base_markdown_defects = 0
    adapter_markdown_defects = 0
    base_han_defects = 0
    adapter_han_defects = 0

    for example_id in sorted(base_ids):
        base = base_index[example_id]
        adapter = adapter_index[example_id]
        _require_equal_pair_contract(example_id, base, adapter)
        contract = contract_from_mapping(base.get("contract") or {})
        base_detail = base_detail_index[example_id]
        adapter_detail = adapter_detail_index[example_id]

        constraint_delta = float(adapter_detail["constraint"]["score"]) - float(base_detail["constraint"]["score"])
        constraint_differences.append(constraint_delta)
        detail: dict[str, Any] = {
            "id": example_id,
            "constraint_score_adapter_minus_base": constraint_delta,
        }

        if contract.requested_language == "ru":
            russian_count += 1
            script_delta = float(adapter_detail["constraint"]["russian_script_score"]) - float(
                base_detail["constraint"]["russian_script_score"]
            )
            russian_script_differences.append(script_delta)
            detail["russian_script_score_adapter_minus_base"] = script_delta
            if base.get("semantic_score") is not None:
                semantic_delta = float(adapter["semantic_score"]) - float(base["semantic_score"])
                semantic_differences.append(semantic_delta)
                detail["semantic_score_adapter_minus_base"] = semantic_delta

        if contract.require_markdown or contract.required_blocks:
            base_valid = bool(base_detail["markdown_valid"])
            adapter_valid = bool(adapter_detail["markdown_valid"])
            base_markdown_defects += int(not base_valid)
            adapter_markdown_defects += int(not adapter_valid)
            markdown_delta = float(adapter_valid) - float(base_valid)
            markdown_improvements.append(markdown_delta)
            detail["markdown_valid_adapter_minus_base"] = markdown_delta

        allow_han = contract.allow_han or contract.requested_language in {"zh", "ja"}
        if not allow_han:
            base_han = int(base_detail["accidental_han_count"])
            adapter_han = int(adapter_detail["accidental_han_count"])
            base_han_defects += int(base_han > 0)
            adapter_han_defects += int(adapter_han > 0)
            example_improvement = float(base_han > 0) - float(adapter_han > 0)
            count_improvement = float(base_han - adapter_han)
            han_example_improvements.append(example_improvement)
            han_count_improvements.append(count_improvement)
            detail["accidental_han_example_base_minus_adapter"] = example_improvement
            detail["accidental_han_count_base_minus_adapter"] = count_improvement
            base_tokens = base.get("completion_token_count")
            adapter_tokens = adapter.get("completion_token_count")
            if base_tokens is not None and adapter_tokens is not None:
                token_improvement = 1000.0 * (base_han / int(base_tokens) - adapter_han / int(adapter_tokens))
                han_per_1000_token_improvements.append(token_improvement)
                detail["accidental_han_per_1000_tokens_base_minus_adapter"] = token_improvement
        paired_details.append(detail)

    metric_seed = bootstrap_seed

    def bootstrap(values: list[float]) -> dict[str, float | int] | None:
        nonlocal metric_seed
        result = paired_bootstrap(values, samples=bootstrap_samples, seed=metric_seed)
        metric_seed += 1
        return result

    metrics = {
        "constraint_score_adapter_minus_base": bootstrap(constraint_differences),
        "russian_script_score_adapter_minus_base": bootstrap(russian_script_differences),
        "russian_semantic_score_adapter_minus_base": bootstrap(semantic_differences),
        "required_markdown_valid_base_to_adapter": bootstrap(markdown_improvements),
        "accidental_han_example_base_to_adapter": bootstrap(han_example_improvements),
        "accidental_han_count_base_to_adapter": bootstrap(han_count_improvements),
        "accidental_han_per_1000_tokens_base_to_adapter": bootstrap(han_per_1000_token_improvements),
    }

    semantic_coverage = len(semantic_differences) / russian_count if russian_count else 0.0
    semantic_status = _metric_status(
        metrics["russian_semantic_score_adapter_minus_base"],
        regression_margin=semantic_noninferiority_margin,
    )
    if semantic_coverage < required_semantic_coverage:
        semantic_status = "PENDING"

    if base_markdown_defects == 0:
        markdown_status = "NOT_REPRODUCED"
    else:
        markdown_status = _metric_status(metrics["required_markdown_valid_base_to_adapter"])

    if base_han_defects == 0:
        han_status = "NOT_REPRODUCED"
    else:
        han_status = _metric_status(metrics["accidental_han_example_base_to_adapter"])

    target_status = {
        "russian_semantic_quality": semantic_status,
        "required_markdown_validity": markdown_status,
        "accidental_han": han_status,
    }
    if "FAIL" in target_status.values():
        overall_status = "FAIL"
    elif all(status == "PASS" for status in target_status.values()):
        overall_status = "PASS"
    else:
        overall_status = "PENDING"

    result = {
        "status": overall_status,
        "target_status": target_status,
        "count": len(base_ids),
        "russian_example_count": russian_count,
        "paired_russian_semantic_count": len(semantic_differences),
        "paired_russian_semantic_coverage": semantic_coverage,
        "required_semantic_coverage": required_semantic_coverage,
        "semantic_noninferiority_margin": semantic_noninferiority_margin,
        "base_required_markdown_defects": base_markdown_defects,
        "adapter_required_markdown_defects": adapter_markdown_defects,
        "base_accidental_han_examples": base_han_defects,
        "adapter_accidental_han_examples": adapter_han_defects,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "base_summary": base_summary,
        "adapter_summary": adapter_summary,
    }
    return result, paired_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_predictions", type=Path)
    parser.add_argument("adapter_predictions", type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=52)
    parser.add_argument("--required-semantic-coverage", type=float, default=1.0)
    parser.add_argument("--semantic-noninferiority-margin", type=float, default=0.02)
    args = parser.parse_args()

    result, details = compare_rows(
        read_jsonl(args.base_predictions),
        read_jsonl(args.adapter_predictions),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        required_semantic_coverage=args.required_semantic_coverage,
        semantic_noninferiority_margin=args.semantic_noninferiority_margin,
    )
    result["base_predictions_sha256"] = _sha256(args.base_predictions)
    result["adapter_predictions_sha256"] = _sha256(args.adapter_predictions)
    if args.details is not None:
        args.details.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details))
        result["details_sha256"] = _sha256(args.details)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
