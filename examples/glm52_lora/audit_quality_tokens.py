#!/usr/bin/env python3
"""Measure GLM-5.2 quality data with the exact model chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from build_quality_dataset import read_jsonl, validate_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot summarize an empty token-length sequence")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty token-length sequence")
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
        "total": sum(values),
    }


def token_count(encoded: Any) -> int:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer output has no input_ids")
        encoded = encoded["input_ids"]
    shape = getattr(encoded, "shape", None)
    if shape is not None:
        if len(shape) not in {1, 2}:
            raise ValueError(f"unsupported input_ids shape: {tuple(shape)}")
        return int(shape[-1])
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("expected one tokenized sequence")
        encoded = encoded[0]
    return len(encoded)


def _family(row: dict[str, Any]) -> str:
    matches = [tag for tag in row["tags"] if tag.startswith(("markdown-", "russian-", "han-"))]
    if len(matches) != 1:
        raise ValueError(f"{row['id']}: expected one quality-family tag, got {matches}")
    return matches[0]


def measure(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    metrics: dict[str, list[int]] = defaultdict(list)
    by_split: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_family: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        prompt_messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
        ]
        full_messages = [*prompt_messages, {"role": "assistant", "content": row["response"]}]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_tokens = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        target_tokens = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        lengths = {
            "full_chat": token_count(full_tokens),
            "prompt_chat": token_count(prompt_tokens),
            "target_text": token_count(target_tokens),
        }
        for name, length in lengths.items():
            metrics[name].append(length)
            by_split[row["split"]][name].append(length)
            by_family[_family(row)][name].append(length)

    return {
        "overall": {name: summarize(values) for name, values in sorted(metrics.items())},
        "by_split": {
            split: {name: summarize(values) for name, values in sorted(group.items())}
            for split, group in sorted(by_split.items())
        },
        "by_family": {
            family: {name: summarize(values) for name, values in sorted(group.items())}
            for family, group in sorted(by_family.items())
        },
    }


def audit(input_jsonl: Path, tokenizer_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started_at = time.monotonic()
    rows = validate_rows(read_jsonl(input_jsonl))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    result = {
        "schema_version": 1,
        "input_jsonl": str(input_jsonl.resolve()),
        "input_sha256": _sha256(input_jsonl),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_json_sha256": _sha256(tokenizer_path / "tokenizer.json"),
        "tokenizer_config_sha256": _sha256(tokenizer_path / "tokenizer_config.json"),
        **measure(rows, tokenizer),
        "wall_seconds": time.monotonic() - started_at,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("tokenizer_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.input_jsonl, args.tokenizer_path)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
