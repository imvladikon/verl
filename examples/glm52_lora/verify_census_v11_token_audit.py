#!/usr/bin/env python3
"""Validate the exact GLM-5.2 token audit for the censused v11 train view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_INPUT_SHA256 = "fa6c257c78d2b2c10f8d2a5d0cf9456a88f6e1e1d4e90189a0d7ec4237938658"
EXPECTED_TOKENIZER_JSON_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"
EXPECTED_FAMILIES = {
    "han-cleanup",
    "han-in-code",
    "han-in-link",
    "han-in-quote",
    "han-retention",
    "markdown-code",
    "markdown-list",
    "markdown-mixed",
    "markdown-table",
    "russian-case-period-restoration",
    "russian-latin-confusable-cleanup",
    "russian-style",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"TOKEN-AUDIT-FAIL: {message}")


def validate(report: dict) -> dict:
    require(report["input_sha256"] == EXPECTED_INPUT_SHA256, "train JSONL drift")
    require(
        report["tokenizer_json_sha256"] == EXPECTED_TOKENIZER_JSON_SHA256,
        "tokenizer.json drift",
    )
    require(
        report["tokenizer_config_sha256"] == EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer_config.json drift",
    )
    require(set(report["by_split"]) == {"train"}, "heldout split entered token audit")
    require(set(report["by_family"]) == EXPECTED_FAMILIES, "quality-family drift")
    expected = {
        "full_chat": {
            "count": 576,
            "min": 54,
            "p50": 141,
            "p99": 504,
            "max": 548,
            "total": 100622,
        },
        "prompt_chat": {
            "count": 576,
            "min": 53,
            "p50": 99,
            "p99": 291,
            "max": 313,
            "total": 67056,
        },
        "target_text": {
            "count": 576,
            "min": 1,
            "p50": 43,
            "p99": 213,
            "max": 235,
            "total": 33566,
        },
    }
    for name, values in expected.items():
        actual = report["overall"][name]
        for metric, value in values.items():
            require(actual[metric] == value, f"{name}.{metric} drift")
        require(
            report["by_split"]["train"][name] == actual,
            f"{name} split aggregate drift",
        )
    return {
        "status": "TOKEN-AUDIT-PASS",
        "train_rows": 576,
        "full_chat_max_tokens": 548,
        "configured_max_length": 576,
        "heldout_rows_used": 0,
        "tokenizer_json_sha256": EXPECTED_TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.report.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
