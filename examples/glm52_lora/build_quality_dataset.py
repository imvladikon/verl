#!/usr/bin/env python3
"""Validate curated GLM-5.2 quality JSONL and write SFT/eval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from quality_reward import contract_from_mapping, score_constraints

VALID_SPLITS = frozenset({"train", "validation", "test"})
WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_SYSTEM = (
    "Отвечай естественно и точно. Соблюдай язык запроса. Возвращай структурно корректный Markdown, когда он требуется."
)


def _normalized_prompt(prompt: str) -> str:
    return WHITESPACE_RE.sub(" ", prompt).strip().casefold()


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(_normalized_prompt(prompt).encode("utf-8")).hexdigest()


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
        raise ValueError(f"empty dataset: {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    prompt_splits: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(rows):
        for field in ("id", "split", "prompt", "response"):
            if not isinstance(raw.get(field), str):
                raise TypeError(f"row {index}: {field} must be a string")
        example_id = raw["id"].strip()
        split = raw["split"].strip()
        prompt = raw["prompt"].strip()
        response = raw["response"].strip()
        if not example_id or example_id in ids:
            raise ValueError(f"row {index}: missing or duplicate id: {example_id!r}")
        if split not in VALID_SPLITS:
            raise ValueError(f"{example_id}: unsupported split: {split!r}")
        if not prompt or not response:
            raise ValueError(f"{example_id}: prompt and response must be nonempty")
        ids.add(example_id)

        digest = prompt_digest(prompt)
        previous = prompt_splits.get(digest)
        if previous is not None:
            raise ValueError(f"prompt leakage/duplicate: {example_id} ({split}) matches {previous[0]} ({previous[1]})")
        prompt_splits[digest] = (example_id, split)

        contract_data = raw.get("contract") or {}
        if not isinstance(contract_data, dict):
            raise TypeError(f"{example_id}: contract must be an object")
        contract = contract_from_mapping(contract_data)
        result = score_constraints(response, contract)
        if result.han_score != 1.0:
            raise ValueError(f"{example_id}: curated response contains accidental Han")
        if result.markdown_defects:
            raise ValueError(f"{example_id}: curated response has Markdown defects: {result.markdown_defects}")
        if contract.requested_language == "ru" and result.cyrillic_count == 0:
            raise ValueError(f"{example_id}: Russian response has no Cyrillic letters")

        system = raw.get("system", DEFAULT_SYSTEM)
        if not isinstance(system, str) or not system.strip():
            raise TypeError(f"{example_id}: system must be a nonempty string")
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise TypeError(f"{example_id}: tags must be a list of strings")
        use_for_rl = raw.get("use_for_constraint_rl_smoke", False)
        if not isinstance(use_for_rl, bool):
            raise TypeError(f"{example_id}: use_for_constraint_rl_smoke must be a boolean")
        review = raw.get("review")
        if not isinstance(review, dict):
            raise TypeError(f"{example_id}: review must be an object")
        if review.get("status") != "accepted":
            raise ValueError(f"{example_id}: review status must be accepted")
        reviewer = review.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise TypeError(f"{example_id}: accepted review must name a reviewer")
        review_notes = review.get("notes", "")
        if not isinstance(review_notes, str):
            raise TypeError(f"{example_id}: review.notes must be a string")
        provenance = raw.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError(f"{example_id}: provenance must be an object")
        required_provenance = (
            "dataset",
            "revision",
            "license",
            "source_split",
            "source_record_id",
        )
        for field in required_provenance:
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{example_id}: provenance.{field} must be a nonempty string")

        validated.append(
            {
                "id": example_id,
                "split": split,
                "prompt": prompt,
                "response": response,
                "system": system.strip(),
                "contract": {
                    "requested_language": contract.requested_language,
                    "allow_han": contract.allow_han,
                    "allow_han_in_blockquotes": contract.allow_han_in_blockquotes,
                    "require_markdown": contract.require_markdown,
                    "required_markdown_blocks": list(contract.required_blocks),
                },
                "tags": sorted(set(tags)),
                "use_for_constraint_rl_smoke": use_for_rl,
                "review": {
                    "status": "accepted",
                    "reviewer": reviewer.strip(),
                    "notes": review_notes,
                },
                "provenance": {field: provenance[field].strip() for field in required_provenance},
                "prompt_sha256": digest,
            }
        )
    return validated


def _sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
            "enable_thinking": False,
            "example_id": row["id"],
            "tags": row["tags"],
            "provenance": row["provenance"],
        }
        for row in rows
    ]


def _rl_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "data_source": "glm52_quality",
            "prompt": [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["prompt"]},
            ],
            "ability": "ru_markdown_accidental_han_constraints",
            "reward_model": {"style": "rule", "ground_truth": row["response"]},
            "extra_info": {
                **row["contract"],
                "example_id": row["id"],
                "constraint_only_smoke": True,
            },
        }
        for row in rows
        if row["use_for_constraint_rl_smoke"]
    ]


def write_artifacts(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["split"] for row in rows)
    for split in sorted(VALID_SPLITS):
        split_rows = [row for row in rows if row["split"] == split]
        if split_rows:
            pq.write_table(
                pa.Table.from_pylist(_sft_rows(split_rows)),
                output_dir / f"sft_{split}.parquet",
            )

    rl_rows = _rl_rows([row for row in rows if row["split"] == "train"])
    if rl_rows:
        pq.write_table(
            pa.Table.from_pylist(rl_rows),
            output_dir / "rl_constraint_smoke.parquet",
        )

    eval_rows = [row for row in rows if row["split"] != "train"]
    eval_path = output_dir / "eval_contracts.jsonl"
    eval_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in eval_rows))
    manifest = {
        "schema_version": 1,
        "counts": dict(sorted(counts.items())),
        "rl_constraint_smoke_count": len(rl_rows),
        "eval_count": len(eval_rows),
        "example_ids": [row["id"] for row in rows],
        "prompt_sha256": {row["id"]: row["prompt_sha256"] for row in rows},
        "source_counts": dict(sorted(Counter(row["provenance"]["dataset"] for row in rows).items())),
        "warning": ("rl_constraint_smoke.parquet has no semantic-quality reward and is not a production RL dataset"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = write_artifacts(validate_rows(read_jsonl(args.input_jsonl)), args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
