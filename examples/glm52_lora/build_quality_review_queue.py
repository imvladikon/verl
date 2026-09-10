#!/usr/bin/env python3
"""Build a pinned human-review queue from permissive Russian instruction data.

The output is intentionally not accepted by ``build_quality_dataset.py``.
Every row has ``review.status=pending`` and must be read, corrected when
necessary, and explicitly accepted by a named reviewer before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from build_quality_dataset import DEFAULT_SYSTEM, prompt_digest
from quality_reward import MARKDOWN, QualityContract, score_constraints

AYA_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
OASST1_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"


@dataclass(frozen=True)
class SourceLock:
    name: str
    dataset: str
    revision: str
    license: str
    splits: tuple[str, ...]


SOURCE_LOCKS = {
    "aya": SourceLock(
        name="aya",
        dataset="CohereLabs/aya_dataset",
        revision=AYA_REVISION,
        license="apache-2.0",
        splits=("train",),
    ),
    "oasst1": SourceLock(
        name="oasst1",
        dataset="OpenAssistant/oasst1",
        revision=OASST1_REVISION,
        license="apache-2.0",
        splits=("train", "validation"),
    ),
}

MARKDOWN_WORD_RE = re.compile(r"\b(?:markdown|маркдаун(?:е|ом)?)\b", re.IGNORECASE)
BLOCK_DEMAND_RE = {
    "heading": re.compile(r"\b(?:заголов\w*|раздел\w*)\b", re.IGNORECASE),
    "list": re.compile(
        r"\b(?:спис\w*|перечисл\w*|перечень|пункт\w*|нумерац\w*|нумерованн\w*|маркированн\w*|bullets?)\b",
        re.IGNORECASE,
    ),
    "table": re.compile(r"\b(?:таблиц\w*|табличн\w*)\b", re.IGNORECASE),
    "code": re.compile(
        r"\b(?:блок\w* (?:кода|markdown)|markdown-блок\w*|fenced-блок\w*|"
        r"напиши код|приведи код|исходн\w* код|скрипт\w*|программ\w*)\b",
        re.IGNORECASE,
    ),
}
TOKEN_TYPES = {
    "heading": {"heading_open"},
    "list": {"bullet_list_open", "ordered_list_open"},
    "table": {"table_open"},
    "code": {"fence"},
}
BAD_OASST_LABELS = frozenset(
    {
        "fails_task",
        "hate_speech",
        "lang_mismatch",
        "not_appropriate",
        "pii",
        "sexual_content",
        "spam",
        "toxicity",
        "violence",
    }
)


def _label_values(labels: Any) -> dict[str, float]:
    if isinstance(labels, dict):
        names = labels.get("name") or []
        values = labels.get("value") or []
        return {str(name): float(value) for name, value in zip(names, values, strict=False) if value is not None}
    if isinstance(labels, list):
        return {
            str(item["name"]): float(item["value"])
            for item in labels
            if isinstance(item, dict) and item.get("name") is not None and item.get("value") is not None
        }
    return {}


def _markdown_contract(prompt: str, response: str) -> tuple[QualityContract, tuple[str, ...]]:
    token_types = {token.type for token in MARKDOWN.parse(response)}
    demanded = tuple(name for name, pattern in BLOCK_DEMAND_RE.items() if pattern.search(prompt))
    missing = tuple(name for name in demanded if not TOKEN_TYPES[name].intersection(token_types))
    require_markdown = bool(demanded) or bool(MARKDOWN_WORD_RE.search(prompt))
    if require_markdown and not demanded:
        structural = set().union(*TOKEN_TYPES.values()) | {"blockquote_open"}
        if not structural.intersection(token_types):
            missing = ("structure",)
    return (
        QualityContract(
            requested_language="ru",
            allow_han=False,
            require_markdown=require_markdown,
            required_blocks=demanded,
        ),
        missing,
    )


def _pair_rejection(prompt: str, response: str) -> tuple[str | None, QualityContract]:
    prompt_result = score_constraints(prompt, QualityContract(requested_language="ru"))
    if prompt_result.han_count:
        return "prompt_han", QualityContract()
    if prompt_result.cyrillic_count < 5:
        return "prompt_too_little_cyrillic", QualityContract()
    if prompt_result.cyrillic_count < prompt_result.latin_count:
        return "prompt_not_predominantly_cyrillic", QualityContract()

    contract, missing = _markdown_contract(prompt, response)
    if missing:
        return f"requested_markdown_missing:{','.join(missing)}", contract
    result = score_constraints(response, contract)
    if result.han_count:
        return "response_han", contract
    if result.cyrillic_count < 20:
        return "response_too_little_cyrillic", contract
    if result.russian_script_score < 1.0:
        return "response_not_predominantly_cyrillic", contract
    if result.markdown_defects:
        return "response_markdown_defect", contract
    if result.visible_character_count < 40:
        return "response_too_short", contract
    if result.visible_character_count > 12_000:
        return "response_too_long", contract
    return None, contract


def _candidate(
    *,
    source: SourceLock,
    source_split: str,
    source_record_id: str,
    prompt: str,
    response: str,
    tags: Iterable[str],
    source_holdout: bool,
    contract: QualityContract,
) -> dict[str, Any]:
    identifier_seed = f"{source.name}\0{source.revision}\0{source_split}\0{source_record_id}"
    identifier = hashlib.sha256(identifier_seed.encode()).hexdigest()[:20]
    return {
        "id": f"{source.name}-{identifier}",
        "split": "test" if source_holdout else "pending",
        "prompt": prompt.strip(),
        "response": response.strip(),
        "system": DEFAULT_SYSTEM,
        "contract": {
            "requested_language": contract.requested_language,
            "allow_han": contract.allow_han,
            "allow_han_in_blockquotes": contract.allow_han_in_blockquotes,
            "require_markdown": contract.require_markdown,
            "required_markdown_blocks": list(contract.required_blocks),
        },
        "tags": sorted(set(tags)),
        "use_for_constraint_rl_smoke": False,
        "review": {
            "status": "pending",
            "reviewer": None,
            "method": "human",
            "notes": "",
        },
        "provenance": {
            "dataset": source.dataset,
            "revision": source.revision,
            "license": source.license,
            "source_split": source_split,
            "source_record_id": source_record_id,
        },
        "source_holdout": source_holdout,
    }


def candidate_from_aya(
    row: dict[str, Any], row_index: int, *, source_split: str = "train"
) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("language_code") != "rus":
        return None, "non_russian"
    prompt = row.get("inputs")
    response = row.get("targets")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None, "invalid_schema"
    rejection, contract = _pair_rejection(prompt, response)
    if rejection:
        return None, rejection
    annotation_type = str(row.get("annotation_type") or "unknown")
    return (
        _candidate(
            source=SOURCE_LOCKS["aya"],
            source_split=source_split,
            source_record_id=f"row:{row_index}",
            prompt=prompt,
            response=response,
            tags=("source:aya", f"annotation:{annotation_type}"),
            source_holdout=source_split != "train",
            contract=contract,
        ),
        None,
    )


def candidates_from_oasst(
    rows: Iterable[dict[str, Any]], *, source_split: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    roots: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        message_id = row.get("message_id")
        if not isinstance(message_id, str):
            rejected["invalid_schema"] += 1
            continue
        if row.get("parent_id") is None:
            roots[message_id] = row
            continue
        if row.get("role") != "assistant" or row.get("lang") != "ru":
            continue
        parent = roots.get(str(row.get("parent_id")))
        if parent is None:
            rejected["not_root_reply"] += 1
            continue
        if parent.get("role") != "prompter" or parent.get("lang") != "ru":
            rejected["parent_language_or_role"] += 1
            continue
        if parent.get("review_result") is not True or parent.get("deleted") is True:
            rejected["parent_not_accepted"] += 1
            continue
        if parent.get("synthetic") is True:
            rejected["synthetic_parent"] += 1
            continue
        if row.get("review_result") is not True or row.get("deleted") is True:
            rejected["not_accepted"] += 1
            continue
        if row.get("synthetic") is True:
            rejected["synthetic"] += 1
            continue
        if row.get("rank") not in (None, 0):
            rejected["not_best_rank"] += 1
            continue
        if int(row.get("review_count") or 0) < 3:
            rejected["insufficient_reviews"] += 1
            continue
        labels = _label_values(row.get("labels"))
        if labels.get("quality", -1.0) < 0.75:
            rejected["quality_below_0.75"] += 1
            continue
        if any(labels.get(label, 0.0) > 0.0 for label in BAD_OASST_LABELS):
            rejected["negative_label"] += 1
            continue
        prompt = parent.get("text")
        response = row.get("text")
        if not isinstance(prompt, str) or not isinstance(response, str):
            rejected["invalid_schema"] += 1
            continue
        rejection, contract = _pair_rejection(prompt, response)
        if rejection:
            rejected[rejection] += 1
            continue
        candidates.append(
            _candidate(
                source=SOURCE_LOCKS["oasst1"],
                source_split=source_split,
                source_record_id=message_id,
                prompt=prompt,
                response=response,
                tags=("source:oasst1", "human-rated", "root-turn"),
                source_holdout=source_split != "train",
                contract=contract,
            )
        )
    return candidates, rejected


def _split_from_digest(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def deduplicate_and_split(
    candidates: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            not row["source_holdout"],
            0 if row["provenance"]["dataset"] == SOURCE_LOCKS["oasst1"].dataset else 1,
            row["id"],
        ),
    )
    kept: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    rejected: Counter[str] = Counter()
    for row in ordered:
        digest = prompt_digest(row["prompt"])
        if digest in seen_prompts:
            rejected["duplicate_prompt"] += 1
            continue
        seen_prompts.add(digest)
        row = dict(row)
        row["split"] = "test" if row.pop("source_holdout") else _split_from_digest(digest)
        row["prompt_sha256"] = digest
        kept.append(row)
    return sorted(kept, key=lambda row: (row["split"], row["id"])), rejected


def _bounded(rows: Iterable[dict[str, Any]], limit: int) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if limit and index >= limit:
            break
        yield row


def _counted(rows: Iterable[dict[str, Any]], counter: Counter[str], key: str) -> Iterator[dict[str, Any]]:
    for row in rows:
        counter[key] += 1
        yield row


def collect_live_candidates(
    source_names: tuple[str, ...], *, max_source_rows: int = 0
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], Counter[str]]:
    from datasets import load_dataset

    candidates: list[dict[str, Any]] = []
    processed: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for source_name in source_names:
        source = SOURCE_LOCKS[source_name]
        for split in source.splits:
            stream = load_dataset(
                source.dataset,
                split=split,
                revision=source.revision,
                streaming=True,
            )
            source_key = f"{source_name}:{split}"
            if source_name == "aya":
                for row_index, row in enumerate(_bounded(stream, max_source_rows)):
                    processed[source_key] += 1
                    candidate, reason = candidate_from_aya(row, row_index, source_split=split)
                    if candidate is not None:
                        candidates.append(candidate)
                        accepted[f"{source_name}:{split}"] += 1
                    else:
                        rejected[f"{source_name}:{split}:{reason}"] += 1
            else:
                source_candidates, source_rejected = candidates_from_oasst(
                    _counted(_bounded(stream, max_source_rows), processed, source_key),
                    source_split=split,
                )
                candidates.extend(source_candidates)
                accepted[f"{source_name}:{split}"] += len(source_candidates)
                for reason, count in source_rejected.items():
                    rejected[f"{source_name}:{split}:{reason}"] += count
    return candidates, processed, accepted, rejected


def write_review_queue(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_names: tuple[str, ...],
    processed: Counter[str],
    accepted_before_dedupe: Counter[str],
    rejected: Counter[str],
    max_source_rows: int,
    started_at: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "quality_review_queue.jsonl"
    queue_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    source_counts = Counter(row["provenance"]["dataset"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    required_block_counts = Counter(block for row in rows for block in row["contract"]["required_markdown_blocks"])
    manifest = {
        "schema_version": 1,
        "warning": "pending human review; build_quality_dataset.py must reject this file",
        "source_locks": {
            name: {
                "dataset": SOURCE_LOCKS[name].dataset,
                "revision": SOURCE_LOCKS[name].revision,
                "license": SOURCE_LOCKS[name].license,
                "splits": list(SOURCE_LOCKS[name].splits),
            }
            for name in source_names
        },
        "max_source_rows": max_source_rows or None,
        "processed_rows": dict(sorted(processed.items())),
        "accepted_before_dedupe": dict(sorted(accepted_before_dedupe.items())),
        "rejected": dict(sorted(rejected.items())),
        "queue_count": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "markdown_required_count": sum(bool(row["contract"]["require_markdown"]) for row in rows),
        "required_block_counts": dict(sorted(required_block_counts.items())),
        "queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic() - started_at,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output_dir / "review_queue_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_LOCKS),
        dest="sources",
        help="repeat to select sources; defaults to every locked source",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=0,
        help="debug-only raw-row limit per source split; zero processes all rows",
    )
    args = parser.parse_args()
    if args.max_source_rows < 0:
        parser.error("--max-source-rows must be nonnegative")
    source_names = tuple(args.sources or SOURCE_LOCKS)
    started_at = time.monotonic()
    candidates, processed, accepted, rejected = collect_live_candidates(
        source_names, max_source_rows=args.max_source_rows
    )
    rows, dedupe_rejections = deduplicate_and_split(candidates)
    rejected.update(dedupe_rejections)
    manifest = write_review_queue(
        rows,
        args.output_dir,
        source_names=source_names,
        processed=processed,
        accepted_before_dedupe=accepted,
        rejected=rejected,
        max_source_rows=args.max_source_rows,
        started_at=started_at,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
