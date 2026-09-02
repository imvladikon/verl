from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import (  # noqa: E402
    RATING_FIELDS,
    adjudicate,
    build_packet,
    canonical_sha256,
    prompt_sha256,
    read_completed_reviews,
    read_contracts,
    require_absent,
)
from compare_quality_outputs import compare_rows  # noqa: E402


def fixture_rows() -> tuple[dict, list[dict], list[dict]]:
    source = {
        "id": "example",
        "split": "validation",
        "system": "Отвечай по-русски.",
        "prompt": "Исправь текст.",
        "response": "Текст исправлен.",
        "prompt_sha256": prompt_sha256("Исправь текст."),
        "contract": {
            "requested_language": "ru",
            "allow_han": False,
            "require_markdown": False,
            "required_markdown_blocks": [],
        },
    }
    request_hash = canonical_sha256(
        [
            {"role": "system", "content": source["system"]},
            {"role": "user", "content": source["prompt"]},
        ]
    )

    def prediction(completion: str, variant: str) -> dict:
        return {
            "id": "example",
            "completion": completion,
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "request_messages_sha256": request_hash,
            "decoding_contract_sha256": "d" * 64,
            "generation_pair_contract_sha256": "a" * 64,
            "generation": {
                "variant": variant,
                "runtime_manifest_sha256": "a" * 64,
                "quality_claim_allowed": True,
            },
        }

    return {
        "example": source
    }, [prediction("Базовый ответ.", "base")], [prediction("Ответ адаптера.", "adapter")]


def completed_packet(packet: list[dict], reviewer: str, a: int, b: int) -> list[dict]:
    rows = deepcopy(packet)
    for row in rows:
        row["review"]["reviewer"] = reviewer
        for label, value in (("candidate_a", a), ("candidate_b", b)):
            for field in RATING_FIELDS:
                row["review"][label][field] = value
            row["review"][label]["severe_error"] = False
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_packet_is_blinded_and_deterministic() -> None:
    contracts, base, adapter = fixture_rows()
    first = build_packet(contracts, base, adapter, blinding_key=b"0123456789abcdef")
    second = build_packet(contracts, base, adapter, blinding_key=b"0123456789abcdef")
    assert first == second
    assert {first[0]["candidate_a"], first[0]["candidate_b"]} == {
        "Базовый ответ.",
        "Ответ адаптера.",
    }
    assert "base" not in first[0]
    assert "adapter" not in first[0]
    assert first[0]["review"]["reviewer"] is None


def test_two_reviews_are_mapped_back_to_the_correct_models(tmp_path: Path) -> None:
    key = b"0123456789abcdef"
    contracts, base, adapter = fixture_rows()
    packet = build_packet(contracts, base, adapter, blinding_key=key)
    base_is_a = packet[0]["candidate_a"] == "Базовый ответ."
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_rows(first, completed_packet(packet, "reviewer-1", 2, 5))
    write_rows(second, completed_packet(packet, "reviewer-2", 2, 5))
    reviews, hashes = read_completed_reviews([first, second], packet, minimum_reviewers=2)
    scored_base, scored_adapter, summary = adjudicate(
        contracts,
        base,
        adapter,
        reviews,
        blinding_key=key,
        review_hashes=hashes,
    )
    expected_base = 0.25 if base_is_a else 1.0
    expected_adapter = 1.0 if base_is_a else 0.25
    assert scored_base[0]["semantic_score"] == expected_base
    assert scored_adapter[0]["semantic_score"] == expected_adapter
    assert summary["reviewers"] == ["reviewer-1", "reviewer-2"]


def test_modified_blinded_item_is_rejected(tmp_path: Path) -> None:
    contracts, base, adapter = fixture_rows()
    packet = build_packet(contracts, base, adapter, blinding_key=b"0123456789abcdef")
    modified = completed_packet(packet, "reviewer", 3, 3)
    modified[0]["candidate_a"] += " Подмена."
    path = tmp_path / "modified.jsonl"
    write_rows(path, modified)
    with pytest.raises(ValueError, match="review item was modified"):
        read_completed_reviews([path], packet, minimum_reviewers=1)


def test_invalid_prediction_payload_fails_closed() -> None:
    contracts, base, adapter = fixture_rows()
    missing_decoding = deepcopy(base)
    missing_decoding[0].pop("decoding_contract_sha256")
    with pytest.raises(ValueError, match="decoding contract must be a SHA-256"):
        build_packet(
            contracts,
            missing_decoding,
            adapter,
            blinding_key=b"0123456789abcdef",
        )

    invalid_completion = deepcopy(base)
    invalid_completion[0]["completion"] = 52
    with pytest.raises(ValueError, match="completion must be a nonempty string"):
        build_packet(
            contracts,
            invalid_completion,
            adapter,
            blinding_key=b"0123456789abcdef",
        )


def test_duplicate_or_insufficient_reviewers_fail_closed(tmp_path: Path) -> None:
    contracts, base, adapter = fixture_rows()
    packet = build_packet(contracts, base, adapter, blinding_key=b"0123456789abcdef")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_rows(first, completed_packet(packet, "same-reviewer", 3, 3))
    write_rows(second, completed_packet(packet, "same-reviewer", 4, 4))
    with pytest.raises(ValueError, match="at least 2"):
        read_completed_reviews([first], packet, minimum_reviewers=2)
    with pytest.raises(ValueError, match="duplicate reviewer"):
        read_completed_reviews([first, second], packet, minimum_reviewers=2)

    blank = tmp_path / "blank.jsonl"
    write_rows(blank, packet)
    with pytest.raises(ValueError, match="nonempty reviewer identity"):
        read_completed_reviews([blank], packet, minimum_reviewers=1)


def test_contract_hash_and_output_overwrite_fail_closed(tmp_path: Path) -> None:
    contracts, _, _ = fixture_rows()
    invalid = tmp_path / "invalid.jsonl"
    row = next(iter(contracts.values()))
    write_rows(invalid, [{**row, "prompt_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="held-out prompt_sha256 is invalid"):
        read_contracts([invalid], split="validation")

    output = tmp_path / "existing.jsonl"
    output.write_text("evidence\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        require_absent([output], overwrite=False)
    require_absent([output], overwrite=True)


def test_adjudicated_scores_feed_the_three_target_comparator(tmp_path: Path) -> None:
    key = b"0123456789abcdef"
    examples = [
        (
            "markdown",
            "Оформи ответ как заголовок и список.",
            "## Ответ\n\n- Пункт.",
            "сломанный markdown",
            "## Ответ\n\n- Пункт.",
            {
                "requested_language": "ru",
                "allow_han": False,
                "require_markdown": True,
                "required_markdown_blocks": ["heading", "list"],
            },
        ),
        (
            "han",
            "Убери случайные китайские знаки.",
            "Русский текст.",
            "Русский текст 中.",
            "Русский текст.",
            {
                "requested_language": "ru",
                "allow_han": False,
                "require_markdown": False,
                "required_markdown_blocks": [],
            },
        ),
    ]
    contracts: dict[str, dict] = {}
    base: list[dict] = []
    adapter: list[dict] = []
    for example_id, prompt, reference, base_text, adapter_text, contract in examples:
        source = {
            "id": example_id,
            "split": "validation",
            "system": "Отвечай по-русски.",
            "prompt": prompt,
            "response": reference,
            "prompt_sha256": prompt_sha256(prompt),
            "contract": contract,
        }
        contracts[example_id] = source
        messages_hash = canonical_sha256(
            [
                {"role": "system", "content": source["system"]},
                {"role": "user", "content": prompt},
            ]
        )
        common = {
            "id": example_id,
            "completion_token_count": 12,
            "contract": contract,
            "prompt_sha256": source["prompt_sha256"],
            "request_messages_sha256": messages_hash,
            "decoding_contract_sha256": "d" * 64,
            "generation_pair_contract_sha256": "a" * 64,
        }
        base.append(
            {
                **common,
                "completion": base_text,
                "generation": {
                    "variant": "base",
                    "runtime_manifest_sha256": "a" * 64,
                    "quality_claim_allowed": True,
                },
            }
        )
        adapter.append(
            {
                **common,
                "completion": adapter_text,
                "generation": {
                    "variant": "adapter",
                    "runtime_manifest_sha256": "a" * 64,
                    "quality_claim_allowed": True,
                },
            }
        )

    packet = build_packet(contracts, base, adapter, blinding_key=key)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_rows = deepcopy(packet)
    second_rows = deepcopy(packet)
    adapter_texts = {example[4] for example in examples}
    for reviewer, rows in (("reviewer-1", first_rows), ("reviewer-2", second_rows)):
        for row in rows:
            row["review"]["reviewer"] = reviewer
            for candidate in ("candidate_a", "candidate_b"):
                rating = 5 if row[candidate] in adapter_texts else 2
                for field in RATING_FIELDS:
                    row["review"][candidate][field] = rating
                row["review"][candidate]["severe_error"] = False
    write_rows(first, first_rows)
    write_rows(second, second_rows)
    reviews, hashes = read_completed_reviews([first, second], packet, minimum_reviewers=2)
    scored_base, scored_adapter, _ = adjudicate(
        contracts,
        base,
        adapter,
        reviews,
        blinding_key=key,
        review_hashes=hashes,
    )

    result, _ = compare_rows(scored_base, scored_adapter, bootstrap_samples=100)

    assert result["status"] == "PASS"
    assert result["target_status"] == {
        "russian_semantic_quality": "PASS",
        "required_markdown_validity": "PASS",
        "accidental_han": "PASS",
    }
