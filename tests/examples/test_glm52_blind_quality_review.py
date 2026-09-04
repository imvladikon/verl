# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import (
    RATING_FIELDS,
    adjudicate,
    build_packet,
    canonical_sha256,
    prompt_sha256,
    read_completed_reviews,
    read_contracts,
    require_absent,
)
from compare_quality_outputs import compare_rows
from generate_full_quality_outputs_sglang import (
    OFFICIAL_MODEL_ARTIFACTS,
    OFFICIAL_TRAINER,
    build_pair_contract,
)


def pair_provenance(source: dict) -> tuple[dict, str, str]:
    source.setdefault("tags", ["russian", "quality"])
    source.setdefault(
        "provenance",
        {
            "dataset": "project-authored/test",
            "license": "apache-2.0",
            "revision": "test-v1",
            "source_split": "generated",
            "source_record_id": source["id"],
        },
    )
    trainer = {
        "model_id": OFFICIAL_TRAINER[0],
        "revision": OFFICIAL_TRAINER[1],
        "revision_verified": True,
        **OFFICIAL_MODEL_ARTIFACTS[OFFICIAL_TRAINER],
    }
    inference_identity = (
        "zai-org/GLM-5.2-FP8",
        "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    )
    inference = {
        "model_id": inference_identity[0],
        "revision": inference_identity[1],
        "revision_verified": True,
        **OFFICIAL_MODEL_ARTIFACTS[inference_identity],
    }
    pair_runtime = {
        "schema_version": 3,
        "artifact_contract": {"trainer_base": trainer, "inference_base": inference},
        "weight_shard_manifest_sha256": {
            "trainer": "1" * 64,
            "inference": "2" * 64,
        },
        "sglang": {
            "repository": "https://github.com/imvladikon/sglang",
            "revision": "0dbdb73509fbf6b3381359df87cde267d453c8d3",
            "tree": "5678fc2ab88fd65411b833c065f510b6d4f5d59c",
        },
        "runtime_script_sha256": {
            "build_quality_sglang_runtime.py": "5" * 64,
            "generate_full_quality_outputs_sglang.py": "6" * 64,
            "launch_quality_sglang_server.py": "7" * 64,
            "build_blind_quality_review.py": "8" * 64,
        },
        "environment_semantics": {
            "python_version": "3.12.11",
            "python_executable_sha256": "9" * 64,
            "installed_distributions_sha256": "a" * 64,
        },
        "server_semantics": {
            "served_base_model": "glm52-base",
            "tp_size": 8,
            "max_model_len": 2048,
        },
    }
    decoding = {
        "schema_version": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_completion_tokens": 512,
        "seed": 52,
        "reasoning_effort": "none",
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    pair = build_pair_contract(
        source,
        runtime={"pair_runtime_contract": pair_runtime},
        decoding=decoding,
    )
    return pair, canonical_sha256(pair_runtime), canonical_sha256(decoding)


def adapter_provenance() -> dict:
    return {
        "name": "quality",
        "artifact_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "verification_sha256": "3" * 64,
        "trainer_base_revision": OFFICIAL_TRAINER[1],
        "profile": "mla-only",
        "rank": 16,
        "alpha": 32,
        "parameter_count": 106_149_888,
        "target_modules": [
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "q_a_proj",
            "q_b_proj",
        ],
    }


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
    pair, pair_runtime_hash, decoding_hash = pair_provenance(source)

    def prediction(completion: str, variant: str) -> dict:
        return {
            "id": "example",
            "split": source["split"],
            "completion": completion,
            "completion_token_count": 4,
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "source_row_sha256": pair["held_out"]["source_row_sha256"],
            "reference_response_sha256": pair["held_out"]["reference_response_sha256"],
            "request_messages_sha256": request_hash,
            "input_han_count": pair["held_out"]["input_han_count"],
            "input_contains_han": pair["held_out"]["input_contains_han"],
            "han_evaluation_mode": pair["held_out"]["han_evaluation_mode"],
            "evaluation_cluster_id": pair["held_out"]["evaluation_cluster_id"],
            "decoding_contract_sha256": decoding_hash,
            "pair_contract": pair,
            "pair_contract_sha256": canonical_sha256(pair),
            "generation": {
                "variant": variant,
                "runtime_mode": variant,
                "runtime_manifest_sha256": ("a" * 64 if variant == "base" else "b" * 64),
                "pair_runtime_contract_sha256": pair_runtime_hash,
                "quality_claim_allowed": True,
                "api_secret_sha256": "b" * 64,
                "trainer_base": pair["runtime"]["artifact_contract"]["trainer_base"],
                "inference_base": pair["runtime"]["artifact_contract"]["inference_base"],
                "adapter": None if variant == "base" else adapter_provenance(),
                "sglang": {
                    "checkout": "/src/sglang",
                    **pair["runtime"]["sglang"],
                },
                "server_instance_id": f"{variant}-instance",
                "response_id": f"response-{variant}",
                "response_model": "glm52-base" if variant == "base" else "quality",
                "finish_reason": "stop",
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            },
        }

    return (
        {"example": source},
        [prediction("Базовый ответ.", "base")],
        [prediction("Ответ адаптера.", "adapter")],
    )


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
    with pytest.raises(ValueError, match="prediction fields are invalid"):
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


def test_pair_contract_cannot_smuggle_adapter_server_fields() -> None:
    contracts, base, adapter = fixture_rows()
    pair = deepcopy(base[0]["pair_contract"])
    pair["runtime"]["server_semantics"]["enable_lora"] = True
    pair_hash = canonical_sha256(pair)
    pair_runtime_hash = canonical_sha256(pair["runtime"])
    for row in (base[0], adapter[0]):
        row["pair_contract"] = deepcopy(pair)
        row["pair_contract_sha256"] = pair_hash
        row["generation"]["pair_runtime_contract_sha256"] = pair_runtime_hash

    with pytest.raises(ValueError, match="server semantic fields are invalid"):
        build_packet(
            contracts,
            base,
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


def test_adjudicated_scores_feed_all_quality_and_retention_gates(
    tmp_path: Path,
) -> None:
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
        (
            "general-russian",
            "Ответь одним обычным русским предложением.",
            "Обычный русский ответ.",
            "Обычный русский ответ.",
            "Обычный русский ответ.",
            {
                "requested_language": "ru",
                "allow_han": False,
                "require_markdown": False,
                "required_markdown_blocks": [],
            },
        ),
        (
            "zh-retention",
            "请用中文回答。",
            "这是同样正确的中文回答。",
            "这是正确的中文回答。",
            "这是同样正确的中文回答。",
            {
                "requested_language": "zh",
                "allow_han": True,
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
            "system": ("请用中文回答。" if contract["requested_language"] == "zh" else "Отвечай по-русски."),
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
        pair, pair_runtime_hash, decoding_hash = pair_provenance(source)
        common = {
            "id": example_id,
            "split": source["split"],
            "completion_token_count": 12,
            "contract": contract,
            "prompt_sha256": source["prompt_sha256"],
            "source_row_sha256": pair["held_out"]["source_row_sha256"],
            "reference_response_sha256": pair["held_out"]["reference_response_sha256"],
            "request_messages_sha256": messages_hash,
            "input_han_count": pair["held_out"]["input_han_count"],
            "input_contains_han": pair["held_out"]["input_contains_han"],
            "han_evaluation_mode": pair["held_out"]["han_evaluation_mode"],
            "evaluation_cluster_id": pair["held_out"]["evaluation_cluster_id"],
            "decoding_contract_sha256": decoding_hash,
            "pair_contract": pair,
            "pair_contract_sha256": canonical_sha256(pair),
        }
        base.append(
            {
                **common,
                "completion": base_text,
                "generation": {
                    "variant": "base",
                    "runtime_mode": "base",
                    "runtime_manifest_sha256": "a" * 64,
                    "pair_runtime_contract_sha256": pair_runtime_hash,
                    "quality_claim_allowed": True,
                    "api_secret_sha256": "b" * 64,
                    "trainer_base": pair["runtime"]["artifact_contract"]["trainer_base"],
                    "inference_base": pair["runtime"]["artifact_contract"]["inference_base"],
                    "adapter": None,
                    "sglang": {"checkout": "/src/sglang", **pair["runtime"]["sglang"]},
                    "server_instance_id": "base-instance",
                    "response_id": f"response-base-{example_id}",
                    "response_model": "glm52-base",
                    "finish_reason": "stop",
                    "prompt_tokens": 8,
                    "completion_tokens": 12,
                    "total_tokens": 20,
                },
            }
        )
        adapter.append(
            {
                **common,
                "completion": adapter_text,
                "generation": {
                    "variant": "adapter",
                    "runtime_mode": "adapter",
                    "runtime_manifest_sha256": "b" * 64,
                    "pair_runtime_contract_sha256": pair_runtime_hash,
                    "quality_claim_allowed": True,
                    "api_secret_sha256": "c" * 64,
                    "trainer_base": pair["runtime"]["artifact_contract"]["trainer_base"],
                    "inference_base": pair["runtime"]["artifact_contract"]["inference_base"],
                    "adapter": adapter_provenance(),
                    "sglang": {"checkout": "/src/sglang", **pair["runtime"]["sglang"]},
                    "server_instance_id": "adapter-instance",
                    "response_id": f"response-adapter-{example_id}",
                    "response_model": "quality",
                    "finish_reason": "stop",
                    "prompt_tokens": 8,
                    "completion_tokens": 12,
                    "total_tokens": 20,
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

    result, _ = compare_rows(
        scored_base,
        scored_adapter,
        bootstrap_samples=100,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
        minimum_slice_rows=1,
        minimum_slice_clusters=1,
    )

    assert result["status"] == "PASS"
    assert result["target_status"] == {
        "russian_semantic_quality": "PASS",
        "required_markdown_validity": "PASS",
        "accidental_han": "PASS",
        "non_russian_semantic_retention": "PASS",
    }
