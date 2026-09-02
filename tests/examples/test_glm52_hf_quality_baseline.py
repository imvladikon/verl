from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from generate_full_quality_baseline_hf import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    REVISION_ACK,
    canonical_sha256,
    build_manifest,
    is_retryable_provider_error,
    prompt_sha256,
    read_contract_rows,
    read_existing,
    response_row,
    select_rows,
    validate_runtime_acks,
)


def source_row(example_id: str, *, split: str = "validation") -> dict:
    prompt = f"Русский запрос {example_id}"
    return {
        "id": example_id,
        "split": split,
        "system": "Отвечай по-русски.",
        "prompt": prompt,
        "prompt_sha256": prompt_sha256(prompt),
        "contract": {
            "requested_language": "ru",
            "allow_han": False,
            "require_markdown": False,
            "required_markdown_blocks": [],
        },
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_reader_verifies_split_ids_and_prompt_hash(tmp_path: Path) -> None:
    source = tmp_path / "contracts.jsonl"
    write_rows(source, [source_row("v"), source_row("t", split="test")])
    assert [row["id"] for row in read_contract_rows([source], split="validation")] == ["v"]

    invalid = source_row("bad")
    invalid["prompt_sha256"] = "0" * 64
    write_rows(source, [invalid])
    with pytest.raises(ValueError, match="prompt SHA-256 mismatch"):
        read_contract_rows([source], split="validation")


def test_prompt_hash_matches_dataset_whitespace_and_case_contract() -> None:
    assert prompt_sha256("  ПРИВЕТ\n\tмир ") == prompt_sha256("привет мир")


def test_explicit_ids_remain_inside_billing_cap() -> None:
    rows = [source_row("a"), source_row("b"), source_row("c")]
    assert [row["id"] for row in select_rows(rows, ids=["c", "a"], max_examples=2)] == ["c", "a"]
    with pytest.raises(ValueError, match="billing cap"):
        select_rows(rows, ids=["a", "b", "c"], max_examples=2)


def test_runtime_requires_exact_dynamic_billing_and_revision_acks() -> None:
    args = argparse.Namespace(
        max_examples=12,
        max_tokens=256,
        billing_ack="max_examples=12,max_tokens=256",
        unverified_revision_ack=REVISION_ACK,
    )
    validate_runtime_acks(args)
    args.billing_ack = "yes"
    with pytest.raises(SystemExit, match="billing acknowledgement mismatch"):
        validate_runtime_acks(args)


def test_response_row_preserves_request_and_unverified_revision() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Корректный ответ."),
                finish_reason="stop",
            )
        ],
        model="glm-5.2",
        request_id="request-1",
        created=123,
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
        ),
    )
    generated = response_row(
        source_row("a"),
        response,
        provider="zai-org",
        decoding_contract_sha256="d" * 64,
    )
    assert generated["completion"] == "Корректный ответ."
    assert generated["completion_token_count"] == 4
    assert generated["generation"] == {
        "provider": "zai-org",
        "requested_model": MODEL_ID,
        "requested_hub_revision": MODEL_REVISION,
        "provider_revision_verified": False,
        "served_model": "glm-5.2",
        "request_id": "request-1",
        "created": 123,
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert len(generated["request_messages_sha256"]) == 64


def test_resume_rejects_a_different_decoding_contract(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps({"id": "a", "decoding_contract_sha256": "old"}) + "\n"
    )
    with pytest.raises(ValueError, match="decoding contract differs"):
        read_existing(output, decoding_hash=canonical_sha256({"new": True}))


def test_resume_rejects_stale_prompt_or_quality_contract(tmp_path: Path) -> None:
    source = source_row("a")
    messages = [
        {"role": "system", "content": source["system"]},
        {"role": "user", "content": source["prompt"]},
    ]
    decoding_hash = canonical_sha256({"temperature": 0.0})
    stored = {
        "id": "a",
        "contract": source["contract"],
        "prompt_sha256": source["prompt_sha256"],
        "request_messages_sha256": canonical_sha256(messages),
        "decoding_contract_sha256": decoding_hash,
    }
    output = tmp_path / "predictions.jsonl"
    write_rows(output, [stored])
    assert read_existing(
        output,
        decoding_hash=decoding_hash,
        expected_sources={"a": source},
    ) == {"a"}

    changed = source_row("a")
    changed["contract"] = {**changed["contract"], "require_markdown": True}
    with pytest.raises(ValueError, match="quality contract differs"):
        read_existing(
            output,
            decoding_hash=decoding_hash,
            expected_sources={"a": changed},
        )


def test_partial_manifest_hashes_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"id":"a"}\n', encoding="utf-8")
    manifest = build_manifest(
        {"selected_count": 2},
        output=output,
        completed_count=1,
        status="PARTIAL",
    )
    assert manifest["completed_count"] == 1
    assert manifest["status"] == "PARTIAL"
    assert len(manifest["output_sha256"]) == 64


def test_billing_and_auth_errors_are_not_retried() -> None:
    for status_code in (400, 401, 402, 403, 404, 409, 422):
        error = RuntimeError("provider")
        error.response = SimpleNamespace(status_code=status_code)
        assert is_retryable_provider_error(error) is False
    assert is_retryable_provider_error(RuntimeError("network")) is True
