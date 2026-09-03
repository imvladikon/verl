from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from compare_quality_outputs import compare_rows, main  # noqa: E402


def _row(
    example_id: str,
    completion: str,
    semantic_score: float | None,
    *,
    require_markdown: bool = False,
    requested_language: str = "ru",
    allow_han: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": example_id,
        "prompt_sha256": f"prompt-{example_id}",
        "decoding_contract_sha256": "greedy-v1",
        "generation_pair_contract_sha256": "a" * 64,
        "generation": {
            "variant": "base",
            "runtime_manifest_sha256": "a" * 64,
            "quality_claim_allowed": True,
        },
        "completion": completion,
        "completion_token_count": 12,
        "contract": {
            "requested_language": requested_language,
            "allow_han": allow_han,
            "require_markdown": require_markdown,
            "required_markdown_blocks": ["heading", "list"] if require_markdown else [],
        },
    }
    if semantic_score is not None:
        row["semantic_score"] = semantic_score
    return row


def _adapter(row: dict[str, object]) -> dict[str, object]:
    row["generation"] = {
        "variant": "adapter",
        "runtime_manifest_sha256": row["generation_pair_contract_sha256"],
        "quality_claim_allowed": True,
    }
    return row


def test_comparator_passes_only_paired_three_target_improvement() -> None:
    base = [
        _row("markdown-a", "сломанный markdown", 0.2, require_markdown=True),
        _row("markdown-b", "ещё один сломанный markdown", 0.3, require_markdown=True),
        _row("han-a", "Русский текст 中", 0.4),
        _row("han-b", "Ещё один русский 结果", 0.5),
    ]
    adapter = [
        _adapter(_row("markdown-a", "## Ответ\n\n1. Первый пункт.", 0.7, require_markdown=True)),
        _adapter(_row("markdown-b", "## Итог\n\n1. Второй пункт.", 0.8, require_markdown=True)),
        _adapter(_row("han-a", "Русский текст.", 0.9)),
        _adapter(_row("han-b", "Ещё один русский текст.", 1.0)),
    ]

    result, details = compare_rows(base, adapter, bootstrap_samples=500)

    assert result["status"] == "PASS"
    assert result["target_status"] == {
        "russian_semantic_quality": "PASS",
        "required_markdown_validity": "PASS",
        "accidental_han": "PASS",
        "non_russian_semantic_retention": "NOT_APPLICABLE",
    }
    assert result["base_required_markdown_defects"] == 2
    assert result["adapter_required_markdown_defects"] == 0
    assert result["base_accidental_han_examples"] == 2
    assert result["adapter_accidental_han_examples"] == 0
    assert len(details) == 4


def test_comparator_stays_pending_without_semantics_or_reproduced_defects() -> None:
    base = [_row("clean", "Корректный русский текст.", None)]
    adapter = [_adapter(_row("clean", "Другой корректный русский текст.", None))]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "PENDING"
    assert result["target_status"] == {
        "russian_semantic_quality": "PENDING",
        "required_markdown_validity": "NOT_REPRODUCED",
        "accidental_han": "NOT_REPRODUCED",
        "non_russian_semantic_retention": "NOT_APPLICABLE",
    }
    assert result["paired_russian_semantic_coverage"] == 0.0


def test_comparator_reports_semantic_regression_as_failure() -> None:
    base = [_row("semantic", "Хороший русский ответ 中", 0.9)]
    adapter = [_adapter(_row("semantic", "Плохой русский ответ.", 0.1))]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "FAIL"
    assert result["target_status"]["russian_semantic_quality"] == "FAIL"


def test_comparator_requires_legitimate_chinese_semantic_retention() -> None:
    base = [
        _row("markdown", "сломанный markdown", 0.3, require_markdown=True),
        _row("han", "Русский текст 中", 0.5),
        _row(
            "zh-retention",
            "这是一个正确的中文回答。",
            0.9,
            requested_language="zh",
            allow_han=True,
        ),
    ]
    adapter = [
        _adapter(
            _row(
                "markdown",
                "## Ответ\n\n- Исправлено.",
                0.6,
                require_markdown=True,
            )
        ),
        _adapter(_row("han", "Русский текст.", 0.8)),
        _adapter(
            _row(
                "zh-retention",
                "Нерелевантный русский ответ.",
                0.1,
                requested_language="zh",
                allow_han=True,
            )
        ),
    ]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "FAIL"
    assert result["target_status"]["non_russian_semantic_retention"] == "FAIL"
    assert result["paired_retention_semantic_coverage"] == 1.0


def test_comparator_accepts_noninferior_chinese_retention() -> None:
    base = [
        _row("markdown", "сломанный markdown", 0.3, require_markdown=True),
        _row("han", "Русский текст 中", 0.5),
        _row(
            "zh-retention",
            "这是一个正确的中文回答。",
            0.8,
            requested_language="zh",
            allow_han=True,
        ),
    ]
    adapter = [
        _adapter(
            _row(
                "markdown",
                "## Ответ\n\n- Исправлено.",
                0.6,
                require_markdown=True,
            )
        ),
        _adapter(_row("han", "Русский текст.", 0.8)),
        _adapter(
            _row(
                "zh-retention",
                "这是一个同样正确的中文回答。",
                0.8,
                requested_language="zh",
                allow_han=True,
            )
        ),
    ]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "PASS"
    assert result["target_status"]["non_russian_semantic_retention"] == "PASS"


def test_comparator_stays_pending_without_chinese_retention_scores() -> None:
    base = [
        _row("markdown", "сломанный markdown", 0.3, require_markdown=True),
        _row("han", "Русский текст 中", 0.5),
        _row(
            "zh-retention",
            "这是一个正确的中文回答。",
            None,
            requested_language="zh",
            allow_han=True,
        ),
    ]
    adapter = [
        _adapter(
            _row(
                "markdown",
                "## Ответ\n\n- Исправлено.",
                0.6,
                require_markdown=True,
            )
        ),
        _adapter(_row("han", "Русский текст.", 0.8)),
        _adapter(
            _row(
                "zh-retention",
                "这是一个同样正确的中文回答。",
                None,
                requested_language="zh",
                allow_han=True,
            )
        ),
    ]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "PENDING"
    assert result["target_status"]["non_russian_semantic_retention"] == "PENDING"
    assert result["paired_retention_semantic_coverage"] == 0.0


@pytest.mark.parametrize(
    ("field", "new_value", "message"),
    [
        ("prompt_sha256", "different", "prompt_sha256"),
        ("request_messages_sha256", "different", "request_messages_sha256"),
        ("decoding_contract_sha256", "different", "decoding_contract_sha256"),
        ("contract", {"requested_language": "zh"}, "contracts differ"),
    ],
)
def test_comparator_rejects_unpaired_contracts(field: str, new_value: object, message: str) -> None:
    base = [_row("same", "Ответ 中", 0.4)]
    adapter_row = _adapter(_row("same", "Ответ.", 0.6))
    if field == "request_messages_sha256":
        base[0][field] = "original"
    adapter_row[field] = new_value

    with pytest.raises(ValueError, match=message):
        compare_rows(base, [adapter_row], bootstrap_samples=100)


def test_comparator_rejects_unpaired_semantic_coverage_and_ids() -> None:
    base = [_row("base", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("base", "Ответ.", None))]
    with pytest.raises(ValueError, match="semantic score coverage"):
        compare_rows(base, adapter, bootstrap_samples=100)

    with pytest.raises(ValueError, match="ids differ"):
        compare_rows(
            base,
            [_adapter(_row("adapter", "Ответ.", 0.6))],
            bootstrap_samples=100,
        )


def test_comparator_rejects_surgery_or_unproven_runtime() -> None:
    base = [_row("test-only", "Русский ответ 中", 0.4)]
    adapter = [_adapter(_row("test-only", "Русский ответ.", 0.8))]
    base[0]["generation"]["quality_claim_allowed"] = False
    adapter[0]["generation"]["quality_claim_allowed"] = False

    with pytest.raises(ValueError, match="not a full-model quality oracle"):
        compare_rows(base, adapter, bootstrap_samples=100)

    del base[0]["generation_pair_contract_sha256"]
    with pytest.raises(ValueError, match="generation_pair_contract_sha256"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_cli_writes_hashed_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    details_path = tmp_path / "details.jsonl"
    base_path.write_text(json.dumps(_row("paired", "Ответ 中", 0.4)) + "\n")
    adapter_path.write_text(json.dumps(_adapter(_row("paired", "Ответ.", 0.7))) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_quality_outputs.py",
            str(base_path),
            str(adapter_path),
            "--details",
            str(details_path),
            "--bootstrap-samples",
            "100",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PENDING"
    assert len(result["base_predictions_sha256"]) == 64
    assert len(result["adapter_predictions_sha256"]) == 64
    assert len(result["details_sha256"]) == 64
    assert json.loads(details_path.read_text())["id"] == "paired"
