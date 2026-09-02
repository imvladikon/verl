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
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": example_id,
        "prompt_sha256": f"prompt-{example_id}",
        "decoding_contract_sha256": "greedy-v1",
        "completion": completion,
        "completion_token_count": 12,
        "contract": {
            "requested_language": "ru",
            "allow_han": False,
            "require_markdown": require_markdown,
            "required_markdown_blocks": ["heading", "list"] if require_markdown else [],
        },
    }
    if semantic_score is not None:
        row["semantic_score"] = semantic_score
    return row


def test_comparator_passes_only_paired_three_target_improvement() -> None:
    base = [
        _row("markdown-a", "сломанный markdown", 0.2, require_markdown=True),
        _row("markdown-b", "ещё один сломанный markdown", 0.3, require_markdown=True),
        _row("han-a", "Русский текст 中", 0.4),
        _row("han-b", "Ещё один русский 结果", 0.5),
    ]
    adapter = [
        _row("markdown-a", "## Ответ\n\n1. Первый пункт.", 0.7, require_markdown=True),
        _row("markdown-b", "## Итог\n\n1. Второй пункт.", 0.8, require_markdown=True),
        _row("han-a", "Русский текст.", 0.9),
        _row("han-b", "Ещё один русский текст.", 1.0),
    ]

    result, details = compare_rows(base, adapter, bootstrap_samples=500)

    assert result["status"] == "PASS"
    assert result["target_status"] == {
        "russian_semantic_quality": "PASS",
        "required_markdown_validity": "PASS",
        "accidental_han": "PASS",
    }
    assert result["base_required_markdown_defects"] == 2
    assert result["adapter_required_markdown_defects"] == 0
    assert result["base_accidental_han_examples"] == 2
    assert result["adapter_accidental_han_examples"] == 0
    assert len(details) == 4


def test_comparator_stays_pending_without_semantics_or_reproduced_defects() -> None:
    base = [_row("clean", "Корректный русский текст.", None)]
    adapter = [_row("clean", "Другой корректный русский текст.", None)]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "PENDING"
    assert result["target_status"] == {
        "russian_semantic_quality": "PENDING",
        "required_markdown_validity": "NOT_REPRODUCED",
        "accidental_han": "NOT_REPRODUCED",
    }
    assert result["paired_russian_semantic_coverage"] == 0.0


def test_comparator_reports_semantic_regression_as_failure() -> None:
    base = [_row("semantic", "Хороший русский ответ 中", 0.9)]
    adapter = [_row("semantic", "Плохой русский ответ.", 0.1)]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "FAIL"
    assert result["target_status"]["russian_semantic_quality"] == "FAIL"


@pytest.mark.parametrize(
    ("field", "new_value", "message"),
    [
        ("prompt_sha256", "different", "prompt_sha256"),
        ("decoding_contract_sha256", "different", "decoding_contract_sha256"),
        ("contract", {"requested_language": "zh"}, "contracts differ"),
    ],
)
def test_comparator_rejects_unpaired_contracts(field: str, new_value: object, message: str) -> None:
    base = [_row("same", "Ответ 中", 0.4)]
    adapter_row = _row("same", "Ответ.", 0.6)
    adapter_row[field] = new_value

    with pytest.raises(ValueError, match=message):
        compare_rows(base, [adapter_row], bootstrap_samples=100)


def test_comparator_rejects_unpaired_semantic_coverage_and_ids() -> None:
    base = [_row("base", "Ответ 中", 0.4)]
    adapter = [_row("base", "Ответ.", None)]
    with pytest.raises(ValueError, match="semantic score coverage"):
        compare_rows(base, adapter, bootstrap_samples=100)

    with pytest.raises(ValueError, match="ids differ"):
        compare_rows(
            base,
            [_row("adapter", "Ответ.", 0.6)],
            bootstrap_samples=100,
        )


def test_comparator_cli_writes_hashed_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    details_path = tmp_path / "details.jsonl"
    base_path.write_text(json.dumps(_row("paired", "Ответ 中", 0.4)) + "\n")
    adapter_path.write_text(json.dumps(_row("paired", "Ответ.", 0.7)) + "\n")
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
