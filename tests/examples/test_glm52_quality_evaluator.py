from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from evaluate_quality_outputs import evaluate_rows  # noqa: E402


def test_evaluator_reports_conditional_han_markdown_and_semantic_metrics() -> None:
    rows = [
        {
            "id": "ru-clean",
            "completion": "## Ответ\n\n- Проверка завершена.",
            "completion_token_count": 12,
            "semantic_score": 0.9,
            "contract": {
                "requested_language": "ru",
                "allow_han": False,
                "require_markdown": True,
                "required_markdown_blocks": ["heading", "list"],
            },
        },
        {
            "id": "ru-dirty",
            "completion": "Русский ответ 中 с **ошибкой",
            "completion_token_count": 10,
            "semantic_score": 0.4,
            "contract": {"requested_language": "ru", "allow_han": False},
        },
        {
            "id": "zh-retention",
            "completion": "这是正确的中文回答。",
            "completion_token_count": 8,
            "contract": {"requested_language": "zh", "allow_han": True},
        },
    ]
    summary, details = evaluate_rows(rows)

    assert summary["count"] == 3
    assert summary["accidental_han_applicable_count"] == 2
    assert summary["accidental_han_count"] == 1
    assert summary["accidental_han_example_rate"] == pytest.approx(1 / 2)
    assert summary["accidental_han_per_1000_tokens"] == pytest.approx(1000 / 22)
    assert summary["token_count_coverage"] == 1.0
    assert summary["russian_example_count"] == 2
    assert summary["markdown_structural_valid_rate"] == pytest.approx(2 / 3)
    assert summary["required_markdown_valid_rate"] == 1.0
    assert summary["semantic_score_mean"] == pytest.approx(0.65)
    assert summary["semantic_score_coverage"] == pytest.approx(2 / 3)
    assert details[2]["accidental_han_count"] == 0


def test_evaluator_does_not_invent_token_or_semantic_metrics() -> None:
    summary, _ = evaluate_rows(
        [
            {
                "id": "ru",
                "completion": "Корректный ответ.",
                "contract": {"requested_language": "ru"},
            }
        ]
    )
    assert summary["accidental_han_per_1000_tokens"] is None
    assert summary["token_count_coverage"] == 0.0
    assert summary["semantic_score_mean"] is None
    assert summary["semantic_score_coverage"] == 0.0
    assert math.isfinite(summary["constraint_score_mean"])


def test_allowed_chinese_examples_do_not_create_fake_han_denominators() -> None:
    summary, _ = evaluate_rows(
        [
            {
                "id": "zh",
                "completion": "这是正确的中文回答。",
                "completion_token_count": 8,
                "contract": {"requested_language": "zh", "allow_han": True},
            }
        ]
    )
    assert summary["accidental_han_applicable_count"] == 0
    assert summary["accidental_han_example_rate"] is None
    assert summary["accidental_han_per_1000_visible_characters"] is None
    assert summary["accidental_han_per_1000_tokens"] is None
    assert summary["token_count_coverage"] is None
    assert summary["russian_script_score_mean"] is None


def test_evaluator_rejects_duplicate_ids_and_bad_scores() -> None:
    row = {
        "id": "same",
        "completion": "Ответ.",
        "contract": {"requested_language": "ru"},
    }
    with pytest.raises(ValueError, match="duplicate id"):
        evaluate_rows([row, row])
    with pytest.raises(ValueError, match="semantic_score"):
        evaluate_rows([{**row, "semantic_score": 1.1}])
