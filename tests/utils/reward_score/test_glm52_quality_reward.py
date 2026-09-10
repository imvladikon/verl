from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from quality_reward import (  # noqa: E402
    QualityContract,
    compute_score,
    score_constraints,
)


def test_clean_russian_text_gets_full_constraint_score() -> None:
    result = score_constraints(
        "Короткий и корректный ответ на русском языке.",
        QualityContract(requested_language="ru"),
    )
    assert result.score == 1.0
    assert result.han_count == 0
    assert result.markdown_defects == ()


def test_empty_text_cannot_receive_a_constraint_reward() -> None:
    result = score_constraints("  \n", QualityContract(requested_language="ru"))
    assert result.nonempty_score == 0.0
    assert result.score == 0.0


def test_accidental_han_in_russian_prose_is_penalized() -> None:
    clean = score_constraints("Это корректный русский ответ.", QualityContract(requested_language="ru"))
    contaminated = score_constraints(
        "Это русский ответ с лишним символом 中.",
        QualityContract(requested_language="ru"),
    )
    assert contaminated.han_count == 1
    assert contaminated.han_score == math.exp(-1.0)
    assert contaminated.score < clean.score


def test_han_in_code_url_and_link_destination_is_not_accidental_prose() -> None:
    text = (
        "Русское объяснение. `变量 = 1`\n\n"
        "```python\n变量 = 2\n```\n\n"
        "[Источник](relative/中文) и https://example.test/中文"
    )
    result = score_constraints(text, QualityContract(requested_language="ru"))
    assert result.han_count == 0
    assert result.han_score == 1.0
    assert result.markdown_defects == ()


def test_inline_code_requires_an_exact_backtick_delimiter_length() -> None:
    result = score_constraints(
        "Русский текст с ``кодом``` без закрытия.",
        QualityContract(requested_language="ru"),
    )
    assert "unclosed_inline_code:line=1:ticks=2" in result.markdown_defects


def test_fence_with_an_info_string_cannot_close_an_open_fence() -> None:
    result = score_constraints(
        "```text\nпример\n```python",
        QualityContract(requested_language="ru", require_markdown=True),
    )
    assert "unclosed_fence:line=1" in result.markdown_defects


def test_han_in_blockquote_requires_an_explicit_contract_flag() -> None:
    text = "Русский перевод:\n\n> 原文"
    strict = score_constraints(text, QualityContract(requested_language="ru"))
    quoted = score_constraints(
        text,
        QualityContract(requested_language="ru", allow_han_in_blockquotes=True),
    )
    assert strict.han_count == 2
    assert quoted.han_count == 0


def test_explicit_chinese_context_can_allow_han() -> None:
    result = score_constraints(
        "这是正确的中文回答。",
        QualityContract(requested_language="zh", allow_han=True),
    )
    assert result.han_count > 0
    assert result.han_score == 1.0
    assert result.score == 1.0


@pytest.mark.parametrize(
    ("text", "defect"),
    [
        ("Объяснение:\n```python\nprint('ok')", "unclosed_fence:line=2"),
        ("Это **важный текст", "unclosed_strong_asterisk:line=1"),
        ("##Заголовок", "heading_without_space:line=1"),
        ("См. [источник](relative/path", "unclosed_link:line=1"),
    ],
)
def test_product_markdown_defects_are_explicit(text: str, defect: str) -> None:
    result = score_constraints(
        text,
        QualityContract(requested_language="ru", require_markdown=True),
    )
    assert defect in result.markdown_defects
    assert result.markdown_score < 1.0


def test_required_markdown_blocks_include_tables() -> None:
    contract = QualityContract(
        requested_language="ru",
        require_markdown=True,
        required_blocks=("heading", "list", "code", "table"),
    )
    valid = score_constraints(
        "# Заголовок\n\n- пункт\n\n| А | Б |\n| - | - |\n| 1 | 2 |\n\n```text\nпример\n```",
        contract,
    )
    invalid = score_constraints("Просто русский текст.", contract)
    assert valid.markdown_defects == ()
    assert set(invalid.markdown_defects) == {
        "missing_required_code",
        "missing_required_heading",
        "missing_required_list",
        "missing_required_markdown_structure",
        "missing_required_table",
    }


def test_unknown_markdown_requirement_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported required Markdown blocks"):
        score_constraints(
            "Текст",
            QualityContract(requested_language="ru", required_blocks=("diagram",)),
        )


def test_verl_entrypoint_is_namespaced_and_finite() -> None:
    scores = compute_score(
        "glm52_quality",
        "## Ответ\n\n- корректный пункт",
        None,
        {
            "requested_language": "ru",
            "require_markdown": True,
            "required_markdown_blocks": ["heading", "list"],
        },
    )
    assert scores.keys() == {
        "score",
        "constraint",
        "nonempty",
        "no_accidental_han",
        "russian_script",
        "markdown",
    }
    assert all(math.isfinite(value) for value in scores.values())
    assert scores["score"] == scores["constraint"] == 1.0
    with pytest.raises(ValueError, match="unexpected data source"):
        compute_score("other", "Ответ", None)
