from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from quality_reward import QualityContract, markdown_defects, score_constraints  # noqa: E402


def test_lazy_blockquote_continuation_is_inside_allowed_han_scope() -> None:
    completion = "> 引用 на китайском\nпродолжение 引用"
    result = score_constraints(
        completion,
        QualityContract(requested_language="ru", allow_han_in_blockquotes=True),
    )
    assert result.han_count == 0


def test_han_after_blockquote_is_not_hidden() -> None:
    completion = "> 引用\n\nСнаружи 引用"
    result = score_constraints(
        completion,
        QualityContract(requested_language="ru", allow_han_in_blockquotes=True),
    )
    assert result.han_count == 2


def test_code_link_and_quote_scopes_do_not_mask_visible_han() -> None:
    completion = (
        "`变量 = 1` [ссылка](https://example.test/中文)\n\n> допустимая цитата 引用\n\nНо этот иероглиф 中 видим."
    )
    result = score_constraints(
        completion,
        QualityContract(requested_language="ru", allow_han_in_blockquotes=True),
    )
    assert result.han_count == 1


def test_common_broken_markdown_patterns_fail_closed() -> None:
    contract = QualityContract(
        requested_language="ru",
        require_markdown=True,
        required_blocks=("heading", "list"),
    )
    defects = markdown_defects("#Заголовок\n\n- пункт с **обрывом", contract)
    assert "heading_without_space:line=1" in defects
    assert "missing_required_heading" in defects
    assert "unclosed_strong_asterisk:line=3" in defects


def test_valid_requested_structure_has_no_markdown_defects() -> None:
    contract = QualityContract(
        requested_language="ru",
        require_markdown=True,
        required_blocks=("heading", "list"),
    )
    assert markdown_defects("## Заголовок\n\n- **Корректный** пункт", contract) == ()
