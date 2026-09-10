from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_quality_dataset import validate_rows  # noqa: E402
from build_teacher_free_russian_corruptions import (  # noqa: E402
    SOURCE_CONFIG,
    SOURCE_DATASET,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    article_sentences,
    generate_rows,
    split_sequence_buckets,
)

SENTENCES = [
    (
        "Русская научная традиция включает подробное описание наблюдений, "
        "проверяемые ссылки на источники и осторожное разграничение "
        "установленных фактов и предположений."
    ),
    (
        "Исследователи сопоставляют независимые свидетельства, фиксируют "
        "ограничения метода и отдельно отмечают выводы, которые требуют "
        "дальнейшей экспериментальной проверки."
    ),
    (
        "Такой порядок работы уменьшает риск случайной ошибки, облегчает "
        "повторение результата и делает обсуждение понятным для специалистов "
        "из смежных областей."
    ),
]


def _article() -> dict:
    return {
        "id": "12345",
        "url": "https://ru.wikipedia.org/wiki/Test",
        "title": "Проверяемая статья",
        "sentences": SENTENCES,
        "text_sha256": "0" * 64,
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "config": SOURCE_CONFIG,
        "license": SOURCE_LICENSE,
    }


def test_teacher_free_corruptions_are_deterministic_and_valid() -> None:
    first = generate_rows([_article()])
    second = generate_rows([_article()])
    assert first == second
    assert len(first) == 4
    assert len({row["split"] for row in first}) == 1
    assert {row["tags"][-1] for row in first} != set()
    validated = validate_rows(first)
    assert len(validated) == 4
    assert any("完成" in row["prompt"] or "错误" in row["prompt"] for row in first)
    assert all("完成" not in row["response"] and "错误" not in row["response"] for row in first)
    assert any(row["contract"]["require_markdown"] for row in first)
    assert all(row["provenance"]["source_url"].startswith("https://ru.wikipedia.org/") for row in first)
    assert all(row["provenance"]["source_text_sha256"] == "0" * 64 for row in first)
    buckets = split_sequence_buckets(validated)
    assert len(buckets["corrections"]) == 3
    assert len(buckets["markdown"]) == 1


def test_source_revision_drift_is_rejected() -> None:
    article = _article()
    article["revision"] = "main"
    try:
        generate_rows([article])
    except ValueError as error:
        assert "revision lock mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mutable source revision was accepted")


def test_malformed_source_punctuation_is_rejected() -> None:
    malformed = (
        "Автор (, род. ) подготовил подробное исследование, в котором "
        "сопоставил независимые свидетельства и аккуратно описал ограничения метода."
    )
    assert article_sentences(malformed) == []


def test_duplicate_prompt_drops_the_whole_second_article_group() -> None:
    first = _article()
    second = _article()
    second["id"] = "67890"
    second["url"] = "https://ru.wikipedia.org/wiki/Duplicate"
    rows = generate_rows([first, second])
    assert len(rows) == 4
    assert {row["provenance"]["source_record_id"].split(":", 1)[0] for row in rows} == {"12345"}
