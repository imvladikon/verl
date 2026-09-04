# ruff: noqa: E402, E501

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUALITY = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(QUALITY))

from audit_split_isolation import audit_rows, source_index_from_records  # noqa: E402
from build_teacher_free_russian_corruptions import (  # noqa: E402
    cluster_source_splits,
    generate_rows,
)
from build_token_bucket_mixture import enforce_split_isolation  # noqa: E402


def source_article(source_id: str = "source-a", *, topic: str = "резервной копии") -> dict:
    sentences = [
        "Выход и продвижение 20 октября 2021 года, группа показала трек-лист для своего второго англоязычного альбома The Dreaming. 10 декабря в один день с выходом трека, вышла экранизация в виде клипа на песню.",
        "Участники группы работают в боулингом клубе, Ю Кихён подметает дорожки для метания мячей, Ли Минхёк протирает сами шары для игры в боулинг, Чё Хёнвон дезинфицирует обувь, Ли Чжухон и Им Чангюн вместе работают на диджейской стойке.",
        "В композиции присутствуют синтезаторные биты, гитарные риффы, как и клип, трек был написан в стиле эры диско 80-х годов.",
    ]
    source_text = "\n".join(sentences)
    return {
        "config": "20231101.ru",
        "dataset": "wikimedia/wikipedia",
        "id": source_id,
        "license": "cc-by-sa-3.0 OR gfdl",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "sentences": sentences,
        "text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "title": f"Проверка {topic}",
        "url": f"https://ru.wikipedia.org/wiki/{source_id}",
    }


def source_articles(*source_ids: str) -> list[dict]:
    assert len(source_ids) == 4
    return [
        source_article(source_ids[0], topic="резервной копии"),
        source_article(source_ids[1], topic="резервной копии"),
        source_article(source_ids[2], topic="тестового релиза"),
        source_article(source_ids[3], topic="тестового релиза"),
    ]


def test_duplicate_article_target_is_removed_before_split_assignment() -> None:
    original = source_article()
    duplicate = copy.deepcopy(original)
    duplicate["id"] = original["id"] + "-duplicate-content"
    duplicate["url"] = original["url"] + "?duplicate=1"

    rows = generate_rows([original, duplicate])

    assert len(rows) == 4
    assert (
        audit_rows(
            rows,
            source_index=source_index_from_records([original]),
            required_source_datasets=["wikimedia/wikipedia"],
        )["status"]
        == "PASS"
    )


def test_source_and_rendered_near_duplicates_are_clustered_atomically() -> None:
    articles = source_articles("9896182", "9896254", "5108281", "5109388")

    assignments = cluster_source_splits(articles)

    for left, right in (("9896182", "9896254"), ("5108281", "5109388")):
        assert assignments[left][1] == assignments[right][1]
        assert assignments[left][0] == assignments[right][0]


def test_mixture_gate_rejects_cross_split_target_reuse() -> None:
    original = source_article()
    rows = generate_rows([original])
    leaked = copy.deepcopy(rows[0])
    leaked["id"] += "-heldout"
    leaked["split"] = "test" if rows[0]["split"] != "test" else "train"

    try:
        enforce_split_isolation([rows[0], leaked])
    except ValueError as error:
        assert "cross-split content leakage detected" in str(error)
    else:
        raise AssertionError("content leakage was accepted")
