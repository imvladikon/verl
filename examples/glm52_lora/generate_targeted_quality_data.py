#!/usr/bin/env python3
"""Generate teacher-free GLM-5.2 Russian, Markdown, and Han training pairs.

Every target is a deterministic rendering or correction of information already
present in the prompt.  No model-generated facts are introduced.  Semantic
groups, rather than individual prompt variants, are assigned to splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_quality_dataset import DEFAULT_SYSTEM, validate_rows, write_artifacts

DATASET_REVISION = "targeted-template-v1"
REVIEWER = "deterministic-template-audit-v1"

LIST_CASES = (
    (
        "Проверка резервной копии",
        ("Создать копию", "Восстановить её в изолированной среде", "Сверить контрольные суммы"),
    ),
    ("Подготовка релиза", ("Запустить тесты", "Обновить журнал изменений", "Создать тег версии")),
    ("Разбор ошибки", ("Воспроизвести сбой", "Изучить журнал", "Проверить исправление")),
    ("Обновление зависимости", ("Прочитать список изменений", "Обновить lock-файл", "Повторно запустить тесты")),
    ("Миграция данных", ("Создать резервную копию", "Выполнить пробный запуск", "Сверить количество записей")),
    ("Проверка доступа", ("Собрать требования", "Выдать минимальные права", "Записать решение в журнал аудита")),
    ("Обработка запроса", ("Проверить входные данные", "Выполнить операцию", "Сохранить результат")),
    ("Настройка мониторинга", ("Выбрать метрики", "Настроить предупреждения", "Собрать панель наблюдения")),
    ("Проверка документации", ("Проверить команды", "Открыть все ссылки", "Повторить примеры")),
    ("Очистка кэша", ("Остановить новые записи", "Удалить устаревшие значения", "Прогреть часто используемые ключи")),
    ("Ротация ключа", ("Создать новый ключ", "Развернуть его в сервисах", "Отозвать старый ключ")),
    ("Восстановление сервиса", ("Изолировать причину", "Откатить изменение", "Проверить состояние сервиса")),
    ("Подготовка отчёта", ("Собрать измерения", "Проверить числа", "Сформулировать выводы")),
    ("Тестирование API", ("Проверить успешный запрос", "Передать неверные данные", "Смоделировать тайм-аут")),
    ("Проверка модели", ("Сверить конфигурацию", "Проверить контрольную сумму весов", "Выполнить тестовый инференс")),
    ("Архивация журнала", ("Закрыть текущий файл", "Вычислить контрольную сумму", "Загрузить архив в хранилище")),
)

LIST_PROMPTS = (
    "Оформи данные в Markdown: заголовок «{title}» и нумерованный список в заданном порядке. Данные: {data}.",
    "Верни корректный Markdown с заголовком «{title}». Ниже нужен нумерованный список без перестановки: {data}.",
    "Структурируй сведения под заголовком «{title}» как нумерованный Markdown-список: {data}.",
    "Сохрани порядок и оформи этапы в Markdown с заголовком «{title}»: {data}.",
    "Подготовь раздел Markdown «{title}» и перечисли шаги числами: {data}.",
    "Преобразуй последовательность в раздел Markdown с заголовком «{title}» и нумерацией: {data}.",
    "Напиши только корректно оформленный раздел «{title}» с нумерованными пунктами: {data}.",
    "Сделай заголовок Markdown «{title}», затем передай эти пункты по порядку: {data}.",
)

TABLE_CONTEXTS = (
    "проверка резервной копии",
    "подготовка релиза",
    "миграция данных",
    "обновление зависимости",
    "проверка API",
    "восстановление сервиса",
    "ротация ключа",
    "архивация журнала",
    "проверка документации",
    "настройка мониторинга",
    "обработка запроса",
    "проверка модели",
    "очистка кэша",
    "разбор ошибки",
    "подготовка отчёта",
    "проверка доступа",
)

TABLE_PROMPTS = (
    "Оформи сведения о процессе «{context}» как Markdown-таблицу с колонками «Статус» и «Действие»: {data}.",
    "Верни корректную Markdown-таблицу для процесса «{context}». Колонки: «Статус», «Действие». Строки: {data}.",
    "Преобразуй данные процесса «{context}» в таблицу Markdown: {data}.",
    "Составь Markdown-таблицу статусов процесса «{context}», не меняя значения: {data}.",
    "Покажи данные «{context}» в валидной Markdown-таблице из двух колонок: {data}.",
    "Структурируй записи процесса «{context}» как таблицу Markdown со статусом и действием: {data}.",
    "Напиши только Markdown-таблицу для «{context}» по следующим строкам: {data}.",
    "Сохрани текст ячеек и оформи «{context}» таблицей Markdown: {data}.",
)

CODE_CASES = (
    ('print("готово")', "Код выводит сообщение «готово»."),
    ("items = [1, 2, 3]\nprint(len(items))", "Код создаёт список из трёх элементов и выводит его длину."),
    ("result = sum([2, 3, 5])\nprint(result)", "Код складывает три числа и выводит результат."),
    ('status = "ok"\nassert status == "ok"', "Код сохраняет статус и проверяет, что он равен строке «ok»."),
    ('for name in ["api", "worker"]:\n    print(name)', "Код по очереди выводит два имени компонентов."),
    (
        'config = {"retries": 3}\nprint(config["retries"])',
        "Код создаёт словарь настроек и выводит число повторных попыток.",
    ),
    ('path = "/tmp/report.txt"\nprint(path)', "Код сохраняет путь к отчёту и выводит его."),
    ("enabled = True\nprint(enabled)", "Код задаёт логический признак и выводит его значение."),
    ("values = [4, 8]\nprint(max(values))", "Код находит и выводит наибольшее число в списке."),
    ('text = "проверка"\nprint(text.upper())', "Код переводит строку в верхний регистр и выводит результат."),
    ("ports = {80, 443}\nprint(443 in ports)", "Код проверяет наличие порта 443 в множестве и выводит итог."),
    ('pairs = {"a": 1, "b": 2}\nprint(sorted(pairs))', "Код сортирует ключи словаря и выводит их список."),
    ("count = 0\ncount += 1\nprint(count)", "Код увеличивает счётчик на единицу и выводит новое значение."),
    ('message = "ошибок нет"\nprint(message)', "Код сохраняет и выводит сообщение об отсутствии ошибок."),
    ('data = ("train", "test")\nprint(data[0])', "Код создаёт кортеж и выводит его первый элемент."),
    ("ready = all([True, True])\nprint(ready)", "Код проверяет истинность всех значений и выводит результат."),
)

CODE_PROMPTS = (
    "Объясни по-русски действие кода и повтори его в закрытом блоке Markdown:\n{code}",
    "Дай краткое русское объяснение, затем верни исходный код в корректном fenced-блоке Markdown:\n{code}",
    "Сначала опиши результат по-русски, после чего оформи этот код закрытым блоком Markdown:\n{code}",
    "Не меняй код. Напиши русское пояснение и помести код в валидный Markdown-блок:\n{code}",
)

MIXED_CASES = tuple(
    (
        title,
        items,
        conclusion,
        f"https://example.test/docs/{index}",
    )
    for index, (title, items) in enumerate(LIST_CASES, start=1)
    for conclusion in (f"Для раздела «{title}» важен проверяемый результат",)
)

MIXED_PROMPTS = (
    "Оформи Markdown-карточку «{title}»: список {data}, жирный вывод «{conclusion}», ссылка {url}.",
    "Верни Markdown для «{title}»: bullets {data}; жирный вывод «{conclusion}»; ссылка {url}.",
    "Собери раздел «{title}»: пункты {data}, жирный вывод «{conclusion}» и ссылка {url}.",
    "Напиши Markdown-карточку «{title}»: список {data}, жирный вывод «{conclusion}», документация {url}.",
)

STYLE_CASES = (
    (
        "сервис перезапущен; ошибки не повторились; журнал сохранён",
        "Сервис перезапущен, ошибки больше не повторяются, а журнал сохранён.",
    ),
    (
        "копия создана; восстановление проверено; суммы совпали",
        "Резервная копия создана, восстановление проверено, а контрольные суммы совпали.",
    ),
    ("тесты завершены; два предупреждения; ошибок нет", "Тесты завершились с двумя предупреждениями, но без ошибок."),
    (
        "запрос проверен; данные корректны; результат записан",
        "Запрос проверен, данные признаны корректными, а результат записан.",
    ),
    ("ключ заменён; сервисы обновлены; старый ключ отозван", "Ключ заменён, сервисы обновлены, а старый ключ отозван."),
    (
        "миграция выполнена; число записей прежнее; потерь нет",
        "Миграция выполнена без потерь, и количество записей не изменилось.",
    ),
    (
        "документация обновлена; команды проверены; ссылки открываются",
        "Документация обновлена: команды проверены, а ссылки открываются.",
    ),
    (
        "метрики поступают; предупреждения настроены; панель доступна",
        "Метрики поступают, предупреждения настроены, а панель доступна.",
    ),
    (
        "изменение отменено; сервис отвечает; очередь обработана",
        "Изменение отменено, сервис снова отвечает, а очередь обработана.",
    ),
    (
        "архив создан; сумма вычислена; загрузка завершена",
        "Архив создан, контрольная сумма вычислена, а загрузка завершена.",
    ),
    (
        "конфигурация сверена; веса проверены; инференс успешен",
        "Конфигурация и веса проверены, а тестовый инференс завершился успешно.",
    ),
    (
        "доступ выдан; права минимальны; решение записано",
        "Доступ выдан с минимальными правами, а решение записано в журнал аудита.",
    ),
    (
        "кэш очищен; ключи прогреты; задержка нормальная",
        "Кэш очищен, ключи прогреты, а задержка вернулась к нормальному уровню.",
    ),
    ("отчёт собран; числа сверены; выводы готовы", "Отчёт собран, числа сверены, а выводы подготовлены."),
    (
        "API отвечает; неверный ввод отклонён; тайм-аут обработан",
        "API отвечает, неверный ввод отклоняется, а тайм-аут обрабатывается корректно.",
    ),
    (
        "зависимость обновлена; lock-файл изменён; тесты прошли",
        "Зависимость и lock-файл обновлены, после чего тесты успешно прошли.",
    ),
)

STYLE_PROMPTS = (
    "Перепиши заметки одним естественным русским предложением без добавления фактов: {notes}.",
    "Сделай из заметок грамотное русское предложение, сохранив все факты: {notes}.",
    "Убери телеграфный стиль и верни одно связное предложение по-русски: {notes}.",
    "Сформулируй данные естественно и кратко на русском языке: {notes}.",
    "Объедини заметки в одно ясное русское предложение: {notes}.",
    "Напиши нейтральную русскую фразу, не меняя содержания заметок: {notes}.",
    "Преобразуй черновые пункты в грамматически связное русское предложение: {notes}.",
    "Верни только отредактированное русское предложение по этим заметкам: {notes}.",
)

CLEAN_SENTENCES = tuple(target for _, target in STYLE_CASES)
HAN_INSERTIONS = ("完成", "错误", "检查", "数据", "结果", "成功", "用户", "系统")
HAN_SCOPES = (
    "变量",
    "结果",
    "状态",
    "用户",
    "配置",
    "检查",
    "数据",
    "服务",
    "完成",
    "错误",
    "成功",
    "文件",
    "路径",
    "消息",
    "请求",
    "响应",
)


def _split(group_index: int) -> str:
    remainder = group_index % 10
    if remainder == 0:
        return "test"
    if remainder == 1:
        return "validation"
    return "train"


def _contract(
    *,
    language: str = "ru",
    allow_han: bool = False,
    allow_han_in_blockquotes: bool = False,
    require_markdown: bool = False,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "requested_language": language,
        "allow_han": allow_han,
        "allow_han_in_blockquotes": allow_han_in_blockquotes,
        "require_markdown": require_markdown,
        "required_markdown_blocks": list(blocks),
    }


def _row(
    family: str,
    group_index: int,
    variant_index: int,
    prompt: str,
    response: str,
    contract: dict[str, Any],
    tags: Iterable[str],
) -> dict[str, Any]:
    example_id = f"targeted-{family}-{group_index:02d}-{variant_index:02d}"
    return {
        "id": example_id,
        "split": _split(group_index),
        "prompt": prompt,
        "response": response,
        "system": DEFAULT_SYSTEM,
        "contract": contract,
        "tags": sorted({"targeted", family, *tags}),
        "use_for_constraint_rl_smoke": False,
        "review": {
            "status": "accepted",
            "reviewer": REVIEWER,
            "method": "deterministic-template-audit",
            "notes": "target is an exact rendering or correction of prompt-supplied data",
        },
        "provenance": {
            "dataset": "project-authored/glm52-targeted-quality",
            "revision": DATASET_REVISION,
            "license": "apache-2.0",
            "source_split": "generated",
            "source_record_id": example_id,
        },
    }


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{index}. {item}." for index, item in enumerate(items, start=1))


def _bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}." for item in items)


def generate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, (title, items) in enumerate(LIST_CASES):
        data = "; ".join(items)
        response = f"## {title}\n\n{_numbered(items)}"
        for variant, template in enumerate(LIST_PROMPTS):
            rows.append(
                _row(
                    "markdown-list",
                    group,
                    variant,
                    template.format(title=title, data=data),
                    response,
                    _contract(require_markdown=True, blocks=("heading", "list")),
                    ("markdown", "list", "russian"),
                )
            )

    table_rows = (("Готово", "Сохранить результат"), ("Ошибка", "Проверить журнал"), ("Ожидание", "Повторить проверку"))
    table_data = "; ".join(f"{status} — {action}" for status, action in table_rows)
    table_body = "\n".join(f"| {status} | {action} |" for status, action in table_rows)
    for group, context in enumerate(TABLE_CONTEXTS):
        response = f"## {context.capitalize()}\n\n| Статус | Действие |\n| --- | --- |\n{table_body}"
        for variant, template in enumerate(TABLE_PROMPTS):
            rows.append(
                _row(
                    "markdown-table",
                    group,
                    variant,
                    template.format(context=context, data=table_data),
                    response,
                    _contract(require_markdown=True, blocks=("heading", "table")),
                    ("markdown", "table", "russian"),
                )
            )

    for group, (code, explanation) in enumerate(CODE_CASES):
        response = f"{explanation}\n\n```python\n{code}\n```"
        for variant, template in enumerate(CODE_PROMPTS):
            rows.append(
                _row(
                    "markdown-code",
                    group,
                    variant,
                    template.format(code=code),
                    response,
                    _contract(require_markdown=True, blocks=("code",)),
                    ("code", "markdown", "russian"),
                )
            )

    for group, (title, items, conclusion, url) in enumerate(MIXED_CASES):
        data = "; ".join(items)
        response = f"## {title}\n\n{_bullets(items)}\n\n**Ключевой вывод:** {conclusion}.\n\n[Документация]({url})"
        for variant, template in enumerate(MIXED_PROMPTS):
            rows.append(
                _row(
                    "markdown-mixed",
                    group,
                    variant,
                    template.format(
                        title=title,
                        data=data,
                        conclusion=conclusion,
                        url=url,
                    ),
                    response,
                    _contract(require_markdown=True, blocks=("heading", "list")),
                    ("link", "markdown", "russian", "strong"),
                )
            )

    for group, (notes, response) in enumerate(STYLE_CASES):
        for variant, template in enumerate(STYLE_PROMPTS):
            rows.append(
                _row(
                    "russian-style",
                    group,
                    variant,
                    template.format(notes=notes),
                    response,
                    _contract(),
                    ("russian", "style"),
                )
            )

    for group, clean in enumerate(CLEAN_SENTENCES):
        words = clean.split()
        for variant, insertion in enumerate(HAN_INSERTIONS):
            position = 1 + (variant * 3) % (len(words) - 1)
            corrupted = " ".join([*words[:position], insertion, *words[position:]])
            prompt = (
                "Удали случайную китайскую вставку из русского черновика и "
                f"верни только исправленный текст: {corrupted}"
            )
            rows.append(
                _row(
                    "han-cleanup",
                    group,
                    variant,
                    prompt,
                    clean,
                    _contract(),
                    ("accidental-han", "russian"),
                )
            )

    for group, value in enumerate(HAN_SCOPES):
        code = f"{value}_{group + 1} = {group + 1}"
        rows.append(
            _row(
                "han-in-code",
                group,
                0,
                f"Объясни по-русски присваивание и сохрани идентификатор без изменений: {code}",
                f"Код присваивает идентификатору числовое значение:\n\n```python\n{code}\n```",
                _contract(require_markdown=True, blocks=("code",)),
                ("accidental-han-control", "code", "russian"),
            )
        )
        rows.append(
            _row(
                "han-in-quote",
                group,
                0,
                f"Процитируй без изменений слово «{value}» в blockquote и поясни по-русски, что это исходный текст.",
                f"Ниже сохранён исходный текст без изменений:\n\n> {value}",
                _contract(allow_han_in_blockquotes=True, require_markdown=True),
                ("accidental-han-control", "blockquote", "russian"),
            )
        )
        url = f"https://example.test/{value}/{group + 1}"
        rows.append(
            _row(
                "han-in-link",
                group,
                0,
                f"Добавь ссылку {url} под русским пояснением, не меняя адрес.",
                f"Адрес сохранён без изменений: [открыть источник]({url}).",
                _contract(),
                ("accidental-han-control", "link", "russian"),
            )
        )

    retention_prompts = (
        "Верни следующую строку без перевода и добавлений: «{value}».",
        "Сохрани китайский пользовательский текст точно как дано: «{value}».",
    )
    for group, value in enumerate(HAN_SCOPES):
        for variant, template in enumerate(retention_prompts):
            rows.append(
                _row(
                    "han-retention",
                    group,
                    variant,
                    template.format(value=value),
                    value,
                    _contract(language="zh", allow_han=True),
                    ("chinese-retention",),
                )
            )
    return rows


def write_targeted_dataset(output_dir: Path) -> dict[str, Any]:
    started_at = time.monotonic()
    rows = validate_rows(generate_rows())
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "targeted_quality.jsonl"
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    artifact_manifest = write_artifacts(rows, output_dir / "artifacts")
    artifact_manifest_path = output_dir / "artifacts" / "manifest.json"
    family_counts = Counter(
        tag for row in rows for tag in row["tags"] if tag.startswith(("markdown-", "russian-", "han-"))
    )
    block_counts = Counter(block for row in rows for block in row["contract"]["required_markdown_blocks"])
    manifest = {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "reviewer": REVIEWER,
        "review_method": "deterministic-template-audit",
        "row_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "family_counts": dict(sorted(family_counts.items())),
        "markdown_required_count": sum(row["contract"]["require_markdown"] for row in rows),
        "required_block_counts": dict(sorted(block_counts.items())),
        "russian_count": sum(row["contract"]["requested_language"] == "ru" for row in rows),
        "han_cleanup_count": sum("accidental-han" in row["tags"] for row in rows),
        "han_scope_control_count": sum("accidental-han-control" in row["tags"] for row in rows),
        "chinese_retention_count": sum("chinese-retention" in row["tags"] for row in rows),
        "dataset_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_counts": artifact_manifest["counts"],
        "artifact_eval_count": artifact_manifest["eval_count"],
        "artifact_manifest_sha256": hashlib.sha256(artifact_manifest_path.read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic() - started_at,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "warning": "targets use only prompt-supplied data; evaluate the full model before deployment",
    }
    (output_dir / "targeted_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            write_targeted_dataset(args.output_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
