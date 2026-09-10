#!/usr/bin/env python3
# ruff: noqa: E501
"""Generate teacher-free GLM-5.2 Russian, Markdown, and Han training pairs.

Every target is a deterministic rendering or correction of information already
present in the prompt. No model-generated facts are introduced. Semantic
groups, rather than individual prompt variants, are assigned to splits. Held-out
groups use prompt and response renderings that are not reused by training rows.
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

DATASET_REVISION = "targeted-template-v4"
REVIEWER = "deterministic-template-audit-v2"

LIST_CASES = (
    (
        "Проверка резервной копии",
        (
            "Создать копию",
            "Восстановить её в изолированной среде",
            "Сверить контрольные суммы",
        ),
    ),
    (
        "Подготовка релиза",
        ("Запустить тесты", "Обновить журнал изменений", "Создать тег версии"),
    ),
    (
        "Разбор ошибки",
        ("Воспроизвести сбой", "Изучить журнал", "Проверить исправление"),
    ),
    (
        "Обновление зависимости",
        (
            "Прочитать список изменений",
            "Обновить lock-файл",
            "Повторно запустить тесты",
        ),
    ),
    (
        "Миграция данных",
        (
            "Создать резервную копию",
            "Выполнить пробный запуск",
            "Сверить количество записей",
        ),
    ),
    (
        "Проверка доступа",
        (
            "Собрать требования",
            "Выдать минимальные права",
            "Записать решение в журнал аудита",
        ),
    ),
    (
        "Обработка запроса",
        ("Проверить входные данные", "Выполнить операцию", "Сохранить результат"),
    ),
    (
        "Настройка мониторинга",
        ("Выбрать метрики", "Настроить предупреждения", "Собрать панель наблюдения"),
    ),
    (
        "Проверка документации",
        ("Проверить команды", "Открыть все ссылки", "Повторить примеры"),
    ),
    (
        "Очистка кэша",
        (
            "Остановить новые записи",
            "Удалить устаревшие значения",
            "Прогреть часто используемые ключи",
        ),
    ),
    (
        "Ротация ключа",
        ("Создать новый ключ", "Развернуть его в сервисах", "Отозвать старый ключ"),
    ),
    (
        "Восстановление сервиса",
        ("Изолировать причину", "Откатить изменение", "Проверить состояние сервиса"),
    ),
    (
        "Подготовка отчёта",
        ("Собрать измерения", "Проверить числа", "Сформулировать выводы"),
    ),
    (
        "Тестирование API",
        (
            "Проверить успешный запрос",
            "Передать неверные данные",
            "Смоделировать тайм-аут",
        ),
    ),
    (
        "Проверка модели",
        (
            "Сверить конфигурацию",
            "Проверить контрольную сумму весов",
            "Выполнить тестовый инференс",
        ),
    ),
    (
        "Архивация журнала",
        (
            "Закрыть текущий файл",
            "Вычислить контрольную сумму",
            "Загрузить архив в хранилище",
        ),
    ),
)

LIST_PROMPTS = (
    "Собери рабочую памятку Markdown «{title}» с заголовком. Расположи эти действия в нумерованном списке: {data}.",
    "Преобразуй последовательность {data} в раздел «{title}» с Markdown-заголовком и нумерованным списком.",
    "Для контрольной выборки напечатай заголовок «{title}», а под ним перечисли числами без перестановки: {data}.",
    "Отобрази процедуру {data} как упорядоченные пункты под Markdown-заголовком «{title}».",
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
    "Сверстай журнал процесса «{context}» таблицей Markdown. Нужны поля «Статус» и «Действие»: {data}.",
    "Разложи записи {data} по двум табличным столбцам для операции «{context}».",
    "В проверочном ответе сохрани значения {data} и представь их Markdown-таблицей со статусом и действием для «{context}».",
    "Покажи карточки этапов «{context}» в двухколоночной Markdown-таблице; исходные пары: {data}.",
)

CODE_CASES = (
    ('print("готово")', "Код выводит сообщение «готово»."),
    (
        "items = [1, 2, 3]\nprint(len(items))",
        "Код создаёт список из трёх элементов и выводит его длину.",
    ),
    (
        "result = sum([2, 3, 5])\nprint(result)",
        "Код складывает три числа и выводит результат.",
    ),
    (
        'status = "ok"\nassert status == "ok"',
        "Код сохраняет статус и проверяет, что он равен строке «ok».",
    ),
    (
        'for name in ["api", "worker"]:\n    print(name)',
        "Код по очереди выводит два имени компонентов.",
    ),
    (
        'config = {"retries": 3}\nprint(config["retries"])',
        "Код создаёт словарь настроек и выводит число повторных попыток.",
    ),
    (
        'path = "/tmp/report.txt"\nprint(path)',
        "Код сохраняет путь к отчёту и выводит его.",
    ),
    (
        "enabled = True\nprint(enabled)",
        "Код задаёт логический признак и выводит его значение.",
    ),
    (
        "values = [4, 8]\nprint(max(values))",
        "Код находит и выводит наибольшее число в списке.",
    ),
    (
        'text = "проверка"\nprint(text.upper())',
        "Код переводит строку в верхний регистр и выводит результат.",
    ),
    (
        "ports = {80, 443}\nprint(443 in ports)",
        "Код проверяет наличие порта 443 в множестве и выводит итог.",
    ),
    (
        'pairs = {"a": 1, "b": 2}\nprint(sorted(pairs))',
        "Код сортирует ключи словаря и выводит их список.",
    ),
    (
        "count = 0\ncount += 1\nprint(count)",
        "Код увеличивает счётчик на единицу и выводит новое значение.",
    ),
    (
        'message = "ошибок нет"\nprint(message)',
        "Код сохраняет и выводит сообщение об отсутствии ошибок.",
    ),
    (
        'data = ("train", "test")\nprint(data[0])',
        "Код создаёт кортеж и выводит его первый элемент.",
    ),
    (
        "ready = all([True, True])\nprint(ready)",
        "Код проверяет истинность всех значений и выводит результат.",
    ),
)

CODE_PROMPTS = (
    "Объясни действие программы по-русски и повтори её в закрытом блоке Markdown:\n{code}",
    "Разбери этот фрагмент, затем дословно помести его в блок Markdown между тройными обратными кавычками:\n{code}",
    "Для проверочного задания кратко опиши наблюдаемый результат и приложи неизменённый fenced-блок:\n{code}",
    "Сформулируй русское пояснение к листингу и воспроизведи исходник в валидном закрытом блоке Markdown:\n{code}",
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
    "Собери Markdown-карточку «{title}»: пункты {data}; выделенный вывод «{conclusion}»; адрес {url}.",
    "Создай краткую техническую заметку с заголовком «{title}», маркированным списком {data}, жирной фразой «{conclusion}» и ссылкой {url}.",
    "В контрольном формате объедини заголовок «{title}», маркированные сведения {data}, акцентированный итог «{conclusion}» и URL {url}.",
    "Представь материалы {data} как маркированный список в Markdown-разделе «{title}»; отдельно выдели «{conclusion}» и добавь переход {url}.",
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
    (
        "тесты завершены; два предупреждения; ошибок нет",
        "Тесты завершились с двумя предупреждениями, но без ошибок.",
    ),
    (
        "запрос проверен; данные корректны; результат записан",
        "Запрос проверен, данные признаны корректными, а результат записан.",
    ),
    (
        "ключ заменён; сервисы обновлены; старый ключ отозван",
        "Ключ заменён, сервисы обновлены, а старый ключ отозван.",
    ),
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
    (
        "отчёт собран; числа сверены; выводы готовы",
        "Отчёт собран, числа сверены, а выводы подготовлены.",
    ),
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
    "Отредактируй телеграфные заметки в одно естественное русское предложение: {notes}.",
    "Свяжи факты {notes} грамматически, ничего к ним не добавляя.",
    "Контрольная редактура: преврати черновик {notes} в единственную нейтральную фразу.",
    "Передай содержание записи {notes} одним гладким предложением на русском языке.",
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


def _variant_indexes(group_index: int) -> tuple[int, ...]:
    """Keep augmentation in train while giving held-out groups one fixed form."""

    split = _split(group_index)
    if split == "train":
        return (0, 1)
    if split == "validation":
        return (2,)
    return (3,)


def _split_rendering(split: str, *, train: str, validation: str, test: str) -> str:
    return {"train": train, "validation": validation, "test": test}[split]


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
        for variant in _variant_indexes(group):
            template = LIST_PROMPTS[variant]
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

    for group, context in enumerate(TABLE_CONTEXTS):
        actions = LIST_CASES[group][1]
        statuses = {
            "train": ("Готово", "Проверка", "Далее"),
            "validation": ("Завершено", "На контроле", "Запланировано"),
            "test": ("Исполнено", "Сверяется", "Предстоит"),
        }[_split(group)]
        table_rows = tuple(zip(statuses, actions, strict=True))
        table_data = "; ".join(f"{status} — {action}" for status, action in table_rows)
        table_body = "\n".join(f"| {status} | {action} |" for status, action in table_rows)
        response = f"| Статус | Действие |\n| --- | --- |\n{table_body}"
        for variant in _variant_indexes(group):
            template = TABLE_PROMPTS[variant]
            rows.append(
                _row(
                    "markdown-table",
                    group,
                    variant,
                    template.format(context=context, data=table_data),
                    response,
                    _contract(require_markdown=True, blocks=("table",)),
                    ("markdown", "table", "russian"),
                )
            )

    for group, (code, explanation) in enumerate(CODE_CASES):
        split = _split(group)
        response = _split_rendering(
            split,
            train=f"{explanation}\n\n```python\n{code}\n```",
            validation=f"Разбор контрольного фрагмента: {explanation}\n\n```python\n{code}\n```",
            test=f"Итог чтения листинга: {explanation}\n\n```python\n{code}\n```",
        )
        for variant in _variant_indexes(group):
            template = CODE_PROMPTS[variant]
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
        for variant in _variant_indexes(group):
            template = MIXED_PROMPTS[variant]
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
        for variant in _variant_indexes(group):
            template = STYLE_PROMPTS[variant]
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
        for variant in _variant_indexes(group):
            insertion = HAN_INSERTIONS[variant]
            position = 1 + (variant * 3) % (len(words) - 1)
            corrupted = " ".join([*words[:position], insertion, *words[position:]])
            prompt = _split_rendering(
                _split(group),
                train=(
                    "Удали случайную китайскую вставку из русского черновика и "
                    f"верни только исправленный текст: {corrupted}"
                ),
                validation=(
                    "Очисти контрольную русскую фразу от внедрённого иероглифического "
                    f"слова; в ответе оставь один восстановленный вариант: {corrupted}"
                ),
                test=(
                    "В предложение затесался посторонний китайский фрагмент. "
                    f"Исключи его, не редактируя остальное: {corrupted}"
                ),
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
        split = _split(group)
        code_prompt = _split_rendering(
            split,
            train=(
                "Объясни присваивание по-русски и сохрани идентификатор "
                f"дословно в закрытом блоке кода Markdown: {code}"
            ),
            validation=(
                "Для проверки области исключения опиши операцию, не переводя имя "
                f"переменной, затем приведи исходную строку в закрытом блоке Markdown: {code}"
            ),
            test=(
                "Разбери выражение, оставь иероглифический идентификатор нетронутым "
                f"и воспроизведи листинг в блоке Markdown между обратными кавычками: {code}"
            ),
        )
        code_response = _split_rendering(
            split,
            train=f"Переменной назначается числовое значение:\n\n```python\n{code}\n```",
            validation=f"Операция сохраняет исходное имя и записывает число:\n\n```python\n{code}\n```",
            test=f"В листинге указан идентификатор и присвоенное ему целое:\n\n```python\n{code}\n```",
        )
        rows.append(
            _row(
                "han-in-code",
                group,
                0,
                code_prompt,
                code_response,
                _contract(require_markdown=True, blocks=("code",)),
                ("accidental-han-control", "code", "russian"),
            )
        )
        quote_prompt = _split_rendering(
            split,
            train=f"Перепечатай слово «{value}» в цитатном блоке и поясни, что оно получено от пользователя.",
            validation=f"Составь blockquote с дословным фрагментом «{value}»; перед ним по-русски укажи происхождение текста.",
            test=f"Помести без перевода выражение «{value}» после русского замечания об исходной формулировке.",
        )
        quote_response = _split_rendering(
            split,
            train=f"Пользователь передал следующий фрагмент:\n\n> {value}",
            validation=f"Ниже приведена дословная исходная цитата:\n\n> {value}",
            test=f"Полученная формулировка воспроизведена без перевода:\n\n> {value}",
        )
        rows.append(
            _row(
                "han-in-quote",
                group,
                0,
                quote_prompt,
                quote_response,
                _contract(allow_han_in_blockquotes=True, require_markdown=True),
                ("accidental-han-control", "blockquote", "russian"),
            )
        )
        url = f"https://example.test/{value}/{group + 1}"
        link_prompt = _split_rendering(
            split,
            train=f"Размести адрес {url} в русской Markdown-ссылке, не исправляя путь.",
            validation=f"Под контрольным пояснением добавь кликабельный URL {url}; символы назначения должны сохраниться.",
            test=f"Оформи переход на {url} русской подписью и оставь исходный сетевой адрес дословным.",
        )
        link_response = _split_rendering(
            split,
            train=f"Источник доступен по [этой ссылке]({url}).",
            validation=f"Контрольный адрес: [открыть материал]({url}).",
            test=f"Переход без изменения назначения: [посмотреть страницу]({url}).",
        )
        rows.append(
            _row(
                "han-in-link",
                group,
                0,
                link_prompt,
                link_response,
                _contract(),
                ("accidental-han-control", "link", "russian"),
            )
        )

    retention_prompts = (
        "Верни следующую строку без перевода и добавлений: «{value}».",
        "Сохрани китайский пользовательский текст точно как дано: «{value}».",
        "Контроль сохранности: воспроизведи полученный фрагмент «{value}» посимвольно.",
        "Не русифицируй выражение «{value}»; ответ должен совпасть с ним целиком.",
    )
    for group, value in enumerate(HAN_SCOPES):
        for variant in _variant_indexes(group):
            template = retention_prompts[variant]
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
