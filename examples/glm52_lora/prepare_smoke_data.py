#!/usr/bin/env python3
"""Create deterministic GLM-5.2 LoRA SFT and RL contract datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM = (
    "Отвечай естественным русским языком. "
    "Не добавляй китайские иероглифы. "
    "Если нужен Markdown, он должен быть синтаксически корректным."
)

SFT_EXAMPLES = (
    (
        "Исправь разметку: **важный текст\n\n```python\nprint('ok')",
        "**Важный текст**\n\n```python\nprint('ok')\n```",
    ),
    (
        "Перепиши без случайных китайских символов: "
        "Отчёт готов 的 и сохранён.",
        "Отчёт готов и сохранён.",
    ),
    (
        "Сделай фразу естественной: "
        "Я осуществил выполнение проверки результата.",
        "Я проверил результат.",
    ),
    (
        "Оформи два шага списком Markdown: "
        "проверить данные; сохранить отчёт.",
        "1. Проверить данные.\n2. Сохранить отчёт.",
    ),
    (
        "Ответь кратко по-русски: почему важно закрывать блоки кода?",
        "Закрывающий маркер отделяет код от последующего текста "
        "и сохраняет корректную разметку.",
    ),
    (
        "Удали языковой мусор: "
        "Модель вернула корректный ответ, 但是 затем испортила формат.",
        "Модель вернула корректный ответ, но затем испортила формат.",
    ),
)

RL_PROMPTS = (
    "Исправь Markdown и верни только исправленный текст: **отчёт готов",
    "Перепиши естественно по-русски без китайских символов: "
    "Проверка 完成 успешно.",
    "Оформи корректный Markdown-список из пунктов: "
    "анализ; исправление; проверка.",
    "Ответь одним русским предложением: "
    "зачем проверять результат после изменения?",
    "Исправь незакрытый блок: ```python\nprint('готово')",
    "Удали случайный символ и сохрани смысл: "
    "Результат 已 записан в журнал.",
    "Сделай фразу менее канцелярской: "
    "Было осуществлено проведение анализа ошибки.",
    "Верни корректный заголовок Markdown второго уровня "
    "со словами Итоги проверки.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "enable_thinking": False,
        }
        for prompt, answer in SFT_EXAMPLES
    ]
    rl_rows = [
        {
            "data_source": "glm52_lora_contract",
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "ability": "ru_markdown_script_hygiene",
            "reward_model": {"style": "rule", "ground_truth": "unused"},
            "extra_info": {"index": index, "smoke_tiebreaker": True},
        }
        for index, prompt in enumerate(RL_PROMPTS)
    ]

    sft_path = args.output_dir / "sft.parquet"
    rl_path = args.output_dir / "rl.parquet"
    pq.write_table(pa.Table.from_pylist(sft_rows), sft_path)
    pq.write_table(pa.Table.from_pylist(rl_rows), rl_path)
    print(f"wrote {len(sft_rows)} SFT examples to {sft_path}")
    print(f"wrote {len(rl_rows)} RL examples to {rl_path}")


if __name__ == "__main__":
    main()
