"""Deterministic rewards for the GLM-5.2 Russian/Markdown contract.

These rules validate objective properties only.  They are suitable for smoke
tests and for a narrow RL formatting stage, but they are not a substitute for
human or judge-model preference data for general Russian prose quality.
"""

from __future__ import annotations

import hashlib
import re

HAN_RE = re.compile(
    "["
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uf900-\ufaff"
    "\U00020000-\U0002fa1f"
    "]"
)
CYRILLIC_RE = re.compile("[А-Яа-яЁё]")
LETTER_RE = re.compile("[A-Za-zА-Яа-яЁё]")


def balanced_markdown(text: str) -> bool:
    fence = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else None
        if marker is None:
            continue
        if fence is None:
            fence = marker
        elif marker == fence:
            fence = None
    if fence is not None:
        return False

    without_fenced_code = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    if without_fenced_code.count("`") % 2:
        return False
    if without_fenced_code.count("**") % 2:
        return False
    if without_fenced_code.count("__") % 2:
        return False
    return all(
        not re.match(r"^#{1,6}[^#\s]", line)
        for line in without_fenced_code.splitlines()
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    del ground_truth, kwargs
    if data_source != "glm52_lora_contract":
        raise ValueError(f"unexpected data source: {data_source!r}")
    extra_info = extra_info or {}
    index = int(extra_info.get("index", -1))
    text = solution_str.strip()

    nonempty = float(3 <= len(text) <= 800)
    no_han = float(HAN_RE.search(text) is None)
    letters = LETTER_RE.findall(text)
    cyrillic_ratio = (
        len(CYRILLIC_RE.findall(text)) / len(letters) if letters else 0.0
    )
    markdown_ok = float(balanced_markdown(text))
    task_score = 0.0
    if index == 0:
        task_score = float(text.count("**") >= 2 and text.count("**") % 2 == 0)
    elif index == 1 or index == 5:
        task_score = no_han
    elif index == 2:
        list_items = sum(
            bool(re.match(r"^\s*(?:[-*+] |\d+[.)] )", line))
            for line in text.splitlines()
        )
        task_score = min(list_items / 3.0, 1.0)
    elif index == 3:
        task_score = float("провер" in text.lower())
    elif index == 4:
        task_score = float(text.count("```") == 2)
    elif index == 6:
        lowered = text.lower()
        task_score = float(
            "осуществлено" not in lowered
            and "проведение анализа" not in lowered
            and ("анализ" in lowered or "ошиб" in lowered)
        )
    elif index == 7:
        task_score = float(bool(re.fullmatch(r"##\s+Итоги проверки\.?", text)))

    score = (
        0.15 * nonempty
        + 0.25 * no_han
        + 0.20 * cyrillic_ratio
        + 0.15 * markdown_ok
        + 0.25 * task_score
    )
    if bool(extra_info.get("smoke_tiebreaker", False)):
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        score += 1e-3 * int.from_bytes(digest, "big") / float((1 << 64) - 1)
    return {
        "score": score,
        "nonempty": nonempty,
        "no_han": no_han,
        "cyrillic_ratio": cyrillic_ratio,
        "markdown_ok": markdown_ok,
        "task_score": task_score,
    }
