#!/usr/bin/env python3
"""Deterministic GLM-5.2 Russian, Markdown, and accidental-Han constraints.

This is a bounded constraint component, not a semantic-quality reward.  SFT
data and RL rewards still need an independent relevance/quality signal so a
model cannot win by emitting short, formally valid Russian text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from markdown_it import MarkdownIt

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
LINK_DEST_RE = re.compile(r"(?<=\]\()[^)\n]+(?=\))")
FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})([^\n]*)$")
BROKEN_LINK_RE = re.compile(r"!?\[[^\]\n]+\]\([^\)\n]*$")
HEADING_WITHOUT_SPACE_RE = re.compile(r"^#{1,6}[^#\s]")
SUPPORTED_BLOCKS = frozenset({"code", "heading", "list", "table"})
MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")


@dataclass(frozen=True)
class QualityContract:
    requested_language: str = "ru"
    allow_han: bool = False
    allow_han_in_blockquotes: bool = False
    require_markdown: bool = False
    required_blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintResult:
    score: float
    nonempty_score: float
    han_score: float
    russian_script_score: float
    markdown_score: float
    han_count: int
    cyrillic_count: int
    latin_count: int
    visible_character_count: int
    markdown_defects: tuple[str, ...]


def _is_han(character: str) -> bool:
    name = unicodedata.name(character, "")
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


def _script_counts(text: str) -> tuple[int, int, int]:
    han = cyrillic = latin = 0
    for character in text:
        name = unicodedata.name(character, "")
        han += int(_is_han(character))
        cyrillic += int("CYRILLIC" in name)
        latin += int("LATIN" in name)
    return han, cyrillic, latin


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return bool(backslashes % 2)


def _find_matching_backtick_run(text: str, start: int, length: int) -> int:
    index = start
    while index < len(text):
        position = text.find("`", index)
        if position < 0:
            return -1
        end = position + 1
        while end < len(text) and text[end] == "`":
            end += 1
        if not _is_escaped(text, position) and end - position == length:
            return position
        index = end
    return -1


def _mask_urls_links_and_inline_code(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = LINK_DEST_RE.sub(" ", text)
    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "`" or _is_escaped(text, index):
            output.append(text[index])
            index += 1
            continue
        run_end = index + 1
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        delimiter = text[index:run_end]
        close = _find_matching_backtick_run(text, run_end, len(delimiter))
        if close < 0:
            output.append(delimiter)
            index = run_end
            continue
        output.append(" " * (close + len(delimiter) - index))
        index = close + len(delimiter)
    return "".join(output)


def visible_prose(text: str, *, exclude_blockquotes: bool = False) -> str:
    """Return prose used for script statistics, excluding code and destinations."""
    output: list[str] = []
    open_fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        match = FENCE_RE.match(stripped)
        if open_fence is None and match:
            marker = match.group(2)
            open_fence = (marker[0], len(marker))
            output.append("\n" if line.endswith("\n") else "")
            continue
        if open_fence is not None:
            if match:
                marker = match.group(2)
                if marker[0] == open_fence[0] and len(marker) >= open_fence[1] and not match.group(3).strip():
                    open_fence = None
            output.append("\n" if line.endswith("\n") else "")
            continue
        if exclude_blockquotes and line.lstrip().startswith(">"):
            output.append("\n" if line.endswith("\n") else "")
            continue
        output.append(_mask_urls_links_and_inline_code(line))
    return "".join(output)


def _fence_defects(text: str) -> list[str]:
    defects: list[str] = []
    open_fence: tuple[str, int, int] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group(2)
        if open_fence is None:
            open_fence = (marker[0], len(marker), line_number)
        elif marker[0] == open_fence[0] and len(marker) >= open_fence[1] and not match.group(3).strip():
            open_fence = None
    if open_fence is not None:
        defects.append(f"unclosed_fence:line={open_fence[2]}")
    return defects


def _inline_backtick_defects(text: str) -> list[str]:
    defects: list[str] = []
    for line_number, line in enumerate(visible_prose(text).splitlines(), start=1):
        runs: dict[int, int] = {}
        index = 0
        while index < len(line):
            if line[index] != "`" or _is_escaped(line, index):
                index += 1
                continue
            end = index + 1
            while end < len(line) and line[end] == "`":
                end += 1
            runs[end - index] = runs.get(end - index, 0) + 1
            index = end
        for run_length, count in runs.items():
            if count % 2:
                defects.append(f"unclosed_inline_code:line={line_number}:ticks={run_length}")
    return defects


def _strong_emphasis_defects(text: str) -> list[str]:
    defects: list[str] = []
    prose = visible_prose(text)
    for line_number, line in enumerate(prose.splitlines(), start=1):
        for delimiter, name in (("**", "asterisk"), ("__", "underscore")):
            count = 0
            start = 0
            while True:
                position = line.find(delimiter, start)
                if position < 0:
                    break
                if position == 0 or line[position - 1] != "\\":
                    count += 1
                start = position + len(delimiter)
            if count % 2:
                defects.append(f"unclosed_strong_{name}:line={line_number}")
    return defects


def markdown_defects(text: str, contract: QualityContract) -> tuple[str, ...]:
    unknown_blocks = set(contract.required_blocks) - SUPPORTED_BLOCKS
    if unknown_blocks:
        raise ValueError(f"unsupported required Markdown blocks: {sorted(unknown_blocks)}")

    defects = _fence_defects(text)
    defects.extend(_inline_backtick_defects(text))
    defects.extend(_strong_emphasis_defects(text))
    for line_number, line in enumerate(text.splitlines(), start=1):
        if BROKEN_LINK_RE.search(line):
            defects.append(f"unclosed_link:line={line_number}")
        if HEADING_WITHOUT_SPACE_RE.match(line):
            defects.append(f"heading_without_space:line={line_number}")

    try:
        tokens = MARKDOWN.parse(text)
    except Exception as error:  # pragma: no cover - parser failures are rare
        defects.append(f"parser_error:{type(error).__name__}")
        tokens = []
    token_types = {token.type for token in tokens}
    required_token_types = {
        "code": {"fence"},
        "heading": {"heading_open"},
        "list": {"bullet_list_open", "ordered_list_open"},
        "table": {"table_open"},
    }
    for required in contract.required_blocks:
        if not required_token_types[required].intersection(token_types):
            defects.append(f"missing_required_{required}")
    structural_types = set().union(*required_token_types.values()) | {"blockquote_open"}
    if contract.require_markdown and not structural_types.intersection(token_types):
        defects.append("missing_required_markdown_structure")
    return tuple(sorted(set(defects)))


def score_constraints(completion: str, contract: QualityContract) -> ConstraintResult:
    prose = visible_prose(
        completion,
        exclude_blockquotes=contract.allow_han_in_blockquotes,
    )
    han_count, cyrillic_count, latin_count = _script_counts(prose)
    defects = markdown_defects(completion, contract)

    if contract.allow_han or contract.requested_language in {"zh", "ja"}:
        han_score = 1.0
    else:
        han_score = math.exp(-float(han_count))

    if contract.requested_language == "ru":
        denominator = cyrillic_count + latin_count + han_count
        ratio = cyrillic_count / denominator if denominator else 0.0
        russian_script_score = min(1.0, max(0.0, ratio / 0.70))
    else:
        russian_script_score = 1.0

    nonempty_score = float(bool(prose.strip()))
    markdown_score = math.exp(-float(len(defects)))
    score = nonempty_score * math.sqrt(han_score * markdown_score) * (0.75 + 0.25 * russian_script_score)
    return ConstraintResult(
        score=score,
        nonempty_score=nonempty_score,
        han_score=han_score,
        russian_script_score=russian_script_score,
        markdown_score=markdown_score,
        han_count=han_count,
        cyrillic_count=cyrillic_count,
        latin_count=latin_count,
        visible_character_count=sum(not character.isspace() for character in prose),
        markdown_defects=defects,
    )


def contract_from_mapping(info: dict[str, Any] | None) -> QualityContract:
    info = info or {}
    requested_language = info.get("requested_language", "ru")
    if not isinstance(requested_language, str) or not requested_language.strip():
        raise TypeError("requested_language must be a nonempty string")
    for field in ("allow_han", "allow_han_in_blockquotes", "require_markdown"):
        if field in info and not isinstance(info[field], bool):
            raise TypeError(f"{field} must be a boolean")
    required_blocks = info.get("required_markdown_blocks", ())
    if not isinstance(required_blocks, (list, tuple)) or any(not isinstance(block, str) for block in required_blocks):
        raise TypeError("required_markdown_blocks must be a list of strings")
    return QualityContract(
        requested_language=requested_language.strip().casefold(),
        allow_han=info.get("allow_han", False),
        allow_han_in_blockquotes=info.get("allow_han_in_blockquotes", False),
        require_markdown=info.get("require_markdown", False),
        required_blocks=tuple(required_blocks),
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """VERL/Miles-compatible deterministic constraint component."""
    del ground_truth, kwargs
    if data_source != "glm52_quality":
        raise ValueError(f"unexpected data source: {data_source!r}")
    result = score_constraints(solution_str, contract_from_mapping(extra_info))
    return {
        "score": result.score,
        "constraint": result.score,
        "nonempty": result.nonempty_score,
        "no_accidental_han": result.han_score,
        "russian_script": result.russian_script_score,
        "markdown": result.markdown_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--allow-han", action="store_true")
    parser.add_argument("--allow-han-in-blockquotes", action="store_true")
    parser.add_argument("--require-markdown", action="store_true")
    parser.add_argument("--required-block", action="append", default=[])
    args = parser.parse_args()
    result = score_constraints(
        args.text,
        QualityContract(
            requested_language=args.language,
            allow_han=args.allow_han,
            allow_han_in_blockquotes=args.allow_han_in_blockquotes,
            require_markdown=args.require_markdown,
            required_blocks=tuple(args.required_block),
        ),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
