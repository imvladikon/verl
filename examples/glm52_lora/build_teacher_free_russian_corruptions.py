#!/usr/bin/env python3
"""Build teacher-free Russian correction and Markdown rows from a locked sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from build_quality_dataset import (
    DEFAULT_SYSTEM,
    prompt_digest,
    validate_rows,
    write_artifacts,
)

SOURCE_DATASET = "wikimedia/wikipedia"
SOURCE_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SOURCE_CONFIG = "20231101.ru"
SOURCE_LICENSE = "cc-by-sa-3.0 OR gfdl"
DATASET_REVISION = "wikipedia-corruption-v3"
REVIEWER = "deterministic-corruption-audit-v2"
CC_BY_SA_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
GFDL_URL = "https://www.gnu.org/licenses/fdl-1.3.html"

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁ«])")
WHITESPACE_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
MARKDOWN_SPECIAL_RE = re.compile(r"[*_\[\]#|<>]")
MALFORMED_PUNCTUATION_RE = re.compile(r"\(\s*[,;:]|[,;:]\s*\)|\(\s*\)")
LATIN_CONFUSABLES = {
    "А": "A",
    "а": "a",
    "Е": "E",
    "е": "e",
    "О": "O",
    "о": "o",
    "Р": "P",
    "р": "p",
    "С": "C",
    "с": "c",
    "Х": "X",
    "х": "x",
    "У": "Y",
    "у": "y",
}
HAN_INSERTIONS = ("完成", "错误", "检查", "数据", "结果", "成功", "用户", "系统")
SPLIT_NEAR_DUPLICATE_THRESHOLD = 0.7
SPLIT_SHINGLE_WIDTH = 5
SPLIT_MIN_NEAR_TOKENS = 5
SPLIT_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPLIT_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class _SplitFingerprint:
    shingles: frozenset[str]
    tokens: tuple[str, ...]
    token_counts: Counter[str]


@dataclass(frozen=True)
class _SourceGroupFingerprint:
    source_fragments: tuple[_SplitFingerprint, ...]
    prompts: tuple[_SplitFingerprint, ...]
    responses: tuple[_SplitFingerprint, ...]


def normalize_target(value: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def article_sentences(text: str) -> list[str]:
    normalized = WHITESPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()
    candidates = SENTENCE_BOUNDARY_RE.split(normalized)
    accepted: list[str] = []
    for sentence in candidates:
        sentence = sentence.strip(" \t\n")
        alpha_count = len(ALPHA_RE.findall(sentence))
        cyrillic_count = len(CYRILLIC_RE.findall(sentence))
        if not 100 <= len(sentence) <= 360:
            continue
        if not sentence.endswith("."):
            continue
        if alpha_count < 70 or cyrillic_count / alpha_count < 0.85:
            continue
        if HAN_RE.search(sentence) or MARKDOWN_SPECIAL_RE.search(sentence):
            continue
        if "http://" in sentence or "https://" in sentence:
            continue
        if sentence.count("(") != sentence.count(")"):
            continue
        if sentence.count("«") != sentence.count("»"):
            continue
        if MALFORMED_PUNCTUATION_RE.search(sentence):
            continue
        accepted.append(sentence)
    return accepted


def _split(source_id: str) -> str:
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _normalize_split_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = "".join(character for character in normalized if unicodedata.category(character) != "Cf")
    return WHITESPACE_RE.sub(" ", normalized).strip().casefold()


def _split_fingerprint(value: str) -> _SplitFingerprint:
    normalized = _normalize_split_text(value)
    shingle_tokens = SPLIT_TOKEN_RE.findall(normalized)
    if len(shingle_tokens) < SPLIT_SHINGLE_WIDTH:
        shingles = frozenset(shingle_tokens)
    else:
        shingles = frozenset(
            " ".join(shingle_tokens[index : index + SPLIT_SHINGLE_WIDTH])
            for index in range(len(shingle_tokens) - SPLIT_SHINGLE_WIDTH + 1)
        )
    tokens = tuple(SPLIT_WORD_RE.findall(normalized))
    return _SplitFingerprint(
        shingles=shingles,
        tokens=tokens,
        token_counts=Counter(tokens),
    )


def _split_near_duplicate(left: _SplitFingerprint, right: _SplitFingerprint) -> bool:
    if not left.shingles or not right.shingles:
        return False
    shared = len(left.shingles & right.shingles)
    jaccard = shared / (len(left.shingles) + len(right.shingles) - shared)
    containment = shared / min(len(left.shingles), len(right.shingles))
    token_containment = 0.0
    if min(len(left.tokens), len(right.tokens)) >= SPLIT_MIN_NEAR_TOKENS:
        shared_tokens = sum(min(count, right.token_counts.get(token, 0)) for token, count in left.token_counts.items())
        token_containment = shared_tokens / min(len(left.tokens), len(right.tokens))
    return max(jaccard, containment, token_containment) >= SPLIT_NEAR_DUPLICATE_THRESHOLD


def cluster_source_splits(
    articles: Iterable[dict[str, Any]],
) -> dict[str, tuple[str, str, int]]:
    """Assign every near-duplicate source component to one deterministic split."""

    materialized = list(articles)
    source_ids: list[str] = []
    fingerprints: list[_SourceGroupFingerprint] = []
    for article in materialized:
        title, sentences = _validate_source_article(article)
        source_id = str(article["id"])
        if source_id in source_ids:
            raise ValueError(f"duplicate source article ID: {source_id}")
        source_ids.append(source_id)
        content = "\n".join((title, *sentences))
        payloads = _article_payloads(title, sentences, source_id)
        fingerprints.append(
            _SourceGroupFingerprint(
                source_fragments=tuple(_split_fingerprint(value) for value in (content, title, *sentences)),
                prompts=tuple(_split_fingerprint(prompt) for _, prompt, _, _ in payloads),
                responses=tuple(_split_fingerprint(response) for _, _, response, _ in payloads),
            )
        )

    parents = list(range(len(materialized)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(materialized)):
        for right in range(left):
            left_group = fingerprints[left]
            right_group = fingerprints[right]
            if (
                any(
                    _split_near_duplicate(left_value, right_value)
                    for left_value in left_group.source_fragments
                    for right_value in right_group.source_fragments
                )
                or any(
                    _split_near_duplicate(left_value, right_value)
                    for left_value in left_group.prompts
                    for right_value in right_group.prompts
                )
                or any(
                    _split_near_duplicate(left_value, right_value)
                    for left_value in left_group.responses
                    for right_value in right_group.responses
                )
            ):
                union(left, right)

    components: dict[int, list[str]] = {}
    for index, source_id in enumerate(source_ids):
        components.setdefault(find(index), []).append(source_id)

    assignments: dict[str, tuple[str, str, int]] = {}
    for members in components.values():
        ordered = sorted(members)
        representative = min(ordered, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        cluster_id = hashlib.sha256("\0".join(ordered).encode()).hexdigest()[:16]
        assignment = (_split(representative), cluster_id, len(ordered))
        assignments.update({source_id: assignment for source_id in ordered})
    return assignments


def _contract(*, markdown: bool = False) -> dict[str, Any]:
    return {
        "requested_language": "ru",
        "allow_han": False,
        "allow_han_in_blockquotes": False,
        "require_markdown": markdown,
        "required_markdown_blocks": ["heading", "list"] if markdown else [],
    }


def _row(
    article: dict[str, Any],
    family: str,
    prompt: str,
    response: str,
    *,
    split: str,
    markdown: bool = False,
) -> dict[str, Any]:
    source_id = str(article["id"])
    short_id = hashlib.sha256(source_id.encode()).hexdigest()[:16]
    return {
        "id": f"wikipedia-ru-{short_id}-{family}",
        "split": split,
        "prompt": prompt,
        "response": response,
        "system": DEFAULT_SYSTEM,
        "contract": _contract(markdown=markdown),
        "tags": sorted({"teacher-free", "russian", family}),
        "use_for_constraint_rl_smoke": False,
        "review": {
            "status": "accepted",
            "reviewer": REVIEWER,
            "method": "deterministic-corruption-audit",
            "notes": "target is source text copied exactly after a deterministic corruption or rendering",
        },
        "provenance": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "source_split": SOURCE_CONFIG,
            "source_record_id": f"{source_id}:{family}",
            "source_url": str(article["url"]),
            "source_title": str(article["title"]),
            "source_text_sha256": str(article["text_sha256"]),
        },
    }


def _inject_han(sentence: str, source_id: str) -> str:
    words = sentence.split()
    digest = hashlib.sha256(f"{source_id}:han".encode()).digest()
    position = 1 + int.from_bytes(digest[:2], "big") % (len(words) - 1)
    insertion = HAN_INSERTIONS[digest[2] % len(HAN_INSERTIONS)]
    return " ".join([*words[:position], insertion, *words[position:]])


def _inject_latin_confusables(sentence: str, source_id: str) -> str:
    eligible = [index for index, character in enumerate(sentence) if character in LATIN_CONFUSABLES]
    if len(eligible) < 2:
        raise ValueError("sentence has fewer than two Cyrillic/Latin-confusable letters")
    digest = hashlib.sha256(f"{source_id}:latin".encode()).digest()
    first = eligible[int.from_bytes(digest[:2], "big") % len(eligible)]
    remaining = [index for index in eligible if index != first]
    second = remaining[int.from_bytes(digest[2:4], "big") % len(remaining)]
    characters = list(sentence)
    for index in sorted((first, second)):
        characters[index] = LATIN_CONFUSABLES[characters[index]]
    return "".join(characters)


def _damage_case_and_period(sentence: str) -> str:
    match = re.search(r"[А-ЯЁ]", sentence)
    if match is None:
        raise ValueError("sentence has no uppercase Cyrillic start")
    characters = list(sentence[:-1])
    characters[match.start()] = characters[match.start()].lower()
    return "".join(characters)


def _validate_source_article(article: dict[str, Any]) -> tuple[str, list[str]]:
    for field in ("id", "url", "title", "sentences", "text_sha256"):
        if field not in article:
            raise ValueError(f"source article is missing {field}")
    if article.get("dataset") != SOURCE_DATASET:
        raise ValueError("source dataset lock mismatch")
    if article.get("revision") != SOURCE_REVISION:
        raise ValueError("source revision lock mismatch")
    if article.get("config") != SOURCE_CONFIG:
        raise ValueError("source config lock mismatch")
    if article.get("license") != SOURCE_LICENSE:
        raise ValueError("source license lock mismatch")
    title = WHITESPACE_RE.sub(" ", str(article["title"])).strip()
    if not 3 <= len(title) <= 100 or MARKDOWN_SPECIAL_RE.search(title):
        raise ValueError("source title is unsafe for deterministic Markdown")
    source_url = str(article["url"])
    if not source_url.startswith("https://ru.wikipedia.org/"):
        raise ValueError("source URL is not a Russian Wikipedia HTTPS URL")
    text_sha256 = str(article["text_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", text_sha256) is None:
        raise ValueError("source text SHA-256 is invalid")
    sentences = article["sentences"]
    if not isinstance(sentences, list) or len(sentences) < 3:
        raise ValueError("source article needs at least three extracted sentences")
    normalized_sentences = [WHITESPACE_RE.sub(" ", str(sentence)).strip() for sentence in sentences[:3]]
    if any(article_sentences(sentence) != [sentence] for sentence in normalized_sentences):
        raise ValueError("source sentence no longer passes the locked filter")
    return title, normalized_sentences


def _article_payloads(title: str, sentences: list[str], source_id: str) -> list[tuple[str, str, str, bool]]:
    clean = sentences[0]
    fragments = "\n".join(f"Фрагмент {index}: {sentence}" for index, sentence in enumerate(sentences, 1))
    markdown_response = f"## {title}\n\n" + "\n".join(f"- {sentence}" for sentence in sentences)
    return [
        (
            "han-cleanup",
            "Удали случайную китайскую вставку и верни только исправленный русский текст: "
            + _inject_han(clean, source_id),
            clean,
            False,
        ),
        (
            "russian-latin-confusable-cleanup",
            "Исправь две случайные латинские буквы-двойники и верни только русский текст: "
            + _inject_latin_confusables(sentences[1], source_id),
            sentences[1],
            False,
        ),
        (
            "russian-case-period-restoration",
            "Восстанови прописную первую букву и конечную точку. Верни только исправленный текст: "
            + _damage_case_and_period(sentences[2]),
            sentences[2],
            False,
        ),
        (
            "markdown-list",
            "Оформи заголовок и маркированный список, не сокращая и не изменяя текст фрагментов.\n"
            f"Заголовок: {title}\n{fragments}",
            markdown_response,
            True,
        ),
    ]


def generate_rows(
    articles: Iterable[dict[str, Any]],
    *,
    split_assignments: dict[str, tuple[str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    materialized = list(articles)
    assignments = split_assignments or cluster_source_splits(materialized)
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    for article in materialized:
        title, sentences = _validate_source_article(article)
        source_id = str(article["id"])
        split, _, _ = assignments[source_id]
        group = [
            _row(
                article,
                family,
                prompt,
                response,
                split=split,
                markdown=markdown,
            )
            for family, prompt, response, markdown in _article_payloads(title, sentences, source_id)
        ]
        group_prompts = {prompt_digest(row["prompt"]) for row in group}
        group_targets = {normalize_target(row["response"]) for row in group}
        if len(group_prompts) != len(group) or group_prompts.intersection(seen_prompts):
            continue
        if len(group_targets) != len(group) or group_targets.intersection(seen_targets):
            continue
        seen_prompts.update(group_prompts)
        seen_targets.update(group_targets)
        rows.extend(group)
    return rows


def read_articles(path: Path) -> list[dict[str, Any]]:
    articles = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not articles:
        raise ValueError(f"empty source sample: {path}")
    return articles


def split_sequence_buckets(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "corrections": [row for row in rows if not row["contract"]["require_markdown"]],
        "markdown": [row for row in rows if row["contract"]["require_markdown"]],
    }
    if any(not bucket_rows for bucket_rows in buckets.values()):
        raise ValueError("both correction and Markdown sequence buckets must be nonempty")
    if sum(len(bucket_rows) for bucket_rows in buckets.values()) != len(rows):
        raise AssertionError("sequence bucket partition lost rows")
    return buckets


def write_sequence_buckets(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, bucket_rows in split_sequence_buckets(rows).items():
        bucket_dir = output_dir / name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        rows_path = bucket_dir / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in bucket_rows))
        bucket_manifest = write_artifacts(bucket_rows, bucket_dir)
        bucket_manifest.update(
            {
                "bucket": name,
                "rows_sha256": sha256(rows_path),
                "token_audit_required": True,
            }
        )
        (bucket_dir / "manifest.json").write_text(
            json.dumps(bucket_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        result[name] = {
            "rows": len(bucket_rows),
            "counts": bucket_manifest["counts"],
            "rows_sha256": bucket_manifest["rows_sha256"],
        }
    return result


def write_attribution(articles: list[dict[str, Any]], rows: list[dict[str, Any]], output_dir: Path) -> Path:
    accepted_ids = {row["provenance"]["source_record_id"].split(":", 1)[0] for row in rows}
    attribution_path = output_dir / "ATTRIBUTION.jsonl"
    attribution_path.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": SOURCE_DATASET,
                    "revision": SOURCE_REVISION,
                    "config": SOURCE_CONFIG,
                    "license": SOURCE_LICENSE,
                    "source_record_id": str(article["id"]),
                    "source_title": str(article["title"]),
                    "source_url": str(article["url"]),
                    "source_text_sha256": str(article["text_sha256"]),
                    "changes": "selected sentences and deterministic corruption/rendering prompts",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for article in articles
            if str(article["id"]) in accepted_ids
        )
    )
    notice_path = output_dir / "NOTICE.md"
    notice_path.write_text(
        "# Source and license notice\n\n"
        "This artifact contains modified excerpts from Russian Wikipedia, pinned to "
        f"`{SOURCE_DATASET}@{SOURCE_REVISION}` (`{SOURCE_CONFIG}`). "
        "Per-record titles, URLs, and source-text hashes are in `ATTRIBUTION.jsonl`.\n\n"
        "Changes are limited to sentence selection and deterministic corruption or Markdown rendering. "
        f"Review and distribute under [CC BY-SA 3.0]({CC_BY_SA_URL}) or "
        f"[GFDL 1.3]({GFDL_URL}), as applicable. Legal review remains required before production use.\n"
    )
    return attribution_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    articles = read_articles(args.source_jsonl)
    split_assignments = cluster_source_splits(articles)
    rows = validate_rows(generate_rows(articles, split_assignments=split_assignments))
    accepted_cluster_members: dict[str, set[str]] = {}
    for row in rows:
        provenance = row["provenance"]
        source_id = provenance["source_record_id"].split(":", 1)[0]
        source_cluster_id = split_assignments[source_id][1]
        accepted_cluster_members.setdefault(source_cluster_id, set()).add(source_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "teacher_free_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    attribution_path = write_attribution(articles, rows, args.output_dir)
    manifest = write_artifacts(rows, args.output_dir)
    sequence_buckets = write_sequence_buckets(rows, args.output_dir)
    manifest.update(
        {
            "dataset_revision": DATASET_REVISION,
            "input_article_count": len(articles),
            "accepted_article_count": len(rows) // 4,
            "duplicate_article_groups_removed": len(articles) - len(rows) // 4,
            "source_cluster_count": len(accepted_cluster_members),
            "multi_article_source_cluster_count": sum(
                len(members) > 1 for members in accepted_cluster_members.values()
            ),
            "largest_accepted_source_cluster": max(map(len, accepted_cluster_members.values())),
            "source_clustering": {
                "assignment": "one deterministic split per connected component",
                "fields": ["source_fragment", "prompt", "response"],
                "near_duplicate_threshold": SPLIT_NEAR_DUPLICATE_THRESHOLD,
                "shingle_width": SPLIT_SHINGLE_WIDTH,
                "minimum_sequence_tokens": SPLIT_MIN_NEAR_TOKENS,
            },
            "source_sample_sha256": sha256(args.source_jsonl),
            "rows_sha256": sha256(rows_path),
            "attribution_sha256": sha256(attribution_path),
            "sequence_buckets": sequence_buckets,
            "family_counts": dict(
                sorted(
                    Counter(
                        tag for row in rows for tag in row["tags"] if tag not in {"teacher-free", "russian"}
                    ).items()
                )
            ),
            "license_gate": "CC-BY-SA-3.0/GFDL attribution and distribution review required before production use",
        }
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
