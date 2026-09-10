#!/usr/bin/env python3
"""Materialize a bounded, revision-locked Russian Wikipedia sentence sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from build_teacher_free_russian_corruptions import (
    SOURCE_CONFIG,
    SOURCE_DATASET,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    article_sentences,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--max-articles", type=int, default=512)
    parser.add_argument("--max-source-rows", type=int, default=20000)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=52053)
    args = parser.parse_args()
    if args.max_articles <= 0 or args.max_source_rows < args.max_articles:
        raise ValueError("max-source-rows must be at least max-articles > 0")

    import datasets
    from datasets import load_dataset

    started = time.monotonic()
    stream = load_dataset(
        SOURCE_DATASET,
        SOURCE_CONFIG,
        split="train",
        revision=SOURCE_REVISION,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    selected: list[dict] = []
    processed = 0
    for article in stream:
        processed += 1
        text = article.get("text")
        title = article.get("title")
        if not isinstance(text, str) or not isinstance(title, str):
            continue
        sentences = article_sentences(text)
        if len(sentences) < 3:
            if processed >= args.max_source_rows:
                break
            continue
        selected.append(
            {
                "id": str(article["id"]),
                "url": str(article["url"]),
                "title": title,
                "sentences": sentences[:3],
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "dataset": SOURCE_DATASET,
                "revision": SOURCE_REVISION,
                "config": SOURCE_CONFIG,
                "license": SOURCE_LICENSE,
            }
        )
        if len(selected) >= args.max_articles or processed >= args.max_source_rows:
            break
    if len(selected) != args.max_articles:
        raise RuntimeError(f"selected only {len(selected)}/{args.max_articles} articles after {processed} rows")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    )
    manifest = {
        "schema_version": 1,
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "config": SOURCE_CONFIG,
        "license": SOURCE_LICENSE,
        "datasets_version": datasets.__version__,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "max_source_rows": args.max_source_rows,
        "processed_rows": processed,
        "selected_articles": len(selected),
        "sample_sha256": sha256(args.output_jsonl),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_seconds": time.monotonic() - started,
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
