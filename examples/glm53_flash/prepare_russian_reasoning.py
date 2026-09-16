# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert attn-signs/russian-reasoning into a LoRA SFT set for GLM-5.3-Flash.

The dataset teaches its own reasoning markup::

    <Thought>...</Thought> <output>...</output>

GLM-5.3's chat template already opens the reasoning block itself — a rendered prompt ends with
``<|assistant|><think>`` — so an assistant target must be the reasoning text, then ``</think>``,
then the answer. Training the dataset's markup verbatim would teach a second, conflicting
convention and leave the model emitting ``<output>`` where the reward parser expects an answer.

Usage::

    python examples/glm53_flash/prepare_russian_reasoning.py \\
        --output-dir /path/to/out --model /path/to/glm53-flash --max-length 8192

Reads the HuggingFace dataset by default; pass --input-parquet to convert local files instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

THOUGHT_RE = re.compile(r"<Thought>(.*?)</Thought>", re.S)
OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.S)

# The dataset's system prompt dictates <Thought>/<output>; keeping it would contradict the target
# format. None means "no system turn", which is what the model's own template expects.
DEFAULT_SYSTEM = None


def split_reasoning(assistant_content: str) -> tuple[str, str] | None:
    """Return (reasoning, answer) from the dataset's markup, or None when it is not there."""
    thought = THOUGHT_RE.search(assistant_content)
    output = OUTPUT_RE.search(assistant_content)
    if not thought or not output:
        return None
    reasoning = thought.group(1).strip()
    answer = output.group(1).strip()
    if not reasoning or not answer:
        return None
    return reasoning, answer


def build_messages(row: dict, system: str | None = DEFAULT_SYSTEM) -> list[dict] | None:
    """One dataset row as chat messages in the model's own reasoning format."""
    conversation = list(row.get("conversation") or [])
    user = next((m["content"] for m in conversation if m.get("role") == "user"), None)
    assistant = next((m["content"] for m in conversation if m.get("role") == "assistant"), None)
    if not user or not assistant:
        return None
    split = split_reasoning(assistant)
    if split is None:
        return None
    reasoning, answer = split
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": user})
    # The template emits "<|assistant|><think>", so the target carries the closing tag, not the
    # opening one.
    messages.append({"role": "assistant", "content": f"{reasoning}</think>{answer}"})
    return messages


def _iter_rows(args):
    if args.input_parquet:
        import pyarrow.parquet as pq

        for path in args.input_parquet:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
                yield from batch.to_pylist()
        return
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    yield from dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="attn-signs/russian-reasoning")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true", help="stream instead of downloading the split")
    parser.add_argument("--input-parquet", nargs="*", help="convert these local parquet files instead")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", help="tokenizer path; without it no length filtering is done")
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="drop samples whose rendered chat exceeds this many tokens (needs --model)",
    )
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1, help="stop after this many input rows")
    parser.add_argument("--system", default=None, help="system turn to prepend (default: none)")
    args = parser.parse_args()

    import pandas as pd

    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    kept: list[dict] = []
    seen = malformed = too_long = 0
    lengths: list[int] = []
    for row in _iter_rows(args):
        if args.limit > 0 and seen >= args.limit:
            break
        seen += 1
        messages = build_messages(row, system=args.system)
        if messages is None:
            malformed += 1
            continue
        if tokenizer is not None:
            # apply_chat_template(tokenize=True) can hand back a mapping rather than ids, so render
            # to text and tokenize that: the count has to be the real one, it decides what is kept.
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if token_count > args.max_length:
                too_long += 1
                continue
            lengths.append(token_count)
        kept.append({"messages": messages})

    if not kept:
        print("no usable samples", file=sys.stderr)
        return 1

    from pathlib import Path

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_size = min(args.val_size, len(kept) // 10)
    pd.DataFrame(kept[val_size:]).to_parquet(out / "train.parquet")
    if val_size:
        pd.DataFrame(kept[:val_size]).to_parquet(out / "val.parquet")

    report = {
        "input_rows": seen,
        "kept": len(kept) - val_size,
        "val": val_size,
        "dropped_malformed": malformed,
        "dropped_too_long": too_long,
        "max_length": args.max_length if tokenizer else None,
    }
    if lengths:
        lengths.sort()
        report["tokens_median"] = lengths[len(lengths) // 2]
        report["tokens_p90"] = lengths[int(0.9 * len(lengths))]
        report["tokens_max"] = lengths[-1]
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
