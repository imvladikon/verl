#!/usr/bin/env python3
"""Generate a bounded GLM-5.2 hosted-inference quality preflight.

This is deliberately not an exact-revision quality oracle: Hugging Face's
conversational provider API selects a live provider deployment and does not
accept a Hub commit revision. The requested Hub revision is verified as model
metadata and the provider limitation is recorded on every row and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

MODEL_ID = "zai-org/GLM-5.2"
MODEL_REVISION = "cf457fa734ab149ffef225f80893eb38c6ff5cdc"
REVISION_ACK = "hosted-provider-revision-is-not-hf-pinned"
SCHEMA_VERSION = 1
WHITESPACE_RE = re.compile(r"\s+")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_sha256(prompt: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", prompt).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_contract_rows(paths: Iterable[Path], *, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            if row.get("split") != split:
                continue
            example_id = str(row.get("id", "")).strip()
            if not example_id or example_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: missing or duplicate id {example_id!r}")
            prompt = row.get("prompt")
            system = row.get("system")
            contract = row.get("contract")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{example_id}: prompt must be nonempty")
            if not isinstance(system, str) or not system:
                raise ValueError(f"{example_id}: system must be nonempty")
            if not isinstance(contract, dict):
                raise TypeError(f"{example_id}: contract must be an object")
            actual_prompt_sha = prompt_sha256(prompt)
            if row.get("prompt_sha256") != actual_prompt_sha:
                raise ValueError(f"{example_id}: prompt SHA-256 mismatch")
            seen_ids.add(example_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"no {split!r} rows found")
    return rows


def read_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("IDs file must contain distinct, nonempty IDs")
    return ids


def select_rows(
    rows: list[dict[str, Any]],
    *,
    ids: list[str] | None,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if ids is None:
        return rows[:max_examples]
    indexed = {row["id"]: row for row in rows}
    missing = [example_id for example_id in ids if example_id not in indexed]
    if missing:
        raise ValueError(f"IDs are absent from the selected split: {missing[:5]}")
    if len(ids) > max_examples:
        raise ValueError("IDs file exceeds max_examples billing cap")
    return [indexed[example_id] for example_id in ids]


def decoding_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "glm52-quality-decoding-v1",
        "message_contract": "system-then-user-verbatim",
        "thinking": {"type": args.thinking},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
    }


def validate_runtime_acks(args: argparse.Namespace) -> None:
    expected_billing_ack = (
        f"max_examples={args.max_examples},max_tokens={args.max_tokens}"
    )
    if args.billing_ack != expected_billing_ack:
        raise SystemExit(
            "billing acknowledgement mismatch; pass exactly "
            f"--billing-ack {expected_billing_ack!r}"
        )
    if args.unverified_revision_ack != REVISION_ACK:
        raise SystemExit(
            "hosted providers cannot pin the Hub commit; pass exactly "
            f"--unverified-revision-ack {REVISION_ACK!r}"
        )


def response_row(
    source: dict[str, Any],
    response: Any,
    *,
    provider: str,
    decoding_contract_sha256: str,
) -> dict[str, Any]:
    choice = response.choices[0]
    completion = choice.message.content or ""
    if not completion.strip():
        raise RuntimeError(
            f"{source['id']}: provider returned an empty visible completion"
        )
    messages = [
        {"role": "system", "content": source["system"]},
        {"role": "user", "content": source["prompt"]},
    ]
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    return {
        "id": source["id"],
        "completion": completion,
        "completion_token_count": completion_tokens,
        "contract": source["contract"],
        "prompt_sha256": source["prompt_sha256"],
        "request_messages_sha256": canonical_sha256(messages),
        "decoding_contract_sha256": decoding_contract_sha256,
        "generation": {
            "provider": provider,
            "requested_model": MODEL_ID,
            "requested_hub_revision": MODEL_REVISION,
            "provider_revision_verified": False,
            "served_model": getattr(response, "model", None),
            "request_id": getattr(response, "request_id", None),
            "created": getattr(response, "created", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": completion_tokens,
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }


def read_existing(
    path: Path,
    *,
    decoding_hash: str,
    expected_sources: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("id", "")).strip()
        if not example_id or example_id in completed:
            raise ValueError(f"{path}:{line_number}: missing or duplicate id")
        if row.get("decoding_contract_sha256") != decoding_hash:
            raise ValueError(f"{example_id}: existing decoding contract differs")
        if expected_sources is not None:
            source = expected_sources.get(example_id)
            if source is None:
                raise ValueError(f"{example_id}: existing ID is outside this bounded plan")
            messages = [
                {"role": "system", "content": source["system"]},
                {"role": "user", "content": source["prompt"]},
            ]
            if row.get("prompt_sha256") != source["prompt_sha256"]:
                raise ValueError(f"{example_id}: existing prompt differs")
            if row.get("request_messages_sha256") != canonical_sha256(messages):
                raise ValueError(f"{example_id}: existing request messages differ")
            if row.get("contract") != source["contract"]:
                raise ValueError(f"{example_id}: existing quality contract differs")
        completed.add(example_id)
    return completed


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_manifest(
    plan: dict[str, Any],
    *,
    output: Path,
    completed_count: int,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        **plan,
        "completed_count": completed_count,
        "output": str(output),
        "output_sha256": file_sha256(output) if output.is_file() else None,
    }


def is_retryable_provider_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code not in {400, 401, 402, 403, 404, 409, 422}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--max-examples", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--provider", default="zai-org")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--billing-ack")
    parser.add_argument("--unverified-revision-ack")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="write a partial manifest for existing rows without an API call",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0 or args.max_tokens > 4096:
        raise SystemExit("max_tokens must be in [1, 4096]")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise SystemExit("top_p must be in (0, 1]")
    if args.retries <= 0 or args.retries > 5:
        raise SystemExit("retries must be in [1, 5]")
    rows = read_contract_rows(args.inputs, split=args.split)
    selected = select_rows(
        rows,
        ids=read_ids(args.ids_file),
        max_examples=args.max_examples,
    )
    contract = decoding_contract(args)
    decoding_hash = canonical_sha256(contract)
    input_hashes = {str(path): file_sha256(path) for path in args.inputs}
    plan = {
        "selected_ids": [row["id"] for row in selected],
        "selected_count": len(selected),
        "maximum_requested_output_tokens": len(selected) * args.max_tokens,
        "provider": args.provider,
        "requested_model": MODEL_ID,
        "requested_hub_revision": MODEL_REVISION,
        "provider_revision_verified": False,
        "decoding_contract": contract,
        "decoding_contract_sha256": decoding_hash,
        "input_sha256": input_hashes,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    expected_sources = {row["id"]: row for row in selected}
    if args.finalize_existing:
        completed = read_existing(
            args.output,
            decoding_hash=decoding_hash,
            expected_sources=expected_sources,
        )
        selected_ids = {row["id"] for row in selected}
        if not completed <= selected_ids:
            raise ValueError("existing output contains IDs outside this bounded plan")
        status = (
            "HOSTED-PREFLIGHT/PROVIDER-REVISION-UNVERIFIED"
            if len(completed) == len(selected)
            else "PARTIAL/HOSTED-PREFLIGHT/PROVIDER-REVISION-UNVERIFIED"
        )
        manifest = build_manifest(
            plan,
            output=args.output,
            completed_count=len(completed),
            status=status,
        )
        write_manifest_atomic(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    validate_runtime_acks(args)
    from huggingface_hub import HfApi, InferenceClient

    metadata = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
    if metadata.sha != MODEL_REVISION:
        raise RuntimeError(
            f"Hub revision mismatch: expected {MODEL_REVISION}, got {metadata.sha}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_existing(
        args.output,
        decoding_hash=decoding_hash,
        expected_sources=expected_sources,
    )
    selected_ids = {row["id"] for row in selected}
    if not completed <= selected_ids:
        raise ValueError("existing output contains IDs outside this bounded plan")
    write_manifest_atomic(
        manifest_path,
        build_manifest(
            plan,
            output=args.output,
            completed_count=len(completed),
            status="PARTIAL/HOSTED-PREFLIGHT/PROVIDER-REVISION-UNVERIFIED",
        ),
    )

    client = InferenceClient(
        model=MODEL_ID,
        provider=args.provider,
        token=True,
        timeout=args.timeout,
    )
    with args.output.open("a", encoding="utf-8") as output:
        for row in selected:
            if row["id"] in completed:
                continue
            messages = [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["prompt"]},
            ]
            last_error: Exception | None = None
            for attempt in range(1, args.retries + 1):
                try:
                    response = client.chat_completion(
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                        extra_body={"thinking": {"type": args.thinking}},
                    )
                    break
                except Exception as error:  # provider/network boundary
                    last_error = error
                    if attempt == args.retries or not is_retryable_provider_error(error):
                        raise
                    time.sleep(min(2 ** (attempt - 1), 4))
            else:  # pragma: no cover - defensive; loop either breaks or raises
                raise RuntimeError("provider retry loop exhausted") from last_error
            generated = response_row(
                row,
                response,
                provider=args.provider,
                decoding_contract_sha256=decoding_hash,
            )
            output.write(json.dumps(generated, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            completed.add(row["id"])
            write_manifest_atomic(
                manifest_path,
                build_manifest(
                    plan,
                    output=args.output,
                    completed_count=len(completed),
                    status="PARTIAL/HOSTED-PREFLIGHT/PROVIDER-REVISION-UNVERIFIED",
                ),
            )
            print(f"completed {len(completed)}/{len(selected)}: {row['id']}", flush=True)

    manifest = build_manifest(
        plan,
        output=args.output,
        completed_count=len(completed),
        status="HOSTED-PREFLIGHT/PROVIDER-REVISION-UNVERIFIED",
    )
    write_manifest_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
