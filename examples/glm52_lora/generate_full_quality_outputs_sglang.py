#!/usr/bin/env python3
"""Generate exact-runtime paired GLM-5.2 quality outputs through SGLang.

The script is intentionally a client, not a server launcher.  A separately
hashed runtime manifest binds the live endpoint to its model revisions,
adapter artifact, SGLang revision, and server arguments.  Official full-model
manifests can produce quality evidence; surgery manifests require an explicit
test acknowledgement and remain engineering evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from build_blind_quality_review import (
    canonical_sha256,
    file_sha256,
    is_sha256,
    read_contracts,
    read_jsonl,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = 1
OFFICIAL_TRAINER = (
    "zai-org/GLM-5.2",
    "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
)
OFFICIAL_INFERENCE_BASES = {
    OFFICIAL_TRAINER,
    (
        "zai-org/GLM-5.2-FP8",
        "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    ),
}
TEST_ACK = "nonofficial-checkpoint-output-is-not-quality-evidence"
OFFICIAL_STATUS = "EXACT-REVISION-SERVER-READY"
TEST_STATUS = "TEST-CHECKPOINT-SERVER-READY"
MLA_TARGETS = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
}
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RequestFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _revision_block(value: Any, label: str) -> tuple[str, str]:
    block = _mapping(value, label)
    model_id = _nonempty_string(block.get("model_id"), f"{label}.model_id")
    revision = block.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{label}.revision must be a 40-character Git revision")
    if block.get("revision_verified") is not True:
        raise ValueError(f"{label}.revision_verified must be true")
    for field in ("config_sha256", "weights_index_sha256"):
        if not is_sha256(block.get(field)) or block[field] == "0" * 64:
            raise ValueError(f"{label}.{field} must be a SHA-256 digest")
    return model_id, revision


def validate_runtime_manifest(
    manifest: Any,
    *,
    test_checkpoint_ack: str | None,
) -> bool:
    manifest = _mapping(manifest, "runtime manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported runtime manifest schema")
    trainer = _revision_block(manifest.get("trainer_base"), "trainer_base")
    inference = _revision_block(manifest.get("inference_base"), "inference_base")

    sglang = _mapping(manifest.get("sglang"), "sglang")
    _nonempty_string(sglang.get("checkout"), "sglang.checkout")
    _nonempty_string(sglang.get("repository"), "sglang.repository")
    revision = sglang.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("sglang.revision must be a 40-character Git revision")
    server_args = _mapping(manifest.get("server_args"), "server_args")
    if manifest.get("server_args_sha256") != canonical_sha256(server_args):
        raise ValueError("server_args_sha256 differs from the embedded server arguments")
    _nonempty_string(manifest.get("endpoint"), "endpoint")
    _nonempty_string(manifest.get("server_instance_id"), "server_instance_id")
    _nonempty_string(manifest.get("served_base_model"), "served_base_model")

    adapter = _mapping(manifest.get("adapter"), "adapter")
    _nonempty_string(adapter.get("name"), "adapter.name")
    for field in ("artifact_sha256", "config_sha256", "verification_sha256"):
        if not is_sha256(adapter.get(field)) or adapter[field] == "0" * 64:
            raise ValueError(f"adapter.{field} must be a SHA-256 digest")
    if adapter.get("trainer_base_revision") != trainer[1]:
        raise ValueError("adapter trainer revision differs from trainer_base")
    if adapter.get("rank") != 16 or adapter.get("alpha") != 32:
        raise ValueError("adapter must use the locked rank-16/alpha-32 contract")
    parameter_count = adapter.get("parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise ValueError("adapter.parameter_count must be positive")
    profile = adapter.get("profile")
    expected_targets = set(MLA_TARGETS)
    if profile == "mla-lm-head":
        expected_targets.add("lm_head")
    elif profile != "mla-only":
        raise ValueError("adapter.profile must be mla-only or mla-lm-head")
    targets = adapter.get("target_modules")
    if not isinstance(targets, list) or set(targets) != expected_targets:
        raise ValueError(f"adapter target modules differ from the locked {profile} profile")

    official = trainer == OFFICIAL_TRAINER and inference in OFFICIAL_INFERENCE_BASES
    expected_status = OFFICIAL_STATUS if official else TEST_STATUS
    if manifest.get("status") != expected_status:
        raise ValueError(f"runtime status must be {expected_status}")
    if not official and test_checkpoint_ack != TEST_ACK:
        raise ValueError(
            "nonofficial runtime requires --test-checkpoint-ack "
            f"{TEST_ACK!r}"
        )
    return official


def load_runtime_manifest(
    path: Path,
    *,
    test_checkpoint_ack: str | None,
) -> tuple[dict[str, Any], str, bool]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    official = validate_runtime_manifest(
        manifest,
        test_checkpoint_ack=test_checkpoint_ack,
    )
    return manifest, file_sha256(path), official


def decoding_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_completion_tokens,
        "seed": args.seed,
        "n": 1,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def request_payload(
    source: dict[str, Any],
    *,
    variant: str,
    runtime: dict[str, Any],
    decoding: dict[str, Any],
) -> dict[str, Any]:
    model = runtime["served_base_model"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": source["system"]},
            {"role": "user", "content": source["prompt"]},
        ],
        **{key: value for key, value in decoding.items() if key != "schema_version"},
    }
    if variant == "adapter":
        adapter_name = runtime["adapter"]["name"]
        payload["model"] = adapter_name
        payload["lora_path"] = adapter_name
    return payload


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read(2048).decode("utf-8", errors="replace")
        raise RequestFailure(
            f"HTTP {error.code}: {body}",
            retryable=error.code in RETRYABLE_HTTP_STATUS,
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RequestFailure(str(error), retryable=True) from error
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise RequestFailure("server returned non-JSON response", retryable=False) from error
    if not isinstance(decoded, dict):
        raise RequestFailure("server response must be an object", retryable=False)
    return decoded


def response_row(
    source: dict[str, Any],
    response: dict[str, Any],
    *,
    variant: str,
    runtime: dict[str, Any],
    runtime_sha256: str,
    official: bool,
    decoding_sha256: str,
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RequestFailure("response must contain exactly one choice", retryable=False)
    choice = _mapping(choices[0], "response choice")
    message = _mapping(choice.get("message"), "response message")
    completion = message.get("content")
    if not isinstance(completion, str) or not completion.strip():
        raise RequestFailure("server returned an empty visible completion", retryable=False)
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise RequestFailure(f"completion did not finish normally: {finish_reason!r}", retryable=False)
    expected_model = (
        runtime["served_base_model"] if variant == "base" else runtime["adapter"]["name"]
    )
    if response.get("model") != expected_model:
        raise RequestFailure("response model differs from the requested runtime variant", retryable=False)
    usage = _mapping(response.get("usage"), "response usage")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) or completion_tokens < 1:
        raise RequestFailure("response must report a positive completion token count", retryable=False)
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
        or isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < prompt_tokens + completion_tokens
    ):
        raise RequestFailure("response token accounting is invalid", retryable=False)
    messages = [
        {"role": "system", "content": source["system"]},
        {"role": "user", "content": source["prompt"]},
    ]
    return {
        "id": source["id"],
        "completion": completion,
        "completion_token_count": completion_tokens,
        "contract": source["contract"],
        "prompt_sha256": source["prompt_sha256"],
        "request_messages_sha256": canonical_sha256(messages),
        "decoding_contract_sha256": decoding_sha256,
        "generation_pair_contract_sha256": runtime_sha256,
        "generation": {
            "variant": variant,
            "runtime_manifest_sha256": runtime_sha256,
            "quality_claim_allowed": official,
            "server_instance_id": runtime["server_instance_id"],
            "trainer_base": runtime["trainer_base"],
            "inference_base": runtime["inference_base"],
            "adapter": runtime["adapter"] if variant == "adapter" else None,
            "sglang": runtime["sglang"],
            "response_id": response.get("id"),
            "response_model": response.get("model"),
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def generate_one(
    source: dict[str, Any],
    *,
    variant: str,
    runtime: dict[str, Any],
    runtime_sha256: str,
    official: bool,
    decoding: dict[str, Any],
    endpoint: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    request_fn: Callable[..., dict[str, Any]] = post_json,
) -> dict[str, Any]:
    payload = request_payload(source, variant=variant, runtime=runtime, decoding=decoding)
    last_error: RequestFailure | None = None
    for attempt in range(retries + 1):
        try:
            response = request_fn(endpoint, payload, timeout=timeout, api_key=api_key)
            return response_row(
                source,
                response,
                variant=variant,
                runtime=runtime,
                runtime_sha256=runtime_sha256,
                official=official,
                decoding_sha256=canonical_sha256(decoding),
            )
        except RequestFailure as error:
            last_error = error
            if not error.retryable or attempt == retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("request retry loop exhausted") from last_error


def validate_existing_rows(
    rows: list[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    *,
    variant: str,
    runtime_sha256: str,
    official: bool,
    decoding_sha256: str,
) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("id", "")).strip()
        if not example_id or example_id in seen:
            raise ValueError(f"existing output has missing or duplicate ID {example_id!r}")
        source = contracts.get(example_id)
        if source is None:
            raise ValueError(f"existing output ID {example_id!r} is outside the selected split")
        expected_messages = [
            {"role": "system", "content": source["system"]},
            {"role": "user", "content": source["prompt"]},
        ]
        expected = {
            "contract": source["contract"],
            "prompt_sha256": source["prompt_sha256"],
            "request_messages_sha256": canonical_sha256(expected_messages),
            "decoding_contract_sha256": decoding_sha256,
            "generation_pair_contract_sha256": runtime_sha256,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{example_id}: existing {field} differs from the current run")
        generation = _mapping(row.get("generation"), f"{example_id}.generation")
        if generation.get("variant") != variant:
            raise ValueError(f"{example_id}: existing generation variant differs")
        if generation.get("runtime_manifest_sha256") != runtime_sha256:
            raise ValueError(f"{example_id}: existing runtime manifest differs")
        if generation.get("quality_claim_allowed") is not official:
            raise ValueError(f"{example_id}: existing quality-claim status differs")
        completion = row.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{example_id}: existing completion is empty")
        token_count = row.get("completion_token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
            raise ValueError(f"{example_id}: existing completion token count is invalid")
        seen.add(example_id)
    return seen


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if not parsed.path.endswith("/v1/chat/completions"):
        raise ValueError("endpoint path must end in /v1/chat/completions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contracts", nargs="+", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-checkpoint-ack")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_endpoint(args.endpoint)
    if not 1 <= args.concurrency <= 64:
        raise ValueError("concurrency must be in [1, 64]")
    if args.max_completion_tokens < 1 or args.request_timeout <= 0 or args.retries < 0:
        raise ValueError("token limit and timeout must be positive; retries must be nonnegative")
    if args.output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume to validate and continue it")
    if args.manifest.exists() and not args.resume:
        raise FileExistsError("manifest exists; pass --resume to validate and continue")

    contracts = read_contracts(args.contracts, split=args.split)
    runtime, runtime_sha256, official = load_runtime_manifest(
        args.runtime_manifest,
        test_checkpoint_ack=args.test_checkpoint_ack,
    )
    if args.endpoint.rstrip("/") != runtime["endpoint"].rstrip("/"):
        raise ValueError("endpoint differs from the hashed runtime manifest")
    decoding = decoding_contract(args)
    decoding_sha256 = canonical_sha256(decoding)
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    completed = validate_existing_rows(
        existing_rows,
        contracts,
        variant=args.variant,
        runtime_sha256=runtime_sha256,
        official=official,
        decoding_sha256=decoding_sha256,
    )
    pending = [contracts[example_id] for example_id in sorted(contracts) if example_id not in completed]
    api_key = args.api_key_file.read_text(encoding="utf-8").strip() if args.api_key_file else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    generate_one,
                    source,
                    variant=args.variant,
                    runtime=runtime,
                    runtime_sha256=runtime_sha256,
                    official=official,
                    decoding=decoding,
                    endpoint=args.endpoint,
                    api_key=api_key,
                    timeout=args.request_timeout,
                    retries=args.retries,
                ): source["id"]
                for source in pending
            }
            for index, future in enumerate(as_completed(futures), len(completed) + 1):
                example_id = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    for queued in futures:
                        queued.cancel()
                    raise RuntimeError(f"generation failed for {example_id}") from error
                with args.output.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    output.flush()
                print(f"[{index}/{len(contracts)}] {example_id}", file=sys.stderr, flush=True)

    rows = read_jsonl(args.output)
    completed = validate_existing_rows(
        rows,
        contracts,
        variant=args.variant,
        runtime_sha256=runtime_sha256,
        official=official,
        decoding_sha256=decoding_sha256,
    )
    if completed != set(contracts):
        raise RuntimeError("output does not cover the complete selected split")
    rows.sort(key=lambda row: row["id"])
    write_jsonl(args.output, rows)
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "QUALITY-OUTPUTS-COMPLETE" if official else "TEST-OUTPUTS-COMPLETE",
        "variant": args.variant,
        "split": args.split,
        "count": len(rows),
        "quality_claim_allowed": official,
        "runtime_manifest_sha256": runtime_sha256,
        "decoding_contract": decoding,
        "decoding_contract_sha256": decoding_sha256,
        "contracts_sha256": {str(path): file_sha256(path) for path in args.contracts},
        "output_sha256": file_sha256(args.output),
    }
    write_json(args.manifest, output_manifest)
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
