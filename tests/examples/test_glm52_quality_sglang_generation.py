from __future__ import annotations

import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import canonical_sha256, prompt_sha256  # noqa: E402
from generate_full_quality_outputs_sglang import (  # noqa: E402
    TEST_ACK,
    RequestFailure,
    decoding_contract,
    generate_one,
    validate_existing_rows,
    validate_runtime_manifest,
)


def runtime_manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "status": "EXACT-REVISION-SERVER-READY",
        "server_instance_id": "glm52-quality-20260902-validation",
        "served_base_model": "glm52-base",
        "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
        "server_args": {
            "model_path": "/models/glm52-fp8",
            "tp_size": 8,
            "enable_lora": True,
            "lora_paths": {"glm52-quality-mla-r16": "/adapters/glm52-quality"},
        },
        "trainer_base": {
            "model_id": "zai-org/GLM-5.2",
            "revision": "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
            "revision_verified": True,
            "config_sha256": "2" * 64,
            "weights_index_sha256": "3" * 64,
        },
        "inference_base": {
            "model_id": "zai-org/GLM-5.2-FP8",
            "revision": "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
            "revision_verified": True,
            "config_sha256": "4" * 64,
            "weights_index_sha256": "5" * 64,
        },
        "adapter": {
            "name": "glm52-quality-mla-r16",
            "artifact_sha256": "6" * 64,
            "config_sha256": "7" * 64,
            "verification_sha256": "8" * 64,
            "trainer_base_revision": "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
            "profile": "mla-only",
            "rank": 16,
            "alpha": 32,
            "parameter_count": 106_149_888,
            "target_modules": [
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "q_a_proj",
                "q_b_proj",
            ],
        },
        "sglang": {
            "checkout": "/src/sglang",
            "repository": "https://github.com/imvladikon/sglang",
            "revision": "328f776c80911dc10b5f0d787e140b6a241eb5b1",
        },
    }
    manifest["server_args_sha256"] = canonical_sha256(manifest["server_args"])
    return manifest


def source_row() -> dict:
    prompt = "Исправь русский текст."
    return {
        "id": "quality-example",
        "split": "validation",
        "system": "Отвечай по-русски.",
        "prompt": prompt,
        "response": "Исправленный русский текст.",
        "prompt_sha256": prompt_sha256(prompt),
        "contract": {
            "requested_language": "ru",
            "allow_han": False,
            "require_markdown": False,
            "required_markdown_blocks": [],
        },
    }


def decoding_args() -> Namespace:
    return Namespace(
        temperature=0.0,
        top_p=1.0,
        max_completion_tokens=512,
        seed=52,
    )


def response(model: str = "glm52-base") -> dict:
    return {
        "id": "response-id",
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": "Исправленный русский текст."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }


def test_official_and_test_runtime_boundaries() -> None:
    assert validate_runtime_manifest(runtime_manifest(), test_checkpoint_ack=None) is True

    test_runtime = deepcopy(runtime_manifest())
    test_runtime["status"] = "TEST-CHECKPOINT-SERVER-READY"
    test_runtime["trainer_base"]["model_id"] = "imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy"
    test_runtime["trainer_base"]["revision"] = "cc2b0f160092e9965d67792bc11fb16a57847ee5"
    test_runtime["adapter"]["trainer_base_revision"] = test_runtime["trainer_base"]["revision"]
    with pytest.raises(ValueError, match="test-checkpoint-ack"):
        validate_runtime_manifest(test_runtime, test_checkpoint_ack=None)
    assert validate_runtime_manifest(test_runtime, test_checkpoint_ack=TEST_ACK) is False


@pytest.mark.parametrize("variant", ["base", "adapter"])
def test_request_and_response_are_bound_to_the_runtime(variant: str) -> None:
    runtime = runtime_manifest()
    source = source_row()
    decoding = decoding_contract(decoding_args())
    observed: dict = {}

    def fake_request(endpoint: str, payload: dict, *, timeout: float, api_key: str | None) -> dict:
        observed.update(
            {"endpoint": endpoint, "payload": payload, "timeout": timeout, "api_key": api_key}
        )
        return response(model=payload["model"])

    row = generate_one(
        source,
        variant=variant,
        runtime=runtime,
        runtime_sha256="a" * 64,
        official=True,
        decoding=decoding,
        endpoint="http://127.0.0.1:30000/v1/chat/completions",
        api_key=None,
        timeout=30.0,
        retries=0,
        request_fn=fake_request,
    )

    assert observed["payload"]["reasoning_effort"] == "none"
    assert observed["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    if variant == "adapter":
        assert observed["payload"]["model"] == runtime["adapter"]["name"]
        assert observed["payload"]["lora_path"] == runtime["adapter"]["name"]
    else:
        assert observed["payload"]["model"] == runtime["served_base_model"]
        assert "lora_path" not in observed["payload"]
    assert row["completion_token_count"] == 6
    assert row["generation_pair_contract_sha256"] == "a" * 64
    assert row["generation"] == {
        "variant": variant,
        "runtime_manifest_sha256": "a" * 64,
        "quality_claim_allowed": True,
        "server_instance_id": runtime["server_instance_id"],
        "trainer_base": runtime["trainer_base"],
        "inference_base": runtime["inference_base"],
        "adapter": runtime["adapter"] if variant == "adapter" else None,
        "sglang": runtime["sglang"],
        "response_id": "response-id",
        "response_model": (
            "glm52-quality-mla-r16" if variant == "adapter" else "glm52-base"
        ),
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
    }


def test_resume_rejects_runtime_or_decoding_drift() -> None:
    runtime = runtime_manifest()
    source = source_row()
    decoding = decoding_contract(decoding_args())

    def fake_request(*args, **kwargs) -> dict:
        return response()

    row = generate_one(
        source,
        variant="base",
        runtime=runtime,
        runtime_sha256="a" * 64,
        official=True,
        decoding=decoding,
        endpoint="http://127.0.0.1:30000/v1/chat/completions",
        api_key=None,
        timeout=30.0,
        retries=0,
        request_fn=fake_request,
    )
    completed = validate_existing_rows(
        [row],
        {source["id"]: source},
        variant="base",
        runtime_sha256="a" * 64,
        official=True,
        decoding_sha256=canonical_sha256(decoding),
    )
    assert completed == {source["id"]}

    with pytest.raises(ValueError, match="generation_pair_contract_sha256"):
        validate_existing_rows(
            [row],
            {source["id"]: source},
            variant="base",
            runtime_sha256="b" * 64,
            official=True,
            decoding_sha256=canonical_sha256(decoding),
        )


def test_truncated_or_unmetered_response_fails_closed() -> None:
    runtime = runtime_manifest()
    source = source_row()
    decoding = decoding_contract(decoding_args())

    for broken_response, message in (
        ({**response(), "usage": {}}, "completion token count"),
        (
            {
                **response(),
                "choices": [
                    {
                        "message": {"content": "Незаконченный ответ"},
                        "finish_reason": "length",
                    }
                ],
            },
            "did not finish normally",
        ),
    ):

        def fake_request(*args, response_payload=broken_response, **kwargs) -> dict:
            return response_payload

        with pytest.raises(RequestFailure, match=message):
            generate_one(
                source,
                variant="base",
                runtime=runtime,
                runtime_sha256="a" * 64,
                official=True,
                decoding=decoding,
                endpoint="http://127.0.0.1:30000/v1/chat/completions",
                api_key=None,
                timeout=30.0,
                retries=0,
                request_fn=fake_request,
            )


def test_response_model_or_token_accounting_drift_fails_closed() -> None:
    runtime = runtime_manifest()
    source = source_row()
    decoding = decoding_contract(decoding_args())

    for broken_response, message in (
        ({**response(), "model": "wrong-model"}, "response model"),
        (
            {
                **response(),
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 15},
            },
            "token accounting",
        ),
    ):

        def fake_request(*args, response_payload=broken_response, **kwargs) -> dict:
            return response_payload

        with pytest.raises(RequestFailure, match=message):
            generate_one(
                source,
                variant="base",
                runtime=runtime,
                runtime_sha256="a" * 64,
                official=True,
                decoding=decoding,
                endpoint="http://127.0.0.1:30000/v1/chat/completions",
                api_key=None,
                timeout=30.0,
                retries=0,
                request_fn=fake_request,
            )
