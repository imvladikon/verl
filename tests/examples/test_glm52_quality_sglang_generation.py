from __future__ import annotations

import json
import sys
import threading
import urllib.request
from argparse import Namespace
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    prompt_sha256,
)
from evaluate_quality_outputs import evaluate_rows  # noqa: E402
from generate_full_quality_outputs_sglang import (  # noqa: E402
    OFFICIAL_MODEL_ARTIFACTS,
    OFFICIAL_TRAINER,
    TEST_ACK,
    TRUSTED_WEIGHT_SHARD_MANIFESTS,
    RequestFailure,
    build_pair_contract,
    build_pair_runtime_contract,
    decoding_contract,
    derive_han_evaluation_mode,
    evaluation_cluster_id,
    generate_one,
    output_manifest_contract,
    post_json,
    require_runtime_mode,
    secret_sha256,
    validate_existing_output_manifest,
    validate_existing_rows,
    validate_runtime_manifest,
)

API_KEY = "test-api-key-0123456789abcdef0123456789"
TRAINER_SHARD_MANIFEST_SHA = "1" * 64
INFERENCE_SHARD_MANIFEST_SHA = "2" * 64
TRUSTED_WEIGHT_SHARD_MANIFESTS[OFFICIAL_TRAINER] = TRAINER_SHARD_MANIFEST_SHA
TRUSTED_WEIGHT_SHARD_MANIFESTS[("zai-org/GLM-5.2-FP8", "f33c6dc501ee5a2c7e35155653b1b1abbc320951")] = (
    INFERENCE_SHARD_MANIFEST_SHA
)


class _RecordingServer(ThreadingHTTPServer):
    requests: list[dict]
    redirect_to: str | None


class _RecordingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "authorization": self.headers.get("Authorization"),
                "body": body,
                "path": self.path,
            }
        )
        redirect_to = self.server.redirect_to  # type: ignore[attr-defined]
        if redirect_to is not None:
            self.send_response(302)
            self.send_header("Location", redirect_to)
            self.end_headers()
            return
        response_body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _recording_server(*, redirect_to: str | None = None) -> _RecordingServer:
    server = _RecordingServer(("127.0.0.1", 0), _RecordingHandler)
    server.requests = []
    server.redirect_to = redirect_to
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def runtime_manifest(mode: str = "adapter") -> dict:
    trainer = {
        "model_id": OFFICIAL_TRAINER[0],
        "revision": OFFICIAL_TRAINER[1],
        "revision_verified": True,
        **OFFICIAL_MODEL_ARTIFACTS[OFFICIAL_TRAINER],
    }
    inference_identity = (
        "zai-org/GLM-5.2-FP8",
        "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    )
    inference = {
        "model_id": inference_identity[0],
        "revision": inference_identity[1],
        "revision_verified": True,
        **OFFICIAL_MODEL_ARTIFACTS[inference_identity],
    }
    server_args = {
        "model_path": "/models/glm52-fp8",
        "served_base_model": "glm52-base",
        "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
        "tp_size": 8,
        "gpu_ids": list(range(8)),
        "max_model_len": 2048,
        "enable_lora": mode == "adapter",
    }
    if mode == "adapter":
        server_args.update(
            {
                "lora_strict_loading": True,
                "lora_paths": {"glm52-quality-mla-r16": "/adapters/glm52-quality"},
                "max_lora_rank": 16,
                "lora_target_modules": [
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                    "o_proj",
                    "q_a_proj",
                    "q_b_proj",
                ],
            }
        )
    manifest = {
        "schema_version": 3,
        "status": "EXACT-REVISION-SERVER-READY",
        "runtime_mode": mode,
        "server_instance_id": "glm52-quality-20260902-validation",
        "served_base_model": "glm52-base",
        "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
        "server_args": server_args,
        "trainer_base": trainer,
        "inference_base": inference,
        "artifact_contract": {"trainer_base": trainer, "inference_base": inference},
        "weight_shard_identity": {
            "trainer": {
                "status": "VERIFIED",
                "model_id": trainer["model_id"],
                "revision": trainer["revision"],
                "manifest_sha256": TRAINER_SHARD_MANIFEST_SHA,
                "local_verification_receipt_sha256": "3" * 64,
                "verification_method": "trusted-sha256-manifest+full-read-once+stat-cache-v1",
                "shard_count": trainer["shard_count"],
                "shard_bytes_on_disk": trainer["shard_bytes_on_disk"],
            },
            "inference": {
                "status": "VERIFIED",
                "model_id": inference["model_id"],
                "revision": inference["revision"],
                "manifest_sha256": INFERENCE_SHARD_MANIFEST_SHA,
                "local_verification_receipt_sha256": "4" * 64,
                "verification_method": "trusted-sha256-manifest+full-read-once+stat-cache-v1",
                "shard_count": inference["shard_count"],
                "shard_bytes_on_disk": inference["shard_bytes_on_disk"],
            },
        },
        "local_artifacts": {
            "trainer_model_path": "/models/glm52-bf16",
            "inference_model_path": "/models/glm52-fp8",
            "trainer_weight_shard_manifest": "/proofs/trainer-shards.json",
            "inference_weight_shard_manifest": "/proofs/inference-shards.json",
            "weight_verification_cache_dir": "/proofs/shard-cache",
            **(
                {
                    "adapter_path": "/adapters/glm52-quality",
                    "adapter_verification_path": "/proofs/glm52-quality.json",
                }
                if mode == "adapter"
                else {}
            ),
        },
        "adapter": None
        if mode == "base"
        else {
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
            "tensor_count": 780,
            "lora_b_tensor_count": 390,
            "tensor_dtype": "torch.bfloat16",
            "topology_sha256": "9" * 64,
            "tensor_validation_status": "FINITE-NONZERO-B-TOPOLOGY-VERIFIED",
        },
        "sglang": {
            "checkout": "/src/sglang",
            "repository": "https://github.com/imvladikon/sglang",
            "revision": "0dbdb73509fbf6b3381359df87cde267d453c8d3",
            "tree": "5678fc2ab88fd65411b833c065f510b6d4f5d59c",
            "live_code_sha256": {
                "python/sglang/launch_server.py": "1" * 64,
                "python/sglang/srt/server_args.py": "2" * 64,
                "python/sglang/srt/models/glm4_moe.py": "3" * 64,
                "python/sglang/srt/models/deepseek_v2.py": "4" * 64,
                "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py": "5" * 64,
                "python/sglang/srt/lora/lora_manager.py": "6" * 64,
                "python/sglang/srt/lora/lora_registry.py": "7" * 64,
            },
        },
        "code_artifacts": {
            "runtime_scripts": {
                name: {"path": f"/src/runtime/{name}", "sha256": character * 64}
                for name, character in (
                    ("build_quality_sglang_runtime.py", "a"),
                    ("generate_full_quality_outputs_sglang.py", "b"),
                    ("launch_quality_sglang_server.py", "c"),
                    ("build_blind_quality_review.py", "d"),
                )
            }
        },
        "environment_artifacts": {
            "python_executable": "/venv/bin/python",
            "python_executable_sha256": "e" * 64,
            "python_version": "3.12.11",
            "installed_distributions_sha256": "f" * 64,
        },
        "api_secret_sha256": secret_sha256(API_KEY),
    }
    manifest["server_args_sha256"] = canonical_sha256(manifest["server_args"])
    manifest["pair_runtime_contract"] = build_pair_runtime_contract(manifest)
    manifest["pair_runtime_contract_sha256"] = canonical_sha256(manifest["pair_runtime_contract"])
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
        "tags": ["russian", "quality"],
        "provenance": {
            "dataset": "project-authored/test",
            "license": "apache-2.0",
            "revision": "test-v1",
            "source_split": "generated",
            "source_record_id": "quality-example",
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
                "message": {
                    "role": "assistant",
                    "content": "Исправленный русский текст.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }


def test_official_and_test_runtime_boundaries() -> None:
    adapter_runtime = runtime_manifest()
    base_runtime = runtime_manifest("base")
    assert validate_runtime_manifest(adapter_runtime, test_checkpoint_ack=None) is True
    assert validate_runtime_manifest(base_runtime, test_checkpoint_ack=None) is True
    assert adapter_runtime["adapter"] is not None
    assert base_runtime["adapter"] is None
    assert adapter_runtime["pair_runtime_contract"] == base_runtime["pair_runtime_contract"]
    assert adapter_runtime["pair_runtime_contract_sha256"] == base_runtime["pair_runtime_contract_sha256"]
    assert canonical_sha256(adapter_runtime) != canonical_sha256(base_runtime)

    test_runtime = deepcopy(adapter_runtime)
    test_runtime["status"] = "TEST-CHECKPOINT-SERVER-READY"
    test_runtime["trainer_base"]["model_id"] = "imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy"
    test_runtime["trainer_base"]["revision"] = "cc2b0f160092e9965d67792bc11fb16a57847ee5"
    test_runtime["adapter"]["trainer_base_revision"] = test_runtime["trainer_base"]["revision"]
    test_runtime["weight_shard_identity"]["trainer"].update(
        {
            "status": "PENDING-TRUSTED-MANIFEST",
            "model_id": test_runtime["trainer_base"]["model_id"],
            "revision": test_runtime["trainer_base"]["revision"],
            "manifest_sha256": None,
            "local_verification_receipt_sha256": None,
        }
    )
    test_runtime["local_artifacts"]["trainer_weight_shard_manifest"] = None
    test_runtime["pair_runtime_contract"] = build_pair_runtime_contract(test_runtime)
    test_runtime["pair_runtime_contract_sha256"] = canonical_sha256(test_runtime["pair_runtime_contract"])
    with pytest.raises(ValueError, match="test-checkpoint-ack"):
        validate_runtime_manifest(test_runtime, test_checkpoint_ack=None)
    assert validate_runtime_manifest(test_runtime, test_checkpoint_ack=TEST_ACK) is False


def test_official_identity_without_trusted_shard_proof_is_pending() -> None:
    runtime = runtime_manifest("base")
    runtime["status"] = "OFFICIAL-QUALITY-PENDING-SHARD-IDENTITY"
    runtime["weight_shard_identity"]["trainer"].update(
        {
            "status": "PENDING-TRUSTED-MANIFEST",
            "manifest_sha256": None,
            "local_verification_receipt_sha256": None,
        }
    )
    runtime["local_artifacts"]["trainer_weight_shard_manifest"] = None
    runtime["pair_runtime_contract"] = build_pair_runtime_contract(runtime)
    runtime["pair_runtime_contract_sha256"] = canonical_sha256(runtime["pair_runtime_contract"])
    with pytest.raises(ValueError, match="test-checkpoint-ack"):
        validate_runtime_manifest(runtime, test_checkpoint_ack=None)
    assert validate_runtime_manifest(runtime, test_checkpoint_ack=TEST_ACK) is False


def test_generation_rejects_runtime_mode_mismatch() -> None:
    with pytest.raises(ValueError, match="base runtime manifest"):
        require_runtime_mode(runtime_manifest("adapter"), "base")
    with pytest.raises(ValueError, match="adapter runtime manifest"):
        require_runtime_mode(runtime_manifest("base"), "adapter")


@pytest.mark.parametrize("variant", ["base", "adapter"])
def test_request_and_response_are_bound_to_the_runtime(variant: str) -> None:
    runtime = runtime_manifest(variant)
    source = source_row()
    decoding = decoding_contract(decoding_args())
    observed: dict = {}

    def fake_request(endpoint: str, payload: dict, *, timeout: float, api_key: str | None) -> dict:
        observed.update(
            {
                "endpoint": endpoint,
                "payload": payload,
                "timeout": timeout,
                "api_key": api_key,
            }
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
        api_key=API_KEY,
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
    assert row["input_han_count"] == 0
    assert row["input_contains_han"] is False
    assert row["han_evaluation_mode"] == "spontaneous"
    assert row["evaluation_cluster_id"].startswith("reference-provenance:")
    pair_contract = build_pair_contract(source, runtime=runtime, decoding=decoding)
    assert row["pair_contract"] == pair_contract
    assert row["pair_contract_sha256"] == canonical_sha256(pair_contract)
    assert row["generation"] == {
        "variant": variant,
        "runtime_mode": variant,
        "runtime_manifest_sha256": "a" * 64,
        "pair_runtime_contract_sha256": runtime["pair_runtime_contract_sha256"],
        "quality_claim_allowed": True,
        "api_secret_sha256": runtime["api_secret_sha256"],
        "server_instance_id": runtime["server_instance_id"],
        "trainer_base": runtime["trainer_base"],
        "inference_base": runtime["inference_base"],
        "adapter": runtime["adapter"] if variant == "adapter" else None,
        "sglang": runtime["sglang"],
        "response_id": "response-id",
        "response_model": ("glm52-quality-mla-r16" if variant == "adapter" else "glm52-base"),
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
    }


def test_resume_rejects_runtime_or_decoding_drift() -> None:
    runtime = runtime_manifest("base")
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
        api_key=API_KEY,
        timeout=30.0,
        retries=0,
        request_fn=fake_request,
    )
    completed = validate_existing_rows(
        [row],
        {source["id"]: source},
        variant="base",
        runtime=runtime,
        runtime_sha256="a" * 64,
        official=True,
        decoding=decoding,
    )
    assert completed == {source["id"]}

    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        validate_existing_rows(
            [row],
            {source["id"]: source},
            variant="base",
            runtime=runtime,
            runtime_sha256="b" * 64,
            official=True,
            decoding=decoding,
        )


def test_resume_rejects_every_bound_row_and_generation_field() -> None:
    runtime = runtime_manifest("base")
    source = source_row()
    decoding = decoding_contract(decoding_args())
    row = generate_one(
        source,
        variant="base",
        runtime=runtime,
        runtime_sha256="a" * 64,
        official=True,
        decoding=decoding,
        endpoint=runtime["endpoint"],
        api_key=API_KEY,
        timeout=30.0,
        retries=0,
        request_fn=lambda *args, **kwargs: response(),
    )

    def reject(candidate: dict) -> None:
        with pytest.raises(ValueError):
            validate_existing_rows(
                [candidate],
                {source["id"]: source},
                variant="base",
                runtime=runtime,
                runtime_sha256="a" * 64,
                official=True,
                decoding=decoding,
            )

    for field in (
        "split",
        "contract",
        "prompt_sha256",
        "source_row_sha256",
        "reference_response_sha256",
        "request_messages_sha256",
        "input_han_count",
        "input_contains_han",
        "han_evaluation_mode",
        "evaluation_cluster_id",
        "decoding_contract_sha256",
        "pair_contract",
        "pair_contract_sha256",
    ):
        candidate = deepcopy(row)
        candidate[field] = None
        reject(candidate)

    generation_mutations = {
        "variant": "adapter",
        "runtime_mode": "adapter",
        "runtime_manifest_sha256": "b" * 64,
        "pair_runtime_contract_sha256": "b" * 64,
        "quality_claim_allowed": False,
        "api_secret_sha256": "b" * 64,
        "server_instance_id": "different-server",
        "trainer_base": {},
        "inference_base": {},
        "adapter": {},
        "sglang": {},
        "response_id": "",
        "response_model": "different-model",
        "finish_reason": "length",
    }
    for field, value in generation_mutations.items():
        candidate = deepcopy(row)
        candidate["generation"][field] = value
        reject(candidate)

    for field, value in (
        ("completion", ""),
        ("completion_token_count", 0),
        ("prompt_tokens", 0),
        ("completion_tokens", 7),
        ("total_tokens", 15),
    ):
        candidate = deepcopy(row)
        if field in candidate["generation"]:
            candidate["generation"][field] = value
        else:
            candidate[field] = value
        reject(candidate)

    candidate = deepcopy(row)
    candidate["unexpected"] = "unbound"
    reject(candidate)
    candidate = deepcopy(row)
    candidate["generation"]["unexpected"] = "unbound"
    reject(candidate)


def test_generated_input_han_provenance_reaches_evaluator() -> None:
    runtime = runtime_manifest("base")
    source = source_row()
    source["prompt"] = "Удали случайный знак 中 из русского текста."
    source["prompt_sha256"] = prompt_sha256(source["prompt"])
    source["tags"] = ["han-cleanup", "russian"]
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
        endpoint=runtime["endpoint"],
        api_key=API_KEY,
        timeout=30.0,
        retries=0,
        request_fn=fake_request,
    )
    summary, details = evaluate_rows([row])

    assert row["input_han_count"] == 1
    assert row["input_contains_han"] is True
    assert row["pair_contract"]["held_out"]["input_han_count"] == 1
    assert row["han_evaluation_mode"] == "input_conditioned_cleanup"
    assert summary["input_conditioned_han_cleanup"]["cleanup_success_rate"] == 1.0
    assert details[0]["han_evaluation_mode"] == "input_conditioned_cleanup"


def test_truncated_or_unmetered_response_fails_closed() -> None:
    runtime = runtime_manifest("base")
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
                api_key=API_KEY,
                timeout=30.0,
                retries=0,
                request_fn=fake_request,
            )


def test_response_model_or_token_accounting_drift_fails_closed() -> None:
    runtime = runtime_manifest("base")
    source = source_row()
    decoding = decoding_contract(decoding_args())

    for broken_response, message in (
        ({**response(), "model": "wrong-model"}, "response model"),
        (
            {
                **response(),
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 6,
                    "total_tokens": 15,
                },
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
                api_key=API_KEY,
                timeout=30.0,
                retries=0,
                request_fn=fake_request,
            )


def _refresh_runtime_hashes(runtime: dict) -> None:
    runtime["server_args_sha256"] = canonical_sha256(runtime["server_args"])
    runtime["pair_runtime_contract"] = build_pair_runtime_contract(runtime)
    runtime["pair_runtime_contract_sha256"] = canonical_sha256(runtime["pair_runtime_contract"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda runtime: runtime["inference_base"].__setitem__("config_sha256", "0" * 64),
            "SHA-256",
        ),
        (
            lambda runtime: runtime["inference_base"].__setitem__("tokenizer_json_sha256", "1" * 64),
            "official model artifact",
        ),
        (
            lambda runtime: runtime["inference_base"].__setitem__("shard_count", 140),
            "shard inventory",
        ),
        (
            lambda runtime: runtime.__setitem__("api_secret_sha256", None),
            "API secret commitment",
        ),
    ),
)
def test_official_runtime_rejects_artifact_or_secret_bypass(mutation, message: str) -> None:
    runtime = runtime_manifest("base")
    mutation(runtime)
    runtime["artifact_contract"] = {
        "trainer_base": runtime["trainer_base"],
        "inference_base": runtime["inference_base"],
    }
    _refresh_runtime_hashes(runtime)
    with pytest.raises(ValueError, match=message):
        validate_runtime_manifest(runtime, test_checkpoint_ack=None)


def test_official_runtime_rejects_nonloopback_endpoint() -> None:
    runtime = runtime_manifest("base")
    endpoint = "http://0.0.0.0:30000/v1/chat/completions"
    runtime["endpoint"] = endpoint
    runtime["server_args"]["endpoint"] = endpoint
    _refresh_runtime_hashes(runtime)
    with pytest.raises(ValueError, match="loopback"):
        validate_runtime_manifest(runtime, test_checkpoint_ack=None)


def test_official_runtime_rejects_dummy_weight_loading() -> None:
    runtime = runtime_manifest("base")
    runtime["server_args"]["load_format"] = "dummy"
    _refresh_runtime_hashes(runtime)

    with pytest.raises(ValueError, match="real safetensors weights"):
        validate_runtime_manifest(runtime, test_checkpoint_ack=None)


def test_post_json_ignores_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    target = _recording_server()
    proxy = _recording_server()
    try:
        target_endpoint = f"http://127.0.0.1:{target.server_port}/v1/chat/completions"
        proxy_endpoint = f"http://127.0.0.1:{proxy.server_port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_endpoint)
        monkeypatch.setenv("http_proxy", proxy_endpoint)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        # Make a default ProxyHandler use the trap even for loopback. The
        # production client must still connect directly through ProxyHandler({}).
        monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)

        assert post_json(
            target_endpoint,
            {"secret_prompt": "не отправлять прокси"},
            timeout=2.0,
            api_key=API_KEY,
        ) == {"ok": True}
        assert len(target.requests) == 1
        assert proxy.requests == []
    finally:
        target.shutdown()
        proxy.shutdown()
        target.server_close()
        proxy.server_close()


def test_post_json_rejects_redirect_without_forwarding_body_or_secret() -> None:
    trap = _recording_server()
    redirect_target = f"http://127.0.0.1:{trap.server_port}/v1/chat/completions"
    source = _recording_server(redirect_to=redirect_target)
    try:
        endpoint = f"http://127.0.0.1:{source.server_port}/v1/chat/completions"
        with pytest.raises(RequestFailure, match="HTTP 302"):
            post_json(
                endpoint,
                {"secret_prompt": "не пересылать по редиректу"},
                timeout=2.0,
                api_key=API_KEY,
            )
        assert len(source.requests) == 1
        assert source.requests[0]["authorization"] == f"Bearer {API_KEY}"
        assert trap.requests == []
    finally:
        source.shutdown()
        trap.shutdown()
        source.server_close()
        trap.server_close()


@pytest.mark.parametrize("location", ("top-level", "server-args", "adapter"))
def test_runtime_manifest_rejects_fields_that_can_smuggle_local_secrets_or_paths(
    location: str,
) -> None:
    runtime = runtime_manifest("adapter")
    if location == "top-level":
        runtime["api_key"] = API_KEY
    elif location == "server-args":
        runtime["server_args"]["api_key"] = API_KEY
        _refresh_runtime_hashes(runtime)
    else:
        runtime["adapter"]["path"] = "/unbound/adapter"

    with pytest.raises(ValueError, match="fields|unsupported"):
        validate_runtime_manifest(runtime, test_checkpoint_ack=None)


def test_generation_requires_exact_runtime_secret() -> None:
    runtime = runtime_manifest("base")
    source = source_row()
    decoding = decoding_contract(decoding_args())

    for key in (None, "wrong-key-that-is-at-least-32-bytes"):
        with pytest.raises(ValueError, match="API key differs"):
            generate_one(
                source,
                variant="base",
                runtime=runtime,
                runtime_sha256="a" * 64,
                official=True,
                decoding=decoding,
                endpoint=runtime["endpoint"],
                api_key=key,
                timeout=30.0,
                retries=0,
                request_fn=lambda *args, **kwargs: response(),
            )


def test_han_modes_fail_closed_and_clusters_group_variants() -> None:
    spontaneous = source_row()
    assert derive_han_evaluation_mode(spontaneous, input_han_count=0) == "spontaneous"

    cleanup = deepcopy(spontaneous)
    cleanup["tags"] = ["han-cleanup"]
    assert derive_han_evaluation_mode(cleanup, input_han_count=1) == "input_conditioned_cleanup"
    scope = deepcopy(spontaneous)
    scope["tags"] = ["accidental-han-control", "han-in-code"]
    assert derive_han_evaluation_mode(scope, input_han_count=1) == "input_conditioned_scope_control"
    allowed = deepcopy(spontaneous)
    allowed["contract"]["requested_language"] = "zh-CN"
    assert derive_han_evaluation_mode(allowed, input_han_count=1) == "excluded_han_allowed"

    with pytest.raises(ValueError, match="must be tagged"):
        derive_han_evaluation_mode(spontaneous, input_han_count=1)
    cleanup["tags"].append("accidental-han-control")
    with pytest.raises(ValueError, match="both Han cleanup and scope control"):
        derive_han_evaluation_mode(cleanup, input_han_count=1)

    variant = deepcopy(spontaneous)
    variant["id"] = "quality-example-variant"
    variant["provenance"]["source_record_id"] = "quality-example-variant"
    assert evaluation_cluster_id(variant) == evaluation_cluster_id(spontaneous)

    wikipedia = deepcopy(spontaneous)
    wikipedia["provenance"] = {
        "dataset": "wikimedia/wikipedia",
        "source_text_sha256": "d" * 64,
    }
    assert evaluation_cluster_id(wikipedia) == f"wikipedia-source:{'d' * 64}"


def test_resume_validates_rows_token_accounting_and_output_manifest(
    tmp_path: Path,
) -> None:
    runtime = runtime_manifest("base")
    source = source_row()
    decoding = decoding_contract(decoding_args())
    row = generate_one(
        source,
        variant="base",
        runtime=runtime,
        runtime_sha256="a" * 64,
        official=True,
        decoding=decoding,
        endpoint=runtime["endpoint"],
        api_key=API_KEY,
        timeout=30.0,
        retries=0,
        request_fn=lambda *args, **kwargs: response(),
    )
    output = tmp_path / "base.jsonl"
    output.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    contracts_path = tmp_path / "contracts.jsonl"
    contracts_path.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = output_manifest_contract(
        status="QUALITY-OUTPUTS-IN-PROGRESS",
        count=1,
        output_sha256=file_sha256(output),
        variant="base",
        split="validation",
        official=True,
        runtime_sha256="a" * 64,
        runtime=runtime,
        decoding=decoding,
        contracts_paths=[contracts_path],
    )
    validate_existing_output_manifest(
        manifest,
        output_path=output,
        rows=[row],
        variant="base",
        split="validation",
        official=True,
        runtime_sha256="a" * 64,
        runtime=runtime,
        decoding=decoding,
        contracts_paths=[contracts_path],
    )

    for field in (
        "variant",
        "split",
        "count",
        "runtime_manifest_sha256",
        "api_secret_sha256",
        "decoding_contract_sha256",
        "contract_artifacts_sha256",
        "output_sha256",
    ):
        broken = deepcopy(manifest)
        broken[field] = 999 if field == "count" else "0" * 64
        with pytest.raises(ValueError, match="existing output manifest differs"):
            validate_existing_output_manifest(
                broken,
                output_path=output,
                rows=[row],
                variant="base",
                split="validation",
                official=True,
                runtime_sha256="a" * 64,
                runtime=runtime,
                decoding=decoding,
                contracts_paths=[contracts_path],
            )

    token_drift = deepcopy(row)
    token_drift["generation"]["completion_tokens"] += 1
    with pytest.raises(ValueError, match="token accounting"):
        validate_existing_rows(
            [token_drift],
            {source["id"]: source},
            variant="base",
            runtime=runtime,
            runtime_sha256="a" * 64,
            official=True,
            decoding=decoding,
        )
