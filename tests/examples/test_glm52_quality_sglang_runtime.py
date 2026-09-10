from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import canonical_sha256, file_sha256  # noqa: E402
from build_quality_sglang_runtime import (  # noqa: E402
    TRUSTED_WEIGHT_SHARD_MANIFESTS,
    validate_adapter,
    validate_model_snapshot,
    validate_server_args,
    validate_sglang_checkout,
    validate_weight_shard_identity,
)
from build_quality_sglang_runtime import main as build_main  # noqa: E402
from generate_full_quality_outputs_sglang import (  # noqa: E402
    OFFICIAL_INFERENCE_BASES,
    OFFICIAL_MODEL_ARTIFACTS,
    OFFICIAL_TRAINER,
    TEST_ACK,
    build_pair_runtime_contract,
    secret_sha256,
    validate_runtime_manifest,
)
from launch_quality_sglang_server import (  # noqa: E402
    build_server_kwargs,
    validate_gpu_lease_attestation,
    validate_live_environment,
    validate_sglang_import_origins,
)

API_KEY = "runtime-test-key-0123456789abcdef0123456789"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_snapshot(
    root: Path,
    revision: str,
    *,
    fp8: bool,
) -> Path:
    snapshot = root / revision
    config = {"model_type": "glm_moe_dsa", "num_hidden_layers": 78}
    if fp8:
        config["quantization_config"] = {"quant_method": "fp8"}
    write_json(snapshot / "config.json", config)
    write_json(
        snapshot / "model.safetensors.index.json",
        {"metadata": {"total_size": 7}, "weight_map": {"weight": "model.safetensors"}},
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("tokenizer-config", encoding="utf-8")
    (snapshot / "chat_template.jinja").write_text("chat-template", encoding="utf-8")
    return snapshot


def make_adapter(root: Path, trainer: Path) -> tuple[Path, Path]:
    adapter = root / "adapter"
    write_json(
        adapter / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "base_model_name_or_path": "portable/model@immutable-revision",
            "target_modules": [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
            ],
        },
    )
    weights = adapter / "adapter_model.safetensors"
    tensors = {}
    for target in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    ):
        prefix = f"base_model.model.model.layers.0.self_attn.{target}"
        tensors[f"{prefix}.lora_A.default.weight"] = torch.ones((16, 2), dtype=torch.bfloat16)
        tensors[f"{prefix}.lora_B.default.weight"] = torch.ones((3, 16), dtype=torch.bfloat16)
    save_file(tensors, weights)
    verification = root / "adapter_verification.json"
    write_json(
        verification,
        {
            "adapter_dir": "/obsolete/machine/local/path",
            "serialized_sha256": file_sha256(weights),
            "all_lora_b_nonzero": True,
            "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        },
    )
    return adapter, verification


def make_clean_checkout(root: Path) -> Path:
    checkout = root / "sglang"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    (checkout / "README.md").write_text("test\n", encoding="utf-8")
    for relative_name in (
        "python/sglang/launch_server.py",
        "python/sglang/srt/server_args.py",
        "python/sglang/srt/models/glm4_moe.py",
        "python/sglang/srt/models/deepseek_v2.py",
        "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py",
        "python/sglang/srt/lora/lora_manager.py",
        "python/sglang/srt/lora/lora_registry.py",
    ):
        source = checkout / relative_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {relative_name}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "test"], check=True)
    return checkout


def test_adapter_contract_is_independent_of_machine_local_paths(tmp_path: Path) -> None:
    blocks = []
    for index in (1, 2):
        adapter, verification = make_adapter(tmp_path / f"machine-{index}", tmp_path)
        config_path = adapter / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["base_model_name_or_path"] = f"/machine-{index}/models/glm52"
        write_json(config_path, config)
        proof = json.loads(verification.read_text(encoding="utf-8"))
        proof["adapter_dir"] = f"/machine-{index}/runs/adapter"
        write_json(verification, proof)
        _, block = validate_adapter(
            adapter,
            verification,
            name="glm52-quality-mla-r16",
            profile="mla-only",
            trainer_revision=OFFICIAL_TRAINER[1],
            official=False,
        )
        blocks.append(block)

    assert blocks[0] == blocks[1]


def test_official_adapter_requires_exact_trainer_shard_provenance(
    tmp_path: Path,
) -> None:
    adapter, verification_path = make_adapter(tmp_path, tmp_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    trainer_model = {
        "model_id": OFFICIAL_TRAINER[0],
        "revision": OFFICIAL_TRAINER[1],
        "config_sha256": "1" * 64,
        "weights_index_sha256": "2" * 64,
    }
    verification["training_provenance"] = {
        **trainer_model,
        "weight_shard_manifest_sha256": "3" * 64,
    }
    write_json(verification_path, verification)

    # The topology is intentionally tiny, so a correct provenance block reaches
    # the later full-78-layer check rather than failing the provenance gate.
    with pytest.raises(ValueError, match="contiguous and complete"):
        validate_adapter(
            adapter,
            verification_path,
            name="test-adapter",
            profile="mla-only",
            trainer_revision=OFFICIAL_TRAINER[1],
            official=True,
            trainer_model=trainer_model,
            trainer_weight_shard_manifest_sha256="3" * 64,
        )

    verification["training_provenance"]["weight_shard_manifest_sha256"] = "4" * 64
    write_json(verification_path, verification)
    with pytest.raises(ValueError, match="does not bind its trainer base shards"):
        validate_adapter(
            adapter,
            verification_path,
            name="test-adapter",
            profile="mla-only",
            trainer_revision=OFFICIAL_TRAINER[1],
            official=True,
            trainer_model=trainer_model,
            trainer_weight_shard_manifest_sha256="3" * 64,
        )


def test_builder_hashes_real_trainer_and_inference_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_snapshot(tmp_path / "trainer", OFFICIAL_TRAINER[1], fp8=False)
    inference_identity = next(identity for identity in OFFICIAL_INFERENCE_BASES if identity != OFFICIAL_TRAINER)
    inference = make_snapshot(tmp_path / "inference", inference_identity[1], fp8=True)
    adapter, verification = make_adapter(tmp_path, trainer)
    checkout = make_clean_checkout(tmp_path)
    server_args_path = tmp_path / "server_args.json"
    write_json(
        server_args_path,
        {
            "model_path": str(inference),
            "served_base_model": "glm52-base",
            "enable_lora": True,
            "lora_strict_loading": True,
            "lora_paths": {"glm52-quality": str(adapter)},
            "max_lora_rank": 16,
            "lora_target_modules": [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
            ],
            "ld_library_paths": [],
            "tp_size": 8,
            "gpu_ids": list(range(8)),
            "max_model_len": 2048,
            "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
        },
    )
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_quality_sglang_runtime.py",
            "--runtime-mode",
            "adapter",
            "--trainer-model-path",
            str(trainer),
            "--model-path",
            str(inference),
            "--adapter-path",
            str(adapter),
            "--adapter-verification",
            str(verification),
            "--adapter-name",
            "glm52-quality",
            "--profile",
            "mla-only",
            "--sglang-checkout",
            str(checkout),
            "--server-args",
            str(server_args_path),
            "--server-instance-id",
            "glm52-quality-test",
            "--trainer-model-id",
            "test/glm52-trainer",
            "--inference-model-id",
            "test/glm52-fp8",
            "--test-checkpoint-ack",
            TEST_ACK,
            "--output",
            str(output),
        ],
    )

    build_main()

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert validate_runtime_manifest(manifest, test_checkpoint_ack=TEST_ACK) is False
    assert manifest["trainer_base"]["config_sha256"] == file_sha256(trainer / "config.json")
    assert manifest["trainer_base"]["weights_index_sha256"] == file_sha256(trainer / "model.safetensors.index.json")
    assert manifest["inference_base"]["config_sha256"] == file_sha256(inference / "config.json")
    assert manifest["runtime_mode"] == "adapter"
    assert "path" not in manifest["adapter"]
    assert manifest["local_artifacts"]["adapter_path"] == str(adapter.resolve())
    assert manifest["pair_runtime_contract"] == build_pair_runtime_contract(manifest)


def test_builder_can_hash_base_runtime_before_adapter_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_snapshot(tmp_path / "trainer", OFFICIAL_TRAINER[1], fp8=False)
    inference_identity = next(identity for identity in OFFICIAL_INFERENCE_BASES if identity != OFFICIAL_TRAINER)
    inference = make_snapshot(tmp_path / "inference", inference_identity[1], fp8=True)
    checkout = make_clean_checkout(tmp_path)
    server_args_path = tmp_path / "base_server_args.json"
    write_json(
        server_args_path,
        {
            "model_path": str(inference),
            "served_base_model": "glm52-base",
            "enable_lora": False,
            "ld_library_paths": [],
            "tp_size": 8,
            "gpu_ids": list(range(8)),
            "max_model_len": 2048,
            "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
        },
    )
    output = tmp_path / "base_runtime.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_quality_sglang_runtime.py",
            "--runtime-mode",
            "base",
            "--trainer-model-path",
            str(trainer),
            "--model-path",
            str(inference),
            "--sglang-checkout",
            str(checkout),
            "--server-args",
            str(server_args_path),
            "--server-instance-id",
            "glm52-base-test",
            "--trainer-model-id",
            "test/glm52-trainer",
            "--inference-model-id",
            "test/glm52-fp8",
            "--test-checkpoint-ack",
            TEST_ACK,
            "--output",
            str(output),
        ],
    )

    build_main()

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert validate_runtime_manifest(manifest, test_checkpoint_ack=TEST_ACK) is False
    assert manifest["runtime_mode"] == "base"
    assert manifest["adapter"] is None
    assert manifest["server_args"]["enable_lora"] is False
    assert "lora_paths" not in manifest["server_args"]
    assert build_server_kwargs(manifest)["enable_lora"] is False
    assert "lora_paths" not in build_server_kwargs(manifest)


def test_server_args_reject_noninteger_gpu_ids(tmp_path: Path) -> None:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    value = {
        "model_path": str(model),
        "enable_lora": True,
        "lora_strict_loading": True,
        "lora_paths": {"quality": str(adapter)},
        "max_lora_rank": 16,
        "lora_target_modules": ["q_a_proj"],
        "ld_library_paths": [],
        "tp_size": 1,
        "gpu_ids": ["5"],
        "max_model_len": 2048,
        "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
    }

    with pytest.raises(ValueError, match="gpu_ids"):
        validate_server_args(
            value,
            runtime_mode="adapter",
            model_path=model.resolve(),
            adapter_path=adapter.resolve(),
            adapter_name="quality",
            target_modules=["q_a_proj"],
            official=False,
            inference_model_id="test/model",
        )


def test_launcher_builds_only_hashed_server_arguments() -> None:
    manifest = {
        "runtime_mode": "adapter",
        "served_base_model": "glm52-base",
        "server_args": {
            "model_path": "/models/glm52",
            "served_base_model": "glm52-base",
            "endpoint": "http://127.0.0.1:30152/v1/chat/completions",
            "tp_size": 1,
            "gpu_ids": [5],
            "enable_lora": True,
            "lora_strict_loading": True,
            "lora_paths": {"quality": "/adapters/quality"},
            "max_lora_rank": 16,
            "lora_target_modules": ["q_a_proj"],
            "ld_library_paths": [],
            "max_model_len": 2048,
            "attention_backend": "flashinfer",
            "dsa_prefill_backend": "torch",
        },
    }

    assert build_server_kwargs(manifest) == {
        "model_path": "/models/glm52",
        "served_model_name": "glm52-base",
        "host": "127.0.0.1",
        "port": 30152,
        "tp_size": 1,
        "enable_lora": True,
        "lora_strict_loading": True,
        "lora_paths": {"quality": "/adapters/quality"},
        "max_lora_rank": 16,
        "lora_target_modules": ["q_a_proj"],
        "context_length": 2048,
        "attention_backend": "flashinfer",
        "dsa_prefill_backend": "torch",
    }

    manifest["server_args"]["unhashed_escape_hatch"] = True
    with pytest.raises(ValueError, match="unsupported server argument"):
        build_server_kwargs(manifest)


def test_official_artifact_constants_are_exact_and_manifest_tampering_fails() -> None:
    bf16 = OFFICIAL_MODEL_ARTIFACTS[OFFICIAL_TRAINER]
    fp8_identity = next(identity for identity in OFFICIAL_INFERENCE_BASES if identity != OFFICIAL_TRAINER)
    fp8 = OFFICIAL_MODEL_ARTIFACTS[fp8_identity]
    assert bf16 == {
        "config_sha256": "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
        "weights_index_sha256": "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
        "tokenizer_json_sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer_config_sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
        "chat_template_sha256": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
        "weight_count": 59_585,
        "shard_count": 282,
        "index_total_size": 1_506_659_919_872,
        "shard_bytes_on_disk": 1_506_667_387_408,
    }
    assert fp8["config_sha256"] == "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff"
    assert fp8["weights_index_sha256"] == "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
    assert fp8["shard_count"] == 141
    assert fp8["index_total_size"] == 755_617_140_416
    assert fp8["shard_bytes_on_disk"] == 755_632_050_320


def test_snapshot_rejects_empty_missing_and_zero_shards(tmp_path: Path) -> None:
    revision = "1" * 40
    snapshot = make_snapshot(tmp_path, revision, fp8=False)
    index_path = snapshot / "model.safetensors.index.json"

    write_json(index_path, {"metadata": {"total_size": 7}, "weight_map": {}})
    with pytest.raises(ValueError, match="empty or invalid weight_map"):
        validate_model_snapshot(
            snapshot,
            model_id="test/model",
            revision=revision,
            official=False,
            label="test model",
        )

    write_json(
        index_path,
        {
            "metadata": {"total_size": 7},
            "weight_map": {"weight": "missing.safetensors"},
        },
    )
    with pytest.raises(FileNotFoundError, match="missing 1 index-referenced"):
        validate_model_snapshot(
            snapshot,
            model_id="test/model",
            revision=revision,
            official=False,
            label="test model",
        )

    write_json(
        index_path,
        {"metadata": {"total_size": 7}, "weight_map": {"weight": "empty.safetensors"}},
    )
    (snapshot / "empty.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty shard"):
        validate_model_snapshot(
            snapshot,
            model_id="test/model",
            revision=revision,
            official=False,
            label="test model",
        )


def test_official_snapshot_rejects_aggregate_or_small_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = OFFICIAL_TRAINER[1]
    snapshot = make_snapshot(tmp_path, revision, fp8=False)
    monkeypatch.setattr(
        "build_quality_sglang_runtime.OFFICIAL_MODEL_ARTIFACTS",
        {},
    )
    _, observed = validate_model_snapshot(
        snapshot,
        model_id=OFFICIAL_TRAINER[0],
        revision=revision,
        official=True,
        label="trainer model",
    )
    expected = {field: observed[field] for field in OFFICIAL_MODEL_ARTIFACTS[OFFICIAL_TRAINER]}
    expected["shard_bytes_on_disk"] += 1
    monkeypatch.setattr(
        "build_quality_sglang_runtime.OFFICIAL_MODEL_ARTIFACTS",
        {OFFICIAL_TRAINER: expected},
    )
    with pytest.raises(ValueError, match="locked official artifact contract"):
        validate_model_snapshot(
            snapshot,
            model_id=OFFICIAL_TRAINER[0],
            revision=revision,
            official=True,
            label="trainer model",
        )


def test_sglang_checkout_rejects_untracked_files(tmp_path: Path) -> None:
    checkout = make_clean_checkout(tmp_path)
    validate_sglang_checkout(checkout)
    (checkout / "untracked.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="including untracked"):
        validate_sglang_checkout(checkout)


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://0.0.0.0:30000/v1/chat/completions",
        "http://192.168.1.5:30000/v1/chat/completions",
        "http://example.test:30000/v1/chat/completions",
        "http://127.0.0.1:30000/v1/chat/completions?escape=1",
    ),
)
def test_server_args_reject_nonloopback_or_ambiguous_endpoint(
    tmp_path: Path,
    endpoint: str,
) -> None:
    model = tmp_path / "model"
    value = {
        "model_path": str(model),
        "served_base_model": "test",
        "enable_lora": False,
        "ld_library_paths": [],
        "tp_size": 1,
        "gpu_ids": [0],
        "max_model_len": 2048,
        "endpoint": endpoint,
    }
    with pytest.raises(ValueError, match="loopback"):
        validate_server_args(
            value,
            runtime_mode="base",
            model_path=model.resolve(),
            adapter_path=None,
            adapter_name=None,
            target_modules=None,
            official=False,
            inference_model_id="test/model",
        )


def test_launcher_requires_exact_secret_without_storing_it() -> None:
    manifest = {
        "runtime_mode": "base",
        "served_base_model": "test",
        "api_secret_sha256": secret_sha256(API_KEY),
        "server_args": {
            "model_path": "/models/test",
            "served_base_model": "test",
            "endpoint": "http://127.0.0.1:30152/v1/chat/completions",
            "tp_size": 1,
            "gpu_ids": [0],
            "enable_lora": False,
            "ld_library_paths": [],
            "max_model_len": 2048,
        },
    }
    with pytest.raises(ValueError, match="requires its bound API key"):
        build_server_kwargs(manifest)
    with pytest.raises(ValueError, match="differs from its commitment"):
        build_server_kwargs(manifest, api_key="wrong-key-that-is-at-least-32-bytes")
    kwargs = build_server_kwargs(manifest, api_key=API_KEY)
    assert kwargs["api_key"] == API_KEY
    assert API_KEY not in json.dumps(manifest)


def build_live_test_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path, Path, Path]:
    revision = "1" * 40
    trainer = make_snapshot(tmp_path / "trainer", revision, fp8=False)
    inference = make_snapshot(tmp_path / "inference", revision, fp8=True)
    adapter, verification = make_adapter(tmp_path, trainer)
    checkout = make_clean_checkout(tmp_path)
    server_args_path = tmp_path / "server_args.json"
    write_json(
        server_args_path,
        {
            "model_path": str(inference),
            "served_base_model": "test-model",
            "enable_lora": True,
            "lora_strict_loading": True,
            "lora_paths": {"test-adapter": str(adapter)},
            "max_lora_rank": 16,
            "lora_target_modules": [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
            ],
            "ld_library_paths": [],
            "tp_size": 1,
            "gpu_ids": [5],
            "max_model_len": 2048,
            "endpoint": "http://127.0.0.1:30000/v1/chat/completions",
        },
    )
    output = tmp_path / "runtime.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_quality_sglang_runtime.py",
            "--runtime-mode",
            "adapter",
            "--trainer-model-path",
            str(trainer),
            "--model-path",
            str(inference),
            "--adapter-path",
            str(adapter),
            "--adapter-verification",
            str(verification),
            "--adapter-name",
            "test-adapter",
            "--profile",
            "mla-only",
            "--sglang-checkout",
            str(checkout),
            "--server-args",
            str(server_args_path),
            "--server-instance-id",
            "test-instance",
            "--trainer-model-id",
            "test/trainer",
            "--trainer-revision",
            revision,
            "--inference-model-id",
            "test/inference",
            "--inference-revision",
            revision,
            "--test-checkpoint-ack",
            TEST_ACK,
            "--output",
            str(output),
        ],
    )
    build_main()
    return json.loads(output.read_text(encoding="utf-8")), checkout, inference, adapter


@pytest.mark.parametrize("drift", ("git", "code", "environment", "model", "adapter"))
def test_launch_revalidates_every_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    manifest, checkout, inference, adapter = build_live_test_manifest(tmp_path, monkeypatch)
    monkeypatch.setenv("PYTHONPATH", str((checkout / "python").resolve()))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    validate_live_environment(manifest, official=False)

    if drift == "git":
        (checkout / "untracked.py").write_text("pass\n", encoding="utf-8")
        message = "including untracked"
    elif drift == "code":
        manifest["code_artifacts"]["runtime_scripts"]["launch_quality_sglang_server.py"]["sha256"] = "0" * 64
        message = "code artifact differs"
    elif drift == "environment":
        manifest["environment_artifacts"]["installed_distributions_sha256"] = "0" * 64
        message = "Python environment differs"
    elif drift == "model":
        with (inference / "config.json").open("a", encoding="utf-8") as output:
            output.write(" \n")
        message = "live inference model differs"
    else:
        with (adapter / "adapter_model.safetensors").open("ab") as output:
            output.write(b"tampered")
        message = "adapter weights differ"

    with pytest.raises(ValueError, match=message):
        validate_live_environment(manifest, official=False)


def test_sglang_import_origins_must_resolve_inside_bound_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "sglang"
    expected = {
        "sglang": checkout / "python/sglang/__init__.py",
        "sglang.launch_server": checkout / "python/sglang/launch_server.py",
        "sglang.srt.server_args": checkout / "python/sglang/srt/server_args.py",
    }
    for path in expected.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test source\n", encoding="utf-8")
    modules = {name: SimpleNamespace(__file__=str(path)) for name, path in expected.items()}

    validate_sglang_import_origins(checkout, modules)

    modules["sglang.launch_server"].__file__ = str(tmp_path / "shadow/launch_server.py")
    with pytest.raises(ValueError, match="outside the bound SGLang checkout"):
        validate_sglang_import_origins(checkout, modules)


def test_sglang_import_origins_require_complete_module_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="module set is incomplete"):
        validate_sglang_import_origins(tmp_path, {})


def test_weight_shards_require_trusted_manifest_then_full_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    snapshot = make_snapshot(tmp_path / "model", revision, fp8=False)
    _, model = validate_model_snapshot(
        snapshot,
        model_id="test/model",
        revision=revision,
        official=False,
        label="test model",
    )
    manifest = {
        "schema_version": 1,
        "model_id": "test/model",
        "revision": revision,
        "weights_index_sha256": model["weights_index_sha256"],
        "hash_algorithm": "sha256",
        "shards": [
            {
                "filename": "model.safetensors",
                "size": 7,
                "sha256": file_sha256(snapshot / "model.safetensors"),
            }
        ],
    }
    manifest_path = tmp_path / "trusted-shards.json"
    write_json(manifest_path, manifest)
    TRUSTED_WEIGHT_SHARD_MANIFESTS[("test/model", revision)] = canonical_sha256(manifest)
    cache = tmp_path / "verification-cache"

    pending = validate_weight_shard_identity(
        snapshot,
        model,
        manifest_path=manifest_path,
        cache_dir=cache,
        verify_full=False,
        label="test model",
    )
    assert pending["status"] == "PENDING-LOCAL-FULL-READ"
    verified = validate_weight_shard_identity(
        snapshot,
        model,
        manifest_path=manifest_path,
        cache_dir=cache,
        verify_full=True,
        label="test model",
    )
    assert verified["status"] == "VERIFIED"

    monkeypatch.setattr(
        "build_quality_sglang_runtime._sha256_stream",
        lambda path: (_ for _ in ()).throw(AssertionError("must not rehash")),
    )
    assert (
        validate_weight_shard_identity(
            snapshot,
            model,
            manifest_path=manifest_path,
            cache_dir=cache,
            verify_full=False,
            label="test model",
        )
        == verified
    )
    os.utime(snapshot / "model.safetensors", None)
    with pytest.raises(ValueError, match="stale or invalid"):
        validate_weight_shard_identity(
            snapshot,
            model,
            manifest_path=manifest_path,
            cache_dir=cache,
            verify_full=False,
            label="test model",
        )


def test_weight_shard_receipt_supports_huggingface_cache_symlinks(
    tmp_path: Path,
) -> None:
    revision = "b" * 40
    snapshot = make_snapshot(tmp_path / "model", revision, fp8=False)
    shard = snapshot / "model.safetensors"
    blob = tmp_path / "hub" / "blobs" / ("c" * 64)
    blob.parent.mkdir(parents=True)
    shard.rename(blob)
    shard.symlink_to(blob)
    _, model = validate_model_snapshot(
        snapshot,
        model_id="test/symlink-model",
        revision=revision,
        official=False,
        label="test model",
    )
    manifest = {
        "schema_version": 1,
        "model_id": "test/symlink-model",
        "revision": revision,
        "weights_index_sha256": model["weights_index_sha256"],
        "hash_algorithm": "sha256",
        "shards": [
            {
                "filename": shard.name,
                "size": blob.stat().st_size,
                "sha256": file_sha256(blob),
            }
        ],
    }
    manifest_path = tmp_path / "trusted-symlink-shards.json"
    write_json(manifest_path, manifest)
    TRUSTED_WEIGHT_SHARD_MANIFESTS[("test/symlink-model", revision)] = canonical_sha256(manifest)

    verified = validate_weight_shard_identity(
        snapshot,
        model,
        manifest_path=manifest_path,
        cache_dir=tmp_path / "verification-cache",
        verify_full=True,
        label="test model",
    )

    assert verified["status"] == "VERIFIED"


@pytest.mark.parametrize("corruption", ("nan", "zero-b", "missing"))
def test_adapter_validation_uses_tensor_contents_not_claimed_boolean(tmp_path: Path, corruption: str) -> None:
    adapter, verification = make_adapter(tmp_path, tmp_path)
    weights = adapter / "adapter_model.safetensors"
    tensors = load_file(weights)
    target_key = next(iter(sorted(tensors)))
    if corruption == "nan":
        tensors[target_key][0, 0] = float("nan")
        message = "non-finite"
    elif corruption == "zero-b":
        target_key = next(key for key in tensors if ".lora_B." in key)
        tensors[target_key].zero_()
        message = "all zero"
    else:
        tensors.pop(target_key)
        message = "missing"
    save_file(tensors, weights)
    proof = json.loads(verification.read_text(encoding="utf-8"))
    proof["serialized_sha256"] = file_sha256(weights)
    proof["all_lora_b_nonzero"] = True
    proof["parameter_count"] = sum(tensor.numel() for tensor in tensors.values())
    write_json(verification, proof)
    with pytest.raises(ValueError, match=message):
        validate_adapter(
            adapter,
            verification,
            name="test-adapter",
            profile="mla-only",
            trainer_revision="a" * 40,
            official=False,
        )


def test_gpu_lease_is_owner_only_runtime_and_hardware_bound(tmp_path: Path) -> None:
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    lease = {
        "schema_version": 1,
        "status": "ACTIVE",
        "lease_id": "communal-gpu-lease-123",
        "owner_uid": os.geteuid(),
        "host": "shared.example",
        "runtime_manifest_sha256": "a" * 64,
        "gpu_ids": [5],
        "gpu_uuids": ["GPU-uuid-5"],
        "issued_at_utc": "2026-09-04T11:55:00Z",
        "expires_at_utc": "2026-09-04T12:30:00Z",
        "shared_host_acknowledged": True,
        "foreign_processes_checked": True,
    }
    path = tmp_path / "gpu-lease.json"
    write_json(path, lease)
    path.chmod(0o600)
    assert (
        validate_gpu_lease_attestation(
            path,
            runtime_manifest_sha256="a" * 64,
            gpu_ids=[5],
            now=now,
            live_inventory={5: "GPU-uuid-5"},
            hostname="shared.example",
        )
        == lease
    )

    lease["expires_at_utc"] = "2026-09-04T11:59:00Z"
    write_json(path, lease)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="expired"):
        validate_gpu_lease_attestation(
            path,
            runtime_manifest_sha256="a" * 64,
            gpu_ids=[5],
            now=now,
            live_inventory={5: "GPU-uuid-5"},
            hostname="shared.example",
        )


def test_gpu_lease_file_must_not_be_group_readable(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    write_json(path, {})
    path.chmod(0o640)
    with pytest.raises(ValueError, match="owner-only"):
        validate_gpu_lease_attestation(
            path,
            runtime_manifest_sha256="a" * 64,
            gpu_ids=[5],
            live_inventory={5: "GPU-uuid-5"},
        )
