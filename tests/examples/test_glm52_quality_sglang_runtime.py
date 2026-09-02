from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import file_sha256  # noqa: E402
from build_quality_sglang_runtime import (  # noqa: E402
    FULL_TRAINABLE_PARAMETERS,
    validate_server_args,
)
from build_quality_sglang_runtime import main as build_main  # noqa: E402
from generate_full_quality_outputs_sglang import (  # noqa: E402
    OFFICIAL_INFERENCE_BASES,
    OFFICIAL_TRAINER,
    validate_runtime_manifest,
)
from launch_quality_sglang_server import build_server_kwargs  # noqa: E402


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
    write_json(snapshot / "model.safetensors.index.json", {"weight_map": {}})
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
            "base_model_name_or_path": str(trainer),
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
    weights.write_bytes(b"verified-adapter")
    verification = root / "adapter_verification.json"
    write_json(
        verification,
        {
            "adapter_dir": str(adapter),
            "serialized_sha256": file_sha256(weights),
            "all_lora_b_nonzero": True,
            "parameter_count": FULL_TRAINABLE_PARAMETERS["mla-only"],
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
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "test"], check=True)
    return checkout


def test_builder_hashes_real_trainer_and_inference_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_snapshot(tmp_path / "trainer", OFFICIAL_TRAINER[1], fp8=False)
    inference_identity = next(
        identity for identity in OFFICIAL_INFERENCE_BASES if identity != OFFICIAL_TRAINER
    )
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
            "--output",
            str(output),
        ],
    )

    build_main()

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert validate_runtime_manifest(manifest, test_checkpoint_ack=None) is True
    assert manifest["trainer_base"]["config_sha256"] == file_sha256(
        trainer / "config.json"
    )
    assert manifest["trainer_base"]["weights_index_sha256"] == file_sha256(
        trainer / "model.safetensors.index.json"
    )
    assert manifest["inference_base"]["config_sha256"] == file_sha256(
        inference / "config.json"
    )


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
            model_path=model.resolve(),
            adapter_path=adapter.resolve(),
            adapter_name="quality",
            target_modules=["q_a_proj"],
            official=False,
            inference_model_id="test/model",
        )


def test_launcher_builds_only_hashed_server_arguments() -> None:
    manifest = {
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
