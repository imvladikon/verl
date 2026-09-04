# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(SCRIPTS))

import build_glm52_surgery_dummy as builder
from glm52_surgery_io import DigestWriter, HubRangeReader, TensorLocation
from plan_glm52_surgery_pair import (
    DEFAULT_LAYER_MAP,
    surgery_pair_id,
    target_config,
    validate_sources,
)
from verify_glm52_surgery_pair import verify_built, verify_pair_contract


def source_config() -> dict:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "vocab_size": 154880,
        "num_hidden_layers": 78,
        "num_nextn_predict_layers": 1,
        "torch_dtype": None,
        "rms_norm_eps": 1e-5,
        "indexer_types": ["full"] * 78,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "layer_types": ["full_attention"] * 78,
    }


def fp8_source_config() -> dict:
    config = source_config()
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "lm_head",
            "model.layers.0.input_layernorm",
            "model.layers.77.post_attention_layernorm",
        ],
    }
    return config


def test_planner_and_verifier_have_only_checked_in_dependencies() -> None:
    for filename in (
        "plan_glm52_surgery_pair.py",
        "verify_glm52_surgery_pair.py",
        "build_glm52_surgery_dummy.py",
    ):
        source = (SCRIPTS / filename).read_text(encoding="utf-8")
        assert "plan_glm53_flash_9b_surgery" not in source
        assert "verify_glm53_flash_9b_surgery" not in source
    assert (SCRIPTS / "glm52_surgery_io.py").is_file()


def test_source_config_comparison_is_strict_outside_quantization() -> None:
    bf16 = source_config()
    fp8 = fp8_source_config()
    validate_sources(bf16, fp8, DEFAULT_LAYER_MAP)
    fp8["rms_norm_eps"] = 2e-5
    with pytest.raises(
        ValueError,
        match=r"(?s)outside quantization_config.*config[.]rms_norm_eps",
    ):
        validate_sources(bf16, fp8, DEFAULT_LAYER_MAP)


def test_target_config_remaps_all_per_layer_arrays_and_exclusions() -> None:
    target = target_config(fp8_source_config(), DEFAULT_LAYER_MAP, precision="fp8-rollout")
    assert target["num_hidden_layers"] == 5
    assert target["indexer_types"] == ["full"] * 5
    assert target["mlp_layer_types"] == ["dense"] * 3 + ["sparse"] * 2
    assert target["layer_types"] == ["full_attention"] * 5
    assert target["quantization_config"]["modules_to_not_convert"] == [
        "lm_head",
        "model.layers.0.input_layernorm",
    ]
    bf16_target = target_config(source_config(), DEFAULT_LAYER_MAP, precision="bf16")
    assert "quantization_config" not in bf16_target


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        content_range: str,
        *,
        status: int = 206,
        content_length: int | None = None,
        content_encoding: str | None = None,
    ):
        self.payload = BytesIO(payload)
        self.status = status
        self.headers = {
            "Content-Range": content_range,
            "Content-Length": str(len(payload) if content_length is None else content_length),
        }
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.payload.read(size)


def test_range_reader_validates_content_range_and_records_receipt() -> None:
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse(b"abc", "bytes 10-12/100")

    reader = HubRangeReader(
        "owner/model",
        "a" * 40,
        {"weight_map": {"weight": "model.safetensors"}},
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    output = BytesIO()
    writer = DigestWriter(output)
    reader.copy_range("model.safetensors", 10, 13, writer)
    assert output.getvalue() == b"abc"
    assert requests[0][0].get_header("Range") == "bytes=10-12"
    assert requests[0][0].get_header("Accept-encoding") == "identity"
    assert reader.remote_bytes == 3
    assert reader.range_receipts == [
        {
            "repository": "owner/model",
            "revision": "a" * 40,
            "shard": "model.safetensors",
            "start": 10,
            "end_exclusive": 13,
            "requested_end_exclusive": 13,
            "content_range": "bytes 10-12/100",
            "reported_total_bytes": 100,
            "bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
            "complete_response": True,
        }
    ]


def test_range_reader_resumes_truncated_response_with_contiguous_receipts() -> None:
    responses = iter(
        [
            FakeResponse(b"ab", "bytes 10-12/100", content_length=3),
            FakeResponse(b"c", "bytes 12-12/100"),
        ]
    )
    reader = HubRangeReader(
        "owner/model",
        "b" * 40,
        {"weight_map": {"weight": "model.safetensors"}},
        opener=lambda _request, timeout: next(responses),
        sleeper=lambda _seconds: None,
    )
    assert reader.read_range("model.safetensors", 10, 13) == b"abc"
    assert [receipt["complete_response"] for receipt in reader.range_receipts] == [
        False,
        True,
    ]
    assert [(receipt["start"], receipt["end_exclusive"]) for receipt in reader.range_receipts] == [(10, 12), (12, 13)]


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(b"abc", "", status=200),
        FakeResponse(b"abc", "bytes 9-11/100"),
        FakeResponse(b"abc", "bytes 10-13/100"),
    ],
)
def test_range_reader_rejects_unproven_source_ranges(response: FakeResponse) -> None:
    reader = HubRangeReader(
        "owner/model",
        "c" * 40,
        {"weight_map": {"weight": "model.safetensors"}},
        retries=1,
        opener=lambda _request, timeout: response,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="failed at model[.]safetensors"):
        reader.read_range("model.safetensors", 10, 13)


def test_range_reader_requires_immutable_revision_and_known_stable_size() -> None:
    with pytest.raises(ValueError, match="immutable 40-character commit SHA"):
        HubRangeReader(
            "owner/model",
            "main",
            {"weight_map": {"weight": "model.safetensors"}},
        )

    responses = iter(
        [
            FakeResponse(b"abc", "bytes 10-12/100"),
            FakeResponse(b"def", "bytes 13-15/101"),
        ]
    )
    reader = HubRangeReader(
        "owner/model",
        "e" * 40,
        {"weight_map": {"weight": "model.safetensors"}},
        retries=1,
        opener=lambda _request, timeout: next(responses),
        sleeper=lambda _seconds: None,
    )
    assert reader.read_range("model.safetensors", 10, 13) == b"abc"
    with pytest.raises(RuntimeError, match="failed at model[.]safetensors"):
        reader.read_range("model.safetensors", 13, 16)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(b"abc", "bytes 10-12/*"),
        FakeResponse(b"abc", "bytes 10-12/100", content_encoding="gzip"),
    ],
)
def test_range_reader_rejects_unknown_size_or_encoded_payload(
    response: FakeResponse,
) -> None:
    reader = HubRangeReader(
        "owner/model",
        "f" * 40,
        {"weight_map": {"weight": "model.safetensors"}},
        retries=1,
        opener=lambda _request, timeout: response,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="failed at model[.]safetensors"):
        reader.read_range("model.safetensors", 10, 13)


def test_surgery_pair_id_binds_metadata_and_expert_selection() -> None:
    values = {
        "bf16_repository": "owner/bf16",
        "bf16_revision": "a" * 40,
        "bf16_config_sha256": "1" * 64,
        "bf16_index_sha256": "2" * 64,
        "fp8_repository": "owner/fp8",
        "fp8_revision": "b" * 40,
        "fp8_config_sha256": "3" * 64,
        "fp8_index_sha256": "4" * 64,
        "layer_map": DEFAULT_LAYER_MAP,
        "selections": {
            3: {
                "method": "deterministic-cosine-pam-v1",
                "selected_source_experts": [2, 7],
                "source_clusters": [[0, 2], [1, 7]],
            }
        },
    }
    baseline = surgery_pair_id(**values)
    changed_metadata = {**values, "bf16_index_sha256": "5" * 64}
    changed_selection = deepcopy(values)
    changed_selection["selections"][3]["selected_source_experts"] = [2, 8]
    changed_diagnostic = deepcopy(values)
    changed_diagnostic["selections"][3]["cosine_distance_objective"] = 0.123
    assert surgery_pair_id(**changed_metadata) != baseline
    assert surgery_pair_id(**changed_selection) != baseline
    assert surgery_pair_id(**changed_diagnostic) == baseline


class LocalReader:
    payload = np.asarray([1.25, -2.5], dtype="<f4").tobytes()

    def __init__(self, repository, revision, source_index, *, chunk_bytes):
        assert source_index["weight_map"] == {"weight": "source.safetensors"}
        self.repository = repository
        self.revision = revision
        self.remote_bytes = 0
        self.range_receipts = []

    def location(self, name: str) -> TensorLocation:
        assert name == "weight"
        return TensorLocation(
            name="weight",
            shard="source.safetensors",
            dtype="F32",
            shape=(2,),
            file_start=100,
            file_end=108,
        )

    def copy_range(self, shard, start, end, writer) -> None:
        assert (shard, start, end) == ("source.safetensors", 100, 108)
        writer.write(self.payload)
        self.remote_bytes += len(self.payload)
        self.range_receipts.append(
            {
                "repository": self.repository,
                "revision": self.revision,
                "shard": shard,
                "start": start,
                "end_exclusive": end,
                "requested_end_exclusive": end,
                "content_range": "bytes 100-107/1000",
                "reported_total_bytes": 1000,
                "bytes": len(self.payload),
                "sha256": hashlib.sha256(self.payload).hexdigest(),
                "complete_response": True,
            }
        )


def write_tiny_plan(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    config_path = metadata / "config.json"
    index_path = metadata / "model.safetensors.index.json"
    config_path.write_text("{}\n", encoding="utf-8")
    index_path.write_text(
        json.dumps({"weight_map": {"weight": "source.safetensors"}}) + "\n",
        encoding="utf-8",
    )
    pair_id = "glm52-test-pair"
    plan = {
        "schema_version": 1,
        "status": "test fixture",
        "pair_id": pair_id,
        "source": {
            "repository": "owner/source",
            "revision": "d" * 40,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        },
        "target": {
            "repository": "owner/target",
            "profile": "unit-test",
            "precision_role": "bf16",
            "model_parameter_count": 2,
            "serialized_bytes": 8,
        },
        "config": {"surgery_pair_id": pair_id, "torch_dtype": "bfloat16"},
        "expert_selection": {},
        "tensors": [
            {
                "kind": "direct",
                "target_name": "weight",
                "dtype": "F32",
                "shape": [2],
                "nbytes": 8,
                "source": {
                    "name": "weight",
                    "shard": "source.safetensors",
                    "dtype": "F32",
                    "shape": [2],
                    "file_start": 100,
                    "file_end": 108,
                },
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path, metadata


def test_plan_driven_builder_and_verifier_are_exact_and_receipted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, metadata = write_tiny_plan(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(builder, "HubRangeReader", LocalReader)
    monkeypatch.setattr(builder, "available_memory_bytes", lambda: 4 * 1024**3)
    args = SimpleNamespace(
        plan=plan_path,
        metadata_dir=metadata,
        output=output,
        max_shard_size_gib=0.001,
        chunk_mib=1,
        execute=True,
    )
    result = builder.build(args)
    assert result["status"] == "built"
    assert result["source_range_receipt_count"] == 1
    resumed = builder.build(args)
    assert resumed["source_range_receipt_count"] == 1
    assert resumed["remote_source_bytes"] == len(LocalReader.payload)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verified = verify_built(output, plan_path, plan)
    assert verified["status"] == "verified"
    manifest = json.loads((output / "surgery_manifest.json").read_text())
    assert manifest["source_range_receipts"][0]["sha256"] == hashlib.sha256(LocalReader.payload).hexdigest()

    broken_manifest = deepcopy(manifest)
    broken_manifest["remote_source_bytes"] += 1
    (output / "surgery_manifest.json").write_text(json.dumps(broken_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt bytes differ"):
        verify_built(output, plan_path, plan)
    (output / "surgery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    broken_manifest = deepcopy(manifest)
    broken_manifest["source"]["revision"] = "e" * 40
    (output / "surgery_manifest.json").write_text(json.dumps(broken_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest source differs from plan"):
        verify_built(output, plan_path, plan)
    (output / "surgery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    broken_manifest = deepcopy(manifest)
    broken_manifest["source_range_receipts"][0]["reported_total_bytes"] = 999
    (output / "surgery_manifest.json").write_text(json.dumps(broken_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="total-size mismatch"):
        verify_built(output, plan_path, plan)
    (output / "surgery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    built_config = json.loads((output / "config.json").read_text())
    built_config["unplanned"] = True
    (output / "config.json").write_text(json.dumps(built_config), encoding="utf-8")
    with pytest.raises(ValueError, match="built config differs from plan"):
        verify_built(output, plan_path, plan)


def test_pair_verifier_rejects_logical_shape_drift() -> None:
    base = {
        "pair_id": "glm52-5l32e-test",
        "expert_selection": {},
        "tensors": [
            {
                "target_name": "weight",
                "dtype": "BF16",
                "shape": [2, 2],
                "nbytes": 8,
            }
        ],
    }
    fp8 = deepcopy(base)
    fp8["tensors"][0]["shape"] = [1, 4]
    with pytest.raises(ValueError, match="shape mismatch"):
        verify_pair_contract(base, fp8)


def test_legacy_launchers_use_the_proven_recompute_null_contract() -> None:
    launchers = {
        "run_surgery_sft_megatron.sh": "engine.override_transformer_config",
        "run_surgery_grpo_megatron_sglang.sh": ("actor_rollout_ref.actor.megatron.override_transformer_config"),
    }
    for filename, prefix in launchers.items():
        source = (SCRIPTS / filename).read_text(encoding="utf-8")
        assert f"{prefix}.recompute_granularity=null" in source
        assert f"{prefix}.recompute_method=null" in source
        assert f"{prefix}.recompute_num_layers=null" in source
        assert f"{prefix}.recompute_granularity=full" not in source


def test_fp8_remote_launcher_is_bound_to_published_contract_and_real_tools() -> None:
    source = (SCRIPTS / "run_glm52_fp8_twin_remote.sh").read_text(encoding="utf-8")
    assert "5eedf18a056d10b37452528c930487cc48dbd63a" in source
    assert "5e0152c0d8dcbc7e0fdb236e4b264ab4a7e997fa6421e20970b4a274c5883181" in source
    assert 'plan_file="$plan_dir/surgery_plan.json"' in source
    for filename in (
        "build_glm52_surgery_dummy.py",
        "verify_glm52_surgery_pair.py",
    ):
        assert filename in source
        assert (SCRIPTS / filename).is_file()
    assert '"$hf_bin" auth whoami --format json' in source
    assert '"$hf_bin" upload "$TARGET_REPO" "$model_dir" .' in source
    assert source.count("--type model") == 3
    assert "upload-large-folder" not in source
    assert "--num-workers" not in source
