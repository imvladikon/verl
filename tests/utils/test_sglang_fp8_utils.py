# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import asyncio
from types import SimpleNamespace

import torch

from verl.utils.sglang.sglang_fp8_utils import (
    SGLangFP8QuantizerHelper,
    build_sglang_fp8_quant_config,
    is_sglang_fp8_quant_config,
)


class MappingLikeConfig:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def test_build_sglang_fp8_quant_config_preserves_defaults(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)

    quant_config = build_sglang_fp8_quant_config()

    assert quant_config == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }


def test_sglang_fp8_quant_config_detection():
    assert is_sglang_fp8_quant_config({"quant_method": "fp8"})
    assert is_sglang_fp8_quant_config(MappingLikeConfig({"quant_method": "FP8"}))
    assert not is_sglang_fp8_quant_config(None)
    assert not is_sglang_fp8_quant_config({"quant_method": "compressed-tensors"})


def test_sglang_fp8_quant_config_merges_hf_ignored_layers(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "ignored_layers": ["model.layers.0.self_attn.q_proj"],
            "modules_to_not_convert": ["model.layers.1.mlp.down_proj"],
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
    ]
    assert not helper.should_quantize_param("model.layers.0.self_attn.q_proj.weight")
    assert not helper.should_quantize_param("model.layers.1.mlp.down_proj.weight")
    assert helper.should_quantize_param("model.layers.2.mlp.down_proj.weight")


def test_sglang_fp8_quant_config_accepts_mapping_like_config(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = MappingLikeConfig(
        {
            "quantization_config": MappingLikeConfig(
                {
                    "ignored_layers": ["model.layers.0.linear_attn"],
                }
            )
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["model.layers.0.linear_attn"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")


def test_sglang_fp8_quantizer_handles_glm53_flash_text_prefixes(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "modules_to_not_convert": [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.o_proj",
                "model.visual.merger.down_proj",
            ],
        }
    )

    helper = SGLangFP8QuantizerHelper(build_sglang_fp8_quant_config(hf_config))

    # HF's multimodal wrapper inserts ``language_model`` into text parameter
    # names, whereas SGLang and the checkpoint ignore list omit it.
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert not helper.should_quantize_param(f"model.language_model.layers.0.self_attn.{projection}.weight")
    assert not helper.should_quantize_param("model.visual.merger.down_proj.weight")


def test_sglang_fp8_quantizer_includes_glm53_flash_dsa_projections(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    helper = SGLangFP8QuantizerHelper(build_sglang_fp8_quant_config())

    for layer_id in (3, 7):
        for projection in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"):
            assert helper.should_quantize_param(f"model.language_model.layers.{layer_id}.self_attn.{projection}.weight")


def test_sglang_fp8_quantizer_matches_regex_ignored_layers(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "ignored_layers": ["re:.*linear_attn.*"],
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["re:.*linear_attn.*"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")
    assert not helper.should_quantize_param("model.layers.0.linear_attn.g_proj.weight")
    assert helper.should_quantize_param("model.layers.0.mlp.experts.0.up_proj.weight")


def test_sglang_fp8_quantizer_reads_sglang_env_ignored_layers(monkeypatch):
    monkeypatch.setenv("SGLANG_FP8_IGNORED_LAYERS", "linear_attn")

    quant_config = build_sglang_fp8_quant_config()
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["linear_attn"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")
    assert not helper.should_quantize_param("model.layers.0.linear_attn.g_proj.weight")
    assert helper.should_quantize_param("model.layers.0.mlp.experts.0.up_proj.weight")


def test_glm53_mixed_bf16_to_fp8_stream_keeps_ignored_layers_bf16(monkeypatch):
    """Exercise the trainer-BF16 to rollout-FP8 wire contract on CPU."""
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    monkeypatch.setattr("verl.utils.kernel.fp8_kernel._DISABLE_TRITON_FP8", True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    config = build_sglang_fp8_quant_config(
        SimpleNamespace(
            quantization_config={
                "modules_to_not_convert": [
                    "model.layers.0.self_attn.q_proj",
                ]
            }
        )
    )
    helper = SGLangFP8QuantizerHelper(config)
    dense_name = "model.language_model.layers.0.mlp.gate_proj.weight"
    ignored_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    dense = torch.linspace(-3, 3, 256 * 128, dtype=torch.bfloat16).reshape(256, 128)
    ignored = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)

    async def collect():
        return [item async for item in helper.quant_weights_by_name([(dense_name, dense), (ignored_name, ignored)])]

    result = asyncio.run(collect())

    assert [name for name, _ in result] == [
        dense_name,
        dense_name + "_scale_inv",
        ignored_name,
    ]
    quantized, scales, unchanged = (value for _, value in result)
    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.shape == dense.shape
    assert scales.dtype == torch.float32
    assert scales.shape == (2, 1)
    assert unchanged is ignored

    reconstructed = torch.empty_like(dense, dtype=torch.float32)
    for row in range(2):
        reconstructed[row * 128 : (row + 1) * 128] = quantized[row * 128 : (row + 1) * 128].float() * scales[row, 0]
    torch.testing.assert_close(reconstructed, dense.float(), rtol=0.07, atol=0.03)


def test_sglang_fp8_quantizer_does_not_silently_send_bf16(monkeypatch):
    helper = SGLangFP8QuantizerHelper(build_sglang_fp8_quant_config())
    monkeypatch.setattr(
        "verl.utils.fp8_utils.scaled_fp8_blockwise",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported fp8 cast")),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    async def collect():
        return [
            item
            async for item in helper.quant_weights_by_name(
                [("model.layers.0.mlp.gate_proj.weight", torch.ones(128, 128, dtype=torch.bfloat16))]
            )
        ]

    try:
        asyncio.run(collect())
    except RuntimeError as error:
        assert "model.layers.0.mlp.gate_proj.weight" in str(error)
    else:
        raise AssertionError("FP8 conversion failure was silently replaced with BF16")
