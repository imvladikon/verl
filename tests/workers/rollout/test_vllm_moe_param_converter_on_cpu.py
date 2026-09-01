# Copyright 2026 Bytedance Ltd. and/or its affiliates
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


import torch

from verl.workers.engine.fsdp.utils import unfuse_moe_params


def _collect(weights, model_type):
    return [item for item in unfuse_moe_params(weights, model_type)]


def test_qwen_moe_packed_weights_are_expanded_per_expert():
    gate_up = torch.randn(2, 6, 8)
    down = torch.randn(2, 8, 3)

    converted = _collect(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ],
        "qwen3_moe",
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert [tensor.shape for _, tensor in converted] == [
        (3, 8),
        (3, 8),
        (3, 8),
        (3, 8),
        (8, 3),
        (8, 3),
    ]


def test_gpt_oss_packed_weights_are_not_expanded():
    gate_up = torch.randn(2, 8, 6)
    down = torch.randn(2, 3, 8)
    weights = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]

    converted = _collect(weights, "gpt_oss")

    assert [name for name, _ in converted] == [name for name, _ in weights]
    assert converted[0][1] is gate_up
    assert converted[1][1] is down
    assert converted[0][1].shape == (2, 8, 6)
    assert converted[1][1].shape == (2, 3, 8)


def test_glm5_next_runtime_names_are_reverted_to_checkpoint_names():
    weights = [
        ("model.layers.0.attn_hc.fn", torch.tensor([1.0])),
        ("model.layers.0.attn_hc.base", torch.tensor([2.0])),
        ("model.layers.0.attn_hc.scale", torch.tensor([3.0])),
        ("model.layers.0.ffn_hc.fn", torch.tensor([4.0])),
        ("model.layers.0.ffn_hc.base", torch.tensor([5.0])),
        ("model.layers.0.ffn_hc.scale", torch.tensor([6.0])),
        ("model.layers.0.self_attn.forget_gate.f_a_proj.weight", torch.tensor([7.0])),
        ("model.layers.0.self_attn.forget_gate.f_b_proj.weight", torch.tensor([8.0])),
        ("model.layers.0.self_attn.forget_gate.dt_bias", torch.tensor([9.0])),
        ("model.layers.0.self_attn.forget_gate.A_log", torch.tensor([10.0])),
    ]

    converted = _collect(weights, "glm5_next")

    assert [name for name, _ in converted] == [
        "model.layers.0.hc_attn_fn",
        "model.layers.0.hc_attn_base",
        "model.layers.0.hc_attn_scale",
        "model.layers.0.hc_ffn_fn",
        "model.layers.0.hc_ffn_base",
        "model.layers.0.hc_ffn_scale",
        "model.layers.0.self_attn.f_a_proj.weight",
        "model.layers.0.self_attn.f_b_proj.weight",
        "model.layers.0.self_attn.dt_bias",
        "model.layers.0.self_attn.A_log",
    ]
    for (_, actual), (_, expected) in zip(converted, weights, strict=True):
        assert actual is expected


def test_glm5_next_fused_conv1d_is_split_without_cross_channel_mixing():
    fused = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(12, 1, 2)

    converted = _collect(
        [("model.layers.0.self_attn.conv1d.weight", fused)],
        "glm5_next_text",
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.self_attn.q_conv1d.weight",
        "model.layers.0.self_attn.k_conv1d.weight",
        "model.layers.0.self_attn.v_conv1d.weight",
    ]
    for index, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, fused[index * 4 : (index + 1) * 4])
        assert tensor.is_contiguous()


def test_glm5_next_fused_conv1d_rejects_invalid_channel_count():
    fused = torch.zeros(10, 1, 2)

    try:
        _collect([("model.layers.0.self_attn.conv1d.weight", fused)], "glm5_next")
    except ValueError as error:
        assert "Invalid GLM-5.3 fused conv1d shape" in str(error)
    else:
        raise AssertionError("Invalid GLM-5.3 fused conv1d shape was accepted")
