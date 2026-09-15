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

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from verl.workers.rollout.sglang_rollout.async_sglang_server import (  # noqa: E402
    describe_sglang_backends,
    uses_dsa_attention,
)


def test_reports_the_resolved_dsa_and_runner_backends(monkeypatch):
    monkeypatch.setenv("SGLANG_DSA_FUSE_TOPK", "false")
    args = SimpleNamespace(
        kv_cache_dtype="bfloat16",
        dsa_prefill_backend="fa3",
        dsa_decode_backend="fa3",
        dsa_topk_backend="sgl-kernel",
        moe_runner_backend="triton",
        disable_cuda_graph=False,
    )
    line = describe_sglang_backends(args)
    for expected in (
        "dsa_prefill_backend=fa3",
        "dsa_decode_backend=fa3",
        "dsa_topk_backend=sgl-kernel",
        "moe_runner_backend=triton",
        "disable_cuda_graph=False",
        "SGLANG_DSA_FUSE_TOPK=false",
    ):
        assert expected in line
    assert "linear_attn_backend" not in line  # fields the SGLang build lacks are skipped


def test_dsa_models_leave_the_attention_backend_to_sglang():
    glm5_next = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"], text_config=SimpleNamespace(index_topk=2048)
    )
    glm_moe_dsa = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], index_topk=2048, text_config=None)
    deepseek_v3 = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
    assert uses_dsa_attention(glm5_next)
    assert uses_dsa_attention(glm_moe_dsa)
    assert not uses_dsa_attention(deepseek_v3)
