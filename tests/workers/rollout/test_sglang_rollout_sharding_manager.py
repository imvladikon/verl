# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import pytest
import torch

from verl.workers.rollout.sglang_rollout import sglang_rollout
from verl.workers.rollout.sglang_rollout.utils import _compact_for_bucket, get_named_tensor_buckets

_TENSOR_1MB = torch.zeros(512, 512)
_BYTES_1MB = 1 << 20


@pytest.mark.parametrize(
    "named_tensors, bucket_size_mb, gt_groups",
    [
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            0.5 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            1 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            1.5 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            2 * _BYTES_1MB,
            [["a", "b"]],
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_named_tensor_buckets(named_tensors, bucket_size_mb, gt_groups: list[list[str]]):
    named_tensors_iter = iter(named_tensors)
    groups = [g async for g in get_named_tensor_buckets(named_tensors_iter, bucket_size_mb)]
    assert len(groups) == len(gt_groups)
    for group, gt_group in zip(groups, gt_groups, strict=True):
        assert len(group) == len(gt_group)
        for (name, _), (gt_name) in zip(group, gt_group, strict=True):
            assert name == gt_name


def test_compact_for_bucket_skips_clone_for_owned_contiguous_tensor():
    # A freshly-allocated contiguous tensor owns tight storage (the DTensor.full_tensor() case).
    # It must be returned as-is so bucketing does not transiently double its peak memory.
    tensor = torch.randn(128, 64)
    assert _compact_for_bucket(tensor) is tensor


def test_compact_for_bucket_clones_view_into_larger_buffer():
    # A contiguous view that spans only part of a larger backing buffer must be compacted,
    # otherwise the whole buffer stays resident / gets shipped.
    base = torch.randn(256, 64)
    view = base[:128]
    assert view.is_contiguous()
    out = _compact_for_bucket(view)
    assert out is not view
    assert out.untyped_storage().nbytes() == out.numel() * out.element_size()
    assert torch.equal(out, view)


def test_compact_for_bucket_clones_non_contiguous_tensor():
    tensor = torch.randn(64, 128).t()
    assert not tensor.is_contiguous()
    out = _compact_for_bucket(tensor)
    assert out is not tensor
    assert out.data_ptr() != tensor.data_ptr()  # cloned into its own fresh storage
    assert torch.equal(out, tensor)


@pytest.mark.asyncio
async def test_get_named_tensor_buckets_preserves_values():
    # The conditional clone must not change the data that ends up in the buckets.
    named_tensors = [("a", torch.randn(64, 64)), ("b", torch.randn(64, 64)), ("c", torch.randn(64, 64))]
    expected = {name: tensor.clone() for name, tensor in named_tensors}
    flat = {}
    async for group in get_named_tensor_buckets(iter(named_tensors), 0.5 * _BYTES_1MB):
        for name, tensor in group:
            flat[name] = tensor
    assert set(flat) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(flat[name], tensor)


@pytest.mark.asyncio
async def test_server_adapter_finalizes_quantized_reload_after_all_buckets(monkeypatch):
    """The last RPC is a transaction boundary, not another data bucket."""

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    class FakeMesh:
        def __getitem__(self, key):
            assert key == "infer_tp"
            return self

        @staticmethod
        def get_local_rank():
            return 0

    class FakeEngine:
        def __init__(self):
            self.flush_count = 0

        async def flush_cache(self):
            self.flush_count += 1

    calls = []

    async def record_update(**kwargs):
        calls.append(
            {
                "names": [name for name, _ in kwargs["params_batch"]],
                "flush_cache": kwargs["flush_cache"],
            }
        )

    async def already_initialized(self):
        return None

    monkeypatch.setattr(sglang_rollout, "sgl_update_weights", record_update)
    monkeypatch.setattr(sglang_rollout, "_to_ipc_device", lambda tensor: tensor)
    monkeypatch.setattr(sglang_rollout.ServerAdapter, "_init_server_adapter", already_initialized)

    adapter = sglang_rollout.ServerAdapter.__new__(sglang_rollout.ServerAdapter)
    adapter.config = AttrDict(
        quantization=None,
        checkpoint_engine=AttrDict(update_weights_bucket_megabytes=1),
    )
    adapter.model_config = None
    adapter.device_mesh = FakeMesh()
    adapter._engine = FakeEngine()
    adapter._pd_role = None

    weights = iter(
        [
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(512, 512)),
            ("model.layers.0.mlp.up_proj.weight", torch.ones(512, 512)),
        ]
    )
    await adapter.update_weights(weights)

    assert calls == [
        {"names": ["model.layers.0.mlp.gate_proj.weight"], "flush_cache": False},
        {"names": ["model.layers.0.mlp.up_proj.weight"], "flush_cache": False},
        {"names": [], "flush_cache": True},
    ]
    assert adapter._engine.flush_count == 1
