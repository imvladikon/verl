# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch

from verl.utils.kernel import fp8_kernel


def _reference(x, block_size=128):
    """Independent per-block oracle with the existing reciprocal-scale contract."""
    result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        ((x.shape[0] + block_size - 1) // block_size, (x.shape[1] + block_size - 1) // block_size, 1),
        dtype=torch.float32,
        device=x.device,
    )
    for row in range(0, x.shape[0], block_size):
        for col in range(0, x.shape[1], block_size):
            block = x[row : row + block_size, col : col + block_size].float()
            amax = block.abs().max()
            multiplier = torch.where((amax == 0) | (amax == torch.inf), torch.ones_like(amax), torch.div(448.0, amax))
            result[row : row + block_size, col : col + block_size] = (block * multiplier).clamp(-448, 448)
            scales[row // block_size, col // block_size, 0] = multiplier.reciprocal()
    return result, scales


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(128, 128), (256, 128), (128, 256), (257, 385)])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_fallback_preserves_input_and_block_values(monkeypatch, dtype, shape, layout):
    device = os.environ.get("FP8_TEST_DEVICE", "cpu")
    generator = torch.Generator().manual_seed(21)
    rows, cols = shape
    backing_shape = (cols, rows) if layout == "transpose" else (rows, cols * (2 if layout == "strided" else 1))
    backing = torch.randn(backing_shape, generator=generator).to(device=device, dtype=dtype)
    x = backing.t() if layout == "transpose" else backing[:, ::2] if layout == "strided" else backing
    before = backing.clone()
    expected, expected_scale = _reference(x)
    # Force both row and column tiling with small fixtures, including partial blocks.
    monkeypatch.setattr(fp8_kernel, "_FP8_PYTORCH_CHUNK_ELEMENTS", 2 * 128**2)
    actual, scale = fp8_kernel._scaled_fp8_blockwise_pytorch(x, [128, 128])
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    assert actual.is_contiguous()


def test_fp32_leaf_remains_usable_for_backward():
    device = os.environ.get("FP8_TEST_DEVICE", "cpu")
    x = torch.ones((256, 128), device=device, requires_grad=True)
    before = x.detach().clone()
    with torch.no_grad():
        fp8_kernel._scaled_fp8_blockwise_pytorch(x, [128, 128])
    x.square().sum().backward()
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, 2 * before, rtol=0, atol=0)


def test_fallback_zero_and_infinite_blocks(monkeypatch):
    device = os.environ.get("FP8_TEST_DEVICE", "cpu")
    x = torch.zeros((256, 256), device=device)
    x[:128, :128] = torch.inf
    x[128:, :128] = -torch.inf
    before = x.clone()
    monkeypatch.setattr(fp8_kernel, "_FP8_PYTORCH_CHUNK_ELEMENTS", 128**2)
    actual, scale = fp8_kernel._scaled_fp8_blockwise_pytorch(x, [128, 128])
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(actual.float(), x.clamp(-448, 448), rtol=0, atol=0)
    torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)
