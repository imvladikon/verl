from types import SimpleNamespace

import pytest
import torch

from verl.models.transformers import glm5_next


def test_safe_eager_kda_masks_future_decay_before_exp():
    torch.manual_seed(17)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 8, 8
    inputs = {
        "query": torch.randn(batch, sequence, heads, key_dim, requires_grad=True),
        "key": torch.randn(batch, sequence, heads, key_dim, requires_grad=True),
        "value": torch.randn(batch, sequence, heads, value_dim, requires_grad=True),
        "g": torch.full((batch, sequence, heads, key_dim), -5.0, requires_grad=True),
        "beta": torch.sigmoid(torch.randn(batch, sequence, heads)).requires_grad_(),
    }

    output, final_state = glm5_next._safe_eager_chunk_kimi_delta_attention(
        **inputs,
        use_qk_l2norm_in_kernel=True,
    )
    assert final_state is None
    assert torch.isfinite(output).all()

    output.float().square().mean().backward()
    for tensor in inputs.values():
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_safe_eager_kda_preserves_causal_decay_values():
    chunk_size, key_dim = 64, 8
    future_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    g = torch.full((1, 1, 1, chunk_size, key_dim), -5.0).cumsum(dim=-2)
    decay_delta = g.unsqueeze(-2) - g.unsqueeze(-3)

    old_decay = decay_delta.exp()
    safe_delta = decay_delta.masked_fill(future_mask.unsqueeze(-1), 0.0)
    safe_decay = safe_delta.exp().masked_fill(future_mask.unsqueeze(-1), 0.0)

    causal = ~future_mask
    torch.testing.assert_close(safe_decay[..., causal, :], old_decay[..., causal, :])
    assert torch.isfinite(safe_decay).all()
    assert torch.count_nonzero(safe_decay[..., future_mask, :]) == 0
    assert torch.isinf(old_decay).any()


def test_patch_replaces_only_plain_torch_fallback(monkeypatch):
    from transformers.models.glm5_next import modeling_glm5_next

    sentinel = object()
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="glm5_next"),
        _use_kernels=False,
        kernel_config=None,
    )
    monkeypatch.setattr(modeling_glm5_next, "chunk_kimi_delta_attention", sentinel)
    monkeypatch.setattr(glm5_next, "_native_fla_kda_available", lambda: False)

    assert glm5_next.patch_glm5_next_eager_kda(model)
    assert modeling_glm5_next.chunk_kimi_delta_attention is glm5_next._safe_eager_chunk_kimi_delta_attention
    assert not glm5_next.patch_glm5_next_eager_kda(model)


@pytest.mark.parametrize("use_kernels,native_fla", [(True, False), (False, True)])
def test_patch_preserves_optimized_kda_implementations(monkeypatch, use_kernels, native_fla):
    from transformers.models.glm5_next import modeling_glm5_next

    sentinel = object()
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="glm5_next"),
        _use_kernels=use_kernels,
        kernel_config=None,
    )
    monkeypatch.setattr(modeling_glm5_next, "chunk_kimi_delta_attention", sentinel)
    monkeypatch.setattr(glm5_next, "_native_fla_kda_available", lambda: native_fla)

    assert not glm5_next.patch_glm5_next_eager_kda(model)
    assert modeling_glm5_next.chunk_kimi_delta_attention is sentinel
