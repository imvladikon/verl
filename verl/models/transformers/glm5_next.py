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
"""Numerically safe compatibility helpers for Transformers GLM-5.3-Flash."""

import torch
import torch.nn.functional as F


def _safe_eager_chunk_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Torch KDA reference with causal decay masked before exponentiation.

    Transformers 5.16.1 masks future positions only after ``exp(g_i - g_j)``.
    With GLM's gate lower bound of -5 and a 64-token chunk, those unused
    positions can reach ``exp(315)``. The forward mask hides the infinities,
    but autograd then evaluates ``0 * inf`` and produces NaN query/key
    gradients. Pre-masking the exponent preserves every causal value while
    making the backward pass finite.
    """
    from transformers.models.glm5_next.modeling_glm5_next import l2norm

    del kwargs
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous().to(torch.float32) for tensor in (query, key, value, beta, g)
    ]

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        tensor.reshape(tensor.shape[0], tensor.shape[1], -1, chunk_size, tensor.shape[-1])
        for tensor in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    g = g.cumsum(dim=-2)
    diagonal_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    future_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    decay_delta = g.unsqueeze(-2) - g.unsqueeze(-3)
    decay_delta = decay_delta.masked_fill(future_mask.unsqueeze(-1), 0.0)
    decay_mask = decay_delta.exp().float().masked_fill(future_mask.unsqueeze(-1), 0.0)

    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(diagonal_mask, 0)
    for index in range(1, chunk_size):
        row = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for index in range(total_sequence_length // chunk_size):
        q_i = query[:, :, index]
        k_i = key[:, :, index]
        v_i = value[:, :, index]
        g_i = g[:, :, index]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (
            (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, index]).sum(dim=-1).masked_fill(future_mask, 0)
        )
        v_prime = k_cumdecay[:, :, index] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, index] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _native_fla_kda_available() -> bool:
    try:
        from fla.ops.kda import chunk_kda  # noqa: F401
    except Exception:
        return False
    return True


def patch_glm5_next_eager_kda(model) -> bool:
    """Patch only the plain Torch fallback, preserving configured KDA kernels."""
    if getattr(model.config, "model_type", None) != "glm5_next":
        return False
    if getattr(model, "_use_kernels", False) or getattr(model, "kernel_config", None) is not None:
        return False
    if _native_fla_kda_available():
        return False

    from transformers.models.glm5_next import modeling_glm5_next

    if modeling_glm5_next.chunk_kimi_delta_attention is _safe_eager_chunk_kimi_delta_attention:
        return False
    modeling_glm5_next.chunk_kimi_delta_attention = _safe_eager_chunk_kimi_delta_attention
    return True
