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


def _causal_decay_products(
    k_beta: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    g: torch.Tensor,
    block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute causal KDA products without a full pairwise decay tensor.

    Cross-block decay factorizes around ``g[mid]`` as
    ``exp(g_i - g_j) = exp(g_i - g_mid) * exp(g_mid - g_j)``. Since cumulative
    GLM log-decay is monotone, both factors are at most one. Only small diagonal
    blocks retain a pairwise tensor, with future positions masked before
    exponentiation.
    """
    chunk_size = g.shape[-2]
    akk = g.new_zeros(*g.shape[:-1], chunk_size)
    aqk = g.new_zeros(*g.shape[:-1], chunk_size)
    non_causal = torch.ones(
        block_size, block_size, dtype=torch.bool, device=g.device
    ).triu(1)

    def fill(lo: int, hi: int) -> None:
        size = hi - lo
        if size <= block_size:
            g_block = g[..., lo:hi, :]
            decay = (
                (g_block.unsqueeze(-2) - g_block.unsqueeze(-3))
                .masked_fill(non_causal[:size, :size, None], float("-inf"))
                .exp()
            )
            keys = key[..., lo:hi, :].unsqueeze(-3)
            akk[..., lo:hi, lo:hi] = (
                k_beta[..., lo:hi, :].unsqueeze(-2) * keys * decay
            ).sum(dim=-1).tril(-1)
            aqk[..., lo:hi, lo:hi] = (
                query[..., lo:hi, :].unsqueeze(-2) * keys * decay
            ).sum(dim=-1)
            return

        mid = (lo + hi) // 2
        fill(lo, mid)
        fill(mid, hi)
        anchor = g[..., mid : mid + 1, :]
        row_decay = (g[..., mid:hi, :] - anchor).exp()
        scaled_key = (
            key[..., lo:mid, :] * (anchor - g[..., lo:mid, :]).exp()
        ).transpose(-1, -2)
        akk[..., mid:hi, lo:mid] = (
            k_beta[..., mid:hi, :] * row_decay
        ) @ scaled_key
        aqk[..., mid:hi, lo:mid] = (
            query[..., mid:hi, :] * row_decay
        ) @ scaled_key

    fill(0, chunk_size)
    return akk, aqk


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
    """Torch KDA reference with stable, blockwise causal decay products.

    Transformers 5.16.1 materializes a pairwise per-channel decay tensor and
    exponentiates future positions before masking them. This implementation
    avoids both the resulting ``0 * inf`` gradients and the quadratic channel
    tensor by using local-reference block factorization.
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
    akk, aqk = _causal_decay_products(k_beta, query, key, g)
    attn = -akk
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
        attn_intra = aqk[:, :, index]
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
    if getattr(model.config, "model_type", None) not in {
        "glm5_next",
        "glm5_next_text",
    }:
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
