# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Give every data-parallel rank a comparable share of a step's work.

The sampler hands each rank the same number of samples, not the same number of tokens, and nothing
downstream moves tokens between ranks: :func:`verl.utils.seqlen_balancing.rearrange_micro_batches`
all-reduces the micro-batch *count* over the dp group and then partitions each rank's own samples.
On a MoE model every layer then meets at the dispatcher's all-gather over the expert group, so the
step costs the slowest rank rather than the average one.

This is the SPMD counterpart of ``trainer.balance_batch`` in the Ray trainers
(``verl/trainer/ppo/ray_trainer.py``), which can partition a global batch because the driver holds
one. Here every rank holds its own shard, so the global batch is reconstructed by an all-gather and
partitioned identically everywhere -- the partition is a pure function of the gathered lengths, so
no second collective is needed to agree on it.

Training is unchanged by this: the loss is normalized by a token count that is itself all-reduced
over the dp group (``verl/workers/engine/megatron/transformer_impl.py``), so moving a sample from
one rank to another permutes the global batch and leaves the summed gradient alone. That is what
makes it safe to turn on and off inside one run without spoiling a comparison.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions


def partition_for_dp(seqlens: list[int], dp_size: int) -> list[list[int]]:
    """Which global sample indices each rank should end up with, equal count per rank.

    A pure function of the gathered lengths, so every rank computes the same answer.
    """
    workloads = calculate_workload(torch.tensor(seqlens, dtype=torch.long)).tolist()
    return get_seqlen_balanced_partitions(workloads, k_partitions=dp_size, equal_size=True)


def balance_batch_across_dp(data: TensorDict, seqlens: list[int], dp_group, dp_size: int, device=None) -> TensorDict:
    """Re-deal the step's samples so every rank carries a comparable amount of work.

    ``seqlens`` is the global, rank-ordered sequence-length list that the trainer already
    all-gathers for its metrics, so the partition itself costs nothing.

    The exchange goes through ``all_gather_object`` on CPU copies rather than a padded all-gather of
    each ragged column. The batch is a few megabytes once per step against seconds of waiting, and
    the padded version has to get every column's ragged dimension right, where a mistake corrupts
    training quietly instead of failing.
    """
    if dp_size <= 1:
        return data
    local_bsz = data.batch_size[0]
    assert len(seqlens) == local_bsz * dp_size, (
        f"seqlens must cover the whole dp group: got {len(seqlens)} for {dp_size} ranks of {local_bsz}"
    )

    rank = dist.get_rank(group=dp_group)
    mine = sorted(partition_for_dp(seqlens, dp_size)[rank])
    # A partition of the wrong size would change the step's sample count, not just its balance.
    assert len(mine) == local_bsz, f"partition for rank {rank} has {len(mine)} samples, expected {local_bsz}"

    # The exchange below is unconditional on purpose. Skipping it for a rank whose partition happens
    # to be its own block is a per-rank decision taken before a collective: the ranks that skip never
    # reach the all_gather and the ranks that do not wait for them forever. The saving would have
    # been one exchange on the rare step where nothing needs to move.
    # One object per rank, not one per sample. Walking a batch column element-wise looks harmless
    # and is not: indexing a tensorclass or NonTensorStack copies the whole column per element, so
    # a 16-sample batch of 8k-token rows spent twelve minutes at 100% CPU and 63 GB of RSS before
    # anyone could tell it apart from a hang.
    gathered: list = [None] * dp_size
    dist.all_gather_object(gathered, data.cpu(), group=dp_group)

    selected = tu.index_select_tensor_dict(tu.concat_tensordict(gathered), mine)
    return selected.to(device) if device is not None else selected
