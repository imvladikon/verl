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
"""Redistributing a step's samples across dp ranks must not lose or duplicate one.

The failure this guards against is silent: a mis-ordered rebuild pairs one sample's tokens with
another's loss mask, and training continues on corrupted supervision rather than raising. So the
checks below are about identity -- which sample ended up where, with its own columns still attached
-- not only about the balance improving.

Runs on gloo/CPU, so it needs no GPU.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.dp_balance import balance_batch_across_dp, partition_for_dp
from verl.utils.tensordict_utils import get_tensordict, nested_tensor_from_tensor_list


def _make_batch(lengths, rope_dim=0):
    """One rank's batch: input_ids keyed so a sample can be identified after it moves."""
    input_ids = [torch.full((length,), fill_value=length, dtype=torch.long) for length in lengths]
    loss_mask = [torch.ones(length, dtype=torch.long) * length for length in lengths]
    columns = {
        "input_ids": torch.nested.as_nested_tensor(input_ids, layout=torch.jagged),
        "loss_mask": torch.nested.as_nested_tensor(loss_mask, layout=torch.jagged),
    }
    if rope_dim:
        columns["position_ids"] = nested_tensor_from_tensor_list(
            [torch.arange(length).repeat(rope_dim, 1) for length in lengths], ragged_idx=2
        )
    return get_tensordict(columns)


# Lengths per rank, deliberately lopsided: rank 0 carries far more than rank 3.
_LENGTHS = [[90, 100, 110, 120], [40, 45, 50, 55], [30, 32, 34, 36], [10, 12, 14, 16]]


def _worker(rank, world_size, rope_dim, result_queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531", RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        seqlens = [length for rank_lengths in _LENGTHS for length in rank_lengths]
        data = _make_batch(_LENGTHS[rank], rope_dim=rope_dim)
        # Pass a device so the placement path is exercised: it is where a ragged column can lose
        # the _ragged_idx attribute that says which dimension varies.
        balanced = balance_batch_across_dp(data, seqlens, dp_group=None, dp_size=world_size, device="cpu")

        rows = []
        for i in range(balanced.batch_size[0]):
            ids = balanced["input_ids"][i]
            mask = balanced["loss_mask"][i]
            row = {"length": ids.shape[0], "id_value": int(ids[0]), "mask_value": int(mask[0])}
            if rope_dim:
                position_ids = balanced["position_ids"][i]
                row["position_shape"] = tuple(position_ids.shape)
                row["position_last"] = int(position_ids[0, -1])
                row["ragged_idx"] = getattr(balanced["position_ids"], "_ragged_idx", None)
            rows.append(row)
        result_queue.put((rank, rows))
    finally:
        dist.destroy_process_group()


def _run(world_size=4, rope_dim=0):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(rank, world_size, rope_dim, queue)) for rank in range(world_size)]
    for proc in procs:
        proc.start()
    results = dict(queue.get(timeout=120) for _ in procs)
    for proc in procs:
        proc.join(timeout=120)
        assert proc.exitcode == 0, f"worker exited {proc.exitcode}"
    return results


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device to move to")
def test_a_ragged_column_keeps_which_dimension_varies_when_moved_to_the_device():
    """The rebuilt batch is assembled on the host and then moved, so this has to survive the move.

    ``_ragged_idx`` is a plain attribute on the nested tensor rather than part of its type, and it
    is what says a multimodal ``position_ids`` is ``(4, seqlen)`` ragged in dim 2 rather than ragged
    in its last dimension. If a torch or tensordict upgrade stops carrying it, every consumer that
    reads it -- index_select_tensor_dict included -- silently starts slicing the wrong dimension,
    which is worth finding here rather than in a training curve.
    """
    parts = [torch.arange(n).repeat(4, 1) for n in (5, 7, 3)]
    column = nested_tensor_from_tensor_list(parts, ragged_idx=2)
    moved = get_tensordict({"position_ids": column}).to("cuda")

    assert getattr(moved["position_ids"], "_ragged_idx", None) == 2
    assert tuple(moved["position_ids"][0].shape) == (4, 5)


def test_partition_is_the_same_on_every_rank_and_equal_sized():
    seqlens = [length for rank_lengths in _LENGTHS for length in rank_lengths]
    partitions = partition_for_dp(seqlens, dp_size=4)
    assert len(partitions) == 4
    assert all(len(part) == 4 for part in partitions)
    assert sorted(i for part in partitions for i in part) == list(range(16))
    # Deterministic: the callers rely on every rank deriving the same answer without agreeing on it.
    assert partition_for_dp(seqlens, dp_size=4) == partitions


@pytest.mark.parametrize("rope_dim", [0, 4])
def test_every_sample_survives_the_exchange_with_its_own_columns(rope_dim):
    results = _run(rope_dim=rope_dim)

    assert sorted(results) == [0, 1, 2, 3]
    assert all(len(rows) == 4 for rows in results.values()), "each rank must keep its sample count"

    rows = [row for rank_rows in results.values() for row in rank_rows]
    # Nothing lost, nothing duplicated: the global multiset of lengths is exactly what went in.
    assert sorted(row["length"] for row in rows) == sorted(
        length for rank_lengths in _LENGTHS for length in rank_lengths
    )
    for row in rows:
        # input_ids and loss_mask were both filled with the sample's own length, so a rebuild that
        # paired columns from different samples shows up here.
        assert row["id_value"] == row["length"], row
        assert row["mask_value"] == row["length"], row
        if rope_dim:
            assert row["position_shape"] == (rope_dim, row["length"]), row
            assert row["position_last"] == row["length"] - 1, row
            # Without this the column silently reads as ragged in its last dimension instead.
            assert row["ragged_idx"] == 2, row


def test_balance_improves_the_spread():
    results = _run()
    per_rank = {rank: sum(row["length"] for row in rows) for rank, rows in results.items()}
    balanced_ratio = max(per_rank.values()) / (sum(per_rank.values()) / len(per_rank))

    before = [sum(rank_lengths) for rank_lengths in _LENGTHS]
    before_ratio = max(before) / (sum(before) / len(before))

    assert balanced_ratio < before_ratio, f"{per_rank} is no better than {before}"
    assert balanced_ratio < 1.1, f"still lopsided: {per_rank}"
