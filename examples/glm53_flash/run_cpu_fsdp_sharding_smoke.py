#!/usr/bin/env python3
"""Run one GLM optimizer step with real multi-process FSDP shards on CPU."""

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForImageTextToText


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("This smoke requires at least two ranks; world-size 1 is NO_SHARD")

    torch.set_num_threads(1)
    torch.manual_seed(17)
    dist.init_process_group("gloo")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
        model.config.use_cache = False
        model.train()
        full_parameter_numel = sum(parameter.numel() for parameter in model.parameters())
        model = FSDP(
            model,
            device_id=torch.device("cpu"),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            limit_all_gathers=True,
        )
        flat_parameter = model._handle.flat_param
        local_shard_before = flat_parameter.detach().clone()
        local_shard_numel = flat_parameter.numel()
        unsharded_numel = flat_parameter._unpadded_unsharded_size.numel()
        expected_shard_numel = (unsharded_numel + world_size - 1) // world_size
        if local_shard_numel != expected_shard_numel:
            raise RuntimeError(f"Rank {rank} owns {local_shard_numel} elements, expected {expected_shard_numel}")

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            use_cache=False,
        )
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Rank {rank} produced a non-finite loss")
        loss.backward()
        local_grad_sq = torch.zeros((), dtype=torch.float64)
        for parameter in model.parameters():
            if parameter.grad is not None:
                local_grad_sq += parameter.grad.detach().double().square().sum()
        dist.all_reduce(local_grad_sq)
        global_grad_norm = local_grad_sq.sqrt()
        if not torch.isfinite(global_grad_norm) or global_grad_norm.item() == 0:
            raise RuntimeError(f"Invalid global gradient norm: {global_grad_norm.item()}")
        optimizer.step()

        local_delta = (flat_parameter.detach() - local_shard_before).abs()
        local_changed = torch.tensor(int(torch.count_nonzero(local_delta)), dtype=torch.long)
        dist.all_reduce(local_changed)
        if local_changed.item() == 0:
            raise RuntimeError("The optimizer step did not change any parameter shard")

        rank_report = {
            "rank": rank,
            "loss": loss.detach().item(),
            "local_shard_numel": local_shard_numel,
            "local_changed_elements": int(torch.count_nonzero(local_delta)),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        reports = [None] * world_size
        dist.all_gather_object(reports, rank_report)
        if rank == 0:
            result = {
                "status": "pass",
                "backend": "gloo",
                "sharding_strategy": "FULL_SHARD",
                "world_size": world_size,
                "full_parameter_numel": full_parameter_numel,
                "flat_unsharded_numel": unsharded_numel,
                "flat_local_shard_numel": local_shard_numel,
                "shard_fraction": local_shard_numel / unsharded_numel,
                "global_grad_norm": global_grad_norm.item(),
                "changed_elements_across_shards": local_changed.item(),
                "ranks": reports,
            }
            rendered = json.dumps(result, indent=2, sort_keys=True)
            print(rendered)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
