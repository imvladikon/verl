#!/usr/bin/env python3
"""Real two-rank FSDP1/PEFT export, nested materialization and adapter reload.

Only the requested source module is loaded process-locally. No model downloads,
mock collectives, fake PEFT filtering, global installs or shared process groups.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-original-bugs", action="store_true")
    args = parser.parse_args()
    rank, local_rank, world = [int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    uuids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if world != 2 or len(uuids) != 2 or not all(x.startswith("GPU-") for x in uuids):
        parser.error("two explicit GPU UUIDs required")
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuids[local_rank])
    if pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        parser.error("GPU occupied; no action taken")
    pynvml.nvmlShutdown()
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    spec = importlib.util.spec_from_file_location("fsdp_utils_candidate", args.source)
    fu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fu)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=3))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), torch=torch.__version__, rank=rank, cases=[]
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2048, 2048, bias=False, device=device)

        def forward(self, x):
            return self.proj(x)

    try:
        for use_orig in (True, False):
            torch.manual_seed(67)
            peft = get_peft_model(Model(), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"]))
            expected = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(peft).items()}
            assert len(expected) == 2
            proj = peft.base_model.model.proj
            child = FSDP(proj.base_layer, use_orig_params=use_orig, device_id=device)
            proj.base_layer = child
            owner = FSDP(proj, use_orig_params=use_orig, device_id=device)
            peft.base_model.model.proj = owner
            root = FSDP(peft, use_orig_params=use_orig, device_id=device)
            # Real forward/backward initializes FSDP's normal lifecycle first.
            root(torch.randn(1, 2048, device=device)).float().sum().backward()
            base_before = child._handle.flat_param.detach().clone()
            observed = []
            original_filter = fu.get_peft_model_state_dict

            def measured_filter(*a, _child=child, _observed=observed, _filter=original_filter, **kw):
                _observed.append(_child._handle.flat_param.numel())
                return _filter(*a, **kw)  # real PEFT filtering, unchanged

            fu.get_peft_model_state_dict = measured_filter
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_allocated = torch.cuda.memory_allocated()
            started = time.monotonic()
            try:
                exported = fu.collect_lora_params(
                    root, layered_summon=True, base_sync_done=True, allow_full_summon_fallback=False
                )
            finally:
                fu.get_peft_model_state_dict = original_filter
            row = dict(
                use_orig_params=use_orig,
                expected_keys=sorted(expected),
                exported_keys=sorted(exported),
                child_local_numel=base_before.numel(),
                child_numel_during_filter=observed,
                cuda_peak_increment=torch.cuda.max_memory_allocated() - start_allocated,
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                export_seconds=time.monotonic() - started,
            )
            if args.expect_original_bugs and not use_orig:
                assert not exported, sorted(exported)
                row["expected_failure"] = (
                    "coarse adapter names hidden by flat_param; checkpoint caller rejects empty state"
                )
            else:
                assert exported.keys() == expected.keys(), row
                for k in expected:
                    torch.testing.assert_close(exported[k], expected[k], rtol=0, atol=0)
                target = {k: v + 0.125 for k, v in exported.items()}
                checkpoint = args.output_dir / f"adapter-orig{use_orig}-rank{rank}.pt"
                torch.save(target, checkpoint)
                loaded = torch.load(checkpoint, weights_only=True)
                fu.layered_load_lora_params(root, loaded)
                reexport = fu.collect_lora_params(
                    root, layered_summon=True, base_sync_done=True, allow_full_summon_fallback=False
                )
                assert reexport.keys() == target.keys()
                for k in target:
                    torch.testing.assert_close(reexport[k], target[k], rtol=0, atol=0)
                row["adapter_file_bytes"] = checkpoint.stat().st_size
                row["reload_exact"] = True
                multiplier = 2 if args.expect_original_bugs else 1
                assert observed and max(observed) == base_before.numel() * multiplier, row
            torch.testing.assert_close(child._handle.flat_param, base_before, rtol=0, atol=0)
            row["base_shard_unchanged"] = True
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            del root, peft, owner, child, proj, base_before
        report["status"] = "EXPECTED_OLD_BUGS_REPRODUCED" if args.expect_original_bugs else "EXPORT_RELOAD_PASS"
        with (args.output_dir / f"rank-{rank}.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
