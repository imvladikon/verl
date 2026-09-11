"""The layered LoRA checkpoint has to survive both use_orig_params modes.

Run it as a gate, not as a probe:

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/utils/test_layered_lora_checkpoint_contract.py <verl/utils/fsdp_utils.py>

Exit code 0 means every combination below held; anything else names what broke.

What it pins down, measured on two A100s before the fix:

* ``use_orig_params=False``. FSDP1 has replaced the adapter with a
  FlatParameter, so ``named_parameters()`` outside a summon returns
  ``...lora_A._flat_param``. That name still contains ``lora_``, so the PEFT
  filter accepts it. Export then wrote four tensors instead of two, adding the
  raw flat parameters to the adapter, and load rejected a correct adapter as
  unknown. Both FULL_SHARD and NO_SHARD, so a single-GPU qualification run is
  affected as well.

* Rejections before any write. ``Tensor.copy_`` broadcasts, so a scalar or a
  transposed factor is otherwise accepted silently and fills the whole matrix
  with the wrong values. A checkpoint missing half the adapter is just as bad:
  the remaining factors stay at their init values and the model is quietly
  wrong. Both, plus an unknown key, must raise, and the adapter must be
  unchanged afterwards.

FSDP, the collectives and PEFT are real; the wrap policy is the one verl uses
for LoRA. Only ``verl.utils.device`` and ``verl.utils.model`` are stubbed, and
the functions under test do not call into them -- importing them for real pulls
in ray and the whole worker stack.
"""

import functools
import importlib.util
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

A_VALUE, B_VALUE = 3.0, 7.0


def stub_verl():
    for name in ("verl", "verl.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    device = types.ModuleType("verl.utils.device")
    device.get_device_id = lambda: torch.device("cpu")
    device.get_device_name = lambda: "cpu"
    device.get_torch_device = lambda: types.SimpleNamespace(empty_cache=lambda: None)
    sys.modules["verl.utils.device"] = device
    model = types.ModuleType("verl.utils.model")
    model.check_exclude_modules = lambda *args, **kwargs: False
    model.check_target_modules = lambda *args, **kwargs: True
    sys.modules["verl.utils.model"] = model


def load_fsdp_utils(path):
    spec = importlib.util.spec_from_file_location("fsdp_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fsdp_utils_under_test"] = module
    spec.loader.exec_module(module)
    return module


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


def lora_wrap_policy():
    """Wrap whatever the layout tagged, so several ownership shapes are reachable.

    verl's own policy wraps the trainable leaves; wrapping the whole
    ``lora.Linear`` instead would mix trainable adapter parameters with the
    frozen base weight, which FSDP1 refuses outright when
    ``use_orig_params=False``.
    """

    def lambda_policy_fn(module):
        return bool(getattr(module, "_wrap_for_test", False))

    return functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)


def tag_layout(model, layout):
    """Decide which modules become their own FSDP unit.

    ``leaf`` is what verl's LoRA policy produces: both factors in their own unit.
    ``root_only`` leaves the whole adapter in the root flat parameter.
    ``root_and_child`` wraps one factor and leaves the other at root, so the
    adapter lives at two levels at once -- the layout where an ownership check
    that compares un-normalized names starts claiming a nested unit's shards.
    """
    for name, module in model.named_modules():
        if layout == "leaf":
            module._wrap_for_test = name.endswith("lora_A.default") or name.endswith("lora_B.default")
        elif layout == "root_and_child":
            module._wrap_for_test = name.endswith("lora_A.default")
        else:
            module._wrap_for_test = False


def build(use_orig_params, strategy, layout):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        Base(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], lora_dropout=0.0, bias="none")
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.fill_(0.25 if "lora_A" in name else 0.5)
    tag_layout(model, layout)
    device = None
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.current_device()
    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=lora_wrap_policy(),
        use_orig_params=use_orig_params,
        device_id=device,
    )
    # One forward: without it the FSDP handles are uninitialized and summon fails.
    with torch.no_grad():
        wrapped(torch.zeros(1, 8, device=next(wrapped.parameters()).device))
    return wrapped


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def run_case(fsdp_utils, wrapped, label, failures):
    exported = fsdp_utils.layered_summon_lora_params(wrapped)
    check(
        set(exported) == {"base_model.model.proj.lora_A.weight", "base_model.model.proj.lora_B.weight"},
        f"{label}: export produced {sorted(exported)}",
        failures,
    )

    payload = {
        name: torch.full_like(tensor, A_VALUE if "lora_A" in name else B_VALUE)
        for name, tensor in exported.items()
    }

    def attempt(state):
        try:
            fsdp_utils.layered_load_lora_params(wrapped, state)
            return None
        except Exception as error:  # noqa: BLE001 - the type is part of what we report
            return f"{type(error).__name__}: {error}"

    error = attempt(payload)
    check(error is None, f"{label}: loading a correct adapter raised {error}", failures)

    reloaded = fsdp_utils.layered_summon_lora_params(wrapped)
    for name, tensor in reloaded.items():
        want = A_VALUE if "lora_A" in name else B_VALUE
        check(
            torch.allclose(tensor.float(), torch.full_like(tensor.float(), want)),
            f"{label}: {name} did not take the checkpoint values",
            failures,
        )

    snapshot = {name: tensor.clone() for name, tensor in reloaded.items()}
    rejections = {
        "missing factor": {k: v for k, v in payload.items() if "lora_A" in k},
        "unknown key": {**payload, "base_model.model.proj.lora_C.weight": torch.zeros(2, 8)},
        "scalar instead of a matrix": {k: torch.tensor([5.0]) for k in payload},
    }
    for why, state in rejections.items():
        check(attempt(state) is not None, f"{label}: accepted a checkpoint with a {why}", failures)
        after = fsdp_utils.layered_summon_lora_params(wrapped)
        for name, tensor in after.items():
            check(
                torch.equal(tensor, snapshot[name]),
                f"{label}: a rejected checkpoint ({why}) still changed {name}",
                failures,
            )


def main():
    stub_verl()
    fsdp_utils = load_fsdp_utils(sys.argv[1])
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    failures = []
    for strategy, strategy_name in (
        (ShardingStrategy.FULL_SHARD, "FULL_SHARD"),
        (ShardingStrategy.NO_SHARD, "NO_SHARD"),
    ):
        for use_orig_params in (True, False):
            for layout in ("leaf", "root_only", "root_and_child"):
                label = f"{strategy_name}/orig={use_orig_params}/{layout}"
                try:
                    wrapped = build(use_orig_params, strategy, layout)
                except ValueError as error:
                    # FSDP1 refuses a flat parameter that mixes frozen and trainable
                    # tensors when use_orig_params=False, so those layouts cannot
                    # exist at all. Not reaching them is the correct outcome.
                    if "uniform `requires_grad`" in str(error):
                        if dist.get_rank() == 0:
                            print(f"  {label}: unreachable layout, FSDP refuses it")
                        continue
                    raise
                run_case(fsdp_utils, wrapped, label, failures)
                if dist.get_rank() == 0 and not failures:
                    print(f"  {label}: export, load and all three rejections held")

    dist.barrier()
    if dist.get_rank() == 0:
        if failures:
            print("FAILED")
            for failure in failures:
                print(f"  {failure}")
        else:
            print("layered LoRA checkpoint contract holds in every mode")
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
