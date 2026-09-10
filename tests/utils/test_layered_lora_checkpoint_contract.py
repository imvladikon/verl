"""An adapter checkpoint must be rejected before it is half-applied.

layered_load_lora_params used to check only "incoming keys we did not consume",
so a checkpoint missing half the adapter loaded without complaint and left those
factors at their init values. There was no shape check either, and Tensor.copy_
broadcasts, so a scalar filled a whole matrix. The one error it did raise came
after the copies, leaving the adapter partly overwritten.

Run with two GPUs:

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        test_layered_lora_checkpoint_contract.py <verl/utils/fsdp_utils.py>

Expected: the full checkpoint loads (A sums to 48, B to 112); a missing factor,
a wrong shape and an unknown key each raise, and in every failing case both
factors stay at zero — nothing was written.

FSDP and PEFT are real. Two verl helper modules are stubbed because importing
them pulls in ray and the rest of the stack; neither is touched by the functions
under test. FSDP1 needs an accelerator in torch 2.13, so this does not run on CPU.
"""
import importlib.util, os, sys, types
import torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def stub_verl():
    for name in ("verl", "verl.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    dev = types.ModuleType("verl.utils.device")
    dev.get_device_id = lambda: torch.device("cpu")
    dev.get_device_name = lambda: "cpu"
    dev.get_torch_device = lambda: types.SimpleNamespace(empty_cache=lambda: None)
    sys.modules["verl.utils.device"] = dev
    mdl = types.ModuleType("verl.utils.model")
    mdl.check_exclude_modules = lambda *a, **k: False
    mdl.check_target_modules = lambda *a, **k: True
    sys.modules["verl.utils.model"] = mdl


def load_fsdp_utils(path):
    spec = importlib.util.spec_from_file_location("fu", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fu"] = mod
    spec.loader.exec_module(mod)
    return mod


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


def build():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(Base(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"],
                                              lora_dropout=0.0, bias="none"))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.fill_(0.0)
    from peft.tuners.lora import Linear as LoraLinear
    if torch.cuda.is_available():
        model = model.cuda()
    wrapped = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                   auto_wrap_policy=ModuleWrapPolicy({LoraLinear}),
                   use_orig_params=True,
                   device_id=torch.cuda.current_device() if torch.cuda.is_available() else None)
    # Один форвард: без него хендлы FSDP не инициализированы и summon падает.
    with torch.no_grad():
        device = next(wrapped.parameters()).device
        wrapped(torch.zeros(1, 8, device=device))
    return wrapped


def case(fu, label, make_params, rank):
    wrapped = build()
    reference = fu.layered_summon_lora_params(wrapped)
    params = make_params(reference)
    error = None
    try:
        fu.layered_load_lora_params(wrapped, params)
    except Exception as exc:
        error = f"{type(exc).__name__}"
    got = fu.layered_summon_lora_params(wrapped)
    sums = {k.split(".")[-2] + "." + k.split(".")[-1]: round(float(v.sum()), 3)
            for k, v in got.items()}
    if rank == 0:
        print(f"  {label:32s} ошибка={str(error):18s} суммы={sums}")


def main():
    stub_verl()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    fu = load_fsdp_utils(sys.argv[1])

    def full(ref):
        return {k: torch.full_like(v, 3.0 if "lora_A" in k else 7.0) for k, v in ref.items()}

    def only_a(ref):
        return {k: v for k, v in full(ref).items() if "lora_A" in k}

    def scalars(ref):
        return {k: torch.tensor([5.0]) for k in ref}

    def extra(ref):
        out = full(ref)
        out["base_model.model.proj.lora_C.weight"] = torch.zeros(2, 8)
        return out

    if rank == 0:
        print("=== layered_load_lora_params: FSDP1, 2 ранга, настоящий PEFT")
        print("    ожидается: A -> сумма 48, B -> сумма 112")
    case(fu, "полный A+B правильной формы", full, rank)
    case(fu, "только A, B отсутствует", only_a, rank)
    case(fu, "A и B формы [1]", scalars, rank)
    case(fu, "A+B плюс неизвестный ключ", extra, rank)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
