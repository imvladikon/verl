"""Which tensors an ``all-linear`` LoRA actually creates, counted on the model.

A target plan that lists ``gate_proj``/``up_proj``/``down_proj`` reads as expert
coverage, but on the pinned Transformers the routed experts are packed
``nn.Parameter`` tensors rather than ``nn.Linear`` modules, and PEFT only wraps
modules. Those suffixes match the *shared* expert linears instead, so the plan
can look complete while every routed expert stays untouched.

Measured on the 9B surgery checkpoint, rank 16: 572 adapter tensors totalling
11.1M parameters, split across attention (420), shared experts (138), dense MLP
(6) and 8 others -- and nothing on the 46 packed routed-expert parameters.

No weights are loaded: only the module structure matters, so the model is built
from the config on the meta device. The counts are unaffected.
"""
import argparse, collections, json, re
import torch


def bucket(name):
    if ".self_attn." in name:
        return "attention"
    if ".shared_expert" in name or ".shared_experts" in name:
        return "shared expert"
    if re.search(r"\.experts\b", name) or ".experts." in name:
        return "routed experts"
    if ".mlp." in name:
        return "dense MLP"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    import transformers
    from transformers import AutoConfig
    from peft import LoraConfig, get_peft_model

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Берём класс из самого конфига: glm5_next зарегистрирован не под AutoModelForCausalLM.
    cls = getattr(transformers, config.architectures[0])
    with torch.device("meta"):
        model = cls(config)

    linears = collections.Counter()
    packed = collections.Counter()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linears[bucket(name)] += 1
    for name, param in model.named_parameters():
        if param.dim() >= 3:
            packed[bucket(name)] += 1

    print("линейные слои (кандидаты all-linear):")
    for key, count in sorted(linears.items()):
        print(f"  {key:16s} {count}")
    print("packed-параметры (LoRA их не видит):")
    for key, count in sorted(packed.items()) or [("нет", 0)]:
        print(f"  {key:16s} {count}")

    peft_model = get_peft_model(
        model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules="all-linear",
                          lora_dropout=0.0, bias="none")
    )
    created = collections.Counter()
    params = collections.Counter()
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue
        created[bucket(name)] += 1
        params[bucket(name)] += param.numel()
    total = sum(params.values())
    print(f"\nсозданные адаптерные тензоры (rank={args.rank}):")
    for key in sorted(set(created) | set(params)):
        print(f"  {key:16s} тензоров {created[key]:5d}  параметров {params[key]:,}")
    print(f"  {'ИТОГО':16s} тензоров {sum(created.values()):5d}  параметров {total:,}")


if __name__ == "__main__":
    main()
