"""Which tensors a LoRA actually creates here, counted on the model.

A target plan that lists ``gate_proj``/``up_proj``/``down_proj`` reads as expert
coverage, but on the pinned Transformers the routed experts are packed
``nn.Parameter`` tensors rather than ``nn.Linear`` modules, and PEFT only wraps
modules. Those suffixes match the *shared* expert linears instead, so the plan
can look complete while every routed expert stays untouched.

By default this counts the route the trainer takes, not the shorthand: verl does
not hand PEFT the string ``all-linear`` -- ``HFModelConfig`` resolves it through
``build_glm5_next_lora_adapter_plan`` first and substitutes its own
``exclude_modules``. Counting the raw shorthand measures a different injection.
``--generic`` reports that one, for comparison.

Measured on the 9B surgery checkpoint at rank 16:

    trainer plan     528 tensors, 10,719,744 parameters
    raw all-linear   572 tensors, 11,106,688 parameters

The 44-tensor gap is the DSA indexer and the embedding/head extras the plan
keeps opt-in. Neither route touches the 46 packed routed-expert parameters.

No weights are loaded: only the module structure matters, so the model is built
from the config on the meta device. The counts are unaffected.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import re

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
    parser.add_argument("--alpha", type=int, default=0)
    parser.add_argument("--exclude", action="append", default=None,
                        help="дополнительные exclude_modules, как в конфиге тренера")
    parser.add_argument("--generic", action="store_true",
                        help="считать по сырому all-linear, без плана тренера")
    parser.add_argument("--json", help="куда записать пофамильный отчёт")
    args = parser.parse_args()

    import transformers
    from transformers import AutoConfig
    from peft import LoraConfig, get_peft_model

    # Резолвер плана не тянет остальной verl (только stdlib), поэтому грузим его
    # файлом: перепись должна работать и там, где пакет не установлен.
    plan_path = pathlib.Path(__file__).resolve().parents[2] / "verl/workers/config/lora_adapter_plan.py"
    spec = importlib.util.spec_from_file_location("lora_adapter_plan", plan_path)
    plan_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_module)
    build_glm5_next_lora_adapter_plan = plan_module.build_glm5_next_lora_adapter_plan

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Берём класс из самого конфига: glm5_next зарегистрирован не под AutoModelForCausalLM.
    cls = getattr(transformers, config.architectures[0])
    with torch.device("meta"):
        model = cls(config)

    # Тренер не отдаёт PEFT строку "all-linear" как есть: HFModelConfig сначала
    # резолвит её планом и подставляет собственные exclude_modules. Перепись по
    # сырому all-linear считала бы другой маршрут, чем тот, что реально обучается.
    alpha = args.alpha or 2 * args.rank
    plan = None if args.generic else build_glm5_next_lora_adapter_plan(
        config, "all-linear", rank=args.rank, alpha=alpha, exclude_modules=args.exclude
    )
    if plan is not None:
        target_modules = plan["target_modules"]
        exclude_modules = plan["trainer_exclude_modules"]
        route = "план тренера"
    else:
        target_modules = "all-linear"
        exclude_modules = args.exclude
        route = "сырой all-linear"
    print(f"маршрут: {route}; целей {len(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules}")

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

    lora_kwargs = dict(r=args.rank, lora_alpha=alpha, target_modules=target_modules,
                       lora_dropout=0.0, bias="none")
    if exclude_modules:
        lora_kwargs["exclude_modules"] = exclude_modules
    peft_model = get_peft_model(model, LoraConfig(**lora_kwargs))
    created = collections.Counter()
    params = collections.Counter()
    tensors = []
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue
        created[bucket(name)] += 1
        params[bucket(name)] += param.numel()
        tensors.append({"name": name, "group": bucket(name), "shape": list(param.shape),
                        "numel": param.numel(), "requires_grad": bool(param.requires_grad)})
    total = sum(params.values())
    print(f"\nсозданные адаптерные тензоры (rank={args.rank}):")
    for key in sorted(set(created) | set(params)):
        print(f"  {key:16s} тензоров {created[key]:5d}  параметров {params[key]:,}")
    print(f"  {'ИТОГО':16s} тензоров {sum(created.values()):5d}  параметров {total:,}")
    trainable = sum(1 for x in tensors if x["requires_grad"])
    print(f"  из них requires_grad: {trainable}")
    if args.json:
        fingerprint = hashlib.sha256(
            json.dumps({"targets": target_modules, "exclude": exclude_modules},
                       sort_keys=True, default=str).encode()
        ).hexdigest()
        pathlib.Path(args.json).write_text(json.dumps(
            {"model": args.model, "route": route, "rank": args.rank, "alpha": alpha,
             "plan_fingerprint": fingerprint, "target_modules": target_modules,
             "exclude_modules": exclude_modules,
             "linear_modules": dict(linears), "packed_parameters": dict(packed),
             "adapter_tensors": tensors,
             "totals": {"tensors": sum(created.values()), "parameters": total}},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  отчёт: {args.json}")


if __name__ == "__main__":
    main()
