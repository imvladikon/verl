"""Which tensors a LoRA actually creates here, counted on the model.

Suffix names do not settle expert coverage. On the pinned Transformers the
routed experts are packed ``nn.Parameter`` tensors rather than ``nn.Linear``
modules, so the module route never reaches them, and the
``gate_proj``/``up_proj``/``down_proj`` entries a target plan lists belong to the
shared experts instead. PEFT can adapt packed parameters through
``target_parameters``; that is a separate route, reported here only when asked
for, because its cost is a different order.

By default this counts the route the trainer takes, not the shorthand: verl does
not hand PEFT the string ``all-linear`` -- ``HFModelConfig`` resolves it through
``build_glm5_next_lora_adapter_plan`` first and substitutes its own
``exclude_modules``. ``--generic`` reports the shorthand for comparison.

Measured at rank 16 on the 24-layer surgery checkpoint (config sha256
76f89e74..., 18 KDA + 6 DSA layers, 128 routed and 1 shared expert,
Transformers 5.16.1, PEFT 0.20):

    trainer plan                       528 tensors,  10,719,744 parameters
    raw all-linear                     572 tensors,  11,106,688 parameters
    plan + packed routed experts       620 tensors, 227,774,976 parameters

The 44-tensor gap between the first two is the DSA indexer and the
embedding/head extras the plan keeps opt-in. The third line is why routed-expert
LoRA is a decision rather than a default: it is twenty times the adapter, with
the memory, optimizer and weight-sync profile that follows.

Counts belong to a configuration, not to a model name, so the JSON report
records the resolved config hash, the geometry it came from and the dependency
versions alongside the plan fingerprint.

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


def _text_value(config, name, default=None):
    """Поле берётся из text_config, если оно там, иначе из корня."""
    text = getattr(config, "text_config", None)
    if text is not None and getattr(text, name, None) is not None:
        return getattr(text, name)
    return getattr(config, name, default)


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
    parser.add_argument("--target-parameters", action="append", default=None,
                        help="суффиксы packed-параметров (down_proj, gate_up_proj): маршрут PEFT "
                             "target_parameters, которым маршрутизируемые эксперты достижимы")
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

    geometry = {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", []) or []),
        "hidden_size": _text_value(config, "hidden_size"),
        "num_hidden_layers": _text_value(config, "num_hidden_layers"),
        "n_routed_experts": _text_value(config, "n_routed_experts"),
        "n_shared_experts": _text_value(config, "n_shared_experts"),
        "layer_types": collections.Counter(_text_value(config, "layer_types") or []),
    }
    config_hash = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()
    print("конфиг: %s слоёв, hidden %s, экспертов %s/%s, sha256 %s" % (
        geometry["num_hidden_layers"], geometry["hidden_size"],
        geometry["n_routed_experts"], geometry["n_shared_experts"], config_hash[:12]))
    print("типы слоёв:", dict(geometry["layer_types"]))
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
    if args.target_parameters:
        # Модульный маршрут обходит packed-эксперты стороной, но PEFT умеет
        # адаптировать и сами параметры. Это другой профиль, поэтому включается явно.
        lora_kwargs["target_parameters"] = args.target_parameters
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
        # Отпечаток берёт и план, и конфиг: числа переписи привязаны к конкретной
        # геометрии, а имя вроде "9B surgery" её не определяет.
        fingerprint = hashlib.sha256(
            json.dumps({"targets": target_modules, "exclude": exclude_modules,
                        "target_parameters": args.target_parameters,
                        "config_sha256": config_hash, "rank": args.rank, "alpha": alpha},
                       sort_keys=True, default=str).encode()
        ).hexdigest()
        pathlib.Path(args.json).write_text(json.dumps(
            {"model": args.model, "route": route, "rank": args.rank, "alpha": alpha,
             "plan_fingerprint": fingerprint, "config_sha256": config_hash,
             "geometry": {**geometry, "layer_types": dict(geometry["layer_types"])},
             "versions": {"transformers": transformers.__version__,
                          "peft": __import__("peft").__version__,
                          "torch": torch.__version__},
             "target_modules": target_modules, "exclude_modules": exclude_modules,
             "target_parameters": args.target_parameters,
             "linear_modules": dict(linears), "packed_parameters": dict(packed),
             "adapter_tensors": tensors,
             "totals": {"tensors": sum(created.values()), "parameters": total}},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  отчёт: {args.json}")


if __name__ == "__main__":
    main()
