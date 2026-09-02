#!/usr/bin/env python3
"""Verify that GLM-5.2 receives finite, non-zero LoRA gradients.

This is deliberately a Transformers + PEFT reference, not a VERL smoke.  It
isolates the model/adapter contract before distributed training and rollout
weight synchronization are introduced.  The base checkpoint remains frozen;
only LoRA A/B matrices may receive gradients or optimizer updates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTENTION_TARGETS = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument(
        "--target-profile",
        choices=("attention", "attention_lm_head"),
        default="attention_lm_head",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def finite_gradient_stats(model: torch.nn.Module) -> tuple[float, int, int]:
    squared_norm = 0.0
    tensors_with_gradient = 0
    nonzero_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise AssertionError(f"frozen parameter received a gradient: {name}")
            continue
        gradient = parameter.grad
        if gradient is None:
            raise AssertionError(f"trainable parameter has no gradient: {name}")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite LoRA gradient: {name}")
        tensors_with_gradient += 1
        nonzero_elements += int(torch.count_nonzero(gradient))
        squared_norm += float(torch.sum(gradient.float().square()))
    return math.sqrt(squared_norm), tensors_with_gradient, nonzero_elements


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cuda_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    ) if device.type == "cuda" else None
    if args.rank <= 0 or args.alpha <= 0 or args.steps <= 0:
        raise ValueError("rank, alpha, and steps must be positive")
    if not torch.cuda.is_available() and device.type == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    weights_are_tied = input_weight is output_weight or (
        input_weight.untyped_storage().data_ptr()
        == output_weight.untyped_storage().data_ptr()
    )
    if bool(model.config.tie_word_embeddings) != weights_are_tied:
        raise AssertionError(
            "tie_word_embeddings disagrees with the loaded parameter storage: "
            f"config={model.config.tie_word_embeddings}, storage={weights_are_tied}"
        )

    targets = list(ATTENTION_TARGETS)
    if args.target_profile == "attention_lm_head":
        targets.append("lm_head")
    # Transformers declares the potential lm_head -> embed_tokens tie in
    # _tied_weights_keys even when tie_word_embeddings=False.  PEFT warns from
    # that declaration alone.  The storage assertion above distinguishes the
    # untied GLM-5.2 checkpoint from a genuinely tied model.
    with warnings.catch_warnings():
        if not weights_are_tied:
            warnings.filterwarnings(
                "ignore",
                message=r"Model has `tie_word_embeddings=True` and a tied layer.*",
                category=UserWarning,
            )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.rank,
                lora_alpha=args.alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
                ensure_weight_tying=weights_are_tied,
            ),
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("PEFT created no trainable parameters")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise AssertionError(f"non-LoRA parameters are trainable: {unexpected[:8]}")
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)

    prompt_messages = [
        {"role": "system", "content": "Отвечай только по-русски."},
        {"role": "user", "content": "Исправь Markdown."},
    ]
    completion = "## Заголовок\n\n- Первый пункт\n- Второй пункт"
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError("the full chat is not prefixed by the generation prompt")
    full_ids = full_ids[: args.max_length]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("max_length leaves no assistant completion tokens")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    supervised_tokens = int(torch.count_nonzero(labels != -100))
    if supervised_tokens != len(full_ids) - len(prompt_ids):
        raise AssertionError("assistant-only label mask has the wrong size")

    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in trainable), lr=args.learning_rate
    )
    step_metrics: list[dict[str, float | int]] = []
    started = time.monotonic()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=labels).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        grad_norm, gradient_tensors, nonzero_gradient_elements = finite_gradient_stats(model)
        if not math.isfinite(grad_norm) or grad_norm <= 0 or nonzero_gradient_elements <= 0:
            raise FloatingPointError(
                f"invalid LoRA gradient at step {step}: norm={grad_norm}, "
                f"nonzero={nonzero_gradient_elements}"
            )
        optimizer.step()
        step_metrics.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": grad_norm,
                "gradient_tensors": gradient_tensors,
                "nonzero_gradient_elements": nonzero_gradient_elements,
            }
        )

    result = {
        "model": str(args.model.resolve()),
        "model_type": model.config.model_type,
        "dtype": "bfloat16",
        "device": str(device),
        "target_modules": targets,
        "weights_are_tied": weights_are_tied,
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_parameters": trainable_parameters,
        "sequence_length": len(full_ids),
        "prompt_tokens": len(prompt_ids),
        "supervised_tokens": supervised_tokens,
        "steps": step_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index)) if cuda_index is not None else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(cuda_index)) if cuda_index is not None else None
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
