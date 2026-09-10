#!/usr/bin/env python3
"""Reload the exported HF adapter and prove that it changes finite logits."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy",
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this reload gate requires an audited CUDA device")
    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(cuda_index)
    torch.cuda.reset_peak_memory_stats(cuda_index)

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()

    batch = tokenizer(
        "Исправь Markdown: #заголовок без пробела",
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)
    with torch.inference_mode():
        with model.disable_adapter():
            base_logits = model(**batch, use_cache=False).logits.float()
        adapter_logits = model(**batch, use_cache=False).logits.float()

    if not bool(torch.isfinite(base_logits).all()):
        raise FloatingPointError("base logits are non-finite after reload")
    if not bool(torch.isfinite(adapter_logits).all()):
        raise FloatingPointError("adapter logits are non-finite after reload")
    delta = adapter_logits - base_logits
    delta_l2 = float(torch.linalg.vector_norm(delta))
    delta_max = float(delta.abs().max())
    if not math.isfinite(delta_l2) or delta_l2 <= 0 or delta_max <= 0:
        raise AssertionError(
            f"reloaded adapter did not change logits: l2={delta_l2}, max={delta_max}"
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "input_tokens": int(batch.input_ids.numel()),
        "adapter_trainable_parameters": trainable,
        "logit_delta_l2": delta_l2,
        "logit_delta_max": delta_max,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_index)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_index)),
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
