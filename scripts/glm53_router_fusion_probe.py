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
"""Compare the fused and unfused MoE router on one batch, in one process.

``TopKRouter.routing`` reads ``fused=self.config.moe_router_fusion`` at call time, so the two paths
can be run against the same weights, the same batch and the same RNG state by flipping one boolean
between forwards. That removes every source of difference except the one under test -- which a
two-run comparison cannot do, because two runs also differ by their own nondeterminism.

What separates the two explanations:

    routing maps differ          the fused kernel picks different experts -> semantic, not rounding
    maps equal, probs differ     same experts, different weights -> arithmetic inside the kernel
    both equal                   the router is not the source; look at permute / shared-expert overlap

Usage: call :func:`compare_router_paths` with a built model, a forward callable and the
TransformerConfig the routers read. Everything else is the caller's harness.
"""

from __future__ import annotations

import torch


def _routers(model):
    """Every module that owns the fused/unfused decision, in a stable order."""
    found = []
    for module in ([model] if not isinstance(model, list) else model):
        for name, child in module.named_modules():
            if type(child).__name__ == "TopKRouter":
                found.append((name, child))
    return found


def expert_bias_scale(model) -> float:
    """Largest absolute expert bias in the model, which decides whether this test can see anything.

    With ``moe_router_enable_expert_bias`` the reference path selects experts by ``scores + bias``
    and weights them by ``scores`` alone. When the bias is all zeros -- an untrained or freshly
    initialised checkpoint -- those two agree, the branch is degenerate, and the fused and unfused
    kernels can only differ by rounding. A verdict of "arithmetic" then says nothing about a trained
    model, where the bias is what makes the selection contestable in the first place.
    """
    largest = 0.0
    for _name, router in _routers(model):
        bias = getattr(router, "expert_bias", None)
        if bias is not None:
            largest = max(largest, float(bias.detach().abs().max()))
    return largest


def _capture(model, sink: dict):
    """Record each router's (probs, routing_map) without touching what it returns."""
    handles = []
    for name, router in _routers(model):

        def hook(_module, _args, output, name=name):
            probs, routing_map = output[0], output[1]
            sink[name] = (probs.detach().float().cpu(), routing_map.detach().cpu())

        handles.append(router.register_forward_hook(hook))
    return handles


def compare_router_paths(model, forward_once, tf_config) -> dict:
    """Run the batch twice, fused and unfused, and report where the two disagree.

    Args:
        model: the built model (or the vpp list), used only to find the routers.
        forward_once: zero-argument callable running one forward on the batch, returning the loss.
        tf_config: the TransformerConfig the routers read ``moe_router_fusion`` from.
    """
    original = tf_config.moe_router_fusion
    results = {}
    try:
        for fused in (False, True):
            tf_config.moe_router_fusion = fused
            sink: dict = {}
            handles = _capture(model, sink)
            try:
                with torch.no_grad():
                    loss = forward_once()
            finally:
                for handle in handles:
                    handle.remove()
            results[fused] = (float(loss), sink)
    finally:
        tf_config.moe_router_fusion = original

    (loss_ref, ref), (loss_fused, fused_sink) = results[False], results[True]
    layers = sorted(set(ref) & set(fused_sink))
    report = {
        "layers_compared": len(layers),
        "loss_unfused": loss_ref,
        "loss_fused": loss_fused,
        "loss_delta": loss_fused - loss_ref,
        "routing_differs_in_layers": [],
        "max_abs_probs_delta": 0.0,
        "tokens_routed_differently": 0,
    }
    for name in layers:
        probs_ref, map_ref = ref[name]
        probs_new, map_new = fused_sink[name]
        if map_ref.shape == map_new.shape and not torch.equal(map_ref, map_new):
            report["routing_differs_in_layers"].append(name)
            # One token counts once, however many of its experts changed.
            differing = (map_ref != map_new).any(dim=-1).sum().item()
            report["tokens_routed_differently"] += int(differing)
        if probs_ref.shape == probs_new.shape:
            delta = (probs_ref - probs_new).abs().max().item()
            report["max_abs_probs_delta"] = max(report["max_abs_probs_delta"], delta)

    report["expert_bias_scale"] = expert_bias_scale(model)
    report["verdict"] = (
        "semantic: the fused kernel selects different experts"
        if report["routing_differs_in_layers"]
        else "arithmetic: same experts, different weights"
        if report["max_abs_probs_delta"] > 0
        else "identical: the router is not the source of the loss difference"
    )
    if not report["routing_differs_in_layers"] and report["expert_bias_scale"] == 0.0:
        # Refuse to let a degenerate run read as a clean bill of health.
        report["verdict"] += (
            " -- but every expert bias is zero, so selection could not have differed here whatever "
            "the kernel does; this says nothing about a trained checkpoint"
        )
    return report
