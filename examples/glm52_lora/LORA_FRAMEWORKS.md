# GLM-5.2 LoRA framework decision

Snapshot date: 2026-09-02.

## Decision

Use VERL's Megatron trainer with NVIDIA Megatron Bridge for the first full
GLM-5.2 LoRA SFT. Import the exact official block-FP8 checkpoint and
dequantize each source tensor to resident BF16 before Megatron mapping. This
is BF16 LoRA training from the official FP8 artifact, not FP8 compute or
QLoRA. Keep SGLang for held-out generation and optional
constraint-GRPO after SFT. Do not put AutoModel's trainer inside VERL and do
not use Slime as the LoRA trainer.

The first full run stays rank 16 / alpha 32 on the five MLA projections:

```text
linear_q_down_proj
linear_q_up_proj
linear_kv_down_proj
linear_kv_up_proj
linear_proj
```

The full checkpoint header gate reads no tensor payload. It found 141 shards,
59,044 E4M3 weights, and exactly 59,044 FP32 inverse-scale tensors, with every
scale matching the declared 128x128 block geometry. The 78-layer policy part
is 745,584,507,456 source bytes and expands to 1,486,754,078,208 bytes in BF16,
exactly matching the previously audited BF16 policy checkpoint. At
W128/TP8/EP128, expert tensors are read once while non-expert tensors are read
by 128 ranks, for 3.061 TiB of logical source traffic. The topology remains a
valid process grid, but the worst observed surgery-memory calibration now
rejects its seq768 envelope on 80-GiB devices. It must not launch at that
length without a separately qualified memory reduction.

Megatron Bridge merged the required
[GLM-5 FP8 import fix](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5851)
as `44c871b`. It applies each weight's own `_scale_inv`, including compound
gated-MLP mappings, before conversion. Our qualified dependency stack stays
on Bridge `d0c6228` and cherry-picks that exact two-file change; the launcher
locks its stable diff SHA-256. This avoids silently upgrading unrelated Bridge
dependencies while preserving the upstream implementation.

The import path was exercised on the 9B surgery pair against a BF16 control.
Both completed two SFT updates with finite gradients and the same 16.852 GiB
CUDA allocation peak. The FP8-source loss was `13.4675 -> 12.3566`; the BF16
control was `14.0332 -> 12.3654`. Both exported 13,608,960-parameter adapters
with every LoRA-B tensor nonzero. This validates import and training mechanics,
not full-model quality.

This is the smallest profile with direct full-model evidence. NVIDIA's current
[verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08fc1c60ecb7cb5421ef3fdab0494d9a5b65e678/examples/model_verification_cards/glm5-2/card.yaml)
records two 100-step BF16 full-model LoRA runs on these five targets at rank 8:
208 H100s at TP1/PP13/EP16 and 192 GB200s at TP1/PP6/EP32. The H100 run's loss
fell from 1.909421 to 0.8835775 with final gradient norm 0.101 and exactly 790
adapter tensors; the GB200 loss fell from 1.831100 to 0.8957523.

Those runs pin `zai-org/GLM-5.2@4d67f66c`. Our current checkpoint lock is
`cf457fa7`; the two `config.json` files differ only by the newer explicit
`"moe_router_dtype": "float32"` field. Weight and runtime validation must
still use our exact newer revision rather than silently inheriting the older
card's claim.

Our rank-16 profile has 106,149,888 trainable parameters over the 78 policy
layers. That is 202.47 MiB for BF16 adapter weights and 1.780 GiB for a
conservative unsharded 18-byte-per-parameter training bundle. Megatron's
documented BF16/FP32-gradient formula is `6 + 12/d` bytes per trainable
parameter with distributed-optimizer DP size `d`; 18 bytes is exact for DP1
and remains a conservative bound for the larger candidate grids. Under our
TP8 layout, Megatron replicates the q/kv down-projection factors but shards the
other three adapters: each rank holds 29,552,640 adapter parameters, or 0.495
GiB under that bound. If the separate
MTP layer is enabled and matched too, the count becomes 107,510,784. The
current VERL quality profile deliberately disables MTP, as does the surgery
fixture; MTP needs its own gate.

## What to borrow from each project

### NeMo AutoModel

The current
[GLM-5.2 LoRA recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/013906e2cdbeacad034030b3e71a833e965e0400/examples/llm_finetune/glm/glm_5.2_lora.yaml)
is a useful independent FSDP2 implementation. It uses EP128, packed THD at
4K, rank 32 / alpha 64, FP32 routing, activation checkpointing, gradient
clipping at 1.0, chunked cross entropy, and a dequantized base checkpoint.
Borrow those numerical and data-path choices.
Do not treat it as a drop-in VERL backend: it owns the model, distributed
layout, optimizer, checkpointing, and training loop.

AutoModel targets MLA, dense/shared MLPs, and every routed expert. At rank 16,
the 78 policy layers would contain:

| Target set | Trainable parameters |
|---|---:|
| Five MLA projections | 106,149,888 |
| MLA + dense/shared MLP | 138,295,296 |
| MLA + all MLP + per-expert LoRA | 5,800,605,696 |

The routed experts alone add 5,662,310,400 parameters: 256 independent
rank-16 expert updates in each of 75 MoE layers. That is a valid large-run
design, but a poor first experiment for 2,728 examples because expert coverage
depends on frozen routing and the adapter ceases to be small. Consider it only
after the MLA-only held-out comparison, with expert-activation coverage and
per-expert gradient measurements.

### Megatron Bridge and Baseten

NVIDIA's
[GLM bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08fc1c60ecb7cb5421ef3fdab0494d9a5b65e678/src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py)
is the conversion and model-provider layer already used by VERL. It preserves
MLA, DSA IndexShare, MoE routing, and reshardable distributed checkpoints.
Use it directly rather than writing a second GLM conversion stack.

The installed Bridge snapshot used by our surgery gates contains NVIDIA's
merged [replicated-adapter fix](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5900).
Without it, duplicated MLA q/kv down projections silently combine different
sequence-parallel token shards and produce wrong adapter outputs and
gradients. Our TP2 save/resume/reload gate and complementary EP2 routed-expert
gate both passed on that code; their evidence roots are recorded in the main
README.

Baseten's
[native GLM-5.2 FP8 expert import](https://github.com/basetenlabs/Megatron-Bridge/commit/e6ab3619a95f)
keeps E4M3 payloads and FP32 inverse scales without dequantizing routed
experts. Its checks require complete 128x128 blocks and rowwise-only
Transformer Engine storage. This is relevant to a future native-FP8/QLoRA
track, not a reason to change the first BF16 trainer run.

### NeMo RL

[NeMo RL LoRA](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/lora.md)
supports SFT, GRPO, and DPO with either the AutoModel DTensor v2 backend or the
Megatron backend. It is a separate end-to-end trainer and a useful parity
oracle. Its AutoModel Triton LoRA path must be disabled for TP greater than
one. Generic LoRA support is not GLM-5.2 qualification: the dedicated
[GLM-5.2 NeMo RL issue](https://github.com/NVIDIA-NeMo/RL/issues/3172) remains
open and explicitly asks for GRPO support. Adopting it would be an explicit
framework comparison, not an AutoModel component inserted into VERL.

### Slime

Slime's
[full GLM-5.2 example](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/docs/en/examples/glm5.2-744B-A40B.md)
is valuable for BF16-training/FP8-rollout topology, torch-dist conversion,
IndexShare-aware pipeline boundaries, and SGLang deployment. The current tree
has no PEFT/LoRA training implementation; `q_lora_rank` and `kv_lora_rank` in
the model script are MLA architecture dimensions. Use Slime later as a
full-parameter RL or rollout parity reference, not for this LoRA SFT.

### Axolotl

Axolotl's
[NVFP4 ScatterMoE recipe](https://github.com/axolotl-ai-cloud/axolotl/blob/6461c03b602bf0410f0388f1d029ba51e84aeaa2/examples/glm_moe_dsa/glm-5.2-nvfp4-lora.yaml)
shows that an aggressively quantized base can make GLM-5.2 LoRA feasible on
two B200s or eight H100s. It uses FSDP2, pure EP, rank 16, MLA plus MLP and
routed-expert adapters. Keep this as a low-memory ablation, not the reference
path: its custom NVFP4/ScatterMoE stack differs from the official checkpoint,
and
[adapter merge is currently unresolved](https://github.com/axolotl-ai-cloud/axolotl/issues/3773)
for that exact model path.

## Surgery boundary

Our BF16/FP8 surgery pair is the primary systems oracle because it keeps the
real width, MLA/DSA dimensions, dense-to-MoE transition, selected real donor
weights, and mixed E4M3/BF16 tensor contract. It deliberately reduces depth to
10, routed experts to 16, and sets `num_nextn_predict_layers=0`. It can qualify
losses, gradients, memory, adapter save/reload, and trainer-to-rollout weight
sync. It cannot establish language quality, all-256-expert coverage, MTP
behavior, or a multi-rank topology that has not itself been run.

The experiment order is therefore:

1. MLA-only rank 16 on the pinned official FP8 checkpoint, dequantized to BF16
   during Bridge import. Keep the independently pinned BF16 artifact as the
   numerical oracle where it is available.
2. Compare base and adapter on identical held-out prompts with the locked
   decoding contract and paired evaluator.
3. If capacity is insufficient, ablate `lm_head`, then dense/shared MLP.
4. Add routed-expert LoRA only with routing-coverage evidence.
5. Run constraint GRPO only after SFT improves Russian, Markdown, and
   accidental-Han metrics without semantic regression.
