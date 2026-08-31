# GLM-5.3-Flash SFT and GRPO lifecycle

This example qualifies the complete tiny-model training loop used by this fork:

- supervised fine-tuning with VERL's AutoModel FSDP2 engine;
- two GRPO optimizer steps with VERL FSDP and asynchronous SGLang rollout;
- exact dependency provenance checks before GPU work starts;
- exact state-dict equality, finite-weight, component-update, inter-step-update,
  and frozen-vision checks after training.

The two environments are intentionally separate. Current AutoModel and SGLang
require incompatible CUDA-kernel dependency stacks, so combining them in one
extra would hide ABI conflicts rather than make the lifecycle reproducible.

## Tracked sources

- SGLang: `imvladikon/sglang@glm-5.3-flash`
- Megatron-Core: `imvladikon/Megatron-LM@glm-5.3-flash`
- AutoModel: `NVIDIA-NeMo/Automodel@9228f33cf73d66a9b2e84256d298aac9a70283f0`

`uv.lock` records the exact commit resolved from each tracked branch, and the
provenance gate compares the active installation with that locked commit.

The authoritative model contract is
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).
Serving references are the
[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) and the
[SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash).

## Run the empirical gates

Set `MODEL_PATH` to a local architecture-matched tiny checkpoint. The scripts
do not download the production checkpoint.

```bash
MODEL_PATH=/path/to/GLM-5.3-Flash-tiny \
  examples/glm53_flash/run_glm53_flash_automodel.sh

MODEL_PATH=/path/to/GLM-5.3-Flash-tiny \
  examples/glm53_flash/run_glm53_flash_fsdp.sh
```

Both scripts default to `uv run --frozen` and their matching extra. Set
`VERL_USE_UV=0` only when running inside an already-provisioned development
environment; the provenance gate still requires the active modules to resolve
to the exact commits in `uv.lock`.

The GRPO script keeps the production TileLang DSA backend enabled. On CUDA 12
hosts whose default GCC is newer than 12, it selects an installed `g++-12` or
`g++-11` before Ray starts. Set `TILELANG_CXX=/path/to/g++` to use a cluster
toolchain explicitly.

The tiny checkpoint must preserve the production architecture contracts that
matter here: `Glm5NextForConditionalGeneration`, KDA and DSA layers, mHC,
routed MoE plus router, and the vision tower. Passing this gate proves that
those paths load, backpropagate, update, checkpoint, roll out, and accept
weight refreshes. It does not by itself prove the memory fit of the 744B
production checkpoint; that requires a separate topology and memory budget.

## One-GPU 9B surgery smoke

The tiny lifecycle is an interface gate, not a numerical proxy for the 9B
surgery checkpoint. In particular, the tiny run uses a much larger learning
rate, does not create multi-billion-element FSDP flat parameters, and does not
exercise the same block-quantized FP8 weight-refresh path. Use the separate 9B
smoke on an otherwise free A100-80G:

```bash
CUDA_VISIBLE_DEVICES=5 \
MODEL_PATH=/path/to/GLM-5.3-Flash-9B-Surgery-Dummy \
  examples/glm53_flash/run_glm53_flash_9b_smoke.sh
```

This profile keeps the actor in BF16, dequantizes the FP8 checkpoint at load,
and requantizes actor updates for the SGLang rollout. At the tested learning
rate of `1e-6`, nearest-rounded BF16 AdamW discards most sampled updates. The
profile therefore uses TorchAO `AdamW8bit` with BF16 stochastic rounding and
`use_orig_params=true`. The latter is required here: TorchAO only quantizes an
optimizer buffer when its size is divisible by the 256-element block, while
the default FSDP flat-buffer lengths are not. In the two-step qualification,
202 parameters covering 8,895,656,704 elements used real `OptimState8bit`
first- and second-moment buffers; the 75 small fallback parameters covered only
15,036 elements. Both steps had finite gradients and completed rollout weight
sync.

The script intentionally does not save checkpoints: each BF16 actor snapshot
is about 18 GB. It is a one-GPU integration and gradient-flow smoke, not an
optimizer recommendation for the 744B production model. Production training
needs a sharded optimizer with an explicit master-weight/rounding policy and a
separate TP/EP memory qualification.

To distinguish real sharding from the world-size-one `NO_SHARD` fallback, run
one CPU optimizer step with multiple FSDP ranks:

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/glm53_flash/run_cpu_fsdp_sharding_smoke.py \
  --model /path/to/GLM-5.3-Flash-sharding-twin \
  --output /tmp/glm53-fsdp-sharding.json
```

The report fails unless every rank owns only its physical `FULL_SHARD` slice,
the distributed gradient is finite and nonzero, and the optimizer changes the
shards. This CPU gate isolates FSDP correctness; it is not a substitute for a
multi-GPU TP/EP throughput test.

Megatron-Core is branch-tracked because it is a dependency of the supported
stacks and its GLM mHC/recompute/routing-replay changes are tested in that
fork. This example does not claim an end-to-end Megatron actor: that path also
needs a GLM-5.3-Flash Megatron-Bridge model provider and checkpoint mapping.
