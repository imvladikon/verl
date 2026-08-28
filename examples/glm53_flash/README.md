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

## Pinned sources

- SGLang: `imvladikon/sglang@b26f2c18c00259a15e2829db9bfef96ea2365972`
- Megatron-Core: `imvladikon/Megatron-LM@366ae1b219a212c36be03923a05a207d174cbe22`
- AutoModel: `NVIDIA-NeMo/Automodel@9228f33cf73d66a9b2e84256d298aac9a70283f0`

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
to the exact commits above.

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

Megatron-Core is pinned because it is a dependency of the supported stacks and
its GLM mHC/recompute/routing-replay changes are tested in that fork. This
example does not claim an end-to-end Megatron actor: that path also needs a
GLM-5.3-Flash Megatron-Bridge model provider and checkpoint mapping, which is
not yet a user-owned pinned dependency.
