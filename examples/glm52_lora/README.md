# GLM-5.2 LoRA validation

This directory validates LoRA SFT and GRPO on our own full-width GLM-5.2
surgery pair. The checkpoints are test fixtures, not chat or benchmark models.

- BF16 trainer anchor:
  [imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy](https://huggingface.co/imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy)
- mixed E4M3/BF16 rollout twin:
  `imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy-FP8`
- pair ID: `glm52-9b-2db222dcbd5d236a`

Both checkpoints keep the original width and MLA/DSA geometry. They use the
same ten donor layers and the same 16 experts in every retained MoE layer.

## Sequence

1. Create the deterministic SFT and GRPO smoke data:

   ```bash
   python examples/glm52_lora/prepare_smoke_data.py /path/to/data
   ```

2. Run two BF16 Megatron Bridge SFT steps on an audited free GPU:

   ```bash
   GPU_ID=5 \
   TRAIN_FILE=/path/to/data/sft.parquet \
   examples/glm52_lora/run_surgery_sft_megatron.sh
   ```

3. Verify the final adapter. The verifier requires the HF export, Megatron
   dist checkpoint, exact rank-16 MLA topology, finite BF16 tensors, and a
   nonzero update in every LoRA-B tensor:

   ```bash
   python examples/glm52_lora/verify_surgery_adapter.py \
     runs/glm52-lora-surgery-sft-megatron/global_step_2
   ```

4. Pass the Megatron adapter checkpoint to the FP8 rollout run:

   ```bash
   GPU_ID=5 \
   TRAIN_FILE=/path/to/data/rl.parquet \
   SFT_ADAPTER_CKPT=/path/to/global_step_2/model/dist_ckpt \
   examples/glm52_lora/run_surgery_grpo_megatron_sglang.sh
   ```

5. Verify the final PPO actor adapter and its Megatron dist checkpoint:

   ```bash
   python examples/glm52_lora/verify_surgery_adapter.py \
     runs/glm52-lora-surgery-grpo-megatron-sglang/global_step_2/actor
   ```

The first profile adapts only the five MLA projections. `lm_head`, the DSA
indexer, shared experts, and routed experts are separate ablations after the
MLA-only save/reload, sharding, hot-sync, and finite-gradient gates pass.

`setup_vm_env.sh` creates a new checkout and venv and refuses to overwrite an
existing environment. It installs all four GitHub forks with normal dependency
resolution, checks out Megatron Bridge at the recorded exact revision, and
installs the matching CUDA-13 Transformer Engine and NVIDIA ModelOpt version
recorded by Bridge (needed by adapter-only checkpoint filtering). Bridge is
imported directly from that exact source checkout because its published
all-model metadata pins an older Transformers and adds unrelated diffusion
dependencies. No dependency is installed with `--no-deps`; existing venvs and
Ray clusters are untouched.

Before downloading model shards, verify that the Hub config selects the GLM
bridge and preserves the surgery model's full-width MLA/DSA/MoE contract:

```bash
python examples/glm52_lora/verify_bridge_contract.py
```
