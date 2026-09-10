#!/usr/bin/env python3
"""Does a Megatron-Bridge HF export claim the training seq_length as the model context?

Bridge's generic ``CONFIG_MAPPING`` pairs HF ``max_position_embeddings`` with Megatron
``seq_length``, so any export that regenerates the HF config from the provider (the
``AutoBridge.from_auto_config`` path behind ``run_conversion.py export``) writes the run's
sequence length where the model's context window belongs, and ``conform_config_to_reference``
does not restore it because the key exists in the reference config too.

This probe runs the real ``megatron_to_hf_config`` and ``conform_config_to_reference`` for
every GLM bridge against an installed Bridge, so the claim can be rechecked on a new pin
instead of taken from a changelog. It writes nothing and needs no GPU.

    python tests/utils/megatron_bridge_export_config_probe.py --training-seq-length 8192

Exit code 1 means at least one bridge lost the reference context length.
"""

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types


def _install_transformer_engine_stub():
    """Let the probe run in an environment without Transformer Engine.

    The two functions under test never touch TE; only the import chain does. The stub is
    installed solely when TE is genuinely absent, so a real environment is used as is.
    """
    if importlib.util.find_spec("transformer_engine") is not None:
        return False

    class _StubMeta(type):
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _StubMeta(name, (cls,), {})

    class _StubAny(metaclass=_StubMeta):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return _StubAny()

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _StubMeta(name, (_StubAny,), {})

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        prefixes = ("transformer_engine", "transformer_engine_torch")

        def find_spec(self, fullname, path=None, target=None):
            if not any(fullname == p or fullname.startswith(p + ".") for p in self.prefixes):
                return None
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

        def create_module(self, spec):
            module = _StubModule(spec.name)
            module.__file__ = f"<{spec.name} stub>"
            module.__path__ = []
            module.__version__ = "2.9.0"
            return module

        def exec_module(self, module):
            return None

    sys.meta_path.insert(0, _StubFinder())
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seq-length", type=int, default=8192)
    parser.add_argument("--reference-max-position-embeddings", type=int, default=1048576)
    args = parser.parse_args()

    stubbed = _install_transformer_engine_stub()

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.utils import conform_config_to_reference
    from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge
    from megatron.bridge.models.glm.glm47_flash_bridge import GLM47FlashBridge
    from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge
    from megatron.bridge.models.gpt_provider import GPTModelProvider
    from megatron.bridge.models.mla_provider import MLAModelProvider

    if stubbed:
        print("transformer_engine is absent; imported Bridge with a stub (unused by the code under test)")

    base = MegatronModelBridge.megatron_to_hf_config.__func__
    small = dict(num_layers=2, hidden_size=128, num_attention_heads=4, seq_length=args.training_seq_length)
    cases = [
        ("GLM-5 / 5.1 / 5.2 (glm_moe_dsa)", GLM5Bridge, MLAModelProvider(**small)),
        ("GLM-4.5 (glm4_moe)", GLM45Bridge, GPTModelProvider(**small)),
        ("GLM-4.7-Flash", GLM47FlashBridge, GPTModelProvider(**small)),
    ]

    reference = {"max_position_embeddings": args.reference_max_position_embeddings, "hidden_size": 128}
    lost = []
    for label, bridge_cls, provider in cases:
        generated = bridge_cls.megatron_to_hf_config(provider)
        conformed = conform_config_to_reference(generated, dict(reference))
        final = conformed.get("max_position_embeddings")
        verdict = "keeps reference" if final == reference["max_position_embeddings"] else "LOSES model context"
        print(
            f"{label:34s} overrides megatron_to_hf_config="
            f"{bridge_cls.megatron_to_hf_config.__func__ is not base!s:5s} "
            f"generated={generated.get('max_position_embeddings')} "
            f"after conform={final} {verdict}"
        )
        if final != reference["max_position_embeddings"]:
            lost.append(label)

    if lost:
        print(f"\n{len(lost)}/{len(cases)} bridges export the training sequence length as the model context")
        return 1
    print("\nall bridges preserve the reference context length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
