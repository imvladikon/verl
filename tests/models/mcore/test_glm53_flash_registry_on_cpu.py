from types import SimpleNamespace

from verl.models.mcore.registry import SupportedVLM, get_mcore_forward_fn, supported_vlm


def test_glm53_flash_is_registered_as_vlm():
    architecture = "Glm5NextForConditionalGeneration"
    assert SupportedVLM.GLM53_FLASH.value == architecture
    assert architecture in supported_vlm
    forward = get_mcore_forward_fn(SimpleNamespace(architectures=[architecture]))
    assert callable(forward)
