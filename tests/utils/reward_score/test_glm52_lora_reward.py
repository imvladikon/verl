import importlib.util
from pathlib import Path


reward_path = Path(__file__).resolve().parents[3] / "examples" / "glm52_lora" / "reward.py"
spec = importlib.util.spec_from_file_location("glm52_lora_reward_under_test", reward_path)
assert spec is not None and spec.loader is not None
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)

HAN_RE = reward.HAN_RE
balanced_markdown = reward.balanced_markdown
compute_score = reward.compute_score


def score(text: str, index: int) -> float:
    return compute_score(
        "glm52_lora_contract",
        text,
        "unused",
        {"index": index},
    )["score"]


def test_han_ranges_and_markdown_balance():
    assert HAN_RE.search("готово 完成")
    assert HAN_RE.search("готово 𠀀")
    assert balanced_markdown("## Заголовок\n\n```python\nprint('ok')\n```")
    assert not balanced_markdown("##Заголовок\n\n```python\nprint('ok')")


def test_objective_repairs_score_above_failures():
    assert score("Отчёт готов и сохранён.", 1) > score(
        "Отчёт готов 的 и сохранён.", 1
    )
    assert score("```python\nprint('готово')\n```", 4) > score("```python\nprint('готово')", 4)
    assert score("## Итоги проверки", 7) > score("#Итоги проверки", 7)
