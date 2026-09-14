import pytest

from verl.utils.reward_score.math_dapo import compute_score


@pytest.mark.parametrize(
    "solution",
    [
        "Answer: 60",
        "Answer: $60$",
        "Answer: \\boxed{60}",
        "Answer: \\(60\\)",
        "Answer: \\[60\\]",
        "Answer: \\( \\boxed{60} \\)",
    ],
)
def test_latex_math_delimiters_do_not_change_the_answer(solution):
    assert compute_score(solution, "60")["acc"]


@pytest.mark.parametrize("solution", ["Answer: \\(61\\)", "Answer: \\[6\\]0", "Answer: \\boxed{\\frac{1}{2}}"])
def test_wrong_answers_stay_wrong(solution):
    assert not compute_score(solution, "60")["acc"]


def test_fraction_inside_parenthesis_delimiters():
    assert compute_score("Answer: \\(\\frac{1}{2}\\)", "\\frac{1}{2}")["acc"]
