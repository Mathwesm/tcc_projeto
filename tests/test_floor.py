"""Tests for the trivial predictors.

These exist to put a number on how much of the score is available to anything that
counts keypresses. If they are wrong, the fine-tuned model gets credit it did not earn
-- or is denied credit it did.
"""

from __future__ import annotations

import math

import pytest

from centauro_lite.core.chunking import IGNORE_INDEX
from centauro_lite.core.floor import (
    LabelledWindow,
    answer_counts,
    marginal_result,
    recovered_fraction,
    scored_labels,
    uniform_result,
)


def _window(experiment: str, answers: list[int]) -> LabelledWindow:
    """Build a window whose scored tokens are exactly ``answers``.

    The leading ignore token stands for the window's first position, which the causal
    model cannot predict and which therefore never enters any score.
    """
    return LabelledWindow(experiment=experiment, participant="p0", labels=[IGNORE_INDEX, *answers])


def test_the_first_position_is_never_scored() -> None:
    """The neural metric drops index 0; the trivial ones must drop it too.

    Scoring it here would put the trivial predictors on a larger set than the model
    they are compared against, which tilts the comparison in a direction nobody chose.
    """
    window = LabelledWindow(experiment="e", participant="p0", labels=[7, 7, IGNORE_INDEX, 9])
    assert list(scored_labels(window)) == [7, 9]


def test_padding_is_not_an_answer() -> None:
    """Ignore tokens are structure, not behaviour."""
    window = _window("e", [IGNORE_INDEX, 5, IGNORE_INDEX])
    assert list(scored_labels(window)) == [5]


def test_answers_are_counted_per_experiment() -> None:
    """Answers are counted per experiment, never pooled.

    Different tasks use different keys, and pooling them would handicap the predictor
    for a reason that has nothing to do with behaviour.
    """
    counts = answer_counts([_window("a", [1, 1, 2]), _window("b", [3])])
    assert counts["a"] == {1: 2, 2: 1}
    assert counts["b"] == {3: 1}


def test_uniform_scores_the_entropy_of_ignorance() -> None:
    """With two answers seen plus one slot held for the unseen, uniform is 1/3."""
    train = [_window("e", [1, 2])]
    result = uniform_result(train, [_window("e", [1, 1, 2])])
    assert result.nll == pytest.approx(-math.log(1 / 3))


def test_marginal_beats_uniform_when_answers_are_skewed() -> None:
    """That gap is the whole point: it is what counting alone buys.

    A model that does not clear it has learned nothing a frequency table does not
    already know.
    """
    train = [_window("e", [1] * 90 + [2] * 10)]
    validation = [_window("e", [1] * 9 + [2])]
    assert marginal_result(train, validation).nll < uniform_result(train, validation).nll


def test_marginal_matches_the_hand_computed_probability() -> None:
    """Add-one smoothing over three slots: two answers seen, one held for the unseen."""
    train = [_window("e", [1, 1, 1, 2])]
    result = marginal_result(train, [_window("e", [1])])
    assert result.nll == pytest.approx(-math.log((3 + 1) / (4 + 3)))


def test_an_unseen_answer_costs_a_finite_amount() -> None:
    """An answer never seen in training must still cost a finite amount.

    Without smoothing one rare keypress would be infinitely surprising and would
    single-handedly decide the aggregate.
    """
    result = marginal_result([_window("e", [1, 1])], [_window("e", [99])])
    assert math.isfinite(result.nll)


def test_each_experiment_is_reported_separately() -> None:
    """Same reporting shape as the neural evaluation, so the rows sit in one table."""
    train = [_window("a", [1, 1, 1, 1]), _window("b", [1, 2, 3, 4])]
    result = marginal_result(train, [_window("a", [1]), _window("b", [1])])
    assert set(result.per_experiment) == {"a", "b"}
    # Answers concentrated on one key are easier to guess than answers spread over four.
    assert result.per_experiment["a"] < result.per_experiment["b"]


def test_recovered_fraction_places_a_score_on_a_measured_range() -> None:
    """Halfway between counting and the reference reads as 50%."""
    assert recovered_fraction(trivial=1.0, model=0.75, reference=0.5) == pytest.approx(50.0)


def test_passing_the_reference_exceeds_one_hundred_percent() -> None:
    """Beating the comparison point must not silently clamp to 100."""
    assert recovered_fraction(trivial=1.0, model=0.4, reference=0.5) > 100.0


def test_an_inverted_range_is_refused() -> None:
    """An inverted range is refused rather than returned.

    A reference worse than the trivial floor makes the percentage meaningless, and
    returning a number anyway would put a nonsense figure into the thesis.
    """
    with pytest.raises(ValueError, match="must be better"):
        recovered_fraction(trivial=0.5, model=0.4, reference=1.0)
