"""Tests for the NLL aggregation.

This is the number the thesis reports. Every bug guarded here produces a plausible
number rather than an error, which is what makes them worth a test.
"""

from __future__ import annotations

import pytest

from centauro_lite.core.chunking import IGNORE_INDEX
from centauro_lite.core.metrics import NllAccumulator, scored_token_count


def test_position_zero_is_not_counted() -> None:
    """A causal model predicts token ``i`` from what precedes it, so index 0 is dropped.

    Counting it would make the reported denominator larger than the one the model
    actually used, and the NLL would come out too low by a hair -- consistently, and
    invisibly.
    """
    labels = [[5, 7, IGNORE_INDEX]]
    assert scored_token_count(labels) == 1


def test_padding_is_not_counted() -> None:
    """Padded positions carry the ignore index and must not dilute the average."""
    labels = [[IGNORE_INDEX, 5, IGNORE_INDEX, IGNORE_INDEX]]
    assert scored_token_count(labels) == 1


def test_counting_spans_the_whole_batch() -> None:
    """Every example in the batch contributes, each dropping only its own index 0."""
    labels = [[IGNORE_INDEX, 1, 2], [IGNORE_INDEX, 3, IGNORE_INDEX]]
    assert scored_token_count(labels) == 3


def test_aggregate_weights_by_token_not_by_batch() -> None:
    """Batches differ in size; averaging their losses would misweight the result.

    Here 100 tokens at loss 1.0 and 1 token at loss 5.0 must land near 1.04, not at
    the 3.0 a naive mean of the two batch losses would give.
    """
    accumulator = NllAccumulator()
    accumulator.add("exp/a.csv", mean_loss=1.0, n_scored=100)
    accumulator.add("exp/a.csv", mean_loss=5.0, n_scored=1)
    assert accumulator.result("test").nll == pytest.approx(105 / 101)


def test_experiments_are_reported_separately() -> None:
    """The aggregate hides a model that is great at one task and useless at another."""
    accumulator = NllAccumulator()
    accumulator.add("easy/a.csv", mean_loss=0.2, n_scored=100)
    accumulator.add("hard/a.csv", mean_loss=2.0, n_scored=100)
    result = accumulator.result("test")
    assert result.per_experiment == pytest.approx({"easy/a.csv": 0.2, "hard/a.csv": 2.0})
    assert result.nll == pytest.approx(1.1)


def test_per_experiment_token_counts_are_reported() -> None:
    """An NLL from 3 tokens must not look as authoritative as one from 3,000."""
    accumulator = NllAccumulator()
    accumulator.add("big/a.csv", mean_loss=1.0, n_scored=3000)
    accumulator.add("small/a.csv", mean_loss=1.0, n_scored=3)
    assert accumulator.result("test").per_experiment_tokens == {
        "big/a.csv": 3000,
        "small/a.csv": 3,
    }


def test_empty_batches_are_ignored_not_averaged_in() -> None:
    """Treating a choice-free batch as zero loss would improve the score for free."""
    accumulator = NllAccumulator()
    accumulator.add("exp/a.csv", mean_loss=1.0, n_scored=10)
    accumulator.add("exp/a.csv", mean_loss=0.0, n_scored=0)
    assert accumulator.result("test").nll == pytest.approx(1.0)
    assert accumulator.n_scored_tokens == 10


def test_an_empty_evaluation_raises() -> None:
    """Scoring nothing means the masking or the split broke.

    Returning 0.0 would report a perfect model, which is the worst possible way to
    surface that failure.
    """
    with pytest.raises(ValueError, match="No scored tokens"):
        NllAccumulator().result("test")


def test_references_travel_with_the_result() -> None:
    """The paper's numbers belong next to ours, labelled as the landmark they are."""
    accumulator = NllAccumulator()
    accumulator.add("exp/a.csv", mean_loss=0.5, n_scored=10)
    comparison = accumulator.result("mine").comparison()
    assert comparison["mine"] == pytest.approx(0.5)
    assert comparison["reference_centaur_70b"] == 0.44
    assert comparison["reference_llama_base"] == 0.58
