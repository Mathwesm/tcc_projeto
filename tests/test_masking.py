"""Tests for locating human choices in a transcript.

Every test here targets a bug that would silently corrupt the reported NLL rather
than raise. That is the whole risk profile of this module: the pipeline runs fine and
the number is wrong.
"""

from __future__ import annotations

import pytest

from centauro_lite.core.masking import (
    count_choices,
    find_choice_spans,
    spans_covering,
    token_choice_ids,
)


def test_span_excludes_the_delimiters() -> None:
    """The loss must land on the choice, not on the ``<<``/``>>`` markup.

    Letting the delimiters into the loss teaches the model to reproduce dataset
    notation -- trivially predictable tokens -- which drags the NLL down for a reason
    unrelated to modelling behaviour.
    """
    text = "You press <<D>>."
    (span,) = find_choice_spans(text)
    assert text[span.start : span.end] == "D"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no choices at all", 0),
        ("You press <<D>>.", 1),
        ("<<A>> then <<B>> then <<C>>", 3),
        ("multi\nline <<yes\nno>> choice", 1),
        ("<<>> empty is not a choice", 0),
        ("<<A>> and <<>> and <<B>>", 2),
        ("unclosed << marker", 0),
        ("<<left>> unclosed <<", 1),
    ],
)
def test_choice_counting(text: str, expected: int) -> None:
    """Counting must survive malformed and multi-line transcripts."""
    assert count_choices(text) == expected


def test_non_greedy_match_does_not_swallow_the_gap() -> None:
    """A greedy regex would merge two choices into one span covering the text between."""
    spans = find_choice_spans("<<A>> filler <<B>>")
    assert len(spans) == 2
    assert spans[0].end < spans[1].start


def test_token_straddling_the_boundary_is_not_scored() -> None:
    """A token that only partly overlaps a choice must stay out of the loss.

    Subword tokenizers merge across ``>>`` boundaries. Scoring such a token would put
    context characters into the loss, so containment has to be strict.
    """
    text = "a<<D>>b"  # the choice content "D" occupies characters [3, 4)
    offsets = [(0, 3), (3, 4), (3, 6)]  # context | the choice | "D>>" merged
    assert spans_covering(text, offsets) == [False, True, False]


def test_special_tokens_are_never_scored() -> None:
    """Special tokens carry an empty ``(0, 0)`` offset and are pure padding of meaning."""
    assert spans_covering("<<abc>>", [(0, 0), (2, 5)]) == [False, True]


def test_no_choices_means_nothing_is_scored() -> None:
    """A transcript window with no choice contributes no loss at all."""
    assert spans_covering("plain text", [(0, 4), (4, 8)]) == [False, False]


def test_every_choice_gets_its_own_index() -> None:
    """Windows snap on choice boundaries, so tokens must know *which* choice they are in."""
    text = "<<A>> gap <<B>>"
    ids = token_choice_ids(text, [(2, 3), (6, 9), (12, 13)])
    assert ids == [0, -1, 1]
