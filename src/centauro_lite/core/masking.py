"""Locating the human choices inside a Psych-101 transcript.

Psych-101 marks every human response with ``<<`` and ``>>`` -- for example
``"You press <<D>>"``. Loss and evaluation are computed over those choices alone;
instructions, stimuli and feedback are context.

The subtlety that decides whether the resulting metric means anything: the loss
belongs on the *content* between the delimiters, not on the delimiters themselves.
``<<`` and ``>>`` are dataset notation, and they are trivially predictable. Letting
them into the loss teaches the model to reproduce markup and drives the reported NLL
down for a reason that has nothing to do with modelling human behaviour.
"""

from __future__ import annotations

import re
from typing import NamedTuple

CHOICE_PATTERN = re.compile(r"<<(.*?)>>", re.DOTALL)
"""Group 1 captures the choice content. Every consumer must use the span of the
group, never the span of the whole match."""


class CharSpan(NamedTuple):
    """Half-open character interval ``[start, end)`` inside a transcript."""

    start: int
    end: int

    def contains(self, other: CharSpan) -> bool:
        """Whether ``other`` lies entirely inside this span.

        Args:
            other: Candidate span, typically a token's character offsets.

        Returns:
            ``True`` when ``other`` is fully enclosed.
        """
        return self.start <= other.start and other.end <= self.end


def find_choice_spans(text: str) -> list[CharSpan]:
    """Return the character spans of every human choice in a transcript.

    Args:
        text: Raw ``text`` field of a Psych-101 row.

    Returns:
        Spans covering the content between ``<<`` and ``>>``, in order of appearance.
        Empty choices (``<<>>``) are skipped: they carry no token to score, and
        keeping them would inflate the choice count without adding signal.
    """
    return [
        CharSpan(match.start(1), match.end(1))
        for match in CHOICE_PATTERN.finditer(text)
        if match.end(1) > match.start(1)
    ]


def count_choices(text: str) -> int:
    """Count the human choices in a transcript.

    Args:
        text: Raw ``text`` field of a Psych-101 row.

    Returns:
        Number of non-empty ``<<...>>`` spans.
    """
    return len(find_choice_spans(text))


def token_choice_ids(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    """Map each token to the choice it belongs to.

    Knowing *which* choice a token belongs to -- not merely that it belongs to one --
    is what lets a window stop before splitting a choice in half, so this returns
    indices rather than flags.

    Args:
        text: The transcript the offsets were produced from.
        offsets: Character offsets per token, as returned by a fast tokenizer with
            ``return_offsets_mapping=True``.

    Returns:
        One entry per token: the index of its choice, or ``-1`` outside any choice.
        Special tokens carry an empty ``(0, 0)`` offset and are always ``-1``.

    Note:
        Offsets are the only reliable way to align characters with tokens. Subword
        tokenization does not respect character boundaries in any way you can predict
        by counting, so anything that reimplements this by hand will be subtly wrong.
    """
    spans = find_choice_spans(text)
    ids = [-1] * len(offsets)
    if not spans:
        return ids

    cursor = 0
    for position, (start, end) in enumerate(offsets):
        if start == end:  # special token, no character span
            continue
        # Choices are sorted and disjoint, so a single forward cursor is enough:
        # advance past every choice that ends before this token starts.
        while cursor < len(spans) and spans[cursor].end <= start:
            cursor += 1
        if cursor < len(spans) and spans[cursor].contains(CharSpan(start, end)):
            ids[position] = cursor
    return ids


def spans_covering(text: str, offsets: list[tuple[int, int]]) -> list[bool]:
    """Mark which tokens fall inside a human choice.

    Args:
        text: The transcript the offsets were produced from.
        offsets: Character offsets per token.

    Returns:
        One flag per token: ``True`` when the token is part of a choice and should be
        scored.
    """
    return [choice != -1 for choice in token_choice_ids(text, offsets)]
