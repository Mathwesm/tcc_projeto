"""Cutting a full session transcript into trainable windows.

A Psych-101 row is an entire participant session -- 4.2k tokens on average, up to
57,729. Only 2 of the 76 experiments fit inside a 2048-token window, so the transcripts
have to be divided.

Truncating instead would keep only each participant's *first* trials: exactly the phase
where the task has not been learned yet and behaviour is least predictable. The NLL
would rise from sampling bias rather than from any property of the model, and stop
being comparable to anything.

Two details keep the windows honest:

*Boundary snapping.* A window that ends mid-choice would score the first half of a
choice with full context and the second half with almost none. Windows therefore end
just before a choice they cannot contain.

*Scoring each choice once.* With overlapping windows the same choice appears twice.
The second occurrence is masked out, so the metric's denominator stays the true number
of choices.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple, Protocol

from centauro_lite.core.masking import token_choice_ids

IGNORE_INDEX = -100
"""Label value ignored by the cross-entropy loss."""


class OffsetTokenizer(Protocol):
    """The slice of a fast tokenizer this module needs."""

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        """Tokenize ``text``, returning ``input_ids`` and ``offset_mapping``."""
        ...


class Window(NamedTuple):
    """One training example: a slice of a transcript with its scored choices."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]

    @property
    def n_scored(self) -> int:
        """Number of tokens contributing to the loss.

        Returns:
            Count of labels that are not :data:`IGNORE_INDEX`.
        """
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


def _snap_end(token_choices: list[int], start: int, end: int, total: int) -> int:
    """Pull a window's end back so it does not cut a choice in half.

    Args:
        token_choices: Per-token choice index.
        start: First token of the window.
        end: Tentative end, exclusive.
        total: Total token count of the transcript.

    Returns:
        The adjusted end. Left untouched when the window reaches the end of the
        transcript (nothing is being cut) or when snapping would empty the window,
        which happens only if a single choice is longer than the whole window.
    """
    if end >= total:
        return end
    trailing = token_choices[end - 1]
    if trailing == -1 or token_choices[end] != trailing:
        return end  # the window already ends on a choice boundary

    snapped = end
    while snapped > start and token_choices[snapped - 1] == trailing:
        snapped -= 1
    return snapped if snapped > start else end


def iter_windows(
    text: str,
    tokenizer: OffsetTokenizer,
    *,
    max_seq_length: int,
    stride: int,
) -> Iterator[Window]:
    """Cut one transcript into windows, scoring every choice exactly once.

    Args:
        text: The participant's full transcript.
        tokenizer: Fast tokenizer providing offset mappings.
        max_seq_length: Window size in tokens.
        stride: Step between window starts. Equal to ``max_seq_length`` for contiguous
            windows; smaller values give the choices after each boundary more context
            at the cost of recomputing the overlap.

    Yields:
        Windows carrying at least one scored choice. Windows with none are dropped:
        they add loss-free compute to training and are skipped by the metric anyway.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: list[int] = list(encoded["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(pair) for pair in encoded["offset_mapping"]]
    token_choices = token_choice_ids(text, offsets)
    total = len(input_ids)

    overlap = max_seq_length - stride
    start = 0
    next_unscored = 0

    while start < total:
        end = _snap_end(token_choices, start, min(start + max_seq_length, total), total)
        window = _build_window(input_ids, token_choices, start, end, next_unscored)
        if window.n_scored:
            yield window

        next_unscored = end
        if end >= total:
            return
        # ``end`` can be pulled back by snapping, so advance from it rather than from
        # ``start``; the max() guarantees forward progress even in pathological cases.
        start = max(start + 1, end - overlap)


def _build_window(
    input_ids: list[int],
    token_choices: list[int],
    start: int,
    end: int,
    next_unscored: int,
) -> Window:
    """Assemble one window, masking everything that is not a fresh choice token.

    Args:
        input_ids: Token ids of the whole transcript.
        token_choices: Per-token choice index.
        start: First token of the window.
        end: End of the window, exclusive.
        next_unscored: First position not yet scored by an earlier window.

    Returns:
        The window, with context and already-scored tokens set to
        :data:`IGNORE_INDEX`.
    """
    ids = input_ids[start:end]
    labels = [
        token_id if token_choices[position] != -1 and position >= next_unscored else IGNORE_INDEX
        for position, token_id in zip(range(start, end), ids, strict=True)
    ]
    return Window(input_ids=ids, attention_mask=[1] * len(ids), labels=labels)
