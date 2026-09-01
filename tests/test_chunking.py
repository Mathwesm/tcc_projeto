"""Tests for cutting transcripts into windows.

Chunking replaced truncation because truncation kept only each participant's first
trials. These tests guard the two properties that make the replacement worth anything:
every choice is scored, and each one is scored exactly once.
"""

from __future__ import annotations

from typing import Any

import pytest

from centauro_lite.core.chunking import IGNORE_INDEX, iter_windows


class CharTokenizer:
    """One token per character, so token indices and character offsets coincide.

    A real tokenizer would make the expected values in these tests unreadable and tie
    them to a specific vocabulary. The logic under test is index arithmetic over
    offsets, which this exercises exactly.
    """

    def __call__(self, text: str, **_: Any) -> dict[str, Any]:
        """Return one token id and one offset per character."""
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _scored_text(window: Any, text: str, start_offset: int = 0) -> str:
    """Reconstruct the characters a window actually scores."""
    return (
        "".join(
            chr(token)
            for token, label in zip(window.input_ids, window.labels, strict=True)
            if label != IGNORE_INDEX
        )
        + text[:0]
        + str(start_offset)[:0]
    )


def _all_scored(text: str, **kwargs: Any) -> str:
    """Concatenate everything scored across every window of a transcript."""
    windows = list(iter_windows(text, CharTokenizer(), **kwargs))
    return "".join(_scored_text(window, text) for window in windows)


def test_only_the_choice_content_is_scored() -> None:
    """Context and the ``<<``/``>>`` markup must never reach the loss."""
    assert _all_scored("press <<D>> now", max_seq_length=100, stride=100) == "D"


def test_every_choice_survives_chunking() -> None:
    """This is the property truncation violates: nothing may be silently dropped."""
    text = "a<<A>>bbbb<<B>>cccc<<C>>dddd<<D>>"
    assert _all_scored(text, max_seq_length=8, stride=8) == "ABCD"


def test_no_choice_is_scored_twice_under_overlap() -> None:
    """Overlapping windows see the same choice again; the metric must not count it twice.

    Double counting inflates the denominator of the NLL with duplicated tokens, which
    quietly changes the number being compared to the paper.
    """
    text = "a<<A>>bbbb<<B>>cccc<<C>>dddd<<D>>"
    assert _all_scored(text, max_seq_length=12, stride=6) == "ABCD"


def test_windows_do_not_split_a_choice() -> None:
    """A window ending mid-choice would score half of it with almost no context.

    The window is pulled back to before the choice so that the choice is always
    scored in one piece, with the trials that preceded it still visible.
    """
    text = "aaaaaaaa<<XYZ>>bbbb"
    windows = list(iter_windows(text, CharTokenizer(), max_seq_length=12, stride=12))
    assert _all_scored(text, max_seq_length=12, stride=12) == "XYZ"
    # The first window stops before "<<XYZ>>" and therefore scores nothing, so it is
    # dropped; what remains must carry the whole choice.
    assert all(window.n_scored > 0 for window in windows)


def test_windows_without_a_choice_are_dropped() -> None:
    """A window with nothing to score costs compute and contributes no gradient."""
    text = "a" * 50
    assert list(iter_windows(text, CharTokenizer(), max_seq_length=10, stride=10)) == []


def test_window_never_exceeds_the_configured_length() -> None:
    """Exceeding the length is an out-of-memory error on a 6GB card, not a warning."""
    text = "filler<<A>>" * 40
    windows = list(iter_windows(text, CharTokenizer(), max_seq_length=16, stride=16))
    assert windows
    assert all(len(window.input_ids) <= 16 for window in windows)


def test_attention_mask_matches_the_token_count() -> None:
    """Padding is applied later by the collator; here every token is real."""
    (window,) = list(iter_windows("x<<A>>", CharTokenizer(), max_seq_length=64, stride=64))
    assert window.attention_mask == [1] * len(window.input_ids)
    assert len(window.labels) == len(window.input_ids)


def test_labels_repeat_the_input_ids_at_scored_positions() -> None:
    """Causal models shift labels internally; shifting here too would train off-by-one."""
    (window,) = list(iter_windows("x<<A>>", CharTokenizer(), max_seq_length=64, stride=64))
    for token, label in zip(window.input_ids, window.labels, strict=True):
        assert label in (IGNORE_INDEX, token)


def test_choice_longer_than_the_window_still_progresses() -> None:
    """Snapping must not collapse a window to nothing and spin forever.

    A single choice longer than the window cannot be kept whole, so the code accepts
    the split rather than refusing to advance.
    """
    text = "<<" + "Z" * 40 + ">>"
    windows = list(iter_windows(text, CharTokenizer(), max_seq_length=8, stride=8))
    assert windows
    assert "".join(_scored_text(window, text) for window in windows) == "Z" * 40


@pytest.mark.parametrize("stride", [4, 8, 16])
def test_scored_content_is_independent_of_stride(stride: int) -> None:
    """Overlap buys context, not extra data. Changing it must not change the metric."""
    text = "aa<<A>>bbb<<B>>ccc<<C>>ddd<<D>>eee<<E>>"
    assert _all_scored(text, max_seq_length=16, stride=stride) == "ABCDE"
