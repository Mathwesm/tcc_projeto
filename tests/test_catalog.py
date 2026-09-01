"""Tests for the experiment catalog.

The catalog is what the case-study experiments get chosen from, and
``fits_in_window_pct`` is what decides whether transcripts must be chunked. A wrong
number here propagates into a wrong methodological decision.
"""

from __future__ import annotations

from typing import Any

from centauro_lite.core.catalog import (
    TranscriptRow,
    build_catalog,
    catalog_totals,
    iter_transcripts,
)


class WordTokenizer:
    """Whitespace tokenizer standing in for a real one.

    Using a fake keeps these tests offline and deterministic; the catalog logic under
    test is arithmetic over token counts, not tokenization itself.
    """

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        """Return one id per whitespace-separated word."""
        return {"input_ids": list(range(len(text.split())))}


def _row(experiment: str, participant: str, n_words: int, n_choices: int) -> TranscriptRow:
    """Build a transcript of a given length carrying a given number of choices."""
    choices = " ".join(f"<<{index}>>" for index in range(n_choices))
    filler = " ".join("word" for _ in range(max(0, n_words - n_choices)))
    return TranscriptRow(experiment=experiment, participant=participant, text=f"{filler} {choices}")


def test_short_transcripts_need_no_chunking() -> None:
    """When everything fits in one window, truncation would discard nothing."""
    rows = [_row("short/exp.csv", str(index), n_words=10, n_choices=2) for index in range(5)]
    (stats,) = build_catalog(rows, WordTokenizer(), max_seq_length=64, sample_per_experiment=10)
    assert stats.fits_in_window_pct == 100.0
    assert stats.windows_mean == 1.0


def test_long_transcripts_expose_how_much_truncation_would_discard() -> None:
    """This is the number that justifies chunking over truncation.

    With transcripts four times the window, truncating keeps only the first quarter --
    and specifically the earliest trials, where the participant has not yet learned
    the task and behaviour is least predictable.
    """
    rows = [_row("long/exp.csv", str(index), n_words=400, n_choices=40) for index in range(5)]
    (stats,) = build_catalog(rows, WordTokenizer(), max_seq_length=100, sample_per_experiment=10)
    assert stats.fits_in_window_pct == 0.0
    assert stats.windows_mean == 4.0


def test_choices_are_counted_across_every_participant() -> None:
    """The choice total is the size of the evaluation set; undercounting hides data loss."""
    rows = [_row("exp/a.csv", str(index), n_words=50, n_choices=3) for index in range(4)]
    (stats,) = build_catalog(rows, WordTokenizer(), max_seq_length=64, sample_per_experiment=10)
    assert stats.n_choices_total == 12
    assert stats.choices_mean == 3.0
    assert stats.n_participants == 4


def test_sampling_limits_tokenization_without_touching_participant_counts() -> None:
    """Token stats come from a sample; participant and choice totals must not.

    Confusing the two would report the size of the sample as the size of the dataset.
    """
    rows = [_row("exp/a.csv", str(index), n_words=20, n_choices=1) for index in range(50)]
    (stats,) = build_catalog(rows, WordTokenizer(), max_seq_length=64, sample_per_experiment=5)
    assert stats.n_sampled == 5
    assert stats.n_participants == 50
    assert stats.n_choices_total == 50


def test_experiments_are_ranked_by_size() -> None:
    """The largest experiments are the usable ones for a small case study."""
    rows = [_row("small/a.csv", "0", 10, 1)]
    rows += [_row("big/a.csv", str(index), 10, 1) for index in range(3)]
    catalog = build_catalog(rows, WordTokenizer(), max_seq_length=64, sample_per_experiment=10)
    assert [stats.experiment for stats in catalog] == ["big/a.csv", "small/a.csv"]


def test_domain_tagging_flags_the_case_study_experiments() -> None:
    """Untagged experiments must stay ``None`` rather than default to a domain."""
    rows = [_row("peterson2021using/e.csv", "0", 10, 1), _row("other/e.csv", "0", 10, 1)]
    catalog = build_catalog(
        rows,
        WordTokenizer(),
        max_seq_length=64,
        sample_per_experiment=10,
        domain_of=lambda name: "risky_choice" if "peterson" in name else None,
    )
    domains = {stats.experiment: stats.domain for stats in catalog}
    assert domains["peterson2021using/e.csv"] == "risky_choice"
    assert domains["other/e.csv"] is None


def test_totals_can_be_checked_against_the_published_figures() -> None:
    """Comparing totals to the dataset card is how a partial download gets noticed."""
    rows = [_row("a/x.csv", "0", 10, 2), _row("b/x.csv", "1", 10, 3)]
    catalog = build_catalog(rows, WordTokenizer(), max_seq_length=64, sample_per_experiment=10)
    assert catalog_totals(catalog) == {
        "n_experiments": 2,
        "n_participants": 2,
        "n_choices": 5,
    }


def test_iter_transcripts_adapts_mapping_rows() -> None:
    """The dataset yields mappings; the catalog needs typed rows."""
    raw = [{"experiment": "a/x.csv", "participant": 7, "text": "You press <<D>>"}]
    (row,) = list(iter_transcripts(raw))
    assert row == TranscriptRow("a/x.csv", "7", "You press <<D>>")
