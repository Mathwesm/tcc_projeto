"""Building the experiment catalog: the EDA stage of the pipeline.

Nothing downstream should choose an experiment from memory or from the paper. This
module produces the real inventory of what Psych-101 contains, so the case study is
selected from evidence.

The catalog carries one column that decides the whole data-preparation design:
``fits_in_window_pct``, the share of participants whose transcript fits inside
``max_seq_length``. Where that number is low, truncation would throw away most of the
choices *and* keep only the earliest trials -- the phase where the participant has not
yet learned the task and behaviour is least predictable. The resulting NLL would rise
for a reason that has nothing to do with the model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from centauro_lite.core.masking import count_choices


class TranscriptRow(NamedTuple):
    """One Psych-101 row: a whole participant's session."""

    experiment: str
    participant: str
    text: str


class LengthTokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this module needs.

    Depending on a narrow protocol rather than the concrete class keeps the core
    logic testable without downloading a tokenizer.
    """

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        """Tokenize ``text`` and return at least an ``input_ids`` entry."""
        ...


class ExperimentStats(BaseModel):
    """Aggregated statistics for one experiment of Psych-101."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    domain: str | None
    n_participants: int
    n_choices_total: int
    choices_mean: float
    chars_mean: float
    chars_p50: float
    chars_p95: float
    chars_max: int
    n_sampled: int
    tokens_mean: float
    tokens_p95: float
    tokens_max: int
    fits_in_window_pct: float
    """Share of sampled participants whose full transcript fits in one window. The
    lower this is, the more truncation would discard."""

    windows_mean: float
    """Average number of windows a participant is cut into. ``1.0`` means chunking
    changes nothing for this experiment."""


class _Accumulator:
    """Per-experiment running collection, kept until percentiles can be computed."""

    def __init__(self) -> None:
        self.char_lengths: list[int] = []
        self.choice_counts: list[int] = []
        self.sample_texts: list[str] = []

    def add(self, text: str, *, keep_sample: bool) -> None:
        """Record one participant.

        Args:
            text: The participant's transcript.
            keep_sample: Whether to retain the text for later tokenization.
        """
        self.char_lengths.append(len(text))
        self.choice_counts.append(count_choices(text))
        if keep_sample:
            self.sample_texts.append(text)


def _token_length(tokenizer: LengthTokenizer, text: str) -> int:
    """Return the token count of ``text`` without special tokens.

    Args:
        tokenizer: Tokenizer of the model that will consume the data.
        text: Transcript to measure.

    Returns:
        Number of tokens.
    """
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def build_catalog(
    rows: Iterable[TranscriptRow],
    tokenizer: LengthTokenizer,
    *,
    max_seq_length: int,
    sample_per_experiment: int,
    domain_of: Any = None,
) -> list[ExperimentStats]:
    """Aggregate the dataset into one row per experiment.

    Args:
        rows: Every transcript of the dataset.
        tokenizer: Tokenizer used to measure real token lengths on a sample.
        max_seq_length: Window size the token statistics are judged against.
        sample_per_experiment: How many participants per experiment to tokenize.
            Tokenizing all 60k transcripts would cost far more than the precision it
            buys, and the spread within an experiment is small by construction.
        domain_of: Optional callable mapping an experiment name to a case-study
            domain, used to flag the target experiments in the catalog.

    Returns:
        Statistics per experiment, sorted by participant count descending so the
        largest experiments -- the useful ones for a small case study -- come first.
    """
    accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)

    for row in rows:
        acc = accumulators[row.experiment]
        acc.add(row.text, keep_sample=len(acc.sample_texts) < sample_per_experiment)

    catalog: list[ExperimentStats] = []
    for experiment, acc in accumulators.items():
        chars = np.asarray(acc.char_lengths, dtype=np.int64)
        choices = np.asarray(acc.choice_counts, dtype=np.int64)
        token_lengths = np.asarray(
            [_token_length(tokenizer, text) for text in acc.sample_texts],
            dtype=np.int64,
        )
        windows = np.ceil(token_lengths / max_seq_length)

        catalog.append(
            ExperimentStats(
                experiment=experiment,
                domain=domain_of(experiment) if domain_of is not None else None,
                n_participants=int(chars.size),
                n_choices_total=int(choices.sum()),
                choices_mean=round(float(choices.mean()), 1),
                chars_mean=round(float(chars.mean()), 1),
                chars_p50=round(float(np.percentile(chars, 50)), 1),
                chars_p95=round(float(np.percentile(chars, 95)), 1),
                chars_max=int(chars.max()),
                n_sampled=int(token_lengths.size),
                tokens_mean=round(float(token_lengths.mean()), 1),
                tokens_p95=round(float(np.percentile(token_lengths, 95)), 1),
                tokens_max=int(token_lengths.max()),
                fits_in_window_pct=round(
                    100.0 * float((token_lengths <= max_seq_length).mean()), 1
                ),
                windows_mean=round(float(windows.mean()), 2),
            )
        )

    catalog.sort(key=lambda stats: stats.n_participants, reverse=True)
    return catalog


def iter_transcripts(dataset: Iterable[Mapping[str, Any]]) -> Iterator[TranscriptRow]:
    """Adapt a Hugging Face dataset into :class:`TranscriptRow` values.

    Args:
        dataset: Any iterable of mappings carrying ``experiment``, ``participant``
            and ``text`` keys.

    Yields:
        One row per participant.
    """
    for row in dataset:
        yield TranscriptRow(
            experiment=str(row["experiment"]),
            participant=str(row["participant"]),
            text=str(row["text"]),
        )


def catalog_totals(catalog: Sequence[ExperimentStats]) -> dict[str, int]:
    """Summarise the catalog for a sanity check against the published figures.

    Psych-101 is documented as 60,092 participants, 160 experiments and 10,681,650
    choices. Comparing against those numbers is the cheapest way to notice that the
    dataset was filtered, truncated or partially downloaded.

    Args:
        catalog: Output of :func:`build_catalog`.

    Returns:
        Totals for experiments, participants and choices.
    """
    return {
        "n_experiments": len(catalog),
        "n_participants": sum(stats.n_participants for stats in catalog),
        "n_choices": sum(stats.n_choices_total for stats in catalog),
    }
